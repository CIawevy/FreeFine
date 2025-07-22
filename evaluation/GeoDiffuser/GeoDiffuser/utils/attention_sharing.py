import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import v2
import abc

from GeoDiffuser.utils.generic_torch import reshape_transform_coords, reshape_attention_mask
# from GeoDiffuser.inversion import NullInversion


LOW_RESOURCE = False
def perform_attention(q, k, v, scale, mask=None):

    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=True):
        
        out = F.scaled_dot_product_attention(q,k,v, scale=scale)

    return out


def baddbmm_bmm(a, b, scale = 1):

    b_in = torch.empty(
                a.shape[0], a.shape[1], b.shape[-1], dtype=a.dtype, device=a.device
            )
    sim = torch.baddbmm(b_in, a, b, beta=0, alpha = scale)

    return sim

def compute_attention(q, k, scale, mask = None, fg_mask_warp = None, fg_mask = None, inpaint_mask = None):

    b_in = torch.empty(
                q.shape[0], q.shape[1], k.shape[1], dtype=q.dtype, device=q.device
            )
    sim = torch.baddbmm(b_in, q, k.permute(0, -1, 1), beta=0, alpha = scale) #* scale

    if fg_mask_warp is not None:
        # Ensure that the forward warped mask is attending to the object of interest!
        sim[:, fg_mask_warp[0, 0].reshape(-1) >= 0.5, :][..., fg_mask[0, 0].reshape(-1) < 0.5] = -1000.0

        # Ensure the inpainting mask does not attend to foreground
        sim[:, inpaint_mask[0, 0].reshape(-1) >= 0.5, :][..., fg_mask[0, 0].reshape(-1) >= 0.5] = -1000.0

    # attention, what we cannot get enough of
    attn = F.softmax(sim, dim=-1)
    
    return attn


def smooth_mask(mask, k = 1):
    
    # print(mask.shape, " old")
    # if mask.shape[-1] <= 32:
    #     k = 1
    # elif mask.shape[-1] <= 64:
    #     k = 1
    # else:
    #     k = 2
    
    
    mask = F.max_pool2d(mask, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
    # print(mask.shape, " mid")
    # mask = F.interpolate(mask, size=(mask.shape[2:]))
    # print(mask.shape, " f")
    return mask

@torch.no_grad()
def interpolate_from_mask(features, foreground_mask, distance):

    ''' 
    Interpolate features for the foreground

    features - b, h, n^2, D
    mask - 1, 1, n^2, 1
    distance - b/1, n^2, n^2

    '''
    # print(foreground_mask.shape)

    # Setting background distances to very high value
    distance_new = distance * 512 / 2.0 + 100000 * (1.0 - (foreground_mask[:1, :1, :, 0] > 0.5) * 1.0)
    inv_distance = 1.0 / (distance_new + 1e-4)
    max_4_inv_distances = torch.topk(inv_distance, k=4, dim=-1, largest=True, sorted=False)



    m_indices = (max_4_inv_distances.indices[:, None, :] * torch.ones_like(features[..., 0])[..., None]).long() # b, h, n^2, 4
    m_values = max_4_inv_distances.values[:, None, :] * torch.ones_like(features[..., 0])[..., None]

    # print(m_indices.shape, features.shape, inv_distance.shape)
    b_idx = (torch.arange(0, features.shape[0])[:, None, None, None].type_as(m_indices) * torch.ones_like(m_indices)).type_as(m_indices)

    m_indices = torch.clip(m_indices, min=0, max = features.shape[2] - 1)
    h_idx = torch.arange(0, features.shape[1])[None, :, None, None].type_as(m_indices)

    h_idx = (h_idx * torch.ones_like(m_indices)).type_as(m_indices)

    features_select = features[b_idx.long(), h_idx.long(), m_indices] # b, h, n^2, 4, D
    feature_inv_distances=m_values # b, h, n^2, 4

    interpolated_features = torch.sum(features_select * feature_inv_distances[..., None], -2) / (torch.sum(feature_inv_distances, -1)[..., None] + 1e-12) # b, h, n^2, D

    distance_weights = torch.exp(- (1 / torch.max(feature_inv_distances, -1).values) / 5) # b, h, n^2

    return interpolated_features, distance_weights
        
        


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t, transform_coords):
        # print("step callback")
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale = None, mask = None):
        raise NotImplementedError

    def __call__(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale=None, mask = None):
       
        if self.cur_att_layer >= self.num_uncond_att_layers:
            
            if (type(self).__name__.startswith("AttentionGeometry")):
                if LOW_RESOURCE:
                    out = self.forward(q, k, v, is_cross, place_in_unet, transform_coord=transform_coords, scale = scale, mask = mask)
                else:
                    out_edit = self.forward(q, k, v, is_cross, place_in_unet, transform_coords = transform_coords, scale = scale, mask = mask)
                    out = out_edit
            else:
                out = self.forward(q, k, v, is_cross, place_in_unet, transform_coords=transform_coords, scale = scale, mask = mask)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return out
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0




class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords=None, scale = None, mask = None):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        attn = compute_attention(q, k, scale, mask)
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn.detach())
            
        out = torch.bmm(attn, v)
        return out

    
    def attn_store(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn.detach())
        return 

    def between_steps(self):
        # print("Running between step")
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.step_store:
                self.attention_store[key] = self.attention_store[key] + self.step_store[key]

                if self.cur_step == 1:
                    self.attention_store["length_" + key] = len(self.step_store[key])

        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def get_base_edit_qkv(q, k, v, batch_size, coords_base = None, coords_edit = None, use_cfg = True):
    '''
    Get Reference and edit q,k,v
    '''

    # print(q.shape, k.shape, v.shape)
    
    if use_cfg:
        h = q.shape[0] // (2 * batch_size)
        q = q.reshape(2 * batch_size, h, *q.shape[1:])
        k = k.reshape(2 * batch_size, h, *k.shape[1:])
        v = v.reshape(2 * batch_size, h, *v.shape[1:])

        # q_base, k_base, v_base = q[2:3], k[2:3], v[2:3] # 1, F, h*w, D
        # q_edit, k_edit, v_edit = q[3:], k[3:], v[3:] # b - 1, F, h*w, D
    else:
        h = q.shape[0] // (batch_size)
        q = q.reshape(batch_size, h, *q.shape[1:])
        k = k.reshape(batch_size, h, *k.shape[1:])
        v = v.reshape(batch_size, h, *v.shape[1:])

    # print(h, "h size", q.shape, k.shape, v.shape)

    
    if coords_base is not None:
        # print(coords_base)
        q_base, k_base, v_base = q[coords_base[0]:coords_base[1]], k[coords_base[0]:coords_base[1]], v[coords_base[0]:coords_base[1]] # 1, F, h*w, D
    
    if coords_edit is not None:
        # print(coords_edit)
        q_edit, k_edit, v_edit = q[coords_edit[0]:coords_edit[1]], k[coords_edit[0]:coords_edit[1]], v[coords_edit[0]:coords_edit[1]] # b - 1, F, h*w, D
    
    return q_base.detach(), k_base.detach(), v_base.detach(), q_edit, k_edit, v_edit




# GeoDiffuser Attention Sharing Mechanism
def attention_sharing(q_ref, k_ref, v_ref, q_edit, k_edit, v_edit):

    



    pass