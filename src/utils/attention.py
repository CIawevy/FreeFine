import math
from typing import Union, Tuple, List, Callable, Dict, Optional
import torch
import torch.nn.functional as nnf
from diffusers import DiffusionPipeline
import numpy as np
# from IPython.display import display
from PIL import Image
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import abc
# import ptp_utils
# import seq_aligner

# from FPE
# def register_attention_control_new(model, controller):
#     def ca_forward(self, place_in_unet):
#         to_out = self.to_out
#         if type(to_out) is torch.nn.modules.container.ModuleList:
#             to_out = self.to_out[0]
#         else:
#             to_out = self.to_out
#
#         def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
#             x = hidden_states
#             context = encoder_hidden_states
#             mask = attention_mask
#             batch_size, sequence_length, dim = x.shape
#             h = self.heads
#             q = self.to_q(x)
#             is_cross = context is not None
#             context = context if is_cross else x
#             k = self.to_k(context)
#             v = self.to_v(context)
#             q = self.head_to_batch_dim(q)
#             k = self.head_to_batch_dim(k)
#             v = self.head_to_batch_dim(v)
#
#             sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
#
#             if mask is not None:
#                 mask = mask.reshape(batch_size, -1)
#                 max_neg_value = -torch.finfo(sim.dtype).max
#                 mask = mask[:, None, :].repeat(h, 1, 1)
#                 sim.masked_fill_(~mask, max_neg_value)
#
#             # attention, what we cannot get enough of
#             attn = sim.softmax(dim=-1)
#             # if is_cross:
#             #     ssss = []
#             attn = controller(attn, is_cross, place_in_unet)
#             out = torch.einsum("b i j, b j d -> b i d", attn, v)
#             out = self.batch_to_head_dim(out)
#             return to_out(out)
#
#         return forward
#
#     class DummyController:
#
#         def __call__(self, *args):
#             return args[0]
#
#         def __init__(self):
#             self.num_att_layers = 0
#
#     if controller is None:
#         controller = DummyController()
#
#     def register_recr(net_, count, place_in_unet):
#         # print(net_.__class__.__name__)
#         if net_.__class__.__name__ == 'Attention':
#             net_.forward = ca_forward(net_, place_in_unet)
#             return count + 1
#         elif hasattr(net_, 'children'):
#             for net__ in net_.children():
#                 count = register_recr(net__, count, place_in_unet)
#         return count
#
#     cross_att_count = 0
#     sub_nets = model.unet.named_children()
#     # print(sub_nets)
#     for net in sub_nets:
#         if "down" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "down")
#         elif "up" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "up")
#         elif "mid" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "mid")
#
#     controller.num_att_layers = cross_att_count


#from p2p
def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, ):
            is_cross = encoder_hidden_states is not None

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            # if controller.use_contrast:
            #     model.scale = self.scale
            #     attention_scores, dtype = model.get_attention_scores(query, key, attention_mask,use_softmax=False)
            #     #contrast operation TODO:check
            #     attention_scores = model.contrast_operation(attention_scores, controller.contrast_beta,clamp=False,dim=-1)
            #     #do softmax after
            #     attention_probs = attention_scores.softmax(dim=-1)
            #     del attention_scores
            #     attention_probs = attention_probs.to(dtype)
            # else:
            attention_probs = self.get_attention_scores(query, key, attention_mask)

            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.LOW_RESOURCE = False
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)

        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):

                    self.attention_store[key][i] += self.step_store[key][i]
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

class Mask_Expansion_SELF_ATTN(AttentionControl):
    @staticmethod
    def get_empty_store():
        # return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #         "down_self": [],  "mid_self": [],  "up_self": []}
        return {"down_self": [], "mid_self": [], "up_self": []}


    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            pass
        else:
            key = f"{place_in_unet}_{'self'}"
            self.step_store[key].append(self.get_correlation_mask(attn))
        return attn
    def view(self,mask, title='Mask'):
        """
        显示输入的mask图像

        参数:
        mask (torch.Tensor): 要显示的mask图像，类型应为torch.bool或torch.float32
        title (str): 图像标题
        """
        # 确保输入的mask是float类型以便于显示
        mask_new = mask.float()
        mask_new = mask_new.detach().cpu()
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_new.numpy(), cmap='gray')
        plt.title(title)
        plt.axis('off')  # 去掉坐标轴
        plt.show()

    def between_steps(self):
        mask_=torch.zeros_like(self.obj_mask)
        h,w = mask_.shape
        attention_map_length = 0
        for k,v in self.step_store.items():
            for f_t in v:
                fh,fw = f_t.shape
                if h/fh > 8 :
                    continue
                    # pass
                if mask_.shape != f_t.shape:
                    f_t = F.interpolate(f_t.unsqueeze(0).unsqueeze(0), size=mask_.shape,
                                                mode='nearest').squeeze(0).squeeze(0)
                mask_ += f_t
                attention_map_length+=1
        mask_ /= attention_map_length
        self.expansion_mask_store[f'step{self.cur_step}'] = mask_
        self.step_store = self.get_empty_store()

    def contrast_operation(self,mask,contrast_beta,clamp=True,min_v=0,max_v=1,dim=-1):
        if dim is not None:
            mean_value = mask.mean(dim=dim)[:,:,None]
        else:
            mean_value = mask.mean()

        #increase varience
        contrasted_mask = (mask - mean_value) * contrast_beta + mean_value
        del mean_value
        if clamp:
            return contrasted_mask.clamp(min=min_v,max=max_v)
        return contrasted_mask
    def get_correlation_mask(self,attn):
        assert self.obj_mask is not None,"input obj mask please!"
        mask = self.obj_mask
        h,seq_len,_ = attn.shape #head
        m_h,m_w = mask.shape
        sample_rate = math.sqrt(m_h*m_w  /seq_len)
        attn_h,attn_w = int(m_h/sample_rate),int(m_w/sample_rate)
        mask = mask.unsqueeze(0).unsqueeze(0)
        downsampled_mask = F.interpolate(mask, size=(attn_h, attn_w), mode='nearest')
        mask_index = (downsampled_mask > 0.5).squeeze(0).squeeze(0).flatten()#obj mask -> down sampled obj mask
        correlate_maps = attn[:,mask_index]
        if self.use_contrast:
            correlate_maps = self.contrast_operation(correlate_maps, self.contrast_beta, clamp=True, min_v=0, max_v=1, dim=-1)
        correlate_maps = correlate_maps.reshape(h,-1,attn_h,attn_w).sum(dim=0).sum(dim=0) #sum of all heads and all pixels for each obj latent position
        return correlate_maps

    def reset(self):
        super(Mask_Expansion_SELF_ATTN, self).reset()
        self.step_store = self.get_empty_store()
        self.expansion_mask_store = {}

    def __init__(self,):
        super(Mask_Expansion_SELF_ATTN, self).__init__()
        self.step_store = self.get_empty_store()
        self.expansion_mask_store={}

class SelfAttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(SelfAttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                pass
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]]):
        super(SelfAttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])

