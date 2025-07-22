import torch, os
import torch.nn.functional as F
import numpy as np
os.environ['HF_HOME'] = "/oscar/scratch/rsajnani/rsajnani/research/.cache/hf"
import diffusers
from geometry_editor_updated import load_model
from PIL import Image
from ui_utils import read_exp, list_exp_details, complete_path, check_if_exp_root
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
from torch.optim import AdamW
import tqdm

# Borrowed from https://github.com/georg-bn/rotation-steerers/blob/main/rotation_steerers/steerers.py

class ContinuousSteerer(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = torch.nn.Parameter(generator)

    @torch.autocast("cuda", torch.half)
    def forward(self, x, angle_radians):

        # print(x.shape)
        # print(self.generator.shape)
        # exit()
        return F.linear(x, torch.matrix_exp(angle_radians * self.generator))

    def steer_descriptions(self, descriptions, angle_radians, normalize=False):
        descriptions = self.forward(descriptions, angle_radians)
        if normalize:
            descriptions = F.normalize(descriptions, dim=-1)
        return descriptions


def preprocess_image_torch(image):

    image = torch.from_numpy(image) / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return image

def encode_image(im, model):
    image = im.to(model.dtype)
    latents = model.vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents

def get_affine_grid(t, angle=0):
    
    angle_rad = np.deg2rad(angle)
    theta = torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad), 0], 
                         [np.sin(angle_rad), np.cos(angle_rad),  0]]).unsqueeze(0)

    grid = torch.nn.functional.affine_grid(theta, t.shape, align_corners=False)

    return grid

def rotate_tensor(t, degree=0, mask_in = None):

    grid = get_affine_grid(t, degree).type_as(t)
    
    if mask_in is None:
        mask = torch.ones_like(t)
    else:
        mask = mask_in

    t_rotated = torch.nn.functional.grid_sample(t, grid)

    mask_rotated = torch.nn.functional.grid_sample(mask, grid)
    return t_rotated, mask_rotated

def descriptor_loss(d_A, d_B, mask):

    d_A_n = F.normalize(d_A, dim=1, eps=1e-6)
    d_B_n = F.normalize(d_B, dim=1, eps=1e-6)


    corr = torch.sum(d_A_n * d_B_n, dim=1)
    loss = (1.0 - corr).mean()

    return loss

def get_generator_4():

    z = torch.zeros(
            [2, 2],
            device='cuda')
    a = torch.tensor([[0., 1],
                          [-1, 0]],
                         device='cuda')
    
    b = torch.tensor([[0., 2],
                      [-2, 0]],
                         device='cuda')
    
    # b = torch.tensor([0, 2])
    generator = torch.block_diag(a, b)

    return generator

def get_generator(descriptor_size):

    generator = torch.block_diag(
        torch.zeros(
            [descriptor_size // 2,
             descriptor_size // 2],
            device='cuda',
        ),
        *(
            torch.tensor([[0., 1],
                          [-1, 0]],
                         device='cuda')
            for _ in range(descriptor_size // 8)
        ),
        *(
            torch.tensor([[0., 2],
                          [-2, 0]],
                         device='cuda')
            for _ in range(descriptor_size // 16)
        ),
        *(
            torch.tensor([[0., 3],
                          [-3, 0]],
                         device='cuda')
            for _ in range(descriptor_size // 32)
        ),
        *(
            torch.tensor([[0., 4],
                          [-4, 0]],
                         device='cuda')
            for _ in range(descriptor_size // 64)
        ),
        *(
            torch.tensor([[0., 5],
                          [-5, 0]],
                         device='cuda')
            for _ in range(descriptor_size // 128)
        ),
        *(
            torch.tensor([[0., 6],
                          [-6, 0]],
                         device='cuda')
            for _ in range(descriptor_size // 128)
        ),
    )

    return generator
    

if __name__=="__main__":
    
    exp_type = "geometry_editor"
    exp_folder = "./ui_outputs/large_scale_study_optimizer/Translation_2D/16/"

    exp_folder = complete_path(exp_folder)
    exp_dict = read_exp(exp_folder)
    list_exp_details(exp_dict)

    image = exp_dict["input_image_png"]
    print(image.max(), image.min())
    # im_path = ""

    ldm_stable, tokenizer, scheduler = load_model()
    # print(ldm_stable.dtype)
    # print(ldm_stable.type)

    im_torch = preprocess_image_torch(image)
    latent_i = encode_image(im_torch, ldm_stable)

    # generator = get_generator(8)
    generator = get_generator_4()
    # print(generator.shape)
    # print(generator)
    # exit()
    steerer = ContinuousSteerer(generator)
    
    params = [{"params": steerer.parameters(), "lr": 2e-3}]
    

    optimizer = AdamW(params, weight_decay = 0.0)
    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()
    grad_accumulation_steps = 8
    num_epochs = 100000 * 6
    for ep in tqdm.tqdm(range(num_epochs)):

        # angle = np.random.randint(-180, 180)
        angle = 30



        im_r, m_r = rotate_tensor(im_torch, degree=angle)
        # im_i, m_i = rotate_tensor(im_r, mask_in=m_r, degree=-angle)

        # print(((im_i - im_torch) * m_i).abs().mean())
        # print(((im_r - im_torch) * m_r).abs().mean())
        with torch.no_grad():
            latent_i = encode_image(im_torch, ldm_stable)
            latent_r = encode_image(im_r, ldm_stable)
        m_r = torch.ones_like(latent_r)
        latent_r_i, m_r_i = rotate_tensor(latent_r, mask_in=m_r, degree=-angle)
        
        angle_rad = np.deg2rad(angle)
        if angle_rad < 0:
            angle_rad = angle_rad + 2 * np.pi

        with torch.enable_grad():
            latent_r_i = steerer(latent_r_i.permute(0, 2, 3, 1), angle_rad).permute(0, -1, 1, 2)

        d_loss = descriptor_loss(latent_r_i, latent_i, m_r_i)
        l2_loss = ((latent_r_i - latent_i) * m_r_i).abs().mean()

        # print(d_loss.item(), l2_loss.item())
        l = (d_loss) / (2.0 * grad_accumulation_steps)


        if grad_scaler is not None:
            grad_scaler.scale(l).backward()
        else:
            l.backward()

        if ep % grad_accumulation_steps == 0:
            if grad_scaler is not None:
                # grad_scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(steerer.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            print(l.item() * grad_accumulation_steps, ep, num_epochs, angle)

        # print(steerer.generator)




    # print(torch.abs(im_r - im_torch).sum())
    # latent = encode_image(im_torch, ldm_stable)


    # print(latent.shape)





    pass
