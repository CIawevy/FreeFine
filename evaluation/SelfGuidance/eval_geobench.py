import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from typing import Union, Optional, Tuple
import torch
from free_guidance import StableDiffusionFreeGuidancePipeline
from torchvision import transforms as tvt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn import init
from utils.guidance_functions import *
import argparse
from diffusers import LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DDIMInverseScheduler,AutoencoderKL
from utils import *
from PIL import Image, ImageChops
torch.cuda.manual_seed_all(1234) 
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
mpl.rcParams['image.cmap'] = 'gray_r'

def load_image(imgname: str, target_size: Optional[Union[int, Tuple[int, int]]] = None) -> torch.Tensor:
    pil_img = Image.open(imgname).convert('RGB')
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return tvt.ToTensor()(pil_img)[None, ...]  # add batch dimension


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents


class GeoData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(f"{data_dir}/annotations.json", "r") as f:
            data = json.load(f)
        
        self.data = data
        img_idx = list(data.keys())
        ## 0-50; 51-100; 101-150; 151-200; 201-250; 251-300; 301-350; 351-400; 401-450; 451-500; 501-550; 551-608
        self.instance_idx = []

        for i in range(len(img_idx)):
            img = data[img_idx[i]]
            instances = img["instances"]
            for key in instances.keys():
                for sub_key in instances[key].keys():
                    self.instance_idx.append([img_idx[i],key,sub_key])
    
    def __len__(self):
        return len(self.instance_idx)
    
    def __getitem__(self, idx):
        inst = self.instance_idx[idx]
        inst = self.data[inst[0]]["instances"][inst[1]][inst[2]]
        orig_img_path = inst["ori_img_path"]
        orig_img = np.array(Image.open(orig_img_path).convert('RGB'))
        orig_mask_path = inst["ori_mask_path"]
        tgt_mask_path = inst["tgt_mask_path"]
        tgt_mask = Image.open(tgt_mask_path).convert("L")
        orig_mask = Image.open(orig_mask_path).convert("L").resize(tgt_mask.size,Image.LANCZOS)
        mask = ImageChops.lighter(orig_mask, tgt_mask)
        tgt_mask = ((np.array(tgt_mask) >127)*255).astype(np.uint8)
        orig_mask = ((np.array(orig_mask) >127)*255).astype(np.uint8)
        mask = (np.array(mask) >127)*255
        # mask = np.empty((orig_mask.size + tgt_mask.size,), dtype=orig_mask.dtype)
        # mask[0:-1:2] = orig_mask
        # mask[1::2] = tgt_mask
        edit_param = inst["edit_param"]
        base_path = "/scratch/bcgq/yimingg8/img_edit_compile/Geo-Bench-3D/target_mask"
        store_dir = os.path.relpath(inst["tgt_mask_path"], base_path)
        store_dir = f"result_self_guide_3D/{store_dir}"
        return orig_img_path,  f"image of {inst['obj_label']}",  inst['obj_label'], edit_param, store_dir




print("Start Inference!")
# parser = argparse.ArgumentParser()
# parser.add_argument('--model_id', type=str, default="/data/zsz/models/storage_file/models/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9")
# parser.add_argument('--seed', type=int, default=None)
# args = parser.parse_args()
# ded79e214aa69e42c24d3f5ac14b76d568679cc2
model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "/data/zsz/models/storage_file/models/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
device = "cuda"
pipe = StableDiffusionFreeGuidancePipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.unet = UNetDistributedDataParallel(pipe.unet, device_ids=[0]).cuda()
# pipe.vae = UNetDistributedDataParallel(pipe.vae, device_ids=[0,1,2]).cuda()
# pipe.text_encoder = UNetDistributedDataParallel(pipe.text_encoder, device_ids=[0,1,2]).cuda()
# pipe.unet = pipe.unet.to(device)
# pipe.text_encoder = UNetDistributedDataParallel(pipe.text_encoder, device_ids=[0,1,2,3,4], output_device=3).cuda()
# pipe.unet.config, pipe.unet.dtype, pipe.unet.attn_processors, pipe.unet.set_attn_processor = pipe.unet.module.config, pipe.unet.module.dtype, pipe.unet.module.attn_processors, pipe.unet.module.set_attn_processor
# pipe.unet.config, pipe.unet.dtype = pipe.unet.module.config, pipe.unet.module.dtype
pipe.unet = pipe.unet.module
pipe = pipe.to(device)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

inversion_pipe = StableDiffusionFreeGuidancePipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
inversion_pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
inversion_pipe = inversion_pipe.to(device)
torch.backends.cudnn.benchmark = True

# seed = int(torch.rand((1,)) * 100000)
seed = 42
generator=torch.manual_seed(21533)
print(seed)

vae = inversion_pipe.vae

dataset = GeoData("/work/nvme/bcgq/yimingg8/Geo-Bench")

loader = DataLoader(dataset, batch_size=1, shuffle=False)


import time
time_total = 0
time_avg = 0
count = 0
for i, (img_path, prompt, obj, edit_param, store_dir) in enumerate(tqdm(loader)):
    input_img = load_image(img_path[0]).to(device=device).to(vae.dtype)
    latents = img_to_latents(input_img, vae)
    object_to_edit = obj

    start = time.time()
    guidance = partial(silhouette, rot=edit_param[5][0], sy=edit_param[7][0], sx=edit_param[6][0], dy=edit_param[1][0], dx=edit_param[0][0])
    inv_latents, _ = inversion_pipe(prompt=prompt[0], negative_prompt=" ", obj_to_edit = object_to_edit[0], guidance_scale=1.5,
                          width=input_img.shape[-1], height=input_img.shape[-2],
                          output_type='latent', return_dict=False, guidance_func = guidance,
                          num_inference_steps=50, max_guidance_iter_per_step=1, inv=True, latents=latents)


    image_list = pipe(prompt[0], obj_to_edit =object_to_edit[0], ref_attn = inversion_pipe.saved_attn, height=512, width=512, num_inference_steps=50, generator=generator,
        max_guidance_iter_per_step=1, guidance_func=guidance, g_weight=15, latents=inv_latents['images'])
    end = time.time()
    time_total += end - start
    count += 1
    time_avg = time_total / count
    print(f"Time taken for {i+1} images: {time_avg} seconds")

    # os.makedirs(os.path.dirname(store_dir[0]), exist_ok=True)
    # image_list[0].images[0].save(store_dir[0])

    




# ls = ['edit', 'ori']
# for i, image in enumerate(image_list):
#     image.images[0].save(f"results/{seed}_{ls[i]}.png")
# show_images([i for i in [image_list[0].images[0], image_list[1].images[0]]], titles=['edited', 'original'])

