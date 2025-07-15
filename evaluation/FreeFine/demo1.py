import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import DDIMScheduler
import random
project_root = Path.cwd().parent
# sys.path.append(str(project_root))
os.chdir(project_root)
from src.demo.model import FreeFinePipeline
from src.utils.attention import (
    Attention_Modulator,
    register_attention_control_4bggen
)
from src.utils.vis_utils import temp_view,temp_view_img,load_json,get_constrain_areas,prepare_mask_pool,re_edit_2d,dilate_mask,read_and_resize_mask,read_and_resize_img

"""
Load FreeFine
"""

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/" #Replace with your own ckpt path
model = FreeFinePipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16).to(device)
model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
controller = Attention_Modulator()
model.controller = controller
register_attention_control_4bggen(model, controller)
model.modify_unet_forward()
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
"""
Case selection 
"""
choice_dict = {
    '0':['Examples/Removal/airplane/source.png','Examples/Removal/airplane/mask.png','empty snowfield scene'],
    '1':['Examples/Removal/rhino/source.png','Examples/Removal/rhino/mask.png','empty forest scene'],
    '2':['Examples/Removal/cat/source.png','Examples/Removal/cat/mask.png','empty wall scene'],
    '3':['Examples/Removal/man/source.png','Examples/Removal/man/mask.png','empty modern street scene'],
    '4':['Examples/Removal/woman/source.png','Examples/Removal/woman/mask.png','empty street scene'],
}
# sample_id = '0'
sample_id = random.choice(list(choice_dict.keys()))
ori_img_path,ori_mask_path,prompt = choice_dict[sample_id]
ori_img_path ="/data/zkl/geovis/rotate_supply/139/source.png"
ori_mask_path = "/data/zkl/geovis/rotate_supply/139/s_mask.png"
prompt = 'wall'

ori_img = read_and_resize_img(ori_img_path)
ori_mask = read_and_resize_mask(ori_mask_path)
mask_pool = [ori_mask] #add more object mask if you have, to avoid inpainting on them
constrain_areas = get_constrain_areas(mask_list=mask_pool,ori_mask=ori_mask)
dilation_factor = 20
dil_ori_mask = dilate_mask(ori_mask, dilation_factor)
dil_ori_mask = np.where(constrain_areas,0,dil_ori_mask)
Image.fromarray(dil_ori_mask).save('obj_remove_mask.png')
temp_view_img(ori_img,'ori_img')
temp_view(dil_ori_mask,'dil_ori_mask')
seed_r = random.randint(0, 10 ** 16)
# seed_r = 42
generated_results = model.FreeFine_background_generation(ori_img, dil_ori_mask, prompt,
                                                         guidance_scale=3.5,eta=1.0, end_step=50,
                                                         num_step=50, end_scale=0.5,
                                                         start_step=2, share_attn=True, method_type='tca',
                                                         local_text_edit=True,
                                                         local_perturbation=True, verbose=True,
                                                         seed=seed_r,
                                                         return_intermediates=False,latent_blended=False,
                                                         )
temp_view_img(generated_results,'bg_img')
save_img = Image.fromarray(generated_results)
save_img.save("/data/Hszhu/139_bg_img.png")
