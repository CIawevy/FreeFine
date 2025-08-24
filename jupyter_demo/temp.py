import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
from pathlib import Path  
import numpy as np
import torch
import cv2
from PIL import Image
import random
# project_root = Path.cwd().parent  
# # sys.path.append(str(project_root))
# os.chdir(project_root)
sys.path.append('/mnt/bn/ocr-doc-nas/zhuhanshen/iccv/FreeFine')# Replace with your path
os.chdir('/mnt/bn/ocr-doc-nas/zhuhanshen/iccv/FreeFine')# Replace with your path
print(f"当前工作目录: {os.getcwd()}")
assert os.getcwd().split('/')[-1] == 'FreeFine', "Current working directory is not FreeFine"
from src.demo.model import FreeFinePipeline
from src.utils.attention import register_attention_control,Attention_Modulator,register_attention_control_4bggen
from src.utils.vis_utils import temp_view,temp_view_img,load_json,get_constrain_areas,prepare_mask_pool,re_edit_2d,dilate_mask,read_and_resize_mask,re_edit_3d,read_and_resize_img,read_and_resize_mask_from_pil,read_and_resize_img_from_pil
from diffusers import DDIMScheduler
from datasets import load_dataset
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
pretrained_model_path = "/mnt/bn/ocr-doc-nas/zhuhanshen/models/stable-diffusion-v1-5" #Replace with your own ckpt path
model = FreeFinePipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16).to(device)
model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
edit_type = '3D'
if edit_type == '3D':
    # dataset_3d = load_dataset("CIawevy/GeoBench", "3d")['data']
    dataset_3d = load_dataset("/mnt/bn/ocr-doc-nas/zhuhanshen/data/CIawevy/GeoBench", "3d")['data']
    type_3d ='sv3d'
    #type_3d ='depth' 
edit_type = '3D'
if edit_type == '3D':
    #debug ID=xxx
    id = random.choice(range(len(dataset_3d)))
    print(f'current sample is ID:{id}')
    sample = dataset_3d[id] 
    edit_param = sample['edit_param']
    ori_img = read_and_resize_img_from_pil(sample['ori_img'])
    ori_mask = read_and_resize_mask_from_pil(sample['ori_mask'])
    obj_label = sample['obj_label']
    if type_3d == 'sv3d':
        coarse_input = read_and_resize_img_from_pil(sample['coarse_input_1'])
        tgt_mask = read_and_resize_mask_from_pil(sample['target_mask_1'])
        draw_mask = None
    elif type_3d == 'depth':
        coarse_input = read_and_resize_img_from_pil(sample['coarse_input_0'])
        tgt_mask = read_and_resize_mask_from_pil(sample['target_mask_0'])   
        draw_mask = read_and_resize_mask_from_pil(sample['draw_mask'])
    

temp_view_img(ori_img,'ori_img')
temp_view_img(coarse_input,'coarse_input')
controller = Attention_Modulator(start_layer=10)
model.controller = controller
register_attention_control(model, controller)
model.modify_unet_forward()
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
seed_r = random.randint(0, 10 ** 16)
params = {
    "ori_img": ori_img,
    "ori_mask": ori_mask,
    "coarse_input": coarse_input,
    "target_mask": tgt_mask,
    "guidance_text": obj_label,
    "guidance_scale": 7.5,
    "eta": 1.0,
    "end_scale": 0.0,
    "end_step": 50,
    "num_step": 50,
    "start_step": 15,
    "seed": 42,
    "draw_mask": draw_mask,
    "return_intermediates" : False,
    "use_auto_draw" : True,
    "reduce_inp_artifacts" : True,
    "cons_area" : tgt_mask,
}

# 生成结果
generated_results = model.FreeFine_generation(**params)
temp_view_img(generated_results)
