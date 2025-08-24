import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path  
import numpy as np
import torch
import cv2
from PIL import Image
import random
# project_root = Path.cwd().parent  
sys.path.append('/mnt/bn/ocr-doc-nas/zhuhanshen/iccv/FreeFine')
# os.chdir(project_root)
os.chdir('/mnt/bn/ocr-doc-nas/zhuhanshen/iccv/FreeFine')# Replace with your path
print(f"当前工作目录: {os.getcwd()}")
assert os.getcwd().split('/')[-1] == 'FreeFine', "Current working directory is not FreeFine"
from src.demo.model import FreeFinePipeline
from src.utils.attention import register_attention_control,Attention_Modulator,register_attention_control_4bggen
from src.utils.vis_utils import temp_view,temp_view_img,load_json,get_constrain_areas,prepare_mask_pool,re_edit_2d,dilate_mask,read_and_resize_mask,re_edit_3d,read_and_resize_img,read_and_resize_mask_from_pil,read_and_resize_img_from_pil
from diffusers import DDIMScheduler
from datasets import load_dataset
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# pretrained_model_path = "/mnt/bn/ocr-doc-nas/zhuhanshen/models/stable-diffusion-v1-5" #Replace with your own ckpt path
# model = FreeFinePipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16).to(device)
# model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)

edit_type = 'SC' #Replace with '3D' or 'SC'
if edit_type == '3D':
    type_3d ='depth' #Replace with 'sv3d'
# For 2D-Editing
# all_data = load_dataset("/mnt/bn/ocr-doc-nas/zhuhanshen/data/CIawevy/GeoBench")
if edit_type == '2D':
    dataset = load_dataset("/mnt/bn/ocr-doc-nas/zhuhanshen/data/CIawevy/GeoBench", "2d")['data']    
elif edit_type == '3D':
    dataset = load_dataset("/mnt/bn/ocr-doc-nas/zhuhanshen/data/CIawevy/GeoBench", "3d")['data']
elif edit_type == 'SC':
    dataset = load_dataset("/mnt/bn/ocr-doc-nas/zhuhanshen/data/CIawevy/GeoBench", "sc")['data']


id = random.choice(range(len(dataset)))
print(f'current sample is ID:{id}')
sample = dataset[id]
if edit_type == '2D':
    #debug ID=4865
    id = 4865
    edit_param = sample['edit_param']
    ori_img = read_and_resize_img_from_pil(sample['ori_img'])
    coarse_input = read_and_resize_img_from_pil(sample['coarse_input'])

    ori_mask = read_and_resize_mask_from_pil(sample['ori_mask'])
    tgt_mask = read_and_resize_mask_from_pil(sample['tgt_mask'])
    obj_label = sample['obj_label']
elif edit_type == '3D':
    edit_param = sample['edit_param']
    ori_img = read_and_resize_img_from_pil(sample['ori_img'])
    ori_mask = read_and_resize_mask_from_pil(sample['ori_mask'])
    obj_label = sample['obj_label']
    if type_3d == 'depth':
        coarse_input = sample['coarse_input_0']
        tgt_mask = read_and_resize_mask_from_pil(sample['target_mask_0'])
        draw_mask = read_and_resize_mask_from_pil(sample['draw_mask'])
    elif type_3d == 'sv3d':
        coarse_input = sample['coarse_input_1']
        tgt_mask = read_and_resize_mask_from_pil(sample['target_mask_1'])
        draw_mask = None
    else:
        raise ValueError(f'Unknown 3D type: {type_3d}')
elif edit_type == 'SC':
    edit_param = None
    ori_img = read_and_resize_img_from_pil(sample['ori_img'])
    coarse_input = sample['coarse_input']
    ori_mask = read_and_resize_mask_from_pil(sample['ori_mask'])
    tgt_mask = read_and_resize_mask_from_pil(sample['tgt_mask'])
    obj_label = sample['obj_label']
    draw_mask = read_and_resize_mask_from_pil(sample['draw_mask'])
