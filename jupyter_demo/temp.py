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
from evaluation.GeoDiffuser.GeoDiffuser.utils.ui_utils2 import get_transformed_mask, get_depth



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
pretrained_model_path = "/mnt/bn/ocr-doc-nas/zhuhanshen/models/stable-diffusion-v1-5" #Replace with your own ckpt path
model = FreeFinePipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16).to(device)
model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
# Using GeoBench as example data 
edit_type = '3D'
if edit_type == '3D':
     # dataset_3d = load_dataset("CIawevy/GeoBench", "3d")['data']
    dataset_3d = load_dataset("/mnt/bn/ocr-doc-nas/zhuhanshen/data/CIawevy/GeoBench", "3d")['data']
edit_type = '3D'
if edit_type == '3D':
    #debug ID=62
    id = random.choice(range(len(dataset_3d)))
    id = 62
    print(f'current sample is ID:{id}')
    sample = dataset_3d[id] 
    edit_param = sample['edit_param']
    ori_img = read_and_resize_img_from_pil(sample['ori_img'])
    ori_mask = read_and_resize_mask_from_pil(sample['ori_mask'])
    obj_label = sample['obj_label']
    #produce yourself
    # if type_3d == 'sv3d': 
    #     coarse_input = read_and_resize_img_from_pil(sample['coarse_input_1'])
    #     tgt_mask = read_and_resize_mask_from_pil(sample['target_mask_1'])
    #     draw_mask = None
    # elif type_3d == 'depth':
    #     coarse_input = read_and_resize_img_from_pil(sample['coarse_input_0'])
    #     tgt_mask = read_and_resize_mask_from_pil(sample['target_mask_0'])   
    #     draw_mask = read_and_resize_mask_from_pil(sample['draw_mask'])
    
temp_view_img(ori_img,'ori_img')
# temp_view_img(coarse_input,'coarse_input')
"""
Step1:Background Generation
"""
constrain_areas = get_constrain_areas(mask_list=[ori_mask],ori_mask=ori_mask) #for practice use you can upload mask list with all the object mask to avoid dilation on other objects
dilation_factor = 20
dil_ori_mask = dilate_mask(ori_mask, dilation_factor)
dil_ori_mask = np.where(constrain_areas,0,dil_ori_mask)
temp_view_img(ori_img,'ori_img')
temp_view(dil_ori_mask,'dil_full_mask')

controller = Attention_Modulator()
model.controller = controller
register_attention_control_4bggen(model, controller)
model.modify_unet_forward()
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
seed_r = random.randint(0, 10 ** 16)
# seed_r = 42
generated_results = model.FreeFine_background_generation(ori_img, dil_ori_mask, 'empty scene',
                                                         guidance_scale=3.5,eta=1.0, end_step=50,
                                                         num_step=50, end_scale=0.5,
                                                         start_step=1, share_attn=True, method_type='tca',
                                                         local_text_edit=True,
                                                         local_perturbation=True, verbose=True,
                                                         seed=seed_r,
                                                         return_intermediates=False,latent_blended=False,
                                                         )  


blended = True
if blended:
    #implement fom Brushnet
    mask_blurred = cv2.GaussianBlur(dil_ori_mask, (1, 1), 0) / 255
    mask_np = 1 - (1 - dil_ori_mask) * (1 - mask_blurred)
    image_pasted = ori_img * (1 - mask_np) + generated_results * mask_np
    image_pasted = image_pasted.astype(generated_results.dtype)
    temp_view_img(image_pasted,'bg_img')
    inp_back_ground = image_pasted
else:
    temp_view_img(generated_results,'bg_img')
    # save_img = Image.fromarray(generated_results)
    # save_img.save("FreeFine/jupyter_demo、bg_img.png") #replace with your save path
    inp_back_ground = generated_results


"""
Step2:Depth-based 3D editing(coarse)
"""
## Modified from the Geodiffuser repo, do image resizing and get depth image. You can select the depth model to use, see the get_depth function
LENGTH = 512

def resize_image_and_get_constant_depth(img):
    original_h, original_w = img.shape[0], img.shape[1]
    input_img = np.array(Image.fromarray(img).resize((LENGTH, LENGTH)))

    depth = np.ones_like(input_img)
    depth_image = np.ones_like(input_img)
    depth, depth_im_vis = get_depth(input_img, "", depth, depth_image, depth_model = "depth_anything") 
    

    return input_img, depth, depth_im_vis, int(original_h), int(original_w)


edit_param = edit_param #replace with your param [dx,dy,dz,rx,ry,rz,sx,sy,sz]
input_image, depth_image, depth_image_vis, H_txt, W_txt = resize_image_and_get_constant_depth(ori_img)


transform_in = np.eye(4)

#transformed_img not used, only for visualization. transform_mat is the transformation matrix used for real editing
transformed_img, mesh_mask,full_mask,point_correspondence = get_transformed_mask(input_image,
                                    ori_mask, 
                                    depth_image,
                                    None, # Basically a None when I checked
                                    translation_x=edit_param[0]/LENGTH, 
                                    translation_y=edit_param[1]/LENGTH, 
                                    translation_z=edit_param[2]/LENGTH, 
                                    rotation_x=edit_param[3], 
                                    rotation_y=edit_param[4], 
                                    rotation_z=edit_param[5],
                                    transform_in=transform_in, # See above, basically an identity matrix, it will be modified in this function
                                    splatting_radius = 1.3, 
                                    background_img = inp_back_ground,
                                    scale_x = edit_param[6],
                                    scale_y = edit_param[7],
                                    scale_z = edit_param[8],
                                    splatting_tau = 1.0,
                                    splatting_points_per_pixel = 15,
                                    focal_length = 550)
md_mask = np.where(mesh_mask,0,full_mask)

coarse_input, target_mask, draw_mask = transformed_img, mesh_mask, md_mask

# temp_view_img(ori_img_3d)
temp_view_img(coarse_input) 
temp_view(target_mask)

