import sys
from xml.dom.minidom import Notation
sys.path.append("/work/nvme/bcgq/yimingg8/GeoDiffuser/GeoDiffuser")
from utils.ui_utils import get_transformed_mask, get_depth, get_mask, get_edited_image
from PIL import Image
import numpy as np
import json
import os


LENGTH = 512

def resize_image_and_get_constant_depth(img):
    original_h, original_w = img.shape[0], img.shape[1]
    input_img = np.array(Image.fromarray(img).resize((LENGTH, LENGTH)))

    depth = np.ones_like(input_img)
    depth_image = np.ones_like(input_img)
    depth, depth_im_vis = get_depth(input_img, "", depth, depth_image, depth_model = "constant_depth") ## Define the depth model here 

    return input_img, depth, depth_im_vis, int(original_h), int(original_w)


# Read input image and convert to numpy array

annotations_path = "/work/nvme/bcgq/yimingg8/geobench/annotations.json"
with open(annotations_path, "r") as f:
    data = json.load(f)

for image_idx in data.keys():
    edit_indices = data[image_idx]["instances"]
    for edit_idx in edit_indices.keys():
        for sub_edit_idx in edit_indices[edit_idx].keys():
            orig_path = edit_indices[edit_idx][sub_edit_idx]["ori_img_path"]
            mask_path = edit_indices[edit_idx][sub_edit_idx]["ori_mask_path"]
            edit_param = edit_indices[edit_idx][sub_edit_idx]["edit_param"]

            input_image = Image.open(orig_path)
            input_image = np.array(input_image)

            ## Also directly copied from the Geodiffuser repo, do image resizing and get depth image. You can select the depth model to use, see the get_depth function
            input_image, depth_image, depth_image_vis, H_txt, W_txt = resize_image_and_get_constant_depth(input_image)

            ## Get the mask image
            mask_image = Image.open(mask_path).convert("L")
            mask_image = mask_image.resize((LENGTH, LENGTH), resample=Image.NEAREST)
            mask_image = np.stack([mask_image]*3, axis=-1)

            transform_in = np.eye(4)

            #trasnformed_img not used, only for visualization. transform_mat is the transformation matrix used for real editing
            trasnformed_img, transform_mat = get_transformed_mask(input_image, 
                                                mask_image, 
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
                                                background_img = None,
                                                scale_x = edit_param[6],
                                                scale_y = edit_param[7],
                                                scale_z = edit_param[8],
                                                splatting_tau = 1.0,
                                                splatting_points_per_pixel = 15,
                                                focal_length = 550)

            ## Real editing happens here
            edited_image = get_edited_image(input_image, depth_image, mask_image, transform_mat, None, guidance_scale = 7.5, skip_steps = 1, num_ddim_steps = 50, lr = 0.03, cross_replace_steps = 0.97, self_replace_steps = 0.97, latent_replace = 0.6, optimize_steps = 0.95, splatting_radius = 1.3,
                movement_sim_loss_w_self = 0.74, movement_sim_loss_w_cross = 0.5, movement_loss_w_self = 6.5, movement_loss_w_cross = 3.34, movement_removal_loss_w_self = 4.34, movement_removal_loss_w_cross = 2.67, movement_smoothness_loss_w_self = 0.0, movement_smoothness_loss_w_cross = 0.0, amodal_loss_w_cross = 3.5, amodal_loss_w_self = 80.5, splatting_tau = 1.0, splatting_points_per_pixel = 15, prompt = "", diffusion_correction = 0.0, unet_path = "runwayml/stable-diffusion-v1-5", removal_loss_value_in = -1.5,
                ldm_stable_model = None, 
                tokenizer_model = None, 
                scheduler_in = None, 
                optimize_embeddings = True,
                optimize_latents = True,
                perform_inversion = False)

            save_dir = edit_indices[edit_idx][sub_edit_idx]["coarse_input_path"].replace("geobench", "geodiffuser_output_depth_anything").replace("/coarse_img","")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            edited_image = Image.fromarray(edited_image).save(save_dir)


## Also directly copied from the Geodiffuser repo, do image resizing and get depth image. You can select the depth model to use, see the get_depth function
input_image, depth_image, depth_image_vis, H_txt, W_txt = resize_image_and_get_constant_depth(input_image)

## Get the mask image
mask_image = Image.open(mask_path).convert("L")
mask_image = mask_image.resize((LENGTH, LENGTH), resample=Image.NEAREST)
mask_image = np.stack([mask_image]*3, axis=-1)

transform_in = np.eye(4)

#trasnformed_img not used, only for visualization. transform_mat is the transformation matrix used for real editing
trasnformed_img, transform_mat = get_transformed_mask(input_image, 
                                    mask_image, 
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
                                    background_img = None,
                                    scale_x = edit_param[6],
                                    scale_y = edit_param[7],
                                    scale_z = edit_param[8],
                                    splatting_tau = 1.0,
                                    splatting_points_per_pixel = 15,
                                    focal_length = 550)

## Real editing happens here
edited_image = get_edited_image(input_image, depth_image, mask_image, transform_mat, None, guidance_scale = 7.5, skip_steps = 1, num_ddim_steps = 50, lr = 0.03, cross_replace_steps = 0.97, self_replace_steps = 0.97, latent_replace = 0.6, optimize_steps = 0.95, splatting_radius = 1.3,
    movement_sim_loss_w_self = 0.74, movement_sim_loss_w_cross = 0.5, movement_loss_w_self = 6.5, movement_loss_w_cross = 3.34, movement_removal_loss_w_self = 4.34, movement_removal_loss_w_cross = 2.67, movement_smoothness_loss_w_self = 0.0, movement_smoothness_loss_w_cross = 0.0, amodal_loss_w_cross = 3.5, amodal_loss_w_self = 80.5, splatting_tau = 1.0, splatting_points_per_pixel = 15, prompt = "", diffusion_correction = 0.0, unet_path = "runwayml/stable-diffusion-v1-5", removal_loss_value_in = -1.5,
    ldm_stable_model = None, 
    tokenizer_model = None, 
    scheduler_in = None, 
    optimize_embeddings = True,
    optimize_latents = True,
    perform_inversion = False)

# edited_image = Image.fromarray(edited_image).save("/work/nvme/bcgq/yimingg8/geodiffuser_output_2D/212/0/9.png")

