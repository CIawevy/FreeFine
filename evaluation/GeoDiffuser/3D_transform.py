import sys
sys.path.append("/work/nvme/bcgq/yimingg8/GeoDiffuser/GeoDiffuser")
from utils.ui_utils import get_transformed_mask, get_depth, get_mask
from PIL import Image
import numpy as np


LENGTH = 512

## A function direclt copied from the geo_diff_ui.py file
def resize_image_and_get_constant_depth(img):
    original_h, original_w = img.shape[0], img.shape[1]
    input_img = np.array(Image.fromarray(img).resize((LENGTH, LENGTH)))

    depth = np.ones_like(input_img)
    depth_image = np.ones_like(input_img)
    depth, depth_im_vis = get_depth(input_img, "", depth, depth_image, depth_model = "constant_depth")

    return input_img, depth, depth_im_vis, int(original_h), int(original_w)

# Read input image and convert to numpy array
input_image = Image.open("/work/nvme/bcgq/yimingg8/GeoDiffuser/assets/dog.png")
input_image = np.array(input_image)

## Also directly copied from the Geodiffuser repo, do image resizing and gte depth image. You can select the depth model to use, see the get_depth function
input_image, depth_image, depth_image_vis, H_txt, W_txt = resize_image_and_get_constant_depth(input_image)

### This line of code cannot direct run since I didn't write the interactive UI to get points, but the idea is to get the mask from SAM, which is not hard to implement
mask_image = get_mask(input_image, mask_image, selected_points, sam_path)

transform_in = np.eye(4)

trasnformed_img, transform_mat = get_transformed_mask(input_img, 
                                    mask_image, 
                                    depth_image,
                                    transform_mask, # Basically a None when I checked
                                    translation_x, 
                                    translation_y, 
                                    translation_z, 
                                    rotation_x, 
                                    rotation_y, 
                                    rotation_z,
                                    transform_in, # See above, basically an identity matrix, it will be modified in this function
                                    splatting_radius = 1.3, 
                                    background_img = None,
                                    scale_x = 1.0,
                                    scale_y = 1.0,
                                    scale_z = 1.0,
                                    splatting_tau = 1.0,
                                    splatting_points_per_pixel = 15,
                                    focal_length = 550)