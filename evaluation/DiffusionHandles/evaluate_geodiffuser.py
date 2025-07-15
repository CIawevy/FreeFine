import os
import glob
from vis_utils import  temp_view,temp_view_img
import torch
import scipy

# from test.utils import crop_and_resize, load_image, save_image, save_depth
from saicinpainting import LamaInpainter

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

import matplotlib.pyplot as plt
import numpy as np
import imageio
from diffhandles import DiffusionHandles

from scipy.spatial.transform import Rotation as R
import cv2
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dpt import DepthAnything

# Borrowed and Edited from: 
# 1. https://github.com/adobe-research/DiffusionHandles/blob/main/test/remove_foreground.py
# 2. https://github.com/adobe-research/DiffusionHandles/blob/main/test/estimate_depth.py
# 3. 
# Thank you to the authors for the code!


import torch
import torchvision
import imageio.v3 as imageio
import imageio.plugins as imageio_plugins

imageio_plugins.freeimage.download() # to load exr files

def load_image(path: str) -> torch.Tensor:
    # img = Image.open(path)
    # img = img.convert('RGB')
    # img = torchvision.transforms.functional.pil_to_tensor(img)

    img = torch.from_numpy(imageio.imread(path))
    if img.dim() == 2:
        img = img[..., None]
    img = img.to(dtype=torch.float32)
    img = img.permute(2, 0, 1)
    img = img / 255.0
    return img

def save_image(img: torch.Tensor, path: str):
    # img = torchvision.transforms.functional.to_pil_image(img)
    # img.save(path)

    img = img.detach().cpu()
    img = img * 255.0
    img = img.permute(1, 2, 0)
    img = img.to(dtype=torch.uint8)
    if img.shape[-1] == 1:
        img = img[..., 0]
    imageio.imwrite(path, img.numpy())
    
def load_depth(path: str) -> torch.Tensor:
    # depth = Image.open(path)
    # depth = torchvision.transforms.functional.pil_to_tensor(depth)[None,...]

    depth = torch.from_numpy(imageio.imread(path))
    if depth.dim() == 2:
        depth = depth[..., None]
    depth = depth.to(dtype=torch.float32)
    depth = depth.permute(2, 0, 1)
    return depth

def save_depth(depth: torch.Tensor, path: str):
    # depth = torchvision.transforms.functional.to_pil_image(depth, mode='F')
    # depth.save(path)

    depth = depth.detach().cpu()
    depth = depth.permute(1, 2, 0)
    depth = depth.to(dtype=torch.float32)
    depth = depth[..., 0]
    imageio.imwrite(path, depth.numpy())

def crop_and_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    if img.shape[-2] != img.shape[-1]:
        img = torchvision.transforms.functional.center_crop(img, min(img.shape[-2], img.shape[-1]))
    img = torchvision.transforms.functional.resize(img, size=(size, size), antialias=True)
    return img



DIFF_HANDLES_MODEL = None
DEPTH_MODEL = None
INPAINTING_MODEL = None
DEPTH_ANYTHING_MODEL = None

def count_folders(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

def create_folder(directory):
    os.makedirs(directory, exist_ok = True)

def complete_path(directory):
    return os.path.join(directory, "")


def read_image(im_path):

    im = plt.imread(im_path)
    # print(im.min(), im.max(), im.shape, im_path)

    if len(im.shape) == 3:
        im = im[..., :3]
    
    if im.max() <= 1.0:
        im = (im * 255.0).astype("uint8")
    # print(im.min(), im.max(), im.shape, im_path)
    return im


def read_txt_file(f_path):

    with open(f_path, "r") as f:
        txt = f.read()

    return txt

def file_exists(f_path):
    return os.path.exists(f_path)

def read_exp(d_path):
    save_folder = complete_path(d_path)


    img_path = save_folder + "input_image.png"
    depth_path = save_folder + "depth.npy"
    mask_path = save_folder + "input_mask.png"
    bg_path = save_folder + "background_image.png"
    depth_vis_path = save_folder + "depth.png"
    transform_path = save_folder + "transform.npy"
    im_shape = save_folder + "image_shape.npy"
    
    transformed_image_path = save_folder + "transformed_image.png"
    result_path = save_folder + "result.png"
    result_ls_path = save_folder + "resized_result_ls.png"
    zero123_result_path = save_folder + "zero123/lama_followed_by_zero123_result.png"
    object_edit_result_path = save_folder + "object_edit/result_object_edit.png"
    resized_input_image = save_folder + "resized_input_image_png.png"
    resized_input_mask = save_folder + "resized_input_mask_png.png"

    prompt_path = save_folder + "prompt.txt"
    
    all_paths = [img_path, depth_path, mask_path, bg_path, depth_vis_path, transform_path, transformed_image_path, result_path, im_shape, result_ls_path, zero123_result_path, resized_input_image, object_edit_result_path, resized_input_mask, prompt_path]
    
    out_dict = {}
    for f_name in all_paths:
        base_name = os.path.basename(f_name)
        key_name = base_name.split(".")[0]
        f_type = base_name.split(".")[1]
        

        if file_exists(f_name):
            if f_type == "png":
                out_dict[key_name + "_png"] = read_image(f_name)
            elif f_type == "npy":
                out_dict[key_name + "_npy"] = np.load(f_name)
            elif f_type == "txt":
                out_dict[key_name + "_txt"] = read_txt_file(f_name)
        else:
            out_dict[key_name + "_" + f_type] = None
    if out_dict["image_shape_npy"] is None:
        out_dict["image_shape_npy"] = np.array([512, 512])
    out_dict["path_name"] = d_path
    return out_dict


IMG_RES = 512


def preprocess_image(image_in, img_res=IMG_RES):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.from_numpy(image_in).unsqueeze(0).permute(0, -1, 1, 2)
    # print(image.shape)
    image = crop_and_resize(img=image, size=img_res).to(device).to(torch.float32)
    
    return image

def remove_foreground(image, fg_mask, img_res=IMG_RES, dilation=10):

    """
    Both image and fg_mask in range [0-1]
    """
    
    global INPAINTING_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if INPAINTING_MODEL is None:
        inpainter = LamaInpainter()
        inpainter.to(device)
        INPAINTING_MODEL = inpainter
    else:
        inpainter = INPAINTING_MODEL

    image = preprocess_image(image, img_res)
    # print(image.shape)

    fg_mask = preprocess_image(fg_mask, img_res)
    
    fg_mask = (fg_mask>0.5).to(device=device, dtype=torch.float32)


    # inpaint the foreground region to get a background image without the foreground object
    if dilation >= 0:
        fg_mask = fg_mask.cpu().numpy() > 0.5
        fg_mask = scipy.ndimage.binary_dilation(fg_mask[0, 0], iterations=dilation)[None, None, ...]
        fg_mask = torch.from_numpy(fg_mask).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        bg_img = inpainter.inpaint(image=image, mask=fg_mask)
    
    return bg_img[0]

@torch.no_grad()
def get_monocular_depth_anything(image, encoder = "vitl", translate_factor=0.1):

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    global DEPTH_ANYTHING_MODEL
    transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        
    if DEPTH_ANYTHING_MODEL is None:
        depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)
        DEPTH_ANYTHING_MODEL = depth_anything

    else:
        depth_anything = DEPTH_ANYTHING_MODEL


    image = image.astype("uint8") / 255.0
        
    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    depth = depth_anything(image)

    depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    # Converts from relative to absolute depth. This works better than using 1/depth
    depth = depth.max() - depth
    

    # Translation factor computation
    # # Pushes object farther off to reduce smearing
    # depth = depth + depth.max() * translate_factor

    return depth


def estimate_depth(image, img_res=IMG_RES):

    global DEPTH_MODEL, DEPTH_ANYTHING_MODEL

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    d_depth_anything = get_monocular_depth_anything(image)

    if DEPTH_MODEL is None:


        conf = get_config("zoedepth_nk", "infer")
        depth_estimator = build_model(conf)
        depth_estimator.to(device)
        DEPTH_MODEL = depth_estimator
    else:
        depth_estimator = DEPTH_MODEL

    image_d = preprocess_image(image)

    with torch.no_grad():
        depth = depth_estimator.infer(image_d)

    
    # print(depth.shape, d_depth_anything.shape)

    return depth[0], d_depth_anything[None]
    

def load_diffhandles_model(config_path=None):

    global DIFF_HANDLES_MODEL


    if DIFF_HANDLES_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diff_handles_config = OmegaConf.load(config_path) if config_path is not None else None
        diff_handles = DiffusionHandles(conf=diff_handles_config)
        diff_handles.to(device)
        DIFF_HANDLES_MODEL = diff_handles
    else:
        diff_handles = DIFF_HANDLES_MODEL

    return diff_handles

def latent_to_image(latent_image, diff_handles):

    # save image reconstructed from inversion
    with torch.no_grad():
        latent_image = 1 / 0.18215 * latent_image.detach()
        recon_image = diff_handles.diffuser.vae.decode(latent_image)['sample']
        recon_image = (recon_image + 1) / 2

    return recon_image


def convert_transform_to_diffhandles(transform_in):

    # Convert to and from pytorch3d coordinate frame
    M = np.eye(4)
    M[0, 0] = -1.0
    M[1, 1] = -1.0
    transform = np.linalg.inv(M) @ transform_in @ M

    # transform = transform_in @ M
    translation = list(transform[:3, -1])
    rotation = transform[:3, :3]
    rot_scipy = R.from_matrix(rotation)
    axis = rot_scipy.as_rotvec(degrees=True)

    angle = np.linalg.norm(axis) + 1e-8


    axis = (axis / angle)

    if np.linalg.norm(axis) < 0.1:
        axis = np.array([0.0, 1.0, 0.0])

    axis = list(axis)
    return axis, angle, translation

def resize_image(image, aspect_ratio):

    # h, w = image.shape[:2]
    ratio = aspect_ratio[1] / aspect_ratio[0]
    h, w = 512, 512

    if ratio < 1:
        new_h, new_w = h / ratio, w
    else:
        new_h, new_w = h, ratio * w

    img = cv2.resize(image, (int(new_w),int(new_h)))

    # input_img = np.array(Image.fromarray(img).resize((w, h), Image.NEAREST))
    return img


def get_depth_error(depth_data, depth, mask):

    return torch.mean(torch.abs(depth_data - depth)[mask >= 0.5])

def get_depth_translation_factor_and_error_geodiffuser(depth_data, depth, mask):
    # Note here we expect BACKGROUND mask not FOREGROUND MASK!
    d_max = depth.max()
    # depth + d_max * t = depth_data
    t_factor = torch.mean((depth_data - depth)[mask >= 0.5]) / d_max
    d_new = depth + d_max * t_factor
    d_error = get_depth_error(depth_data, d_new, mask)

    print("[INFO]: t_factor: ", t_factor, " and d_error: ", d_error)
    return t_factor, d_error, d_new



def get_best_depth(depth_data_in, d_depth_zoe, d_depth_anything, mask_in):

    depth_data = depth_data_in[0]
    mask = mask_in[0]
    print("[info]: depths: ", depth_data.max(), d_depth_zoe.max(), d_depth_anything.max(), mask.max(), mask.min(), mask.shape)
    
    t_factor, d_error, d_1 = get_depth_translation_factor_and_error_geodiffuser(depth_data, d_depth_anything, mask)

    t_factor_2, d_error_2, d_2 = get_depth_translation_factor_and_error_geodiffuser(depth_data, d_depth_zoe, mask)


    if d_error < d_error_2:
        return d_1
    else:
        return d_2

def run_geodiff_folder(exp_dict, diff_handles):

    print("[INFO]: Running Diffusion Handles on exp: ", exp_dict["path_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir_dh = exp_dict["path_name"] + "diffhandles/"
    os.makedirs(exp_dir_dh, exist_ok=True)
    
    image = exp_dict["input_image_png"] / 255.0
    mask = exp_dict["input_mask_png"][..., :1] / 255.0

    prompt = exp_dict["prompt_txt"]

    if prompt is None:
        prompt = ""
    print("[INFO]: prompt: ", prompt)
    # exit()

    rot_axis, rot_angle, translation = convert_transform_to_diffhandles(exp_dict["transform_npy"])

    # print(rot_angle, rot_axis, translation)

    # exit()

    im_removed = remove_foreground(image, mask)
    im_removed_np = im_removed.permute(1, 2, 0).detach().cpu().numpy()

    imageio.imwrite(exp_dir_dh + "im_removed.png", (im_removed_np * 255.0).astype(np.uint8))

    im_removed_depth, im_removed_depth_anything = estimate_depth(im_removed_np)

    fg_mask = preprocess_image(mask)

    depth = preprocess_image(exp_dict["depth_npy"][..., None])
    im_removed_depth = get_best_depth(depth, im_removed_depth, im_removed_depth_anything, 1.0 - fg_mask)
    # exit()
    bg_depth = im_removed_depth

    is_2D_transform = False
    if np.all(depth[0,0].detach().cpu().numpy() == 0.5):
        is_2D_transform = True
    # exit()
    # Normalize depth
    depth = depth / (depth.max() + 1e-8) + 1e-2
    depth[depth > 0.95] = 1.0
    bg_depth = bg_depth / (bg_depth.max() + 1e-8) + 1e-2

    if is_2D_transform:
        print(fg_mask.shape, depth.shape)
        depth[fg_mask > 0.5] = 0.5

    depth_2D = None
    if is_2D_transform:
        print("[INFO]: 2D Transform")
        # depth_2D = depth
        # depth = estimate_depth(image)[None]
        # bg_depth = diff_handles.set_foreground(depth=depth, fg_mask=fg_mask, bg_depth=(bg_depth[None]))

        # bg_depth = depth
    # else:
    bg_depth = diff_handles.set_foreground(depth=depth, fg_mask=fg_mask, bg_depth=(bg_depth[None]))

    # exit()
    im_removed_depth_np = bg_depth[0, 0].detach().cpu().numpy()
    np.save(exp_dir_dh + "bg_depth_diffhandles.npy", im_removed_depth_np)
    im_removed_depth_np_norm = im_removed_depth_np / (im_removed_depth_np.max() + 1e-6)
    imageio.imwrite(exp_dir_dh + "im_removed_depth.png", (im_removed_depth_np_norm * 255.0).astype(np.uint8))



    img = preprocess_image(image)

    null_text_emb, init_noise = diff_handles.invert_input_image(img, depth, prompt)
    null_text_emb, init_noise, activations, latent_image = diff_handles.generate_input_image(
                depth=depth, prompt=prompt, null_text_emb=null_text_emb, init_noise=init_noise)


    recon_image = latent_to_image(latent_image, diff_handles)
    save_image(recon_image.clamp(min=0, max=1)[0], exp_dir_dh + "recon.png")
    # print("Saved Reconstruction Image for Check")



    # get transformation parameters
    translation = torch.tensor(translation, dtype=torch.float32)
    rot_axis = torch.tensor(rot_axis, dtype=torch.float32) 
    rot_angle = float(rot_angle) 

    results = diff_handles.transform_foreground(
    depth=depth, prompt=prompt,
    fg_mask=fg_mask, bg_depth=bg_depth,
    null_text_emb=null_text_emb, init_noise=init_noise,
    activations=activations,
    rot_angle=rot_angle, rot_axis=rot_axis, translation=translation,
    use_input_depth_normalization=False)

    if diff_handles.conf.guided_diffuser.save_denoising_steps:
        edited_img, edited_disparity, denoising_steps = results
    else:
        edited_img, edited_disparity = results
        denoising_steps = None

    save_image((edited_disparity/edited_disparity.max())[0], exp_dir_dh + "im_disparity_transformed.png")
    save_image(edited_img[0], exp_dir_dh + "im_edited_diffhandles_square.png")


    resized_edit_image = resize_image(edited_img[0].permute(1, 2, 0).detach().cpu().numpy(), exp_dict["image_shape_npy"])


    plt.imsave(exp_dir_dh + "im_edited_diffhandles.png", resized_edit_image)
    print("[INFO]: Saving Edited Image to location: ", exp_dir_dh + "im_edited_diffhandles.png")
    
def get_exp_types():
    
    exp_types = ["Removal", "Rotation_3D", "Rotation_2D", "Translation_3D", "Scaling", "Mix", "Translation_2D"]

    return exp_types

def check_if_exp_root(exp_root_folder, folder_list = None):

    if folder_list is None:    
        folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    
    exp_types = get_exp_types()


    for f in folder_list:
        # print(f.split("/"))
        if f.split("/")[-2] in exp_types:
            return True

    return False



def run_geodiff(exp_root_folder, diff_handles):

    folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    folder_list.sort()

    # print(folder_list)

    if check_if_exp_root(exp_root_folder):
        root_folders_list = folder_list
        for f in root_folders_list:
            folder_list = glob.glob(complete_path(f) + "**/")
            folder_list.sort()

            exp_cat = f.split("/")[-2]
            # if not (exp_cat == "Translation_2D"):
            #     continue
            if exp_cat == "Removal" or exp_cat == "Translation_2D":
                continue


            for exp_folder in folder_list:

                # run_dragon_diffusion_single(exp_folder, dragon_diff_model)
                exp_dict = read_exp(exp_folder)
                run_geodiff_folder(exp_dict, diff_handles)
    
            # exit()
    else:
        for exp_folder in folder_list:
            exp_dict = read_exp(exp_folder)
            run_geodiff_folder(exp_dict, diff_handles)

if __name__ == '__main__':

    print("Running Script!!!!!!!!!!!!!!")

    
    diff_handles = load_diffhandles_model()



    exp_path = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/large_scale_study_all/large_scale_study_dataset_metrics_2/"


    run_geodiff(exp_path, diff_handles)
    # exp_dict = read_exp(exp_path)
    # run_geodiff_folder(exp_dict, diff_handles)

    exit()

    