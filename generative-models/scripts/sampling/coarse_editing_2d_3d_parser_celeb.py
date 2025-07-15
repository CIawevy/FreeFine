import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
# sys.path.append('/data/Hszhu/Reggio')
import json
import os.path as osp
from tqdm import  tqdm
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import imageio
import time
import numpy as np
import torch
import argparse
from einops import rearrange, repeat
# from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor
from edit_prompt_set import generate_random_instructions,my_seed_everything,generate_instruction_celeb

Rotate_3D_list = ['person', 'hat']
def temp_view_img(image: Image.Image, title: str = None) -> None:
    # PIL -> ndarray OR ndarray->PIL->ndarray
    if not isinstance(image, Image.Image):  # ndarray
        # image_array = Image.fromarray(image).convert('RGB')
        image_array = image
    else:  # PIL
        if image.mode != 'RGB':
            image.convert('RGB')
        image_array = np.array(image)

    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()
def visualize_rgb_image(image: Image.Image, title: str = None) -> None:
    """
    Visualize an RGB image from a PIL Image format with an optional title.

    Parameters:
    image (PIL.Image.Image): The RGB image represented as a PIL Image.
    title (str, optional): The title to display above the image.

    Raises:
    ValueError: If the input is not a PIL Image or is not in RGB mode.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    if image.mode != 'RGB':
        raise ValueError("Input image must be in RGB mode.")

    image_array = np.array(image)

    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()
def temp_view( mask, title='Mask', name=None):
    """
    显示输入的mask图像

    参数:
    mask (torch.Tensor): 要显示的mask图像，类型应为torch.bool或torch.float32
    title (str): 图像标题
    """
    # 确保输入的mask是float类型以便于显示
    if isinstance(mask, np.ndarray):
        mask_new = mask
    else:
        mask_new = mask.float()
        mask_new = mask_new.detach().cpu()
        mask_new = mask_new.numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(mask_new, cmap='gray')
    plt.title(title)
    plt.axis('off')  # 去掉坐标轴
    # plt.savefig(name+'.png')
    plt.show()


def save_mask(mask, dst_dir, da_name, ins_name,sample_id):
    da_name = str(da_name)
    ins_name = str(ins_name)
    sample_id = str(sample_id)
    # 创建da子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 创建ins子文件夹
    ins_subfolder_path = os.path.join(subfolder_path, ins_name)
    os.makedirs(ins_subfolder_path, exist_ok=True)

    # 保存mask到ins子文件夹中
    mask_path = os.path.join(ins_subfolder_path, f"{sample_id}.png")
    cv2.imwrite(mask_path, mask.astype(np.uint8)*255)
    print(f"Saved mask to {mask_path}")

    return mask_path


def save_img(img, dst_dir, da_name, ins_name,sample_id):
    da_name = str(da_name)
    ins_name = str(ins_name)
    sample_id = str(sample_id)
    # 创建da子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 创建ins子文件夹
    ins_subfolder_path = os.path.join(subfolder_path, ins_name)
    os.makedirs(ins_subfolder_path, exist_ok=True)

    # 保存img到ins子文件夹中
    img_path = os.path.join(ins_subfolder_path, f"{sample_id}.png")
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved image to {img_path}")

    return img_path

def save_json(data_dict, file_path):
    """
    将字典保存为 JSON 文件

    Args:
        data_dict (dict): 需要保存的字典
        file_path (str): JSON 文件的保存路径
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)


def load_json(file_path):
    """
    加载指定路径的JSON文件并返回数据。

    :param file_path: JSON文件的路径
    :return: 从JSON文件中加载的数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        print(f"文件格式错误: {file_path}")
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
    return None
import random
def split_data(data, num_splits, subset_num=None,seed=None):
    if seed is not None:
        random.seed(seed)
    data_keys = list(data.keys())

    # 如果需要从数据中随机抽取100个
    if subset_num is not None:
        data_keys = random.sample(data_keys, subset_num)  # 随机抽取subset_num个键
    else:
        random.shuffle(data_keys)  # 随机打乱数据键

    chunk_size = len(data_keys) // num_splits
    data_parts = []

    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_splits - 1 else len(data_keys)
        data_part = {k: data[k] for k in data_keys[start_idx:end_idx]}
        data_parts.append(data_part)

    return data_parts

def frames_to_images(samples,base_count,output_folder):
    samples = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
    for idx,spl in enumerate(samples):
        spl = Image.fromarray(spl)
        imageio.imwrite(
            os.path.join(output_folder, f"{base_count:06d}_{idx}.jpg"), spl
        )
def frames_to_images_list(samples):
    samples = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
    return samples


def generate_elevations(elevations_deg, num_frames):
    if elevations_deg==0:
        return [elevations_deg]*num_frames
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        # Generate the first half of the sequence: 0 to max, max to 0
        up = np.linspace(0, elevations_deg, num_frames // 4 + 1)
        down = np.linspace(elevations_deg, 0, num_frames // 4 + 1)[1:]
        # Generate the second half of the sequence: 0 to -max, -max to 0
        negative_up = np.linspace(0, -elevations_deg, num_frames // 4 + 1)[1:]
        negative_down = np.linspace(-elevations_deg, 0, num_frames // 4 + 1)[1:]

        # Concatenate all parts to get the full cyclic sequence
        elevations_deg = np.concatenate([up, down, negative_up, negative_down])

        # If the number of frames is odd, adjust the sequence by trimming or repeating the last value
        if len(elevations_deg) < num_frames:
            elevations_deg = np.concatenate([elevations_deg, [elevations_deg[-1]]])
        elif len(elevations_deg) > num_frames:
            elevations_deg = elevations_deg[:num_frames]

    assert len(
        elevations_deg) == num_frames, f"Please provide 1 value, or a list of {num_frames} values for elevations_deg! Given {len(elevations_deg)}"
    return elevations_deg
# [0, 0, 10, 0, 0, 0, 15, 0, 0, 20, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 30]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def load_sv3d(
    num_steps: Optional[int] = None,
    version: str = "svd",
    device: str = "cuda",
    verbose: Optional[bool] = False,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    if version == "sv3d_u":
        num_frames = 21
        num_steps = default(num_steps, 50)
        model_config = "scripts/sampling/configs/sv3d_u.yaml"
    elif version == "sv3d_p":
        num_frames = 21
        num_steps = default(num_steps, 50)
        model_config = "scripts/sampling/configs/sv3d_p.yaml"

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    return  model,filter

def sv3d_sample(
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    elevations_deg: Optional[float | List[float]] = 10.0,  # For SV3D
    azimuths_deg: Optional[List[float]] = None,  # For SV3D
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
    obj_mask=None,
    input_img=None,
    sv3d_model=None,
    sv3d_filter=None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    if version == "sv3d_u":
        num_frames = 21
        num_steps = default(num_steps, 50)
        model_config = "scripts/sampling/configs/sv3d_u.yaml"
        cond_aug = 1e-5
    elif version == "sv3d_p":
        num_frames = 21
        num_steps = default(num_steps, 50)
        model_config = "scripts/sampling/configs/sv3d_p.yaml"
        cond_aug = 1e-5
        if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
            # elevations_deg = [elevations_deg] * num_frames
            elevations_deg = np.linspace(0, elevations_deg, num_frames + 1)[1:] % elevations_deg
        assert (
            len(elevations_deg) == num_frames
        ), f"Please provide 1 value, or a list of {num_frames} values for elevations_deg! Given {len(elevations_deg)}"
        # elevations_deg = generate_elevations(elevations_deg,num_frames)
        # elevations_deg = [  0, 6,12,18,24,30,30,30,30,30,30,30, 30, 30, 30,30,24,18,12,6,0]
        polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
        if isinstance(azimuths_deg, float) or isinstance(azimuths_deg, int):#List by default
            if azimuths_deg !=0:#todo:适应性修改+resize copy resize尺寸
                azimuths_deg = np.linspace(0, azimuths_deg, num_frames + 1)[1:] % azimuths_deg
            else:
                azimuths_deg = [azimuths_deg] * num_frames
        assert (
            len(azimuths_deg) == num_frames
        ), f"Please provide a list of {num_frames} values for azimuths_deg! Given {len(azimuths_deg)}"
        azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
        azimuths_rad[:-1].sort()
    else:
        raise ValueError(f"Version {version} does not exist.")
    if sv3d_model is not None and sv3d_filter is not None:
        model,filter = sv3d_model,sv3d_filter
    else:
        model, filter = load_model(
            model_config,
            device,
            num_frames,
            num_steps,
            verbose,
        )
    torch.manual_seed(seed)


    if "sv3d" in version:
        image_arr = np.where(obj_mask.astype(bool)[:,:,None], input_img, 255)
        image_arr = np.array(Image.fromarray(image_arr).convert("RGBA"))
        in_w, in_h = image_arr.shape[:2]
        x, y, w, h = cv2.boundingRect(obj_mask)
        max_size = max(w, h)
        side_len = (
            int(max_size / image_frame_ratio)
            if image_frame_ratio is not None
            else in_w
        )
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[
            center - h // 2 : center - h // 2 + h,
            center - w // 2 : center - w // 2 + w,
        ] = image_arr[y : y + h, x : x + w]
        # resize frame to 576x576
        rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
        # white bg
        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))

        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024) and "sv3d" not in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if (H, W) != (576, 576) and "sv3d" in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        if "sv3d_p" in version:
            value_dict["polars_rad"] = polars_rad
            value_dict["azimuths_rad"] = azimuths_rad

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

                randn = torch.randn(shape, device=device)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    2, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                model.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model.decode_first_stage(samples_z)
                if "sv3d" in version:
                    samples_x[-1:] = value_dict["cond_frames_without_noise"]
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                # os.makedirs(output_folder, exist_ok=True)
                # base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                #
                # imageio.imwrite(
                #     os.path.join(output_folder, f"{base_count:06d}.jpg"), input_image
                # )

                samples = embed_watermark(samples)
                samples = filter(samples)
                return (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8),in_w


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter



def resize_img(trans_img,size=None,):
    if isinstance(trans_img,np.ndarray):
        trans_img = cv2.cvtColor(trans_img,cv2.COLOR_RGB2BGR)
        trans_img = Image.fromarray(trans_img) #PIL img
    assert size is not None,"for resize pipe,size should be given"
    trans_img.thumbnail(size, Image.Resampling.LANCZOS)
    return_img = np.array(trans_img.copy())
    return_img = cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB)
    return return_img


def read_img(image_path):
    img = cv2.imread(image_path)  # bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def coarse_edit_func_v2_celeb(img,mask_cur,inp_cur,constrain_area,obj_label,sv3d_model=None,sv3d_filter=None):
    # 3D EDITING SAMPLING FUNC v2 for cxeleb
    # SAMPLE all the edit all the possibility
    #RESIZE IMG

    np.random.seed(int(time.time()))
    random_seed = np.random.randint(0, 2 ** 32 - 1)

    instructions_2D = generate_instruction_celeb(sample_type='2D',seed=random_seed,label = obj_label)

    edit_prompt_list,coarse_res_list,tgt_mask_list,edit_param_list= [], [], [],[]
    edit_prompt_list_3d=[]
    angle_list_3d = [] #for easier fetch
    edit_param_list_3d=[]
    for instruction in instructions_2D:
        try:
            coarse_edit_res,target_mask,edit_prompt,edit_param = sample_edit_func_2d_celeb(img,mask_cur,inp_cur,constrain_area,obj_label,instruction)
            edit_prompt_list.append(edit_prompt)
            coarse_res_list.append(coarse_edit_res)
            tgt_mask_list.append(target_mask)
            edit_param_list.append(edit_param)
        except AssertionError as e:
            print(f"AssertionError caught: {e}")
            continue
    if obj_label in Rotate_3D_list:
        instructions_3D = generate_instruction_celeb(sample_type='3D', seed=random_seed, label=obj_label)
        for instruction in instructions_3D:
            edit_prompt,sample_degree,edit_param = generate_editing_config_3d(obj_label, instruction)
            edit_prompt_list_3d.append(edit_prompt)
            angle_list_3d.append(sample_degree)
            edit_param_list_3d.append(edit_param)

        if len(angle_list_3d)>0:
            edit_config_3D = {
                'elevations_deg': 10,  # default
                'azimuths_deg': generate_azimuth_angles(n_views_sv3d=21, angle_list_3d=angle_list_3d)
            }
            coarse_edit_list_3d,tgt_mask_list_3d,valid_prompt_list_3d,valid_param_list_3d= transform_3d(img,mask_cur,inp_cur,angle_list_3d,constrain_area,edit_prompt_list_3d,edit_param_list_3d,
                         elevations_deg=edit_config_3D['elevations_deg'],
                         azimuths_deg=edit_config_3D['azimuths_deg'],
                        sv3d_model=sv3d_model,sv3d_filter=sv3d_filter)
            coarse_res_list.extend(coarse_edit_list_3d)
            tgt_mask_list.extend(tgt_mask_list_3d)
            edit_prompt_list.extend(valid_prompt_list_3d)
            edit_param_list.extend(valid_param_list_3d)



    return edit_prompt_list,edit_param_list,coarse_res_list,tgt_mask_list
def get_mask_from_rembg(trans_img,size=None,need_mask=True):
    if isinstance(trans_img,np.ndarray):
        trans_img = cv2.cvtColor(trans_img,cv2.COLOR_RGB2BGR)
        trans_img = Image.fromarray(trans_img) #PIL img
    if need_mask:
        if size is not None:
            trans_img.thumbnail(size, Image.Resampling.LANCZOS)
            print(f'resized_img shape:{trans_img.size}')

        return_img = np.array(trans_img.copy())
        return_img = cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB)
        trans_img = remove(trans_img.convert("RGBA"), alpha_matting=True)
        image_arr = np.array(trans_img)
        in_w, in_h = image_arr.shape[:2]
        _, trans_mask = cv2.threshold(
            np.array(trans_img.split()[-1]), 128, 255, cv2.THRESH_BINARY
        )
        return trans_mask,return_img
def transform_2d(ori_img,ori_mask,inp_cur,edit_config,constrain_area,ignore_constrain):
    rotation_angle=edit_config['rotation_angle']
    resize_scale = edit_config['resize_scale']
    if isinstance(resize_scale, float) or isinstance(resize_scale, int):
        resize_scale = (resize_scale,resize_scale)
    dx,dy = edit_config['dx'],edit_config['dy']
    flip_horizontal=edit_config['flip_horizontal']
    flip_vertical = edit_config['flip_vertical']
    # Prepare foreground
    height, width = ori_img.shape[:2]
    y_indices, x_indices = np.where(ori_mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        top, bottom = np.min(y_indices), np.max(y_indices)
        left, right = np.min(x_indices), np.max(x_indices)
        # mask_roi = mask[top:bottom + 1, left:right + 1]
        # image_roi = image[top:bottom + 1, left:right + 1]
        mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2
    else:
        assert False, 'mask no center error, discard'
    #将resize_scale解耦出来，实现x，y的单独缩放
    rotation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle, 1)
    #当rotation angle=0且resize scale!=1时，由mask 中心会影响dx,dy的初始值
    #计算公式默认rotation angle=0
    tx,ty = (1-resize_scale[0])*mask_center_x,(1-resize_scale[1])*mask_center_y
    dx+=tx
    dy+=ty
    rotation_matrix[0, 2] += dx
    rotation_matrix[1, 2] += dy
    rotation_matrix[0, 0]*=resize_scale[0]
    rotation_matrix[1, 1]*=resize_scale[1]

    transformed_image = cv2.warpAffine(ori_img, rotation_matrix, (width, height))
    # transformed_mask_exp = cv2.warpAffine(exp_mask.astype(np.uint8), rotation_matrix, (width, height),
    #                                       flags=cv2.INTER_NEAREST).astype(bool)
    transformed_mask = cv2.warpAffine(ori_mask.astype(np.uint8), rotation_matrix, (width, height),
                                      flags=cv2.INTER_NEAREST).astype(bool)

    # 检查是否需要水平翻转
    if flip_horizontal:
        transformed_image = cv2.flip(transformed_image, 1)
        transformed_mask = cv2.flip(transformed_mask.astype(np.uint8), 1).astype(bool)
        # transformed_mask_exp = cv2.flip(transformed_mask_exp.astype(np.uint8), 1).astype(bool)

    # 检查是否需要垂直翻转
    if flip_vertical:
        transformed_image = cv2.flip(transformed_image, 0)
        transformed_mask = cv2.flip(transformed_mask.astype(np.uint8), 0).astype(bool)
        # transformed_mask_exp = cv2.flip(transformed_mask_exp.astype(np.uint8), 0).astype(bool)

    # ddpm_region = transformed_mask_exp * (1 - transformed_mask)
    final_image = np.where(transformed_mask[:, :, None], transformed_image,
                           inp_cur)  # move with expansion pixels but inpaint
    if ignore_constrain:
        constrain_area = np.zeros_like(constrain_area)
    assert (transformed_mask & constrain_area.astype(bool)).sum()==0,'overlap with other objects, discard'
    return final_image,transformed_mask
def pasted_sv3d_back_to_img(ori_img, ori_mask, inp_cur, trans_img,constrain_area,ignore_constrain):
    # trans_img = cv2.resize(trans_img, (ori_img.shape[1], ori_img.shape[0]))
    trans_mask, trans_img = get_mask_from_rembg(trans_img)

    x, y, w, h = cv2.boundingRect(ori_mask)
    cent_h, cent_w = y + h // 2, x + w // 2

    x_t, y_t, w_t, h_t = cv2.boundingRect(trans_mask)

    # 计算粘贴区域的起始和结束位置
    start_h = max(cent_h - h_t // 2, 0)
    start_w = max(cent_w - w_t // 2, 0)
    end_h = min(cent_h - h_t // 2 + h_t, ori_mask.shape[0])
    end_w = min(cent_w - w_t // 2 + w_t, ori_mask.shape[1])

    # 调整 trans_mask 的区域
    src_end_h = y_t + (end_h - start_h)
    src_end_w = x_t + (end_w - start_w)

    # 创建与 ori_mask 和 ori_img 大小一致的空白图像和掩码
    repl_trans_mask = np.zeros_like(ori_mask)
    repl_trans_img = np.zeros_like(ori_img)

    # 粘贴 trans_mask 和 trans_img 到相应位置
    repl_trans_mask[start_h:end_h, start_w:end_w] = trans_mask[y_t:src_end_h, x_t:src_end_w]
    repl_trans_img[start_h:end_h, start_w:end_w] = trans_img[y_t:src_end_h, x_t:src_end_w]

    repl_trans_mask = (repl_trans_mask > 0).astype(bool)
    if ignore_constrain:
        constrain_area = np.zeros_like(constrain_area)
    assert (repl_trans_mask & constrain_area.astype(bool)).sum() == 0,'3D trans overlap problem, discard'
    # ddpm_region = exp_mask_cur * (1 - repl_trans_mask)
    final_image = np.where(repl_trans_mask[:, :, None], repl_trans_img, inp_cur)

    return final_image, repl_trans_mask
def transform_3d(ori_img,ori_mask,inp_cur,angle_list_3d,constrain_area,edit_prompt_list_3d,edit_param_list_3d,elevations_deg=None,azimuths_deg=None,sv3d_model=None,sv3d_filter=None,ignore_constrain=False):
    coarse_edit_list_3d, tgt_mask_list_3d,ddpm_region_list_3d = [],[],[]
    img_list,in_w = sv3d_sample(version='sv3d_p', decoding_t=5, elevations_deg=elevations_deg,
                azimuths_deg=azimuths_deg,input_img=ori_img,obj_mask=ori_mask,sv3d_model=sv3d_model,sv3d_filter=sv3d_filter)
    valid_edit_prompt_list = []
    valid_edit_param_list = []
    for edit_index,angle_select in enumerate(angle_list_3d):
        cur_prompt = edit_prompt_list_3d[edit_index]
        cur_param = edit_param_list_3d[edit_index]
        if angle_select<0:
            angle_select = 360+angle_select
        idx = np.array(azimuths_deg)==angle_select
        #idx ->
        trans_img = img_list[idx][0]
        trans_img = cv2.resize(trans_img, (in_w, in_w))
        try:
            coarse_edit_res,transfered_mask = pasted_sv3d_back_to_img(ori_img, ori_mask,inp_cur,trans_img,constrain_area,ignore_constrain)
            coarse_edit_list_3d.append(coarse_edit_res)
            tgt_mask_list_3d.append(transfered_mask)
            valid_edit_prompt_list.append(cur_prompt)
            valid_edit_param_list.append(cur_param)
        except AssertionError as e:
            print(f"AssertionError caught: {e}")
            continue

    return coarse_edit_list_3d,tgt_mask_list_3d,valid_edit_prompt_list,valid_edit_param_list





def sample_move_func(exp_mask, constrain_area, direction,level):
    """
    compare to full image and item itself
    level1:
    level2:
    level3:
    """
    #dx dy sample from direction and level
    #clip the boundary or fail case
    dx,dy = 0,0
    H, W = constrain_area.shape  # 图像的高度和宽度
    x, y, w, h = cv2.boundingRect(exp_mask)
    if level == 'level_1':
        range_x =  (int(0.05*W),int(0.1*W))
        range_y =  (int(0.05*H),int(0.1*H))
    elif level == 'level_2':
        range_x = (int(0.1 * W), int(0.2 * W))
        range_y = (int(0.1 * H), int(0.2 * H))
    elif level == 'level_3':
        range_x = (int(0.2 * W), int(0.4 * W))
        range_y = (int(0.2 * H), int(0.4 * H))
        # Sample movement based on direction
    if 'left' in direction:
        dx = -np.random.randint(range_x[0], range_x[1])
        assert x-range_x[0]>=0,'move left lower bound error, discard'
    elif 'right' in direction:
        dx = np.random.randint(range_x[0], range_x[1])
        assert x + w + range_x[0] <= W, 'move right lower bound error, discard'
    if 'up' in direction or 'upper' in direction:
        dy = -np.random.randint(range_y[0], range_y[1])
        assert y - range_y[0] >= 0, 'move up lower bound error, discard'
    elif 'down' in direction or 'lower' in direction:
        dy = np.random.randint(range_y[0], range_y[1])
        assert y + h + range_y[0] <= H, 'move down lower bound error, discard'

    return dx, dy

def calculate_max_scale_1d(pos_center, length_half, limit, max_scale, scale_ratio=0.5):
    # 计算任意比例下的缩放边界
    # limit 为图像的边界，scale_ratio 为所考虑的缩放后超出部分的比例（如 0.5 表示一半）

    scale_positive = (limit - pos_center) / (scale_ratio * length_half) if limit > pos_center else np.inf  # 正方向
    scale_negative = pos_center / (scale_ratio * length_half) if pos_center > 0 else np.inf  # 负方向

    max_scale_corrected = min(max_scale, scale_positive, scale_negative)

    return max_scale_corrected
def sample_scale_func(exp_mask, constrain_area, edit_class, direction,level):
    H, W = constrain_area.shape  # 整个图像的高度和宽度
    x, y, w, h = cv2.boundingRect(exp_mask)  # 获取exp_mask的外接矩形框
    if level == 'level_1':
        range_enlarge = (1.1, 1.3)
        range_shrink = (0.9, 0.8)
    elif level == 'level_2':
        range_enlarge = (1.3, 1.5)
        range_shrink = (0.6, 0.8)
    elif level == 'level_3':
        range_enlarge = (1.5, 3.0)
        range_shrink = (0.4, 0.6)
        # Sample movement based on direction
    # 计算中心点和半长半宽
    cx, cy = x + w / 2, y + h / 2  # 中心点
    half_w, half_h = w / 2, h / 2  # 半宽和半高

    if edit_class =='shrink':
        if direction == 'uniformly':
            scale_x = scale_y = np.random.uniform(range_shrink[0],range_shrink[1])
        else:
            scale_x = np.random.uniform(range_shrink[0],range_shrink[1]) if direction == 'horizontally' else 1.0
            scale_y = np.random.uniform(range_shrink[0],range_shrink[1]) if direction == 'vertically' else 1.0
    elif edit_class == 'enlarge':
        if direction == 'uniformly':
            boundary_scale_x = max(W-cx,cx) / half_w
            boundary_scale_y = max(H-cy,y) / half_h
            scale_x = scale_y = min(boundary_scale_x,boundary_scale_y,np.random.uniform(range_enlarge[0], range_enlarge[1]))
            assert scale_x > range_enlarge[0],'resize lower bound error, discard'

        elif direction == 'horizontally':
            boundary_scale_x = max(W - cx, cx) / half_w
            scale_x = min(boundary_scale_x,np.random.uniform(range_enlarge[0], range_enlarge[1])) if direction != 'vertically' else 1.0
            assert scale_x > range_enlarge[0], 'resize lower bound error, discard'
            scale_y = 1.0
        elif  direction == 'vertically':
            boundary_scale_y = max(H - cy, y) / half_h
            scale_y = min(boundary_scale_y ,np.random.uniform(range_enlarge[0], range_enlarge[1])) if direction != 'horizontally' else 1.0
            assert scale_y > range_enlarge[0], 'resize lower bound error, discard'
            scale_x = 1.0


    resize_scale = (scale_x, scale_y)
    return resize_scale


def calculate_rotation_boundaries(cx, cy, r, constrain_area, direction, scale_ratio, max_angle):
    H, W = constrain_area.shape
    free_area = 1 - constrain_area  # 反转constrain_area以获取可行域

    # 生成所有可能的旋转角度和对应的坐标
    angles = np.arange(1, max_angle + 1)
    if 'counterclockwise' in direction:
        angles = -angles


    rad_angles = np.radians(angles)

    # 基于中心点和半径计算四个顶点的新坐标
    corner_offsets = np.array([[np.cos(-np.pi/4), np.sin(-np.pi/4)],
                               [np.cos(np.pi/4), np.sin(np.pi/4)],
                               [np.cos(3*np.pi/4), np.sin(3*np.pi/4)],
                               [np.cos(-3*np.pi/4), np.sin(-3*np.pi/4)]]) * r
    new_xs = cx + np.outer(np.cos(rad_angles), corner_offsets[:, 0])
    new_ys = cy + np.outer(np.sin(rad_angles), corner_offsets[:, 1])

    # 检查顶点是否在图像范围内

    in_image_bounds = (new_xs >= 0) & (new_xs < W) & (new_ys >= 0) & (new_ys < H)
    conditions_in_image = np.zeros_like(new_xs, dtype=bool)
    conditions_in_image[in_image_bounds] = free_area[new_ys[in_image_bounds].astype(int), new_xs[in_image_bounds].astype(
        int)] == 1

    # 对于超出图像范围的点，检查是否满足 scale_ratio 的条件
    conditions_outside_image = np.zeros_like(new_xs, dtype=bool)
    conditions_outside_image[~in_image_bounds] = (
                                       abs(new_xs[~in_image_bounds] - cx) <= scale_ratio * r) & (abs(new_ys[~in_image_bounds] - cy) <= scale_ratio * r
                                                                              )

    # 合并所有条件
    conditions = np.all(conditions_in_image | conditions_outside_image,axis=1)


    if np.all(~conditions):
        return 0
    elif np.all(conditions):
        return max_angle
    else:
        # 找到第一个不满足条件的角度
        first_invalid_angle_index = np.argmax(~conditions)
        return angles[first_invalid_angle_index-1]

def sample_rotate_func_2d(exp_mask, constrain_area, direction, level):

    x, y, w, h = cv2.boundingRect(exp_mask)  # 获取exp_mask的外接矩形框
    cx, cy = x + w / 2, y + h / 2  # 中心点
    r = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)  # 计算对角线的半径
    if level == 'level_1':
        range_rotate= (5, 10)
    elif level == 'level_2':
        range_rotate = (10, 20)
    elif level == 'level_3':
        range_rotate = (20, 40)
    # 计算顺时针和逆时针的最大旋转角度
    if 'counterclockwise' in direction:
        max_rotation_angle = calculate_rotation_boundaries(cx, cy, r, constrain_area, 'counterclockwise',
                                                           0.1, 40)
    else:
        max_rotation_angle = calculate_rotation_boundaries(cx, cy, r, constrain_area, 'clockwise', 0.1,
                                                           40)

    # 返回最终的最大旋转角度
    rotation_angle = min(max_rotation_angle,np.random.uniform(range_rotate[0],range_rotate[1]))
    assert rotation_angle > range_rotate[0],'rotate lower bound error, discard'
    final_angle =  int(np.round(rotation_angle,2))
    if 'counterclockwise' in direction:
        final_angle = -final_angle
    return final_angle
def has_significant_difference(new_degree, existing_degrees, threshold=0.1):
    """
    检查当前采样的 degree 是否与已存在的 degree 有显著差异。
    如果新 degree 与已有的任一 degree 差异小于阈值，则返回 False；否则返回 True。
    """
    for existing_degree in existing_degrees:
        if abs(new_degree - existing_degree) < threshold:
            return False
    return True








def gen_2D_edit_config_v2( exp_mask, constrain_area, edit_class, direction,level,ignore_constrain):
    dx, dy, rotation_angle, resize_scale, flip_horizontal, flip_vertical = 0, 0, 0, (1,1), False, False  # default
    np.random.seed(int(time.time()))
    random_seed = np.random.randint(0, 2 ** 32 - 1)
    my_seed_everything(random_seed)
    if ignore_constrain:
        constrain_area = np.zeros_like(constrain_area)
    if edit_class == 'move':
        dx, dy = sample_move_func(exp_mask, constrain_area, direction,level)


    elif edit_class == 'enlarge' or edit_class == 'shrink':
        resize_scale = sample_scale_func(exp_mask, constrain_area, edit_class, direction,level)


    elif edit_class == 'flip':
        flip_horizontal = direction == 'horizontally'
        flip_vertical = not flip_horizontal


    elif edit_class == 'rotate':
        rotation_angle = sample_rotate_func_2d(exp_mask, constrain_area, direction,level)


    else:
        raise ValueError(
            f"Invalid edit class '{edit_class}'. Expected 'move', 'enlarge', 'shrink', 'flip', or 'rotate'.")
    edit_config = {
        'dx': dx,
        'dy': dy,
        'rotation_angle': rotation_angle,
        'resize_scale': resize_scale,
        'flip_horizontal': flip_horizontal,
        'flip_vertical': flip_vertical
    }
    #edit_param=[dx,dy,dz,rx,ry,rz,sx,sy,sz]
    #for 2D edit dz=rx=ry=0,sz=1
    edit_param = [dx,dy,0,0,0,rotation_angle,resize_scale[0],resize_scale[1],1]
    return edit_config,edit_param


def generate_azimuth_angles(n_views_sv3d=21,angle_list_3d=None):
    # 指定顺时针方向的角度
    half_len = len(angle_list_3d) // 2
    forward_angles = np.array(angle_list_3d[:half_len])
    # 指定逆时针方向的角度
    backward_angles = np.array([360 + angle for angle in angle_list_3d[half_len:]])

    # 确保特定角度数量不超过总帧数
    assert len(forward_angles) + len(backward_angles) < n_views_sv3d, "指定的角度数量不能超过总帧数"

    # 剩余帧数
    remaining_frames = n_views_sv3d - len(forward_angles) - len(backward_angles) - 1  # 减去最后一个0

    # 在剩余范围内均匀分布剩余角度
    if remaining_frames > 0:
        remaining_azimuths = np.linspace(0, 360, remaining_frames + 1)[:-1] % 360
    else:
        remaining_azimuths = np.array([])

    # 合并所有角度，并确保最后一个角度为0
    azimuths_deg = np.concatenate((forward_angles, backward_angles, remaining_azimuths))
    azimuths_deg = np.sort(azimuths_deg) % 360
    azimuths_deg = np.concatenate((azimuths_deg, [0.0]))

    return list(azimuths_deg)


def generate_editing_config_2d(exp_mask,constrain_area,obj_label,instructions,ignore_constrain):

    # Step 1: 根据指令确定操作类型
    edit_class = instructions['type']
    prompt = instructions['prompt']
    direction = instructions['direction']
    level = instructions['degree']

    edit_config,edit_param= gen_2D_edit_config_v2(exp_mask,constrain_area,edit_class,direction,level,ignore_constrain)
    #'with regard to its center'
    edit_prompt = prompt.replace("{object}", obj_label)

    return  edit_prompt,edit_config,edit_param

def generate_editing_config_3d(obj_label, instructions):
    # Step 1: 根据指令确定操作类型
    edit_class = instructions['type']
    prompt = instructions['prompt']
    direction = instructions['direction']
    level = instructions['degree']
    assert  edit_class == 'rotate','Not implent other 3d operation yet'
    if level == 'level_1':
        range_rotate= (5, 10)
    elif level == 'level_2':
        range_rotate = (15, 20)
    elif level == 'level_3':
        range_rotate = (25, 40)
    sampled_degree = int(np.round(np.random.uniform(range_rotate[0],range_rotate[1]),2))
    if 'counterclockwise' in direction:
        sampled_degree = -sampled_degree
    #edit_param = [dx,dy,dz,rx,ry,rz,sx,sy,sz] default clock-wise
    edit_param = [0,0,0,0,0,sampled_degree,1,1,1]
    edit_prompt = prompt.replace("{object}", obj_label)
    return edit_prompt,sampled_degree,edit_param

def judge_2d_3d(instuction):
    edit_class = instuction['type']
    prompt = instuction['prompt']
    if edit_class == 'move':
        edit_func_type = "2D"
        #dx,dy=sample_move()
    elif edit_class == 'enlarge':
        edit_func_type = "2D"
    elif edit_class == 'shrink':
        edit_func_type = "2D"
    elif edit_class == 'flip':
        edit_func_type = "2D"
    elif edit_class == 'rotate':
        if 'z-axis' in prompt:
            edit_func_type = "2D"
        elif 'y-axis' in prompt:
            edit_func_type = "3D"
    return edit_func_type


def sample_edit_func_2d_celeb(ori_img,ori_mask,inp_cur,constrain_area,obj_label,instructions):
    #2D注意单独缩放x，y的矩阵手动实现。
    omit_constrain_list = ['person', 'hat']
    ignore_constrain = obj_label in omit_constrain_list
    edit_prompt,edit_config,edit_param = generate_editing_config_2d(ori_mask,constrain_area,obj_label,instructions,ignore_constrain)

    coarse_edit_res,target_mask=transform_2d(ori_img,ori_mask,inp_cur,edit_config,constrain_area,ignore_constrain)
    return coarse_edit_res,target_mask,edit_prompt,edit_param
def get_constrain_areas(mask_list_path):
    mask_list = [cv2.imread(pa) for pa in mask_list_path]
    if len(mask_list)>0:
        constrain_areas = np.zeros_like(mask_list[0])
    for mask in mask_list:
        mask[mask>0] = 1
        constrain_areas +=mask
    constrain_areas[constrain_areas>0] =1
    return constrain_areas[:,:,0]
def get_constrain_areas_celeb(mask_list_path,obj_label_list):
    mask_list = [cv2.imread(pa) for pa in mask_list_path]
    if len(mask_list)>0:
        constrain_areas = np.zeros_like(mask_list[0])
    for idx,mask in enumerate(mask_list):
        if obj_label_list[idx] == 'person':
            continue
        mask[mask>0] = 1
        constrain_areas +=mask
    constrain_areas[constrain_areas>0] =1
    return constrain_areas[:,:,0]





def main(data_id, base_dir):
    dst_base = osp.join(base_dir, f'Subset_{data_id}')
    if osp.exists(osp.join(dst_base,f"coarse_input_full_pack_{data_id}.json")):
        print(f'coarse edit for {data_id} already finish!')
        return
    if osp.exists(osp.join(dst_base,f"temp_file_coarse.json")): #resume
        new_data = load_json(osp.join(dst_base,f"temp_file_coarse.json"))
    else:
        new_data = dict()
    dataset_json_file = osp.join(dst_base,f"mat_fooocus_inpainting_{data_id}.json")
    dst_coarse_inp_path = osp.join(dst_base,"coarse_input")
    dst_target_msk_path = osp.join(dst_base,"target_mask")
    # dst_ddpm_reg_path = osp.join(dst_base,"ddpm_region")
    data = load_json(dataset_json_file)
    model,filter=load_sv3d(version='sv3d_p')
    # data_parts = split_data(data, 3,subset_num=210,seed=42)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #load data：
        #0：
            #'image_path'
            #'instances':
                #'mask_path','exp_math_path','inp_img_path','obj_label_path'
    #save data:
        # 0：img_id
        # 'instances':attr
            #0 ins_id
                 #0 sample_id
                    # edit_prompt','ori_mask_path','tgt_mask_path','ori_img_path','coarse_input_path',;obj_label'
                 #1
                    # edit_prompt','ori_mask_path','tgt_mask_path','ori_img_path','coarse_input_path',;obj_label'
            #1:...
        #...

    for da_n, da in tqdm(data.items(), desc=f'Proceeding coarse editing (Part {data_id})'):

        if 'instances' not in da.keys():
            print(f'skip {da_n} for not valid instance')
            continue
        image_path = da['src_img_path']
        caption = da['caption']
        instances = da['instances']
        if da_n in new_data.keys() and '0' in new_data[da_n]['instances'].keys():
            print(f'\n skip {da_n} for already exist')
            continue

        new_data[da_n] = dict()
        new_data[da_n]['src_img_path'] = image_path
        new_data[da_n]['caption'] = caption
        new_data[da_n]['instances'] = dict()


        mask_list = instances['mask_path'] #modified with completion masks
        # exp_mask_list = instances['exp_mask_path']
        inp_img_list = instances['inp_img_path']
        obj_label_list = instances['obj_label']
        #other configs
        img = read_img(image_path)
        img = resize_img(img,size=[512,512])
        w, h = img.shape[:2]
        #resize img
        #rm bg
        constrain_areas = get_constrain_areas_celeb(mask_list,obj_label_list)
        # temp_view(constrain_areas)
        # constrain_areas_strict = get_constrain_areas(exp_mask_list)
        for ins_id in range(len(inp_img_list)):#ins_id
            sample_dict = dict()
            mask_path,inp_path,obj_label = mask_list[ins_id],inp_img_list[ins_id],obj_label_list[ins_id]
            mask_cur =  cv2.resize( cv2.imread(mask_path)[:,:,0], (h, w))
            constrain_areas = cv2.resize(constrain_areas, (h, w))
            inp_cur = cv2.resize(read_img(inp_path), (h, w))
            # exp_mask_cur =  cv2.resize(cv2.imread(exp_mask_path)[:,:,0], (h, w))
            cons_area = np.where(mask_cur.astype(bool),0,constrain_areas)
            edit_prompt_list,edit_param_list,coarse_res_list,tgt_mask_list= coarse_edit_func_v2_celeb(img,mask_cur,inp_cur,cons_area,obj_label,sv3d_model=model,sv3d_filter=filter)

            for sample_id,edit_prompt in enumerate(edit_prompt_list):#sample_id
                coarse_input = coarse_res_list[sample_id]
                target_mask = tgt_mask_list[sample_id]
                edit_param = edit_param_list[sample_id]
                # ddpm_mask = ddpm_region_list[sample_id]
                #save coarse img and mask
                tgt_mask_path = save_mask(target_mask,dst_target_msk_path,da_n,ins_id,sample_id)#save_masks(target_mask)
                coarse_img_path = save_img(coarse_input,dst_coarse_inp_path,da_n,ins_id,sample_id)  # save_img(coarse_input)
                # ddpm_region_path = save_mask(ddpm_mask ,dst_ddpm_reg_path,da_n,ins_id,sample_id)
                per_edit_data = dict()
                per_edit_data['edit_prompt'] = edit_prompt
                per_edit_data['src_img_path'] = image_path
                # per_edit_data['tag_caption'] = caption
                per_edit_data['obj_label'] = obj_label
                per_edit_data['ori_mask_path'] = mask_path
                per_edit_data['tgt_mask_path'] = tgt_mask_path
                per_edit_data['coarse_input_path'] = coarse_img_path
                per_edit_data['edit_param'] = edit_param
                # per_edit_data['ddpm_region_path'] = ddpm_region_path
                sample_dict[sample_id] = per_edit_data
                if len(sample_dict)==0:
                    continue
            new_data[da_n]['instances'][ins_id] = sample_dict
            #save temp file for resume
            save_json(new_data, osp.join(dst_base, f"temp_file_coarse.json"))
    save_json(new_data,osp.join(dst_base,f"coarse_input_full_pack_{data_id}.json"))
    #remove temp file
    os.remove(osp.join(dst_base, f"temp_file_coarse.json"))


if __name__ == "__main__":

    #this code is designed specially for celebA-HQ dataset
    #for partial moving can be constrained very specific
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="GroundingSAM processing script")
    parser.add_argument('--data_id', type=int, required=True, help="Data ID to process")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory path for dataset")
    # parser.add_argument('--gpu_id', type=int, default=0, help="Specify the GPU to use. Default is GPU 0")

    args = parser.parse_args()




    # 在需要时使用 device
    # model.to(device)
    # tensor = tensor.to(device)

    # 调用主逻辑并传入设备
    main(args.data_id, args.base_dir)