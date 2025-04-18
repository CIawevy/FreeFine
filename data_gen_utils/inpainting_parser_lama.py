import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import os.path as osp
import json
from tqdm import tqdm
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from lama import lama_with_refine
sys.path.append('/data/Hszhu/Reggio')
# from simple_lama_inpainting import SimpleLama
# from lama import lama_with_refine

import torch
import cv2



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

from pytorch_lightning import seed_everything
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


def temp_view(mask, title='Mask', name=None):
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


def replace_mask(mask, src_mask_path):
    # 保存mask到ins子文件夹中
    cv2.imwrite(src_mask_path, mask.astype(np.uint8) * 255)
    print(f"Saved mask to {src_mask_path}")
    return src_mask_path


def save_mask(mask, dst_dir, da_name, ins_name, sample_id):
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
    cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
    print(f"Saved mask to {mask_path}")

    return mask_path


def save_img(img, dst_dir, da_name, ins_name, sample_id):
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


def save_masks(masks, dst_dir, da_name):
    # 创建子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 用于存储保存的mask路径
    mask_paths = []

    # 保存每个mask到子文件夹中
    for idx, mask in enumerate(masks):
        mask_path = os.path.join(subfolder_path, f"mask_{idx + 1}.png")
        cv2.imwrite(mask_path, mask)  # 将mask保存为png图片 (注意：mask是二值图，乘以255以得到可见的结果)
        print(f"Saved mask {idx + 1} to {mask_path}")
        mask_paths.append(mask_path)

    return mask_paths


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


def split_data(data, num_splits, subset_num=None, seed=None):
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


def get_constrain_areas(mask_list_path):
    mask_list = [cv2.imread(pa) for pa in mask_list_path]
    if len(mask_list) > 0:
        constrain_areas = np.zeros_like(mask_list[0])
    for mask in mask_list:
        mask[mask > 0] = 1
        constrain_areas += mask
    constrain_areas[constrain_areas > 0] = 255
    return constrain_areas[:, :]


def prepare_mask_pool(instances):
    mask_pool = []
    for i,ins in instances.items():
        if len(ins) == 0:
            continue

        # 获取字典的第一个键
        first_key = next(iter(ins))

        # 将第一个键对应的 'ori_mask_path' 添加到 mask_pool
        mask_pool.append(ins[first_key]['ori_mask_path'])

    return mask_pool
def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

# data_parts = split_data(data, 2 , seed=42)


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

inpainter = lama_with_refine(device)
# seed = 42


from copy import deepcopy



base_dir = "/data/Hszhu/dataset/Geo-Bench-SC"
json_path = osp.join(base_dir, "annotations.json")
# 调用主逻辑并传入设备
data = load_json(json_path)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# new_data = dict()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
for da_n, da in tqdm(data.items(), desc='proceeding reggio inpainting'):
    instances = da['instances']
    for ins_n, inst in instances.items():
        ins_already_paint_stat = False
        for edit_n, coarse_input_pack in inst.items():
            if ins_already_paint_stat:
                print(f'skip {edit_n} for already inpainted ')
                continue
            mask_pool = prepare_mask_pool(instances)
            constrain_areas_strict = get_constrain_areas(mask_pool)
            constrain_areas_strict = cv2.resize(constrain_areas_strict, dsize=(512, 512),
                                                interpolation=cv2.INTER_NEAREST)

            # edit_prompt = coarse_input_pack['edit_prompt']
            # edit_param = coarse_input_pack['edit_param']

            ori_img = cv2.imread(coarse_input_pack['ori_img_path'])  # bgr
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            # ori_caption  = coarse_input_pack['tag_caption']
            ori_mask = cv2.imread(coarse_input_pack['ori_mask_path'])
            # obj_label = coarse_input_pack['obj_label']
            target_mask = cv2.imread(coarse_input_pack['tgt_mask_path'])
            # ddpm_region_mask =  cv2.imread(coarse_input_pack['ddpm_region_path'])
            # coarse_input = cv2.imread(coarse_input_pack['coarse_input_path'])  # bgr
            # coarse_input = cv2.cvtColor( coarse_input , cv2.COLOR_BGR2RGB)
            # temp_view_img(coarse_input)
            ori_mask = cv2.resize(ori_mask, dsize=target_mask.shape[:2], interpolation=cv2.INTER_NEAREST)
            dilation_factor = 30
            forbit_area = constrain_areas_strict - ori_mask
            dil_ori_mask = dilate_mask(ori_mask, dilation_factor)
            dil_ori_mask = np.where(forbit_area, 0, dil_ori_mask)
            # #re editing 2D for vis
            seed_r = 3787517166  # 3787517166 #4255183641 #4255183641 #2391550765
            seed_r = random.randint(0, 10 ** 16)

            seed_everything(seed_r)

            # to_inpaint_img = Image.fromarray(ori_img)
            # repair_mask = Image.fromarray(dil_ori_mask)
            # print(f'img:{to_inpaint_img.size}')
            # print(f'msk:{repair_mask.size}')
            inpainted_image = inpainter(ori_img, dil_ori_mask[:,:,0])#lama
            generated_results = inpainted_image
            # temp_view_img(generated_results)
            blended = False
            # output_image = cv2.resize(output_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            if blended:
                # TODO, assert ndarray

                # blur, you can adjust the parameters for better performance
                mask_blurred = cv2.GaussianBlur(ori_mask, (21, 21), 0) / 255
                mask_np = 1 - (1 - ori_mask) * (1 - mask_blurred)
                image_pasted = ori_img * (1 - mask_np) + generated_results * mask_np
                generated_results = image_pasted.astype(generated_results.dtype)
                # temp_view_img(image_pasted)

            inp_img_dir = os.path.join(base_dir, 'inp_img_lama')
            os.makedirs(inp_img_dir, exist_ok=True)
            subfolder_path = os.path.join(inp_img_dir, str(da_n))
            os.makedirs(subfolder_path, exist_ok=True)

            ins_subfolder_path = os.path.join(subfolder_path, str(ins_n))
            os.makedirs(ins_subfolder_path, exist_ok=True)
            final_path = os.path.join(ins_subfolder_path, f"inp_img.png")
            save_img = Image.fromarray(generated_results)
            save_img.save(final_path)  # 保存为PNG格式（单通道
            ins_already_paint_stat = True



