import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import os.path as osp
import json
from tqdm import  tqdm
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import random
sys.path.append('/data/Hszhu/Reggio')
# from simple_lama_inpainting import SimpleLama
from lama import lama_with_refine
import argparse
from src.demo.model import AutoPipeReggio
import torch
import cv2
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
import clip
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline,UNet2DConditionModel
from MAT.generate_image import create_and_load_mat_model,mat_forward
def load_clip_on_the_main_Model(main_model,device):
    # 加载CLIP模型和处理器
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    main_model.clip = model
    main_model.clip_process = preprocess
    return main_model
import clip
import warnings
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
def save_imgs(imgs, dst_dir, da_name):
    # 创建子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 用于存储保存的图片路径
    img_paths = []

    # 保存每个图片到子文件夹中
    for idx, img in enumerate(imgs):
        img_path = os.path.join(subfolder_path, f"img_{idx + 1}.png")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 将图片保存为png格式 (注意：需转换为BGR格式)
        print(f"Saved image {idx + 1} to {img_path}")
        img_paths.append(img_path)

    return img_paths

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
def get_matching_score(pos_features,neg_features,target_features):
    sim_pos = torch.matmul(target_features, pos_features.T).mean(dim=-1)
    sim_neg = torch.matmul(target_features,neg_features.T).mean(dim=-1)
    return sim_pos - sim_neg



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
def judge_exist(da_n,dst_dir,instances):
    # 创建子文件夹
    inp_save_path = os.path.join(dst_dir, da_n)
    ins_num = len(instances['obj_label'])
    img_path = os.path.join(inp_save_path, f"img_{ins_num}.png")
    #judge whether last instance is saved
    return osp.exists(img_path)



def main(data_id, base_dir):
    dst_base = osp.join(base_dir, f'Subset_{data_id}')
    # if osp.exists(osp.join(dst_base, f"packed_data_full_INP_{data_id}.json")):
    #     print(f'expansion inpainting for {data_id} already finish!')
    #     return
    # dataset_json_file = osp.join(dst_base, f"packed_data_full_tag_{data_id}.json") #pie
    dataset_json_file = osp.join(dst_base, f"mask_tag_relabelled_lmm_{data_id}.json") #GRIT AND SUBJECT
    data = load_json(dataset_json_file)

    # dst_dir_path_exp = osp.join(dst_base, "EXP_masks/")
    dst_dir_path_inp = osp.join(dst_base, "inp_imgs/")

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-inpainting/"
    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
    vae_path = "default"
    precision = torch.float32
    model = AutoPipeReggio.from_pretrained(pretrained_model_path, torch_dtype=precision).to(device)
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    model.scheduler = DDIMScheduler.from_config(model.scheduler.config, )
    model.inpainter = lama_with_refine(device)
    # model.sd_inpainter = sd_inpainter
    # model.tag2text = tag2text_model
    model = load_clip_on_the_main_Model(model, device)
    controller = Mask_Expansion_SELF_ATTN(block_size=8, drop_rate=0.5, start_layer=10)
    controller.contrast_beta = 1.67
    controller.use_contrast = True
    model.controller = controller
    register_attention_control(model, controller)
    model.modify_unet_forward()
    model.enable_attention_slicing()
    model.enable_xformers_memory_efficient_attention()
    assist_prompt = ['shadow', ]
    # omit_label_list = ['field','sky']
    # for da_n,da in tqdm(data.items(),desc='proceeding inpainting:'):
    new_data = dict()
    for da_n, da in tqdm(data.items(), desc=f'Proceeding inpainting (data_parts {data_id})'):
        random_stat = True
        if random_stat:
            random_seed =  random.randint(0, 2**32 - 1)
            random.seed(random_seed)
            key_list = list(data.keys())
            da_n = key_list[random.randint(0, len(key_list) - 1)]
            # da_n = '15156'
            da = data[da_n]
        #for debug vis
        print(f'proceeding {da_n}')
        if da_n in ['15154','15155']:
            continue
        # da_n = '3' #378
        # da = data[da_n]

        # da = data[da_n]
        if 'instances' not in da.keys():
            print(f'skip {da_n} for not valid instance')
            continue
        image_path = da['src_img_path']
        instances_list = da['instances']
        new_data[da_n] = da
        new_data[da_n]['instances'] = []
        for instances in instances_list:
            level = instances['level']
            level_dst_dir_path_inp = osp.join(dst_dir_path_inp,f'level{level}')
            if judge_exist(da_n,level_dst_dir_path_inp,instances):
                instances['inp_img_path'] = [osp.join(level_dst_dir_path_inp, da_n,f"img_{id+1}.png") for id in range(len(instances['obj_label'])) ]
                instances['level'] = level
                new_data[da_n]['instances'].append(instances)
                print(f'skip level{level} {da_n} for already exist')
                continue
            try:
                print(f'here we are focusing on {da_n}')
                mask_list = [cv2.imread(path) for path in instances['mask_path']]  # load all masks
                img = cv2.imread(image_path)  # bgr
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # mask_list,obj_label_list,stat = filter_out_invalid_label(model,img,mask_list,obj_label_list)
                # if not stat:
                # TODO: 1.complete the obj parts and replace original mask
                # TODO: 2.use semantic ca to get surroundings as exp mask，eg: shadow
                expansion_mask_list, inpainting_imgs_list = model.expansion_and_inpainting_func_exp(img, mask_list,
                                                                                                max_resolution=512,
                                                                                                expansion_step=10,
                                                                                                samples_per_time=1,
                                                                                                assist_prompt=assist_prompt,
                                                                                                sem_expansion=True)
                # mask_path = save_masks(expansion_mask_list, dst_dir_path_exp, da_n)
                # instances['exp_mask_path'] = mask_path

                # lama_inp_path = save_imgs(lama_inp_list, dst_dir_path_inp_lama, da_n)
                # instances['lama_inp_img_path'] = lama_inp_path
                if len(inpainting_imgs_list) > 0:  # if all filtered with no sd inpainting results
                    best_inp_path = save_imgs(inpainting_imgs_list, level_dst_dir_path_inp, da_n)
                    instances['inp_img_path'] = best_inp_path
                new_data[da_n]['instances'].append(instances)  # add inp_img_path to data json
            except Exception as e:
                print(f"skip error case for: {e}")
                model.controller.reset() #avoid
                continue
    save_json(data, osp.join(dst_base, f"packed_data_full_INP_{data_id}.json"))

if __name__ == "__main__":
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