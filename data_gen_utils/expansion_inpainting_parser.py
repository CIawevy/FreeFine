import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import os.path as osp
import json
from tqdm import  tqdm
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/data/Hszhu/Reggio')
# from simple_lama_inpainting import SimpleLama
from lama import lama_with_refine
from ram.models import tag2text
import argparse
from src.demo.model import AutoPipeReggio
import torch
import cv2
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
import clip
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline,UNet2DConditionModel

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




def filter_out_invalid_label(model,img, mask_list, obj_label_list):
    """
    过滤掉不合适的标签。
    1.label 本身的前景属性不够高的部分
    2.mask部分分割不够好的部分


    参数:
    model: 提供已经载入的CLIP模型
    mask_list (list): 对应每个标签的掩码列表。
    obj_label_list (list): 对应每个对象的标签列表。
    clip_threshold (float): 用于CLIP相似度的阈值（可选）。

    返回:
    filtered_mask_list, filtered_label_list: 过滤后的掩码和标签列表。
    """
    with torch.no_grad():
        #crop img feat
        mask_list = [mask[:,:,0] for mask in mask_list]
        # cropped_img_list = [model.crop_image_with_mask(img, mask) for mask in mask_list]
        processed_img_list = model.pre_process_with_mask_list(img, mask_list)
        cropped_img_list = [model.crop_image_with_mask(im, mask) for im,mask in zip(processed_img_list,mask_list)]
        processed_cropped_img_list = [model.clip_process(model.numpy_to_pil(crop_img)[0]) for crop_img in cropped_img_list]
        image_features = [model.clip.encode_image(pro_img.to(model.device).unsqueeze(0)) for pro_img in processed_cropped_img_list]
        image_features = torch.stack([feat[0] for feat in image_features ],dim=0)
        #text feat
        # object_synonyms = [
        #     'item', 'thing', 'entity',  'substance',
        #     'material', 'artifact', 'creature', 'being', 'organism',
        #     'object',
        # ]
        # object_antonyms = [
        #     'background', 'backdrop', 'surroundings', 'context',
        #     'environment', 'scenery', 'setting', 'ambience',
        #     'space', 'void', 'nothingness', 'emptiness',]
        object_synonyms = [
             'object', 'item'
        ]
        object_antonyms = [
             'surroundings','environment', 'scenery' ]
        label_tokens = clip.tokenize(obj_label_list).to(model.device)
        object_synonyms_tokens = clip.tokenize(object_synonyms).to(model.device)
        object_antonyms_tokens = clip.tokenize(object_antonyms).to(model.device)
        label_features = model.clip.encode_text(label_tokens)
        object_synonyms_features = model.clip.encode_text(object_synonyms_tokens)
        object_antonym_features = model.clip.encode_text( object_antonyms_tokens)


        #normalize
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        label_features = label_features / label_features.norm(dim=1, keepdim=True)
        object_synonyms_features = object_synonyms_features /object_synonyms_features.norm(dim=1, keepdim=True)
        object_antonym_features = object_antonym_features / object_antonym_features.norm(dim=1, keepdim=True)

        # 通过矩阵乘法计算标签与无效词之间的余弦相似度
        # label_features: (N, D), invalid_features: (M, D)
        # sim: (N, M) 相似度矩阵

        score_t = get_matching_score(pos_features=object_synonyms_features,neg_features=object_antonym_features,target_features=label_features)
        score_i = get_matching_score(pos_features=object_synonyms_features, neg_features=object_antonym_features,target_features=image_features)
        valid_idx = [i for i, sc in enumerate(score_t) if (sc>0 and score_i[i]>0)]
        # valid_idx = [i for i, sc in enumerate(final_score) if sc>0]
        if len(valid_idx)>0:
            filtered_idx = [i for i, sc in enumerate(score_t) if i not in valid_idx]
            abandom_list = np.array(obj_label_list)[filtered_idx].tolist()
            print(f'{abandom_list} is filtered out')
            # omit_label_list=list(set(omit_label_list + abandom_list))
            filtered_mask_list = [mask_list[i] for i in valid_idx]
            filtered_label_list = [obj_label_list[i] for i in valid_idx]

            return filtered_mask_list, filtered_label_list,True,
        else:
            print(f'{obj_label_list} is all filtered out')
            return [],[],False

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
    if osp.exists(osp.join(dst_base, f"packed_data_full_INP_{data_id}.json")):
        print(f'expansion inpainting for {data_id} already finish!')
        return
    # dataset_json_file = osp.join(dst_base, f"packed_data_full_tag_{data_id}.json")
    dataset_json_file = osp.join(dst_base, f"mask_tag_relabelled_lmm_{data_id}.json")


    data = load_json(dataset_json_file)
    inp_mode = 'sd'

    # dst_dir_path_exp = osp.join(dst_base, "EXP_masks/")
    dst_dir_path_inp = osp.join(dst_base, "inp_imgs/")

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-inpainting/"
    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
    vae_path = "default"
    # Tag2Text
    TAG2TEXT_THRESHOLD = 0.64  # 0.64
    ckpt_base_dir = "/data/Hszhu/prompt-to-prompt/GroundingSAM_ckpts"
    TAG2TEXT_CHECKPOINT_PATH = osp.join(ckpt_base_dir, "tag2text_swin_14m.pth")
    RAM_CHECKPOINT_PATH = osp.join(ckpt_base_dir, "ram_swin_large_14m.pth")
    DELETE_TAG_INDEX = []  # filter out attributes and action which are difficult to be grounded
    for idx in range(3012, 3429):
        DELETE_TAG_INDEX.append(idx)

    tag2text_model = tag2text(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                              image_size=384,
                              vit='swin_b',
                              delete_tag_index=DELETE_TAG_INDEX,
                              text_encoder_type='/data/Hszhu/prompt-to-prompt/bert-base-uncased')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    tag2text_model.threshold = TAG2TEXT_THRESHOLD
    tag2text_model.eval()
    tag2text_model = tag2text_model.to(device)

    # pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-2-inpainting/"
    sd_inpainter = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_inpaint_model_path,
        safety_checker=None,
        requires_safety_checker=False,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)
    sd_inpainter._progress_bar_config = {"disable": True}
    sd_inpainter.enable_attention_slicing()
    precision = torch.float32
    model = AutoPipeReggio.from_pretrained(pretrained_model_path, torch_dtype=precision).to(device)
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    model.scheduler = DDIMScheduler.from_config(model.scheduler.config, )
    model.inpainter = lama_with_refine(device)
    model.sd_inpainter = sd_inpainter
    model.tag2text = tag2text_model
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

        # da_n = '3' #378
        # da = data[da_n]
        # da_n = list(data.keys())[202]
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
                mask_list = [cv2.imread(path) for path in instances['mask_path']]  # load all masks
                obj_label_list = instances['obj_label']
                img = cv2.imread(image_path)  # bgr
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # mask_list,obj_label_list,stat = filter_out_invalid_label(model,img,mask_list,obj_label_list)
                # if not stat:
                # TODO: 1.complete the obj parts and replace original mask
                # TODO: 2.use semantic ca to get surroundings as exp mask，eg: shadow
                expansion_mask_list, inpainting_imgs_list = model.expansion_and_inpainting_func(img, mask_list,
                                                                                                obj_label_list,
                                                                                                max_resolution=512,
                                                                                                expansion_step=10,
                                                                                                max_try_times=10,
                                                                                                samples_per_time=5,
                                                                                                assist_prompt=assist_prompt,
                                                                                                mode=inp_mode,
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