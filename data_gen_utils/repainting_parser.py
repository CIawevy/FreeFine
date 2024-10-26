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
from src.demo.model import AutoPipeReggio
import torch
import cv2
import argparse
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
import clip
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline,UNet2DConditionModel
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
def replace_mask(mask,src_mask_path):
    # 保存mask到ins子文件夹中
    cv2.imwrite(src_mask_path, mask.astype(np.uint8) * 255)
    print(f"Saved mask to {src_mask_path}")
    return src_mask_path
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


def get_constrain_areas(mask_list_path):
    mask_list = [cv2.imread(pa) for pa in mask_list_path]
    if len(mask_list)>0:
        constrain_areas = np.zeros_like(mask_list[0])
    for mask in mask_list:
        mask[mask>0] = 1
        constrain_areas +=mask
    constrain_areas[constrain_areas>0] =1
    return constrain_areas[:,:,0]

def prepare_mask_pool(instances):
    mask_pool = []
    for i in range(len(instances)):
        ins = instances[str(i)]
        if len(ins) == 0:
            continue
        mask_pool.append(ins['0']['ori_mask_path'])
    return mask_pool
def main(data_id, base_dir):
    dst_base = osp.join(base_dir, f'Subset_{data_id}')
    if osp.exists(osp.join(dst_base,f"generated_dataset_full_pack_{data_id}.json")):
        print(f'repainting for {data_id} already finish!')
        return
    dataset_json_file = osp.join(dst_base,f"coarse_input_full_pack_{data_id}.json")

    data = load_json(dataset_json_file)
    # data_parts = split_data(data, 2 , seed=42)


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
    # vae_path = "default"
    # pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-2-inpainting/"

    precision=torch.float32
    model = AutoPipeReggio.from_pretrained(pretrained_model_path,torch_dtype=precision).to(device)
    # if vae_path != "default":
    #     model.vae = AutoencoderKL.from_pretrained(
    #         vae_path
    #     ).to(model.vae.device, model.vae.dtype)

    model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
    # model.inpainter = lama_with_refine(device)
    controller = Mask_Expansion_SELF_ATTN(block_size=8,drop_rate=0.5,start_layer=10)
    controller.contrast_beta = 1.67
    controller.use_contrast = True
    model.controller = controller
    register_attention_control(model, controller)
    model.modify_unet_forward()
    model.enable_attention_slicing()
    model.enable_xformers_memory_efficient_attention()
    dst_dir_path_gen = osp.join(dst_base,"Gen_results/")
    #dict(edit_prompt: coarse input ,ori_Img ,ori_mask,target_mask)
    if osp.exists(osp.join(dst_base, f"temp_file_repaint.json")):  # resume
        new_data = load_json(osp.join(dst_base, f"temp_file_repaint.json"))
    else:
        new_data = dict()

    for da_n, da in tqdm(data.items(), desc=f'Proceeding inpainting (Part {data_id})'):
        # if 'instances' not in da.keys():
        #     print(f'skip {da_n} for not valid instance')
        #     continue
        if da_n in new_data.keys() and 'instances' in new_data[da_n].keys():
            print(f'skip {da_n} for already exist')
            continue
        instances = da['instances']
        for ins_id,current_ins in instances.items():
            if len(current_ins)==0:
                print(f'skip empty ins')
                continue
            # if ins_id =='0':
            #     continue
            mask_pool = prepare_mask_pool(instances)
            constrain_areas_strict = get_constrain_areas(mask_pool)
            for edit_ins,coarse_input_pack in current_ins.items():
                edit_prompt = coarse_input_pack['edit_prompt']
                ori_img = cv2.imread(coarse_input_pack['src_img_path'])  # bgr
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
                ori_caption  = coarse_input_pack['tag_caption']
                ori_mask = cv2.imread(coarse_input_pack['ori_mask_path'])
                obj_label = coarse_input_pack['obj_label']
                target_mask = cv2.imread(coarse_input_pack['tgt_mask_path'])
                # ddpm_region_mask =  cv2.imread(coarse_input_pack['ddpm_region_path'])
                coarse_input = cv2.imread(coarse_input_pack['coarse_input_path'])  # bgr
                coarse_input = cv2.cvtColor( coarse_input , cv2.COLOR_BGR2RGB)

                generated_results,exp_target_mask = model.generated_refine_results(ori_img,ori_mask,coarse_input,target_mask,constrain_areas_strict,obj_label,guidance_scale=7.5,eta=1.0,contrast_beta = 1.67,
                                                                   end_step = 0, num_step = 50, start_step = 25,use_mtsa = True,feature_injection=False,local_text_edit=True,local_ddpm=True,verbose=True, obj_label= obj_label)#add gen_res in input_pack
                gen_img_path = save_img(generated_results, dst_dir_path_gen, da_n,ins_id,edit_ins)
                tgt_mask_path = replace_mask(exp_target_mask,coarse_input_pack['tgt_mask_path'])
                coarse_input_pack['gen_img_path'] = gen_img_path
                current_ins[edit_ins] = coarse_input_pack
            instances[ins_id] = current_ins
        new_data[da_n] = dict()
        new_data[da_n]['instances'] = instances
        # save temp file for resume
        save_json(new_data, osp.join(dst_base, f"temp_file_repaint.json"))

    save_json(new_data,osp.join(dst_base,f"generated_dataset_full_pack_{data_id}.json"))
    # remove temp file
    os.remove(osp.join(dst_base, f"temp_file_repaint.json"))

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

