import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json
from tqdm import  tqdm
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2



from src.demo.model import DesignEdit
import torch
# sys.path.append('/data/Hszhu/Reggio')
os.makedirs('models', exist_ok=True)
# subprocess.run(shlex.split(
#     'wget https://hf-mirror.com/Adapter/DragonDiffusion/resolve/main/model/efficient_sam_vits.pt -O models/efficient_sam_vits.pt'))
# from src.demo.demo import *
def temp_view_img(image: Image.Image, title: str = None) -> None:
    # Convert to ndarray if the input is not already in that format
    if not isinstance(image, Image.Image):  # ndarray
        image_array = image
    else:  # PIL
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)

    # Function to crop white borders
    def crop_white_borders(img_array):
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
        mask = gray < 255  # Mask of non-white pixels
        coords = np.argwhere(mask)  # Find the coordinates of the non-white pixels
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        return img_array[x0:x1+1, y0:y1+1]

    # Crop the white borders
    cropped_image_array = crop_white_borders(image_array)

    # Display the cropped image
    fig, ax = plt.subplots()
    ax.imshow(cropped_image_array)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')  # Hide the axis

    # Remove the white border around the figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)

    # Set the position of the axes to fill the entire figure
    ax.set_position([0, 0, 1, 1])

    # Show the image
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
    # 保存img到ins子文件夹中
    img_path = get_save_path_edit(dst_dir, da_name, ins_name,sample_id)
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved image to {img_path}")

    return img_path
def get_save_path_edit(dst_dir, da_name, ins_name,sample_id):
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
def read_and_resize_img(ori_img_path,dsize=(512,512)):
    ori_img = cv2.imread(ori_img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_img = cv2.resize(ori_img, dsize=dsize, interpolation=cv2.INTER_LANCZOS4)
    return ori_img
def read_and_resize_mask(ori_mask_path,dsize=(512,512)):
    #return 3-channel mask ndarray
    ori_mask = cv2.imread(ori_mask_path)

    ori_mask = cv2.resize(ori_mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    return ori_mask

def clear_gpu_memory():
    """基础显存清理函数"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 收集IPC内存
def main():
    base_dir = "/data/Hszhu/dataset/Geo-Bench/"
    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-xl-base-1.0"
    model = DesignEdit(pretrained_model_path=pretrained_model_path)



    # if osp.exists(osp.join(base_dir,f"generated_results_DesignEdit.json")):
    #     print(f'infering for DesignEdit already finish!')
    #     return
    dataset_json_file = osp.join(base_dir ,f"annotations.json")

    data = load_json(dataset_json_file)
    # data = split_data(data, 1 , seed=42,subset_num=min(len(data),50))[0]


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #TODO:controller here

    dst_dir_path_gen = osp.join(base_dir,"Gen_results_DesignEdit/")
    os.makedirs(dst_dir_path_gen,exist_ok=True)
    #dict(edit_prompt: coarse input ,ori_Img ,ori_mask,target_mask)
    # if osp.exists(osp.join(base_dir, f"temp_file_repaint_ed.json")):  # resume
    #     new_data = load_json(osp.join(base_dir, f"temp_file_repaint_ed.json"))
    # else:
    new_data = dict()

    for da_n, da in tqdm(data.items(), desc=f'Proceeding editing'):
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

            for edit_ins,coarse_input_pack in current_ins.items():
                # edit_prompt = coarse_input_pack['edit_prompt']
                ori_img = cv2.imread(coarse_input_pack['ori_img_path'])  # bgr
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
                # ori_caption  = coarse_input_pack['tag_caption']
                ori_mask = cv2.imread(coarse_input_pack['ori_mask_path'])
                # obj_label = coarse_input_pack['obj_label']
                # target_mask = cv2.imread(coarse_input_pack['tgt_mask_path'])
                # ddpm_region_mask =  cv2.imread(coarse_input_pack['ddpm_region_path'])
                # coarse_input = cv2.imread(coarse_input_pack['coarse_input_path'])  # bgr
                # coarse_input = cv2.cvtColor( coarse_input , cv2.COLOR_BGR2RGB)
                ori_mask = cv2.resize(ori_mask,dsize = ori_img.shape[:2],interpolation=cv2.INTER_NEAREST)
                # draw_mask = np.zeros_like(ori_mask)

                # generated_results,exp_target_mask = model.generated_refine_results(ori_img,ori_mask,coarse_input,target_mask,constrain_areas_strict,obj_label,guidance_scale=7.5,eta=1.0,contrast_beta = 1.67,
                #                                                    end_step = 0, num_step = 50, start_step = 25,use_mtsa = True,feature_injection=False,local_text_edit=True,local_ddpm=True,verbose=True, obj_label= obj_label)#add gen_res in input_pack
                edit_param = coarse_input_pack['edit_param']
                dx,dy,_,_,_,rz,sx,_,_ = edit_param
                dx/=512
                dy/=-512
                rz = -rz
                with torch.no_grad():
                    generated_results = model.infer_2d_edit(
                        ori_img, ori_img, ori_mask, dx, dy, sx, rz,
                    )
                temp_view_img(cv2.cvtColor(cv2.imread(coarse_input_pack['coarse_input_path']),cv2.COLOR_BGR2RGB))
                temp_view_img(generated_results[0])
                gen_img_path = save_img(generated_results[0], dst_dir_path_gen, da_n,ins_id,edit_ins)
                coarse_input_pack['gen_img_path'] = gen_img_path
                clear_gpu_memory()
                current_ins[edit_ins] = coarse_input_pack
            instances[ins_id] = current_ins
        new_data[da_n] = dict()
        new_data[da_n]['instances'] = instances
        # save temp file for resume
        save_json(new_data, osp.join(base_dir, f"temp_file_repaint_ed.json"))

    save_json(new_data,osp.join(base_dir,f"generated_results_DesignEdit.json"))
    # remove temp file
    os.remove(osp.join(base_dir, f"temp_file_repaint_ed.json"))







"""
Designed by Clawer ,producing DesignEdit infer results
"""
if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    main()

