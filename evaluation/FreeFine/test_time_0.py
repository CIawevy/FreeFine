import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"
import os.path as osp
import json
from tqdm import  tqdm
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


sys.path.append('/data/Hszhu/FreeFine')
# from simple_lama_inpainting import SimpleLama
# from lama import lama_with_refine
from src.demo.model import FreeFinePipeline
import torch
import cv2
import argparse
from src.utils.attention import AttentionStore,register_attention_control,Attention_Modulator
import clip
import time
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline,UNet2DConditionModel
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
    for i,ins in instances.items():
        if len(ins) == 0:
            continue

        # 获取字典的第一个键
        first_key = next(iter(ins))

        # 将第一个键对应的 'ori_mask_path' 添加到 mask_pool
        mask_pool.append(ins[first_key]['ori_mask_path'])

    return mask_pool


def re_edit_2d(src_img, src_mask, edit_param, inp_cur):
    if len(src_mask.shape) == 3:
        src_mask = src_mask[:, :, 0]
    dx, dy, dz, rx, ry, rz, sx, sy, sz = edit_param
    rotation_angle = rz
    resize_scale = (sx, sy)
    flip_horizontal = False
    flip_vertical = False
    # Prepare foreground
    height, width = src_mask.shape[:2]
    y_indices, x_indices = np.where(src_mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        top, bottom = np.min(y_indices), np.max(y_indices)
        left, right = np.min(x_indices), np.max(x_indices)
        # mask_roi = mask[top:bottom + 1, left:right + 1]
        # image_roi = image[top:bottom + 1, left:right + 1]
        mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2
        # 检查是否有移动操作（dx 或 dy 不为零）
        if dx != 0 or dy != 0:
            # 计算物体移动后的新边界
            new_left = left + dx
            new_right = right + dx
            new_top = top + dy
            new_bottom = bottom + dy

            # 检查新边界是否超出图像的边界
            if new_left < 0 or new_right > width or new_top < 0 or new_bottom > height:
                # 如果超出边界，则丢弃或做其他处理
                assert False, 'The transformed object is out of image boundary after move, discard'

    # 将resize_scale解耦出来，实现x，y的单独缩放
    rotation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle, 1)
    # 当rotation angle=0且resize scale!=1时，由mask 中心会影响dx,dy的初始值
    # 计算公式默认rotation angle=0
    tx, ty = (1 - resize_scale[0]) * mask_center_x, (1 - resize_scale[1]) * mask_center_y
    dx += tx
    dy += ty
    rotation_matrix[0, 2] += dx
    rotation_matrix[1, 2] += dy
    rotation_matrix[0, 0] *= resize_scale[0]
    rotation_matrix[1, 1] *= resize_scale[1]

    transformed_image = cv2.warpAffine(src_img, rotation_matrix, (width, height))
    transformed_mask = cv2.warpAffine(src_mask.astype(np.uint8), rotation_matrix, (width, height),
                                      flags=cv2.INTER_NEAREST).astype(bool)

    # # 检查是否需要水平翻转
    # if flip_horizontal:
    #     transformed_mask = cv2.flip(transformed_mask.astype(np.uint8), 1).astype(bool)
    #     # transformed_mask_exp = cv2.flip(transformed_mask_exp.astype(np.uint8), 1).astype(bool)
    #
    # # 检查是否需要垂直翻转
    # if flip_vertical:
    #     transformed_mask = cv2.flip(transformed_mask.astype(np.uint8), 0).astype(bool)
    #     # transformed_mask_exp = cv2.flip(transformed_mask_exp.astype(np.uint8), 0).astype(bool)
    # if np.array_equal(transformed_mask.astype(np.uint8)*255, tgt_mask):
    #     return True,transformed_mask
    # else:
    #     return False,transformed_mask
    final_image = np.where(transformed_mask[:, :, None], transformed_image,
                           inp_cur)  # move with expansion pixels but inpaint
    return final_image, transformed_mask.astype(np.uint8)*255
def re_edit_3d(coarse_input_ori,ori_target_mask, inp_back_ground):
    reblend_img = np.where(ori_target_mask,coarse_input_ori,inp_back_ground)
    return reblend_img,ori_target_mask
from copy import deepcopy


def save_file(file_path, image):
    """
    递归创建子文件夹并保存图像文件。

    参数:
    - file_path: 目标文件路径（包括文件名）。
    - image: 要保存的图像（PIL.Image 或 NumPy 数组）。
    """
    # 获取目标文件夹路径
    dir_path = os.path.dirname(file_path)

    # 如果文件夹不存在，递归创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

    # 如果 image 是 NumPy 数组，转换为 PIL.Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # 保存图像
    image.save(file_path)
    print(f"Saved image to: {file_path}")


def draw_mask_vis(image, draw_mask, draw_mask_path):
    """
    可视化掩码并保存结果。

    参数:
    - image: 原始图像（RGB NumPy 数组）。
    - draw_mask: 掩码图像（单通道或三通道 NumPy 数组）。
    - draw_mask_path: 掩码文件路径（用于生成可视化文件路径）。
    """
    # 检查是否已存在可视化文件
    vis_path = draw_mask_path.replace('draw_mask', 'draw_mask_vis')
    if os.path.exists(vis_path):
        print(f"Skip {vis_path} (already exists)")
        return

    # 处理掩码
    if draw_mask.ndim == 3:  # 如果是三通道掩码，提取单通道
        draw_mask = draw_mask[:, :, 0]
    _, draw_mask = cv2.threshold(draw_mask, 127, 255, cv2.THRESH_BINARY)  # 二值化

    # 创建颜色层
    color_layer = image.copy()
    color_layer[draw_mask == 255] = (255, 255, 0)  # 使用 SCI 配色的绿色

    # 混合原始图像和颜色层
    a = 0.5  # 透明度
    blended_image = cv2.addWeighted(image, 1 - a, color_layer, a, 0)

    # 保存结果
    save_file(vis_path, blended_image)

def main(dst_base,method_name,data_id=None):
    total_time = 0.0
    sample_count = 0
    max_samples = 100  # 设置最大样本数
    diy = False
    # save_json_name = osp.join(dst_base, f"generated_results_refine_{method_name}.json")
    # if osp.exists(save_json_name):
    #     print(f'results already finish!')
    #     return
    dst_dir_path_gen = osp.join(dst_base, f"Gen_results_refine_ours_35_no_blend")
    # dst_dir_path_coar = osp.join(dst_base, "coarse_inputs/")
    # dict(edit_prompt: coarse input ,ori_Img ,ori_mask,target_mask)
    os.makedirs(dst_dir_path_gen, exist_ok=True)
    dataset_json_file = osp.join(dst_base,f"annotations.json")

    data = load_json(dataset_json_file)
    if diy:
        data_id = None
    if data_id is not None:
        data_parts = split_data(data, num_splits=4, seed=42)
        process_data = data_parts[int(data_id)]
    else:
        process_data = data


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
    # vae_path = "default"
    # pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-2-inpainting/"

    precision=torch.float32
    model = FreeFinePipeline.from_pretrained(pretrained_model_path, torch_dtype=precision).to(device)
    # if vae_path != "default":
    #     model.vae = AutoencoderKL.from_pretrained(
    #         vae_path
    #     ).to(model.vae.device, model.vae.dtype)
    model._progress_bar_config = {"disable": True}
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
    # model.inpainter = lama_with_refine(device)
    controller = Attention_Modulator(start_layer=10)
    model.controller = controller
    register_attention_control(model, controller)
    model.modify_unet_forward()
    model.enable_attention_slicing()
    model.enable_xformers_memory_efficient_attention()
    temp_file_name = osp.join(dst_base, f"temp_file_repaint_refine_{method_name}_{data_id}.json")
    if osp.exists(temp_file_name):  # resume
        new_data = load_json(temp_file_name)
    else:
        new_data = dict()


    for da_n, da in tqdm(process_data.items(), desc=f'Proceeding repainting ours inference)'):
        # if 'instances' not in da.keys():
        #     print(f'skip {da_n} for not valid instance')
        #     continue

        if da_n in new_data.keys() and 'instances' in new_data[da_n].keys() and not diy:
            print(f'skip {da_n} for already exist')
            continue
        instances = deepcopy(da['instances'])
        for ins_id,current_ins in instances.items():
            new_inst = deepcopy(current_ins)
            # if len(current_ins)==0:
            #     print(f'skip empty ins')
            #     continue
            # # if ins_id =='0':
            # #     continue

            for edit_ins,coarse_input_pack in current_ins.items():
                if sample_count >= max_samples:
                    print(f"\n完成 {sample_count} 个样本的推理测试")
                    print(f"总推理时间: {total_time:.4f} 秒")
                    print(
                        f"平均每个样本推理时间: {total_time / sample_count:.4f} 秒 ({(total_time / sample_count) * 1000:.2f} 毫秒) freefine edit")
                    return

                # if osp.exists(get_save_path_edit(dst_dir_path_gen, da_n, ins_id, edit_ins)) and not diy:
                #     print(f'skip {da_n}_{ins_id}_{edit_ins} for already exist')
                #     continue
                # re editing 2D for vis
                # new_edit_param = [0,0,0,0,0,0,0.4,0.4,1]
                # coarse_input,target_mask = re_edit_2d(ori_img,ori_mask,new_edit_param,inp_back_ground)
                # save_img_cr = Image.fromarray(coarse_input)
                # save_img_cr.save("new_edit_cr.png")
                # temp_view_img(coarse_input)
                # draw_mask =  cv2.imread("/data/Hszhu/Reggio/data_gen_utils/draw_mask_temp.png")  # bgr
                # draw_mask = cv2.imread("/data/Hszhu/Reggio/draw_mask_penguin.png")  # bgr
                # temp_view(draw_mask)

                # if diy:
                #     da_n = '215'  # 27 102
                #     ins_id = '0'
                #     edit_ins = '4'
                #
                #     da = data[da_n]
                #
                #     instances = deepcopy(da['instances'])
                #     edit_meta = instances[ins_id]
                #
                #     coarse_input_pack = edit_meta[edit_ins]


                mask_pool = prepare_mask_pool(instances)
                constrain_areas_strict = get_constrain_areas(mask_pool)
                constrain_areas_strict = cv2.resize(constrain_areas_strict, dsize=(512, 512),
                                                    interpolation=cv2.INTER_NEAREST)
                edit_prompt = coarse_input_pack['edit_prompt']
                edit_param = coarse_input_pack['edit_param']
                coarse_input_ori = cv2.imread(coarse_input_pack['coarse_input_path'])
                coarse_input_ori = cv2.cvtColor(coarse_input_ori, cv2.COLOR_BGR2RGB)
                ori_img = cv2.imread(coarse_input_pack['ori_img_path'])  # bgr
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
                # ori_caption  = coarse_input_pack['tag_caption']
                ori_mask = cv2.imread(coarse_input_pack['ori_mask_path'])
                ori_target_mask = cv2.imread(coarse_input_pack['tgt_mask_path'])
                ori_mask = cv2.resize(ori_mask, dsize=ori_target_mask.shape[:2], interpolation=cv2.INTER_NEAREST)
                obj_label = coarse_input_pack['obj_label']
                '''
                # target_mask = cv2.imread(coarse_input_pack['tgt_mask_path'])
                # ddpm_region_mask =  cv2.imread(coarse_input_pack['ddpm_region_path'])
                coarse_input = cv2.imread(coarse_input_pack['coarse_input_path'])  # bgr
                coarse_input = cv2.cvtColor( coarse_input , cv2.COLOR_BGR2RGB)
                '''
                # seed_r = 3787517166  # 3787517166 #4255183641 #4255183641 #2391550765
                # seed_r = random.randint(0, 10 ** 16)
                seed_r = 42
                inp_back_ground = cv2.cvtColor(cv2.imread(osp.join(dst_base,f'inp_img_no_blend/{da_n}/{ins_id}/inp_img.png')),cv2.COLOR_BGR2RGB )  # bgr
                # save_img_cr = Image.fromarray(coarse_input)
                # save_img_cr.save("new_edit_cr.png")
                type_3d=False
                if type_3d:
                    coarse_input, target_mask = re_edit_3d(coarse_input_ori,ori_target_mask, inp_back_ground)
                else:
                    coarse_input, target_mask = re_edit_2d(ori_img, ori_mask, edit_param, inp_back_ground)
                # coarse_input_path = save_img(coarse_input, dst_dir_path_coar, da_n, ins_id, edit_ins)
                # coarse_input_pack['coarse_input_path_2'] = coarse_input_path #override
                # draw_mask = cv2.imread(
                #     f"/data/Hszhu/dataset/Geo-Bench-SC/draw_mask/{da_n}/{ins_id}/draw_{edit_ins}.png")
                draw_mask = np.ones_like(ori_mask)


                # if diy:
                #     temp_view_img(coarse_input)
                #     temp_view(draw_mask)
                # # seed_r = 4255183641
                start_time = time.time()


                generated_results = model.FreeFine_generation(ori_img, ori_mask, coarse_input,
                                                              target_mask,
                                                                   "", guidance_scale=7.5,
                                                              eta=1.0, end_scale=0.0,
                                                              end_step=50, num_step=50,
                                                              start_step=25, draw_mask=draw_mask,
                                                              return_intermediates=False,
                                                              use_auto_draw=True,
                                                              reduce_inp_artifacts=False,
                                                              cons_area=constrain_areas_strict,
                                                              )  # add gen_res in input_pack
                # torch.cuda.synchronize()  # 同步GPU
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                sample_count += 1

                # 输出进度
                if sample_count % 10 == 0:
                    print(
                        f"已处理 {sample_count}/{max_samples} 样本，当前平均时间: {(total_time / sample_count) * 1000:.2f} 毫秒 freefine edit")

                # 保存图像（排除在计时外）
                # generated_results = generated_results[0].permute(1, 2, 0).detach().cpu().numpy()*255
                # gen_img_path = save_img(generated_results, dst_dir_path_gen, da_n, ins_id, edit_ins)

                # 处理完所有数据后的统计
        if sample_count > 0:
            print(f"\n完成 {sample_count} 个样本的推理测试")
            print(f"总推理时间: {total_time:.4f} 秒")
            print(
                f"平均每个样本推理时间: {total_time / sample_count:.4f} 秒 ({(total_time / sample_count) * 1000:.2f} 毫秒) freefine edit")
        else:
            print("未处理任何样本")

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    # parser = argparse.ArgumentParser(description="no sc repainting processing script")
    # parser.add_argument('--data_id', type=int, required=True, help="Data ID to process")
    # parser.add_argument('--method_name', type=str, required=True, help="attention method for sc")
    # parser.add_argument('--base_dir', type=str, required=True, help="base_dir")
    # # # parser.add_argument('--gpu_id', type=int, default=0, help="Specify the GPU to use. Default is GPU 0")
    #
    # args = parser.parse_args()




    # 在需要时使用 device
    # model.to(device)
    # tensor = tensor.to(device)
    base_dir = "/data/Hszhu/dataset/Geo-Bench/"
    main(base_dir,'tca',1)

