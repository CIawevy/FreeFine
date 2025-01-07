import json
import os.path as osp
from tqdm import  tqdm
import json
import os
import shutil
from PIL import Image, ImageOps
from PIL import Image, ImageFile
import numpy as np
import argparse
import cv2
# import matplotlib.pyplot as plt
# 启用加载损坏的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
def crop_to_content(image):
    """
    自动裁剪图片的白边。
    :param image: PIL.Image 对象
    :return: 裁剪后的 PIL.Image 对象
    """
    # 根据图像模式设置背景颜色
    if image.mode == "RGB":
        bg_color = (255, 255, 255)
    elif image.mode == "L":
        bg_color = 255
    elif image.mode == "1":
        bg_color = True
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")

    # 去掉明显的白边
    bg = Image.new(image.mode, image.size, bg_color)
    diff = ImageOps.invert(ImageOps.autocontrast(image)).convert("L")
    bbox = diff.getbbox()
    return image.crop(bbox) if bbox else image


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


def process(i, dst_base):
    data = load_json(os.path.join(dst_base, f"Subset_{i}/mask_tag_relabelled_lmm_v2_{i}.json"))
    # data = load_json(os.path.join(dst_base, f"Subset_{i}/mask_label_filtered_{i}.json"))
    stat_json_path = osp.join(dst_base, f"Subset_{i}/flask_preprocess_stat.json")


    if os.path.exists(stat_json_path):
        stat_data = load_json(stat_json_path)  # 恢复已保存的状态
    else:
        stat_data = {da_n: {'stat': 'unprocessed'} for da_n in data.keys()}  # 文件不存在时初始化

    if '200K' in dst_base:
        replace_name = 'Subjects200K/'
    elif 'PIE' in dst_base:
        replace_name = 'PIE-Bench_v1/'
    elif 'GRIT' in dst_base:
        replace_name = 'GRIT/'
    elif 'SA' in dst_base:
        replace_name = 'SA-1B/'
    elif 'Celeb' in dst_base:
        replace_name = 'CelebAMask-HQ/'
    for da_n, da in tqdm(data.items()):
        if 'instances' not in da:
            continue
        if da_n not in stat_data:
            stat_data[da_n] = {'stat': 'unprocessed'}
        if stat_data[da_n].get('stat') == 'processed':
            continue
        src_img_path = da['src_img_path']
        # 打开原图
        ori_img = Image.open(src_img_path).convert("RGB")
        """
        """
        for lvl in range(len(da['instances'])):
            instances = da['instances'][lvl]
            anno_path = osp.join(dst_base, f'Subset_{i}/masks_tag/level{lvl+1}/{da_n}/anotated_img.png')
            anno_img = Image.open(anno_path)
            target_size = ori_img.size  # 获取原图尺寸作为目标大
            print( anno_path)
            anno_img = crop_to_content(anno_img)  # 去掉白边
            anno_img = anno_img.resize(target_size, Image.Resampling.LANCZOS)
            anno_img.save(anno_path)
            # 保存所有掩码图像
            for mask_id, mask_path in enumerate(instances['mask_path']):
                # 替换原始路径的PIE-Bench_v1后加一个文件夹vis_temp
                vis_mask_path = mask_path.replace(f'{replace_name}', f'{replace_name}vis_temp/')
                # 确保vis_temp文件夹存在
                os.makedirs(os.path.dirname(vis_mask_path), exist_ok=True)

                # 获取掩码图像
                mask_img = Image.open(mask_path).convert("L")

                # 调整掩码尺寸以匹配原图
                if mask_img.size != target_size:
                    mask_img = mask_img.resize(target_size, Image.NEAREST)

                # 生成合成图像：裁剪掩码区域，其他部分填充为白色
                ori_img_nd = np.array(ori_img)
                mask_img = np.array(mask_img)
                new_img = np.where(mask_img[:,:,None],ori_img_nd,255)
                new_img= cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

                # 使用cv2.imwrite保存图像
                cv2.imwrite(vis_mask_path, new_img)
                # temp_view_img(np.array(crop_img))
        stat_data[da_n]['stat']= 'processed'
        save_json(stat_data, stat_json_path)

def process2(i, dst_base):
    # data = load_json(os.path.join(dst_base, f"Subset_{i}/mask_tag_relabelled_lmm_v2_{i}.json"))
    data = load_json(os.path.join(dst_base, f"Subset_{i}/mask_label_filtered_{i}.json"))


    stat_json_path = osp.join(dst_base, f"Subset_{i}/flask_preprocess_stat.json")


    if os.path.exists(stat_json_path):
        stat_data = load_json(stat_json_path)  # 恢复已保存的状态
    else:
        stat_data = {da_n: {'stat': 'unprocessed'} for da_n in data.keys()}  # 文件不存在时初始化

    if '200K' in dst_base:
        replace_name = 'Subjects200K/'
    elif 'PIE' in dst_base:
        replace_name = 'PIE-Bench_v1/'
    elif 'GRIT' in dst_base:
        replace_name = 'GRIT/'
    elif 'SA' in dst_base:
        replace_name = 'SA-1B/'
    elif 'Celeb' in dst_base:
        replace_name = 'CelebAMask-HQ/'
    for da_n, da in tqdm(data.items()):
        if 'instances' not in da:
            continue
        if da_n not in stat_data:
            stat_data[da_n] = {'stat': 'unprocessed'}
        if stat_data[da_n].get('stat') == 'processed':
            continue
        src_img_path = da['src_img_path']
        # 打开原图
        ori_img = Image.open(src_img_path).convert("RGB")
        """
        """
        for lvl in range(len(da['instances'])):
            instances = da['instances'][lvl]
            # 保存所有掩码图像
            for mask_id, mask_path in enumerate(instances['mask_path']):
                # 替换原始路径的PIE-Bench_v1后加一个文件夹vis_temp
                vis_mask_path = mask_path.replace(f'{replace_name}', f'{replace_name}vis_temp/')
                # vis_mask_path ="/data/Hszhu/dataset/Subjects200K/vis_temp/Subset_6/masks_tag/level1/22588/mask_1.png"
                # 获取掩码图像
                mask_img = cv2.imread(vis_mask_path, cv2.IMREAD_UNCHANGED)
                # temp_view_img(mask_img)
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
                # 使用cv2.imwrite保存图像
                cv2.imwrite(vis_mask_path, mask_img)
                # temp_view_img(mask_img)
        stat_data[da_n]['stat']= 'processed'
        save_json(stat_data, stat_json_path)
def process_celeb(i, dst_base):
    # data = load_json(os.path.join(dst_base, f"Subset_{i}/mask_tag_relabelled_lmm_v2_{i}.json"))
    data = load_json(os.path.join(dst_base, f"Subset_{i}/mask_label_filtered_{i}.json"))
    stat_json_path = osp.join(dst_base, f"Subset_{i}/flask_preprocess_stat.json")


    if os.path.exists(stat_json_path):
        stat_data = load_json(stat_json_path)  # 恢复已保存的状态
    else:
        stat_data = {da_n: {'stat': 'unprocessed'} for da_n in data.keys()}  # 文件不存在时初始化


    if 'Celeb' in dst_base:
        replace_name = 'CelebAMask-HQ/'
    for da_n, da in tqdm(data.items()):
        if 'instances' not in da:
            continue
        if da_n not in stat_data:
            stat_data[da_n] = {'stat': 'unprocessed'}
        if stat_data[da_n].get('stat') == 'processed':
            continue
        src_img_path = da['src_img_path']
        # 打开原图
        ori_img = Image.open(src_img_path).convert("RGB")
        """
        """
        instances = da['instances']
        target_size = ori_img.size  # 获取原图尺寸作为目标
        # 保存所有掩码图像
        for mask_id, mask_path in enumerate(instances['mask_path']):
            # 替换原始路径的PIE-Bench_v1后加一个文件夹vis_temp
            vis_mask_path = mask_path.replace(f'{replace_name}', f'{replace_name}vis_temp/')
            # 确保vis_temp文件夹存在
            os.makedirs(os.path.dirname(vis_mask_path), exist_ok=True)

            # 获取掩码图像
            mask_img = Image.open(mask_path).convert("L")

            # 调整掩码尺寸以匹配原图
            if mask_img.size != target_size:
                mask_img = mask_img.resize(target_size, Image.NEAREST)

            # 生成合成图像：裁剪掩码区域，其他部分填充为白色
            ori_img_nd = np.array(ori_img)
            mask_img = np.array(mask_img)
            new_img = np.where(mask_img[:,:,None],ori_img_nd,255)
            new_img= cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

            # 使用cv2.imwrite保存图像
            cv2.imwrite(vis_mask_path, new_img)
            # temp_view_img(np.array(crop_img))
        stat_data[da_n]['stat']= 'processed'
        save_json(stat_data, stat_json_path)
def temp_view_img(image, title: str = None) -> None:
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
def main(i,dst_base):
    process(i, dst_base)
    print('finish')

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="preprocess image for UI")
    parser.add_argument('--data_id', type=int, required=True, help="Data ID to process")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory path for dataset")
    # parser.add_argument('--gpu_id', type=int, default=0, help="Specify the GPU to use. Default is GPU 0")

    args = parser.parse_args()




    # 在需要时使用 device
    # model.to(device)
    # tensor = tensor.to(device)

    # 调用主逻辑并传入设备
    main(args.data_id, args.base_dir)

