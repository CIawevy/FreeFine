import json
import os.path as osp
from tqdm import  tqdm
import json
import os
import shutil
from PIL import Image, ImageOps
from PIL import Image, ImageEnhance
from PIL import  Image
from PIL import Image, ImageFile
import argparse


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

def resize_to_match(image, target_size):
    """
    将图片调整为目标大小。
    :param image: PIL.Image 对象
    :param target_size: 目标尺寸 (宽, 高)
    :return: 调整后的 PIL.Image 对象
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)
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
    for da_n, da in tqdm(data.items()):
        if 'instances' not in da:
            continue
        if da_n not in stat_data:
            stat_data[da_n] = {'stat': 'unprocessed'}
        if stat_data[da_n].get('stat') == 'processed':
            continue
        src_img_path = da['src_img_path']
        # 打开原图
        ori_img = Image.open(src_img_path).convert("RGBA")
        """
        """
        for lvl in range(len(da['instances'])):
            instances = da['instances'][lvl]
            anno_path = osp.join(dst_base, f'Subset_{i}/masks_tag/level{lvl+1}/{da_n}/anotated_img.png')
            anno_img = Image.open(anno_path)
            target_size = ori_img.size  # 获取原图尺寸作为目标大
            anno_img = crop_to_content(anno_img)  # 去掉白边
            anno_img = resize_to_match(anno_img, target_size)  # 调整为原图尺寸
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
                if mask_img.size != ori_img.size:
                    mask_img = mask_img.resize(ori_img.size, Image.NEAREST)

                # 生成合成图像：裁剪掩码区域，其他部分填充为白色
                white_background = Image.new("RGBA", ori_img.size, (255, 255, 255, 255))  # 白色背景
                crop_img = Image.composite(ori_img, white_background, mask_img)  # 保留掩码区域
                # 保存合成图像
                crop_img.save(vis_mask_path)
        stat_data[da_n]['stat']= 'processed'
        save_json(stat_data, stat_json_path)


def get_num_dict(data):
    """根据子集数据生成 num_dict"""
    return {
        da_n: {
            inst['level']: len(inst['mask_path']) for inst in da.get('instances', [])
        } for da_n, da in data.items()
    }

if __name__ == "__main__":
    name_list = ['nick','zhenzhu','zkl','Clawer']
    progress_data =load_json("/data/Hszhu/dataset/Subjects200K/progress_stat")
    for id,i in tqdm(enumerate([0,1,5,6])):
        name = name_list[id]
        ori_data_stat_path = f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/mask_tag_relabelled_lmm_v2_{i}.json"
        data_stat_path = f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/mask_label_filter_stat.json"
        meta_data_path = f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/mask_label_filtered_{i}.json"
        user_info_path = f"/data/Hszhu/dataset/Subjects200K/users/{name}/user_info.json"
        user_info = load_json(user_info_path)
        data = load_json(data_stat_path)
        meta_data = load_json(meta_data_path)
        ori_data = load_json(ori_data_stat_path)
        num_dict = get_num_dict(ori_data)
        da_n_list = list(meta_data.keys())
        processed_num = 0
        for da_n, da in data.items():
            processed_num += sum(num_dict[da_n].values())
            print(f'finish {da_n}')
            data[da_n]['status'] = 'completed'
            if da_n == da_n_list[-1]:
                break
        data['processed_mask_results'] = processed_num
        progress = "{:.2f}".format(
            (data['processed_mask_results'] / data['total_mask_results']) * 100
        )
        progress_data[str(i)]['progress'] = float(progress)
        user_info['contributions'][i] = float(progress)
        user_info['start_progress'] = float(progress)
        save_json(user_info, user_info_path)
        save_json(data, data_stat_path)
        save_json(progress_data, "/data/Hszhu/dataset/Subjects200K/progress_stat")

