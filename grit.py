import os
import random

import json
from tqdm import  tqdm

def save_data_dict_to_json(data_dict, output_path):
    """
    将 data_dict 保存为 JSON 文件。

    参数:
    - data_dict: 要保存的字典数据
    - output_path: JSON 文件的输出路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as json_file:
            # 使用 ensure_ascii=False 以支持保存非 ASCII 字符，indent=4 以美观的方式缩进保存
            json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
        print(f"数据成功保存到 {output_path}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")





def get_all_object_names(root_folder):
    """
    遍历根文件夹，返回文件的路径前缀，去除 .jpg, .txt, .json 后缀。
    """
    object_prefixes = set()

    # 遍历根文件夹下所有子文件夹
    for subdir in tqdm(os.listdir(root_folder), desc="Scanning files"):
        subdir_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subdir_path):
            # 遍历子文件夹中的所有文件
            for file in os.listdir(subdir_path):
                # 获取文件的前缀路径（去掉后缀）
                file_prefix = os.path.splitext(os.path.join(subdir_path, file))[0]
                object_prefixes.add(file_prefix)

    return list(object_prefixes)


def build_data_dict(root_folder, limit=None, random_sample=False):
    """
    根据文件的前缀路径，构建数据字典。
    data_dict 格式:
    {
        "00001": {
            "img_path": "path/to/00001.jpg",
            "caption": "This is the caption from the txt file."
        },
        ...
    }
    """
    data_dict = {}

    # 获取所有文件的前缀路径
    object_prefixes = get_all_object_names(root_folder)

    # 如果需要随机抽样，则打乱顺序
    if random_sample:
        random.shuffle(object_prefixes)
    #get full data dict and shuffle when using

    # 控制遍历样本数量，如果指定了 limit
    if limit:
        object_prefixes = object_prefixes[:limit]

    # 遍历前缀路径，构建数据字典
    for prefix in tqdm(object_prefixes, desc="Building data dict"):
        img_path = prefix + ".jpg"
        txt_path = prefix + ".txt"

        # 检查 .jpg 和 .txt 文件是否存在
        if os.path.exists(img_path) and os.path.exists(txt_path):
            # 读取 txt 文件中的 caption
            with open(txt_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()

            # 获取去掉路径后的文件名部分作为 object_id
            object_id = os.path.basename(prefix)

            # 构建字典并加入 data_dict
            data_dict[object_id] = {
                "img_path": img_path,
                "caption": caption
            }
        else:
            print(f"Warning: Could not find files for {prefix}")

    return data_dict



# 使用示例
root_folder = "/data/Hszhu/dataset/GRIT/srcs/"
data_dict = build_data_dict(root_folder, limit=None, random_sample=False)
# 使用示例
output_json_path = "/data/Hszhu/dataset/GRIT/meta_data.json"
save_data_dict_to_json(data_dict, output_json_path)