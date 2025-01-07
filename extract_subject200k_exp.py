import os
import json
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import  tqdm
# 示例用法
from datasets import load_dataset

from data_gen_utils.split_dataset import load_json


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

def save_images_from_parquet(parquet_dir, output_dir):
    """
    从指定的 parquet 文件中提取图像并保存到指定的文件夹，同时生成一个包含图像路径和描述信息的 json 文件。

    参数:
    parquet_dir (str): 存放 parquet 文件的目录。
    output_dir (str): 用于存放提取图像的目标文件夹。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果目标文件夹不存在，则创建

    meta_data = []  # 存储所有图像的元数据（路径和描述）

    # 获取当前目录下所有的 .parquet 文件
    parquet_files = sorted([f for f in Path(parquet_dir).glob('*.parquet')])

    for parquet_file in parquet_files:
        # 读取 parquet 文件
        df = pd.read_parquet(parquet_file)

        for index, row in df.iterrows():
            # 提取图像数据并解码
            image_data = row['image']['bytes']

            # 将二进制数据转换为图像
            image = Image.open(BytesIO(image_data))

            # 创建唯一的图像文件名
            image_filename = f"image_{index}.jpg"  # 你可以根据需要修改文件格式
            image_path = os.path.join(output_dir, image_filename)

            # 保存图像到目标目录
            image.save(image_path)

            # 提取描述信息
            description = row['description']['item']  # 假设描述在 'item' 字段下

            # 将图像路径和描述信息存储到 meta_data 列表
            meta_data.append({
                'image_path': image_path,
                'description': description
            })

    # 将 meta_data 保存为 json 文件
    meta_data_file = os.path.join(output_dir, "meta_data.json")
    with open(meta_data_file, 'w') as json_file:
        json.dump(meta_data, json_file, ensure_ascii=False, indent=4)

    print(f"图像已保存到 {output_dir}，并生成了 meta_data.json 文件")


def center_crop_to_size(image, size):
    """
    将图像居中裁剪到指定大小。

    参数：
    image (PIL.Image): 需要裁剪的图像。
    size (int): 目标裁剪大小（宽和高相等）。

    返回：
    PIL.Image: 裁剪后的图像。
    """
    width, height = image.size
    left = (width - size) // 2
    upper = (height - size) // 2
    right = (width + size) // 2
    lower = (height + size) // 2
    return image.crop((left, upper, right, lower))
# # 示例：提取当前目录下所有 .parquet 文件，并保存图像到指定文件夹
# parquet_dir = '/data/Hszhu/dataset/Subjects200K/data/'  # 替换为你的 parquet 文件目录
# output_dir = '/data/Hszhu/dataset/Subjects200K/srcs/'  # 替换为你要保存图像的目标文件夹
# save_images_from_parquet(parquet_dir, output_dir)
def save_images_from_hf_dataset(dataset, output_dir):
    """
    从 Hugging Face 数据集中提取图像并保存到指定的文件夹，同时生成一个包含图像路径和描述信息的 JSON 文件。

    参数:
    dataset (DatasetDict): Hugging Face 加载的数据集。
    output_dir (str): 用于存放提取图像的目标文件夹。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果目标文件夹不存在，则创建

    meta_data = {}  # 存储所有图像的元数据（路径和描述）
    id = 0

    for split in dataset.keys():  # 遍历数据集中的每个 split（如 train, validation）
        split_data = dataset[split]
        print(f"Processing split: {split}, total samples: {len(split_data)}")

        for index, row in tqdm(enumerate(split_data), desc=f'extracting Subject200K', total=len(split_data)):
            # 提取图像数据
            image = row['image']  # 假设图像数据存储在 'image' 字段中
            collection = row['collection']
            full_description = row['description']  # 假设描述在 'description' 字段中
            item = full_description['item']
            category = full_description['category']
            des_valid = full_description['description_valid']

            if not des_valid:
                continue

            # 需要先水平切分图像，然后进行中心裁剪，得到两个 512x512 像素的图像对
            width, height = image.size
            half_width = width // 2  # 水平切分的中心位置

            # 切分图像
            left_image = image.crop((0, 0, half_width, height))  # 左侧图像
            right_image = image.crop((half_width, 0, width, height))  # 右侧图像

            # 对两个图像分别进行中心裁剪，使其大小为 512x512
            left_image = center_crop_to_size(left_image, 512)
            right_image = center_crop_to_size(right_image, 512)

            # 保存切分后的图像
            for j,img in enumerate([left_image,right_image]):
                des = full_description[f'description_{j}']
                if 'studio' in des:
                    continue
                image_path = os.path.join(output_dir, f"image_{id}.png")
                img.save(image_path)
                meta_data[str(id)]={'image_path':image_path,'description':des,'category':category,'item':item}
                id+=1



    # 将 meta_data 保存为 JSON 文件
    meta_data_file = os.path.join(base_dir, "meta_data.json")
    with open(meta_data_file, 'w', encoding='utf-8') as json_file:
        json.dump(meta_data, json_file, ensure_ascii=False, indent=4)

    print(f"图像已保存到 {output_dir}，并生成了 meta_data.json 文件")


base_dir = "/data/Hszhu/dataset/Subjects200K/"
# 从 Hugging Face 加载数据集
dataset = load_dataset(base_dir)  # 替换为你的数据集名称

# 指定图像保存目录
output_dir = "/data/Hszhu/dataset/Subjects200K/srcs/"
save_images_from_hf_dataset(dataset, output_dir)

data = load_json('/data/Hszhu/dataset/Subjects200K/meta_data.json')
data = data