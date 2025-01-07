import os
import json
import random
import re
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
# 设置路径
img_dir = '/data/Hszhu/dataset/CelebAMask-HQ/CelebA-HQ-img/'
mask_dir = '/data/Hszhu/dataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno/'

# # 可排除的类，类名为英文标签
excluded_classes = ["cloth","skin","hair","neck","necklace","eyeglass"]  # 例：排除 "cloth" 类

# 设置参数
shuffle = True  # 是否打乱顺序
seed = 42  # 随机种子
subsetnum = 10000  # 设置子集数量，默认为 None 使用所有数据，设置为具体数值时，选择该数量的子集

# 设置随机种子
random.seed(seed)

# 获取所有 mask 文件
mask_files = []
for root, dirs, files in os.walk(mask_dir):
    for file in files:
        if file.endswith(".png"):
            mask_files.append(os.path.join(root, file))

# 创建文件名到 mask 文件路径的映射
mask_dict = defaultdict(list)


for mask_file in mask_files:
    # 提取 mask 名称中的数字部分（例如：21999_u_lip.png -> 21999, u_lip）
    mask_name = os.path.basename(mask_file)

    if 'ear_r' in mask_name:
        label = 'earring'
    elif 'l_eye' in mask_name or 'r_eye' in mask_name:  #exclude eyeglass
        label = 'eye'
    elif 'l_ear' in mask_name or 'r_ear' in mask_name:
        label = 'ear'
    elif 'l_brow' in mask_name or 'r_brow' in mask_name:
        label = 'eyebrow'
    elif 'nose' in mask_name:
        label = 'nose'
    elif 'u_lip' in mask_name or 'l_lip' in mask_name or 'mouth' in mask_name:
        label = 'mouth'
    elif 'hat' in mask_name:
        label = 'hat'
    else:
        label = 'discard'

    # print(f'{mask_name}:{label}')
    img_id = mask_name.split('_')[0]  # 提取图片的 ID，假设为文件名的第一个部分
    # 如果标签在排除列表中，则跳过
    if label not in excluded_classes:
        mask_dict[img_id].append({"mask_path": mask_file, "obj_label": label})



# 获取图像文件名
img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

# 如果需要打乱顺序
if shuffle:
    random.shuffle(img_files)

# 如果需要使用子集数据
if subsetnum is not None:
    img_files = img_files[:subsetnum]
# 将图像文件列表均分成 4 个子集
subset_count = 4
subset_size = len(img_files) // subset_count
j= 0
i= 0
annotations = {}
for img_file in tqdm(img_files, desc=f"Processing Subset {i}"):
    if j == 1000:
        output_path = f'/data/Hszhu/dataset/CelebAMask-HQ/Subset_{i}/mask_label_filtered_{i}.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目标文件夹存在
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=4)
        print(f"Annotation JSON file for Subset {i} has been saved to {output_path}")
        i+=1
        if i==4:
            break
        j=0
        annotations={}

    img_id = img_file.split('.')[0]  # 假设文件名是唯一的并且不包含路径
    img_path = os.path.join(img_dir, img_file)
    if img_id in mask_dict:
        j+=1
        # 根据 img_id 构建 json 格式
        annotations[img_id] = {
            "instances": {
                "mask_path": [item["mask_path"] for item in mask_dict[img_id]],
                "obj_label": [item["obj_label"] for item in mask_dict[img_id]],
            },
            "src_img_path": img_path,
            "caption": ""  # 按照要求，caption 字段为空
        }


