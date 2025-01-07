import os
import json


def create_image_json(directory):
    # 获取文件夹下所有文件
    file_list = os.listdir(directory)

    # 过滤出图像文件（假设是常见的图片格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = [file for file in file_list if any(file.lower().endswith(ext) for ext in image_extensions)]

    # 创建一个字典，用于保存结构
    result = {}

    # 遍历所有图片文件，按顺序为其赋予 id
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(directory, image_file)
        result[f'{idx}'] = {
                'image_path': image_path
        }

    # 将结果保存为 JSON 文件
    json_filename = os.path.join("/data/Hszhu/dataset/CelebAMask-HQ/", 'meta_data.json')
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=4, ensure_ascii=False)

    print(f"JSON file saved as {json_filename}")


# 使用该函数来读取文件夹中的图片并生成 JSON
directory_path = "/data/Hszhu/dataset/CelebAMask-HQ/srcs/"  # 替换为实际的文件夹路径
create_image_json(directory_path)
