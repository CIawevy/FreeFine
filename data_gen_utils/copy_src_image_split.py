import json
import os
import shutil


def copy_images_from_json(json_file, target_dir):
    # 用来维护已复制的文件路径
    copied_files = set()

    # 加载 JSON 数据
    with open(json_file, 'r') as f:
        data = json.load(f)

    def copy_image(image_path, target_path):
        """复制图像并避免重复"""
        if image_path and image_path not in copied_files:
            # 创建目标目录如果不存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            try:
                shutil.copy2(image_path, target_path)  # copy2 保留元数据
                copied_files.add(image_path)
                print(f"Copied {image_path} to {target_path}")
            except Exception as e:
                print(f"Failed to copy {image_path}: {e}")

    # 递归提取图像路径并复制
    def extract_and_copy(data):
        for key, value in data.items():
            if isinstance(value, dict):
                extract_and_copy(value)  # 如果是字典，递归调用
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        extract_and_copy(item)
            else:
                # 如果当前项是图像路径
                if isinstance(value, str):
                    if "src_img_path" ==  key:
                        # 处理原始图像路径
                        source_path = value
                        target_path = os.path.join(target_dir, os.path.basename(source_path))
                        copy_image(source_path, target_path)

    # 调用递归函数
    extract_and_copy(data)


# 示例使用
source_json = "/data/Hszhu/dataset/CelebAMask-HQ/Subset_0/mask_label_filtered_0.json"  # 输入的 JSON 文件路径
target_directory = "/data/Hszhu/dataset/CelebAMask-HQ/Subset_0/source_img_temp/"  # 目标目录

copy_images_from_json(source_json, target_directory)
