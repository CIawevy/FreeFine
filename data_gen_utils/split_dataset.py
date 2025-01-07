import json
import os
from sklearn.model_selection import train_test_split


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


def save_json(data_dict, file_path):
    """
    将字典保存为 JSON 文件

    Args:
        data_dict (dict): 需要保存的字典
        file_path (str): JSON 文件的保存路径
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)


def split_dataset_by_ratio(data, train_ratio, dest_dir):
    """
    根据图像数量的比例将数据划分为训练集和测试集。

    Args:
        data (dict): 原始的完整数据集
        train_ratio (float): 训练集所占的比例，范围在 (0, 1)
        dest_dir (str): 目标文件夹路径
    """
    # 获取所有图像ID
    img_ids = list(data.keys())

    # 使用 sklearn 的 train_test_split 按照指定比例划分图像ID
    train_img_ids, test_img_ids = train_test_split(img_ids, train_size=train_ratio, random_state=42)

    # 根据划分的图像ID构建训练集和测试集
    train_data = {img_id: data[img_id] for img_id in train_img_ids}
    test_data = {img_id: data[img_id] for img_id in test_img_ids}

    # 保存划分后的训练集和测试集
    train_json_path = os.path.join(dest_dir, 'annotations_train.json')
    test_json_path = os.path.join(dest_dir, 'annotations_test.json')
    save_json(train_data, train_json_path)
    save_json(test_data, test_json_path)

    print(f"数据集已划分并保存：")
    print(f"训练集路径：{train_json_path}")
    print(f"测试集路径：{test_json_path}")
    print(f"训练集数量：{len(train_data)}")
    print(f"测试集数量：{len(test_data)}")


# 指定路径
import os.path as osp
dest_dir = "/data/Hszhu/dataset/Edit-PIE"
annotations_path = osp.join(dest_dir,"annotations.json")
train_ratio = 0.95  # 例如 80% 训练集

# 加载数据
data = load_json(annotations_path)

# 划分数据集
split_dataset_by_ratio(data, train_ratio, dest_dir)
