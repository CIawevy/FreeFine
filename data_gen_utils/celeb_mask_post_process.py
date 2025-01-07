import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt

def temp_view_img(image: Image.Image, title: str = None) -> None:
    # 如果输入不是 PIL 图像，假设是 ndarray
    if not isinstance(image, Image.Image):
        image_array = image
    else:  # 如果是 PIL 图像，转换为 ndarray
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)

    # 去除图像白边
    def remove_white_border(image_array):
        # 转换为灰度图
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 240 以上视为白色
        coords = cv2.findNonZero(thresh)  # 非零像素坐标
        x, y, w, h = cv2.boundingRect(coords)  # 获取边界框
        cropped_image = image_array[y:y + h, x:x + w]  # 裁剪图像
        return cropped_image

    # 调用去白边函数
    image_array_no_border = remove_white_border(image_array)

    # 显示图像
    fig, ax = plt.subplots(figsize=(8, 8))  # 自定义画布大小
    ax.imshow(image_array_no_border)
    ax.axis('off')  # 关闭坐标轴

    if title is not None:
        ax.set_title(title)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除多余边距
    plt.tight_layout(pad=0)  # 紧凑布局
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
    plt.margins(0, 0)  # 关闭所有边距
    plt.tight_layout(pad=0)  # 紧凑布局
    plt.show()

def retain_two_largest_connected_components(mask):
    """
    仅保留mask中面积最大的两个连通组件，其余部分设置为0。
    如果只有一个连通组件，则只返回一个。

    参数:
    mask (numpy.ndarray): 输入的二值化掩码图像。

    返回:
    list: 一个包含最多两个连通组件的掩码图像列表。
    """
    # 确保输入掩码是单通道的灰度图
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 1  # 将掩码图像二值化

    # 寻找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 如果没有连通组件，直接返回空图像列表
    if num_labels <= 1:  # 只有背景
        return []

    # 获取所有连通组件的面积（排除背景，背景索引为0）
    areas = stats[1:, cv2.CC_STAT_AREA]  # 不包括背景的面积

    # 获取面积最大的两个组件的索引
    if len(areas) > 1:
        largest_two_idx = np.argsort(areas)[-2:] + 1  # 找到最大两个组件的索引，+1 因为跳过了背景
    else:
        # 如果只有一个组件，返回它
        largest_two_idx = [np.argmax(areas) + 1]

    # 创建一个空白图像列表
    filtered_masks = []

    # 仅保留面积最大的两个组件
    for idx in largest_two_idx:
        filtered_mask = np.zeros_like(mask)
        filtered_mask[labels == idx] = 255
        if len(filtered_mask.shape) == 2:  # 仅有一个通道
            # 将二值掩码扩展到三通道
            filtered_mask = np.stack([filtered_mask] * 3, axis=-1)
        filtered_masks.append(filtered_mask)

    return filtered_masks

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
# 设置路径
output_mask_dir = '/data/Hszhu/dataset/CelebAMask-HQ/Processed_Masks/'

# 如果没有目标文件夹则创建
os.makedirs(output_mask_dir, exist_ok=True)

# 读取每个子集的 JSON 文件并进行 mask 叠加
for i in range(4):
    json_file = f'/data/Hszhu/dataset/CelebAMask-HQ/Subset_{i}/mask_label_filtered_{i}.json'


    annotations = load_json(json_file)

    # 遍历每个图像和其对应的 mask 路径
    for img_id, annotation in tqdm(annotations.items()):
        new_instances = {"mask_path": [], "obj_label": []}
        final_mask = np.zeros((512, 512, 3), dtype=np.uint8)
        mouth_mask = np.zeros((512, 512, 3), dtype=np.uint8)
        for id in range(len(annotation["instances"]["mask_path"])):
            # 读取 mask 文件
            mask_path = annotation["instances"]["mask_path"][id]
            label = annotation["instances"]["obj_label"][id]
            mask = np.array(Image.open(mask_path))
            final_mask += mask
            # 合并 lip 和 mouth 掩码
            if label == "mouth":
                mouth_mask += mask
            elif label == "earring":
                # 处理耳环的mask
                process_masks = retain_two_largest_connected_components(mask)

                # 处理分开保存两个耳环掩码
                earring_id = 0
                for earring_mask in process_masks:
                    earring_id += 1  # 更新耳环编号
                    # 保存单独的耳环掩码
                    earring_mask_path = os.path.join(output_mask_dir, f'earring_mask_{img_id}_{earring_id}_{i}.png')
                    Image.fromarray(earring_mask).save(earring_mask_path)
                    # 更新注释，单独记录每个耳环掩码
                    new_instances["mask_path"].append(earring_mask_path)
                    new_instances["obj_label"].append("earring")
            elif label != 'discard':
                new_instances["mask_path"].append(mask_path)
                new_instances["obj_label"].append(label)

        # 将大于 0 的值设置为 255，表示有效的 mask 区域
        final_mask[final_mask > 0] = 255
        mouth_mask[mouth_mask > 0] = 255


        person_mask_path = os.path.join(output_mask_dir, f'person_mask_{img_id}_{i}.png')
        Image.fromarray(final_mask).save(person_mask_path)
        # 更新原有的注释，设置 obj_label 为 'person'
        new_instances["mask_path"].append(person_mask_path)
        new_instances["obj_label"].append("person")

        mouth_mask_path = os.path.join(output_mask_dir, f'mouth_mask_{img_id}_{i}.png')
        Image.fromarray(mouth_mask).save(mouth_mask_path)
        # 更新原有的注释，设置 obj_label 为 'person'
        new_instances["mask_path"].append(mouth_mask_path)
        new_instances["obj_label"].append("mouth")
        annotations[img_id]["instances"] = new_instances


    save_json(annotations, json_file)
    print(f"Finish for Subset_{i}")
