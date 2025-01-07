import os
import json
import time
from operator import invert

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

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


def read_img(image_path):
    img = cv2.imread(image_path)  # bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


import random
def save_imgs(imgs, dst_dir, da_name):
    # 创建子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 用于存储保存的图片路径
    img_paths = []

    # 保存每个图片到子文件夹中
    for idx, img in enumerate(imgs):
        img_path = os.path.join(subfolder_path, f"img_{idx + 1}.png")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 将图片保存为png格式 (注意：需转换为BGR格式)
        print(f"Saved image {idx + 1} to {img_path}")
        img_paths.append(img_path)

    return img_paths

def visualize_images_column( image_list):
    n = len(image_list)
    rows = n  # 每列显示一张图，所以行数就是图像的数量
    cols = 1  # 只有一列

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # 如果返回的是单一的 Axes 对象，变成列表
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i in range(n):
        axes[i].imshow(image_list[i], cmap='gray')  # 显示图像，假设是灰度图
        axes[i].axis('off')  # 关闭坐标轴

    for i in range(n, len(axes)):  # 如果多余的子图没有被使用，隐藏它们
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)  # 设置子图之间的间距为0
    plt.margins(0, 0)  # 关闭所有边距
    plt.tight_layout(pad=0)  # 紧凑布局
    plt.show()

def visualize_images( image_list):
    n = len(image_list)
    cols = 3  # 每行3张图
    rows = (n + cols - 1) // cols  # 计算需要多少行

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # 如果返回的是单一的 Axes 对象，变成列表
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for i in range(n):
        axes[i].imshow(image_list[i], cmap='gray')  # 显示图像，假设是灰度图
        axes[i].axis('off')  # 关闭坐标轴

    for i in range(n, len(axes)):  # 如果多余的子图没有被使用，隐藏它们
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)  # 设置子图之间的间距为0
    plt.margins(0, 0)  # 关闭所有边距
    plt.tight_layout(pad=0)  # 紧凑布局
    plt.show()


def gaussian_mask(width, height, center, sigma_x, sigma_y):
    """
    生成二维各向异性高斯掩码
    width: 图像宽度
    height: 图像高度
    center: 高斯分布的中心 (cx, cy)
    sigma_x: 水平方向的标准差
    sigma_y: 垂直方向的标准差
    """
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    # 计算高斯分布
    gaussian = np.exp(-((X - center[0])**2 / (2 * sigma_x**2) + (Y - center[1])**2 / (2 * sigma_y**2)))
    return gaussian

def create_gaussian_mask_for_target(tgt_mask_path):
    """
    生成目标掩码的高斯掩码

    Args:
        tgt_mask_path (str): 目标掩码路径

    Returns:
        Image: 生成的高斯掩码图像
    """
    # 加载目标掩码图像
    tgt_mask = Image.open(tgt_mask_path).convert("L")  # 转为灰度图像，单通道

    # 转为 numpy 数组
    tgt_mask_np = np.array(tgt_mask)

    # 找到掩码的非零区域 (目标区域)
    non_zero_pixels = np.argwhere(tgt_mask_np > 0)

    # 获取目标区域的中心
    center_y, center_x = non_zero_pixels.mean(axis=0)

    # 计算目标区域的 bounding box
    top_left = np.min(non_zero_pixels, axis=0)
    bottom_right = np.max(non_zero_pixels, axis=0)
    box_height = bottom_right[0] - top_left[0]
    box_width = bottom_right[1] - top_left[1]
    print(f'box_height: {box_height}, box_width: {box_width}')

    # 计算 sigma_x 和 sigma_y（使用长宽的一半作为 sigma）
    sigma_x = box_width // 2
    sigma_y = box_height // 2
    print(f'box_height: {box_height}, box_width: {box_width}')
    print(f'sigma_x: {sigma_x}, sigma_y: {sigma_y}')

    # 生成高斯掩码
    gaussian = gaussian_mask(tgt_mask_np.shape[1], tgt_mask_np.shape[0], (center_x, center_y), sigma_x, sigma_y)

    # 将高斯掩码转换为图像
    gaussian_image = Image.fromarray((gaussian * 255).astype(np.uint8))

    return gaussian_image


import json
import random
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def gaussian_blob_mask(width, height, xc, yc, rmajor, rminor, theta, scale=0.4, scale_bond=0.4,inverted=False):
    """
    基于 blob 参数生成各向异性高斯掩码，并支持反转（边界值高，中心值低）

    Args:
        width (int): 图像宽度
        height (int): 图像高度
        xc (float): 椭圆中心的 x 坐标
        yc (float): 椭圆中心的 y 坐标
        rmajor (float): 椭圆的半长轴
        rminor (float): 椭圆的半短轴
        theta (float): 椭圆的旋转角度（以角度为单位）
        scale (float): 高斯分布范围的缩放比例
        inverted (bool): 是否生成反转的高斯掩码（边界值高，中心值低）

    Returns:
        np.ndarray: 生成的高斯掩码
    """

    # 生成网格坐标，注意坐标顺序
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    # 转换角度为弧度
    theta_rad = np.deg2rad(theta)

    # 缩放 sigma
    sigma_x = rmajor * scale
    sigma_y = rminor * scale

    # 旋转坐标系，将 (X, Y) 平移到 (xc, yc) 再旋转
    X_shifted = X - xc
    Y_shifted = Y - yc
    X_rot = X_shifted * np.cos(theta_rad) + Y_shifted * np.sin(theta_rad)
    Y_rot = -X_shifted * np.sin(theta_rad) + Y_shifted * np.cos(theta_rad)
    if not inverted:
        # 计算高斯分布
        gaussian = np.exp(-((X_rot ** 2) / (2 * sigma_x ** 2) + (Y_rot ** 2) / (2 * sigma_y ** 2)))

    else:
        episilon=0.01
        # 创建全零的图像
        ellipse_boundary_mask = np.ones((height, width), dtype=np.uint8)
        # 椭圆边界mask带入scipy计算边界distance map
        boundary_condition = np.abs((X_rot ** 2 / rmajor ** 2 + Y_rot ** 2 / rminor ** 2) - 1) < episilon
        ellipse_boundary_mask[boundary_condition] = 0
        dist_transform = distance_transform_edt( ellipse_boundary_mask)
        sigma_bond = min(sigma_x,sigma_y)*scale_bond
        gaussian = np.exp(-((dist_transform) ** 2) / (2 * sigma_bond ** 2))

    # 归一化到 [0, 1]
    gaussian = gaussian / np.max(gaussian)


    return gaussian




def plot_gaussian_with_ellipse(gaussian_mask, xc, yc, rmajor, rminor, theta, title):
    """
    可视化高斯掩码，并叠加绘制原始的椭球形轮廓
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gaussian_mask, cmap='hot', extent=(0, gaussian_mask.shape[1], gaussian_mask.shape[0], 0))
    ax.set_title(title)
    ax.axis('off')

    ellipse = patches.Ellipse(
        (xc, yc), width=2*rmajor, height=2*rminor, angle=theta, edgecolor='blue', facecolor='none', lw=2
    )
    ax.add_patch(ellipse)
    plt.show()
def extract_largest_blob_from_mask(mask_path=None,mask_np=None):
    """
    从二值掩码中提取最大的 blob 参数

    Args:
        mask_path (str): 掩码文件路径

    Returns:
        tuple: 最大的 blob 参数 (xc, yc, rmajor, rminor, theta)，如果没有 blob 则返回 None
    """
    # 加载掩码图像
    if mask_np is None:
        mask = Image.open(mask_path).convert("L")  # 转为灰度图
        mask_np = np.array(mask)

    # 二值化处理
    _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)

    # 找到轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有轮廓，返回 None
    if not contours:
        print("No contours found in the mask.")
        return None

    # 找到面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 如果轮廓点数少于 5，则无法拟合椭圆
    if len(largest_contour) < 5:
        print("Largest contour has less than 5 points, cannot fit an ellipse.")
        return None

    # 椭圆拟合
    ellipse = cv2.fitEllipse(largest_contour)
    (xc, yc), (rmajor, rminor), theta = ellipse

    # 返回最大的 blob 参数
    return (xc, yc, rmajor / 2, rminor / 2, theta)  # 注意半长轴和半短轴为 rmajor/2, rminor/2
def generate_gaussian_mask(mask_np, scale=0.7,inverted=False,gs_bond_scale=0.7):
    """
    从目标掩码中提取最大的 blob 并生成高斯掩码

    Args:
        mask_np: channel 3
        scale (float): 高斯掩码的缩放比例

    Returns:
        np.ndarray: 生成的高斯掩码
    """

    pil_mask = Image.fromarray(mask_np)
    # Step 2: 从掩码中提取最大的 blob
    largest_blob = extract_largest_blob_from_mask(mask_np=mask_np)

    xc, yc, rmajor, rminor, theta = largest_blob
    # print(f"Largest Blob: (xc={xc:.2f}, yc={yc:.2f}, rmajor={rmajor:.2f}, rminor={rminor:.2f}, theta={theta:.2f})")

    # Step 3: 根据最大的 blob 生成高斯掩码
    img_width, img_height = pil_mask.size
    gaussian_mask = gaussian_blob_mask(img_width, img_height, xc, yc, rmajor, rminor, theta, scale=scale,scale_bond=gs_bond_scale,inverted=inverted)*255
    gaussian_mask = gaussian_mask.astype(np.uint8)
    # # 可视化高斯掩码和椭球轮廓
    # plot_gaussian_with_ellipse(gaussian_mask, xc, yc, rmajor, rminor, theta, title="Gaussian Mask for Largest Blob")
    return gaussian_mask


def generate_paired_gaussian_mask(src_mask_np,item=None,vis=False,param=None):
    """
    从源掩码生成高斯掩码，并通过 edit_param 中的 dx, dy, sx, sy 将其变换为目标掩码

    Args:
        src_mask_np: 源掩码图像 (channel 3)
        target_mask_np: 目标掩码图像 (channel 3)
        item: 包含 edit_param 和 edit_prompt 的字典

    Returns:
        np.ndarray: 生成的源和目标高斯掩码
    """
    # 获取 edit_param 和 edit_prompt
    # edit_prompt = item['edit_prompt']
    if param is not None:
        dx, dy, _, _, _, rz, sx, sy, _ = param
    else:
        dx, dy, _, _, _, rz, sx, sy, _ = item['edit_param']

    pil_mask = Image.fromarray(src_mask_np)
    img_width, img_height = pil_mask.size
    if vis and item is not None:
        temp_view_img(Image.open(item['ori_img_path']))
        temp_view_img(pil_mask)
        temp_view_img(Image.open(item['gen_img_path']))
        # temp_view_img(Image.fromarray(target_mask_np))

    # Step 2: 从源掩码中提取最大的 blob
    largest_blob = extract_largest_blob_from_mask(mask_np=src_mask_np)

    xc, yc, rmajor, rminor, theta = largest_blob
    # 输出最大的 blob 参数，调试时可以开启
    # print(f"Largest Blob: (xc={xc:.2f}, yc={yc:.2f}, rmajor={rmajor:.2f}, rminor={rminor:.2f}, theta={theta:.2f})")

    # Step 3: 根据 edit_param 中的 dx, dy, sx, sy 对源椭圆参数进行调整，得到目标椭圆
    # 调整源椭圆中心和长短轴
    tgt_xc = xc + dx  # 目标椭圆中心 x 坐标
    tgt_yc = yc + dy  # 目标椭圆中心 y 坐标
    tgt_rmajor = rmajor * sx  # 目标椭圆长轴
    tgt_rminor = rminor * sy  # 目标椭圆短轴
    tgt_theta = theta + rz  # 目标旋转角

    # 生成源高斯掩码
    src_gaussian_mask = gaussian_blob_mask(img_width, img_height, xc, yc, rmajor, rminor, theta, scale=0.7) * 255
    src_gaussian_mask = src_gaussian_mask.astype(np.uint8)

    # 生成目标高斯掩码，使用变换后的目标椭圆参数
    tgt_gaussian_mask = gaussian_blob_mask(img_width, img_height, tgt_xc, tgt_yc, tgt_rmajor, tgt_rminor, tgt_theta,
                                           scale=0.7) * 255
    tgt_gaussian_mask = tgt_gaussian_mask.astype(np.uint8)
    if vis:
        temp_view_img(Image.fromarray((src_gaussian_mask).astype(np.uint8)).convert('RGB'), title=f'Src Gaussian Mask')
        temp_view_img(Image.fromarray((tgt_gaussian_mask).astype(np.uint8)).convert('RGB'), title=f'Tgt Gaussian Mask')
        # 可视化源和目标高斯掩码及椭圆轮廓
        plot_gaussian_with_ellipse(src_gaussian_mask, xc, yc, rmajor, rminor, theta, title="Source Gaussian Mask")
        plot_gaussian_with_ellipse(tgt_gaussian_mask, tgt_xc, tgt_yc, tgt_rmajor, tgt_rminor, tgt_theta,
                                   title="Target Gaussian Mask")

    return src_gaussian_mask, tgt_gaussian_mask
if __name__ == "__main__":
    """
    高斯掩码表示探索实验
    """
    # Step 1: 从 JSON 文件中加载数据并随机选择样本
    json_path = "/data/Hszhu/dataset/Edit-PIE/GeoEdit_annotations_train.json"
    data = json.load(open(json_path, 'r', encoding='utf-8'))

    random.seed(time.time())
    idx = random.randint(0, len(data) - 1)
    print(f'displaying idx:{idx}')

    random_image_data = data[idx]

    # 加载目标掩码路径
    tgt_mask_path = random_image_data["tgt_mask_path"]
    temp_view_img(Image.open(tgt_mask_path).convert("RGB"), title='Target Mask')

    # Step 2: 从掩码中提取最大的 blob
    largest_blob = extract_largest_blob_from_mask(mask_path=tgt_mask_path)

    if largest_blob is None:
        print("No valid blobs found in the mask.")
    else:
        xc, yc, rmajor, rminor, theta = largest_blob
        print(f"Largest Blob: (xc={xc:.2f}, yc={yc:.2f}, rmajor={rmajor:.2f}, rminor={rminor:.2f}, theta={theta:.2f})")

        # Step 3: 根据最大的 blob 生成高斯掩码
        img_width, img_height = Image.open(tgt_mask_path).size
        gaussian_mask = gaussian_blob_mask(img_width, img_height, xc, yc, rmajor, rminor, theta, scale=0.7)

        # 可视化高斯掩码和椭球轮廓
        plot_gaussian_with_ellipse(gaussian_mask, xc, yc, rmajor, rminor, theta, title="Gaussian Mask for Largest Blob")
        temp_view_img(Image.fromarray((gaussian_mask*255).astype(np.uint8)).convert('RGB'), title=f'Gaussian Mask')


