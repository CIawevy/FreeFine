from vis_utils import temp_view, temp_view_img, save_img, save_mask
import sys
sys.path.append("/data/Hszhu/GeoDiffuser/GeoDiffuser")
from utils.ui_utils2 import get_transformed_mask, get_depth, get_mask, get_edited_image
from PIL import Image
import numpy as np
import json
import os
import cv2
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
# 全局参数
LENGTH = 512  # 图像处理尺寸


def save_correspondence(coords, dst_dir, da_name, ins_name, sample_id):
    """保存点对应关系为 NPY 文件，按层级结构组织"""
    da_name = str(da_name)
    ins_name = str(ins_name)
    sample_id = str(sample_id)

    # 创建保存目录
    da_subfolder = os.path.join(dst_dir, da_name)
    os.makedirs(da_subfolder, exist_ok=True)
    ins_subfolder = os.path.join(da_subfolder, ins_name)
    os.makedirs(ins_subfolder, exist_ok=True)

    # 保存文件
    file_name = f"{sample_id}.npy"
    save_path = os.path.join(ins_subfolder, file_name)
    np.save(save_path, coords)
    print(f"对应关系已保存至: {save_path}")
    return save_path


def denormalize_coordinates(coords, image_shape):
    """将归一化坐标 ([-1, 1]) 转换为绝对像素坐标"""
    height, width = image_shape
    x_abs = ((coords[..., 0] + 1.0) / 2.0) * (width - 1)
    y_abs = ((coords[..., 1] + 1.0) / 2.0) * (height - 1)
    return np.stack([x_abs, y_abs], axis=-1)


def visualize_sampled_correspondence(src_mask, point_correspondence, transformed_img, save_path=None, sample_num=10):
    """可视化采样点的对应关系并检查合法性"""
    h, w = src_mask.shape
    abs_coords = denormalize_coordinates(point_correspondence, (h, w))

    # 提取前景像素坐标
    src_points = np.argwhere(src_mask > 0)
    if len(src_points) == 0:
        print("警告：src_mask中无前景像素！")
        return None

    # 随机采样
    sample_indices = random.sample(range(len(src_points)), min(sample_num, len(src_points)))
    sampled_src = src_points[sample_indices]
    sampled_tgt = abs_coords[sampled_src[:, 0], sampled_src[:, 1]]

    # 检查目标点合法性
    valid_mask = (sampled_tgt[:, 0] >= 0) & (sampled_tgt[:, 0] < w) & (sampled_tgt[:, 1] >= 0) & (sampled_tgt[:, 1] < h)
    valid_tgt = sampled_tgt[valid_mask]
    invalid_tgt = sampled_tgt[~valid_mask]

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.imshow(transformed_img)
    plt.scatter(sampled_src[:, 1], sampled_src[:, 0], c='green', marker='o', s=50, label='Source Points')
    plt.scatter(valid_tgt[:, 0], valid_tgt[:, 1], c='blue', marker='o', s=30, label='Valid Target Points')
    plt.scatter(invalid_tgt[:, 0], invalid_tgt[:, 1], c='red', marker='o', s=40, label='Invalid Target Points')

    # 添加箭头
    for src, tgt in zip(sampled_src, sampled_tgt):
        plt.arrow(src[1], src[0], tgt[0] - src[1], tgt[1] - src[0], color='black', alpha=0.5)

    plt.legend()
    plt.title(f"Sampled Correspondence (Sample Num={sample_num})")
    plt.axis('on')

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {
        'total_samples': len(sampled_src),
        'valid_samples': len(valid_tgt),
        'invalid_samples': len(invalid_tgt)
    }


def resize_image_and_get_constant_depth(img, depth_model="depth_anything"):
    """调整图像大小并获取深度图"""
    original_h, original_w = img.shape[0], img.shape[1]
    input_img = np.array(Image.fromarray(img).resize((LENGTH, LENGTH)))

    depth = np.ones_like(input_img)
    depth_image = np.ones_like(input_img)
    depth, depth_im_vis = get_depth(input_img, "", depth, depth_image, depth_model=depth_model)
    return input_img, depth, depth_im_vis, int(original_h), int(original_w)


def process_single_sample(
    orig_path,          # 原始图像路径
    mask_path,          # 掩码图像路径
    inp_bg_path,        # 背景图像路径
    edit_param,         # 编辑参数 [tx, ty, tz, rx, ry, rz, sx, sy, sz]
    dst_base,           # 保存根目录
    image_idx="single_sample",  # 图像标识（用于路径组织）
    edit_idx="0",       # 编辑标识
    sub_edit_idx="0",   # 子编辑标识
    depth_model="depth_anything"  # 深度模型
):
    """
    处理单个样本的推理流程

    Args:
        orig_path: 原始图像路径
        mask_path: 掩码图像路径
        inp_bg_path: 背景图像路径
        edit_param: 编辑参数列表，共9个元素
                    [translation_x, translation_y, translation_z,
                     rotation_x, rotation_y, rotation_z,
                     scale_x, scale_y, scale_z]
        dst_base: 保存结果的根目录
        image_idx: 图像标识（用于组织保存路径）
        edit_idx: 编辑标识
        sub_edit_idx: 子编辑标识
        depth_model: 深度模型名称

    Returns:
        保存路径字典
    """
    # 创建保存目录
    coarse_dir = osp.join(dst_base, f"depth_coarse3d_{depth_model}_no_blend")
    mesh_mask_dir = osp.join(dst_base, f"depth_mesh_mask")
    md_mask_dir = osp.join(dst_base, f"depth_md_mask")
    corre_point_dir = osp.join(dst_base, f"depth_correspondence")
    os.makedirs(coarse_dir, exist_ok=True)
    os.makedirs(mesh_mask_dir, exist_ok=True)
    os.makedirs(md_mask_dir, exist_ok=True)
    os.makedirs(corre_point_dir, exist_ok=True)

    # 读取图像
    try:
        # 读取原始图像
        input_image = Image.open(orig_path)
        input_image = np.array(input_image)
        # 读取背景图像
        if inp_bg_path is not None:
            inp_back_ground = cv2.cvtColor(cv2.imread(inp_bg_path), cv2.COLOR_BGR2RGB)
        else:
            # 若未提供背景图，使用原始图像非掩码区域作为背景（需先临时读取掩码）
            temp_mask = Image.open(mask_path).convert("L")
            temp_mask = temp_mask.resize(input_image.shape[:2][::-1], resample=Image.NEAREST)  # 匹配原始图像尺寸 (w, h)
            temp_mask = np.array(temp_mask) / 255.0  # 归一化到 [0,1]
            inp_back_ground = np.where(temp_mask[..., None] > 0.5, 0, input_image)  # 掩码区域置0，保留背景

        # 读取掩码图像
        mask_image = Image.open(mask_path).convert("L")
    except Exception as e:
        print(f"图像读取错误: {e}")
        return None

    # 调整图像大小并获取深度图
    input_image_resized, depth_image, _, H_txt, W_txt = resize_image_and_get_constant_depth(
        input_image, depth_model=depth_model
    )

    # 处理掩码图像
    mask_image = mask_image.resize((LENGTH, LENGTH), resample=Image.NEAREST)
    mask_image = np.stack([mask_image] * 3, axis=-1)  # 转为三通道

    # 初始化变换矩阵
    transform_in = np.eye(4)

    # 应用变换获取结果
    transformed_img, mesh_mask, full_mask, point_correspondence = get_transformed_mask(
        input_image_resized,
        mask_image,
        depth_image,
        None,
        translation_x=edit_param[0] / LENGTH,
        translation_y=edit_param[1] / LENGTH,
        translation_z=edit_param[2] / LENGTH,
        rotation_x=edit_param[3],
        rotation_y=edit_param[4],
        rotation_z=edit_param[5],
        transform_in=transform_in,
        splatting_radius=1.3,
        background_img=inp_back_ground,
        scale_x=edit_param[6],
        scale_y=edit_param[7],
        scale_z=edit_param[8],
        splatting_tau=1.0,
        splatting_points_per_pixel=15,
        focal_length=550
    )

    # 计算md_mask
    md_mask = np.where(mesh_mask, 0, full_mask)

    # 保存变换后的图像和掩码
    save_paths = {}
    save_paths["coarse_img"] = save_img(transformed_img, coarse_dir, image_idx, edit_idx, sub_edit_idx)
    save_paths["mesh_mask"] = save_mask(mesh_mask, mesh_mask_dir, image_idx, edit_idx, sub_edit_idx)
    save_paths["md_mask"] = save_mask(md_mask, md_mask_dir, image_idx, edit_idx, sub_edit_idx)

    # 处理并保存点对应关系
    h, w = input_image_resized.shape[:2]
    point_correspondence_abs = denormalize_coordinates(point_correspondence, (h, w))
    save_paths["correspondence"] = save_correspondence(
        point_correspondence_abs, corre_point_dir, image_idx, edit_idx, sub_edit_idx
    )

    # 可视化对应关系
    vis_stats = visualize_sampled_correspondence(
        src_mask=mask_image[..., 0],  # 取单通道掩码
        point_correspondence=point_correspondence,
        transformed_img=transformed_img,
        save_path=osp.join(coarse_dir, image_idx, edit_idx, f"{sub_edit_idx}_correspondence_vis.png"),
        sample_num=20
    )
    print("对应关系可视化统计:", vis_stats)

    print(f"单个样本处理完成，结果保存至: {dst_base}")
    return save_paths


# 示例用法
if __name__ == "__main__":
    # 配置参数
    # image_path = "/data/zkl/web_vis/static/3d/ori_img_path/40.png"
    # mask_path = "/data/zkl/web_vis/static/3d/ori_mask_path/40.png"
    ORIG_PATH =   "/data/zkl/web_vis/static/3d/ori_img_path/40.png"  # 替换为实际路径
    MASK_PATH = "/data/zkl/web_vis/static/3d/ori_mask_path/40.png"      # 替换为实际路径
    # INP_BG_PATH = "/data/zkl/web_vis/static/3d/inp_img_no_blend/40.png" # 替换为实际路径
    INP_BG_PATH = None
    EDIT_PARAM = [0, 0, 0, 0, -75, 0, 1.0, 1.0, 1.0]  # 示例编辑参数
    DST_BASE = "/data/Hszhu/temp2/"                # 替换为保存目录

    # 处理单个样本
    save_paths = process_single_sample(
        orig_path=ORIG_PATH,
        mask_path=MASK_PATH,
        inp_bg_path=INP_BG_PATH,
        edit_param=EDIT_PARAM,
        dst_base=DST_BASE,
        image_idx="sample_001",
        edit_idx="0",
        sub_edit_idx="0",
        depth_model="depth_anything"
    )

    # 打印保存路径
    print("保存路径:")
    for key, path in save_paths.items():
        print(f"{key}: {path}")