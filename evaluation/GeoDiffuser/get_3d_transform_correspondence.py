from vis_utils import  temp_view,temp_view_img,save_img,save_mask
import sys
# from xml.dom.minidom import Notation
sys.path.append("/data/Hszhu/GeoDiffuser/")
from utils.ui_utils import get_transformed_mask, get_depth, get_mask, get_edited_image
from PIL import Image
import numpy as np
import json
import os
import cv2
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
LENGTH = 512
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def save_correspondence(coords, dst_dir, da_name, ins_name, sample_id):
    """
    保存点对应关系为 NPY 文件，按层级结构组织

    Args:
        coords: 点对应关系数组 (H, W, 2)
        dst_dir: 目标根目录
        da_name: 数据标识（如 image_idx）
        ins_name: 实例标识（如 edit_idx）
        sample_id: 样本标识（如 sub_edit_idx）

    Returns:
        保存路径
    """
    # 转换为字符串
    da_name = str(da_name)
    ins_name = str(ins_name)
    sample_id = str(sample_id)

    # 创建数据子文件夹
    da_subfolder = os.path.join(dst_dir, da_name)
    os.makedirs(da_subfolder, exist_ok=True)

    # 创建实例子文件夹
    ins_subfolder = os.path.join(da_subfolder, ins_name)
    os.makedirs(ins_subfolder, exist_ok=True)

    # 保存 NPY 文件
    file_name = f"{sample_id}.npy"
    save_path = os.path.join(ins_subfolder, file_name)
    np.save(save_path, coords)

    print(f"对应关系已保存至: {save_path}")
    return save_path
def denormalize_coordinates(coords, image_shape):
    """
    将归一化坐标 ([-1, 1]) 转换为绝对像素坐标 (0, W-1, 0, H-1)

    Args:
        coords: 归一化坐标 (H, W, 2)
        image_shape: 图像尺寸 (height, width)

    Returns:
        abs_coords: 绝对像素坐标 (H, W, 2)
    """
    height, width = image_shape
    x_abs = ((coords[..., 0] + 1.0) / 2.0) * (width - 1)
    y_abs = ((coords[..., 1] + 1.0) / 2.0) * (height - 1)
    return np.stack([x_abs, y_abs], axis=-1)


import numpy as np
import matplotlib.pyplot as plt
import random


def visualize_sampled_correspondence(src_mask, point_correspondence, transformed_img, save_path=None, sample_num=10):
    """
    可视化采样点的对应关系，并检查目标位置合法性

    Args:
        src_mask: 原始掩码 (H, W)，二值图像
        point_correspondence: 归一化对应坐标 (H, W, 2)
        transformed_img: 变换后的图像 (H, W, 3)
        save_path: 保存路径（可选）
        sample_num: 采样点数（默认10）
    """
    # 转换为绝对像素坐标
    h, w = src_mask.shape
    abs_coords = denormalize_coordinates(point_correspondence, (h, w))

    # 提取src_mask中的前景像素坐标
    src_points = np.argwhere(src_mask > 0)  # (N, 2)，格式为 (y, x)
    if len(src_points) == 0:
        print("警告：src_mask中无前景像素！")
        return

    # 随机采样10个点（或取前10个）
    sample_indices = random.sample(range(len(src_points)), min(sample_num, len(src_points)))
    sampled_src = src_points[sample_indices]
    sampled_tgt = abs_coords[sampled_src[:, 0], sampled_src[:, 1]]

    # 检查目标点合法性（是否在图像范围内）
    valid_mask = (sampled_tgt[:, 0] >= 0) & (sampled_tgt[:, 0] < w) & (sampled_tgt[:, 1] >= 0) & (sampled_tgt[:, 1] < h)
    valid_tgt = sampled_tgt[valid_mask]
    invalid_tgt = sampled_tgt[~valid_mask]

    # 可视化设置
    plt.figure(figsize=(12, 6))
    plt.imshow(transformed_img)

    # 绘制原始点（src_mask中的采样点，绿色点）
    plt.scatter(sampled_src[:, 1], sampled_src[:, 0], c='green', marker='o', s=50, label='Source Points')

    # 绘制有效目标点（蓝色点）
    plt.scatter(valid_tgt[:, 0], valid_tgt[:, 1], c='blue', marker='o', s=30, label='Valid Target Points')

    # 绘制无效目标点（红色点）
    plt.scatter(invalid_tgt[:, 0], invalid_tgt[:, 1], c='red', marker='o', s=40, label='Invalid Target Points')

    # 添加从源点到目标点的箭头
    for src, tgt in zip(sampled_src, sampled_tgt):
        plt.arrow(src[1], src[0], tgt[0] - src[1], tgt[1] - src[0], color='black', alpha=0.5)

    # 添加图例和标题
    plt.legend()
    plt.title(f"Sampled Correspondence (Sample Num={sample_num})")
    plt.axis('on')

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # 返回合法性统计
    return {
        'total_samples': len(sampled_src),
        'valid_samples': len(valid_tgt),
        'invalid_samples': len(invalid_tgt)
    }


def save_correspondence(coords, dst_dir, da_name, ins_name, sample_id):
    """
    保存点对应关系为 NPY 文件，按层级结构组织

    Args:
        coords: 点对应关系数组 (H, W, 2)
        dst_dir: 目标根目录
        da_name: 数据标识（如 image_idx）
        ins_name: 实例标识（如 edit_idx）
        sample_id: 样本标识（如 sub_edit_idx）

    Returns:
        保存路径
    """
    # 转换为字符串
    da_name = str(da_name)
    ins_name = str(ins_name)
    sample_id = str(sample_id)

    # 创建数据子文件夹
    da_subfolder = os.path.join(dst_dir, da_name)
    os.makedirs(da_subfolder, exist_ok=True)

    # 创建实例子文件夹
    ins_subfolder = os.path.join(da_subfolder, ins_name)
    os.makedirs(ins_subfolder, exist_ok=True)

    # 保存 NPY 文件
    file_name = f"{sample_id}.npy"
    save_path = os.path.join(ins_subfolder, file_name)
    np.save(save_path, coords)

    print(f"对应关系已保存至: {save_path}")
    return save_path
def resize_image_and_get_constant_depth(img):
    original_h, original_w = img.shape[0], img.shape[1]
    input_img = np.array(Image.fromarray(img).resize((LENGTH, LENGTH)))

    depth = np.ones_like(input_img)
    depth_image = np.ones_like(input_img)
    depth, depth_im_vis = get_depth(input_img, "", depth, depth_image, depth_model = "depth_anything") ## Define the depth model here 

    return input_img, depth, depth_im_vis, int(original_h), int(original_w)


# Read input image and convert to numpy array

annotations_path = "/data/Hszhu/dataset/Geo-Bench-3D/annotations.json"
dst_base = "/data/Hszhu/dataset/Geo-Bench-3D/"
coarse_dir = osp.join(dst_base, f"coarse3d_depth_anything_no_blend")
mesh_mask_dir = osp.join(dst_base, f"mesh_mask")
md_mask_dir = osp.join(dst_base, f"md_mask")
corre_point_dir = osp.join(dst_base, f"correspondence")
# dst_dir_path_coar = osp.join(dst_base, "coarse_inputs/")
# dict(edit_prompt: coarse input ,ori_Img ,ori_mask,target_mask)
os.makedirs(coarse_dir, exist_ok=True)
os.makedirs(mesh_mask_dir, exist_ok=True)
os.makedirs(md_mask_dir, exist_ok=True)
os.makedirs(corre_point_dir, exist_ok=True)
with open(annotations_path, "r") as f:
    data = json.load(f)

for image_idx in data.keys():
    edit_indices = data[image_idx]["instances"]
    for edit_idx in edit_indices.keys():
        for sub_edit_idx in edit_indices[edit_idx].keys():
            # inp_back_ground = cv2.cvtColor(cv2.imread(osp.join(dst_base,f'inp_img_blended/{image_idx}/{edit_idx}/inp_img.png')),cv2.COLOR_BGR2RGB )  # bgr
            inp_back_ground = cv2.cvtColor(
                cv2.imread(osp.join(dst_base, f'inp_img_no_blend/{image_idx}/{edit_idx}/inp_img.png')),
                cv2.COLOR_BGR2RGB)  # bgr
            orig_path = edit_indices[edit_idx][sub_edit_idx]["ori_img_path"]
            mask_path = edit_indices[edit_idx][sub_edit_idx]["ori_mask_path"]
            edit_param = edit_indices[edit_idx][sub_edit_idx]["edit_param"]

            input_image = Image.open(orig_path)
            input_image = np.array(input_image)

            ## Also directly copied from the Geodiffuser repo, do image resizing and get depth image. You can select the depth model to use, see the get_depth function
            input_image, depth_image, depth_image_vis, H_txt, W_txt = resize_image_and_get_constant_depth(input_image)

            ## Get the mask image
            mask_image = Image.open(mask_path).convert("L")
            mask_image = mask_image.resize((LENGTH, LENGTH), resample=Image.NEAREST)
            mask_image = np.stack([mask_image]*3, axis=-1)

            transform_in = np.eye(4)

            #trasnformed_img not used, only for visualization. transform_mat is the transformation matrix used for real editing
            trasnformed_img, mesh_mask,full_mask,point_correspondence = get_transformed_mask(input_image,
                                                mask_image, 
                                                depth_image,
                                                None, # Basically a None when I checked
                                                translation_x=edit_param[0]/LENGTH, 
                                                translation_y=edit_param[1]/LENGTH, 
                                                translation_z=edit_param[2]/LENGTH, 
                                                rotation_x=edit_param[3], 
                                                rotation_y=edit_param[4], 
                                                rotation_z=edit_param[5],
                                                transform_in=transform_in, # See above, basically an identity matrix, it will be modified in this function
                                                splatting_radius = 1.3, 
                                                background_img = inp_back_ground,
                                                scale_x = edit_param[6],
                                                scale_y = edit_param[7],
                                                scale_z = edit_param[8],
                                                splatting_tau = 1.0,
                                                splatting_points_per_pixel = 15,
                                                focal_length = 550)
            md_mask = np.where(mesh_mask,0,full_mask)
            new_path = save_img(trasnformed_img, coarse_dir, image_idx,edit_idx,sub_edit_idx)
            mesh_mask_path = save_mask(mesh_mask, mesh_mask_dir, image_idx,edit_idx,sub_edit_idx)
            md_mask_path = save_mask(md_mask, md_mask_dir, image_idx,edit_idx,sub_edit_idx)
            # point_correspondence_path = save_mask(point_correspondence, corre_point_dir, image_idx,edit_idx,sub_edit_idx)# new func here needed
            h, w = input_image.shape[:2]

            # 1. 去归一化转换（假设point_correspondence是归一化坐标）
            point_correspondence_abs = denormalize_coordinates(point_correspondence, (h, w))

            # # 2. 保存对应关系（NPY格式）
            # save_correspondence(
            #     coords=point_correspondence_abs,
            #     save_dir=corre_point_dir,
            #     image_idx=image_idx,
            #     edit_idx=edit_idx,
            #     sub_edit_idx=sub_edit_idx
            # )

            # 3. 可视化采样点（从src_mask中采样10个点）
            # temp_view_img(trasnformed_img)
            # temp_view(mesh_mask)
            visualize_sampled_correspondence(
                src_mask=mask_image[..., 0],  # 假设mask_image是单通道掩码
                point_correspondence=point_correspondence,
                transformed_img=trasnformed_img,
                save_path=None,
                sample_num=20
            )
            # 2. 保存对应关系（NPY格式）
            save_correspondence(
                coords=point_correspondence_abs,
                dst_dir=corre_point_dir,
                da_name=image_idx,
                ins_name=edit_idx,
                sample_id=sub_edit_idx
            )
            print(f'finish for {image_idx} {edit_idx} {sub_edit_idx}')



