import numpy as np
import cv2
import time
import torch
import os
import torch
from torch import nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points
# from pytorch3d.renderer import FoVPerspectiveCameras, RasterizationSettings, PointsRasterizer, PointsRenderer

def normalize_radius(radius, image_size):
    height, width = image_size
    min_dim = min(height, width)
    normalized_radius = float(radius) / float(min_dim) * 2.0
    return normalized_radius
def  RasterizePointsXYsBlending(pts3D, src, size, radius,points_per_pixel,rad_pow,tau,accumulation):
    #src image :
    bs = src.size(0)
    image_size = size

    # Make sure these have been arranged in the same way
    assert pts3D.size(2) == 3
    assert pts3D.size(1) == src.size(2)

    pts3D[:,:,1] = - pts3D[:,:,1]
    pts3D[:,:,0] = - pts3D[:,:,0]

    # Add on the default feature to the end of the src
    # src = torch.cat((src, self.default_feature.repeat(bs, 1, 1)), 2)

    # radius = float(radius) / float(image_size) * 2.0
    pts3D = pts3D.float()
    src = src.float()
    radius = normalize_radius(radius, image_size)

    pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))
    points_idx, _, dist = rasterize_points(
        pts3D, image_size, radius, points_per_pixel
    )

    dist = dist / pow(radius, rad_pow)

    alphas = (
        (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
        .pow(tau)
        .permute(0, 3, 1, 2)
    )

    if accumulation == 'alphacomposite':
        transformed_src_alphas = compositing.alpha_composite(
            points_idx.permute(0, 3, 1, 2).long(),
            alphas,
            pts3D.features_packed().permute(1,0),
        )
    elif accumulation == 'wsum':
        transformed_src_alphas = compositing.weighted_sum(
            points_idx.permute(0, 3, 1, 2).long(),
            alphas,
            pts3D.features_packed().permute(1,0),
        )
    elif accumulation == 'wsumnorm':
        transformed_src_alphas = compositing.weighted_sum_norm(
            points_idx.permute(0, 3, 1, 2).long(),
            alphas,
            pts3D.features_packed().permute(1,0),
        )

    return transformed_src_alphas

def cal_shifting_coords(x_coords,dx):
    if dx == 0:
        return 0
    x_length = x_coords.max()-x_coords.min()
    return x_length*dx
def refine_transforms(transforms,points_3d):
    dx,dy,dz = transforms[:3]
    dx_new = cal_shifting_coords(points_3d[:,0],dx)
    dy_new = cal_shifting_coords(points_3d[:, 1],dy)
    dz_new = cal_shifting_coords(points_3d[:, 2],dz)
    new_trans = [dx_new,dy_new,dz_new]
    new_trans.extend(transforms[3:])
    return new_trans
def get_transformation_matrix(tx, ty, tz, rx, ry, rz, sx, sy, sz):
    """
    构建3D变换矩阵，包括平移、旋转和缩放。

    参数:
    tx, ty, tz: float
        分别沿x, y, z轴的平移量。
    rx, ry, rz: float
        分别绕x, y, z轴的旋转角度（以度为单位）。
    sx, sy, sz: float
        分别沿x, y, z轴的缩放比例。

    返回:
    numpy.ndarray
        4x4的3D变换矩阵。
    """
    # 平移矩阵
    translation_matrix = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # 绕X轴旋转矩阵
    rx = np.radians(rx)  # 将角度转换为弧度
    rotation_x_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # 绕Y轴旋转矩阵
    ry = np.radians(ry)  # 将角度转换为弧度
    rotation_y_matrix = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # 绕Z轴旋转矩阵
    rz = np.radians(rz)  # 将角度转换为弧度
    rotation_z_matrix = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # 缩放矩阵
    scaling_matrix = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # 组合所有变换矩阵
    transformation_matrix = translation_matrix @ rotation_x_matrix @ rotation_y_matrix @ rotation_z_matrix @ scaling_matrix

    return transformation_matrix




def dilate_mask(mask, dilate_factor=3):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask





def Integrated3DTransformAndInpaint(img, depth,transforms, focal_length_x, focal_length_y, mask,object_only=True):
    """
    Clawer made fantastic 3D transformation function
    """

    FINAL_WIDTH = img.shape[1]
    FINAL_HEIGHT = img.shape[0]
    if object_only:
        image = img.copy()
        img = np.where((mask>128).astype(bool)[:,:,None],img,0)
    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))

    # Normalize pixel coordinates
    x_normalized = (x - FINAL_WIDTH / 2) / focal_length_x
    y_normalized = (y - FINAL_HEIGHT / 2) / focal_length_y
    z = np.array(depth)


    # Stack coordinates and depth to create 3D points
    points_3d = np.stack((np.multiply(x_normalized, z), np.multiply(y_normalized, z), z,), axis=-1).reshape(-1, 3)

    # Convert mask to 3D points and calculate the 3D center
    mask_index = (mask.flatten()>128).astype(bool)
    masked_points_3d = points_3d[mask_index]
    center_3d = np.array([masked_points_3d[:,0].mean(), masked_points_3d[:,1].mean(), masked_points_3d[:,2].mean()])

    # Translate to object center
    masked_points_3d -= center_3d


    #refine shifting transforms

    transforms = refine_transforms(transforms,points_3d)
    T = get_transformation_matrix(*transforms)

    # Apply 3D transformation matrix on masked points
    masked_points_3d = np.hstack((masked_points_3d,np.ones_like(masked_points_3d[:,0,None])))
    transformed_masked_points_3d = (masked_points_3d @ T.T)[:, :3]

    # Translate back from object center
    transformed_masked_points_3d += center_3d[:3]
    points_3d[mask_index] = transformed_masked_points_3d



    # Reproject 3D coordinates back to 2D

    new_depth_image = np.ones_like(depth)*np.inf
    new_color_image = np.zeros_like(img)
    new_mask_image = np.zeros_like(mask)
    # 开始计时
    start_time = time.time()
    if object_only:
        for idx in np.where(mask_index)[0]:
            x, y, z = points_3d[idx]
            i = idx // FINAL_WIDTH
            j = idx % FINAL_WIDTH
            u = int((x * focal_length_x / z) + FINAL_WIDTH / 2)
            v = int((y * focal_length_y / z) + FINAL_HEIGHT / 2)
            if 0 <= u < FINAL_WIDTH and 0 <= v < FINAL_HEIGHT:
                if z < new_depth_image[v, u]:
                    new_depth_image[v, u] = z
                    new_color_image[v, u] = img[i, j]
                    new_mask_image[v, u] = mask[i, j]
    else:
        for idx, (x, y, z) in enumerate(points_3d):
            i = idx // FINAL_WIDTH
            j = idx % FINAL_WIDTH
            u = int((x * focal_length_x / z) + FINAL_WIDTH / 2)
            v = int((y * focal_length_y / z) + FINAL_HEIGHT / 2)
            if 0 <= u < FINAL_WIDTH and 0 <= v < FINAL_HEIGHT:
                if z < new_depth_image[v, u]:
                    new_depth_image[v, u] = z
                    new_color_image[v, u] = img[i, j]
                    new_mask_image[v, u] = mask[i, j]
    # 结束计时
    end_time = time.time()
    # 计算执行时间
    elapsed_time = end_time - start_time
    print(f"object-only:{object_only} \n 代码执行时间: {elapsed_time} 秒")
    dilation_mask = dilate_mask(new_mask_image, dilate_factor=15)
    inpaint_mask = ((dilation_mask>128).astype(bool) & ~(new_mask_image>128).astype(bool)).astype(np.uint8)*255
    if object_only:
        concat_mask = (dilation_mask>128).astype(bool) | (mask>128).astype(bool)
        new_color_image = np.where(concat_mask[:,:,None],new_color_image,image)#防止黑边
    new_color_image= cv2.inpaint(new_color_image, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # new_depth_image = cv2.inpaint(new_depth_image, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return new_color_image, dilation_mask, new_depth_image, inpaint_mask

def Integrated3DTransformAndRasterize(img, depth,transforms, focal_length_x, focal_length_y, mask,object_only=True,
                                      splatting_radius=1.3,splatting_tau=0,splatting_points_per_pixel=15,device=None):
    """
    Clawer made fantastic 3D transformation function
    """
    EPS = 1e-2
    FINAL_WIDTH = img.shape[1]
    FINAL_HEIGHT = img.shape[0]
    if object_only:
        image = img.copy()
        img = np.where((mask>128).astype(bool)[:,:,None],img,0)
    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))

    # Normalize pixel coordinates
    x_normalized = (x - FINAL_WIDTH / 2) / focal_length_x
    y_normalized = (y - FINAL_HEIGHT / 2) / focal_length_y
    z = np.array(depth)


    # Stack coordinates and depth to create 3D points
    points_3d = np.stack((np.multiply(x_normalized, z), np.multiply(y_normalized, z), z,), axis=-1).reshape(-1, 3)

    # Convert mask to 3D points and calculate the 3D center
    mask_index = (mask.flatten()>128).astype(bool)
    masked_points_3d = points_3d[mask_index]
    center_3d = np.array([masked_points_3d[:,0].mean(), masked_points_3d[:,1].mean(), masked_points_3d[:,2].mean()])

    # Translate to object center
    masked_points_3d -= center_3d


    #refine shifting transforms

    transforms = refine_transforms(transforms,points_3d)
    T = get_transformation_matrix(*transforms)

    # Apply 3D transformation matrix on masked points
    masked_points_3d = np.hstack((masked_points_3d,np.ones_like(masked_points_3d[:,0,None])))
    transformed_masked_points_3d = (masked_points_3d @ T.T)[:, :3]

    # Translate back from object center
    transformed_masked_points_3d += center_3d[:3]
    points_3d[mask_index] = transformed_masked_points_3d #[N,3]

    #->NDC coords /z
    invalid_mask = (np.abs(points_3d[:, -1]) < EPS) #invalid depth
    zs = points_3d[:, -1]
    zs[invalid_mask] = EPS
    points_3d[:,:2] = points_3d[:,:2] / zs[:,None]
    points_3d[invalid_mask] = -10 #将这些无效点在 x、y、z 三个维度上的值全部设置为 -10。



    #TODO: points_3d_coords -> point_cloud
    pts3D = torch.tensor(points_3d,device=device).unsqueeze(0) # 1 N 3
    src, msk = torch.tensor(img,device=device),torch.tensor(mask[:,:,None],device=device)
    src = torch.cat((src,msk),dim=-1).reshape(pts3D.shape[0],pts3D.shape[1],-1).permute(0,2,1) # 1 4 N
    # 开始计时
    start_time = time.time()

    transformed_src_alphas = RasterizePointsXYsBlending(pts3D, src,
                                                        size=(FINAL_HEIGHT,FINAL_WIDTH),
                                                        radius=splatting_radius,
                                                        points_per_pixel=splatting_points_per_pixel,
                                                        rad_pow=2,
                                                        tau=splatting_tau,
                                                        accumulation='alphacomposite').squeeze(0)
    transformed_img, transformed_mask = transformed_src_alphas[:3].permute(1, 2, 0), transformed_src_alphas[3]
    # 结束计时
    end_time = time.time()
    # 计算执行时间
    elapsed_time = end_time - start_time
    print(f"object-only:{object_only} \n 代码执行时间: {elapsed_time} 秒")

    return transformed_img.detach().cpu().numpy() ,transformed_mask.detach().cpu().numpy()







