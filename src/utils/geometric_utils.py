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
from pytorch3d.renderer import FoVPerspectiveCameras, RasterizationSettings, PointsRasterizer, PointsRenderer


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





def Integrated3DTransformAndInpaint(img, depth,transforms, focal_length_x, focal_length_y, mask,object_only=True,inpaint=True):
    """
    Clawer made fantastic 3D transformation function
    """

    FINAL_WIDTH = img.shape[1]
    FINAL_HEIGHT = img.shape[0]
    if object_only:
        # image = img.copy()
        img = np.where((mask>128).astype(bool)[:,:,None],img,0)
        # img = np.where(mask[:, :, None], img, 0)
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
    # mask_index = mask.flatten()
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
    new_color_image = np.full_like(img,0)
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
    if inpaint:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilation_mask = cv2.morphologyEx(new_mask_image, cv2.MORPH_CLOSE, kernel)
        inpaint_mask = ((dilation_mask>128).astype(bool) & ~(new_mask_image>128).astype(bool)).astype(np.uint8)*255
        new_color_image= cv2.inpaint(new_color_image, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        # new_depth_image = cv2.inpaint(new_depth_image, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return new_color_image, dilation_mask, new_depth_image, inpaint_mask
    else:
        return new_color_image, new_mask_image, new_depth_image

# def IntegratedP3DTransRasterBlending(img, depth,transforms, focal_length_x, focal_length_y, mask,object_only=True,
#                                       splatting_radius=1.3,splatting_tau=0,splatting_points_per_pixel=15,device=None):
#     """
#     Clawer made fantastic 3D transformation function using p3d
#     Learning p3d : https://pytorch3d.org/tutorials/render_colored_points
#     Pipe:
#     (1): rgb-d -> point_clouds using p3d
#     (2): point_clouds transformation ,given T using p3d
#     (3): convert to NDC coords and Rasterize and compositing
#     (4): final img / mask returned
#
#     """
#     EPS = 1e-2
#     FINAL_WIDTH = img.shape[1]
#     FINAL_HEIGHT = img.shape[0]
#     if object_only:
#         image = img.copy()
#         img = np.where((mask>128).astype(bool)[:,:,None],img,0)
#     # Create meshgrid for pixel coordinates
#     x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
#
#     # Normalize pixel coordinates
#     x_normalized = (x - FINAL_WIDTH / 2) / focal_length_x
#     y_normalized = (y - FINAL_HEIGHT / 2) / focal_length_y
#     z = np.array(depth)
#
#
#     # Stack coordinates and depth to create 3D points
#     points_3d = np.stack((np.multiply(x_normalized, z), np.multiply(y_normalized, z), z,), axis=-1).reshape(-1, 3)
#
#     # Convert mask to 3D points and calculate the 3D center
#     mask_index = (mask.flatten()>128).astype(bool)
#     masked_points_3d = points_3d[mask_index]
#     center_3d = np.array([masked_points_3d[:,0].mean(), masked_points_3d[:,1].mean(), masked_points_3d[:,2].mean()])
#
#     # Translate to object center
#     masked_points_3d -= center_3d
#
#
#     #refine shifting transforms
#
#     transforms = refine_transforms(transforms,points_3d)
#     T = get_transformation_matrix(*transforms)
#
#     # Apply 3D transformation matrix on masked points
#     masked_points_3d = np.hstack((masked_points_3d,np.ones_like(masked_points_3d[:,0,None])))
#     transformed_masked_points_3d = (masked_points_3d @ T.T)[:, :3]
#
#     # Translate back from object center
#     transformed_masked_points_3d += center_3d[:3]
#     points_3d[mask_index] = transformed_masked_points_3d #[N,3]
#
#     #->NDC coords /z
#     # TODO: pytorch3D camera transform
#     invalid_mask = (np.abs(points_3d[:, -1]) < EPS) #invalid depth
#     zs = points_3d[:, -1]
#     zs[invalid_mask] = EPS
#     points_3d[:,:2] = points_3d[:,:2] / zs[:,None]
#     points_3d[invalid_mask] = -10 #将这些无效点在 x、y、z 三个维度上的值全部设置为 -10。
#
#
#
#     #TODO: points_3d_coords -> point_cloud
#     pts3D = torch.tensor(points_3d,device=device).unsqueeze(0) # 1 N 3
#     src, msk = torch.tensor(img,device=device),torch.tensor(mask[:,:,None],device=device)
#     src = torch.cat((src,msk),dim=-1).reshape(pts3D.shape[0],pts3D.shape[1],-1).permute(0,2,1) # 1 4 N
#     # 开始计时
#     start_time = time.time()
#
#     transformed_src_alphas =
#     transformed_img, transformed_mask = transformed_src_alphas[:3].permute(1, 2, 0), transformed_src_alphas[3]
#     # 结束计时
#     end_time = time.time()
#     # 计算执行时间
#     elapsed_time = end_time - start_time
#     print(f"object-only:{object_only} \n 代码执行时间: {elapsed_time} 秒")
#
#     return transformed_img.detach().cpu().numpy() ,transformed_mask.detach().cpu().numpy()





class My3DTransformTools(nn.Module):
    def __init__(self,H,W,pp_pixel=15,radius=1.3,min_z=1,max_z=100):
        super().__init__()

        # 3D Points transformer
        # self.pts_transformer = MYPtsManipulator(H,W,C=3,pp_pixel=pp_pixel,radius=radius)
        self.min_z = min_z
        self.max_z = max_z

    def forward(self,input_img,depth_img,K,P,use_rgb_features=True,process_depth=True,use_inverse_depth=False):
        """ Forward pass of a view synthesis model with a voxel latent field.
        """

        # Camera parameters
        K = K
        K_inv = np.linalg.inv(K)

        identity = (torch.eye(4).unsqueeze(0).repeat(input_img.size(0), 1, 1).cuda())

        input_RT = identity
        input_RTinv = identity


        if torch.cuda.is_available():
            input_img = input_img.cuda()
            depth_img = depth_img.cuda()


            K = K.cuda()
            K_inv = K_inv.cuda()

            input_RT = input_RT.cuda()
            input_RTinv = input_RTinv.cuda()


        if use_rgb_features:
            fs = input_img
        else:
            print(f'Not implenmentation for feature projection')

        # Regressed points
        if process_depth:
            if not use_inverse_depth:
                depth_after_process = (
                    nn.Sigmoid()(depth_img)
                    * (self.max_z - self.min_z)
                    + self.min_z
                )
            else:
                # Use the inverse for datasets with landscapes, where there
                # is a long tail on the depth distribution
                depth_after_process = 1. / (nn.Sigmoid()(depth_img) * 10 + 0.01)
        else:
            depth_after_process = depth_img

        gen_fs = self.pts_transformer.forward_justpts(
            fs,
            depth_after_process,
            K,
            K_inv,
            None, #input_RT, which is not used in point projection
            input_RTinv,
            input_RT, #output_RT which is equal to input_RT in my case
            None, #output_RT_inv which is not used
        )

        return gen_fs
import torch
from pytorch3d.renderer import PerspectiveCameras, camera_conversions

def opencv_project(points, opencv_camera):
    """
    Project 3D points with OpenCV pinhole camera.

    Args:
        points: Tensor of shape (P, 3) or (N, P, 3)
        opencv_camera: Tensor of shape (3, 3) or (N, 3, 3)
    Returns:
        new_points: projected 3D points with the same shape as the input
    """
    points_batch = points.clone()
    if points_batch.dim() != 2 and points_batch.dim() != 3 and points_batch.shape[-1] != 3:
        msg = "Expected points to have shape (P, 3) or (N, P, 3): got shape %r"
        raise ValueError(msg % repr(points.shape))

    if points_batch.dim() == 2:
        opencv_pix = opencv_camera.mm(points_batch.transpose(0, 1))  # (3,3)*(3,P) -> (3,P)
        opencv_pix = opencv_pix.transpose(0, 1)  # (3,P) -> (P,3)
    if points_batch.dim() == 3:
        N, P, _3 = points_batch.shape
        if opencv_camera.dim() == 2:
            opencv_camera = torch.cat(N*[opencv_camera[None]])
        opencv_pix = opencv_camera.bmm(points_batch.transpose(1, 2))  # (N,3,3)*(N,3,P) -> (N,3,P)
        opencv_pix = opencv_pix.transpose(1, 2)  # (N,3,P) -> (N,P,3)
    opencv_pix[..., :2] = opencv_pix[..., :2] / opencv_pix[..., 2:]  # divide z
    opencv_pix[..., 2:] = 1.0 / opencv_pix[..., 2:]  # keep the same as Pytorch3D
    return opencv_pix

def points_projection(points,W,H,f=550):
    # input opencv coordinates points N,3
    # return screen coordinates H,W,Z

    _R = torch.eye(3)[None]  # (1, 3, 3)
    _T = torch.zeros(1, 3)  # (1, 3)
    cx, cy = 0.5 * W, 0.5 * H
    cam_mat = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]])  # (3, 3)

    # points = torch.tensor([[2.0, 1.0, 3.0], [1.2, 3.2, 1.5]])  # (2, 3)
    # points = torch.rand((2, 2, 3))
    # print('Input 3D points\n', points.shape)

    # ocvcam_pix = opencv_project(points, cam_mat)
    # print('\nOpenCV projection\n', ocvcam_pix.shape, '\n', ocvcam_pix)


    #open-cv coordinates
    p3docv_cam = camera_conversions._cameras_from_opencv_projection(_R, _T, cam_mat[None], torch.tensor([[W, H]]))
    p3docv_pix = p3docv_cam.transform_points_screen(points)

    #p3d coordinates
    p3dpsp_cam = PerspectiveCameras(focal_length=f, principal_point=((cx, cy),), image_size=((W, H),), in_ndc=False)
    points_flipxy = points.clone()
    points_flipxy[..., 0:2] = -points_flipxy[...,
                               0:2]  # Convert points from OpenCV (+X right, +Y down) to Pytorch3D (+X left, +Y up) camera space
    p3dpsp_pix = p3dpsp_cam.transform_points_screen(points_flipxy)

    return "good"







