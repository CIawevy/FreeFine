import numpy as np
import time
import cv2
import torch
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    PointsRasterizationSettings, PointsRenderer, PointsRasterizer,
    AlphaCompositor, NormWeightedCompositor, FoVPerspectiveCameras
)
from pytorch3d.renderer.points import rasterize_points
from pytorch3d.transforms import Transform3d, Rotate, Translate, Scale, euler_angles_to_matrix
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.renderer import compositing
from pytorch3d.renderer import PerspectiveCameras, camera_conversions
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from queue import PriorityQueue

import torch


def calculate_cosine_similarity_between_batches(features, has_reference_image=False, cos_threshold=0.8):
    batch, channels, height, width = features.size()

    # 将特征展平为 (batch, L, C)
    features_flat = features.view(batch, channels, height * width).transpose(1, 2)

    # 归一化特征向量
    norms = features_flat.norm(dim=2, keepdim=True) + 1e-8  # 防止除以零
    normalized_features = features_flat / norms

    # 编辑图像特征和参考图像特征
    edit_features = normalized_features[0]
    ref_features = normalized_features[1]

    # 计算编辑图像每个位置与参考图像每个位置的余弦相似度
    cosine_similarity = torch.matmul(edit_features, ref_features.transpose(0, 1))  # (height * width, height * width)

    # 处理相似度阈值
    max_sim, max_indices = torch.max(cosine_similarity, dim=1)

    # 初始化输出张量
    # final_max_indices = torch.zeros((batch, height, width, 4), dtype=torch.long)
    final_max_indices = torch.empty((height, width, 2), dtype=torch.long).to(features.device)

    for h in range(height):
        for w in range(width):
            hw = h * width + w
            max_ref_hw = max_indices[hw]
            max_ref_h = max_ref_hw // width
            max_ref_w = max_ref_hw % width
            max_sim_value = max_sim[hw]

            # 应用相似度阈值
            if max_sim_value < cos_threshold:
                max_ref_h = -1
                max_ref_w = -1
                scaled_cos_sim = -1
            else:
                scaled_cos_sim = max_sim_value

            final_max_indices[h, w] = torch.tensor([max_ref_h, max_ref_w], dtype=features.dtype)
            # final_max_indices[h, w] = torch.tensor([max_ref_h, max_ref_w, scaled_cos_sim], dtype=features.dtype)
            # final_max_indices[b, h, w] = torch.tensor([max_b, max_h, max_w, cos_sim * 1000])

    return final_max_indices




def calculate_cosine_similarity_between_batches_ori(features):
    batch, channels, height, width = features.size()
    features = features.view(batch, channels, -1).transpose(1, 2).view(batch, -1, channels) #B,L,C
    norms = features.norm(dim=2, keepdim=True)
    normalized_features = features / norms

    normalized_features_expanded = normalized_features.unsqueeze(0)
    normalized_features_tiled = normalized_features.unsqueeze(1)

    # すべてのバッチペアの間でコサイン類似度を計算します。
    cosine_similarity = torch.matmul(
        normalized_features_expanded, normalized_features_tiled.transpose(2, 3)
    ).squeeze()

    if args.reference_image is not None:
        # リファレンスイメージ以外との類似度を除去します。
        batch_indices = torch.arange(batch).view(batch, 1, 1)
        cosine_similarity[batch_indices, batch_indices, :] = -1
        cosine_similarity[1:batch, 1:batch, :] = -1
    else:
        # 自身との類似度を除去します。
        batch_indices = torch.arange(batch).view(batch, 1, 1)
        cosine_similarity[batch_indices, batch_indices, :] = -1

    print("cosine_similarity shape", cosine_similarity.shape)
    # 類似度が閾値以上の要素のインデックスを取得します。
    max_sim, max_indices = torch.max(cosine_similarity, dim=1)
    print("max_sim, max_indices shape 1", max_sim.shape, max_indices.shape)
    print("max_sim, max_indices", max_sim, max_indices)

    max_sim2, max_indices2 = torch.max(max_sim, dim=1)
    print("max_sim2, max_indices2 shape 2 ", max_sim2.shape, max_indices2.shape)
    print("max_sim2, max_indices2", max_sim2, max_indices2)

    # max_indices : (batch, width * height, width * height)
    # max_indices2 : (batch, width * height)

    # final_max_indeces : (batch, height, width, 3) from max_indices and max_indices2
    final_max_indices = torch.zeros((batch, height, width, 4), dtype=torch.long)

    for b in range(batch):
        for hw in range(height * width):

            h = hw // width
            w = hw % width
            max_hw = max_indices2[b, hw]
            max_h = max_hw // width
            max_w = max_hw % width
            max_b = max_indices[b, hw, max_hw]

            cos_sim = max_sim[max_b, max_h, max_w]

            if cos_sim < cos_threshold:
                max_b = -1
                max_h = -1
                max_w = -1

            if args.reference_image is not None and b == 0:
                max_b = -1
                max_h = -1
                max_w = -1

            final_max_indices[b, h, w] = torch.tensor([max_b, max_h, max_w, cos_sim * 1000])

    # max_indices_b = max_indices[
    # max_sim, max_indices = torch.max(max_sim, dim=1)
    # print("max_sim, max_indices shape 3", max_sim.shape, max_indices.shape)
    # print("max_sim, max_indices",  max_sim, max_indices)

    # 閾値以上のものだけを残します。
    # max_indices[max_sim < cos_threshold] = -1  # 閾値未満のインデックスは-1に設定

    # バッチ内のインデックスを高さと幅のインデックスに変換します。
    # max_indices_batch = max_indices
    # max_indices_h = max_indices2 // width
    # max_indices_w = max_indices2 % width
    # print("max_indices_batch.shape, max_indices_h.shape, max_indices_w.shape", max_indices_batch.shape, max_indices_h.shape, max_indices_w.shape)

    # バッチインデックスも含めた結果を返します。
    # max_indices_batch = max_indices #  // (height * width)
    # max_indices_h = max_indices_batch // width
    # max_indices_w = max_indices_batch % width

    print("final_max_indices", final_max_indices)
    print("final_max_indices shape", final_max_indices.shape)
    return final_max_indices
def tensor_inpaint_fmm(tensor_image, tensor_mask):
    """
    使用快速行进法对Tensor图像进行修复。

    :param tensor_image: 输入图像的Tensor，形状为(B, C, H, W)
    :param tensor_mask: 修复区域掩码的Tensor，形状为(H, W)
    :return: 修复后的图像Tensor，形状为(B, C, H, W)
    """
    assert len(tensor_image.shape) == 4, "tensor_image的形状应为(B, C, H, W)"
    assert len(tensor_mask.shape) == 2, "tensor_mask的形状应为(H, W)"

    B, C, H, W = tensor_image.shape

    # 初始化修复区域
    inpainted_image = tensor_image.clone()
    mask = tensor_mask.clone().bool()

    # 获取修复区域的坐标
    inpaint_coords = torch.nonzero(mask, as_tuple=False)

    # 定义邻域
    neighbors = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=tensor_image.device)

    # 创建优先级队列
    pq = PriorityQueue()

    # 初始化优先级队列，计算边界像素的优先级
    for y, x in inpaint_coords:
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not mask[ny, nx]:
                distance = torch.sqrt(dy.float() ** 2 + dx.float() ** 2)
                pq.put((distance.item(), (y.item(), x.item())))
                break

    # 使用快速行进法填充修复区域
    while not pq.empty():
        _, (y, x) = pq.get()
        if not mask[y, x]:
            continue

        # 获取邻域像素的值
        values = []
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not mask[ny, nx]:
                values.append(inpainted_image[:, :, ny, nx])

        if values:
            # 计算平均值进行填充
            values = torch.stack(values, dim=0)
            inpainted_image[:, :, y, x] = values.mean(dim=0)
            mask[y, x] = False

            # 将新填充的像素加入优先级队列
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                    distance = torch.sqrt(dy.float() ** 2 + dx.float() ** 2)
                    pq.put((distance.item(), (ny.item(), nx.item())))

    return inpainted_image


class PartialConvInterpolation(nn.Module):
    def __init__(self, kernel_size, channels, feature_weight):
        super(PartialConvInterpolation, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels

        # 手动输入的特征卷积权重
        self.feature_weight = nn.Parameter(
            torch.tensor(feature_weight, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1),
            requires_grad=False)

        # 掩码卷积的权重为全1
        self.mask_weight = nn.Parameter(torch.ones(self.channels, 1, kernel_size, kernel_size), requires_grad=False)

    def temp_view(self, mask, title='Mask', name=None):
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
        # plt.savefig(name+'.png')
        plt.show()

    def forward(self, input, mask,ref_mask, max_iterations=20):
        input_masked = input * mask
        input_conv = input_masked #init
        ref_mask = (ref_mask>0).float()
        for _ in range(max_iterations):

            input_conv = F.conv2d(input_conv, self.feature_weight, padding=self.kernel_size // 2,
                                  groups=self.channels)
            mask_conv = F.conv2d(mask, self.mask_weight, padding=self.kernel_size // 2, groups=self.channels)
            mask_sum = mask_conv.masked_fill(mask_conv == 0, 1.0)
            output = input_conv / mask_sum

            new_mask = (mask_conv > 0).float()
            stat = ((new_mask)[0,0] * ref_mask).sum() == ref_mask.sum()
            if stat or new_mask.equal(mask):
                break
            mask = new_mask

        return output

def param2theta(param, w, h):
    param = np.concatenate([param, np.array([0, 0, 1], dtype=param.dtype)[None]])  # for求逆
    param = np.linalg.inv(param)
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta

def wrapAffine_tensor( tensor, theta, dsize, mode='bilinear', padding_mode='zeros', align_corners=False,
                      border_value=0):
    """
    对给定的张量进行仿射变换，仿照 cv2.warpAffine 的功能。

    参数：
    - tensor: 要变换的张量，形状为 (C, H, W) 或 (H, W) 或(N,C,H,W)
    - theta: 2x3 变换矩阵，形状为 (2, 3)
    - dsize: 输出尺寸 (width, height)
    - mode: 插值方法，可选 'bilinear', 'nearest', 'bicubic'
    - padding_mode: 边界填充模式，可选 'zeros', 'border', 'reflection'
    - align_corners: 是否对齐角点，默认为 False
    - border_value: 填充值

    返回：
    - transformed_tensor: 变换后的张量，形状与输入张量相同
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 4:
        pass

    # 生成变换的 grid
    grid = F.affine_grid(theta.unsqueeze(0), [tensor.size(0), tensor.size(1), dsize[1], dsize[0]],
                         align_corners=align_corners)

    # 进行 grid 采样
    output = F.grid_sample(tensor, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    transformed_tensor = output.squeeze(0)

    # 使用填充值填充边界
    if padding_mode == 'zeros' and border_value != 0:
        mask = (grid.abs() > 1).any(dim=-1, keepdim=True)
        transformed_tensor[mask] = border_value

    return transformed_tensor
def get_transformation(tx, ty, tz, rx, ry, rz, sx, sy, sz, device):
    """
    构建PyTorch3D的变换对象，包括平移、旋转和缩放。

    参数:
    tx, ty, tz: float
        分别沿x, y, z轴的平移量。
    rx, ry, rz: float
        分别绕x, y, z轴的旋转角度（以度为单位）。
    sx, sy, sz: float
        分别沿x, y, z轴的缩放比例。
    device: torch.device
        PyTorch计算设备（CPU或GPU）。

    返回:
    Transform3d
        PyTorch3D的变换对象。
    """
    # 创建平移对象
    translation = Translate(tx, ty, tz, device=device)

    # 创建旋转对象，使用欧拉角转换为旋转矩阵
    # 将欧拉角从度转换为弧度
    rx, ry, rz = torch.deg2rad(torch.tensor([rx, ry, rz], device=device))
    rotation_matrix = euler_angles_to_matrix(torch.tensor([rx, ry, rz], device=device), convention="XYZ")
    rotation = Rotate(R=rotation_matrix, device=device)

    # 创建缩放对象
    scaling = Scale(sx, sy, sz, device=device)

    # 组合变换对象
    transform = translation.compose(rotation).compose(scaling)

    return transform


def apply_transformation(point_cloud, transform):
    """
    将变换对象应用于点云。

    参数:
    point_cloud: Pointclouds
        要变换的点云。
    transform: Transform3d
        PyTorch3D的变换对象。

    返回:
    Pointclouds
        变换后的点云。
    """
    # 对点云应用变换
    transformed_points = transform.transform_points(point_cloud.points_packed())

    point_cloud = point_cloud.update_padded(transformed_points.unsqueeze(0))
    # # 创建新的点云对象
    # transformed_point_cloud = Pointclouds(points=[transformed_points],
    #                                       features=[point_cloud.features_packed()])

    return point_cloud

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


def transform_point_cloud(point_cloud,transforms,device):
    #relative tx -> absolute tx
    points_3d = point_cloud.points_packed()
    transforms = refine_transforms(transforms, points_3d)
    # build
    transform = get_transformation(*transforms, device)
    # apply
    transformed_point_cloud = apply_transformation(point_cloud, transform)

    return transformed_point_cloud
def IntegratedP3DTransRasterBlendingFull(img, depth, transforms, focal_length_x, focal_length_y, mask, object_only=True,
                                     splatting_radius=0.1, splatting_tau=0.1, splatting_points_per_pixel=5,return_mask=True,device=None,
                                        ):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        h, w = depth.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        i, j = torch.tensor(i, device=device), torch.tensor(j, device=device)
        z = torch.tensor(depth, device=device)
        x = (i - w / 2) * z / focal_length_x
        y = (j - h / 2) * z / focal_length_y
        znear, zfar = z.min(), z.max()
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)

        rgb = torch.tensor(img, device=device, dtype=torch.float32).reshape(-1, 3)
        mask = torch.tensor(mask, device=device).reshape(-1)

        if object_only:
            points = points[mask > 0]
            rgb = rgb[mask > 0]


        # open-cv world -> pytorch3d world
        points[:, :2] = - points[:, :2]
        #get cameta coordinates params


        # Translate to object center as world coordinates
        center_shift = points.mean(dim=0)
        points -= center_shift

        # pytorch3d coords point cloud
        point_cloud = Pointclouds(points=[points], features=[rgb])
        # Transform in world coordinates
        point_cloud = transform_point_cloud(point_cloud, transforms, device)
        #开始计时
        start_time = time.time()
        


        # construct my R T based on center_shifting
        # R, T = look_at_view_transform(20, 0, 0)
        # Construct R and T based on obj_center_shift
        R = torch.eye(3, device=device)[None]  # Identity matrix for rotation
        T = center_shift.view(1, 3)  # Translation vector
        # 开始计时

        # cameras =FoVOrthographicCameras(device=device, R=R, T=T, znear=znear, zfar=zfar,max_y=ymax,min_y=ymin,max_x=xmax,min_x=xmin)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear,zfar=zfar,)
        raster_settings = PointsRasterizationSettings(
            image_size=(h, w),
            radius=float(splatting_radius),
            points_per_pixel=splatting_points_per_pixel,
            bin_size=None, #0 naive, None:heuristic
            max_points_per_bin = 100000 #default 10000
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        compositor = AlphaCompositor(background_color=(0, 0, 0))
        if return_mask:
            fragments = rasterizer(point_cloud, )

            # Construct weights based on the distance of a point to the true point.
            # However, this could be done differently: e.g. predicted as opposed
            # to a function of the weights.
            r = rasterizer.raster_settings.radius

            dists2 = fragments.dists.permute(0, 3, 1, 2)
            weights = 1 - dists2 / (r * r)
            images = compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                weights,
                point_cloud.features_packed().permute(1, 0),
            )
            # permute so image comes at the end
            images = images.permute(0, 2, 3, 1)
            #TODO: fetch_mask from fragment
            # 结束计时
            end_time = time.time()
            # 计算执行时间
            elapsed_time = end_time - start_time
            print(f"object-only:{object_only} \n 代码执行时间: {elapsed_time} 秒")
            rendered_image = images[0, ..., :3].cpu().numpy().astype(np.uint8)
            rendered_mask = torch.any((fragments.idx[0][:,:].sum(dim=-1,keepdim=True)!=-30),dim=-1)
            rendered_mask = rendered_mask.cpu().numpy().astype(np.uint8)*255
            return rendered_image , rendered_mask
        else:
            renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
            images = renderer(point_cloud)


            # 结束计时
            end_time = time.time()
            # 计算执行时间
            elapsed_time = end_time - start_time
            print(f"object-only:{object_only} \n 代码执行时间: {elapsed_time} 秒")
            rendered_image = images[0, ..., :3].cpu().numpy().astype(np.uint8)

            return rendered_image

def IntegratedP3DTransRasterBlending(img, depth, transforms, focal_length_x, focal_length_y, mask, object_only=True,
                                     splatting_radius=0.1, splatting_tau=0.1, splatting_points_per_pixel=5,return_mask=True,device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        h, w = depth.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        i, j = torch.tensor(i, device=device), torch.tensor(j, device=device)
        z = torch.tensor(depth, device=device)
        x = (i - w / 2) * z / focal_length_x
        y = (j - h / 2) * z / focal_length_y
        znear, zfar = z.min(), z.max()
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)

        rgb = torch.tensor(img, device=device, dtype=torch.float32).reshape(-1, 3)
        mask = torch.tensor(mask, device=device).reshape(-1)

        if object_only:
            points = points[mask > 0]
            rgb = rgb[mask > 0]


        # open-cv world -> pytorch3d world
        points[:, :2] = - points[:, :2]
        #get cameta coordinates params


        # Translate to object center as world coordinates
        center_shift = points.mean(dim=0)
        points -= center_shift

        # pytorch3d coords point cloud
        point_cloud = Pointclouds(points=[points], features=[rgb])
        # Transform in world coordinates
        point_cloud = transform_point_cloud(point_cloud, transforms, device)
        #开始计时
        start_time = time.time()
        diy = False
        if diy:
            #Transform to NDC bugs here
            trans_points = point_cloud.points_packed()
            trans_points += center_shift
            trans_points[:,:2] = points[:,:2] / points[:,2,None]
            ndc_point_clouds = point_cloud.update_padded(trans_points.unsqueeze(0))
            raster_settings = PointsRasterizationSettings(
                image_size=(h, w),
                radius=float(splatting_radius),
                points_per_pixel=splatting_points_per_pixel)
            idx, zbuf, dists2 = rasterize_points(
                ndc_point_clouds,
                image_size=raster_settings.image_size,
                radius=raster_settings.radius,
                points_per_pixel=raster_settings.points_per_pixel,
                bin_size=raster_settings.bin_size,
                max_points_per_bin=raster_settings.max_points_per_bin,
            )

            # Construct weights based on the distance of a point to the true point.
            # However, this could be done differently: e.g. predicted as opposed
            # to a function of the weights.
            r = splatting_radius

            # dists2 = dists2.permute(0, 3, 1, 2)

            dist = 1 - dists2 / (r * r)
            weights = (
                (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
                .pow(splatting_tau)
                .permute(0, 3, 1, 2)
            )

            # compositor = AlphaCompositor(background_color=(0, 0, 0))
            # images =compositor(
            #     idx.long().permute(0, 3, 1, 2),
            #     weights,
            #     ndc_point_clouds .features_packed().permute(1, 0),
            # )
            accumulation = 'alphacomposite'
            # accumulation =  'wsum'


            if accumulation == 'alphacomposite':
                images = compositing.alpha_composite(
                    idx.long().permute(0, 3, 1, 2),
                    weights,
                    ndc_point_clouds.features_packed().permute(1, 0),
                )
            elif accumulation == 'wsum':
                images = compositing.weighted_sum(
                    idx.long().permute(0, 3, 1, 2),
                    weights,
                    ndc_point_clouds.features_packed().permute(1, 0),
                )

            # permute so image comes at the end
            images = images.permute(0, 2, 3, 1)
            rendered_image = images[0, ..., :3].cpu().numpy().astype(np.uint8)

            return rendered_image , rendered_image

        else:
            # construct my R T based on center_shifting
            # R, T = look_at_view_transform(20, 0, 0)
            # Construct R and T based on obj_center_shift
            R = torch.eye(3, device=device)[None]  # Identity matrix for rotation
            T = center_shift.view(1, 3)  # Translation vector
            # 开始计时

            # cameras =FoVOrthographicCameras(device=device, R=R, T=T, znear=znear, zfar=zfar,max_y=ymax,min_y=ymin,max_x=xmax,min_x=xmin)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear,zfar=zfar,)
            raster_settings = PointsRasterizationSettings(
                image_size=(h, w),
                radius=float(splatting_radius),
                points_per_pixel=splatting_points_per_pixel,
                bin_size=None, #0 naive, None:heuristic
                max_points_per_bin = 100000 #default 10000
            )
            rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
            compositor = AlphaCompositor(background_color=(0, 0, 0))
            if return_mask:
                fragments = rasterizer(point_cloud, )

                # Construct weights based on the distance of a point to the true point.
                # However, this could be done differently: e.g. predicted as opposed
                # to a function of the weights.
                r = rasterizer.raster_settings.radius

                dists2 = fragments.dists.permute(0, 3, 1, 2)
                weights = 1 - dists2 / (r * r)
                images = compositor(
                    fragments.idx.long().permute(0, 3, 1, 2),
                    weights,
                    point_cloud.features_packed().permute(1, 0),
                )
                # permute so image comes at the end
                images = images.permute(0, 2, 3, 1)
                #TODO: fetch_mask from fragment
                # 结束计时
                end_time = time.time()
                # 计算执行时间
                elapsed_time = end_time - start_time
                print(f"object-only:{object_only} \n 代码执行时间: {elapsed_time} 秒")
                rendered_image = images[0, ..., :3].cpu().numpy().astype(np.uint8)
                rendered_mask = torch.any((fragments.idx[0][:,:].sum(dim=-1,keepdim=True)!=-30),dim=-1)
                rendered_mask = rendered_mask.cpu().numpy().astype(np.uint8)*255
                return rendered_image , rendered_mask
            else:
                renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
                images = renderer(point_cloud)


                # 结束计时
                end_time = time.time()
                # 计算执行时间
                elapsed_time = end_time - start_time
                print(f"object-only:{object_only} \n 代码执行时间: {elapsed_time} 秒")
                rendered_image = images[0, ..., :3].cpu().numpy().astype(np.uint8)

                return rendered_image



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 创建一个简单的点云（只包含一个点）
    points = torch.tensor([[[0, 0, 1]]], dtype=torch.float32)  # 中心点
    colors = torch.tensor([[[1, 0, 0]]], dtype=torch.float32)  # 红色
    point_cloud = Pointclouds(points=points, features=colors)

    # 渲染设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cameras = FoVPerspectiveCameras(device=device)
    raster_settings = PointsRasterizationSettings(
        image_size=(512, 512),  # 指定渲染图像大小
        radius=0.01,
        points_per_pixel=1
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = NormWeightedCompositor()
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    # 渲染
    images = renderer(point_cloud.to(device))

    # 转换为numpy并显示
    rendered_image = images[0, ..., :3].cpu().numpy()
    plt.imshow(rendered_image)
    plt.title('Rendered Point Cloud')
    plt.show()




# while not fully transformed:
#     if 2D:
#         #move/rotate/resize some degree
#     elif 3D:
#         #1.calculate partial 3D transform matrix from fully transformed matrix
#         #2.apply and transform some degree
#         #3.reproject to 2D , also change some degree
#         #4. optimization and complete small unseen parts #!!!!TODO: Need Design , how to use prior here
#         #5. grad backward to bottleneck feature
#         #5 forward again

