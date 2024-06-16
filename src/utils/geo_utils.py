import numpy as np
import time
import cv2
import torch
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
