import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


import numpy as np

def create_ellipse_boundary_mask(width, height, xc, yc, rmajor, rminor, theta, boundary_thickness=0.1):
    """
    生成椭圆的边界二值掩码，边界区域为1，其他区域为0
    参数：
        width: 图像宽度
        height: 图像高度
        xc, yc: 椭圆的中心坐标
        rmajor, rminor: 椭圆的长短轴半径
        theta: 椭圆的旋转角度（以度为单位）
        boundary_thickness: 椭圆边界的厚度（默认1）
    返回：
        二值边界掩码
    """
    # 创建全零的图像
    mask = np.zeros((height, width), dtype=np.uint8)

    # 网格坐标
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)

    # 转换角度为弧度
    theta_rad = np.deg2rad(theta)

    # 旋转坐标系，将 (X, Y) 平移到 (xc, yc) 再旋转
    X_shifted = X - xc
    Y_shifted = Y - yc
    X_rot = X_shifted * np.cos(theta_rad) + Y_shifted * np.sin(theta_rad)
    Y_rot = -X_shifted * np.sin(theta_rad) + Y_shifted * np.cos(theta_rad)

    # 计算椭圆的标准方程： (X_rot^2 / rmajor^2) + (Y_rot^2 / rminor^2) ≈ 1
    # 使用容差来判断像素是否在椭圆的边界附近
    boundary_condition = np.abs((X_rot ** 2 / rmajor ** 2 + Y_rot ** 2 / rminor ** 2) - 1) < boundary_thickness

    mask[boundary_condition] = 1  # 设置边界处为1，其他区域为0

    return mask



def compute_distance_to_ellipse_boundary(width, height, xc, yc, rmajor, rminor, theta, boundary_thickness=0.1):
    """
    计算图像中每个像素到椭圆边界的最短距离
    """
    # 生成椭圆边界掩码
    ellipse_boundary_mask = create_ellipse_boundary_mask(width, height, xc, yc, rmajor, rminor, theta,
                                                         boundary_thickness)

    # 使用距离变换计算每个像素到椭圆边界的最短距离
    # 计算到椭圆边界的距离：非椭圆边界区域到椭圆边界的距离
    dist_transform = distance_transform_edt(1 - ellipse_boundary_mask)

    return dist_transform, ellipse_boundary_mask


# 测试：生成椭圆边界距离变换图
width = 200
height = 200
xc, yc = 100, 100  # 椭圆中心
rmajor, rminor = 60, 40  # 长短轴
theta = 0  # 旋转角度

# 计算距离变换
dist_transform, ellipse_boundary_mask = compute_distance_to_ellipse_boundary(width, height, xc, yc, rmajor, rminor,
                                                                             theta)

# 可视化：显示椭圆边界掩码
plt.figure(figsize=(10, 5))

# 显示椭圆边界掩码
plt.subplot(1, 2, 1)
plt.imshow(ellipse_boundary_mask, cmap='gray')
plt.title("Ellipse Boundary Mask")
plt.colorbar()

# 显示距离变换结果
plt.subplot(1, 2, 2)
plt.imshow(dist_transform, cmap='hot')
plt.colorbar()
plt.title("Distance Transform to Ellipse Boundary")

plt.show()

# 验证特殊点
print("验证特殊点:")
# 椭圆中心（xc, yc）的距离应该为半短轴
center_distance = dist_transform[int(yc), int(xc)]
print(f"椭圆中心到边界的距离：{center_distance:.2f} (应该为半短轴 rminor={rminor})")

# 椭圆边界上的一点（例如，(xc + rmajor, yc)）的距离应该接近 0
boundary_point_distance = dist_transform[int(yc), int(xc + rmajor)]
print(f"椭圆上的点到边界的距离：{boundary_point_distance:.2f} (应该为接近0)")

# 椭圆边界上的一点（例如，(xc, yc + rminor)）的距离应该接近 0
boundary_point_distance_2 = dist_transform[int(yc + rminor), int(xc)]
print(f"椭圆上的点到边界的距离：{boundary_point_distance_2:.2f} (应该为接近0)")