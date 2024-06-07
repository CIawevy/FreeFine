import numpy as np
import cv2

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


def warpAffine3D(img, depth,transforms, focal_length_x, focal_length_y, mask):
    """
    Clawer made fantastic 3D transformation function
    build upon nothing
    acknowledgements: supported by GPT4-turbo
    """
    channels = img.shape[-1]
    FINAL_WIDTH = img.shape[1]
    FINAL_HEIGHT = img.shape[0]

    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))

    # Normalize pixel coordinates
    x_normalized = (x - FINAL_WIDTH / 2) / focal_length_x
    y_normalized = (y - FINAL_HEIGHT / 2) / focal_length_y
    z = np.array(depth)
    ones = np.ones_like(z)

    # Stack coordinates and depth to create 3D points
    points_3d = np.stack((np.multiply(x_normalized, z), np.multiply(y_normalized, z), z, ones), axis=-1).reshape(-1, 4)

    # Convert mask to 3D points and calculate the 3D center
    mask_indices = np.where(mask > 128)
    mask_depth = z[mask_indices]
    mask_x = x_normalized[mask_indices] * mask_depth
    mask_y = y_normalized[mask_indices] * mask_depth
    center_3d = np.array([np.mean(mask_x), np.mean(mask_y), np.mean(mask_depth), 0.0])

    # Translate to object center
    points_3d -= center_3d

    transforms = refine_transforms(transforms,points_3d)
    T = get_transformation_matrix(*transforms)

    # Apply 3D transformation matrix
    transformed_points_3d = (points_3d @ T.T)[:, :3]

    # Translate back from object center
    transformed_points_3d += center_3d[:3]

    # Reproject 3D coordinates back to 2D
    projected_x = (transformed_points_3d[:, 0] * focal_length_x / transformed_points_3d[:, 2]) + FINAL_WIDTH / 2
    projected_y = (transformed_points_3d[:, 1] * focal_length_y / transformed_points_3d[:, 2]) + FINAL_HEIGHT / 2

    # Reshape projected coordinates to match image shape
    projected_x = projected_x.reshape((FINAL_HEIGHT, FINAL_WIDTH)).astype(np.float32)
    projected_y = projected_y.reshape((FINAL_HEIGHT, FINAL_WIDTH)).astype(np.float32)

    # Use remap function from OpenCV for interpolation
    new_depth_image = cv2.remap(depth, projected_x, projected_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if channels == 3:  # image
        new_color_image = cv2.remap(img, projected_x, projected_y, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif channels == 1:  # mask
        new_color_image = cv2.remap(img, projected_x, projected_y, interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return new_color_image, new_depth_image