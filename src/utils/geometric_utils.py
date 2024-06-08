import numpy as np
import cv2
import time


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


    return new_color_image, dilation_mask, new_depth_image,inpaint_mask


