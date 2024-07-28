import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import os


def param2theta(param, w, h):
    param = np.concatenate([param, np.array([0, 0, 1], dtype=param.dtype)[None]])  # for求逆
    param = np.linalg.inv(param)
    theta = np.zeros([2,3])
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
    - tensor: 要变换的张量，形状为 (C, H, W) 或 (H, W)
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
    # plt.savefig(name+'.png')
    plt.show()



os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
prompt = 'car'
motion_split_steps=20
seed=42
selected_points = [[1000, 526], [615, 635]]
guidance_scale=3.5
num_step=50
max_resolution=512
dilate_kernel_size=30
start_step=15
mask_ref=None
eta=1
use_mask_expansion=0
standard_drawing=1
contrast_beta= 1.67
resize_scale=1.0
strong_inpaint=1
flip_horizontal=0
flip_vertical=0
cross_enhance=0
mask_threshold=0.1
mask_threshold_target=0.1
blending_alpha=0.7
# 示例用法
# image_path= "/data/Hszhu/prompt-to-prompt/CPIG/1.jpg"
mask_path = "/data/Hszhu/prompt-to-prompt/masks/1.png"
rotation_angle = 20  # 旋转30度
dx = 0  # 沿x轴平移10个单位
dy = 0 # 沿y轴平移5个单位
resize_scale = 1  # 缩放比例为1.5
mask = cv2.imread(mask_path)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask = (mask > 128).astype(np.uint8)
# 获取图像尺寸
temp_view(mask)
height, width = mask.shape[:2]
y_indices, x_indices = np.where(mask.astype(bool))
if len(y_indices) > 0 and len(x_indices) > 0:
    top, bottom = np.min(y_indices), np.max(y_indices)
    left, right = np.min(x_indices), np.max(x_indices)
    mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2
transformation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle,
                                                        resize_scale)
transformation_matrix[0, 2] += dx
transformation_matrix[1, 2] += dy
t_mask = cv2.warpAffine(mask, transformation_matrix, (width, height),
                                                   flags=cv2.INTER_NEAREST)
temp_view(t_mask)
# 将 cv2 的变换矩阵转换为 tensor 的变换矩阵
# theta = torch.tensor([
#     [transformation_matrix[0, 0], transformation_matrix[0, 1], -dx / width],
#     [transformation_matrix[1, 0], transformation_matrix[1, 1], -dy / height]
# ], dtype=torch.float32, device=device)
center = (mask_center_x, mask_center_y)
img_size = (height, width)

# 获取仿射变换矩阵

# transform mask and feat
mask = torch.tensor(mask,dtype=torch.float32,device=device)
theta = torch.tensor(param2theta(transformation_matrix,width,height),dtype=torch.float32,device=device)
# 进行仿射变换
trajectory_next_mask = wrapAffine_tensor(mask, theta, (width, height),mode='nearest')[0]
temp_view(trajectory_next_mask)

