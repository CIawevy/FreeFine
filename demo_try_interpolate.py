from src.demo.download import download_all
# download_all()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
from simple_lama_inpainting import SimpleLama
from src.demo.demo import create_my_demo,create_my_demo_full_2D,create_my_demo_full_3D,create_my_demo_full_2D_ctn
from src.demo.model import ClawerModels,ClawerModel_v2
from src.unet.unet_2d_condition import DragonUNet2DConditionModel
from src.utils.geo_utils import IntegratedP3DTransRasterBlendingFull,param2theta,wrapAffine_tensor,PartialConvInterpolation,tensor_inpaint_fmm
import torch
import cv2
import torch
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
import gradio as gr
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything_v2.dpt import DepthAnythingV2
from torchvision.transforms import Compose
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline
# main demo
# pretrained_model_path = "runwayml/stable-diffusion-v1-5"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

    def forward(self, input, mask, max_iterations=20):
        input_masked = input * mask
        input_conv = input_masked #init
        for _ in range(max_iterations):

            input_conv = F.conv2d(input_conv, self.feature_weight, padding=self.kernel_size // 2,
                                  groups=self.channels)
            mask_conv = F.conv2d(mask, self.mask_weight, padding=self.kernel_size // 2, groups=self.channels)
            mask_sum = mask_conv.masked_fill(mask_conv == 0, 1.0)
            output = input_conv / mask_sum

            new_mask = (mask_conv > 0).float()
            if new_mask.equal(mask):
                break
            mask = new_mask

        return output
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
@torch.no_grad()
def get_mask_center(mask):
    y_indices, x_indices = torch.where(mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        top, bottom = torch.min(y_indices), torch.max(y_indices)
        left, right = torch.min(x_indices), torch.max(x_indices)
        mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2
    return mask_center_x.item(), mask_center_y.item()

def visualize_rgb_image(image: Image.Image, title: str = None) -> None:
    """
    Visualize an RGB image from a PIL Image format with an optional title.

    Parameters:
    image (PIL.Image.Image): The RGB image represented as a PIL Image.
    title (str, optional): The title to display above the image.

    Raises:
    ValueError: If the input is not a PIL Image or is not in RGB mode.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    if image.mode != 'RGB':
        raise ValueError("Input image must be in RGB mode.")

    image_array = np.array(image)

    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()
# Dummy wrapAffine_tensor function for demonstration
# Manual interpolation function
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




def create_feature_weight(kernel_size, min_w=0.5, max_w=1.0):
    # center = kernel_size // 2
    # kernel = torch.zeros((kernel_size, kernel_size))
    # for i in range(kernel_size):
    #     for j in range(kernel_size):
    #         distance = max(abs(i - center), abs(j - center)) / center
    #         weight = max_w - (max_w - min_w) * distance
    #         kernel[i, j] = weight
    # kernel /= kernel.sum()
    kernel = torch.ones((kernel_size, kernel_size))
    return kernel
def edit_init_code(init_code, theta, current_mask, next_mask):
    batch_size, channels, height, width = init_code.shape

    current_mask = F.interpolate(current_mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode='nearest').squeeze(0).squeeze(0)
    next_mask = F.interpolate(next_mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode='nearest').squeeze(0).squeeze(0)

    moved_init_code = wrapAffine_tensor(init_code, theta, (width,height), mode='bilinear').unsqueeze(0)

    intersection_mask = (current_mask.bool() & next_mask.bool()).float()
    non_intersect_current_mask = (current_mask.bool() & ~intersection_mask.bool()).float()
    union_mask = (current_mask.bool() | next_mask.bool()).float()


    # kernel_size = 5
    # channels = 3
    # feature_weight = create_feature_weight(kernel_size)
    # # feature_weight = [[0.5, 0.8, 0.8, 0.8, 0.5],
    # #                   [0.8, 1.0, 1.0, 1.0, 0.8],
    # #                   [0.8, 1.0, 0.0, 1.0, 0.8],
    # #                   [0.8, 1.0, 1.0, 1.0, 0.8],
    # #                   [0.5, 0.8, 0.8, 0.8, 0.5]]
    #
    # partial_conv = PartialConvInterpolation(kernel_size, channels, feature_weight).to(device)
    # inpaint_mask = current_mask.unsqueeze(0).repeat(channels, 1, 1)  # 为指定通道重复掩码
    # interpolated_original = partial_conv(init_code, 1 - inpaint_mask,10) #1 means valid ,0 means to be repaired
    interpolated_original = tensor_inpaint_fmm(init_code,union_mask)

    next_mask = next_mask.unsqueeze(0).repeat(channels, 1, 1).unsqueeze(0)
    non_intersect_current_mask = non_intersect_current_mask.unsqueeze(0).repeat(channels, 1, 1).unsqueeze(0)
    result_code_0 = torch.where(next_mask>0,moved_init_code,init_code)
    result_code = torch.where(non_intersect_current_mask>0,interpolated_original,result_code_0)

    return result_code,interpolated_original
# Load and preprocess image and masks
def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)
def plot_tensor_image(tensor, title):
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_mask(mask, title):
    mask = mask.cpu().numpy()
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def resize_numpy_image(image, max_resolution=768 * 768, resize_short_edge=None,mask_input=False):
    h, w = image.shape[:2]
    w_org = image.shape[1]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    if not mask_input:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
    scale = w/w_org
    return image, scale

# Main demonstration code
if __name__ == "__main__":
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 输入路径
    image_path = "/data/Hszhu/prompt-to-prompt/CPIG/1.jpg"
    mask_path = "/data/Hszhu/prompt-to-prompt/masks/1.png"

    # 加载图像和mask
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    init_code = load_image_as_tensor(image_path).to(device)
    batch_size, channels, height, width = init_code.shape

    motion_split_steps = 20
    # 获取移动的坐标
    selected_points = [[1095, 525], [701, 879]]
    x = [selected_points[0][0]]
    y = [selected_points[0][1]]
    x_cur = [selected_points[1][0]]
    y_cur = [selected_points[1][1]]

    dx = int(x_cur[0] - x[0]) / motion_split_steps
    dy = int(y_cur[0] - y[0]) / motion_split_steps
    resize_scale = 1.0 ** (1 / motion_split_steps)
    rotation_angle = 0  / motion_split_steps

    # 转换mask为tensor
    mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)

    # 获取mask中心
    mask_center_x, mask_center_y = get_mask_center(mask_tensor)

    # 获取变换矩阵
    transformation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle, resize_scale)
    transformation_matrix[0, 2] += dx
    transformation_matrix[1, 2] += dy

    # 转换为theta
    theta = torch.tensor(param2theta(transformation_matrix, width, height), dtype=init_code.dtype, device=device)

    # 进行仿射变换
    trajectory_current_mask = mask_tensor
    trajectory_next_mask = wrapAffine_tensor(trajectory_current_mask, theta, (width, height), mode='nearest')[0]

    # 编辑代码
    result_code,all_interpolated= edit_init_code(init_code, theta, trajectory_current_mask, trajectory_next_mask)

    # 显示图像
    plot_tensor_image(init_code, "Original Image")
    plot_tensor_image(all_interpolated, "interpolated Image")
    plot_tensor_image(result_code, "Edited Image")


