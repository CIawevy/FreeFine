from src.utils.geometric_utils import Integrated3DTransformAndInpaint
from src.utils.geo_utils import IntegratedP3DTransRasterBlending
import os

import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import torch
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from depth_anything_v2.dpt import DepthAnythingV2





def plot_depth(depth_plot,title='title',norm=False):
    if norm:
        depth_plot = ((depth_plot-depth_plot.min())/(depth_plot.max()-depth_plot.min())*255).astype(np.uint8)
    grayscale = True
    if grayscale:
        depth_plot = np.repeat(depth_plot[..., np.newaxis], 3, axis=-1)
    else:
        depth_plot = cv2.applyColorMap(depth_plot, cv2.COLORMAP_INFERNO)
    plt.imshow(depth_plot)
    plt.title(title)
    plt.show()
    plt.close()
def view(img,title='image'):
    plt.imshow(img)
    plt.title(title)
    plt.show()
    plt.close()



if __name__ == "__main__":
    # Load model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


    # model_configs = {
    #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    # }
    #
    # encoder = 'vitl'  # or 'vitb', 'vits'
    # depth_anything = DepthAnything(model_configs[encoder])
    # depth_anything.load_state_dict(torch.load(f'/data/Hszhu/prompt-to-prompt/depth-anything/depth_anything_{encoder}14.pth'))
    # depth_anything.to(device).eval()
    # transform = Compose([
    #     Resize(
    #         width=518,
    #         height=518,
    #         resize_target=False,
    #         keep_aspect_ratio=True,
    #         ensure_multiple_of=14,
    #         resize_method='lower_bound',
    #         image_interpolation_method=cv2.INTER_CUBIC,
    #     ),
    #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     PrepareForNet(),
    # ])

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vits'  # or 'vits', 'vitb', 'vitg'

    depth_anything_v2 = DepthAnythingV2(**model_configs[encoder])
    depth_anything_v2.load_state_dict(torch.load(f'/data/Hszhu/prompt-to-prompt/depth-anything/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything_v2.to(device).eval()

    image_path = "/data/Hszhu/Reggio/examples/Expansion_Mask/CPIG/1.jpg"
    mask_path = "/data/Hszhu/Reggio/examples/Expansion_Mask/masks/1.png"
    # dst_path = "/data/Hszhu/DragonDiffusion/examples/3D-Trans"
    # Path(dst_path).mkdir(parents=True, exist_ok=True)
    # image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0
    color_image = Image.open(image_path).convert('RGB')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image_raw = np.array(color_image)
    # image_input = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)

    # plt.imshow(image_raw)
    # plt.title("RAW Image")
    # plt.show()
    h, w = image_raw.shape[:2]
    """
    #preprocess image for depth estimate
    image = image_raw / 255
    image = transform({'image': image})['image']

    # Reshape transformed image to (H, W, C) for plotting
    image_plot = np.transpose(image, (1, 2, 0))


    image = torch.from_numpy(image).unsqueeze(0).to(device)
    # Load image from image path to PIL image
    base_depth = 0.1
    max_z = 255
    min_z = 10
    EPISILON = 1e-8

    with torch.no_grad():
        depth = depth_anything(image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    """
    base_depth = 0.1
    max_z = 255
    min_z = 10
    depth = depth_anything_v2.infer_image(image_raw)#cv2 read RGB ndarray
    plot_depth(depth, 'RAW True depth',norm=True)
    process = 'Geo'
    #GeoDiffuser Processor
    # Regressed points
    if process=='Geo':
        depth = depth.max() - depth  # Negating depth as relative depth estimators assign high values to close objects. You can also try 1/depth (inverse depth, but we found this to work better prima facie)
        depth = depth + depth.max() * base_depth  # This helps in reducing depth smearing where translate_factor is between 0 to 1.
        # depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalizes from 0-1.
        #modified by clawer
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * (max_z - min_z) + min_z
    elif process =='for':
        depth = depth.max() - depth
        depth = depth + depth.max() * base_depth
        depth = (
                nn.Sigmoid()(depth)
                * (max_z - min_z)
                + min_z
        )
    elif process=='inv':
            # Use the inverse for datasets with landscapes, where there
            # is a long tail on the depth distribution
            depth = 1. / (nn.Sigmoid()(depth) * 10 + 0.01)



    """
    depth = depth.cpu().numpy().astype(np.uint8)
    """
    # plot_depth( (normalized_depth.cpu().numpy() * 255.0).astype(np.uint8),'network ouput depth')
    plot_depth(depth,'Converted True depth',norm=True)

    image_np = image_raw
    depth_map = depth



    # 模拟用户输入
    tx, ty, tz = 0.2, -0.3, 0.2  # 相对平移量 定义在三维坐标系上
    rx, ry, rz = 0, -60, 0  # 旋转角度（度数）
    sx, sy, sz = 1, 1, 1  # 缩放比例 >1为缩小


    transforms = [tx,ty,tz,rx,ry,rz,sx,sy,sz]
    # 获取3D变换矩阵
    # T = get_transformation_matrix(tx, ty, tz, rx, ry, rz, sx, sy, sz)


    # Global settings
    # FL = 715.0873
    # FY = 256 * 0.6
    # FX = 256 * 0.6
    # NYU_DATA = False
    # FINAL_HEIGHT = 256
    # FINAL_WIDTH = 256

    FINAL_WIDTH = image_np.shape[1]
    FINAL_HEIGHT = image_np.shape[0]
    # FX = FINAL_WIDTH * 0.6
    # FY = FINAL_HEIGHT * 0.6
    #GeoDiffuser
    FX = 1080
    FY = 1080
    splatting_radius = 0.015
    splatting_tau = 0.0
    splatting_points_per_pixel = 30


    # transformed_image, transformed_mask,transformed_depth,inpaint_mask = Integrated3DTransformAndInpaint(image_np,depth_map,transforms,FX,FY,mask,object_only=True)
    # view(transformed_image,title="transformed_image")
    # # plot_depth(transformed_depth.astype(np.uint8), 'transformed depth')
    # plot_depth(transformed_mask.astype(np.uint8), 'transformed_mask')
    # plot_depth(inpaint_mask.astype(np.uint8), 'inpaint_mask')
    # mask = (mask > 128).astype(bool)
    # transformed_mask = (transformed_mask > 128).astype(bool)
    # inpaint_mask = (inpaint_mask>128).astype(bool)
    # repair_mask = (mask & ~transformed_mask)| inpaint_mask
    # # plot_depth(repair_mask.astype(np.uint8)*255, 'repair_mask')
    #
    # image_with_hole = np.where(mask[:, :, None], 0, image_np).astype(np.uint8)  # for visualization use
    # new_image = np.where(transformed_mask[:, :, None], transformed_image,image_with_hole)
    # view(new_image,'blended_image')

    # old_transformed_mask = transformed_mask & ~inpaint_mask
    # image_with_hole = np.where(repair_mask[:, :, None], 0, image_np).astype(np.uint8)  # for visualization use
    # image_with_to_be_inpaint = np.where(old_transformed_mask[:, :, None], transformed_image,image_with_hole)  # for visualization use
    # view(image_with_to_be_inpaint, 'image_with_to_be_inpaint')
    # print('finish')

    ##############################################################################################################################
    ##############################################################################################################################

    transformed_image, transformed_mask = IntegratedP3DTransRasterBlending(image_np, depth_map,transforms,FX, FY, mask,object_only=True,
                                                                                                        splatting_radius=splatting_radius,
                                                                                                        splatting_tau=splatting_tau,
                                                                                                        splatting_points_per_pixel=splatting_points_per_pixel,
                                                                                                        return_mask=True,
                                                                                                        device = device)
    view(transformed_image,title="transformed_image")
    plot_depth(transformed_mask.astype(np.uint8), 'transformed_mask')
    mask = (mask > 128).astype(bool)
    transformed_mask = (transformed_mask > 128).astype(bool)
    repair_mask = (mask & ~transformed_mask)
    plot_depth(repair_mask.astype(np.uint8)*255, 'repair_mask')
    image_with_hole = np.where(mask[:, :, None], 0, image_np).astype(np.uint8)  # for visualization use
    new_image = np.where(transformed_mask[:, :, None], transformed_image,image_with_hole)
    view(new_image,'blended_image')








