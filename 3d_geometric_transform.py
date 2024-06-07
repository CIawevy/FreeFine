from src.utils.geometric_utils import get_transformation_matrix,warpAffine3D
import os


os.environ["CUDA_VISIBLE_DEVICES"]="0"
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






def plot_depth(depth_plot,title='title'):
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


    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }

    encoder = 'vitl'  # or 'vitb', 'vits'
    depth_anything = DepthAnything(model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_{encoder}14.pth'))
    depth_anything.to(device).eval()
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    image_path = "/data/Hszhu/DragonDiffusion/examples/Expansion_Mask/CPIG/1.jpg"
    mask_path = "/data/Hszhu/DragonDiffusion/examples/Expansion_Mask/masks/1.png"
    # dst_path = "/data/Hszhu/DragonDiffusion/examples/3D-Trans"
    # Path(dst_path).mkdir(parents=True, exist_ok=True)
    # image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0
    color_image = Image.open(image_path).convert('RGB')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image_raw = np.array(color_image)
    plt.imshow(image_raw)
    plt.title("RAW Image")
    plt.show()
    h, w = image_raw.shape[:2]
    #preprocess image for depth estimate
    image = image_raw / 255
    image = transform({'image': image})['image']

    # Reshape transformed image to (H, W, C) for plotting
    image_plot = np.transpose(image, (1, 2, 0))


    image = torch.from_numpy(image).unsqueeze(0).to(device)
    # Load image from image path to PIL image
    base_depth = 0.1
    depth_unit = 255
    EPISILON = 1e-8

    with torch.no_grad():
        depth = depth_anything(image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth_unit * torch.clip((base_depth + 1 - normalized_depth),max=1)
    depth = depth.cpu().numpy().astype(np.uint8)
    # plot_depth( (normalized_depth.cpu().numpy() * 255.0).astype(np.uint8),'network ouput depth')
    plot_depth(depth,'Converted True depth')

    image_np = image_raw
    depth_map = depth



    # 模拟用户输入
    tx, ty, tz = 0.1, 0, 0  # 相对平移量 定义在三维坐标系上
    rx, ry, rz = 0, 0, 0  # 旋转角度（度数）
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
    FX = FINAL_WIDTH * 0.6
    FY = FINAL_HEIGHT * 0.6



    # resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
    # resized_pred = Image.fromarray(depth).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)
    transformed_image, transformed_depth = warpAffine3D(image_np,depth_map,transforms,FX,FY,mask,)
    view(transformed_image,title="transformed_image")
    plot_depth(transformed_depth.astype(np.uint8), 'transformed depth')

    transformed_mask,_ = warpAffine3D(mask[:,:,None],depth_map,transforms,FX,FY,mask)
    plot_depth(transformed_mask.astype(np.uint8), 'transformed_mask')
    mask = (mask > 128).astype(bool)
    transformed_mask = (transformed_mask > 128).astype(bool)

    image_with_hole = np.where(mask[:,:,None], 0, image_np).astype(np.uint8)  # for visualization use
    new_image = np.where(transformed_mask[:, :, None], transformed_image,image_with_hole)

    view(new_image,'blended_image')




