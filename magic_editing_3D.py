import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from simple_lama_inpainting import SimpleLama
from src.demo.demo import create_my_demo,create_my_demo_full_2D,create_my_demo_full_3D_magic,create_my_demo_full_2D_ctn
from src.demo.model import ClawerModels,ClawerModel_v2
from src.unet.unet_2d_condition import DragonUNet2DConditionModel
import torch
import cv2
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
import gradio as gr
from depth_anything_v2.dpt import DepthAnythingV2
from torchvision.transforms import Compose
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline,UNet2DConditionModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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
# main demo
# pretrained_model_path = "runwayml/stable-diffusion-v1-5"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-inpainting/"
# pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-2-inpainting/"
sd_inpainter = StableDiffusionInpaintPipeline.from_pretrained(
    pretrained_inpaint_model_path,
    revision="fp16",
    torch_dtype=torch.float16,
).to(device)
sd_inpainter.enable_attention_slicing()


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'  # or 'vits', 'vitb', 'vitg'

depth_anything_v2 = DepthAnythingV2(**model_configs[encoder])
depth_anything_v2.load_state_dict(
    torch.load(f'/data/Hszhu/prompt-to-prompt/depth-anything/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything_v2.to(device).eval()


pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path")
# vae_path = "/data/Hszhu/prompt-to-prompt/sd-vae-ft-mse"
vae_path = "default"
#implement from p2p & FPE they have the same scheduler config which is different from official code
# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
# model = ClawerModels.from_pretrained(pretrained_model_path,scheduler=scheduler).to(device)
precision=torch.float32
model = ClawerModel_v2.from_pretrained(pretrained_model_path,torch_dtype=precision).to(device)

if vae_path != "default":
    model.vae = AutoencoderKL.from_pretrained(
        vae_path
    ).to(model.vae.device, model.vae.dtype)
# model.precision = torch.float32
# model.precision=precision
# Set up a DDIM scheduler
model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
model.inpainter = SimpleLama()
# model.unet = DragonUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", torch_dtype=precision).to(device)
# model.inpainter = None
model.sd_inpainter = sd_inpainter
model.depth_anything = depth_anything_v2
# model.depth_anything = depth_anything
# model.transform = transform
controller = Mask_Expansion_SELF_ATTN(block_size=8,drop_rate=0.5,start_layer=10)
controller.contrast_beta = 1.67
controller.use_contrast = True
model.controller = controller
register_attention_control(model, controller)
# prepare for drag diffusion module
model.modify_unet_forward()
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
image_path= "/data/Hszhu/prompt-to-prompt/CPIG/1.jpg"
mask_path = "/data/Hszhu/prompt-to-prompt/masks/1.png"
#cv2 loading to ndarray

#####################################################
original_image= cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path)

guidance_scale = 7.5
eta = 0.3
num_step = 50
start_step = 15
end_step = 5
feature_injection=True
use_sdsa = True
#SDSA ['down','mid','up']
# FI_range=(782,680)
FI_range=(682,640)
sim_thr = 0.5
DIFT_LAYER_IDX = [0,1,2,3] #[1,2,3]
mask_threshold =  0.2
mask_threshold_target = 0.5
mode =2
use_mask_expansion =1,
strong_inpaint = 1
cross_enhance = 0
standard_drawing = 1
blending_alpha = 1.0
max_resolution = 512
dilate_kernel_size = 30
contrast_beta = 1.67

seed = 42
splatting_radius = 0.015
splatting_tau =  0.0
splatting_points_per_pixel = 30
focal_length =  340
sx =  1
sy = 1
sz = 1
rx = 0
ry = 0
rz = 0
tx = 0.2
ty = -0.2
tz =  0.3

prompt="car"
INP_prompt='empty scene'





output_edit, refer_edit,INP_IMG, INP_Mask, TGT_MSK, depth_map = model.Magic_Editing_Baseline_3D(original_image, mask, prompt, INP_prompt, seed, guidance_scale, num_step,
                                 max_resolution, mode, dilate_kernel_size,
                                 start_step, tx, ty, tz, rx, ry, rz, sx, sy, sz, None, eta, use_mask_expansion,
                                 standard_drawing, contrast_beta, strong_inpaint, cross_enhance,
                                 mask_threshold, mask_threshold_target, blending_alpha, splatting_radius, splatting_tau,
                                 splatting_points_per_pixel, focal_length,end_step,feature_injection,FI_range,sim_thr,DIFT_LAYER_IDX,use_sdsa                             )
visualize_rgb_image(output_edit[0], title="output_edit")
# visualize_rgb_image(refer_edit[0], title="refer_edit")
# visualize_rgb_image(INP_IMG[0], title="INP_IMG")
# visualize_rgb_image(noised_optimized_image[0], title="noised_optimized_image")
# model.temp_view(TGT_MSK[0])


model.prepare_h_feature()
model.Details_Preserving_regeneration()

model.invert()
model.forward_sampling_BG()
#TODO:Size 变回原图size 经过一次resize的
Mask_Expansion_SELF_ATTN()
register_attention_control(model, controller)
model.modify_unet_forward()

def otsu_thresholding_torch(image):
    # 画像のヒストグラムを計算
    hist = torch.histc(image.float(), bins=256, min=0, max=255).to(image.device)

    # 各閾値でのクラス内分散とクラス間分散を計算
    pixel_sum = torch.sum(hist)
    weight1 = torch.cumsum(hist, 0)
    weight2 = pixel_sum - weight1

    bin_edges = torch.arange(256).float().to(image.device)

    # ゼロ除算を避ける
    mean1 = torch.cumsum(hist * bin_edges, 0) / (weight1 + (weight1 == 0))
    mean2 = (torch.cumsum(hist.flip(0) * bin_edges.flip(0), 0) / (weight2.flip(0) + (weight2.flip(0) == 0))).flip(0)


    # クラス間分散を最大にする閾値を求める
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    threshold = torch.argmax(inter_class_variance)

    # 二値化処理
    binary_image = torch.where(image <= threshold, 0, 255)
    return binary_image.type(torch.uint8)
def mask_with_otsu_pytorch(tensor : torch.Tensor):
    # Tensorをnumpy配列に変換し、範囲を0-255に変換
    image = (tensor * 255).to(torch.uint8)

    # 大津の二値化を適用
    binary_image = otsu_thresholding_torch(image)

    # 二値化された画像を0~1のTensorに変換
    return (binary_image / 255).to(tensor.dtype)



