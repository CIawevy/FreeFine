import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from simple_lama_inpainting import SimpleLama
from lama import lama_with_refine
from lora.lora_utils import train_lora_pipe
from src.demo.model import AutoPipeReggio
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

import clip
import warnings

def load_clip_on_the_main_Model(main_model,device):
    # 加载CLIP模型和处理器
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    main_model.clip = model
    main_model.clip_process = preprocess
    return main_model
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

#####################################################
image_path= "/data/Hszhu/prompt-to-prompt/CPIG/1.jpg"
mask_path = "/data/Hszhu/prompt-to-prompt/masks/1.png"
original_image= cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path)
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


pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
lora_path = "./lora_tmp"
# vae_path = "/data/Hszhu/prompt-to-prompt/sd-vae-ft-mse"
vae_path = "default"
train_lora_state = False
lora_prompt =  ''
lora_step =60
lora_lr = 0.0005
lora_batch_size =4
lora_rank = 16
#Train Lora
if train_lora_state:
    train_lora_pipe(original_image,#ndarray
                     lora_prompt,
                     lora_path,
                     lora_step,
                     lora_lr,
                     lora_batch_size,
                     lora_rank,)

#implement from p2p & FPE they have the same scheduler config which is different from official code
# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
# model = ClawerModels.from_pretrained(pretrained_model_path,scheduler=scheduler).to(device)
precision=torch.float32
model = AutoPipeReggio.from_pretrained(pretrained_model_path,torch_dtype=precision).to(device)

if vae_path != "default":
    model.vae = AutoencoderKL.from_pretrained(
        vae_path
    ).to(model.vae.device, model.vae.dtype)

model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
model.inpainter = lama_with_refine(device)
model.sd_inpainter = sd_inpainter
# model.depth_anything = depth_anything_v2
model = load_clip_on_the_main_Model(model,device)
controller = Mask_Expansion_SELF_ATTN(block_size=8,drop_rate=0.5,start_layer=10)
controller.contrast_beta = 1.67
controller.use_contrast = True
model.controller = controller
register_attention_control(model, controller)
model.modify_unet_forward()
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()



# selected_points = [[669, 426], [340, 540]]#car
# selected_points = [[185, 185], [133, 185]]#apple right
# selected_points = [[418, 406], [582, 528]] #mid apple
# selected_points= [[372, 300], [508, 296]] #AS
selected_points = [[157, 159], [195, 162]] #man



resize_scale = 1.0
rotation_angle =0
flip_horizontal = 0
flip_vertical = 0
guidance_scale = 1.1
eta = 1.0
num_step = 50
start_step = 25
end_step = 0
feature_injection=True
use_sdsa = True
#SDSA ['down','mid','up']
# FI_range=(782,680)
FI_range=(682,640)
sim_thr = 0.5
DIFT_LAYER_IDX = [0,1,2,3] #[1,2,3]
mask_threshold =  0.2
mask_threshold_target = 0.5
mode = 2
use_mask_expansion =1,
strong_inpaint = 1
cross_enhance = 0
standard_drawing = 1
blending_alpha = 1.0
max_resolution = 512
dilate_kernel_size = 30
contrast_beta = 1.67
seed = 42


prompt="a photo of a blue car driving down the road"
# prompt = 'a photo of apples lying on the wooden table with cracks,realistic style'
# prompt = 'a photo of sun,with blue sky and white clouds,and a tree'
# prompt = 'an astronaut is playing football on the moon'
# prompt='tennis woman'
# prompt = 'Hipster businessman with idea, light bulb above his head, isolated on black background'

INP_prompt = 'empty scene'
INP_prompt='blur empty scene, highest quality'
INP_prompt='a photo of a background, a photo of an empty place'
# prompt = 'car'
assist_prompt = ['shadow','light']
# assist_prompt=['beam']


output_edit, refer_edit,INP_IMG, INP_Mask, TGT_MSK,inp_bg = model.Magic_Editing_Baseline_2D(original_image,prompt, INP_prompt,selected_points,
                                  seed, guidance_scale, num_step, max_resolution, mode,start_step, resize_scale,
                                  rotation_angle, flip_horizontal,flip_vertical,eta=1, use_mask_expansion=True,expansion_step=20,contrast_beta=1.67,  end_step=0,
                                feature_injection=True, FI_range=(900, 680),sim_thr=0.5, DIFT_LAYER_IDX=[0, 1, 2, 3], use_mtsa=True,select_mask=mask,assist_prompt=assist_prompt
                                  )
visualize_rgb_image(output_edit[0], title="output_edit")
visualize_rgb_image(refer_edit[0], title="refer_edit")
visualize_rgb_image(INP_IMG[0], title="INP_IMG")
visualize_rgb_image(inp_bg[0], title="repaired BG")
# visualize_rgb_image(noised_optimized_image[0], title="noised_optimized_image")
# model.temp_view(TGT_MSK[0])
visualize_rgb_image(INP_Mask[0], title="Inp_mask")


model.prepare_h_feature()
model.Details_Preserving_regeneration()
model.invert()
model.forward_sampling_BG()
#TODO:Size 变回原图size 经过一次resize的
Mask_Expansion_SELF_ATTN()
Mask_Expansion_SELF_ATTN()
register_attention_control(model, controller)
model.modify_unet_forward()

