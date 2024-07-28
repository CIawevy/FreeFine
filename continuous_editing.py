from src.demo.download import download_all
# download_all()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from simple_lama_inpainting import SimpleLama
from src.demo.demo import create_my_demo,create_my_demo_full_2D,create_my_demo_full_3D,create_my_demo_full_2D_ctn
from src.demo.model import ClawerModels,ClawerModel_v2
from src.unet.unet_2d_condition import DragonUNet2DConditionModel
import torch
import cv2
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
vae_path = "/data/Hszhu/prompt-to-prompt/sd-vae-ft-mse"
lora_path = "/data/Hszhu/DragNoise/lora_tmp/"
#implement from p2p & FPE they have the same scheduler config which is different from official code
# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
# model = ClawerModels.from_pretrained(pretrained_model_path,scheduler=scheduler).to(device)
precision=torch.float32
model = ClawerModel_v2.from_pretrained(pretrained_model_path,torch_dtype=precision).to(device)
# model.precision = torch.float32
# model.precision=precision
# Set up a DDIM scheduler
if vae_path != "default":
    model.vae = AutoencoderKL.from_pretrained(
        vae_path
    ).to(model.vae.device, model.vae.dtype)
# # set lora
# if lora_path == "":
#     print("applying default parameters")
#     model.unet.set_default_attn_processor()
# else:
#     print("applying lora: " + lora_path)
#     model.unet.load_attn_procs(lora_path)
model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
model.inpainter = SimpleLama()
# model.unet = DragonUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", torch_dtype=precision).to(device)
# model.inpainter = None
model.depth_anything = depth_anything_v2
# model.depth_anything = depth_anything
# model.transform = transform
controller = Mask_Expansion_SELF_ATTN()
controller.contrast_beta = 1.67
controller.use_contrast = True
model.controller = controller
register_attention_control(model, controller)
model.modify_unet_forward()
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
DESCRIPTION = '# 🐉🐉[Reggio V1.0](https://github.com/CIawevy/Reggio)🐉🐉'

DESCRIPTION += f'<p>Gradio demo for [Reggio](https://arxiv.org/abs/2307.02421) and [DiffEditor](https://arxiv.org/abs/2307.02421). If it is helpful, please help to recommend [[GitHub Repo]](https://github.com/CIawevy/Reggio) to your friends 😊 </p>'

image_path= "/data/Hszhu/prompt-to-prompt/CPIG/1.jpg"
mask_path = "/data/Hszhu/prompt-to-prompt/masks/1.png"
#cv2 loading to ndarray
prompt = 'car'
motion_split_steps=10
seed=42
selected_points = [[1000, 526], [615, 635]]

# selected_points = [[1095, 525],[1095, 525]]
# selected_points = [[1095, 525], [701, 879]]
selected_points = [[1095, 525], [1095, 525]]
guidance_scale=1
num_step=50
max_resolution=512
dilate_kernel_size=30
start_step=15
end_step = 10
mask_ref=None
eta=0
use_mask_expansion=0
standard_drawing=1
contrast_beta= 1.67
resize_scale=1.0
rotation_angle= 0
strong_inpaint=1
flip_horizontal=0
flip_vertical=0
cross_enhance=0
mask_threshold=0.2
mask_threshold_target=0.2
blending_alpha=0.7
max_times=1
sim_thr=0.90
original_image= cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path)
lr=0.1
lam=0.01
ori_gen_image, edit_gen_image, noised_optimized_image, candidate_mask=\
    model.Continuous_Editing_Baseline_2D(original_image, mask, prompt, motion_split_steps, seed, selected_points, guidance_scale,
                         num_step, max_resolution, dilate_kernel_size,max_times,sim_thr,lr,lam,
                         start_step, end_step , mask_ref, eta, use_mask_expansion, standard_drawing, contrast_beta, resize_scale,
                         rotation_angle, strong_inpaint, flip_horizontal, flip_vertical, cross_enhance,
                         mask_threshold, mask_threshold_target, blending_alpha)
visualize_rgb_image(ori_gen_image[0], title="ori_gen_image")
# visualize_rgb_image(edit_gen_image[0], title="edit_gen_image")
# visualize_rgb_image(noised_optimized_image[0], title="noised_optimized_image")
# model.temp_view(candidate_mask[0])


