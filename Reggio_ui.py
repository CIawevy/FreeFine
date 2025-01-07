from src.demo.download import download_all
# download_all()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from simple_lama_inpainting import SimpleLama
from lama import lama_with_refine
from src.demo.demo import create_my_demo_full_3D_magic
from src.demo.demo_v2 import create_my_demo_full_SV3D_magic,create_my_demo_full_SV3D_multi_obj_case,create_my_demo_full_2D_magic
from src.demo.src_model import ClawerModels,ClawerModel_v2
from src.unet.unet_2d_condition import DragonUNet2DConditionModel
import torch
import cv2
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
import gradio as gr

from depth_anything_v2.dpt import DepthAnythingV2
from torchvision.transforms import Compose
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline
from transformers import  CLIPTokenizer
import clip
def load_clip_on_the_main_Model(main_model,device):
    # 加载CLIP模型和处理器
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    main_model.clip = model
    main_model.clip_process = preprocess
    return main_model
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


# model_configs = {
#     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
# }
#
# encoder = 'vits'  # or 'vits', 'vitb', 'vitg'
#
# depth_anything_v2 = DepthAnythingV2(**model_configs[encoder])
# depth_anything_v2.load_state_dict(
#     torch.load(f'/data/Hszhu/prompt-to-prompt/depth-anything/depth_anything_v2_{encoder}.pth', map_location='cpu'))
# depth_anything_v2.to(device).eval()


pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path")
vae_path = "/data/Hszhu/prompt-to-prompt/sd-vae-ft-mse"
# vae_path='default'
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
model.inpainter = lama_with_refine(device)
model = load_clip_on_the_main_Model(model,device)
# model.unet = DragonUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", torch_dtype=precision).to(device)
# model.inpainter = None
model.sd_inpainter = sd_inpainter
# model.depth_anything = depth_anything_v2
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
DESCRIPTION = '# 🐉🐉[Reggio V1.0](https://github.com/CIawevy/Reggio)🐉🐉'

DESCRIPTION += f'<p>Gradio demo for [Reggio](https://arxiv.org/abs/2307.02421). If it is helpful, please help to recommend [[GitHub Repo]](https://github.com/CIawevy/Reggio) to your friends 😊 </p>'

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('2D Magical editing pipeline'):
            create_my_demo_full_2D_magic(model.Magic_Editing_Baseline_2D)
        with gr.TabItem('3D Magical editing pipeline'):
            create_my_demo_full_3D_magic(model.Magic_Editing_Baseline_3D)
        with gr.TabItem('SV3D Magical Auto-Mask Editing pipeline'):
            create_my_demo_full_SV3D_magic(model.Magic_Editing_Baseline_SV3D)
        # with gr.TabItem('SV3D Magical Multi-Obj-selected Editing pipeline'):
        #     create_my_demo_full_SV3D_multi_obj_case(model.Magic_Editing_Baseline_SV3D)

demo.queue(concurrency_count=3, max_size=20)
# demo.launch(server_name="0.0.0.0")
# demo.launch(server_name="0.0.0.0",share=True)
demo.launch(server_name="127.0.0.1")

