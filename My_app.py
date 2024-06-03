from src.demo.download import download_all
# download_all()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from simple_lama_inpainting import SimpleLama
from src.demo.demo import create_my_demo,create_my_demo_full
from src.demo.model import ClawerModels
from src.unet.unet_2d_condition import DragonUNet2DConditionModel
import torch
import cv2
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
import gradio as gr
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline
# main demo
# pretrained_model_path = "runwayml/stable-diffusion-v1-5"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))


# pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-inpainting/"
pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-2-inpainting/"
sd_inpainter = StableDiffusionInpaintPipeline.from_pretrained(
    pretrained_inpaint_model_path,
    revision="fp16",
    torch_dtype=torch.float16,
).to(device)
sd_inpainter.enable_attention_slicing()



pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
#implement from p2p & FPE they have the same scheduler config which is different from official code
# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
# model = ClawerModels.from_pretrained(pretrained_model_path,scheduler=scheduler).to(device)
precision=torch.float32
model = ClawerModels.from_pretrained(pretrained_model_path,torch_dtype=precision).to(device)
# model.precision = torch.float32
# model.precision=precision
# Set up a DDIM scheduler
model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
model.inpainter = SimpleLama()
# model.unet = DragonUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", torch_dtype=precision).to(device)
# model.inpainter = None
model.sd_inpainter = sd_inpainter
controller = Mask_Expansion_SELF_ATTN()
controller.contrast_beta = 1.67
controller.use_contrast = True
model.controller = controller
register_attention_control(model, controller)
model.enable_attention_slicing()
model.enable_xformers_memory_efficient_attention()
DESCRIPTION = '# 🐉🐉[DragonDiffusion V1.0](https://github.com/MC-E/DragonDiffusion)🐉🐉'

DESCRIPTION += f'<p>Gradio demo for [DragonDiffusion](https://arxiv.org/abs/2307.02421) and [DiffEditor](https://arxiv.org/abs/2307.02421). If it is helpful, please help to recommend [[GitHub Repo]](https://github.com/MC-E/DragonDiffusion) to your friends 😊 </p>'

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        # with gr.TabItem('Simple Copy-Paste & Inpainting'):
            # create_my_demo(model.run_my_Baseline)
        with gr.TabItem('Expansion Mask geometric editing'):
            create_my_demo_full(model.run_my_Baseline_full)

demo.queue(concurrency_count=3, max_size=20)
# demo.launch(server_name="0.0.0.0")
# demo.launch(server_name="0.0.0.0",share=True)
demo.launch(server_name="127.0.0.1")

