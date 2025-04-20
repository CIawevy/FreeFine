import os
import subprocess
import shlex
import torch
from src.demo.model import FreeFine

model_path = 'models/efficient_sam_vits.pt'
os.makedirs('models', exist_ok=True)  # Ensure models directory exists

if not os.path.exists(model_path):  # Only download if file doesn't exist
    print("Downloading model file...")
    subprocess.run(shlex.split(f'wget https://hf-mirror.com/Adapter/DragonDiffusion/resolve/main/model/efficient_sam_vits.pt -O {model_path}'))
    print("Download completed!")
else:
    print("Model file already exists, skipping download")

from src.demo.demo import *
import cv2
import gradio as gr
device = torch.device('cuda:0')
pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
model =  FreeFine(pretrained_model_path=pretrained_model_path,device=device)

DESCRIPTION = """
<div class="header">
    <h1>🖼️ FreeFine: Training-Free Geometric Image Editing</h1>
    <div style="font-size: 1.1rem; color: #666; margin-top: 1rem;">
        <p>Official implementation of the FreeFine framework for geometric-aware image editing</p>
        <div style="margin-top: 1rem;">
            <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank" style="color: var(--secondary-color); text-decoration: none; margin: 0 1rem;">
                📜 Research Paper
            </a>
            <a href="https://github.com/CIawevy/FreeFine/" target="_blank" style="color: var(--accent-color); text-decoration: none; margin: 0 1rem;">
                💻 GitHub Repository
            </a>
        </div>
    </div>
</div>
"""


with gr.Blocks(css='mystyle.css') as demo:
    gr.HTML(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('✨ Object Removal'):
            create_demo_remove(model.run_remove)
        # with gr.TabItem('🔷 Geometric Editing'):
        #     create_demo_edit(model.run_edit)
        # with gr.TabItem('🎨 Appearance Transfer'):
        #     create_demo_compose(model.run_compose)
        # with gr.TabItem('🎨 Cross-Image Composition'):
        #     create_demo_compose(model.run_compose)
    # 添加页脚
    gr.HTML("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>Made with ❤️ by FreeFine Team | 
        <a href="https://github.com/CIawevy/FreeFine/issues" target="_blank" style="color: var(--secondary-color);">Report Issues</a></p>
    </div>
    """)

demo.queue(concurrency_count=3, max_size=20)
demo.launch(server_name="127.0.0.1")

