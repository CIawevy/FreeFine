from src.demo.download import download_all
# download_all()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

from src.demo.demo import create_demo_move, create_demo_appearance, create_demo_drag, create_demo_face_drag, create_demo_paste,create_my_demo
from src.demo.model import DragonModels
import torch
import cv2
import gradio as gr


# main demo
# pretrained_model_path = "runwayml/stable-diffusion-v1-5"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
model = DragonModels(pretrained_model_path=pretrained_model_path,device=device)

DESCRIPTION = '# 🐉🐉[DragonDiffusion V1.0](https://github.com/MC-E/DragonDiffusion)🐉🐉'

DESCRIPTION += f'<p>Gradio demo for [DragonDiffusion](https://arxiv.org/abs/2307.02421) and [DiffEditor](https://arxiv.org/abs/2307.02421). If it is helpful, please help to recommend [[GitHub Repo]](https://github.com/MC-E/DragonDiffusion) to your friends 😊 </p>'

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        # with gr.TabItem('Appearance Modulation'):
            # create_demo_appearance(model.run_appearance)
        with gr.TabItem('Object Moving & Resizing'):
            create_demo_move(model.run_move)
        # with gr.TabItem('Face Modulation'):
            create_demo_face_drag(model.run_drag_face)
        # with gr.TabItem('Content Dragging'):
        #     create_demo_drag(model.run_drag)
        # with gr.TabItem('Object Pasting'):
            create_demo_paste(model.run_paste)
        # with gr.TabItem('Simple Copy-Paste & Inpainting'):
        #     create_my_demo(model.run_my_baseline)

demo.queue(concurrency_count=3, max_size=20)
# demo.launch(server_name="0.0.0.0")
# demo.launch(server_name="0.0.0.0",share=True)
demo.launch(server_name="127.0.0.1")

