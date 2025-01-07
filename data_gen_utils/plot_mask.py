import gradio as gr
import numpy as np
import cv2
from PIL import Image
import json
# 处理绘制的图像，转换为掩膜
def process_drawing(drawing):
    # 将绘制的图像转换为灰度图像
    mask = drawing["mask"].convert("L")
    # 将灰度图像转换为numpy数组
    mask_array = np.asarray(mask)
    # 将灰度值大于128的区域设为1，其余为0
    mask_array = np.where(mask_array > 128, 1, 0).astype(np.uint8)

    # 创建单通道掩膜图像并保存，确保是L模式（单通道灰度图）
    mask_image = Image.fromarray(mask_array * 255, mode="L")  # 使用 "L" 模式来确保单通道
    save_path = "draw_mask2.png"
    mask_image.save(save_path)  # 保存为PNG格式（单通道）

    # 返回处理后的掩膜图像（单通道）
    return mask_image



def create_ui(image_path):
    with gr.Blocks() as demo: # 显示原始图像
        # 创建绘图区域，启用图像编辑
        drawing = gr.Image(value=image_path, label="Draw the Mask", type="pil", interactive=True, tool="sketch",height=512, width=512,brush_color='#FFFFFF', mask_opacity=0.5, brush_radius=30)

        # 创建保存按钮
        save_button = gr.Button("Process")

        # 定义保存按钮的行为：处理绘制的图像
        save_button.click(process_drawing, inputs=[drawing], outputs=[gr.Image(type="pil", label="Processed Mask")])


    return demo
def load_json(file_path):
    """
    加载指定路径的JSON文件并返回数据。

    :param file_path: JSON文件的路径
    :return: 从JSON文件中加载的数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        print(f"文件格式错误: {file_path}")
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
    return None


data = load_json("/data/Hszhu/dataset/PIE-Bench_v1/Subset_0/coarse_input_full_pack_0.json")
da_n = '102'
da = data[da_n]
instances = da['instances']
edit_meta = instances['0']
coarse_input_pack = edit_meta['0']
# 定义图像路径
image_path = coarse_input_pack['coarse_input_path']

# 创建并启动 Gradio UI
ui = create_ui(image_path)
ui.launch()