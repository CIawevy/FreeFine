import gradio as gr
import numpy as np
from PIL import Image
import json
import os

import time

from click import clear
from tqdm import tqdm

# 全局变量
# DATA = None
# STAT_DATA = None
CURRENT_INDEX = 0  # 当前图像的索引
CURRENT_INS_INDEX = 0 # 当前实例的索引
CURRENT_EDIT_INDEX = 0  # 当前编辑的索引
# 加载或创建掩码数据的函数
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
# 保存筛选后的数据
def save_json_data(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
# 加载数据的通用函数：如果文件不存在则创建空字典并保存
def load_or_create_json(path,DATA):
    if os.path.exists(path):
        return load_json_data(path)
    else:
        ori_data_stat = DATA
        # 文件不存在时，初始化为空字典
        # 如果文件不存在，初始化状态数据
        for da_n , da in tqdm(ori_data_stat.items(),desc='initializing'):
            ori_data_stat[da_n].pop('4v_caption')
            ori_data_stat[da_n]['status'] = 'unprocessed'
            for ins_n,instances_12 in da['instances'].items():
                ori_data_stat[da_n]['instances'][ins_n] = {'stat': 'unprocessed', 'processed_id': []}
        ori_data_stat['progress'] = '0.00%'

        save_json_data(ori_data_stat, path)
        return ori_data_stat
# 处理绘制的图像，转换为掩膜
def save_stat(DATA,STAT_DATA):
    global CURRENT_INDEX, CURRENT_INS_INDEX, CURRENT_EDIT_INDEX
    if CURRENT_INDEX not in STAT_DATA:
        STAT_DATA[CURRENT_INDEX] = {'status': 'processing', 'instances': {}}
    if CURRENT_INS_INDEX not in STAT_DATA[CURRENT_INDEX]['instances']:
        STAT_DATA[CURRENT_INDEX]['instances'][CURRENT_INS_INDEX] = {'stat': 'processing', 'processed_id': []}
    STAT_DATA[CURRENT_INDEX]['instances'][CURRENT_INS_INDEX]['processed_id'].append(CURRENT_EDIT_INDEX)
    if len(STAT_DATA[CURRENT_INDEX]['instances'][CURRENT_INS_INDEX]['processed_id']) == len(DATA[CURRENT_INDEX]['instances'][CURRENT_INS_INDEX]):
        STAT_DATA[CURRENT_INDEX]['instances'][CURRENT_INS_INDEX]['stat'] = 'done'
    if len(STAT_DATA[CURRENT_INDEX]['instances']) == len(DATA[CURRENT_INDEX]['instances']) and all(STAT_DATA[CURRENT_INDEX]['instances'][ins_id]['stat'] == 'done' for ins_id in DATA[CURRENT_INDEX]['instances']):
        STAT_DATA[CURRENT_INDEX]['status'] = 'done'
    save_json_data(STAT_DATA,"/data/Hszhu/dataset/Geo-Bench-SC/plot_stat.json")
    valid_idx=[]

    for da_n,da in STAT_DATA.items():
        if da_n == 'progress':
            continue
        if da['status'] == 'done':
            valid_idx.append(1)
    progress = len(valid_idx) / len(DATA) * 100
    STAT_DATA['progress'] = f'{progress:.2f}%'
def process_drawing(drawing,progress,DATA,STAT_DATA):
    data = DATA[0]
    stat_data = STAT_DATA[0]
    global CURRENT_INDEX, CURRENT_INS_INDEX, CURRENT_EDIT_INDEX
    dest_dir = "/data/Hszhu/dataset/Geo-Bench-SC"
    draw_mask_dir = os.path.join(dest_dir, 'draw_mask')

    os.makedirs(draw_mask_dir, exist_ok=True)
    # 将绘制的图像转换为灰度图像
    mask = drawing["mask"].convert("L")
    # 将灰度图像转换为numpy数组
    mask_array = np.asarray(mask)
    # 将灰度值大于128的区域设为1，其余为0
    mask_array = np.where(mask_array > 128, 1, 0).astype(np.uint8)

    # 创建单通道掩膜图像并保存，确保是L模式（单通道灰度图）
    mask_image = Image.fromarray(mask_array * 255, mode="L")  # 使用 "L" 模式来确保单通道

    subfolder_path = os.path.join(draw_mask_dir , CURRENT_INDEX)
    os.makedirs(subfolder_path, exist_ok=True)

    ins_subfolder_path = os.path.join(subfolder_path, CURRENT_INS_INDEX)
    os.makedirs(ins_subfolder_path, exist_ok=True)
    final_path  = os.path.join(ins_subfolder_path, f"draw_{CURRENT_EDIT_INDEX}.png")


    # save_path = "draw_mask_penguin.png"
    mask_image.save(final_path)  # 保存为PNG格式（单通道）
    # 返回处理后的掩膜图像（单通道）
    progress = stat_data['progress']
    save_stat(data, stat_data)
    return mask_image,progress,[data],[stat_data]

# 加载JSON文件
def load_json(file_path):
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

# 获取当前实例和编辑的路径
def get_current_path(da_n,ins_n,edit_n,data):
    return data[da_n]['instances'][ins_n][edit_n]['ori_img_path'],data[da_n]['instances'][ins_n][edit_n]['coarse_input_path']

# 获取当前实例和编辑的路径
def get_edit_info(da_n,ins_n,edit_n,data):
    p= data[da_n]['instances'][ins_n][edit_n]['edit_prompt']
    param = data[da_n]['instances'][ins_n][edit_n]['edit_param']
    pa= f'dx={param[0]},dy={param[1]},r={param[5]} s={param[6]}'
    return p,pa
# 更新图像显示
def update_image(da_n,ins_n,edit_n,data):
    global CURRENT_INDEX, CURRENT_INS_INDEX, CURRENT_EDIT_INDEX
    CURRENT_INDEX = da_n  # 当前图像的索引
    CURRENT_INS_INDEX = ins_n  # 当前实例的索引
    CURRENT_EDIT_INDEX = edit_n  # 当前编辑的
    ori_path,path = get_current_path(da_n,ins_n,edit_n,data)
    edit_prompt,edit_param = get_edit_info(da_n,ins_n,edit_n,data)
    # 加载原始图像
    image = Image.open(path).convert("RGBA")

    return ori_path, image, edit_prompt,edit_param

# 点击Next按钮的逻辑
def next_instance(image, drawing, prompt, param, progress,DATA,STAT_DATA):
    data=DATA[0]
    stat_data=STAT_DATA[0]
    for da_n,da in data.items():
        if da_n =='progress':
            continue

        try:
            if  stat_data[da_n]['status'] == 'done':
                continue
        except:
            print(F'stat data:{STAT_DATA}')
            print(f'DATA:{DATA}')
        for ins_n,instances_12 in da['instances'].items():
            if   stat_data[da_n]['instances'][ins_n]['stat'] == 'done':
                continue
            for edit_n,edit in instances_12.items():
                if edit_n in  stat_data[da_n]['instances'][ins_n]['processed_id']:
                    continue
                else:
                    image, drawing, prompt, param  = update_image(da_n,ins_n,edit_n,data)
                    progress, DATA, STAT_DATA = stat_data['progress'],[data],[stat_data]
                    return image, drawing, prompt, param, progress,DATA,STAT_DATA


# 点击Current按钮的逻辑
def current_instance(image, drawing, prompt, param,da_n, ins_n, edit_n,DATA):
    image, drawing, prompt, param = update_image(str(da_n),str(ins_n),str(edit_n),DATA[0])
    return image, drawing, prompt, param

# 点击DIY Edit按钮的逻辑
def diy_edit(image, dx, dy, rz, s):
    # 这里可以调用你后续写的编辑函数
    # 例如：image = your_edit_function(image, dx, dy, rz, s)
    # 假设我们只是简单地返回原始图像和绘图区域
    return image, None
def process_and_next(drawing, image, prompt, param, progress, data_state, stat_data_state):
    # 执行 process_drawing 的逻辑
    processed_mask, progress, data_state, stat_data_state = process_drawing(drawing, progress, data_state, stat_data_state)
    # 执行 next_instance 的逻辑
    image, drawing, prompt, param, progress, data_state, stat_data_state = next_instance(image, drawing, prompt, param, progress, data_state, stat_data_state)

    # 返回所有需要更新的组件
    return drawing, image, prompt, param, progress, data_state, stat_data_state
# 创建UI
def create_ui(DATA,STAT_DATA):
    # 加载数据

    with gr.Blocks() as demo:
        # 使用 State 组件保存 DATA 和 STAT_DATA
        data_state = gr.State(value=[DATA])
        stat_data_state = gr.State(value=[STAT_DATA])


        # 参数输入行
        with gr.Row():
            da_input = gr.Textbox(label="案例号 (da_n)", value="0")
            ins_input = gr.Textbox(label="实例ID (ins_id)", value="0")
            edit_input = gr.Textbox(label="编辑ID (edit_id)", value="0")
            dx = gr.Textbox(label="dx", value="0")
            dy = gr.Textbox(label="dy", value="0")
            rz = gr.Textbox(label="rz", value="0")
            s = gr.Textbox(label="s", value="0")

        # 显示图像
        image = gr.Image(label="Image", type="filepath", height=512, width=512)

        # 创建绘图区域
        drawing = gr.Image(label="Draw the Mask", type="pil", interactive=True, tool="sketch", height=512, width=512, brush_color='#FFFFFF', mask_opacity=0.5, brush_radius=30)
        draw_mask = gr.Image(type="pil", label="Processed Mask")
        # 创建文本显示区域
        prompt = gr.Textbox(label="edit_prompt", interactive=False)
        param = gr.Textbox(label="edit_param", interactive=False)
        progress = gr.Textbox(label="progress", interactive=False)

        # 创建按钮
        with gr.Row():
            plot_button = gr.Button("Plot & SAVE")
            next_button = gr.Button("Next")
            # clear_button = gr.Button("Clear")
            current_button = gr.Button("Current")
            diy_edit_button = gr.Button("DIY Edit")

        # # 定义按钮的行为
        next_button.click(
            next_instance,
            inputs=[image, drawing, prompt, param, progress,data_state, stat_data_state],  # 使用 State 组件作为输入
            outputs=[image, drawing, prompt, param, progress, data_state, stat_data_state]  # 返回更新后的 State
        )
        # clear_button.click(
        #     clear,
        #     inputs=[ drawing],  # 使用 State 组件作为输入
        #     outputs=[drawing]  # 返回更新后的 State
        # )
        current_button.click(
            current_instance,
            inputs=[image, drawing, prompt, param,da_input, ins_input, edit_input, data_state],
            outputs=[image, drawing, prompt, param]
        )
        diy_edit_button.click(
            diy_edit,
            inputs=[image, dx, dy, rz, s, data_state], #TODO TOBE MODIFIED
            outputs=[image, drawing, prompt, param, data_state]
        )
        plot_button.click(
            process_drawing,  # 新的函数，合并 process_drawing 和 next_instance 的逻辑
            inputs=[drawing,  progress, data_state, stat_data_state],
            outputs=[draw_mask, progress, data_state, stat_data_state]
        )
        # print(f'sleep for 3 seconds to avoid bug')
        # time.sleep(3)
        # 初始化图像
        print(stat_data_state)
        demo.load(
            next_instance,
            inputs=[image, drawing, prompt, param, progress, data_state, stat_data_state],
            outputs=[image, drawing, prompt, param, progress, data_state, stat_data_state],
        )

    return demo

# 创建并启动 Gradio UI
DATA = load_json("/data/Hszhu/dataset/Geo-Bench-SC/annotations.json")
STAT_DATA = load_or_create_json("/data/Hszhu/dataset/Geo-Bench-SC/plot_stat.json", DATA)
ui = create_ui(DATA,STAT_DATA )
ui.launch()