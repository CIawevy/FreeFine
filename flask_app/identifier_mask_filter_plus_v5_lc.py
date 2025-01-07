import json
import argparse
import threading
import base64
import random
from flask import Flask, render_template, request, redirect, url_for, send_file, session,flash,get_flashed_messages
import os.path as osp
import secrets
from flask_session import Session  # 新增
from flask import g  # 新增
import  time
from typing import  List
from io import BytesIO
import os
import re
from tqdm import tqdm
import requests
import time
from PIL import Image, ImageOps
from PIL import Image, ImageEnhance
import hashlib
import uuid
# 假设脚本内部定义的密码
correct_password = 'ICCV2025'
# youdao_url = 'https://openapi.youdao.com/api'  # 有道api地址
# app_key = "uxrIpJUaxprcn4tvQBUFMSlMXRbzOfnI"  # 应用密钥
# app_id = "746c5a9d6c19a99d"  # 应用id 有道免费10元额度
# import requests
# import json

# def translate_youdao(translate_text):
#     # 翻译文本生成sign前进行:的处理
#     input_text = ""
#
#     # 当文本长度小于等于20时，取文本
#     if (len(translate_text) <= 20):
#         input_text = translate_text
#
#     # 当文本长度大于20时，进行特殊处理
#     elif (len(translate_text) > 20):
#         input_text = translate_text[:10] + str(len(translate_text)) + translate_text[-10:]
#
#     time_curtime = int(time.time())  # 秒级时间戳获取
#
#     uu_id = uuid.uuid4()  # 随机生成的uuid数，为了每次都生成一个不重复的数。
#     sign = hashlib.sha256(
#         (app_id + input_text + str(uu_id) + str(time_curtime) + app_key).encode('utf-8')).hexdigest()  # sign生成
#
#     data = {
#         'q': translate_text,  # 翻译文本
#         'from': "en",  # 源语言
#         'to': "zh-CHS",  # 翻译语言
#         'appKey': app_id,  # 应用id
#         'salt': uu_id,  # 随机生产的uuid码
#         'sign': sign,  # 签名
#         'signType': "v3",  # 签名类型，固定值
#         'curtime': time_curtime,  # 秒级时间戳
#     }
#
#     r = requests.get(youdao_url, params=data).json()  # 获取返回的json()内容
#     return r["translation"][0]
API_URL = "http://www.trans-home.com/api/index/translate"
TOKEN = "szjsaFIeoBUgoJ1yslzq" # 请替换为实际的 API token

def translate(keywords, target_language):
    """
    调用第三方翻译 API 进行文本翻译

    :param keywords: 需要翻译的文本
    :param target_language: 目标语言代码，如 'en' (英语), 'de' (德语) 等
    :return: 翻译后的文本或错误信息
    """
    # 构建请求的完整 URL
    url = f"{API_URL}?token={TOKEN}"

    # 构建请求数据
    payload = json.dumps({
        "keywords": keywords,          # 需要翻译的文本
        "targetLanguage": target_language  # 目标语言
    })

    # 请求头设置
    headers = {
        'Content-Type': 'application/json'  # 设置请求数据格式为 JSON
    }

    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=payload)

    # 如果请求成功，处理返回的 JSON 数据
    if response.status_code == 200:
        result = response.json()  # 解析返回的 JSON 数据


        # 判断翻译是否成功
        if result.get("code") == 1:
            # 翻译成功，返回翻译后的文本
            return result["data"]["text"]
        else:
            # 翻译失败，返回错误信息
            return f"Error: {result.get('info')}"
    else:
        # 请求失败，返回 HTTP 错误状态码
        return f"Request failed with status code: {response.status_code}"
global_data = {}
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_PERMANENT=True,
    SESSION_FILE_DIR='/data/Hszhu/dataset/flask_session/',
    SESSION_USE_SIGNER=True,
    SECRET_KEY='your_secret_key_here',
)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)
# @app.template_filter('b64encode')
# def b64encode_filter(image_bytes):
#     """
#     Custom Jinja filter to base64 encode image bytes.
#     :param image_bytes: BytesIO object or image data
#     :return: Base64 encoded string
#     """
#     return base64.b64encode(image_bytes.getvalue()).decode('utf-8')
@app.template_filter('random_color')
def random_color_filter(index):
    colors = [
        '#FF5733',  # 鲜红橙色
        # '#33FF57',  # 鲜绿色（过亮，难以辨识）
        '#3357FF',  # 中等蓝色
        # '#F1C40F',  # 金黄色
        '#9B59B6',  # 深紫色
        '#1ABC9C',  # 青绿色
        '#E74C3C',  # 鲜红色
        '#34495E',  # 深蓝灰色
        '#2ECC71',  # 草绿色
        '#3498DB',  # 浅蓝色
        '#8E44AD',  # 紫罗兰色
        '#27AE60',  # 深绿色
        '#E67E22',  # 橙色
        '#E84393',  # 粉红色
    ]
    return colors[index % len(colors)]



def get_user_info(username):
    """动态加载用户信息"""
    user_dir = osp.join(os.environ.get("DATASET_DIR"), 'users', username)  # 用户文件夹路径
    user_info_path = osp.join(user_dir, 'user_info.json')  # 用户数据文件路径

    # 如果文件夹不存在，则初始化
    if not os.path.exists(user_dir):
        os.makedirs(user_dir, exist_ok=True)
        user_info = {
            'selected_subset': None,
            'contributions': {},
            'start_progress': 0,
        }
        print(f'initializing user info for {username}')
        save_user_info(username, user_info)  # 保存初始用户信息
    else:
        with open(user_info_path, 'r') as f:
            user_info = json.load(f)


    return user_info
@app.route('/file/<path:filename>')
def serve_image(filename):
    file_path = filename
    if not file_path.startswith('/'):  # 如果路径不是以斜杠开头，手动添加斜杠
        file_path = '/' + file_path

    print(f"Trying to load file: {file_path}")
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "File not found", 404


def save_user_info(username, user_info):
    """保存用户信息到用户文件夹"""
    user_dir = osp.join(os.environ.get("DATASET_DIR"), 'users', username)  # 用户文件夹路径
    user_info_path = osp.join(user_dir, 'user_info.json')  # 用户数据文件路径

    # 保存用户信息到文件
    with open(user_info_path, 'w') as f:
        json.dump(user_info, f, indent=4)

def crop_to_content(image):
    """
    自动裁剪图片的白边。
    :param image: PIL.Image 对象
    :return: 裁剪后的 PIL.Image 对象
    """
    # 根据图像模式设置背景颜色
    if image.mode == "RGB":
        bg_color = (255, 255, 255)
    elif image.mode == "L":
        bg_color = 255
    elif image.mode == "1":
        bg_color = True
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")

    # 去掉明显的白边
    bg = Image.new(image.mode, image.size, bg_color)
    diff = ImageOps.invert(ImageOps.autocontrast(image)).convert("L")
    bbox = diff.getbbox()
    return image.crop(bbox) if bbox else image

def resize_to_match(image, target_size):
    """
    将图片调整为目标大小。
    :param image: PIL.Image 对象
    :param target_size: 目标尺寸 (宽, 高)
    :return: 调整后的 PIL.Image 对象
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)
def image_to_bytes(image):
    """
    将 PIL 图像转换为字节流。
    :param image: PIL.Image 对象
    :return: 字节流
    """
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')  # 保存为 PNG 格式
    img_byte_arr.seek(0)  # 返回到流的开始位置
    return img_byte_arr
def initialize_ori_data_stat(data, ori_data_stat_path, num_dict):
    """
    初始化或加载 ori_data_stat 的逻辑。

    :param data: 原始子集数据
    :param ori_data_stat_path: 保存 ori_data_stat 的文件路径
    :param num_dict: 每张图片掩码数量的字典
    :return: 初始化或加载的 ori_data_stat 字典
    """
    # 计算总 mask 数量，仅包含有效的 instances
    total_mask_results = sum(
        sum(level_masks.values()) for level_masks in num_dict.values()
    )

    # 加载状态数据
    if os.path.exists(ori_data_stat_path):
        # 如果文件存在，加载状态数据
        ori_data_stat = load_json_data(ori_data_stat_path)
    else:
        # 如果文件不存在，初始化状态数据
        ori_data_stat = {
            da_n: {
                'status': 'unprocessed',
                'cur_level': None,
                'max_level': None,
                'selected_masks': {},
                'processed_masks':{},
            } for da_n in data.keys()
        }
        ori_data_stat['total_mask_results'] = total_mask_results  # 总 mask 数量
        ori_data_stat['processed_mask_results'] = 0  # 已处理 mask 数量
        # 保存初始化的状态数据
        save_json_data(ori_data_stat, ori_data_stat_path)

    return ori_data_stat
def load_or_initialize_global_data(root_dir):
    """加载或初始化全局变量"""
    global global_data

    # 过滤只保留以 Subset_ 开头的子集目录
    subset_dirs = sorted(
        [d for d in os.listdir(root_dir) if d.startswith("Subset_") and os.path.isdir(os.path.join(root_dir, d))],
        key=lambda x: int(x.split("_")[1])  # 提取数字部分进行排序
    )

    # 初始化进度条
    with tqdm(total=len(subset_dirs), desc="Initializing datasets...") as pbar:
        for subset_dir in subset_dirs:
            subset_id = subset_dir.split("_")[1]
            subset_path = os.path.join(root_dir, subset_dir)

            # 如果 global_data 中没有此子集，加载或初始化它
            if subset_id not in global_data:
                global_data[subset_id] = initialize_subset_data(subset_path, subset_id)

            # 每完成一个子集，更新进度条
            pbar.update(1)


def save_global_data_periodically(root_dir, interval=300):
    """定期保存全局变量到磁盘"""
    global global_data
    global_data_path = os.path.join(root_dir, 'global_data.json')

    def save_loop():
        while True:
            with open(global_data_path, 'w') as f:
                json.dump(global_data, f, indent=4)
            time.sleep(interval)

    threading.Thread(target=save_loop, daemon=True).start()

# 加载或创建掩码数据的函数
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
# 加载数据的通用函数：如果文件不存在则创建空字典并保存
def load_or_create_json(path):
    if os.path.exists(path):
        return load_json_data(path)
    else:
        # 文件不存在时，初始化为空字典
        save_json_data({}, path)
        return {}
# 保存筛选后的数据
def save_json_data(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


# # 重新定义 undo_stack 的最大长度
# MAX_UNDO_STACK_SIZE = 100000000
def load_or_create_undo_stack(file_path):
    """加载或创建撤销栈"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                undo_stack = json.load(f)
                # 如果不是列表，返回空列表
                if isinstance(undo_stack, list):
                    return undo_stack
                else:
                    print(f"Warning: Undo stack at {file_path} is not a list. Initializing empty stack.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading undo stack from {file_path}: {e}. Initializing empty stack.")
    # 如果文件不存在或读取失败，创建空列表并保存
    save_undo_stack([], file_path)
    return []
def initialize_subset_data(subset_path, subset_id):
    """
    初始化或加载子集的数据。

    :param subset_path: 子集目录路径
    :param subset_id: 子集的唯一标识
    :return: 初始化的子集数据结构
    """
    # 动态加载或创建数据文件
    data_path = os.path.join(subset_path, f"mask_tag_relabelled_lmm_v2_{subset_id}.json")
    ori_data_stat_path = os.path.join(subset_path, "mask_label_filter_stat.json")
    undo_stack_path = os.path.join(subset_path, "mask_label_filter_undo_stack.json")
    new_data_path = os.path.join(subset_path, f"mask_label_filtered_{subset_id}.json")

    # 加载或创建 JSON 数据
    data = load_or_create_json(data_path)
    new_data = load_or_create_json(new_data_path)

    # 动态生成 num_dict
    num_dict = get_num_dict(data)

    # 初始化或加载 ori_data_stat
    ori_data_stat = initialize_ori_data_stat(data, ori_data_stat_path, num_dict)

    # 加载或创建撤销栈
    undo_stack = load_or_create_undo_stack(undo_stack_path)

    # 返回完整的子集数据结构
    return {
        "data": data,
        "ori_data_stat": ori_data_stat,
        "undo_stack": undo_stack,
        "new_data": new_data,
        "num_dict": num_dict,
        "paths": {
            "data": data_path,
            "ori_data_stat": ori_data_stat_path,
            "undo_stack": undo_stack_path,
            "new_data": new_data_path,
        }
    }
def get_num_dict(data):
    """根据子集数据生成 num_dict"""
    return {
        da_n: {
            inst['level']: len(inst['mask_path']) for inst in da.get('instances', [])
        } for da_n, da in data.items()
    }
# 定义保存和加载 undo_stack 的方法
def save_undo_stack(undo_stack, undo_stack_path):
    # if len(undo_stack) > MAX_UNDO_STACK_SIZE:
    #     undo_stack = undo_stack[-MAX_UNDO_STACK_SIZE:]  # 保留最近的10次操作
    with open(undo_stack_path, 'w') as f:
        json.dump(undo_stack, f, indent=4)

def load_undo_stack(undo_stack_path):
    if os.path.exists(undo_stack_path):
        with open(undo_stack_path, 'r') as f:
            undo_stack = json.load(f)
            return undo_stack if isinstance(undo_stack, list) else []
    return []








# 用户登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 验证密码
        if password == correct_password:
            user_info = get_user_info(username)  # 动态加载用户信息
            return redirect(url_for('select_subset', username=username))  # 跳转到选择子集页面
        else:
            flash('密码错误，请重新输入！')  # 显示密码错误的提示

    return render_template('login.html',
                           error_message='密码错误，请重新输入！' if '密码错误' in get_flashed_messages() else None)

def update_and_collect_user_contributions(progress_stat):
    """
    更新所有用户的贡献并返回聚合的贡献数据。
    :param progress_stat: 当前的进度状态
    :return: contributions_data 聚合后的贡献数据
    """
    contributions_data = {}  # 用于存储所有用户的贡献
    user_base_dir = osp.join(os.environ.get("DATASET_DIR"), 'users')  # 用户文件夹根目录

    if not os.path.exists(user_base_dir):
        print("No user data found!")
        return contributions_data

    for user_folder in os.listdir(user_base_dir):
        user_dir = osp.join(user_base_dir, user_folder)
        if not os.path.isdir(user_dir):
            continue  # 跳过非文件夹项

        user_info_path = osp.join(user_dir, 'user_info.json')
        if not os.path.exists(user_info_path):
            continue

        with open(user_info_path, 'r') as f:
            user_info = json.load(f)

        # 如果用户选择了子集，则更新贡献
        if user_info.get('selected_subset') is not None:
            subset_id = user_info['selected_subset']
            cur_progress = progress_stat[subset_id]['progress']
            start_progress = user_info.get('start_progress', 0)
            contribution = cur_progress - start_progress

            if contribution > 0:
                user_info['contributions'][subset_id] = (
                    user_info['contributions'].get(subset_id, 0) + contribution
                )
                user_info['start_progress'] = cur_progress

            # 保存更新后的用户信息
            with open(user_info_path, 'w') as f:
                json.dump(user_info, f, indent=4)

        # 聚合当前用户的贡献数据
        for subset_id, contribution in user_info.get('contributions', {}).items():
            if subset_id not in contributions_data:
                contributions_data[subset_id] = {}
            contributions_data[subset_id][user_folder] = contribution

    return contributions_data
# 选择子集页面
# 选择子集页面
# 修改 /select_subset/<username> 路由
@app.route('/select_subset/<username>', methods=['GET', 'POST'])
def select_subset(username):
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'logout':
            return redirect(url_for('logout', username=username))  # 重定向到登录页面
        progress_stat = get_progress_stat()  # 动态加载 progress_stat
        user_info = get_user_info(username)
        subset_id = request.form.get('subset_id')
        subset_id = str(subset_id)  # 确保 subset_id 是字符串
        if progress_stat[subset_id]['user'] != 'unselected':
            return redirect(url_for('select_subset', username=username))
        else:
            user_info['selected_subset'] = subset_id  # 更新用户选择的子集
            progress_stat[subset_id]['user'] = username  # 更新子集的用户信息
            user_info['start_progress'] = progress_stat[subset_id]['progress']  # 记录当前进度
            save_user_info(username, user_info)  # 保存用户信息
            save_progress_stat(progress_stat)  # 保存 progress_stat 到文件
            return redirect(url_for('inner_init', username=username, subset_id=subset_id))  # 选择子集后跳转到主页面

    progress_stat = get_progress_stat()  # 动态加载 progress_stat
    user_info = get_user_info(username)
    # 先赞停当前用户选择的子集并收集贡献，避免重复统计
    #每个子集同时只能被一个人选择或者不选择，每个人当前选择也只能有一个子集
    if user_info.get('selected_subset') is not None:
        subset_id = user_info['selected_subset']
        cur_progress = progress_stat[subset_id]['progress']
        start_progress = user_info.get('start_progress', 0)
        contribution = cur_progress - start_progress

        if contribution > 0:
            if subset_id not in user_info['contributions']:
                user_info['contributions'][subset_id] = contribution
            else:
                user_info['contributions'][subset_id] += contribution

        # 更新 start_progress 防止重复计算
        user_info['start_progress'] = cur_progress

    # 清除当前用户的子集选择
    user_info['selected_subset'] = None
    save_user_info(username, user_info)  # 保存用户信息
    # 更新并收集所有用户的贡献数据
    contributions_data = update_and_collect_user_contributions(progress_stat)
    print(contributions_data)
    #先收集 再暂时unselected
    for subset_id, data in progress_stat.items():
        if data['user'] == username:
            progress_stat[subset_id]['user'] = 'unselected'  # 重置为未选中状态
    save_progress_stat(progress_stat)  # 保存 progress_stat 到文件

    # 构造子集的显示信息
    subsets = []
    for subset_id, data in progress_stat.items():
        disable = data['user'] != 'unselected' and username != data['user'] or data['status'] == 'completed'
        subsets.append({
            'subset_id': subset_id,
            'progress': data['progress'],
            'status': data['status'],
            'user': data['user'],
            'is_disabled': disable
        })

    # 从 contributions_data 构建用户索引信息 分配可视化颜色
    user_indices = {}
    for subset_id, user_contributions in contributions_data.items():
        for user in user_contributions.keys():
            if user not in user_indices:
                user_indices[user] = len(user_indices)

    return render_template(
        'select_subset.html',
        progress_stat=subsets,
        username=username,
        contributions_data=contributions_data,
        user_indices=user_indices  # 传递用户索引
    )

# 退出登录
@app.route('/logout/<username>', methods=['GET'])
def logout(username):
    progress_stat = get_progress_stat()  # 动态加载进度状态
    user_info = get_user_info(username)  # 动态加载当前用户信息

    # 更新 progress_stat，释放子集
    for subset_id, data in progress_stat.items():
        if data['user'] == username:
            data['user'] = 'unselected'  # 重置为未选中状态

    # 保存 progress_stat
    save_progress_stat(progress_stat)

    # 统计贡献并更新用户信息
    if user_info.get('selected_subset') is not None:
        subset_id = user_info['selected_subset']
        cur_progress = progress_stat[subset_id]['progress']
        start_progress = user_info.get('start_progress', 0)
        contribution = cur_progress - start_progress

        if contribution > 0:
            if subset_id not in user_info['contributions']:
                user_info['contributions'][subset_id] = contribution
            else:
                user_info['contributions'][subset_id] += contribution

        # 更新 start_progress 防止重复计算
        user_info['start_progress'] = cur_progress

    # 清除当前用户的子集选择
    user_info['selected_subset'] = None

    # 保存更新后的用户信息
    print(f'save user info for {username}')
    save_user_info(username, user_info)

    return redirect(url_for('login'))  # 返回登录页面



def apply_random_color_to_mask(mask_img, color_index=0):
    """
    根据提供的颜色索引为掩码区域着色。
    """
    def random_color_filter(index):
        colors = [
            '#FF5733',  # 鲜红橙色
            '#3357FF',  # 中等蓝色
            '#9B59B6',  # 深紫色
            '#1ABC9C',  # 青绿色
            '#E74C3C',  # 鲜红色
            '#34495E',  # 深蓝灰色
            '#2ECC71',  # 草绿色
            '#3498DB',  # 浅蓝色
            '#8E44AD',  # 紫罗兰色
            '#27AE60',  # 深绿色
            '#E67E22',  # 橙色
            '#E84393',  # 粉红色
        ]
        return colors[index % len(colors)]

    def hex_to_rgba(hex_color, alpha=150):
        hex_color = hex_color.lstrip('#')
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
            alpha
        )

    # 获取掩码数据
    mask_data = mask_img.getdata()
    selected_color = hex_to_rgba(random_color_filter(color_index), alpha=150)

    new_mask_data = [
        selected_color if value > 0 else (0, 0, 0, 0)  # 非零区域使用颜色
        for value in mask_data
    ]
    mask_img_colored = Image.new("RGBA", mask_img.size)
    mask_img_colored.putdata(new_mask_data)
    return mask_img_colored

def combine_with_transparent_background(original_img, mask_img_colored, alpha=0.5):
    """
    将原图透明化后，与着色的掩码进行叠加。
    """
    # 降低原图亮度/透明度
    transparent_bg = Image.new("RGBA", original_img.size, (255, 255, 255, int(255 * (1 - alpha))))
    original_transparent = Image.alpha_composite(transparent_bg, original_img)

    # 将掩码叠加到透明化的原图上
    combined_img = Image.alpha_composite(original_transparent, mask_img_colored)
    return combined_img

def normalize_labels(label_str):
    """
    将翻译结果中的所有可能的分隔符（如逗号、顿号等）统一转换为一个标准分隔符（如|），
    并返回标准化后的字符串。
    """
    # 将逗号、顿号和其他可能的分隔符统一替换为 |
    label_str = re.sub(r'[, ，、；：\n]', '|', label_str)  # 可根据实际需要调整正则表达式
    return label_str
def filter_instance_label(data, ori_data_stat, da_n,level,mask_id):
    # 获取原图路径和用户选择的掩码信息
    src_img_path = data[da_n]['src_img_path']
    selected_masks = ori_data_stat[da_n]["selected_masks"]

    mask_path = data[da_n]['instances'][int(level) - 1]['mask_path'][int(mask_id)]
    ori_label = data[da_n]['instances'][int(level) - 1]['obj_label'][int(mask_id)]
    if not isinstance(ori_label, List):
        ori_label = [ori_label]
    # 将 ori_label 转换为英文逗号分隔的字符串
    label_str = ",".join(ori_label)
    try:
        chinese_label_str = translate(label_str, 'zh-CHS')  # 翻译逗号分隔的字符串
        # print(chinese_label_str)
        # 将翻译后的字符串按逗号分割，恢复为列表
        # 将翻译后的字符串中的所有分隔符统一替换为标准符号（例如|）
        chinese_label_str = normalize_labels(chinese_label_str)

        # 根据标准符号分割翻译结果
        chinese_labels = chinese_label_str.split('|')
        # print(chinese_labels)
    except (requests.exceptions.RequestException, ValueError) as e:
        chinese_labels = ["API ERROR"] * len(ori_label)
        print(f"Translation error: {e}")
    # 获取掩码图像
    replace_name = os.environ.get("DATASET_DIR")
    vis_mask_path = mask_path.replace(f'{replace_name}', f'{replace_name}vis_temp/')


    # 返回数据
    return {
        'ori_img':  src_img_path,  # 原图
        'crop_img': vis_mask_path,  # 所有掩码图像
        'label': ori_label,  #label
        'chinese_labels':chinese_labels
    }

def correct_path(path):
    # 需要将文件路径中的原始前缀替换成新的 URL 前缀
    original_path_prefix = '/data/Hszhu/dataset/'
    new_url_prefix = 'http://vlrlab-monkey.xyz:7684/img/'

    # 如果文件路径包含原来的前缀，就进行替换
    if path.startswith(original_path_prefix):
        path = path.replace(original_path_prefix, new_url_prefix, 1)
    return path
def filter_instance(data, da_n, level):
    # 获取原图路径和实例信息
    instances = data[da_n]['instances'][level - 1]
    replace_name = os.environ.get("DATASET_DIR")
    # 保存所有掩码图像
    crop_images = []

    for mask_id, mask_path in enumerate(instances['mask_path']):
        vis_mask_path = mask_path.replace(f'{replace_name}', f'{replace_name}vis_temp/')
        crop_images.append(vis_mask_path)  # 只需要保存裁剪后的合成图像

    # 返回数据
    return {
        'ori_img': data[da_n]['src_img_path'],  # 原图路径
        'crop_images': crop_images,  # 所有掩码图像路径(处理后)
    }

from flask import Flask, send_file
import os




@app.route('/inner_select/<username>/<subset_id>', methods=['GET', 'POST'])
def inner_select(username, subset_id):
    global global_data

    # 获取子集数据
    subset_data = global_data[subset_id]
    data = subset_data['data']
    ori_data_stat = subset_data['ori_data_stat']

    # 遍历子集中的图片
    for da_n, da in data.items():
        # 跳过没有有效 instances 的图片
        if 'instances' not in da or ori_data_stat[da_n]['status'] == 'completed':
            continue
        return redirect(url_for('inner_select_single',username=username,subset_id=subset_id,da_n=da_n))

    # 如果所有图片均已完成，更新全局进度
    progress_stat = get_progress_stat()
    progress_stat[subset_id]['status'] = 'completed'
    save_progress_stat(progress_stat)

    return redirect(url_for('select_subset', username=username))
@app.route('/inner_select_single/<username>/<subset_id>/<da_n>', methods=['GET', 'POST'])
def inner_select_single(username, subset_id,da_n):
    global global_data

    # 获取子集数据
    subset_data = global_data[subset_id]
    ori_data_stat = subset_data['ori_data_stat']
    stat = ori_data_stat[da_n]['status']
    print(f'come to inner_select current stat:{stat}')
    # print(ori_data_stat[da_n]['processed_masks'])
    # 判断是否进入 level 选择
    if stat == 'unprocessed':
        return redirect(url_for('select_level', username=username, subset_id=subset_id, da_n=da_n))
    # 判断是否进入 mask 选择
    if stat == 'level':
        # 继续处理 mask
        return redirect(url_for('select_mask', username=username, subset_id=subset_id, da_n=da_n, level=ori_data_stat[da_n]['cur_level']))
    if stat == 'mask':
        for lvl , mask_ids in ori_data_stat[da_n]['selected_masks'].items():
            for mask_id in mask_ids: #str '1' in selected masks
                mask_id = int(mask_id) - 1 #0 in processed masks
                if lvl not in ori_data_stat[da_n]['processed_masks']:
                    ori_data_stat[da_n]['processed_masks'][str(lvl)] = []
                if mask_id in ori_data_stat[da_n]['processed_masks'][str(lvl)]:
                    continue
                return redirect(url_for('select_label', username=username, subset_id=subset_id, da_n=da_n,level=lvl,mask_id=mask_id))
    return redirect(url_for('inner_select', username=username, subset_id=subset_id))



@app.route('/inner_init/<username>/<subset_id>', methods=['GET', 'POST'])
def inner_init(username, subset_id):
    global global_data
    dataset_dir = os.environ.get("DATASET_DIR")
    directory = os.path.join(dataset_dir, f'Subset_{subset_id}')

    if subset_id not in global_data:
        # 加载或初始化子集数据
        json_path = os.path.join(directory, f"mask_tag_relabelled_lmm_v2_{subset_id}.json")
        new_data_path = os.path.join(directory, f'mask_label_filtered_{subset_id}.json')
        undo_stack_path = os.path.join(directory, 'mask_label_filter_undo_stack.json')
        ori_data_stat_path = os.path.join(directory, 'mask_label_filter_stat.json')
        data = load_json_data(json_path)
        num_dict = {
            da_n: {
                inst['level']: len(inst['mask_path']) for inst in da['instances']
            } if 'instances' in da else {}
            for da_n, da in data.items()
        }
        total_mask_results = sum(sum(level_masks.values()) for level_masks in num_dict.values())

        # 加载或初始化状态
        if os.path.exists(ori_data_stat_path):
            ori_data_stat = load_json_data(ori_data_stat_path)
        else:
            ori_data_stat = {
                da_n: {
                    'status': 'unprocessed',
                    'cur_level': None,
                    'max_level':None,
                    'selected_masks':{},
                    'processed_masks':{}
                } for da_n in data.keys()
            }
            ori_data_stat['total_mask_results'] = total_mask_results
            ori_data_stat['processed_mask_results'] = 0
            save_json_data(ori_data_stat, ori_data_stat_path)

        # 加载 undo_stack 和 new_data
        undo_stack = load_undo_stack(undo_stack_path)
        new_data = load_or_create_json(new_data_path)

        # 存储到全局变量
        global_data[subset_id] = {
            'data': data,
            'num_dict': num_dict,
            'ori_data_stat': ori_data_stat,
            'undo_stack': undo_stack,
            'new_data': new_data,
            'paths': {
                'ori_data_stat': ori_data_stat_path,
                'undo_stack': undo_stack_path,
                'new_data': new_data_path,
            }
        }

    return redirect(url_for('inner_select', username=username, subset_id=subset_id))
@app.route('/')
def index():
    return redirect(url_for('login'))
    # 遍历数据中的图片


@app.route('/select_level/<username>/<subset_id>/<da_n>', methods=['GET', 'POST'])
def select_level(username, subset_id, da_n):
    global global_data
    if request.method == 'POST':
        # 获取子集数据
        subset_data = global_data[subset_id]
        ori_data_stat = subset_data['ori_data_stat']
        undo_stack = subset_data['undo_stack']
        num_dict = subset_data['num_dict']
        new_data = subset_data['new_data']
        action = request.form.get('action')  # 获取用户的动作
        selected_level = request.form.get('level')  # 获取用户选择的分割级别

        # 处理不同的动作
        if action == 'skip_image':
            # 跳过整张图片
            total_edits_in_image = sum(num_dict[da_n].values())
            ori_data_stat[da_n]['status'] = 'completed'
            ori_data_stat['processed_mask_results'] += total_edits_in_image
            undo_stack.append(('skip_image', da_n, total_edits_in_image))

        elif action == 'undo':
            # 撤销上一步操作
            if undo_stack:
                last_action = undo_stack.pop()
                action_type, undo_da_n, *details = last_action
                print(action_type)
                if action_type == 'skip_image':
                    # 撤回跳过图片的操作
                    ori_data_stat[undo_da_n]['status'] = 'unprocessed' if action_type == 'skip_image' else 'selected'
                    ori_data_stat['processed_mask_results'] -= details[0]
                elif action_type == 'keep' or action_type =='skip':
                    ori_data_stat[undo_da_n]['status'] = 'mask'
                    level = details[0] #str
                    if action_type=='keep':
                        new_data[undo_da_n]['instances']['mask_path'].pop()
                    ori_data_stat[undo_da_n]['processed_masks'][level].pop()
                    ori_data_stat['processed_mask_results'] -= 1
                elif action_type =='discard-all':
                    undo_level,undo_process_num = details[0],details[1]
                    ori_data_stat[undo_da_n]['status'] = 'level'
                    ori_data_stat['processed_mask_results'] -= undo_process_num
                    ori_data_stat[da_n]['cur_level'] = undo_level


            return redirect(url_for('inner_select_single', username=username, subset_id=subset_id, da_n=undo_da_n))

        elif action == 'logout':
            # 登出用户并重定向到登录页面
            return redirect(url_for('logout', username=username))  # 重定向到登录页面

        elif action == 'go_to_main':
            # 在返回主页时先更新所有用户的贡献
            return redirect(url_for('select_subset',username=username))

        elif selected_level:
            # 用户选择了一个分割级别
            selected_level = int(selected_level)
            ori_data_stat[da_n]['status'] = 'level'
            ori_data_stat[da_n]['cur_level'] = selected_level


        # 保存状态
        save_json_data(ori_data_stat, subset_data['paths']['ori_data_stat'])
        save_undo_stack(undo_stack, subset_data['paths']['undo_stack'])

        # 重定向到内部处理逻辑页面
        return redirect(url_for('inner_select_single', username=username, subset_id=subset_id,da_n=da_n))
    subset_data = global_data[subset_id]
    data = subset_data['data']
    ori_data_stat = subset_data['ori_data_stat']
    undo_stack = subset_data['undo_stack']
    num_dict = subset_data['num_dict']
    # 准备渲染数据
    directory = subset_data['paths']['ori_data_stat'].rsplit('/', 1)[0]
    level_images = {
        level: osp.join(directory, f'masks_tag/level{level}/{da_n}/anotated_img.png')
        for level in num_dict[da_n].keys()
    }
    level_images['Original'] = data[da_n]['src_img_path']

    ori_data_stat[da_n]['max_level'] = len(level_images)-1
    # 计算进度百分比
    progress_stat = get_progress_stat()
    progress = "{:.2f}".format(
        (ori_data_stat['processed_mask_results'] / ori_data_stat['total_mask_results']) * 100
    )
    progress_stat[subset_id]['progress'] = float(progress)
    save_progress_stat(progress_stat)
    save_json_data(ori_data_stat, subset_data['paths']['ori_data_stat'])
    # 检查是否禁用撤回按钮
    disable_undo = len(undo_stack) == 0

    # 渲染模板
    return render_template(
        'level_selection_plus_fast_local.html',
        da_n=da_n,
        subset_id=subset_id,
        username=username,
        level_images=level_images,  # 图像路径字典传入
        progress=progress,
        undo_stack=undo_stack[-20:],  # 只传递后 N 条记录
        disable_undo=disable_undo
    )

@app.route('/select_mask/<username>/<subset_id>/<da_n>/<level>', methods=['GET', 'POST'])
def select_mask(username, subset_id, da_n, level):
    global global_data
    if request.method == 'POST':
        # 获取子集数据
        subset_data = global_data[subset_id]
        ori_data_stat = subset_data['ori_data_stat']
        undo_stack = subset_data['undo_stack']

        selected_mask_ids = request.form.get('selected_mask_ids')
        print(f'selected_mask_ids:{selected_mask_ids}')
        selected_mask_ids = json.loads(selected_mask_ids)  # 解析JSON



        # 更新整个 selected_masks 字典
        ori_data_stat[da_n]['selected_masks'] = selected_mask_ids

        # 获取决策按钮的值
        decision = request.form.get('decision', '')
        if decision == 'Next':
            selected_num = sum(len(v) for k,v in selected_mask_ids.items())
            process_num = sum(subset_data['num_dict'][da_n].values()) - selected_num
            ori_data_stat['processed_mask_results'] += process_num
            if selected_num >0:#move on to label filter
                ori_data_stat[da_n]['status'] = 'mask'
            else: #skip to next case but allow undo
                ori_data_stat[da_n]['status'] = 'completed'
                undo_stack.append(('discard-all',da_n,level,process_num))

        elif decision == 'Back':
            ori_data_stat[da_n]['status'] = 'unprocessed'

        save_json_data(ori_data_stat, subset_data['paths']['ori_data_stat'])
        return redirect(url_for('inner_select_single', username=username, subset_id=subset_id,da_n=da_n))

    # 获取子集数据渲染网页
    subset_data = global_data[subset_id]
    data = subset_data['data']
    ori_data_stat = subset_data['ori_data_stat']


    progress_stat = get_progress_stat()
    progress = "{:.2f}".format(
        (ori_data_stat['processed_mask_results'] / ori_data_stat['total_mask_results']) * 100
    )
    progress_stat[subset_id]['progress'] = float(progress)
    save_progress_stat(progress_stat)
    level = int(level)
    # 获取当前level的所有掩码信息
    instance_info = filter_instance(data, da_n, level)
    print(f"max_Level:{ori_data_stat[da_n]['max_level']}")
    if len(ori_data_stat[da_n]['selected_masks']) ==0:
        for lvl in range(1,ori_data_stat[da_n]['max_level']+1):
            ori_data_stat[da_n]['selected_masks'][str(lvl)] = []
    print(f"init selection:{ori_data_stat[da_n]['selected_masks']}")
    save_json_data(ori_data_stat, subset_data['paths']['ori_data_stat'])
    # 渲染模板并传递数据
    return render_template(
         'mask_filter_flatten_fast_local.html',
        username=username,
        subset_id=subset_id,
        ori_img=instance_info['ori_img'],
        crop_images=instance_info['crop_images'],
        progress=progress,
        selected_masks=ori_data_stat[da_n]['selected_masks'],
        level=level,  # 将当前 level 传递给前端
    )


@app.route('/filter/<username>/<subset_id>/<da_n>/<level>/<mask_id>', methods=['GET', 'POST'])
def select_label(username, subset_id, da_n, level,mask_id):
    global global_data

    if request.method == 'POST':
        # 获取子集数据
        subset_data = global_data[subset_id]
        data = subset_data['data']
        ori_data_stat = subset_data['ori_data_stat']
        undo_stack = subset_data['undo_stack']
        new_data = subset_data['new_data']
        mask_id = int(mask_id)
        decision = request.form['decision']

        # 获取最终标签
        if decision == 'keep':
            # 获取用户选择的标签ID（这个值是从前端传递过来的）
            selected_label_id = int(request.form['selected_label_id'])  # 获取选择的标签ID
            print(selected_label_id)
            if da_n not in new_data:
                new_data[da_n] = {'instances': {'mask_path': [], 'obj_label': []}}
            # 保留
            new_data[da_n]['instances']['mask_path'].append(
                data[da_n]['instances'][int(level) - 1]['mask_path'][mask_id]
            )
            print(data[da_n]['instances'][int(level) - 1]['obj_label'][mask_id][selected_label_id])
            new_data[da_n]['instances']['obj_label'].append(
                data[da_n]['instances'][int(level) - 1]['obj_label'][mask_id][selected_label_id]
            )

            # 更新处理状态
            print(level)
            print(ori_data_stat[da_n]['processed_masks'])
            ori_data_stat[da_n]['processed_masks'][level].append(mask_id)
            ori_data_stat['processed_mask_results'] += 1
            # print(ori_data_stat[da_n]['selected_masks'])
            # print(ori_data_stat[da_n]['processed_masks'])
            if sum(len(v) for k,v in ori_data_stat[da_n]['processed_masks'].items()) == sum(len(v) for k, v in ori_data_stat[da_n]['selected_masks'].items()):
                ori_data_stat[da_n]['status'] = 'completed'
            undo_stack.append(('keep',da_n,level,mask_id))

        elif decision == 'skip':
            # 跳过
            ori_data_stat[da_n]['processed_masks'][level].append(mask_id)
            ori_data_stat['processed_mask_results'] += 1
            if sum(len(v) for k,v in ori_data_stat[da_n]['processed_masks'].items()) == sum(len(v) for k, v in ori_data_stat[da_n]['selected_masks'].items()):
                ori_data_stat[da_n]['status'] = 'completed'
            undo_stack.append(('skip',da_n,level,mask_id))

        elif decision == 'undo':
            ori_data_stat[da_n]['status'] = 'mask'
            if undo_stack:
                last_action,undo_da_n,undo_level,undo_mask_id = undo_stack.pop()
                action_type = last_action

                if action_type == 'keep':
                    # 撤回“保留”操作
                    new_data[undo_da_n]['instances']['mask_path'].pop()
                    new_data[undo_da_n]['instances']['obj_label'].pop()
                    ori_data_stat[undo_da_n]['processed_masks'][undo_level].pop()
                    ori_data_stat['processed_mask_results'] -= 1

                elif action_type == 'skip':
                    # 撤回“跳过”操作
                    ori_data_stat[undo_da_n]['processed_masks'][undo_level].pop()
                    ori_data_stat['processed_mask_results'] -= 1


        elif decision == 'Back':
            #回溯
            ori_data_stat[da_n]['processed_masks']={}
            if da_n in new_data:
                del new_data[da_n]

            ori_data_stat[da_n]['status'] = 'level'
            # 回溯undo_stack，直到遇到与da_n不同的操作
            label_process_num = 0
            while undo_stack and undo_stack[-1][1] == da_n:
                last_action,undo_da_n,undo_level,undo_process_num = undo_stack.pop()
                if last_action != 'discard-all':
                    label_process_num +=1
                else:
                    label_process_num+=undo_process_num

            ori_data_stat['processed_mask_results'] -= label_process_num

        # 保存更新后的状态
        save_json_data(new_data, subset_data['paths']['new_data'])
        save_json_data(ori_data_stat, subset_data['paths']['ori_data_stat'])
        save_undo_stack(undo_stack, subset_data['paths']['undo_stack'])

        return redirect(url_for('inner_select_single', username=username, subset_id=subset_id,da_n=da_n))

    # 获取子集数据
    subset_data = global_data[subset_id]
    data = subset_data['data']
    ori_data_stat = subset_data['ori_data_stat']
    undo_stack = subset_data['undo_stack']

    level = int(level)
    mask_id = int(mask_id)
    progress_stat = get_progress_stat()
    # 计算进度百分比
    progress = "{:.2f}".format(
        (ori_data_stat['processed_mask_results'] / ori_data_stat['total_mask_results']) * 100
    )
    progress_stat[subset_id]['progress'] = float(progress)
    save_progress_stat(progress_stat)

    # 获取 mask 实例信息
    instance_info = filter_instance_label(data,ori_data_stat, da_n, level, mask_id)
    # 检查是否禁用撤回按钮
    disable_undo = len(undo_stack)== 0 or undo_stack[-1][1] != da_n
    # print(f'undo_stack last:{undo_stack[-1]}')
    # 渲染模板，并传递所需数据
    return render_template(
        'label_select_plus_fast_local.html',
        username=username,
        subset_id=subset_id,
        instance=instance_info,
        progress=progress,
        disable_undo = disable_undo,
        undo_stack=undo_stack[-20:]
    )

def get_progress_stat():
    """从文件动态加载 progress_stat"""
    if not hasattr(g, 'progress_stat'):
        progress_stat_path = osp.join(os.environ.get("DATASET_DIR"), 'progress_stat')  # 从环境变量获取路径
        if os.path.exists(progress_stat_path):
            with open(progress_stat_path, 'r') as f:
                g.progress_stat = json.load(f)
        else:
            # 如果文件不存在，初始化默认值并保存到文件
            g.progress_stat = {str(i): {'progress': 0, 'status': 'unprocessed', 'user': 'unselected'} for i in range(int(os.environ.get("subset_num")))}
            save_json_data(g.progress_stat, progress_stat_path)
    return g.progress_stat

# 保存 progress_stat
def save_progress_stat(progress_stat):
    """保存 progress_stat 到文件"""
    progress_stat_path = osp.join(os.environ.get("DATASET_DIR"), 'progress_stat')
    save_json_data(progress_stat, progress_stat_path)

if __name__ == '__main__':
    # 使用 argparse 传递参数
    parser = argparse.ArgumentParser(description="Flask data filtering application.")
    parser.add_argument('--dir', type=str, required=True, help="Path to the directory containing the JSON file")
    parser.add_argument('--port', type=int, default=8860, help="Port to run the Flask application on (default: 8860)")
    parser.add_argument('--debug', action='store_true', help="Enable Flask debug mode for development")
    args = parser.parse_args()

    # 初始化全局变量
    load_or_initialize_global_data(args.dir)
    if '200K' in args.dir:
        replace_name = 'Subject200K/'
        os.environ["subset_num"] = '4'
    elif 'PIE' in args.dir:
        replace_name = 'PIE-Bench_v1/'
        os.environ["subset_num"] = '4'
    elif 'GRIT' in args.dir:
        replace_name = 'GRIT/'
        os.environ["subset_num"] = '4'
    elif 'SA' in args.dir:
        replace_name = 'SA-1B/'
        os.environ["subset_num"] = '4'
    # 设置环境变量
    os.environ["DATASET_DIR"] = args.dir  # 数据集路径传递为环境变量
    os.environ["REPLACE_NAME"] = replace_name  # 数据集路径传递为环境变量


    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=True)