import os
import cv2
import json
import argparse
from tqdm import tqdm
from flask_app import Flask, render_template, request, redirect, url_for, send_file, session
import os.path as osp
import secrets
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # 生成一个16字节的随机密钥
# 加载或创建掩码数据的函数
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# 保存筛选后的数据
def save_json_data(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

# 重新定义 undo_stack 的最大长度
MAX_UNDO_STACK_SIZE = 10

# 定义保存和加载 undo_stack 的方法
def save_undo_stack(undo_stack, undo_stack_path):
    if len(undo_stack) > MAX_UNDO_STACK_SIZE:
        undo_stack = undo_stack[-MAX_UNDO_STACK_SIZE:]  # 保留最近的10次操作
    with open(undo_stack_path, 'w') as f:
        json.dump(undo_stack, f, indent=4)

def load_undo_stack(undo_stack_path):
    if os.path.exists(undo_stack_path):
        with open(undo_stack_path, 'r') as f:
            undo_stack = json.load(f)
            return undo_stack if isinstance(undo_stack, list) else []
    return []


def filter_instance(data, da_n,mask_id):
    src_img_path = data[da_n]['src_img_path']
    instances = data[da_n]['instances']
    return {
        'ori_img': src_img_path,
        'ori_mask': instances['mask_path'][int(mask_id)],
        'obj_label': instances['obj_label'][int(mask_id)],
    }
# 动态加载图片的路由
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


@app.route('/')
def index():
    # 遍历数据中的图片
    for da_n, da in data.items():
        # 跳过没有有效 instances 的图片
        if 'instances' not in da or 'mask_path' not in da['instances']:
            continue

        # 检查该图片的处理状态
        if ori_data_stat[da_n]['status'] == 'completed':
            continue  # 如果图片已经处理完毕，跳过

        # 找到第一个未处理的 mask
        mask_paths = da['instances']['mask_path']
        processed_masks = [int(mask) for mask in ori_data_stat[da_n].get('processed_masks', [])]  # 转换为整数

        for mask_id, mask_path in enumerate(mask_paths):
            if mask_id not in processed_masks:
                # 返回当前未处理的 mask
                return redirect(url_for('filter', da_n=da_n, mask_id=mask_id))

    # 若所有图片和mask都已处理完毕，删除相关文件
    os.remove(undo_stack_path)
    os.remove(ori_data_stat_path)
    return "All instances processed!"



@app.route('/filter/<da_n>/<mask_id>', methods=['GET', 'POST'])
def filter(da_n, mask_id):
    session.setdefault('disable_skip_image', False)
    if request.method == 'POST':
        decision = request.form['decision']

        if decision == 'keep':
            session['disable_skip_image'] = True
            # 保留该 mask 结果到 new_data
            if da_n not in new_data:
                new_data[da_n] = {'instances': {'mask_path': []}}
            new_data[da_n]['instances']['mask_path'].append(data[da_n]['instances']['mask_path'][int(mask_id)])

            # 更新 ori_data_stat，标记为已处理
            ori_data_stat['processed_mask_results'] += 1
            ori_data_stat[da_n]['processed_masks'].append(mask_id)
            if len(ori_data_stat[da_n]['processed_masks']) == num_dict[da_n]:
                ori_data_stat[da_n]['status'] = 'completed'
                session['disable_skip_image'] = False

            # 记录操作历史
            undo_stack.append(('keep', da_n, mask_id,session['disable_skip_image'],1))

        elif decision == 'skip':
            # 跳过当前 mask 并保存撤回记录
            ori_data_stat['processed_mask_results'] += 1
            ori_data_stat[da_n]['processed_masks'].append(mask_id)
            if len(ori_data_stat[da_n]['processed_masks']) == num_dict[da_n]:
                ori_data_stat[da_n]['status'] = 'completed'
                session['disable_skip_image'] = False
            undo_stack.append(('skip', da_n, mask_id,session['disable_skip_image'],1))
        elif decision == 'skip_img':
            # 如果图片级别不保留，跳过该图片的所有实例和编辑
            total_edits_in_image = num_dict[da_n]-len(ori_data_stat[da_n]['processed_masks'])
            ori_data_stat['processed_mask_results'] += total_edits_in_image  # 增加跳过的所有编辑
            ori_data_stat[da_n]['status'] = 'completed'  # 标记该图片为已完成
            session['disable_skip_image'] = False
            undo_stack.append(('skip_img', da_n, mask_id, session['disable_skip_image'], total_edits_in_image))  # 记录操作历史




        elif decision == 'undo':

            # 撤回上一次操作

            if undo_stack:

                last_action = undo_stack.pop()

                action_type, undo_da_n, undo_mask_id,button_stat_img,edit_num = last_action
                session['disable_skip_image'] = button_stat_img
                ori_data_stat['processed_mask_results'] -= edit_num

                # 撤回“保留”操作

                if action_type == 'keep':

                    # 从 new_data 中弹出最后一个保留的 mask_path

                    if undo_da_n in new_data:
                        new_data[undo_da_n]['instances']['mask_path'].pop()

                    # 从 processed_masks 列表中移除撤回的 mask_id

                    if ori_data_stat[undo_da_n]['processed_masks'][-1] == undo_mask_id:
                        ori_data_stat[undo_da_n]['processed_masks'].pop()



                # 撤回“跳过”操作

                elif action_type == 'skip':

                    # 仅从 processed_masks 中移除撤回的 mask_id

                    if ori_data_stat[undo_da_n]['processed_masks'][-1] == undo_mask_id:
                        ori_data_stat[undo_da_n]['processed_masks'].pop()


                # 更新状态

                ori_data_stat[undo_da_n]['status'] = 'unprocessed'

                save_json_data(new_data, new_data_path)

                save_json_data(ori_data_stat, ori_data_stat_path)

                save_undo_stack(undo_stack, undo_stack_path)
                # 重定向到当前 mask
                return redirect(url_for('filter', da_n=undo_da_n, mask_id=undo_mask_id))

        # 保存当前操作状态
        save_json_data(new_data, new_data_path)
        save_json_data(ori_data_stat, ori_data_stat_path)
        save_undo_stack(undo_stack, undo_stack_path)
        return redirect(url_for('index'))

    progress = "{:.2f}".format((ori_data_stat['processed_mask_results'] / ori_data_stat['total_mask_results']) * 100)
    instance_info = filter_instance(data, da_n,mask_id)
    # 渲染过滤页面
    return render_template('Grounding_filter.html', instance=instance_info, progress=progress, undo_stack=undo_stack)

# 主程序入口
if __name__ == '__main__':
    # 使用 argparse 传递参数
    parser = argparse.ArgumentParser(description="Flask data filtering application.")
    parser.add_argument('--dir', type=str, required=True, help="Path to the directory containing the JSON file")
    parser.add_argument('--port', type=int, default=8860, help="Port to run the Flask application on (default: 8860)")
    parser.add_argument('--subset_id', type=int, default=0, help="id of the subset data")

    args = parser.parse_args()

    # 获取目录路径
    directory = osp.join(args.dir,f'Subset_{args.subset_id}')

    # 查找目录中的 JSON 文件路径
    json_path = osp.join(directory, f"packed_data_full_tag_{args.subset_id}.json")
    new_data_path = osp.join(directory, f'Grounding_edit_data.json')
    undo_stack_path = osp.join(directory, 'Grounding_undo_stack.json')
    ori_data_stat_path = osp.join(directory, 'Grounding_stat.json')


    # 加载数据的通用函数：如果文件不存在则创建空字典并保存
    def load_or_create_json(path):
        if os.path.exists(path):
            return load_json_data(path)
        else:
            # 文件不存在时，初始化为空字典
            save_json_data({}, path)
            return {}

    # 加载或初始化数据
    data = load_json_data(json_path)
    # 计算每个图片的 mask 数量，更新 num_dict
    num_dict = {
        da_n: len(da['instances']['mask_path']) if 'instances' in da and 'mask_path' in da['instances'] else 0
        for da_n, da in data.items()
    }

    # 计算总 mask 数量，仅包含有效的 instances
    total_mask_results = sum(value for value in num_dict.values() if value > 0)
    # 加载状态数据
    if os.path.exists(ori_data_stat_path):
        ori_data_stat = load_json_data(ori_data_stat_path)
    else:
        ori_data_stat = {
            da_n: {
                'status': 'unprocessed',
                'processed_masks': []  # 用于记录已处理的mask
            } for da_n in data.keys()
        }
        ori_data_stat['total_mask_results'] = total_mask_results  # 总 mask 数量
        ori_data_stat['processed_mask_results'] = 0  # 已处理 mask 数量
        save_json_data(ori_data_stat, ori_data_stat_path)

    # 加载撤回栈
    undo_stack = load_undo_stack(undo_stack_path)

    # 加载已筛选的数据
    new_data = load_or_create_json(new_data_path)

    # 启动 Flask 应用
    app.run(port=args.port, debug=True)
