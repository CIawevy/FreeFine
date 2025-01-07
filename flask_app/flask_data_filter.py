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
# 加载数据的函数
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# 保存筛选后的数据
def save_json_data(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
# 设置最大撤回操作数
MAX_UNDO_STACK_SIZE = 10

# 保存 undo_stack
def save_undo_stack(undo_stack, undo_stack_path):
    if len(undo_stack) > MAX_UNDO_STACK_SIZE:
        undo_stack = undo_stack[-MAX_UNDO_STACK_SIZE:]  # 只保留最近的 MAX_UNDO_STACK_SIZE 次操作

    with open(undo_stack_path, 'w') as f:
        json.dump(undo_stack, f, indent=4)

# 加载 undo_stack
def load_undo_stack(undo_stack_path):
    if os.path.exists(undo_stack_path):
        with open(undo_stack_path, 'r') as f:
            undo_stack = json.load(f)
            if isinstance(undo_stack, list):  # 确保 undo_stack 是列表
                return undo_stack
            else:
                return []  # 如果不是列表，初始化为空列表
    return []

def filter_instance(data, da_n, ins_id, edit_id):
    instances = data[da_n]['instances']
    current_ins = instances[ins_id]
    meta = current_ins[edit_id]
    #edit_prompt convert to Chinese


    return {
        'edit_prompt': meta['edit_prompt'],
        'edit_param': meta['edit_param'],
        'ori_img': meta['src_img_path'],
        'ori_mask': meta['ori_mask_path'],
        'obj_label': meta['obj_label'],
        'edit_result':meta['gen_img_path'],
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
    # 获取未处理的样本
    for da_n, da in data.items():
        if ori_data_stat[da_n]['status'] == 'completed':
            continue  # 如果图片已经处理完毕，跳过

        instances = da['instances']
        for ins_id, current_ins in instances.items():
            # 如果实例的状态是 completed，跳过该实例
            if ori_data_stat[da_n]['processed_ins'][ins_id]['status'] == 'completed':
                continue

            for edit_id, coarse_input_pack in current_ins.items():
                # 如果编辑已经处理，跳过该编辑
                if edit_id in ori_data_stat[da_n]['processed_ins'][ins_id]['processed_edit']:
                    continue

                # 返回当前待处理的实例
                return redirect(url_for('filter', da_n=da_n, ins_id=ins_id, edit_id=edit_id))
    os.remove(undo_stack_path)
    os.remove(ori_data_stat_path)
    return "All data processed!"





@app.route('/filter/<da_n>/<ins_id>/<edit_id>', methods=['GET', 'POST'])
def filter(da_n,ins_id, edit_id):
    session.setdefault('disable_skip_image', False)
    session.setdefault('disable_skip_instance', False)

    if request.method == 'POST':
        # 获取用户的决定
        decision = request.form['decision']

        if decision == 'skip_img':
            # 如果图片级别不保留，跳过该图片的所有实例和编辑
            total_edits_in_image = sum(num_dict[da_n].values())
            ori_data_stat['processed_edit_results'] += total_edits_in_image # 增加跳过的所有编辑
            ori_data_stat[da_n]['status'] = 'completed'  # 标记该图片为已完成
            session['disable_skip_image'] = False
            undo_stack.append(('skip_img', da_n,ins_id,edit_id,session['disable_skip_image'],session['disable_skip_instance'],total_edits_in_image))  # 记录操作历史




        elif decision == 'skip_instance':
            # 如果实例级别不保留，跳过该实例的所有编辑结果
            total_edits_in_instance = num_dict[da_n][ins_id] - len(ori_data_stat[da_n]['processed_ins'][ins_id]['processed_edit'])
            ori_data_stat['processed_edit_results'] += total_edits_in_instance  # 增加跳过的所有编辑
            ori_data_stat[da_n]['processed_ins'][ins_id]['status'] = 'completed'  # 标记实例为已处理
            if all(v['status'] == 'completed' for k, v in ori_data_stat[da_n]['processed_ins'].items()):
                ori_data_stat[da_n]['status'] = 'completed'  # 标记整个图像为完成
                session['disable_skip_image'] = False
            session['disable_skip_instance'] = False
            undo_stack.append(('skip_instance', da_n, ins_id,edit_id,session['disable_skip_image'],session['disable_skip_instance'],total_edits_in_instance))



        elif decision == 'keep':
            # 保留该实例的编辑结果到 new_data
            session['disable_skip_image'] = True
            session['disable_skip_instance']= True

            if da_n not in new_data:
                new_data[da_n] = {'instances': {}}
            if ins_id not in new_data[da_n]['instances']:
                new_data[da_n]['instances'][ins_id] = {}
            new_data[da_n]['instances'][ins_id][edit_id] = data[da_n]['instances'][ins_id][edit_id]
            # 标记实例为已处理，更新 ori_data_stat


            # 更新该实例的编辑状态
            # 增加 processed_edit_results 计数
            ori_data_stat['processed_edit_results'] += 1
            ori_data_stat[da_n]['processed_ins'][ins_id]['processed_edit'].append(edit_id)

            if len(ori_data_stat[da_n]['processed_ins'][ins_id]['processed_edit']) == num_dict[da_n][ins_id]:
                ori_data_stat[da_n]['processed_ins'][ins_id]['status'] = 'completed'  # 标记实例筛选完成
                session['disable_skip_instance'] = False

            if all(v['status'] == 'completed' for k, v in ori_data_stat[da_n]['processed_ins'].items()):
                ori_data_stat[da_n]['status'] = 'completed'  # 标记图像筛选完成
                session['disable_skip_image'] = False


            # 记录操作历史并保存
            undo_stack.append(('keep', da_n, ins_id, edit_id,session['disable_skip_image'],session['disable_skip_instance'], 1))

        elif decision == 'skip':
            # 更新该实例的编辑状态
            ori_data_stat['processed_edit_results'] += 1  # 增加已处理的编辑结果计数
            ori_data_stat[da_n]['processed_ins'][ins_id]['processed_edit'].append(edit_id)
            if len(ori_data_stat[da_n]['processed_ins'][ins_id]['processed_edit']) == num_dict[da_n][ins_id]:
                ori_data_stat[da_n]['processed_ins'][ins_id]['status'] = 'completed'  # 标记实例筛选完成
                session['disable_skip_instance'] = False

            if all(v['status'] == 'completed' for k, v in ori_data_stat[da_n]['processed_ins'].items()):
                ori_data_stat[da_n]['status'] = 'completed'  # 标记图像筛选完成
                session['disable_skip_image'] = False
            # 跳过当前实例的编辑结果并保存撤回记录
            undo_stack.append(('skip', da_n, ins_id, edit_id,session['disable_skip_image'],session['disable_skip_instance'],1))


        elif decision == 'undo':
            # 撤回上一次操作
            if undo_stack:
                last_action = undo_stack.pop()
                action_type, undo_da_n, undo_ins_id, undo_edit_id,button_stat_img,button_stat_ins, edit_num = last_action  # *undo_edit_id allows for flexible number of items
                session['disable_skip_image'] = button_stat_img
                session['disable_skip_instance'] = button_stat_ins
                ori_data_stat['processed_edit_results'] -= edit_num  # 减少已处理的编辑结果计数
                if action_type == 'keep':
                    # 撤回“保留”操作
                    if undo_ins_id in new_data[undo_da_n]['instances']:
                        # 删除 new_data 中的编辑
                        del new_data[undo_da_n]['instances'][undo_ins_id][undo_edit_id]
                        # 如果该实例的所有编辑都已被删除，删除该实例
                        if not new_data[undo_da_n]['instances'][undo_ins_id]:
                            del new_data[undo_da_n]['instances'][undo_ins_id]
                        # 如果该 da_n 中的所有实例都被删除，删除该 da_n
                        if not new_data[undo_da_n]['instances']:
                            del new_data[undo_da_n]

                    # 从 ori_data_stat 中撤销处理过的编辑
                    if undo_edit_id in ori_data_stat[undo_da_n]['processed_ins'][undo_ins_id]['processed_edit']:
                        ori_data_stat[undo_da_n]['processed_ins'][undo_ins_id]['processed_edit'].remove(undo_edit_id)
                        ori_data_stat[undo_da_n]['processed_ins'][undo_ins_id]['status'] = 'unprocessed'
                        ori_data_stat[undo_da_n]['status'] = 'unprocessed'


                elif action_type == 'skip':
                    if undo_edit_id in ori_data_stat[undo_da_n]['processed_ins'][undo_ins_id]['processed_edit']:
                        ori_data_stat[undo_da_n]['processed_ins'][undo_ins_id]['processed_edit'].remove(undo_edit_id)
                        ori_data_stat[undo_da_n]['processed_ins'][undo_ins_id]['status'] = 'unprocessed'
                        ori_data_stat[undo_da_n]['status'] = 'unprocessed'


                elif action_type == 'skip_instance':
                    # 撤回“跳过实例”，将整个实例标记为未处理
                    ori_data_stat[da_n]['processed_ins'][ins_id]['status'] = 'unprocessed'
                    ori_data_stat[undo_da_n]['status'] = 'unprocessed'


                elif action_type == 'skip_img':
                    # 撤回“跳过图片”，将图片标记为未处理
                    ori_data_stat[undo_da_n]['status'] = 'unprocessed'

                # 重定向回撤回的实例或图片
                save_json_data(new_data, new_data_path)  # 保存新数据
                save_json_data(ori_data_stat, ori_data_stat_path)  # 保存状态
                save_undo_stack(undo_stack, undo_stack_path)  # 保存撤回栈
                return redirect(url_for('filter', da_n=undo_da_n, ins_id=undo_ins_id, edit_id=undo_edit_id))

        # 每一步保存操作方便随时Resume
        save_json_data(new_data, new_data_path)  # 保存新数据
        save_json_data(ori_data_stat, ori_data_stat_path)  # 保存状态
        save_undo_stack(undo_stack, undo_stack_path)  # 保存撤回栈
        # 继续处理下一个样本
        return redirect(url_for('index'))

    # 获取当前进度
    progress = "{:.2f}".format((ori_data_stat['processed_edit_results'] / ori_data_stat['total_edit_results']) * 100)

    # 获取当前实例信息
    instance_info = filter_instance(data, da_n, ins_id, edit_id)
    # 渲染展示页面
    return render_template('filter.html', instance=instance_info,
                           disable_skip_image=session['disable_skip_image'],
                           disable_skip_instance=session['disable_skip_instance'],
                           progress=progress,
                           undo_stack=undo_stack,
    )
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
    json_path = osp.join(directory, f"generated_dataset_full_pack_{args.subset_id}.json")
    new_data_path = osp.join(directory, 'new_data.json')
    undo_stack_path = osp.join(directory, 'undo_stack.json')
    ori_data_stat_path = osp.join(directory, 'ori_data_stat.json')


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
    num_dict = {
        da_n: {
            ins_id: len(ins_data)
            for ins_id, ins_data in da['instances'].items()
        }
        for da_n, da in data.items()
    }

    # 计算 total_edit_results
    total_edit_results = sum(
        sum(instance_count for instance_count in instances.values())
        for instances in num_dict.values()
    )
    # 加载已处理状态的数据，如果文件不存在则初始化为包含图片的状态结构
    if os.path.exists(ori_data_stat_path):
        ori_data_stat = load_json_data(ori_data_stat_path)
    else:
        ori_data_stat = {
            da_n: {
                'status': 'unprocessed',
                'processed_ins': {
                    ins_id: {
                        'status': 'unprocessed',
                        'processed_edit': []
                    } for ins_id in data[da_n]['instances'].keys()
                },

            } for da_n in data.keys()
        }
        ori_data_stat['total_edit_results'] = total_edit_results  # 总编辑数量
        ori_data_stat['processed_edit_results']= 0 #初始化编辑数量为0
        save_json_data(ori_data_stat, ori_data_stat_path)

    # 加载撤回栈
    undo_stack = load_undo_stack(undo_stack_path)

    # 加载已筛选的数据
    new_data = load_or_create_json(new_data_path)

    # 启动 Flask 应用
    app.run(port=args.port, debug=True)
