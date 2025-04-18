import os
import json
import shutil
import cv2
import os.path as osp
from data_gen_utils.temp import temp_view,temp_view_img

def copy_data(img_path, dst_dir, da_name, ins_name,sample_id=None,is_mask=False,is_source=False,is_inp=False):
    da_name = str(da_name)
    if is_mask:
        ins_name = str(ins_name)
        # 创建da子文件夹
        subfolder_path = os.path.join(dst_dir, da_name)
        os.makedirs(subfolder_path, exist_ok=True)
        final_path = os.path.join(subfolder_path, f"{ins_name}.png")
        shutil.copy(img_path, final_path)
        # print(f'copy_mask_to:{final_path}')
        return  final_path
    elif is_source:
        # 创建da子文件夹
        final_path = os.path.join(dst_dir, f"{da_name}.png")
        shutil.copy(img_path, final_path)
        # print(f'copy_src_to:{final_path}')
        return final_path
    elif is_inp:
        ins_name = str(ins_name)
        # 创建da子文件夹
        subfolder_path = os.path.join(dst_dir, da_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # 创建ins子文件夹
        ins_subfolder_path = os.path.join(subfolder_path, ins_name)
        os.makedirs(ins_subfolder_path, exist_ok=True)
        final_path = os.path.join(ins_subfolder_path, f"inp_img.png")
        shutil.copy(img_path, final_path)
        # print(f'copy_img_to:{final_path}')
        return final_path
    else:
        ins_name = str(ins_name)
        sample_id = str(sample_id)
        # 创建da子文件夹
        subfolder_path = os.path.join(dst_dir, da_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # 创建ins子文件夹
        ins_subfolder_path = os.path.join(subfolder_path, ins_name)
        os.makedirs(ins_subfolder_path, exist_ok=True)
        final_path  = os.path.join(ins_subfolder_path, f"{sample_id}.png")
        shutil.copy(img_path, final_path)
        # print(f'copy_img_to:{final_path}')
        return final_path
def save_json(data_dict, file_path):
    """
    将字典保存为 JSON 文件

    Args:
        data_dict (dict): 需要保存的字典
        file_path (str): JSON 文件的保存路径
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
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
import numpy as np
from tqdm import  tqdm
def merge_and_copy_json_data(json_path_list, dest_dir):
    # 创建指定文件夹
    source_img_dir = os.path.join(dest_dir, 'source_img')
    gen_img_dir = os.path.join(dest_dir, 'gen_img')
    source_mask_dir = os.path.join(dest_dir, 'source_mask')
    target_mask_dir = os.path.join(dest_dir, 'target_mask')

    # 确保目标文件夹存在
    os.makedirs(source_img_dir, exist_ok=True)
    os.makedirs(gen_img_dir, exist_ok=True)
    os.makedirs(source_mask_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)

    # 用于存储合并后的数据
    merged_data = dict()

    # 遍历json文件路径列表
    for json_path in tqdm(json_path_list,desc=f'proceeding json file'):
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                # 遍历每个实例
                for img_id, instances in data.items():
                    new_instance = dict()
                    for ins_id , edits in instances['instances'].items():
                        operations_per_ins = dict()
                        for case_id, gt_datas in edits.items():
                            new_case = {}
                            gen_img_path = gt_datas.get('gen_img_path')
                            # 如果 gen_img_path 存在，保留此条目
                            if not(gen_img_path and os.path.exists(gen_img_path)):
                                continue
                            else:
                                #移动文件,更新dict
                                ori_img_path = copy_data(gt_datas['ori_img_path'],source_img_dir,img_id,ins_id,case_id,is_source=True)
                                gen_img_path = copy_data(gt_datas['gen_img_path'], gen_img_dir, img_id, ins_id, case_id)
                                ori_mask_path = copy_data(gt_datas['ori_mask_path'], source_mask_dir, img_id, ins_id, case_id,is_mask=True)
                                tgt_mask_path = copy_data(gt_datas['tgt_mask_path'], target_mask_dir, img_id, ins_id, case_id,is_mask=True)
                                new_case['edit_prompt'] = gt_datas['edit_prompt']
                                new_case['ori_img_path'] = ori_img_path
                                new_case['gen_img_path'] = gen_img_path
                                new_case['ori_mask_path'] = ori_mask_path
                                new_case['tgt_mask_path']= tgt_mask_path
                            operations_per_ins[case_id] = new_case #保留命令
                        if len(operations_per_ins) > 0 :
                            new_instance[ins_id] = operations_per_ins#保留该instance
                        else:
                            continue
                    if len(new_instance) > 0 :
                        merged_data[img_id] = new_instance
                    else:
                        continue


    # 将合并后的数据保存为一个新的json文件
    merged_json_path = os.path.join(dest_dir, 'annotations.json')
    save_json(merged_data,merged_json_path)
    print(f"数据已合并并复制，合并后的JSON文件路径: {merged_json_path}")
def merge_and_copy_json_data_v2(json_path_list,assist_json_list,c_assist_json_list, dest_dir):
    error_num = 0
    img_num = 0
    # 创建指定文件夹
    source_img_dir = os.path.join(dest_dir, 'source_img')
    coarse_img_dir = os.path.join(dest_dir, 'coarse_img')
    source_mask_dir = os.path.join(dest_dir, 'source_mask')
    target_mask_dir = os.path.join(dest_dir, 'target_mask')
    inp_img_dir = os.path.join(dest_dir, 'inp_img')

    # 确保目标文件夹存在
    os.makedirs(source_img_dir, exist_ok=True)
    os.makedirs(coarse_img_dir, exist_ok=True)
    os.makedirs(source_mask_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)
    os.makedirs(inp_img_dir, exist_ok=True)

    # 用于存储合并后的数据
    merged_data = dict()

    # 遍历json文件路径列表
    for json_path, assist_json,c_assist_json in tqdm(zip(json_path_list, assist_json_list,c_assist_json_list), desc=f'proceeding json files'):
        # 读取 json 文件
        if 'PIE' in json_path:
            type='PIE'
        elif 'Subject' in json_path:
            type='Sub'
        with open(json_path, 'r') as f:
            data = json.load(f)

        with open(assist_json, 'r') as f:
            assist_data = json.load(f)
        with open(c_assist_json, 'r') as f:
            c_assist_data = json.load(f)
        # 遍历每个实例
        for img_id, instances in data.items():
            new_instance = dict()
            caption = c_assist_data[img_id]['4v_caption']
            for ins_id , edits in instances['instances'].items():
                operations_per_ins = dict()
                if len(edits)==0:
                    print(f'no edit for {img_id},{json_path}')
                inp_img_path = assist_data[img_id]['instances']['inp_img_path'][int(ins_id)]
                for case_id, gt_datas in edits.items():
                    new_case = {}
                    coarse_img_path = gt_datas.get('coarse_input_path')
                    if not(coarse_img_path and os.path.exists(coarse_img_path)):
                        print(f'no coarse img path for {img_id},{json_path}')
                        continue
                    else:
                        ori_img_path = copy_data(gt_datas['src_img_path'],source_img_dir,img_num,ins_id,case_id,is_source=True)
                        coarse_input_path = copy_data(gt_datas['coarse_input_path'], coarse_img_dir, img_num, ins_id, case_id)
                        ori_mask_path = copy_data(gt_datas['ori_mask_path'], source_mask_dir, img_num, ins_id, case_id,is_mask=True)
                        tgt_mask_path = copy_data(gt_datas['tgt_mask_path'], target_mask_dir, img_num, ins_id, case_id)#is mask =False to allow case id named save
                        new_inp_img_path = copy_data(inp_img_path,inp_img_dir,img_num,ins_id,is_inp=True)
                        new_case['edit_prompt'] = gt_datas['edit_prompt']
                        new_case['edit_param'] = gt_datas['edit_param']
                        new_case['ori_img_path'] = ori_img_path
                        new_case['coarse_input_path'] = coarse_input_path
                        new_case['ori_mask_path'] = ori_mask_path
                        new_case['tgt_mask_path']= tgt_mask_path
                        new_case['inp_img_path'] = new_inp_img_path
                        new_case['obj_label'] = gt_datas['obj_label']
                    operations_per_ins[case_id] = new_case #保留命令
                if len(operations_per_ins) > 0 :
                    new_instance[ins_id] = operations_per_ins#保留该instance
                else:

                    continue
            if len(new_instance) > 0 :
                merged_data[img_num]={}
                merged_data[img_num]['instances'] = new_instance
                merged_data[img_num]['4v_caption'] = caption
                merged_data[img_num]['type'] = type
                img_num+=1
            else:
                error_num+=1
                print(f'no instances v2 for {img_id},{json_path}')
                continue
        print(f'cur_img_num:{img_num}')


    # 将合并后的数据保存为一个新的json文件
    merged_json_path = os.path.join(dest_dir, 'annotations.json')
    save_json(merged_data,merged_json_path)
    print(f"数据已合并并复制，合并后的JSON文件路径: {merged_json_path}")
    print(f'error num{error_num}')
# 使用的JSON路径列表
def merge_and_copy_json_data_3d(json_path_list,assist_json_list,c_assist_json_list, dest_dir):
    error_num = 0
    img_num = 0
    # 创建指定文件夹
    source_img_dir = os.path.join(dest_dir, 'source_img')
    coarse_img_dir = os.path.join(dest_dir, 'coarse_img')
    source_mask_dir = os.path.join(dest_dir, 'source_mask')
    target_mask_dir = os.path.join(dest_dir, 'target_mask')
    gen_img_dir = os.path.join(dest_dir, 'gen_img')

    # 确保目标文件夹存在
    os.makedirs(source_img_dir, exist_ok=True)
    os.makedirs(coarse_img_dir, exist_ok=True)
    os.makedirs(source_mask_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)
    os.makedirs(gen_img_dir, exist_ok=True)

    # 用于存储合并后的数据
    merged_data = dict()

    # 遍历json文件路径列表
    for json_path, assist_json,c_assist_json in tqdm(zip(json_path_list, assist_json_list,c_assist_json_list), desc=f'proceeding json files'):
        # 读取 json 文件
        if 'PIE' in json_path:
            type='PIE'
        elif 'Subject' in json_path:
            type='Sub'
        with open(json_path, 'r') as f:
            data = json.load(f)

        # with open(assist_json, 'r') as f:
        #     assist_data = json.load(f)
        with open(c_assist_json, 'r') as f:
            c_assist_data = json.load(f)
        # 遍历每个实例
        for img_id, instances in data.items():
            new_instance = dict()
            caption = c_assist_data[img_id]['4v_caption']
            for ins_id , edits in instances['instances'].items():
                operations_per_ins = dict()
                if len(edits)==0:
                    print(f'no edit for {img_id},{json_path}')
                # inp_img_path = assist_data[img_id]['instances']['inp_img_path'][int(ins_id)]
                for case_id, gt_datas in edits.items():
                    new_case = {}
                    coarse_img_path = gt_datas.get('coarse_input_path')
                    if not(coarse_img_path and os.path.exists(coarse_img_path)):
                        print(f'no coarse img path for {img_id},{json_path}')
                        continue
                    else:
                        ori_img_path = copy_data(gt_datas['src_img_path'],source_img_dir,img_num,ins_id,case_id,is_source=True)
                        coarse_input_path = copy_data(gt_datas['coarse_input_path'], coarse_img_dir, img_num, ins_id, case_id)
                        ori_mask_path = copy_data(gt_datas['ori_mask_path'], source_mask_dir, img_num, ins_id, case_id,is_mask=True)
                        tgt_mask_path = copy_data(gt_datas['tgt_mask_path'], target_mask_dir, img_num, ins_id, case_id)#is mask =False to allow case id named save
                        # new_inp_img_path = copy_data(inp_img_path,inp_img_dir,img_num,ins_id,is_inp=True)
                        gen_img_path = copy_data(gt_datas['gen_img_path'],gen_img_dir, img_num, ins_id, case_id)
                        new_case['edit_prompt'] = gt_datas['edit_prompt']
                        new_case['edit_param'] = gt_datas['edit_param']
                        new_case['ori_img_path'] = ori_img_path
                        new_case['coarse_input_path'] = coarse_input_path
                        new_case['ori_mask_path'] = ori_mask_path
                        new_case['tgt_mask_path']= tgt_mask_path
                        new_case['gen_img_path'] = gen_img_path
                        new_case['obj_label'] = gt_datas['obj_label']
                    operations_per_ins[case_id] = new_case #保留命令
                if len(operations_per_ins) > 0 :
                    new_instance[ins_id] = operations_per_ins#保留该instance
                else:

                    continue
            if len(new_instance) > 0 :
                merged_data[img_num]={}
                merged_data[img_num]['instances'] = new_instance
                merged_data[img_num]['4v_caption'] = caption
                merged_data[img_num]['type'] = type
                img_num+=1
            else:
                error_num+=1
                print(f'no instances v2 for {img_id},{json_path}')
                continue
        print(f'cur_img_num:{img_num}')


    # 将合并后的数据保存为一个新的json文件
    merged_json_path = os.path.join(dest_dir, 'annotations.json')
    save_json(merged_data,merged_json_path)
    print(f"数据已合并并复制，合并后的JSON文件路径: {merged_json_path}")
    print(f'error num{error_num}')

# json_path_list = [f"/data/Hszhu/dataset/PIE-Bench_v1/Subset_{i}/full_3d_edit_sample_{i}.json" for i in range(4)]
# json_path_list_2 = [f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/full_3d_edit_sample_{i}.json" for i in range(4)]
# json_path_full = json_path_list + json_path_list_2
json_path_list = [f"/data/Hszhu/dataset/PIE-Bench_v1/Subset_{i}/structure_completion_select_{i}.json" for i in range(4)]
json_path_list_2 = [f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/structure_completion_select_{i}.json" for i in range(4)]
json_path_full = json_path_list + json_path_list_2

c_ass_json_path_list = [f"/data/Hszhu/dataset/PIE-Bench_v1/Subset_{i}/coarse_input_full_pack_{i}.json" for i in range(4)]
c_ass_json_path_list_2 = [f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/coarse_input_full_pack_{i}.json" for i in range(4)]
c_ass_json_path_full = c_ass_json_path_list + c_ass_json_path_list_2

ass_json_path_list = [f"/data/Hszhu/dataset/PIE-Bench_v1/Subset_{i}/mat_fooocus_inpainting_{i}.json" for i in range(4)]
ass_json_path_list_2 = [f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/mat_fooocus_inpainting_{i}.json" for i in range(4)]
ass_json_path_full = ass_json_path_list + ass_json_path_list_2
dest_dir = "/data/Hszhu/dataset/Geo-Bench-SC/"

# 调用函数
merge_and_copy_json_data_v2(json_path_full,ass_json_path_full,c_ass_json_path_full, dest_dir)
def classify_edit_prompt(edit_prompt, degrees):
    """
    根据 edit_prompt 来定位属于哪个level

    Args:
        edit_prompt (str): 需要分类的编辑提示
        degrees (dict): 包含每个级别的描述信息

    Returns:
        str: 返回匹配到的level，未匹配返回 'unknown'
    """
    for level, data in degrees.items():
        for description in data['description']:
            if description in edit_prompt.lower():  # 转为小写匹配
                return level
    return 'unknown'

# 定义 degrees 字典
degrees = {
    "level_1": {
        "description": ["lightly", "slightly", "gently", "mildly"]
    },
    "level_2": {
        "description": ["moderately", "markedly", "appreciably"]
    },
    "level_3": {
        "description": ["heavily", "intensely", "significantly", "strongly"]
    }
}
level_counts_3d = {"level_1": 0, "level_2": 0, "level_3": 0, "unknown": 0}
level_counts_2d = {"level_1": 0, "level_2": 0, "level_3": 0, "unknown": 0}
# json_path_list = [f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/full_2d_edit_sample_{i}.json" for i in range(4)]
# json_path_list = [f"/data/Hszhu/dataset/PIE-Bench_v1/Subset_{i}/full_2d_edit_sample_{i}.json" for i in range(4)]
json_path_list = [f"/data/Hszhu/dataset/Geo-Bench-SC/annotations.json"]
img_num=0
ins_num=0
edit_num=0
error_num=0
for file in json_path_list:
    data = load_json(file)
    for da_n,da in data.items():
        img_stat=False
        ins_stat=False

        ins_data = da['instances']
        for ins_id ,ins in ins_data.items():

            for case_id, gt_data in ins.items():
                img_stat=True
                ins_stat=True
                edit_num+=1
                edit_prompt = gt_data.get('edit_prompt', '')
                edit_type = '3d' if 'y-axis' in edit_prompt else '2d'
                level = classify_edit_prompt(edit_prompt, degrees)
                if edit_type =='3d':
                    level_counts_3d[level]+=1
                else:
                    level_counts_2d[level]+=1
            if ins_stat:
                ins_num+=1

        if img_stat:
            img_num+=1
        else:
            error_num+=1
    print(f'cur img num:{img_num}')
# 输出统计结果
print(f"2D Edit Level 1 count: {level_counts_2d['level_1']}")
print(f"2D Edit Level 2 count: {level_counts_2d['level_2']}")
print(f"2D Edit Level 3 count: {level_counts_2d['level_3']}")

# print(f"2D Unknown count: {level_counts_2d['unknown']}")
# 输出统计结果
print(f"3D Edit Level 1 count: {level_counts_3d['level_1']}")
print(f"3D Edit Level 2 count: {level_counts_3d['level_2']}")
print(f"3D Edit Level 3 count: {level_counts_3d['level_3']}")
# print(f"3D Unknown count: {level_counts_3d['unknown']}")
print(f'img_num:{img_num}, ins_num:{ins_num}, edit_num:{edit_num} error_num:{error_num}')