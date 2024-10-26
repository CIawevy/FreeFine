import os
import json
import shutil
import cv2
import os.path as osp


def copy_data(img_path, dst_dir, da_name, ins_name,sample_id=None,is_mask=False,is_source=False):
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
def merge_and_copy_json_data_v2(json_path_list, dest_dir):
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
                                ori_img_path = copy_data(gt_datas['src_img_path'],source_img_dir,img_id,ins_id,case_id,is_source=True)
                                gen_img_path = copy_data(gt_datas['gen_img_path'], gen_img_dir, img_id, ins_id, case_id)
                                ori_mask_path = copy_data(gt_datas['ori_mask_path'], source_mask_dir, img_id, ins_id, case_id,is_mask=True)
                                tgt_mask_path = copy_data(gt_datas['tgt_mask_path'], target_mask_dir, img_id, ins_id, case_id,is_mask=True)
                                new_case['edit_prompt'] = gt_datas['edit_prompt']
                                new_case['tag_caption'] = gt_datas['tag_caption']
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
# 使用的JSON路径列表
json_path_list = [f"/data/Hszhu/dataset/PIE-Bench_v1/Subset_{i}/generated_dataset_full_pack_{i}.json" for i in range(4)]
# json_path_list.extend([f"/data/Hszhu/dataset/PIE-Bench_v1/Subset_unseen_{i}/generated_dataset_full_pack_v2.json" for i in range(2)])

# 指定目标文件夹
dest_dir = "/data/Hszhu/dataset/Gedi_full/"

# 调用函数
# merge_and_copy_json_data(json_path_list, dest_dir)
# merge_and_copy_json_data_v2(json_path_list, dest_dir)
#print info
data = load_json(osp.join(dest_dir,"annotations.json"))
ins_length = np.array([len(v.keys()) for k,v in data.items()])
print(f'img num :{len(ins_length)}')
print(f'ins num :{np.sum(ins_length)}')
print(f'average ins:{np.mean(ins_length)}')
ins_dict_list= np.array([v for k,v in data.items()])
edit_length = []
for ins_d in ins_dict_list: #dict('0','1','2') per img
    for k,v in ins_d.items():#dict('0','1','2') per instance
        edit_length.append(len(v.keys()))
edit_length = np.array(edit_length) #ins num
print(f'full edit result pair num:{np.sum(edit_length)}')
print(f'average edit result per ins:{np.mean(edit_length)}')
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
def count_levels_in_data(data, degrees):
    """
    统计数据中每个level的数量

    Args:
        data (dict): 合并后的数据
        degrees (dict): 包含每个级别的描述信息

    Returns:
        dict: 各个level的数量统计
    """
    level_counts = {"level_1": 0, "level_2": 0, "level_3": 0, "unknown": 0}

    # 遍历每个图像和实例
    for img_id, instances in data.items():
        for ins_id, edits in instances.items():
            for case_id, gt_data in edits.items():
                edit_prompt = gt_data.get('edit_prompt', '')
                level = classify_edit_prompt(edit_prompt, degrees)
                level_counts[level] += 1

    return level_counts
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

# 读取合并后的 JSON 文件
data = load_json(osp.join(dest_dir, "annotations.json"))

# 统计每个 level 的数量
level_counts = count_levels_in_data(data, degrees)

# 输出统计结果
print(f"Level 1 count: {level_counts['level_1']}")
print(f"Level 2 count: {level_counts['level_2']}")
print(f"Level 3 count: {level_counts['level_3']}")
print(f"Unknown count: {level_counts['unknown']}")