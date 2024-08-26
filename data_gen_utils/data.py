import torch
import os
import os.path as osp
from tqdm import  tqdm
import json
from data_gen_utils.blip2 import  BLIP2
from data_gen_utils.Myclip import Myclip

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
def replace_first_folder(original_path, new_base_path):
    """
    将路径中的第一个文件夹替换为指定的新路径。

    Args:
        original_path (str): 原始的文件路径。
        new_base_path (str): 要替换的新的基础路径。

    Returns:
        str: 修改后的新文件路径。
    """
    # 分割路径为目录部分和文件名部分
    dirname, filename = os.path.split(original_path)

    # 分割目录部分为各级目录
    path_parts = dirname.split(os.sep)

    # 找到第一个非空部分，将其替换为新路径
    if path_parts[0] == "":
        path_parts[1] = os.path.normpath(new_base_path)
    else:
        path_parts[0] = os.path.normpath(new_base_path)

    # 重新组合路径
    new_dirname = os.sep.join(path_parts)
    new_image_path = os.path.join(new_dirname, filename)

    return new_image_path
def split_data_dict(data_dict):
    label_data = dict()
    unlabel_data =dict()
    for k,v in data_dict.items():
        if v['original_prompt'] == '':
            unlabel_data[k] = v
        else:
            label_data[k]=v
    return unlabel_data,label_data


def PIE_data_preprocessor(base_dir):
    already_exist_list = []
    ti2i_benchmark = osp.join(base_dir,"mapping_file_ti2i_benchmark.json")
    PIE_benchmark = osp.join(base_dir,"mapping_file.json")
    ti2i_benchmark_json = load_json(ti2i_benchmark)
    PIE_benchmark_json = load_json(PIE_benchmark)
    unlab_ti2i,_ = split_data_dict(ti2i_benchmark_json)
    _,lab_PIE = split_data_dict(PIE_benchmark_json)
    blip2 = BLIP2("/data/Hszhu/prompt-to-prompt/blip2-opt-2.7b/")
    # clip_model = clip("ViT-B/32")
    # sam = build_efficient_sam_vits()
    packed_data_dict = dict()
    i=0
    new_base_path = '/data/Hszhu/dataset/PIE-Bench_v1/'
    save_json_path = osp.join(new_base_path,"packed_data.json")
    for k, v in tqdm(unlab_ti2i.items(), desc="Processing unlabelled ti2i data"):
        image_path = replace_first_folder(v['image_path'], new_base_path)
        if image_path in already_exist_list:
            continue
        else:
            already_exist_list.append(image_path)
        blip2_caption = blip2(image_path)
        #masks = sam(image_path) #efficient sam 必须要Box input,不然推荐使用原本的SAM，推理慢且效果有限，Segment all，#尝试直接使用改进版本
        #for msk in masks:
            #mask_path = save_mask(masks)
            #obj_text = clip_model(image_path,mask_path,blip2_caption) #grounding 范式下不需要做匹配，会分配openset标签
        file_info = dict()
        file_info['image_path'] = image_path
        file_info['original_prompt'] = blip2_caption
        file_info['data_type'] = 'ti2i'
        packed_data_dict[i] = file_info
        i+=1
    already_exist_list=[]
    for k, v in tqdm(lab_PIE.items(), desc="Processing labelled PIE data"):
        file_info = dict()
        image_path = osp.join(new_base_path,'annotation_images',v['image_path'])
        if image_path in already_exist_list:
            continue
        else:
            already_exist_list.append(image_path)
        file_info['image_path'] = image_path
        file_info['original_prompt'] = v['original_prompt']
        file_info['data_type'] = 'PIE'
        packed_data_dict[i]=file_info
        i+=1
    print('Done!')
    save_json(packed_data_dict,save_json_path)
    return packed_data_dict


def toy_data_loader(input_dir,dataset_name,):
    return 'OK'



