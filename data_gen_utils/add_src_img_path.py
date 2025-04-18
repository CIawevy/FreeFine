import json
import os.path as osp
from tqdm import  tqdm
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
def add_src_img_path_sp(i,dst_base):
    src_source = osp.join(dst_base,f"Subset_{i}/mask_tag_relabelled_lmm_v2_{i}.json")
    tgt_path = osp.join(dst_base, f"Subset_{i}/mat_fooocus_inpainting_{i}.json")
    # tgt_path = osp.join(dst_base, f"Subset_{i}/mat_fooocus_inpainting_{i}.json")
    src_data = load_json(src_source)
    meta = load_json(tgt_path)
    new_meta = {}
    for da_n,da in meta.items():
        print(da_n)
        if 'instances' not in da or len(da['instances']['mask_path']) == 0:
            # del meta[da_n]
            continue
        new_meta[da_n] = da
        new_meta[da_n].update({"src_img_path": src_data[da_n]["src_img_path"]})
        new_meta[da_n].update({"4v_caption": src_data[da_n]["4v_caption"]})
    save_json(new_meta,tgt_path)
def add_src_img_path(i,dst_base):
    src_source = osp.join(dst_base,f"Subset_{i}/mask_tag_relabelled_lmm_v2_{i}.json")
    tgt_path = osp.join(dst_base, f"Subset_{i}/mask_label_filtered_{i}.json")
    # tgt_path = osp.join(dst_base, f"Subset_{i}/mat_fooocus_inpainting_{i}.json")
    src_data = load_json(src_source)
    meta = load_json(tgt_path)
    new_meta = {}
    for da_n,da in meta.items():
        print(da_n)
        if 'instances' not in da or len(da['instances']['mask_path'])==0:
            # del meta[da_n]
            continue
        new_meta[da_n] = da
        new_meta[da_n].update({"src_img_path": src_data[da_n]["src_img_path"]})
        new_meta[da_n].update({"4v_caption": src_data[da_n]["4v_caption"]})
    save_json(new_meta,tgt_path)
# dst_base = "/data/Hszhu/dataset/GRIT/"
# for i in tqdm(range(4)):
#     add_src_img_path_sp(i,dst_base)
# print('finish')

# add_src_img_path_sp(1,dst_base)
# def add_caption(i,dst_base):
#     src_source = osp.join(dst_base,f"Subset_{i}/mask_tag_relabelled_lmm_v2_{i}.json")
#     tgt_path = osp.join(dst_base, f"Subset_{i}/mask_label_filtered_{i}.json")
#     # tgt_path = osp.join(dst_base, f"Subset_{i}/mat_fooocus_inpainting_{i}.json")
#     src_data = load_json(src_source)
#     meta = load_json(tgt_path)
#     new_meta = {}
#     for da_n,da in meta.items():
#         print(da_n)
#         if 'instances' not in da:
#             # del meta[da_n]
#             continue
#         new_meta[da_n] = da
#         new_meta[da_n].update({"src_img_path": src_data[da_n]["src_img_path"]})
#         new_meta[da_n].update({"4v_caption": src_data[da_n]["4v_caption"]})
#     save_json(new_meta,tgt_path)
def add_caption_2(i,dst_base):
    src_source = "/data/Hszhu/dataset/Geo-Bench/annotations.json"
    tgt_path = "/data/Hszhu/dataset/Geo-Bench/generated_results.json"
    # tgt_path = osp.join(dst_base, f"Subset_{i}/mat_fooocus_inpainting_{i}.json")
    src_data = load_json(src_source)
    meta = load_json(tgt_path)
    new_meta = {}
    for da_n,da in meta.items():
        print(da_n)
        if 'instances' not in da:
            # del meta[da_n]
            continue
        new_meta[da_n] = da
        new_meta[da_n].update({"type": src_data[da_n]["type"]})
        new_meta[da_n].update({"4v_caption": src_data[da_n]["4v_caption"]})
    save_json(new_meta,tgt_path)
dst_base = "/data/Hszhu/dataset/Geo-Bench"
for i in tqdm(range(4)):
    add_caption_2(i,dst_base)
print('finish')

