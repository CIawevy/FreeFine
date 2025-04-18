import json
import os.path as osp

from h5py.tests.test_file_alignment import dataset_name
from tqdm import  tqdm
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



operations = {
    "move": {
        "descriptions": ["Move", "Shift", "Slide", "Drag"],
        "directions": ["upward", "downward", "leftward", "rightward", "upper-left", "upper-right", "lower-left",
                       "lower-right"]
    },
    "rotate": { #只有旋转需要指定轴向
        "descriptions": ["Rotate", "Spin", "Turn", "Swivel"],
        "directions": {
            "2D": ["around the z-axis clockwise", "around the z-axis counterclockwise"],#3d_z
            # "3D_x": ["around the x-axis clockwise", "around the x-axis counterclockwise"], 暂时不用
            "3D_y": ["around the y-axis clockwise", "around the y-axis counterclockwise"],#SV3D
        }
    },
    "enlarge": {
        "descriptions": ["Enlarge", "Expand","zoom",'amplify'], #为什么考虑x,y单独缩放的场景，考虑一个咖啡杯或者一个物体我只想让他矮一点而不是全面变小
        "directions": ["uniformly"] #这里与drag还是不一样的，因为drag数据集构建一般都是对partial parts操作
        # "directions": ["uniformly","horizontally", "vertically",]
    },
    "shrink": {
        "descriptions": ["Shrink","Contract"],
        "directions": ["uniformly"],
        # "directions": ["uniformly","horizontally", "vertically",]

    },
    # "flip": {
    #     "descriptions": ["Flip"],
    #     "directions": ["horizontally", "vertically"]
    # }
}
celeb_degrees = {
    "level_1": {
        "description": ["lightly", "slightly", "gently", "mildly"]
    },
}
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
def find_motion_type(prompt):
    for motion_type, motion_meta in operations.items():
        if any(motion_word in prompt for motion_word in motion_meta['descriptions']):
            return motion_type
    # 如果没有匹配到任何动作，抛出自定义的异常
    assert False , f"No matched motion found for prompt: {prompt}"

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
def temp_view( mask, title='Mask', name=None):
    """
    显示输入的mask图像

    参数:
    mask (torch.Tensor): 要显示的mask图像，类型应为torch.bool或torch.float32
    title (str): 图像标题
    """
    # 确保输入的mask是float类型以便于显示
    if isinstance(mask, np.ndarray):
        mask_new = mask
    else:
        mask_new = mask.float()
        mask_new = mask_new.detach().cpu()
        mask_new = mask_new.numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(mask_new, cmap='gray')
    plt.title(title)
    plt.axis('off')  # 去掉坐标轴
    # plt.savefig(name+'.png')
    plt.show()




def temp_view_img(image: Image.Image, title: str = None) -> None:
    # PIL -> ndarray OR ndarray->PIL->ndarray
    if not isinstance(image, Image.Image):  # ndarray
        # image_array = Image.fromarray(image).convert('RGB')
        image_array = image
    else:  # PIL
        if image.mode != 'RGB':
            image.convert('RGB')
        image_array = np.array(image)

    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()


# data = load_json(json_path)
# for i,da in enumerate(data):
#     #check param
#     if 'y-axis' in da['edit_prompt']:
#         if not (da['edit_param'][4]!=0 and da['edit_param'][5]==0):
#             print(da['edit_param'])
#             print(da['edit_prompt'])
#             print(f'error case y {i}')
#             ry, rz = da['edit_param'][4], da['edit_param'][5]
#             da['edit_param'][4] = rz
#             da['edit_param'][5] = 0
#             print(f'cur param {da["edit_param"]}')
# save_json(data,json_path)
# json_path  = "/data/Hszhu/dataset/CelebAMask-HQ/Subset_3/coarse_input_full_pack_3.json"
#
# data = load_json(json_path)
# for da_n,da in data.items():
#     if 'instances' not in da:
#         print(f'skip no instances {da_n}')
#         continue
#     for ins_n,ins_data in da['instances'].items():
#         for edit_id,edit_meta in ins_data.items():
#             #check param
#             if 'y-axis' in edit_meta['edit_prompt']:
#                 if not (edit_meta['edit_param'][4]!=0 and edit_meta['edit_param'][5]==0):
#                     print(edit_meta['edit_param'])
#                     print(edit_meta['edit_prompt'])
#                     print(f'error case y {da_n}')
#                     ry, rz = edit_meta['edit_param'][4], edit_meta['edit_param'][5]
#                     edit_meta['edit_param'][4] = rz
#                     edit_meta['edit_param'][5] = 0
#                     print(f'cur param {edit_meta["edit_param"]}')
#                     data[da_n]['instances'][ins_n][edit_id] = edit_meta
# save_json(data,json_path)
# def post_process_coarse_edit(edit_prompt_list):
#     move_list = []
#     valid_idx = []
#     for idx,edit_prompt in enumerate(edit_prompt_list):
#         motion_type = find_motion_type(edit_prompt)
#         if motion_type == 'move':
#             move_list.append(idx)
#         else:
#             valid_idx.append(idx)
#         # 2. 如果 move 操作多于三个，随机选择三个
#     if len(move_list) > 3:
#         sampled_move_idx = random.sample(move_list, 3)
#     else:
#         sampled_move_idx = move_list  # 如果少于三个，保留所有 move 操作
#     valid_idx = sampled_move_idx + valid_idx
#     return valid_idx
# json_path  = "/data/Hszhu/dataset/CelebAMask-HQ/Subset_0/coarse_input_full_pack_0.json"
#
# data = load_json(json_path)
# for da_n,da in data.items():
#     if 'instances' not in da:
#         print(f'skip no instances {da_n}')
#         continue
#     for ins_n,ins_data in da['instances'].items():
#         ins_edit_prompt_list = [ins_data[edit_id]['edit_prompt'] for edit_id in ins_data]
#         valid_ins_edit_id_list = [edit_id for edit_id in ins_data]
#         valid_idx = post_process_coarse_edit(ins_edit_prompt_list)
#         valid_ins_edit_id_list = np.array(valid_ins_edit_id_list)[valid_idx].tolist()
#         print(f'ori len {len(ins_edit_prompt_list)} , now len {len(valid_ins_edit_id_list)}')
#         for edit_id in ins_data.copy():
#             if edit_id not in valid_ins_edit_id_list:
#                 del ins_data[edit_id]
# save_json(data,json_path)
# json_path_list = [f"/data/Hszhu/dataset/Subjects200K/Subset_{i}/mask_label_filtered_{i}.json" for i in range(4)]
# img_num=0
# ins_num=0
# for file in json_path_list:
#     data = load_json(file)
#     for da_n,da in data.items():
#         if 'instances' not in da:
#             print(f'skip no instances {da_n}')
#             continue
#         img_num+=1
#         ins_num+=len(da['instances']['obj_label'])
# print(f'final img num {img_num} ins num {ins_num}')
# print(f'final img num {img_num} ')
# new_meta = {}
# s_data = load_json("/data/Hszhu/dataset/Subjects200K/meta_data_unique.json")
# cat_info={}
# cat_num=0
# class_num=0
# for da_n,da in s_data.items():
#     cat = da['category']
#     if cat not in cat_info.keys():
#         cat_info[cat] = []
#         cat_num+=1
#     class_name = da.get('item',None)
#     if class_name is not None:
#         if class_name not in cat_info[cat]:
#             cat_info[cat].append(class_name)
#             class_num+=1
#             new_meta[da_n] = da
# print(f'cat num {cat_num} class num {class_num}')
# # save_json(new_meta,'/data/Hszhu/dataset/Subjects200K/meta_data_unique.json')
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
if __name__ == '__main__':
    # 定义 degrees 字典
    dataset_name='Subjects200K'
    id=0
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
    json_path_list = [f"/data/Hszhu/dataset/{dataset_name}/Subset_{id}/structure_completion_{id}.json"]
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
    # print(f"3D Edit Level 1 count: {level_counts_3d['level_1']}")
    # print(f"3D Edit Level 2 count: {level_counts_3d['level_2']}")
    # print(f"3D Edit Level 3 count: {level_counts_3d['level_3']}")
    # print(f"3D Unknown count: {level_counts_3d['unknown']}")
    print(f'img_num:{img_num}, ins_num:{ins_num}, edit_num:{edit_num} error_num:{error_num}')


