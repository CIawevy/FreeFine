import random
import numpy as np
import torch
import os
from typing import Optional
# 映射表：定义每个类别支持的操作类型
celeb_category_operations = {
    "eye": ["move", "enlarge", "shrink"],
    "eyebrow": ["move", "enlarge", "shrink", "rotate"],
    "nose": ["move", "enlarge", "shrink", "rotate"],
    "ear": ["move", "enlarge", "shrink"],
    "mouth": ["move", "enlarge", "shrink", "rotate"],
    "hat": ["move", "enlarge", "shrink", "rotate"],
    "earring": ["enlarge", "shrink","rotate"],
    "person": ["enlarge", "shrink", "rotate"],
}
slight_edit_list = ['eye','eyebrow','nose','ear','mouth']

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
def find_direction(prompt,motion_type):
     directions = operations[motion_type]['directions']
     for direction in directions:
          if direction in prompt:
              return direction


def find_degree(prompt):
    for lvl,des in degrees.items():
        for adj in des['description']:
            if adj in prompt:
                return lvl
    return error


# def post_process_coarse_edit(edit_prompt_list,out_of_img_boundary_list):
#     #modified for evaluation purpose
#     #so far we have every possible edit for every possible direction
#     # now we will do a cliping operation to suppress direction of edit
#     valid_idx = []
#     move_direction_dict = {}
#     for idx,edit_prompt in enumerate(edit_prompt_list):
#         motion_type = find_motion_type(edit_prompt)
#         if motion_type == 'move' and not out_of_img_boundary_list[idx]:
#             direction = find_direction(edit_prompt, motion_type)
#             if direction not in move_direction_dict:
#                 move_direction_dict[direction]={}
#             else:
#                 valid_idx.append(idx)
#             if len(move_direction_dict)<3:
#                 valid_idx.append(idx)
#         else:
#             valid_idx.append(idx)
#
#     return valid_idx

def post_process_coarse_edit(edit_prompt_list,out_of_img_boundary_list):
    #modified for evaluation purpose
    #so far we have every possible edit for every possible direction
    # now we will do a cliping operation to suppress direction of edit
    edit_dict = dict()
    for idx,edit_prompt in enumerate(edit_prompt_list):
        motion_type = find_motion_type(edit_prompt)
        # direction = find_direction(edit_prompt, motion_type)
        degree = find_degree(edit_prompt)
        if motion_type not in edit_dict:
            edit_dict[motion_type]={}
        if degree not in edit_dict[motion_type]:
            edit_dict[motion_type][degree]= []
        edit_dict[motion_type][degree].append(idx)
    #TODO: 任意motion 的任意 degree 采样一个方向
    valid_idx = []
    for motion_type,meta in edit_dict.items():
        for degree , id_list in meta.items():
            valid_idx.append(random.choice(id_list))


    return valid_idx


def my_seed_everything(seed: Optional[int] = None) -> int:
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    In addition, sets the env variable `PL_GLOBAL_SEED` which will be passed to
    spawned subprocesses (e.g. ddp_spawn backend).

    Args:
        seed: the integer value seed for global random state.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """

    def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
        return random.randint(min_seed_value, max_seed_value)

    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
        print(f"No correct seed found, seed set to {seed}")

    if not (min_seed_value <= seed <= max_seed_value):
        print(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
def sample_degree_celeb():
    level = random.choice(list(celeb_degrees.keys()))
    return random.choice(degrees[level]['description'])
def sample_degree():
    level = random.choice(list(degrees.keys()))
    return random.choice(degrees[level]['description'])
def check_degree(descript):
    for lvl in ['level_1','level_2','level_3']:
        lv1_lst = degrees[lvl]['description']
        if descript in lv1_lst:
            return lvl
    return 'unknown error'
def generate_instruction(sample_type='2D',seed=None):
    """
    遍历operations生成随机编辑指令 with degrees
    """
    if seed is not None:
        my_seed_everything(seed)

    instructions = []

    for edit_type,details in operations.items():
        if edit_type!='rotate':
            if sample_type!='2D':
                continue
            description_list = details['descriptions']
            direction_list = details['directions']
            for direction in direction_list:
                description = random.choice(description_list)
                for level in  degrees.keys():
                    degree = random.choice(degrees[level]['description'])
                    # degree = sample_degree()
                    current_ins = f"{description} the {{object}} {direction} {degree}"
                    instructions.append(dict(type=edit_type, prompt=current_ins, direction=direction,degree=check_degree(degree)))
                    # instructions.append(current_ins)
        else: #rotate
            description_list = details['descriptions']
            if sample_type == '2D':
                direction_list = details['directions'][sample_type]
                for direction in direction_list:
                    description = random.choice(description_list)
                    for level in degrees.keys():
                        degree = random.choice(degrees[level]['description'])
                        # degree = sample_degree()
                        current_ins = f"{description} the {{object}} {direction} {degree}"
                        instructions.append(dict(type=edit_type, prompt=current_ins, direction=direction,degree=check_degree(degree)))
            elif sample_type =='3D':
                direction_list = details['directions']['3D_y']
                for direction in direction_list:
                    for lvl in ['level_1','level_2','level_3'] :
                        description = random.choice(description_list)
                        degree = random.choice(degrees[lvl]['description'])
                        current_ins = f"{description} the {{object}} {direction} {degree}"
                        instructions.append(dict(type=edit_type, prompt=current_ins, direction=direction,degree=lvl))
    return instructions
def generate_instruction_celeb(sample_type='2D',seed=None,label=None):
    """
    遍历operations生成随机编辑指令 with degrees
    """
    if seed is not None:
        my_seed_everything(seed)

    instructions = []
    #get allowed operations for specific label
    allow_operations =  celeb_category_operations[label]
    for edit_type in allow_operations:
        details = operations[edit_type]
        if edit_type!='rotate':
            if sample_type!='2D':
                continue
            description_list = details['descriptions']
            direction_list = details['directions']
            for direction in direction_list:
                description = random.choice(description_list)
                if label in slight_edit_list:
                    degree = sample_degree_celeb()
                else:
                    degree = sample_degree()
                current_ins = f"{description} the {{object}} {direction} {degree}"
                instructions.append(dict(type=edit_type, prompt=current_ins, direction=direction,degree=check_degree(degree)))
                # instructions.append(current_ins)
        else: #rotate
            description_list = details['descriptions']
            if sample_type == '2D':
                direction_list = details['directions'][sample_type]
                for direction in direction_list:
                    description = random.choice(description_list)
                    if label in slight_edit_list:
                        degree = sample_degree_celeb()
                    else:
                        degree = sample_degree()
                    current_ins = f"{description} the {{object}} {direction} {degree}"
                    instructions.append(dict(type=edit_type, prompt=current_ins, direction=direction,degree=check_degree(degree)))
            elif sample_type =='3D':
                direction_list = details['directions']['3D_y']
                for direction in direction_list:
                    for lvl in ['level_1','level_2','level_3'] :
                        description = random.choice(description_list)
                        degree = random.choice(degrees[lvl]['description'])
                        current_ins = f"{description} the {{object}} {direction} {degree}"
                        instructions.append(dict(type=edit_type, prompt=current_ins, direction=direction,degree=lvl))
    return instructions
def generate_random_instructions(total_instructions=10, class_ratios=None, direction_ratios=None,
                                 seed=None):
    """
    生成随机编辑指令集。

    参数:
    - operations: 操作类型及其描述词、方向定义的字典。
    - total_instructions: 要生成的总指令数量。
    - class_ratios: 各操作类别生成的占比，默认为均匀分布。
    - direction_ratios: 各方向生成的占比，默认为均匀分布。
    - seed: 随机种子，用于控制随机性。

    返回:
    - 一个包含生成指令的列表。
    """
    if seed is not None:
        my_seed_everything(seed)

    instructions = []

    # 默认情况下，各操作类别均匀分布
    if class_ratios is None:
        class_ratios = {key: 1 / len(operations) for key in operations.keys()}

    # 根据类比例计算每个类需要生成的指令数量
    class_counts = {key: int(class_ratios[key] * total_instructions) for key in operations.keys()}

    # 处理舍入误差，确保总数为 total_instructions
    remaining = total_instructions - sum(class_counts.values())
    for i, key in enumerate(class_counts):
        if i < remaining:
            class_counts[key] += 1

    for operation, details in operations.items():
        for _ in range(class_counts[operation]):
            description = random.choice(details["descriptions"])

            # 选择方向时，根据指定的方向比例进行采样
            if "directions" in details:
                if isinstance(details["directions"], dict):
                    axis_category = random.choice(list(details["directions"].keys()))
                    if direction_ratios and axis_category in direction_ratios:
                        direction = \
                        random.choices(details["directions"][axis_category], weights=direction_ratios[axis_category])[0]
                    else:
                        direction = random.choice(details["directions"][axis_category])
                else:
                    if direction_ratios:
                        direction = random.choices(details["directions"], weights=direction_ratios)[0]
                    else:
                        direction = random.choice(details["directions"])
            else:
                direction = ""

            # 构建指令，留出物体标签和程度填充位置
            instruction = f"{description} the {{object}} {direction} {{degree}}"
            instructions.append(dict(type=operation,prompt=instruction,direction=direction))
    return instructions
#一些更自然但可能有歧义的描述
# "rotate_3D": {
#     "descriptions": ["Rotate", "Spin", "Turn", "Swivel"],
#     "directions": ["to the right", "to the left"] #to the up # to the down 虽然是自然语言但感觉会有歧义
# },
# "scale": {
#         "descriptions": ['push forward','pull backward'],
#     },
# 定义操作及其对应的方向集合和多样化描述





