import os
import sys
sys.path.append('/data/Hszhu/FreeFine') #replace with your own repo path

from src.demo.model import FreeFinePipeline
import torch
import argparse
from src.utils.attention import AttentionStore,register_attention_control,Attention_Modulator
from diffusers import DDIMScheduler
from src.utils.vis_utils import temp_view,temp_view_img,load_json,get_constrain_areas,prepare_mask_pool,re_edit_2d,dilate_mask,read_and_resize_mask,re_edit_3d,read_and_resize_img,save_json,save_img
import os
import json
from tqdm import tqdm
import sys
import os.path as osp
from PIL import Image
import numpy as np
import torch
import cv2
import argparse
import torch.distributed as dist
from src.demo.model import FreeFinePipeline
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import random

def re_edit_2d(src_img, src_mask, edit_param, inp_cur):
    if len(src_mask.shape) == 3:
        src_mask = src_mask[:, :, 0]
    dx, dy, dz, rx, ry, rz, sx, sy, sz = edit_param
    rotation_angle = rz
    resize_scale = (sx, sy)
    flip_horizontal = False
    flip_vertical = False
    # Prepare foreground
    height, width = src_mask.shape[:2]
    y_indices, x_indices = np.where(src_mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        top, bottom = np.min(y_indices), np.max(y_indices)
        left, right = np.min(x_indices), np.max(x_indices)
        # mask_roi = mask[top:bottom + 1, left:right + 1]
        # image_roi = image[top:bottom + 1, left:right + 1]
        mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2
        # 检查是否有移动操作（dx 或 dy 不为零）
        if dx != 0 or dy != 0:
            # 计算物体移动后的新边界
            new_left = left + dx
            new_right = right + dx
            new_top = top + dy
            new_bottom = bottom + dy

            # 检查新边界是否超出图像的边界
            if new_left < 0 or new_right > width or new_top < 0 or new_bottom > height:
                # 如果超出边界，则丢弃或做其他处理
                assert False, 'The transformed object is out of image boundary after move, discard'

    # 将resize_scale解耦出来，实现x，y的单独缩放
    rotation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle, 1)
    # 当rotation angle=0且resize scale!=1时，由mask 中心会影响dx,dy的初始值
    # 计算公式默认rotation angle=0
    tx, ty = (1 - resize_scale[0]) * mask_center_x, (1 - resize_scale[1]) * mask_center_y
    dx += tx
    dy += ty
    rotation_matrix[0, 2] += dx
    rotation_matrix[1, 2] += dy
    rotation_matrix[0, 0] *= resize_scale[0]
    rotation_matrix[1, 1] *= resize_scale[1]

    transformed_image = cv2.warpAffine(src_img, rotation_matrix, (width, height))
    transformed_mask = cv2.warpAffine(src_mask.astype(np.uint8), rotation_matrix, (width, height),
                                      flags=cv2.INTER_NEAREST).astype(bool)

    # # 检查是否需要水平翻转
    # if flip_horizontal:
    #     transformed_mask = cv2.flip(transformed_mask.astype(np.uint8), 1).astype(bool)
    #     # transformed_mask_exp = cv2.flip(transformed_mask_exp.astype(np.uint8), 1).astype(bool)
    #
    # # 检查是否需要垂直翻转
    # if flip_vertical:
    #     transformed_mask = cv2.flip(transformed_mask.astype(np.uint8), 0).astype(bool)
    #     # transformed_mask_exp = cv2.flip(transformed_mask_exp.astype(np.uint8), 0).astype(bool)
    # if np.array_equal(transformed_mask.astype(np.uint8)*255, tgt_mask):
    #     return True,transformed_mask
    # else:
    #     return False,transformed_mask
    final_image = np.where(transformed_mask[:, :, None], transformed_image,
                           inp_cur)  # move with expansion pixels but inpaint
    return final_image, transformed_mask.astype(np.uint8)*255



class CustomDataset(Dataset):
    def __init__(self, data, dst_dir_path_gen, check_exist=True):
        self.cases = []
        self.existing_results = []
        self.dst_dir_path_gen = dst_dir_path_gen

        for da_n, da in data.items():
            instances = da.get('instances', {})
            for ins_id, current_ins in instances.items():
                for edit_ins, input_pack in current_ins.items():
                    expected_path = self._get_expected_path(da_n, ins_id, edit_ins)
                    item = {
                        'da_n': da_n,
                        'ins_id': ins_id,
                        'edit_ins': edit_ins,
                        **input_pack  # 包含input_pack所有原始信息
                    }
                    if check_exist and osp.exists(expected_path):
                        item['gen_img_path'] = expected_path
                        self.existing_results.append(item)
                    else:
                        self.cases.append(item)

        print(f"Found {len(self.existing_results)} existing results")
        print(f"New cases to process: {len(self.cases)}")

    def _get_expected_path(self, da_n, ins_id, edit_ins):
        da_n = str(da_n)
        ins_id = str(ins_id)
        edit_ins = str(edit_ins)
        ins_subfolder_path = os.path.join(self.dst_dir_path_gen, da_n, ins_id)
        os.makedirs(ins_subfolder_path, exist_ok=True)
        return os.path.join(ins_subfolder_path, f"{edit_ins}.png")

    def get_existing_results(self):
        return self.existing_results

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        return self.cases[idx]


def custom_collate(batch):
    return batch


def main(dst_base):
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 模型加载
    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"  #replace with your own
    model = FreeFinePipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float32).to(device)
    model._progress_bar_config = {"disable": True}
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    controller = Attention_Modulator(start_layer=10)
    model.controller = controller
    register_attention_control(model, controller)
    model.modify_unet_forward()
    model.enable_attention_slicing()
    model.enable_xformers_memory_efficient_attention()

    # 数据准备
    dataset_json = osp.join(dst_base, "annotations_2d.json")
    dst_gen_dir = osp.join(dst_base, "Geo-Bench-2D/Gen_results_FreeFine_2d")
    if local_rank == 0:
        os.makedirs(dst_gen_dir, exist_ok=True)
    data = load_json(dataset_json)

    dataset = CustomDataset(data, dst_gen_dir)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,#Currently, we do not support batchsize > 1 , the attention modulation needs to be modified
        sampler=sampler,
        collate_fn=custom_collate
    )

    image_info = []

    for batch in tqdm(dataloader, desc=f"GPU {local_rank} Processing"):
        case = batch[0]
        da_n, ins_id, edit_ins = case['da_n'], case['ins_id'], case['edit_ins']

        # 加载输入数据
        ori_img_path = case['ori_img_path']
        ori_mask_path = case['ori_mask_path']
        # coarse_input_path = case['coarse_input_path']
        # tgt_mask_path = case['tgt_mask_path']
        edit_param = case['edit_param']
        inp_img_path = os.path.join(dst_base,f"Geo-Bench-2D/inp_img_blended/{da_n}/{ins_id}/inp_img.png")
        inp_back_ground = read_and_resize_img(inp_img_path)
        
        ori_img = read_and_resize_img(ori_img_path)
        # target_mask = read_and_resize_mask(tgt_mask_path)
        # coarse_input = read_and_resize_img(coarse_input_path)

        coarse_input, target_mask = re_edit_2d(ori_img, ori_mask, edit_param, inp_back_ground)
        obj_label = ""
        ori_mask = read_and_resize_mask(ori_mask_path)
        draw_mask = np.ones_like(ori_mask)

        # mask_pool = prepare_mask_pool(data[da_n]['instances'])

        # # 自定义输入处理（保持与原有逻辑一致）
        # inp_back_ground = cv2.cvtColor(
        #     cv2.imread(osp.join(dst_base, f"inp_img_no_blend/{da_n}/{ins_id}/inp_img.png")),
        #     cv2.COLOR_BGR2RGB
        # )

        # ori_img = cv2.cvtColor(cv2.imread(ori_img_path), cv2.COLOR_BGR2RGB)
        # coarse_input = cv2.cvtColor(cv2.imread(coarse_input_path), cv2.COLOR_BGR2RGB)
        # target_mask = cv2.imread(tgt_mask_path, cv2.IMREAD_GRAYSCALE)[:, :, None] / 255.0
        # seed_r = random.randint(0, 10 ** 16)
        # 推理参数
        params = {
            "ori_img": ori_img,
            "ori_mask": ori_mask,
            "coarse_input": coarse_input,
            "target_mask": target_mask,
            "guidance_text": obj_label,
            "guidance_scale": 7.5,
            "eta": 1.0,
            "end_scale": 0.0,
            "end_step": 50,
            "num_step": 50,
            "start_step": 35,
            "seed": 42,
            "draw_mask": draw_mask,
            "return_intermediates" : False,
            "use_auto_draw" : True,
            "reduce_inp_artifacts" : True,
            "cons_area" : target_mask,
        }

        # 生成结果
        generated_results = model.FreeFine_generation(**params)
        gen_img_path = save_img(generated_results, dst_gen_dir, da_n, ins_id, edit_ins)

        # 收集信息（保持原始格式，但增加完整路径信息）
        image_info.append({**case, 'gen_img_path': gen_img_path})

        # clear_gpu_memory()

        # 收集所有进程的 image_info
    all_image_info_list = [None] * world_size
    dist.all_gather_object(all_image_info_list, image_info)

    if local_rank == 0:
        # 合并新生成的结果和已存在的结果
        final_results = dataset.get_existing_results()
        for res in all_image_info_list:
            final_results.extend(res)

        new_data = {}
        for item in final_results:
            da_n, ins_id, edit_ins = item['da_n'], item['ins_id'], item['edit_ins']
            if da_n not in new_data:
                new_data[da_n] = {'instances': {}}
            if ins_id not in new_data[da_n]['instances']:
                new_data[da_n]['instances'][ins_id] = {}
            new_data[da_n]['instances'][ins_id][edit_ins] = item  # 直接保存完整item字典

        # 修正保存路径，使用dst_base而不是未定义的base_dir
        save_json(new_data, os.path.join(dst_base, f"generated_results_freefine_2d.json"))
        print(f"Total images processed: {len(final_results)}")

    # 清理分布式进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    base_dir = "/data/Hszhu/dataset/GeoBenchMeta/"  #replace with your own
    main(base_dir)