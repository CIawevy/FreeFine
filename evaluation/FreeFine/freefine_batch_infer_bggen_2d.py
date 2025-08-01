import os
import sys
sys.path.append('/data/Hszhu/FreeFine') #replace with your own repo path

from src.demo.model import FreeFinePipeline
import torch
import argparse
from src.utils.attention import AttentionStore,register_attention_control,Attention_Modulator,register_attention_control_4bggen
from diffusers import  DDIMScheduler
from src.utils.vis_utils import temp_view,temp_view_img,load_json,get_constrain_areas,prepare_mask_pool,re_edit_2d,dilate_mask,read_and_resize_mask,re_edit_3d,read_and_resize_img,save_json,save_img,read_and_resize_mask_with_dilation
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

def save_inp_img(img, dst_dir_path_gen, da_n, ins_id):
    da_n = str(da_n)
    ins_id = str(ins_id)
    # 创建 da_n 子文件夹
    subfolder_path = os.path.join(dst_dir_path_gen, da_n)
    os.makedirs(subfolder_path, exist_ok=True)

    # 创建 ins_id 子文件夹
    ins_subfolder_path = os.path.join(subfolder_path, ins_id)
    os.makedirs(ins_subfolder_path, exist_ok=True)
    img_path = os.path.join(ins_subfolder_path, f"inp_img.png")

    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved image to {img_path}")
    return img_path
    
class CustomDatasetInpaint(Dataset):
    def __init__(self, data, dst_dir_path_gen, check_exist=True):
        self.cases = []
        self.existing_results = []
        self.dst_dir_path_gen = dst_dir_path_gen

        for da_n, da in tqdm(data.items(), desc='Processing dataset'):
            instances = da.get('instances', {})
            for ins_id, current_ins in instances.items():
                # 只取第一个 edit_ins 的 input_pack，因为不依赖 edit_id
                first_edit_ins = next(iter(current_ins.keys())) if current_ins else None
                if first_edit_ins:
                    input_pack = current_ins[first_edit_ins]
                    expected_path = self._get_expected_path(da_n, ins_id)
                    item = {
                        'da_n': da_n,
                        'ins_id': ins_id,
                        **input_pack  # 包含 input_pack 所有原始信息
                    }
                    if check_exist and osp.exists(expected_path):
                        item['gen_img_path'] = expected_path
                        self.existing_results.append(item)
                    else:
                        self.cases.append(item)
        print(f"Found {len(self.existing_results)} existing results")
        print(f"New cases to process: {len(self.cases)}")

    def _get_expected_path(self, da_n, ins_id):
        da_n = str(da_n)
        ins_id = str(ins_id)
        # 创建 da_n 子文件夹
        subfolder_path = os.path.join(self.dst_dir_path_gen, da_n)
        os.makedirs(subfolder_path, exist_ok=True)

        # 创建 ins_id 子文件夹
        ins_subfolder_path = os.path.join(subfolder_path, ins_id)
        os.makedirs(ins_subfolder_path, exist_ok=True)

        return os.path.join(ins_subfolder_path, f"inp_img.png")

    def get_existing_results(self):
        return self.existing_results

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        return self.cases[idx]


def custom_collate(batch):
    return batch


def main(dst_base,blending):
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 模型加载
    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/" #replace with your own
    model = FreeFinePipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float32).to(device)
    model._progress_bar_config = {"disable": True}
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    controller = Attention_Modulator(start_layer=10)
    model.controller = controller
    register_attention_control_4bggen(model, controller)
    model.modify_unet_forward()
    model.enable_attention_slicing()
    model.enable_xformers_memory_efficient_attention()

    # 数据准备
    dataset_json = osp.join(dst_base, "annotations_2d.json")
    if not blending:
        inp_img_dir = os.path.join(base_dir, f'Geo-Bench-2D/inp_img_no_blend')
    else:
        inp_img_dir = os.path.join(base_dir, f'Geo-Bench-2D/inp_img_blended')

  
    if local_rank == 0:
        os.makedirs(inp_img_dir, exist_ok=True)
    data = load_json(dataset_json)

    dataset = CustomDatasetInpaint(data, inp_img_dir)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1, #Currently, we do not support batchsize > 1 , the attention modulation needs to be modified
        sampler=sampler,
        collate_fn=custom_collate
    )

    image_info = []

    for batch in tqdm(dataloader, desc=f"GPU {local_rank} Processing"):
        case = batch[0]
        da_n, ins_id = case['da_n'], case['ins_id']

        # 加载输入数据
        ori_img_path = case['ori_img_path']
        ori_mask_path = case['ori_mask_path']
       

        ori_img = read_and_resize_img(ori_img_path)
        ori_mask = read_and_resize_mask_with_dilation(ori_mask_path, dilation_factor=30, forbit_area=None)
      
     
        
        # mask_pool = prepare_mask_pool(data[da_n]['instances'])

        # # 自定义输入处理（保持与原有逻辑一致）
        # inp_back_ground = cv2.cvtColor(
        #     cv2.imread(osp.join(dst_base, f"inp_img_no_blend/{da_n}/{ins_id}/inp_img.png")),
        #     cv2.COLOR_BGR2RGB
        # )

        # ori_img = cv2.cvtColor(cv2.imread(ori_img_path), cv2.COLOR_BGR2RGB)
        # coarse_input = cv2.cvtColor(cv2.imread(coarse_input_path), cv2.COLOR_BGR2RGB)
        # target_mask = cv2.imread(tgt_mask_path, cv2.IMREAD_GRAYSCALE)[:, :, None] / 255.0
        seed_r = random.randint(0, 10 ** 16) #bring more diversity to background gen
        # 推理参数
        params = {
            "ori_img": ori_img,
            "ori_mask": ori_mask,
            "guidance_text": "empty scene",
            "guidance_scale": 7.5,
            "eta": 1.0,
            "end_scale": 0.5,
            "end_step": 35,
            "num_step": 50,
            "start_step": 1,
            "seed": seed_r,
            "return_intermediates" : False,
            "use_auto_draw" : False,
            "reduce_inp_artifacts" : True,
        }

        # 生成结果
        generated_results = model.FreeFine_background_generation(**params)
        if blending: #implemented from brushnet
            mask_blurred = cv2.GaussianBlur(ori_mask, (21, 21), 0) / 255
            mask_np = 1 - (1 - ori_mask) * (1 - mask_blurred)
            image_pasted = ori_img * (1 - mask_np) + generated_results * mask_np
            generated_results = image_pasted.astype(generated_results.dtype)
            
        inp_img_path = save_inp_img(generated_results, inp_img_dir, da_n, ins_id)
        # can be index with GeoBenchMeta/Geo-Bench-2D/{INP_IMG_DIR}/{DA_N}/{INS_ID}/inp_img.png

        # 收集信息（保持原始格式，但增加完整路径信息）
        # image_info.append({**case, 'inp_img_path': inp_img_path})

       
        # 收集所有进程的 image_info
    all_image_info_list = [None] * world_size
    dist.all_gather_object(all_image_info_list, image_info)

    # if local_rank == 0:
    #     # 合并新生成的结果和已存在的结果
    #     final_results = dataset.get_existing_results()
    #     for res in all_image_info_list:
    #         final_results.extend(res)

    #     new_data = {}
    #     for item in final_results:
    #         da_n, ins_id, edit_ins = item['da_n'], item['ins_id'], item['edit_ins']
    #         if da_n not in new_data:
    #             new_data[da_n] = {'instances': {}}
    #         if ins_id not in new_data[da_n]['instances']:
    #             new_data[da_n]['instances'][ins_id] = {}
    #         new_data[da_n]['instances'][ins_id][edit_ins] = item  # 直接保存完整item字典

    #     # 修正保存路径，使用dst_base而不是未定义的base_dir
    #     save_json(new_data, os.path.join(dst_base, f"generated_results_freefine_2d_with_inp.json"))
    #     print(f"Total images processed: {len(final_results)}")

    # 清理分布式进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    base_dir = "/data/Hszhu/dataset/GeoBenchMeta/"  #replace with your own
    main(base_dir,blending=True)