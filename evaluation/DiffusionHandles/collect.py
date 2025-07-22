

from vis_utils import  temp_view,temp_view_img,load_json,save_img,save_json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from tqdm import tqdm
import sys
import os.path as osp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import argparse
import scipy
import torch.distributed as dist

from omegaconf import OmegaConf

import imageio
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import Compose
import torchvision
from torch.utils.data import Dataset,DistributedSampler,DataLoader

class MultiModalDataset(Dataset):
    def __init__(self, data_dict, dst_dir_path_gen, check_exist=True):
        self.cases = []
        self.existing_results = []
        self.dst_dir_path_gen = dst_dir_path_gen

        for da_n, da in data_dict.items():
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
        # os.makedirs(ins_subfolder_path, exist_ok=True)
        return os.path.join(ins_subfolder_path, f"{edit_ins}.png")

    def get_existing_results(self):
        return self.existing_results

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        return self.cases[idx]

def latent_to_image(latent_image, diff_handles):

    # save image reconstructed from inversion
    with torch.no_grad():
        latent_image = 1 / 0.18215 * latent_image.detach()
        recon_image = diff_handles.diffuser.vae.decode(latent_image)['sample']
        recon_image = (recon_image + 1) / 2

    return recon_image
def crop_and_resize(img: torch.Tensor, size: int) -> torch.Tensor:
    if img.shape[-2] != img.shape[-1]:
        img = torchvision.transforms.functional.center_crop(img, min(img.shape[-2], img.shape[-1]))
    img = torchvision.transforms.functional.resize(img, size=(size, size), antialias=True)
    return img
def resize_image(image, aspect_ratio):

    # h, w = image.shape[:2]
    ratio = aspect_ratio[1] / aspect_ratio[0]
    h, w = 512, 512

    if ratio < 1:
        new_h, new_w = h / ratio, w
    else:
        new_h, new_w = h, ratio * w

    img = cv2.resize(image, (int(new_w),int(new_h)))

    # input_img = np.array(Image.fromarray(img).resize((w, h), Image.NEAREST))
    return img
def preprocess_image(image_in, img_res=IMG_RES,local_rank=None):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = f'cuda:{local_rank}'
    image = torch.from_numpy(image_in).unsqueeze(0).permute(0, -1, 1, 2)
    # print(image.shape)
    image = crop_and_resize(img=image, size=img_res).to(device).to(torch.float32)

    return image

def dilate_mask(mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask


def custom_collate(batch):
    """自定义collate函数，直接返回原始数据不做任何处理"""
    return batch  # 直接返回原始batch，不做任何修改


def resize_image_and_mask(image, mask, target_size=(512, 512)):
    """
    将图像和掩码统一缩放到目标尺寸

    Args:
        image: 输入图像 (numpy array)
        mask: 输入掩码 (numpy array)
        target_size: 目标尺寸 (width, height)

    Returns:
        resized_image: 缩放后的图像
        resized_mask: 缩放后的掩码
    """
    # 缩放图像 (使用高质量插值方法)
    resized_image = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_LANCZOS4)

    # 缩放掩码 (使用最近邻插值以保持掩码的离散值)
    resized_mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)

    return resized_image, resized_mask
def main(dst_base):
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

   

    # 加载数据
    dataset_json_file = osp.join(dst_base, "annotations.json")
    # 创建保存图像的文件夹
    dst_dir_path_gen = osp.join(dst_base, f"Gen_results_DesignEdit")
    if local_rank == 0:
        os.makedirs(dst_dir_path_gen, exist_ok=True)

    data = load_json(dataset_json_file)
    dataset =MultiModalDataset(data,dst_dir_path_gen=dst_dir_path_gen)
    # 2. 创建分布式采样器
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        shuffle=False,
        drop_last=False
    )

    # 3. 创建数据加载器（无collate_fn，直接返回原始案例）
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        pin_memory=True, collate_fn=custom_collate,
    )


    # 用于存储图像路径和信息的列表
    image_info = []
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

        save_json(new_data, osp.join(base_dir, f"generated_results_DesignEdit.json"))
        print(f"Total images processed: {len(final_results)}")

    # 清理分布式进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    base_dir = "/data/Hszhu/dataset/Geo-Bench/"
    main(base_dir)