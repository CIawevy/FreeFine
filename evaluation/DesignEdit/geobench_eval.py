import os
import os.path as osp
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
from tqdm import  tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.demo.model import DesignEdit
import torch
def read_and_resize_img(ori_img_path,dsize=(512,512)):
    ori_img = cv2.imread(ori_img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_img = cv2.resize(ori_img, dsize=dsize, interpolation=cv2.INTER_LANCZOS4)
    return ori_img
def read_and_resize_mask(ori_mask_path,dsize=(512,512)):
    #return 3-channel mask ndarray
    ori_mask = cv2.imread(ori_mask_path)

    ori_mask = cv2.resize(ori_mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
    return ori_mask

def cleanup():
    dist.destroy_process_group()
def temp_view_img(image: Image.Image, title: str = None) -> None:
    # Convert to ndarray if the input is not already in that format
    if not isinstance(image, Image.Image):  # ndarray
        image_array = image
    else:  # PIL
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)

    # Function to crop white borders
    def crop_white_borders(img_array):
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
        mask = gray < 255  # Mask of non-white pixels
        coords = np.argwhere(mask)  # Find the coordinates of the non-white pixels
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        return img_array[x0:x1+1, y0:y1+1]

    # Crop the white borders
    cropped_image_array = crop_white_borders(image_array)

    # Display the cropped image
    fig, ax = plt.subplots()
    ax.imshow(cropped_image_array)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')  # Hide the axis

    # Remove the white border around the figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)

    # Set the position of the axes to fill the entire figure
    ax.set_position([0, 0, 1, 1])

    # Show the image
    plt.show()
def visualize_rgb_image(image: Image.Image, title: str = None) -> None:
    """
    Visualize an RGB image from a PIL Image format with an optional title.

    Parameters:
    image (PIL.Image.Image): The RGB image represented as a PIL Image.
    title (str, optional): The title to display above the image.

    Raises:
    ValueError: If the input is not a PIL Image or is not in RGB mode.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    if image.mode != 'RGB':
        raise ValueError("Input image must be in RGB mode.")

    image_array = np.array(image)

    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()
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
def save_mask(mask, dst_dir, da_name, ins_name,sample_id):
    da_name = str(da_name)
    ins_name = str(ins_name)
    sample_id = str(sample_id)
    # 创建da子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 创建ins子文件夹
    ins_subfolder_path = os.path.join(subfolder_path, ins_name)
    os.makedirs(ins_subfolder_path, exist_ok=True)

    # 保存mask到ins子文件夹中
    mask_path = os.path.join(ins_subfolder_path, f"{sample_id}.png")
    cv2.imwrite(mask_path, mask.astype(np.uint8)*255)
    print(f"Saved mask to {mask_path}")

    return mask_path
def save_img(img, dst_dir, da_name, ins_name,sample_id):
    # 保存img到ins子文件夹中
    img_path = get_save_path_edit(dst_dir, da_name, ins_name,sample_id)
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved image to {img_path}")

    return img_path
def get_save_path_edit(dst_dir, da_name, ins_name,sample_id):
    da_name = str(da_name)
    ins_name = str(ins_name)
    sample_id = str(sample_id)
    # 创建da子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 创建ins子文件夹
    ins_subfolder_path = os.path.join(subfolder_path, ins_name)
    os.makedirs(ins_subfolder_path, exist_ok=True)

    # 保存img到ins子文件夹中
    img_path = os.path.join(ins_subfolder_path, f"{sample_id}.png")

    return img_path
def save_json(data_dict, file_path):
    """
    将字典保存为 JSON 文件

    Args:
        data_dict (dict): 需要保存的字典
        file_path (str): JSON 文件的保存路径
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
def save_masks(masks, dst_dir, da_name):
    # 创建子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 用于存储保存的mask路径
    mask_paths = []

    # 保存每个mask到子文件夹中
    for idx, mask in enumerate(masks):
        mask_path = os.path.join(subfolder_path, f"mask_{idx + 1}.png")
        cv2.imwrite(mask_path, mask)  # 将mask保存为png图片 (注意：mask是二值图，乘以255以得到可见的结果)
        print(f"Saved mask {idx + 1} to {mask_path}")
        mask_paths.append(mask_path)

    return mask_paths
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
import random
def split_data(data, num_splits, subset_num=None,seed=None):
    if seed is not None:
        random.seed(seed)
    data_keys = list(data.keys())

    # 如果需要从数据中随机抽取100个
    if subset_num is not None:
        data_keys = random.sample(data_keys, subset_num)  # 随机抽取subset_num个键
    else:
        random.shuffle(data_keys)  # 随机打乱数据键

    chunk_size = len(data_keys) // num_splits
    data_parts = []

    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_splits - 1 else len(data_keys)
        data_part = {k: data[k] for k in data_keys[start_idx:end_idx]}
        data_parts.append(data_part)

    return data_parts


class GeoBenchDataset(Dataset):
    def __init__(self, data,existing_data):
        self.ori_data = data
        self.keys = list(data.keys())
        self.samples = self._prepare_samples(data,existing_data)


    def _prepare_samples(self, data,existing_data):
        samples = []
        for da_n, da in data.items():
            instances = da.get('instances', {})
            for ins_id, current_ins in instances.items():
                if not current_ins:
                    continue
                for edit_ins, coarse_input_pack in current_ins.items():
                    is_processed = da_n in existing_data and \
                                  ins_id in existing_data[da_n].get('instances', {}) and \
                                  edit_ins in existing_data[da_n]['instances'][ins_id]
                    if is_processed:
                        print(f'da_n:{da_n},ins_id:{ins_id},edit_ins:{edit_ins} already exist')
                        continue
                    else:
                        samples.append({
                            'da_n': da_n,
                            'ins_id': ins_id,
                            'edit_ins': edit_ins,
                            'coarse_input_pack': coarse_input_pack
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # 实际加载数据（示例）
        pack = item['coarse_input_pack']
        try:
            ori_img = cv2.imread(pack['ori_img_path'])
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            ori_mask = cv2.imread(pack['ori_mask_path'])
            return {
                'da_n': item['da_n'],
                'ins_id': item['ins_id'],
                'edit_ins': item['edit_ins'],
                'ori_img': ori_img,  # 实际加载数据
                'ori_mask': ori_mask,  # 实际加载数据
                'edit_param': pack['edit_param'],
                'coarse_input_pack': pack
            }
        except Exception as e:
            print(f"Error loading {pack['ori_img_path']}: {str(e)}")
            return None  # 或实现跳过逻辑


def collate_fn(batch):
    # 过滤可能的None（加载失败的情况）
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # 按字段重组batch（适用于分布式训练）
    collated = {
        'da_n': [item['da_n'] for item in batch],
        'ins_id': [item['ins_id'] for item in batch],
        'edit_ins': [item['edit_ins'] for item in batch],
        'ori_img': [item['ori_img'] for item in batch],
        'ori_mask': [item['ori_mask'] for item in batch],
        'edit_param': [item['edit_param'] for item in batch],
        'coarse_input_pack': [item['coarse_input_pack'] for item in batch]
    }
    return collated
from torch.distributed import init_process_group,destroy_process_group
def set_up():
    local_rank = int(os.environ['LOCAL_RANK'])
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
def main():
    # torchrun自动初始化
    set_up()
    rank = int(os.environ['RANK'])
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda:{}".format(local_rank))

    seed =42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 初始化路径和模型
    base_dir = "/data/Hszhu/dataset/Geo-Bench/"
    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-xl-base-1.0"
    dataset_json_file = osp.join(base_dir, "annotations.json")
    dst_dir_path_gen = osp.join(base_dir, "Gen_results_DesignEdit/")
    results_file = osp.join(base_dir, "generated_results_DesignEdit.json")

    # 仅rank 0创建目录
    if rank == 0:
        os.makedirs(dst_dir_path_gen, exist_ok=True)


    # 加载模型
    model = DesignEdit(pretrained_model_path=pretrained_model_path,device=dist.get_rank())
    if torch.cuda.device_count()>1:
        model = DataParallel(model)

    existing_data = load_json(results_file) if osp.exists(results_file) else {}


    # 数据加载器
    dataset = GeoBenchDataset(load_json(dataset_json_file),existing_data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            collate_fn=collate_fn)

    # 主循环
    for batch in tqdm(dataloader, desc=f'Rank {rank} processing', disable=rank != 0):
        item = batch  # 因为batch_size=1
        da_n, ins_id, edit_ins = item['da_n'][0], item['ins_id'][0], item['edit_ins'][0]

        # 检查是否已处理（通过广播）
        if rank == 0:
            is_processed = da_n in existing_data and \
                           ins_id in existing_data[da_n].get('instances', {}) and \
                           edit_ins in existing_data[da_n]['instances'][ins_id]
        else:
            is_processed = None#好像没什么用

        # 广播处理状态
        is_processed = torch.tensor([int(is_processed)], device=local_rank) if rank == 0 else torch.tensor([0],
                                                                                                           device=local_rank)
        dist.broadcast(is_processed, src=0)

        if is_processed.item():
            continue

        # 执行推理
        dx, dy, _, _, _, rz, sx, _, _ = item['edit_param'][0]
        with torch.no_grad():
            results = model.module.infer_2d_edit(
                item['ori_img'][0], item['ori_img'][0], item['ori_mask'][0],
                dx / 512, dy / -512, sx, -rz
            )

        # 保存生成图像
        gen_img_path = get_save_path_edit(dst_dir_path_gen, da_n, ins_id, edit_ins)
        cv2.imwrite(gen_img_path, cv2.cvtColor(results[0], cv2.COLOR_RGB2BGR))

        # 准备更新数据
        update_data = {
            da_n: {
                'instances': {
                    ins_id: {
                        edit_ins: {**item['coarse_input_pack'][0], 'gen_img_path': gen_img_path}
                    }
                }
            }
        }

        # 仅rank 0更新并保存结果
        if rank == 0:
            # 深度合并字典
            def deep_update(target, src):
                for k, v in src.items():
                    if isinstance(v, dict):
                        target[k] = deep_update(target.get(k, {}), v)
                    else:
                        target[k] = v
                return target

            existing_data = deep_update(existing_data, update_data)
            save_json(existing_data, results_file)

        dist.barrier()  # 确保所有rank同步
        torch.cuda.empty_cache()

    cleanup()

if __name__ == "__main__":
    main()