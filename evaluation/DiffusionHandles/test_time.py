
from vis_utils import  temp_view,temp_view_img,load_json,save_img,save_json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
import time
from simple_lama_inpainting import SimpleLama
from omegaconf import OmegaConf
from diffhandles import DiffusionHandles  # 新增：导入DiffusionHandles模型
# from test.utils import crop_and_resize, load_image, save_image, save_depth
from saicinpainting import LamaInpainter
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import imageio
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dpt import DepthAnything
import torchvision
# Borrowed and Edited from:
# 1. https://github.com/adobe-research/DiffusionHandles/blob/main/test/remove_foreground.py
# 2. https://github.com/adobe-research/DiffusionHandles/blob/main/test/estimate_depth.py
# 3.
# Thank you to the authors for the code!


# import torch
# import torchvision
# import imageio.v3 as imageio
# import imageio.plugins as imageio_plugins
#
# imageio_plugins.freeimage.download() # to load exr files
DIFF_HANDLES_MODEL = None
DEPTH_MODEL = None
INPAINTING_MODEL = None
DEPTH_ANYTHING_MODEL = None

IMG_RES = 512
def clear_gpu_memory():
    """基础显存清理函数"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.ipc_collect()  # 收集IPC内存
def convert_transform_to_diffhandles(transform_in):

    # Convert to and from pytorch3d coordinate frame
    M = np.eye(4)
    M[0, 0] = -1.0
    M[1, 1] = -1.0
    transform = np.linalg.inv(M) @ transform_in @ M

    # transform = transform_in @ M
    translation = list(transform[:3, -1])
    rotation = transform[:3, :3]
    rot_scipy = R.from_matrix(rotation)
    axis = rot_scipy.as_rotvec(degrees=True)

    angle = np.linalg.norm(axis) + 1e-8


    axis = (axis / angle)

    if np.linalg.norm(axis) < 0.1:
        axis = np.array([0.0, 1.0, 0.0])

    axis = list(axis)
    return axis, angle, translation
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
def preprocess_image(image_in, img_res=IMG_RES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def remove_foreground(image, fg_mask, img_res=IMG_RES, dilation=10):
    """
    Both image and fg_mask in range [0-1]
    """

    global INPAINTING_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if INPAINTING_MODEL is None:
        # inpainter = LamaInpainter()
        inpainter = SimpleLama()
        # inpainter.to(device)
        INPAINTING_MODEL = inpainter
    else:
        inpainter = INPAINTING_MODEL

    # image = preprocess_image(image, img_res)
    # print(image.shape)

    # fg_mask = preprocess_image(fg_mask, img_res)

    # fg_mask = (fg_mask > 0.5).to(device=device, dtype=torch.float32)

    # inpaint the foreground region to get a background image without the foreground object
    fg_mask = dilate_mask(fg_mask, dilation)
    # if dilation >= 0:
    #     fg_mask = fg_mask.cpu().numpy() > 0.5
    #     fg_mask = scipy.ndimage.binary_dilation(fg_mask[0, 0], iterations=dilation)[None, None, ...]
    #     fg_mask = torch.from_numpy(fg_mask).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(fg_mask).convert('L')
        bg_img = inpainter(image_pil, mask_pil)
        # bg_img = inpainter.inpaint(image=image, mask=fg_mask)

    return bg_img


@torch.no_grad()
def get_monocular_depth_anything(image, encoder="vitl", translate_factor=0.1):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    global DEPTH_ANYTHING_MODEL
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if DEPTH_ANYTHING_MODEL is None:
        # depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)
        # DEPTH_ANYTHING_MODEL = depth_anything
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }

        # encoder = 'vitl'  # or 'vitb', 'vits'
        depth_anything = DepthAnything(model_configs[encoder])
        depth_anything.load_state_dict(torch.load(f'/data/Hszhu/prompt-to-prompt/depth-anything/depth_anything_{encoder}14.pth'))
        depth_anything.to(DEVICE).eval()
        DEPTH_ANYTHING_MODEL = depth_anything

    else:
        depth_anything = DEPTH_ANYTHING_MODEL

    image = image.astype("uint8") / 255.0

    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    depth = depth_anything(image)

    depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    # Converts from relative to absolute depth. This works better than using 1/depth
    depth = depth.max() - depth

    # Translation factor computation
    # # Pushes object farther off to reduce smearing
    # depth = depth + depth.max() * translate_factor

    return depth.unsqueeze(0).unsqueeze(0)


def estimate_depth(image, img_res=IMG_RES):
    global DEPTH_MODEL, DEPTH_ANYTHING_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_depth_anything = get_monocular_depth_anything(image)

    if DEPTH_MODEL is None:

        conf = get_config("zoedepth_nk", "infer")
        depth_estimator = build_model(conf)
        depth_estimator.to(device)
        DEPTH_MODEL = depth_estimator
    else:
        depth_estimator = DEPTH_MODEL

    image_d = preprocess_image(image)

    with torch.no_grad():
        depth = depth_estimator.infer(image_d)

    # print(depth.shape, d_depth_anything.shape)

    return depth[0], d_depth_anything[None]
def estimate_depth_zoe(image, img_res=IMG_RES):
    global DEPTH_MODEL, DEPTH_ANYTHING_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # d_depth_anything = get_monocular_depth_anything(image)

    if DEPTH_MODEL is None:

        conf = get_config("zoedepth_nk", "infer")
        depth_estimator = build_model(conf)
        depth_estimator.to(device)
        DEPTH_MODEL = depth_estimator
    else:
        depth_estimator = DEPTH_MODEL

    image_d = preprocess_image(image)

    with torch.no_grad():
        depth = depth_estimator.infer(image_d)

    # print(depth.shape, d_depth_anything.shape)
    return depth[0]
def load_diffhandles_model(config_path=None):

    global DIFF_HANDLES_MODEL
    if DIFF_HANDLES_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diff_handles_config = OmegaConf.load(config_path) if config_path is not None else None
        diff_handles = DiffusionHandles(conf=diff_handles_config)
        diff_handles.to(device)
        DIFF_HANDLES_MODEL = diff_handles
    else:
        diff_handles = DIFF_HANDLES_MODEL

    return diff_handles

def split_data(data, num_splits):
    # if seed is not None:
    #     random.seed(seed)
    data_keys = list(data.keys())
    #
    # # 如果需要从数据中随机抽取100个
    # if subset_num is not None:
    #     data_keys = random.sample(data_keys, subset_num)  # 随机抽取subset_num个键
    # else:
    #     random.shuffle(data_keys)  # 随机打乱数据键

    chunk_size = len(data_keys) // num_splits
    data_parts = []

    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_splits - 1 else len(data_keys)
        data_part = {k: data[k] for k in data_keys[start_idx:end_idx]}
        data_parts.append(data_part)

    return data_parts
def get_expected_path( dst_dir,da_n, ins_id, edit_ins):
    da_n = str(da_n)
    ins_id = str(ins_id)
    edit_ins = str(edit_ins)
    ins_subfolder_path = os.path.join(dst_dir, da_n, ins_id)
    # os.makedirs(ins_subfolder_path, exist_ok=True)
    return os.path.join(ins_subfolder_path, f"{edit_ins}.png")


def main(dst_base, data_id):
    total_time = 0.0
    sample_count = 0
    max_samples = 100  # 设置最大样本数
    diy = False
    dst_dir_path_gen = osp.join(dst_base, f"Gen_results_diffusionhandles")
    os.makedirs(dst_dir_path_gen, exist_ok=True)
    dataset_json_file = osp.join(dst_base, "annotations.json")
    data = load_json(dataset_json_file)
    data_parts = split_data(data, 8)

    # 设备设置（保持一致）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    new_data = dict()
    diff_handles = load_diffhandles_model()

    for da_n, da in tqdm(data_parts[data_id].items(), desc=f'Proceeding editing {data_id}'):
        if da_n in new_data.keys() and 'instances' in new_data[da_n].keys():
            print(f'skip {da_n} for already exist')
            continue
        instances = da['instances']
        for ins_id, current_ins in instances.items():
            if len(current_ins) == 0:
                print(f'skip empty ins')
                continue

            for edit_ins, coarse_input_pack in current_ins.items():
                # 达到最大样本数时停止
                if sample_count >= max_samples:
                    print(f"\n完成 {sample_count} 个样本的推理测试")
                    print(f"总推理时间: {total_time:.4f} 秒")
                    print(
                        f"平均每个样本推理时间: {total_time / sample_count:.4f} 秒 ({(total_time / sample_count) * 1000:.2f} 毫秒) diffusion handles")
                    return

                # expected_path = get_expected_path(dst_dir_path_gen, da_n, ins_id, edit_ins)
                # if osp.exists(expected_path):
                #     print(f'skip {expected_path} for already exist')
                #     continue

                # 加载图像（排除在计时外）
                ori_img = cv2.imread(coarse_input_pack['ori_img_path'])  # bgr
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
                image = ori_img / 255.0

                ori_mask = cv2.imread(coarse_input_pack['ori_mask_path'])
                ori_mask = cv2.resize(ori_mask, dsize=ori_img.shape[:2], interpolation=cv2.INTER_NEAREST)
                mask = ori_mask[..., :1] / 255.0

                # 解析参数
                edit_param = coarse_input_pack['edit_param']
                dx, dy, _, _, _, rz, sx, sy, _ = edit_param
                dx /= -512
                dy /= -512
                if rz != 0:
                    rot_axis = [0.0, 0.0, 1.0]
                    rot_angle = rz
                    translation = [dx, dy, 0.0]
                    scale_factor = [sx, sy, 1.0]
                else:
                    rot_axis = [0.0, 1.0, 0.0]
                    rot_angle = 0.0
                    translation = [dx, dy, 0.0]
                    scale_factor = [sx, sy, 1.0]

                # 图像预处理（排除在计时外）
                start_time = time.time()
                im_removed = remove_foreground(ori_img, mask)
                im_removed_np = np.array(im_removed)
                # torch.cuda.synchronize()  # 同步GPU
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                im_removed_depth = get_monocular_depth_anything(im_removed_np)
                fg_mask = preprocess_image(mask)
                depth = get_monocular_depth_anything(ori_img)
                bg_depth = im_removed_depth

                # 归一化处理（排除在计时外）
                depth = depth / (depth.max() + 1e-8) + 1e-2
                depth[depth > 0.95] = 1.0
                bg_depth = bg_depth / (bg_depth.max() + 1e-8) + 1e-2
                prompt = ""
                is_2D_transform= True
                if is_2D_transform:
                    depth[fg_mask > 0.5] = 0.5

                # 开始计时（仅测量模型推理部分）
                # torch.cuda.synchronize()  # 同步GPU


                # 模型推理部分
                start_time = time.time()
                bg_depth = diff_handles.set_foreground(depth=depth, fg_mask=fg_mask, bg_depth=bg_depth)
                img = preprocess_image(image)
                null_text_emb, init_noise = diff_handles.invert_input_image(img, depth, prompt)
                null_text_emb, init_noise, activations, latent_image = diff_handles.generate_input_image(
                    depth=depth, prompt=prompt, null_text_emb=null_text_emb, init_noise=init_noise)
                recon_image = latent_to_image(latent_image, diff_handles)

                # 转换参数并执行变换
                translation = torch.tensor(translation, dtype=torch.float32)
                rot_axis = torch.tensor(rot_axis, dtype=torch.float32)
                scale_factor = torch.tensor(scale_factor, dtype=torch.float32)
                rot_angle = float(rot_angle)

                generated_results = diff_handles.transform_foreground(
                    depth=depth, prompt=prompt,
                    fg_mask=fg_mask, bg_depth=bg_depth,
                    null_text_emb=null_text_emb, init_noise=init_noise,
                    activations=activations,
                    rot_angle=rot_angle, rot_axis=rot_axis,
                    translation=translation, scale_factor=scale_factor,
                    use_input_depth_normalization=False)[0]

                # 结束计时
                # torch.cuda.synchronize()  # 同步GPU
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                sample_count += 1

                # 输出进度
                if sample_count % 10 == 0:
                    print(
                        f"已处理 {sample_count}/{max_samples} 样本，当前平均时间: {(total_time / sample_count) * 1000:.2f} 毫秒")

                # 保存图像（排除在计时外）
                # generated_results = generated_results[0].permute(1, 2, 0).detach().cpu().numpy()*255
                # gen_img_path = save_img(generated_results, dst_dir_path_gen, da_n, ins_id, edit_ins)

    # 处理完所有数据后的统计
    if sample_count > 0:
        print(f"\n完成 {sample_count} 个样本的推理测试")
        print(f"总推理时间: {total_time:.4f} 秒")
        print(
            f"平均每个样本推理时间: {total_time / sample_count:.4f} 秒 ({(total_time / sample_count) * 1000:.2f} 毫秒) Diffusion Handles")
    else:
        print("未处理任何样本")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="单卡分批处理脚本")
    parser.add_argument('--data_id', type=int, default=0, help='批次ID (从0开始)')
    args = parser.parse_args()
    base_dir = "/data/Hszhu/dataset/Geo-Bench/"
    main(base_dir,args.data_id)