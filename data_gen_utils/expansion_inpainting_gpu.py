import os
import torch
import torch.multiprocessing as mp
import json
from tqdm import tqdm
import cv2
from src.demo.model import AutoPipeReggio
from lama import lama_with_refine
from ram.models import tag2text
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, AutoencoderKL
from src.utils.attention import register_attention_control, Mask_Expansion_SELF_ATTN
import os.path as osp
import torch.nn as nn


def set_cuda_visible_devices(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def split_data(data, num_splits):
    data_keys = list(data.keys())
    chunk_size = len(data_keys) // num_splits

    data_parts = []
    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_splits - 1 else len(data_keys)
        data_part = {k: data[k] for k in data_keys[start_idx:end_idx]}
        data_parts.append(data_part)

    return data_parts


def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"JSON format error: {file_path}")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
    return None


def save_json(data_dict, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)


def save_imgs(imgs, dst_dir, da_name):
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)
    img_paths = []
    for idx, img in enumerate(imgs):
        img_path = os.path.join(subfolder_path, f"img_{idx + 1}.png")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_paths.append(img_path)
    return img_paths


def save_masks(masks, dst_dir, da_name):
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)
    mask_paths = []
    for idx, mask in enumerate(masks):
        mask_path = os.path.join(subfolder_path, f"mask_{idx + 1}.png")
        cv2.imwrite(mask_path, mask)
        mask_paths.append(mask_path)
    return mask_paths


def load_clip_on_the_main_Model(main_model, device):
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    main_model.clip = model
    main_model.clip_process = preprocess
    return main_model


def process_data_on_gpu(gpu_id, data_part, gpu_count):
    torch.cuda.set_device(gpu_id)  # 确保每个进程使用独立的GPU

    # 需要设置的路径参数
    pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-inpainting/"
    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
    vae_path = "default"
    dst_dir_path_exp = "/data/Hszhu/dataset/PIE-Bench_v1/EXP_masks/"
    dst_dir_path_inp = "/data/Hszhu/dataset/PIE-Bench_v1/inp_imgs/"
    TAG2TEXT_THRESHOLD = 0.64
    ckpt_base_dir = "/data/Hszhu/prompt-to-prompt/GroundingSAM_ckpts"
    TAG2TEXT_CHECKPOINT_PATH = osp.join(ckpt_base_dir, "tag2text_swin_14m.pth")
    DELETE_TAG_INDEX = [idx for idx in range(3012, 3429)]

    device = torch.device(f"cuda:{gpu_id % gpu_count}")

    # 模型和其他资源在子进程中加载
    tag2text_model = tag2text(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                              image_size=384,
                              vit='swin_b',
                              delete_tag_index=DELETE_TAG_INDEX,
                              text_encoder_type='/data/Hszhu/prompt-to-prompt/bert-base-uncased')
    tag2text_model.threshold = TAG2TEXT_THRESHOLD
    tag2text_model.eval()
    tag2text_model = tag2text_model.to(device)

    sd_inpainter = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_inpaint_model_path,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)
    sd_inpainter._progress_bar_config = {"disable": True}
    sd_inpainter.enable_attention_slicing()
    model = AutoPipeReggio.from_pretrained(pretrained_model_path, torch_dtype=torch.float32).to(device)
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(vae_path).to(model.vae.device, model.vae.dtype)

    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    model.inpainter = lama_with_refine(device)
    model.sd_inpainter = sd_inpainter
    model.tag2text = tag2text_model
    model = load_clip_on_the_main_Model(model, device)

    controller = Mask_Expansion_SELF_ATTN(block_size=8, drop_rate=0.5, start_layer=10)
    controller.contrast_beta = 1.67
    controller.use_contrast = True
    model.controller = controller
    register_attention_control(model, controller)
    model.modify_unet_forward()
    model.enable_attention_slicing()
    model.enable_xformers_memory_efficient_attention()
    assist_prompt = ['shadow', 'light']

    # 处理数据部分
    for da_n, da in tqdm(data_part.items(), desc=f'Processing on GPU {gpu_id}'):
        image_path = da['image_path']
        if 'instances' not in da.keys():
            print(f'skip {da_n} for not valid instance')
            continue
        instances = da['instances']
        mask_list = [cv2.imread(path) for path in instances['mask_path']]
        obj_label_list = instances['obj_label']
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        expansion_mask_list, inpainting_imgs_list, lama_inp_list = model.expansion_and_inpainting_func(
            img, mask_list, obj_label_list, max_resolution=512, expansion_step=10,
            max_try_times=5, samples_per_time=5, assist_prompt=assist_prompt)

        mask_path = save_masks(expansion_mask_list, dst_dir_path_exp, da_n)
        instances['exp_mask_path'] = mask_path

        if len(inpainting_imgs_list) > 0:
            best_inp_path = save_imgs(inpainting_imgs_list, dst_dir_path_inp, da_n)
            instances['inp_img_path'] = best_inp_path

        data_part[da_n]['instances'] = instances

    save_json(data_part, f"/data/Hszhu/dataset/PIE-Bench_v1/packed_data_gpu_{gpu_id}_EXP_INP.json")


def main():
    #TODO: contains memory BUG unknown
    dataset_json_file = "/data/Hszhu/dataset/PIE-Bench_v1/packed_data_full_tag.json"
    # 使用spawn启动方法
    mp.set_start_method('spawn', force=True)

    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # gpu_ids = [0,1,2,3]
    # gpu_ids = [0,1]
    set_cuda_visible_devices(gpu_ids)

    data = load_json(dataset_json_file)
    data_parts = split_data(data, len(gpu_ids))


    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=process_data_on_gpu, args=(gpu_id, data_parts[i], len(gpu_ids)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()