import os

os.environ["HF_HOME"] = "/data/zkl/huggingface/"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import cv2
import json
import torch
import time
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from losses import FlowLoss
from torchvision import utils
from omegaconf import OmegaConf
from scipy.ndimage import center_of_mass
from generate import load_model_from_config
from torchvision.transforms.functional import to_tensor
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from flow_viz import flow_to_image
from flow_models.raft.raft import  RAFT
import argparse
from flow_models.raft.raft_utils.utils import InputPadder
from flow_models.raft.raft_utils import flow_viz


def gen_flow(edit_param, mask_path, size):
    if edit_param[0]!=0 or edit_param[1]!=0:
        assert all(x == 0 for x in edit_param[2:9]) == 0
        mask = np.array(Image.open(mask_path).resize((size, size)))
        flow = np.zeros((1, 2, mask.shape[0], mask.shape[1]), dtype=np.float32)
        flow[0, 0, mask == 255] = edit_param[0]
        flow[0, 1, mask == 255] = edit_param[1]
        flow = torch.tensor(flow)
        return flow
    else:
        if edit_param[5]!=0:
            assert all(x == 0 for x in edit_param[6:9]) == 0
            mask = np.array(Image.open(mask_path).resize((size, size)))
            center = center_of_mass(mask)
            matrix = cv2.getRotationMatrix2D(center, -edit_param[5], scale=1.0)
        elif edit_param[6]!=0:
            assert edit_param[6] == edit_param[7]
            mask = np.array(Image.open(mask_path).resize((size, size)))
            center = center_of_mass(mask)
            scale = edit_param[6]
            matrix = np.array([
                [scale, 0, (1 - scale) * center[0]],
                [0, scale, (1 - scale) * center[1]]
            ])
        points = np.where(mask == 255)
        points_array = np.array([points[0], points[1], [1]*len(points[0])])
        rotated_point = np.dot(matrix, points_array)
        distance = rotated_point - np.array(points)
        flow = np.zeros((1, 2, mask.shape[0], mask.shape[1]), dtype=np.float32)
        flow[0, :, mask == 255] = distance.transpose(1,0)
        flow = torch.tensor(flow)
        return flow


def get_model():
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "./chkpts/sd-v1-4.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()
    sampler = DDIMSamplerWithGrad(model)
    torch.set_grad_enabled(False)
    return model, sampler


def gen_image(model, sampler, target_flow, src_img, prompt, guidance_schedule, save_path, num_ddim_steps=50, edit_mask=torch.zeros(1,4,64,64).bool()):
    guidance_energy = FlowLoss(100.0, 3.0, oracle=False, target_flow=target_flow, occlusion_masking=True).cuda()
    uncond_embed = model.get_learned_conditioning([""])
    cond_embed = model.get_learned_conditioning([prompt])

    # Sample
    sample, start_zt, info = sampler.sample(
        num_ddim_steps=num_ddim_steps,
        cond_embed=cond_embed,
        uncond_embed=uncond_embed,
        batch_size=1,
        shape=[4, 64, 64],
        CFG_scale=7.5,
        eta=0.0,
        src_img=src_img,
        start_zt=None,
        guidance_schedule=guidance_schedule,
        cached_latents=None,
        edit_mask=edit_mask,
        num_recursive_steps=10,
        clip_grad=200.0,
        guidance_weight=300.0,
        log_freq=0,
        results_folder='test',
        guidance_energy=guidance_energy
    )
    # Decode sampled latent
    sample_img = model.decode_first_stage(sample)
    sample_img = torch.clamp((sample_img + 1.0) / 2.0, min=0.0, max=1.0)
    # utils.save_image(sample_img, save_path)


def main():
    file = '/data/Hszhu/dataset/Geo-Bench/annotations.json'
    data = json.load(open(file))
    model, sampler = get_model()
    guidance_schedule = np.load('guidance_schedule.npy')
    # wrong = []

    start_time = time.time()
    num =0
    for k in list(data.keys()):
        image = data[k]
        prompt = image["4v_caption"]
        instances = image["instances"]
        for instance_key in instances.keys():
            instance = instances[instance_key]
            for sample_key in instance.keys():
                sample = instance[sample_key]

                save_path = sample['coarse_input_path'].replace("coarse_img", "motion").replace("Hszhu", "zkl")
                # if os.path.exists(save_path):
                #     data[k]["instances"][instance_key][sample_key]["gen_img_path"] = save_path
                #     continue
                # else:
                #     sample["prompt"] = prompt
                #     wrong.append(sample)
                #     continue

                # dirname = os.path.dirname(save_path)
                # if not os.path.exists(dirname):
                #     os.makedirs(dirname)

                edit_param = sample["edit_param"]

                image_path = sample["ori_img_path"]
                mask = sample["ori_mask_path"]

                src_img = to_tensor(Image.open(image_path))[None] * 2 - 1
                src_img = src_img.cuda()
                target_flow = gen_flow(edit_param, mask, src_img.shape[-1])
                gen_image(model, sampler, target_flow, src_img, prompt, guidance_schedule, save_path)
                num += 1
                print(num)
                if num == 10:

                # data[k]["instances"][instance_key][sample_key]["gen_img_path"] = 
                                
                    end_time = time.time()
                    print(f"运行时间: {end_time - start_time:.2f} 秒")
                    exit()
    # json.dump(data, open("regiondrag-sc.json", "w"))
    # json.dump(wrong, open("motion-wrong.json", "w"))


def main_3D():
    file = '3D-sample.json'
    data = json.load(open(file))
    l = int(len(data)/5)
    i = 4
    model, sampler = get_model()
    guidance_schedule = np.load('data/guidance_schedule.npy')
    for sample in tqdm(data[l*i: l*(i+1)]):
        save_path = sample['coarse_input_path'].replace("coarse_img", "motion").replace("Hszhu", "zkl")
        if os.path.exists(save_path):
            continue
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        edit_param = sample["edit_param"]
        image_path = sample["ori_img_path"]
        mask = sample["ori_mask_path"]
        prompt = sample['prompt']

        src_img = to_tensor(Image.open(image_path))[None] * 2 - 1
        src_img = src_img.cuda()
        # target_flow = gen_flow(edit_param, mask, src_img.shape[-1])
        target_flow = get_flow(image_path, sample['coarse_input_path'])
        gen_image(model, sampler, target_flow, src_img, prompt, guidance_schedule, save_path)


def signle_image():
    sample = {'edit_prompt': 'Contract the wine bottles uniformly appreciably', 'edit_param': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6197456854570983, 0.6197456854570983, 1.0], 'ori_img_path': 'dataset/Geo-Bench/source_img/455.png', 'coarse_input_path': '/data/Hszhu/dataset/Geo-Bench/coarse_img/455/0/3.png', 'ori_mask_path': 'dataset/Geo-Bench/source_mask/455/0.png', 'tgt_mask_path': 'dataset/Geo-Bench/target_mask/455/0/3.png', 'obj_label': 'wine bottles', '4v_caption': 'The image captures a serene moment in a vineyard at sunset. Two bottles of Chateau Ste. Michelle wine, one green and one red, stand side by side on a wooden crate. The green bottle is on the left, while the red bottle is on the right. Between the two bottles, a small amount of red wine has been poured into a wine glass, adding a touch of elegance to the scene.\n\nThe vineyard stretches out in the background, with rows upon rows of grapevines bathed in the warm glow of the setting sun. The sky above is a clear blue, providing a beautiful contrast to the vibrant colors of the vineyard and the wine bottles.\n\nThe perspective of the image is from the side of the bottles, giving a sense of depth and dimension to the scene. The overall composition of the image suggests a tranquil evening spent in the heart of nature, enjoying the beauty of the vineyard and the taste of fine wine.', 'ours': 'dataset/Geo-Bench/ours/455/0/3.png', 'selfguide': 'dataset/Geo-Bench/selfguide/455/0/3.png', 'dragon': 'dataset/Geo-Bench/dragon/455/0/3.png', 'motion': 'dataset/Geo-Bench/motion/455/0/3.png', 'regiondrag': 'dataset/Geo-Bench/regiondrag/455/0/3.png', 'prompt': 'dataset/Geo-Bench/prompt/455/0/3.png'}

    model, sampler = get_model()
    guidance_schedule = np.load('data/guidance_schedule.npy')
    edit_param = sample["edit_param"]
    image_path = sample["ori_img_path"]
    coarse_input_path = sample["coarse_input_path"]
    mask = sample["ori_mask_path"]
    t_mask = sample["tgt_mask_path"]
    prompt = sample['4v_caption']

    src_img = to_tensor(Image.open(image_path))[None] * 2 - 1
    src_img = src_img.cuda()
    target_flow = gen_flow(edit_param, mask, src_img.shape[-1])
    # target_flow = get_flow(image_path, coarse_input_path)

    flow = target_flow.squeeze(0).permute(1,2,0).cpu().numpy()
    flow = flow_to_image(flow)
    flow_image = Image.fromarray(flow)
    flow_image.save("flow.png")

    t_mask = np.array(Image.open(t_mask).resize((64, 64)))
    o_mask = np.array(Image.open(mask).resize((64, 64)))
    edit_mask = ((t_mask + o_mask) > 0)
    x, y = np.where(edit_mask > 0)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    edit_mask[x_min:x_max+1, y_min:y_max+1] = 1
    Image.fromarray(edit_mask).save("edit_mask.png")
    edit_mask = torch.logical_not(torch.tensor(edit_mask).unsqueeze(0).unsqueeze(0)).repeat(1, 4, 1, 1)

    gen_image(model, sampler, target_flow, src_img, prompt, guidance_schedule, "test.png", 500)


def get_flow(image_path, coarse_input_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="chkpts/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to("cuda")
    model.eval()

    def load_image(imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to("cuda")
    with torch.no_grad():
        image1 = load_image(image_path)
        image2 = load_image(coarse_input_path)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    return flow_up


if __name__ == '__main__':
    main()
