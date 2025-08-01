import os
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from types import SimpleNamespace
from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from drag_pipeline import DragPipeline
from utils.drag_utils import drag_diffusion_update
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from pytorch_lightning import seed_everything
from utils.lora_utils import train_lora
from tqdm import tqdm
from scipy.ndimage import center_of_mass
from einops import rearrange
import json
from PIL import Image
import cv2
import random


def get_transform_coordinates(edit_param, size, mask):
    if edit_param[0]!=0 or edit_param[1]!=0:
        assert all(x == 0 for x in edit_param[2:9]) == 0
        points = np.zeros((size[0], size[1], 2))
        for i in range(size[0]):
            for j in range(size[1]):
                points[i, j, 0] = i + edit_param[1]
                points[i, j, 1] = j + edit_param[0]
        return points
    else:
        center = center_of_mass(mask)
        if edit_param[5]!=0:
            assert all(x == 0 for x in edit_param[6:9]) == 0
            matrix = cv2.getRotationMatrix2D(center, edit_param[5], scale=1.0)
        elif edit_param[6]!=0:
            assert edit_param[6] == edit_param[7]
            scale = edit_param[6]
            matrix = np.array([
                [scale, 0, (1 - scale) * center[0]],
                [0, scale, (1 - scale) * center[1]]
            ])
        y, x = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
        ones = np.ones_like(x)
        points = np.stack((x, y, ones), axis=-1).reshape(-1, 3)
        rotated_point = np.dot(points, matrix.T).reshape(size[0], size[1], 2)
        return rotated_point


def preprocess_image(image, device, dtype=torch.float32):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device, dtype)
    return image


def run_dragdiffusion_backbone(
    source_image,
    source_image_mask,
    merge_image_mask,
    prompt,
    points,
    inversion_strength=0.7,
    lam=0.1,
    latent_lr=0.01,
    n_pix_step=80,
    model_path="runwayml/stable-diffusion-v1-5",
    vae_path="default",
    lora_path="",
    start_step=0,
    start_layer=10,
    unet_feature_idx=3,
    seed=42,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                             beta_schedule="scaled_linear", clip_sample=False,
                             set_alpha_to_one=False, steps_offset=1)
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16)
    model.modify_unet_forward()

    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    model.enable_model_cpu_offload()
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0
    args.unet_feature_idx = [unet_feature_idx]
    args.r_m = 1
    args.r_p = 3
    args.lam = lam
    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5 * full_h)
    args.sup_res_w = int(0.5 * full_w)

    # Get all coordinates (y, x) where source_image_mask is non-zero
    original_mask_coords_yx = np.argwhere(source_image_mask > 0) # These are [row, col] -> [y, x]

    if len(original_mask_coords_yx) == 0:
        print("Warning: source_image_mask is empty. No handle points will be selected.")
        non_scaled_handle_points = []
    elif len(original_mask_coords_yx) <= 30:
        print(f"Warning: Fewer than 30 points in source_image_mask. Using all {len(original_mask_coords_yx)} points.")
        non_scaled_handle_points = original_mask_coords_yx.tolist()
    else:
        # Randomly sample 30 points
        sample_indices = np.random.choice(len(original_mask_coords_yx), size=30, replace=False)
        non_scaled_handle_points = original_mask_coords_yx[sample_indices].tolist() # list of [y, x]

    mask = torch.from_numpy(merge_image_mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").to(device)
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")

    handle_points = []
    # Scale non_scaled_handle_points to supervision resolution
    for p_y_orig, p_x_orig in non_scaled_handle_points:
        h_y_sup_res = p_y_orig / full_h * args.sup_res_h
        h_x_sup_res = p_x_orig / full_w * args.sup_res_w
        # Consistent with user's apparent preference for (X,Y) in current_target_point
        current_handle_point = torch.tensor([h_y_sup_res, h_x_sup_res], dtype=torch.float32)
        # drag_utils expects integer coordinates for handle points for feature map indexing
        # however, the optimization step itself might use float for sub-pixel precision if drag_utils handles it
        # For now, aligning with the observed (X,Y) and keeping float, will cast to long if error persists
        handle_points.append(current_handle_point)
            
    target_points = []
    for handle_point_orig_yx in non_scaled_handle_points: # These are [y_orig, x_orig]
        # points is indexed by [y][x]
        t_y_sup_orig, t_x_sup_orig = points[handle_point_orig_yx[0]][handle_point_orig_yx[1]]
        t_y_sup_res = t_y_sup_orig / full_h * args.sup_res_h
        t_x_sup_res = t_x_sup_orig / full_w * args.sup_res_w
        # current_target_point = torch.round(torch.tensor([t_x_sup_res, t_y_sup_res])) # Original (X,Y)
        current_target_point = torch.tensor([t_y_sup_res, t_x_sup_res], dtype=torch.float32) # Keep as float, (X,Y)
        target_points.append(current_target_point)

    source_image = preprocess_image(source_image, device, dtype=torch.float16)
    
    if lora_path == "":
        model.unet.set_default_attn_processor()
    else:
        model.unet.load_attn_procs(lora_path)

    text_embeddings = model.get_text_embeddings(prompt)
    invert_code = model.invert(
        source_image,
        prompt,
        encoder_hidden_states=text_embeddings,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step
    )
    torch.cuda.empty_cache()

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    init_code = init_code.float()
    text_embeddings = text_embeddings.float()
    model.unet = model.unet.float()

    updated_init_code = drag_diffusion_update(
        model,
        init_code,
        text_embeddings,
        t,
        handle_points,
        target_points,
        mask,
        args)

    updated_init_code = updated_init_code.half()
    text_embeddings = text_embeddings.half()
    model.unet = model.unet.half()
    torch.cuda.empty_cache()

    editor = MutualSelfAttentionControl(start_step=start_step,
                                        start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    gen_image = model(
        prompt=args.prompt,
        encoder_hidden_states=torch.cat([text_embeddings] * 2, dim=0),
        batch_size=2,
        latents=torch.cat([init_code_orig, updated_init_code], dim=0),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step
    )[1].unsqueeze(dim=0)

    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')
    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image


class SimpleProgress:
    def tqdm(self, *args, **kwargs):
        return tqdm(*args, **kwargs)

def visualize_drag_points(image, points, n=50, radius=8, thickness=1, seed=42):
    """
    Visualize DragDiffusion source and target points on an image.
    First, sample n random source points, then for each source point, get the corresponding target point.
    Args:
        image: np.ndarray, shape (H, W, 3), uint8
        points: nested list, e.g. points[i][0] is the target, and each point is [y, x]
        n: int, number of random source points to sample
        radius: int, circle radius
        thickness: int, circle/arrow thickness
        seed: int, random seed for reproducibility
    Returns:
        img_vis: np.ndarray, image with points/arrows drawn
    """
    img_vis = image.copy()
    H, W = img_vis.shape[:2]
    random.seed(seed)
    np.random.seed(seed)

    # Sample n random source points
    source_points = []
    for _ in range(n):
        src_y = random.randint(0, H-1)
        src_x = random.randint(0, W-1)
        source_points.append((int(round(src_x)), int(round(src_y))))  # (x, y)

    for src_point in source_points:
        src_x, src_y = src_point
        tgt_y, tgt_x = points[src_y][src_x]
        cv2.circle(img_vis, (src_x, src_y), radius, (0, 0, 255), thickness)
        cv2.circle(img_vis, (int(round(tgt_x)), int(round(tgt_y))), radius, (255, 0, 0), thickness)
        cv2.arrowedLine(img_vis, (src_x, src_y), (int(round(tgt_x)), int(round(tgt_y))), (255, 255, 255), thickness, tipLength=0.1)
    return img_vis



def train_lora_backbone(
    image,
    prompt,
    model_path,
    vae_path,
    lora_path,
    lora_step=80,
    lora_lr=0.0005,
    lora_batch_size=4,
    lora_rank=16,
    save_interval=-1,
):
    """
    Fine-tune LoRA weights for DragDiffusion backbone (no UI).
    Args:
        image: np.ndarray, input image (H, W, 3)
        prompt: str, text prompt
        model_path: str, base model path
        vae_path: str, VAE path or 'default'
        lora_path: str, directory to save LoRA weights
        lora_step: int, training steps
        lora_lr: float, learning rate
        lora_batch_size: int, batch size
        lora_rank: int, LoRA rank
        save_interval: int, save interval (default -1, only save at end)
    """
    progress = SimpleProgress()
    train_lora(
        image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress,
        save_interval=save_interval
    )


LENGTH = 512

if __name__ == '__main__':
    
    annotations_path = "/work/nvme/bcgq/yimingg8/geobench/annotations.json"
    with open(annotations_path, "r") as f:
        data = json.load(f)
    

    import time
    time_total = 0
    time_avg = 0
    count = 0
    for image_idx in data.keys():
        if int(image_idx) < 110:
            edit_indices = data[image_idx]["instances"]
            for edit_idx in edit_indices.keys():
                for sub_edit_idx in edit_indices[edit_idx].keys():
                    orig_path = edit_indices[edit_idx][sub_edit_idx]["ori_img_path"]
                    mask_path = edit_indices[edit_idx][sub_edit_idx]["ori_mask_path"]
                    target_mask_path = edit_indices[edit_idx][sub_edit_idx]["tgt_mask_path"]
                    edit_param = edit_indices[edit_idx][sub_edit_idx]["edit_param"]
                    # edit_param[5] = -edit_param[5]

                    input_image = np.array(Image.open(orig_path).resize((LENGTH, LENGTH)))
                    

                    mask_image = Image.open(mask_path).convert("L")
                    target_mask_image = Image.open(target_mask_path).convert("L")
                    mask_image = mask_image.resize((LENGTH, LENGTH), resample=Image.NEAREST)
                    target_mask_image = target_mask_image.resize((LENGTH, LENGTH), resample=Image.NEAREST)
                    mask_image = np.array(mask_image)
                    target_mask_image = np.array(target_mask_image)
                    # Create union mask by combining both masks using logical OR
                    union_mask = np.logical_or(mask_image > 0, target_mask_image > 0).astype(mask_image.dtype)

                    points =  get_transform_coordinates(edit_param, (LENGTH, LENGTH), mask_image)

                    ##For visualization
                    # vis_img = visualize_drag_points(input_image, points)
                    # cv2.imwrite('vis_points.png', vis_img)

                    lora_save_path = "./lora_tmp_test"

                    start = time.time()

                    prompt = " "

                    print("Starting LoRA fine-tuning...")
                    train_lora_backbone(
                        input_image,
                        prompt,
                        model_path="runwayml/stable-diffusion-v1-5",
                        vae_path="default",
                        lora_path=lora_save_path,
                        lora_step=10,  # use small number for quick test
                        lora_lr=0.0005,
                        lora_batch_size=2,
                        lora_rank=4,
                        save_interval=-1
                    )
                    print("LoRA fine-tuning complete.")

                    print("Running DragDiffusion backbone evaluation with LoRA weights...")
                    out = run_dragdiffusion_backbone(
                        input_image,
                        mask_image,
                        union_mask,
                        prompt,
                        points,
                        inversion_strength=0.7,
                        lam=0.1,
                        latent_lr=0.01,
                        n_pix_step=80,
                        model_path="runwayml/stable-diffusion-v1-5",
                        vae_path="default",
                        lora_path=lora_save_path,
                        start_step=0,
                        start_layer=10,
                    )
                    print("Output image shape after LoRA fine-tuning:", out.shape)
                    end = time.time()
                    time_total += end - start
                    count += 1
                    time_avg = time_total / count
                    print(f"Time taken for {count} images: {time_avg} seconds")
                    
                    # out = Image.fromarray(out)
                    # save_dir = edit_indices[edit_idx][sub_edit_idx]["coarse_input_path"].replace("geobench", "dragdiffuser_output_2D").replace("/coarse_img","")
                    # os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                    # out.save(save_dir)
                    # print(f"Saved output to {save_dir}")






