from src.models.dragondiff import DragonPipeline
from src.utils.utils import resize_numpy_image,get_inw, split_ldm, process_move, process_drag_face, process_drag, process_appearance, process_paste
from src.utils.inversion import DDIMInversion
from src.utils.geometric_utils import Integrated3DTransformAndInpaint
from src.utils.geo_utils import IntegratedP3DTransRasterBlendingFull,param2theta,wrapAffine_tensor,PartialConvInterpolation,tensor_inpaint_fmm,calculate_cosine_similarity_between_batches
from diffusers.utils.torch_utils import  randn_tensor
import time
import sys
sys.path.append('/data/Hszhu/Reggio')
from ram import inference_tag2text
from typing import List, Union
import clip
from einops import rearrange
import spacy
import os
from copy import deepcopy
import cv2
import copy
import random
from pytorch_lightning import seed_everything
from torchvision.transforms import PILToTensor
import numpy as np
from basicsr.utils import img2tensor
from src.utils.alignment import align_face, get_landmark

import dlib
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
from src.utils.attention import override_forward
from rembg import remove
from typing import Optional
from pytorch_lightning.utilities import rank_zero_warn
NUM_DDIM_STEPS = 50
SIZES = {
    0:4,
    1:2,
    2:1,
    3:1,
}
import torchvision.transforms as TS


def preprocess_tag_pil_img(image_pillow,device):
    normalize = TS.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = TS.Compose(
        [
            TS.Resize((384, 384)),
            TS.ToTensor(),
            normalize
        ]
    )
    image_pillow = image_pillow.resize((384, 384))
    image_pillow = transform(image_pillow).unsqueeze(0).to(device)
    return image_pillow
def my_seed_everything(seed: Optional[int] = None) -> int:
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    In addition, sets the env variable `PL_GLOBAL_SEED` which will be passed to
    spawned subprocesses (e.g. ddp_spawn backend).

    Args:
        seed: the integer value seed for global random state in Lightning.
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
        rank_zero_warn(f"No correct seed found, seed set to {seed}")

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
class DragonModels():
    def __init__(self, pretrained_model_path,device):
        self.device = device #modified by clawer
        self.ip_scale = 0.1
        self.precision = torch.float16
        self.editor = DragonPipeline(sd_id=pretrained_model_path, NUM_DDIM_STEPS=NUM_DDIM_STEPS, precision=self.precision, ip_scale=self.ip_scale,device=self.device)
        self.up_ft_index = [1,2] # fixed in gradio demo
        self.up_scale = 2        # fixed in gradio demo

        # face editing
        SHAPE_PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
        self.face_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)


    def move_and_inpaint(self, image, mask, dx, dy):
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask > 128

        mask = mask.astype(bool)

        # 获取图像尺寸
        height, width = image.shape[:2]

        # 创建移动后的图像和掩码的存储
        shifted_image = np.zeros_like(image)
        shifted_mask = np.zeros_like(mask)

        # 计算移动后的新坐标，确保不会超出边界
        for y in range(height):
            for x in range(width):
                # new_x = (x + dx) % width
                # new_y = (y + dy) % height
                new_x = (x + dx)
                new_y = (y + dy)

                # 使用取模操作可以选择性地允许或禁止环绕效果
                if 0 <= new_x < width and 0 <= new_y < height:
                    shifted_image[new_y, new_x] = image[y, x]
                    shifted_mask[new_y, new_x] = mask[y, x]

        # 创建一个新的图像，物体移动到新位置
        new_image = np.where(shifted_mask[:, :, None], shifted_image, image)

        # 将原始物体位置像素设置为0
        image_with_hole = np.where(mask[:, :, None], 0, image).astype(np.uint8)

        # 创建修复掩码，只覆盖原始物体位置
        repair_mask = (mask.astype(np.uint8) * 255)

        # 对原始位置进行修复
        inpainted_image = cv2.inpaint(image_with_hole, repair_mask, 3, cv2.INPAINT_TELEA)

        # 计算需要从修复后的图像中填充的区域：原始位置且非新位置
        fill_mask = mask & ~shifted_mask

        # 将修复后的背景与新图像合并
        final_image = np.where(fill_mask[:, :, None], inpainted_image, new_image)

        return final_image

    def run_my_baseline(self, original_image, mask, mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint,
                 seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        # energy_scale = energy_scale * 1e3
        # get mask shifting values
        img = original_image
        x = []
        y = []
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx % 2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        dx = x_cur[0] - x[0]
        dy = y_cur[0] - y[0]
        #TODO: copy-paste-inpaint to get initial img
        img_preprocess = self.move_and_inpaint(img, mask, dx, dy)

        return img_preprocess
    def run_move(self, original_image, mask, mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution*max_resolution)
        h, w = img.shape[1], img.shape[0]
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        if mask_ref is not None and np.sum(mask_ref)!=0:
            mask_ref = np.repeat(mask_ref[:,:,None], 3, 2)
        else:
            mask_ref = None

        emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        latent = self.editor.image2latent(img_tensor)
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt)
        latent_in = ddim_latents[-1].squeeze(2)

        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        x=[]
        y=[]
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx%2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        dx = x_cur[0]-x[0]
        dy = y_cur[0]-y[0]

        edit_kwargs = process_move(
            path_mask=mask,
            h=h,
            w=w,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            resize_scale=resize_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_content=w_content,
            w_contrast=w_contrast,
            w_inpaint=w_inpaint,
            precision=self.precision,
            path_mask_ref=mask_ref
        )
        # pre-process zT
        mask_tmp = (F.interpolate(img2tensor(mask)[0].unsqueeze(0).unsqueeze(0), (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))>0).float().to('cuda', dtype=latent_in.dtype)
        latent_tmp = F.interpolate(latent_in, (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))
        mask_tmp = torch.roll(mask_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
        latent_tmp = torch.roll(latent_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
        pad_size_x = abs(mask_tmp.shape[-1]-latent_in.shape[-1])//2
        pad_size_y = abs(mask_tmp.shape[-2]-latent_in.shape[-2])//2
        if resize_scale>1:
            sum_before = torch.sum(mask_tmp)
            mask_tmp = mask_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
            latent_tmp = latent_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
            sum_after = torch.sum(mask_tmp)
            if sum_after != sum_before:
                raise ValueError('Resize out of bounds.')
                exit(0)
        elif resize_scale<1:
            temp = torch.zeros(1,1,latent_in.shape[-2], latent_in.shape[-1]).to(latent_in.device, dtype=latent_in.dtype)
            temp[:,:,pad_size_y:pad_size_y+mask_tmp.shape[-2],pad_size_x:pad_size_x+mask_tmp.shape[-1]]=mask_tmp
            mask_tmp =(temp>0.5).float()
            temp = torch.zeros_like(latent_in)
            temp[:,:,pad_size_y:pad_size_y+latent_tmp.shape[-2],pad_size_x:pad_size_x+latent_tmp.shape[-1]]=latent_tmp
            latent_tmp = temp
        latent_in = (latent_in*(1-mask_tmp)+latent_tmp*mask_tmp).to(dtype=latent_in.dtype)

        latent_rec = self.editor.pipe.edit(
            mode = 'move',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in,
            prompt=prompt,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref = ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]

    def run_appearance(self, img_base, mask_base, img_replace, mask_replace, prompt, prompt_replace, w_edit, w_content, seed, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img_base, input_scale = resize_numpy_image(img_base, max_resolution*max_resolution)
        h, w = img_base.shape[1], img_base.shape[0]
        img_base = Image.fromarray(img_base)
        img_prompt_base = img_base.resize((256, 256))
        img_base_tensor = (PILToTensor()(img_base) / 255.0 - 0.5) * 2
        img_base_tensor = img_base_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        img_replace = Image.fromarray(img_replace)
        img_prompt_replace = img_replace.resize((256, 256))
        img_replace = img_replace.resize((img_base_tensor.shape[-1], img_base_tensor.shape[-2]))
        img_replace_tensor = (PILToTensor()(img_replace) / 255.0 - 0.5) * 2
        img_replace_tensor = img_replace_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        mask_replace = np.repeat(mask_replace[:,:,None], 3, 2) if len(mask_replace.shape)==2 else mask_replace
        mask_base = np.repeat(mask_base[:,:,None], 3, 2) if len(mask_base.shape)==2 else mask_base

        emb_im_base, emb_im_uncond_base = self.editor.get_image_embeds(img_prompt_base)
        emb_im_replace, emb_im_uncond_replace = self.editor.get_image_embeds(img_prompt_replace)
        emb_im = torch.cat([emb_im_base, emb_im_replace], dim=1)
        emb_im_uncond = torch.cat([emb_im_uncond_base, emb_im_uncond_replace], dim=1)

        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        latent_base = self.editor.image2latent(img_base_tensor)
        latent_replace = self.editor.image2latent(img_replace_tensor)
        ddim_latents = self.editor.ddim_inv(latent=torch.cat([latent_base, latent_replace]), prompt=[prompt, prompt_replace])
        latent_in = ddim_latents[-1][:1].squeeze(2)

        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_appearance(
            path_mask = mask_base,
            path_mask_replace = mask_replace,
            h = h,
            w = w,
            scale = scale,
            input_scale = input_scale,
            up_scale = self.up_scale,
            up_ft_index = self.up_ft_index,
            w_edit = w_edit,
            w_content = w_content,
            precision = self.precision
        )
        latent_rec = self.editor.pipe.edit(
            mode = 'appearance',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in,
            prompt=prompt,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref = ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]

    def run_drag_face(self, original_image, reference_image, w_edit, w_inpaint, seed, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=0.05):
        seed_everything(seed)
        prompt = 'a photo of a human face'
        energy_scale = energy_scale*1e3
        original_image = np.array(align_face(original_image, self.face_predictor, 1024))
        reference_image = np.array(align_face(reference_image, self.face_predictor, 1024))
        ldm = get_landmark(original_image, self.face_predictor)
        ldm_ref = get_landmark(reference_image, self.face_predictor)
        x_cur, y_cur = split_ldm(ldm_ref)
        x, y = split_ldm(ldm)
        original_image, input_scale = resize_numpy_image(original_image, max_resolution*max_resolution)
        reference_image, _ = resize_numpy_image(reference_image, max_resolution*max_resolution)
        img = original_image
        h, w = img.shape[1], img.shape[0]
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)

        latent = self.editor.image2latent(img_tensor)
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt)
        latent_in = ddim_latents[-1].squeeze(2)

        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_drag_face(
            h=h,
            w=w,
            x=x,
            y=y,
            x_cur=x_cur,
            y_cur=y_cur,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_inpaint=w_inpaint,
            precision=self.precision,
        )
        latent_rec = self.editor.pipe.edit(
            mode = 'landmark',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in,
            prompt=prompt,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref = ddim_latents,
            edit_kwargs=edit_kwargs,
            SDE_strength_un=SDE_strength,
            SDE_strength = SDE_strength,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()
        # draw editing direction
        for x_cur_i, y_cur_i in zip(x_cur, y_cur):
            reference_image = cv2.circle(reference_image, (x_cur_i, y_cur_i), 8,(255,0,0),-1)
        for x_i, y_i, x_cur_i, y_cur_i in zip(x, y, x_cur, y_cur):
            cv2.arrowedLine(original_image, (x_i, y_i), (x_cur_i, y_cur_i), (255, 255, 255), 4, tipLength=0.2)
            original_image = cv2.circle(original_image, (x_i, y_i), 6,(0,0,255),-1)
            original_image = cv2.circle(original_image, (x_cur_i, y_cur_i), 6,(255,0,0),-1)

        return [img_rec, reference_image, original_image]

    def run_drag(self, original_image, mask, prompt, w_edit, w_content, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution*max_resolution)
        h, w = img.shape[1], img.shape[0]
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
        mask = np.repeat(mask[:,:,None], 3, 2) if len(mask.shape)==2 else mask

        emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)

        latent = self.editor.image2latent(img_tensor)
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt)
        latent_in = ddim_latents[-1].squeeze(2)

        x=[]
        y=[]
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx%2 == 0:
                y.append(point[1]*input_scale)
                x.append(point[0]*input_scale)
            else:
                y_cur.append(point[1]*input_scale)
                x_cur.append(point[0]*input_scale)

        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_drag(
            latent_in = latent_in,
            path_mask=mask,
            h=h,
            w=w,
            x=x,
            y=y,
            x_cur=x_cur,
            y_cur=y_cur,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit=w_edit,
            w_content=w_content,
            w_inpaint=w_inpaint,
            precision=self.precision,
        )
        latent_in = edit_kwargs.pop('latent_in')
        latent_rec = self.editor.pipe.edit(
            mode = 'drag',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in,
            prompt=prompt,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref = ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]

    def run_paste(self, img_base, mask_base, img_replace, prompt, prompt_replace, w_edit, w_content, seed, guidance_scale, energy_scale, dx, dy, resize_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img_base, input_scale = resize_numpy_image(img_base, max_resolution*max_resolution)
        h, w = img_base.shape[1], img_base.shape[0]
        img_base = Image.fromarray(img_base)
        img_prompt_base = img_base.resize((256, 256))
        img_base_tensor = (PILToTensor()(img_base) / 255.0 - 0.5) * 2
        img_base_tensor = img_base_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        img_replace = Image.fromarray(img_replace)
        img_prompt_replace = img_replace.resize((256, 256))
        img_replace = img_replace.resize((img_base_tensor.shape[-1], img_base_tensor.shape[-2]))
        img_replace_tensor = (PILToTensor()(img_replace) / 255.0 - 0.5) * 2
        img_replace_tensor = img_replace_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        mask_base = np.repeat(mask_base[:,:,None], 3, 2) if len(mask_base.shape)==2 else mask_base

        emb_im_base, emb_im_uncond_base = self.editor.get_image_embeds(img_prompt_base)
        emb_im_replace, emb_im_uncond_replace = self.editor.get_image_embeds(img_prompt_replace)
        emb_im = torch.cat([emb_im_base, emb_im_replace], dim=1)
        emb_im_uncond = torch.cat([emb_im_uncond_base, emb_im_uncond_replace], dim=1)

        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        latent_base = self.editor.image2latent(img_base_tensor)
        if resize_scale != 1:
            hr, wr = img_replace_tensor.shape[-2], img_replace_tensor.shape[-1]
            img_replace_tensor = F.interpolate(img_replace_tensor, (int(hr*resize_scale), int(wr*resize_scale)))
            pad_size_x = abs(img_replace_tensor.shape[-1]-wr)//2
            pad_size_y = abs(img_replace_tensor.shape[-2]-hr)//2
            if resize_scale>1:
                img_replace_tensor = img_replace_tensor[:,:,pad_size_y:pad_size_y+hr,pad_size_x:pad_size_x+wr]
            else:
                temp = torch.zeros(1,3,hr, wr).to(self.device, dtype=self.precision)
                temp[:,:,pad_size_y:pad_size_y+img_replace_tensor.shape[-2],pad_size_x:pad_size_x+img_replace_tensor.shape[-1]]=img_replace_tensor
                img_replace_tensor = temp

        latent_replace = self.editor.image2latent(img_replace_tensor)
        ddim_latents = self.editor.ddim_inv(latent=torch.cat([latent_base, latent_replace]), prompt=[prompt, prompt_replace])
        latent_in = ddim_latents[-1][:1].squeeze(2)

        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_paste(
            path_mask=mask_base,
            h=h,
            w=w,
            dx=dx,
            dy=dy,
            scale=scale,
            input_scale=input_scale,
            up_scale=self.up_scale,
            up_ft_index=self.up_ft_index,
            w_edit = w_edit,
            w_content = w_content,
            precision = self.precision,
            resize_scale=resize_scale
        )
        mask_tmp = (F.interpolate(edit_kwargs['mask_base_cur'].float(), (latent_in.shape[-2], latent_in.shape[-1]))>0).float()
        latent_tmp = torch.roll(ddim_latents[-1][1:].squeeze(2), (int(dy/(w/latent_in.shape[-2])), int(dx/(w/latent_in.shape[-2]))), (-2,-1))
        latent_in = (latent_in*(1-mask_tmp)+latent_tmp*mask_tmp).to(dtype=latent_in.dtype)

        latent_rec = self.editor.pipe.edit(
            mode = 'paste',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in,
            prompt=prompt,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            latent_noise_ref = ddim_latents,
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]


class ClawerModels(StableDiffusionPipeline):
    """
    implemented from FPE code
    utilize DDIM inversion for our baseline
    """
    # must call this function when initialize
    # def modify_unet_forward(self):
    #     self.unet.forward = override_forward(self.unet)
    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
    ):
        """
        predict the sample of the next step in the denoise process. DDIM denoising regardless of varience
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        self.device
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    # Sample function (regular DDIM)

    def erode_dilate_mask(self, mask, iterations=1, erode_amount=1, dilate_amount=1):
        # 使用腐蚀操作来模糊边缘
        eroded_mask = cv2.erode(mask, np.ones((erode_amount, erode_amount), np.uint8), iterations=iterations)
        # 使用膨胀操作来扩张物体区域
        dilated_mask = cv2.dilate(eroded_mask, np.ones((dilate_amount, dilate_amount), np.uint8), iterations=iterations)
        return dilated_mask

    def dilate_mask(self,mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def erode_mask(self,mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.erode(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def get_attention_scores(self, query, key, attention_mask=None,use_softmax=True):
        dtype = query.dtype

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if not use_softmax:
            return attention_scores,dtype

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_continuous_transformed_mask(self, mask, x_shift, y_shift, resize_scale=1.0, rotation_angle=0,
                                                 flip_horizontal=False, flip_vertical=False,split_stage_num=10):

        transformed_mask_list=[]
        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask > 128

        # Continuous editing is not designed for flip operation,
        # since it is simple, inpaining can solve this promblem
        # regardless of this for now
        ## 检查是否需要水平翻转
        # if flip_horizontal:
        #     transformed_mask = cv2.flip(transformed_mask, 1)
        #
        # # 检查是否需要垂直翻转
        # if flip_vertical:
        #     transformed_mask = cv2.flip(transformed_mask, 0)

        # 获取图像尺寸
        height, width = mask.shape[:2]
        y_indices, x_indices = np.where(mask.astype(bool))
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = np.min(y_indices), np.max(y_indices)
            left, right = np.min(x_indices), np.max(x_indices)
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2

        rotation_angle_step = rotation_angle / split_stage_num
        resize_scale_step = resize_scale**(1/split_stage_num)
        dx = x_shift / split_stage_num
        dy = y_shift / split_stage_num

        transformed_mask_temp = mask
        transformed_mask_list.append(transformed_mask_temp)
        for step in range(split_stage_num):
            #single step matrix for current center
            transformation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle_step, resize_scale_step)
            transformation_matrix[0, 2] += dx
            transformation_matrix[1, 2] += dy
            # print(f'step:{step} matrix:{transformation_matrix}')
            transformed_mask_temp = cv2.warpAffine(transformed_mask_temp, transformation_matrix, (width, height),
                                              flags=cv2.INTER_NEAREST)
            transformed_mask_list.append(transformed_mask_temp)
            mask_center_x += dx
            mask_center_y += dy
        # output_dir = 'temp_dir_vis'
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # for idx, mask in enumerate(transformed_mask_list):
        #     plt.figure()
        #     plt.imshow(mask, cmap='gray')
        #     plt.title(f'Step {idx}')
        #     plt.axis('off')
        #     plt.savefig(os.path.join(output_dir, f'mask_step_{idx}.png'))
        #     plt.close()
        return transformed_mask_list
    def move_and_inpaint_with_expansion_mask_new(self, image, mask, dx, dy, inpainter=None, mode=None,
                                             dilate_kernel_size=15, inp_prompt=None,
                                             resize_scale=1.0, rotation_angle=0,target_mask=None,flip_horizontal=False,flip_vertical=False):

        if isinstance(image, Image.Image):
            image = np.array(image)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask > 128


        if target_mask.ndim == 3 and target_mask.shape[2] == 3:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
            target_mask =target_mask > 128

        mask = mask.astype(bool)
        target_mask =target_mask.astype(bool)


        # 获取图像尺寸
        height, width = image.shape[:2]


        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = np.min(y_indices), np.max(y_indices)
            left, right = np.min(x_indices), np.max(x_indices)
            # mask_roi = mask[top:bottom + 1, left:right + 1]
            # image_roi = image[top:bottom + 1, left:right + 1]
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2


        rotation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle, resize_scale)
        rotation_matrix[0, 2] += dx
        rotation_matrix[1, 2] += dy

        transformed_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        transformed_mask = cv2.warpAffine(mask.astype(np.uint8), rotation_matrix, (width, height),
                                          flags=cv2.INTER_NEAREST).astype(bool)
        transformed_target = cv2.warpAffine(target_mask.astype(np.uint8), rotation_matrix, (width, height),).astype(bool)

        # 检查是否需要水平翻转
        if flip_horizontal:
            transformed_image = cv2.flip(transformed_image, 1)
            transformed_mask = cv2.flip(transformed_mask, 1)
            transformed_target = cv2.flip(transformed_target, 1)

        # 检查是否需要垂直翻转
        if flip_vertical:
            transformed_image = cv2.flip(transformed_image, 0)
            transformed_mask = cv2.flip(transformed_mask, 0)
            transformed_target = cv2.flip(transformed_target, 0)

        # 创建一个新的图像，物体移动到新位置
        new_image = np.where(transformed_mask[:, :, None], transformed_image, image) #move with expansion pixels but inpaint
        # new_image = np.where(transformed_target[:, :, None], transformed_image, image) #only move the target w/o expansion

        # how to define inpaint mask and image
        # (1) original image + original mask
        # (2) new_moved_image + original mask - (original mask & shifted mask)
        # seems like (2) is more suitable for latest inpainting models
        # inpaint_mode = 1
        # if inpaint_mode:
        repair_region = ((mask | transformed_mask) & ~transformed_target)
        repair_mask = repair_region.astype(np.uint8) * 255  # (2)
        # repair_mask = self.dilate_mask(repair_mask, dilate_factor=dilate_kernel_size)
        image_with_hole = np.where(repair_mask[:, :, None], 0, new_image).astype(np.uint8)  # for visualization use
        to_inpaint_img = new_image
        # else:
        #     repair_mask = (mask.astype(np.uint8) * 255)  #(1)
        #     # repair_mask = self.dilate_mask(repair_mask, dilate_factor=dilate_kernel_size) already expanision and dilation
        #     image_with_hole = np.where(repair_mask[:, :, None], 0, image).astype(np.uint8)  # for visualization use
        #     to_inpaint_img = image
        if mode == 1:
            # lama inpainting
            to_inpaint_img = Image.fromarray(to_inpaint_img)
            repair_mask = Image.fromarray(repair_mask)
            inpainted_image = inpainter(to_inpaint_img, repair_mask)
        elif mode == 2:
            # sd inpainting
            to_inpaint_img = Image.fromarray(to_inpaint_img)
            repair_mask = Image.fromarray(repair_mask)
            # print(f'img:{to_inpaint_img.size}')
            # print(f'msk:{repair_mask.size}')
            inpainted_image = self.sd_inpainter(prompt=inp_prompt, image=to_inpaint_img, mask_image=repair_mask).images[
                0]
        elif mode == 3:
            # lama inpainting first
            print('lama inpaint')
            to_inpaint_img = Image.fromarray(to_inpaint_img)
            repair_mask = Image.fromarray(repair_mask)
            inpainted_image = inpainter(to_inpaint_img, repair_mask)
            # sd inpainting second
            print('lama inpaint')
            inpainted_image = self.sd_inpainter(prompt=inp_prompt, image=inpainted_image, mask_image=repair_mask).images[
                0]

        if inpainted_image.size != to_inpaint_img.size:
            print(f'inpainted image {inpainted_image.size} -> original size {to_inpaint_img.size}')
            inpainted_image = inpainted_image.resize(to_inpaint_img.size)
            retain_mask = ~repair_region
            inpainted_image = np.where(retain_mask[:, :, None], new_image, inpainted_image)

        # if inpaint_mode==0:
        #     final_image = np.where(transformed_mask[:, :, None], new_image, inpainted_image)
        # else:
        final_image = inpainted_image
        #mask retain
        retain_mask =  dict()
        retain_mask['obj_region'] = transformed_target
        retain_mask['ori_expansion'] = mask


        return final_image, image_with_hole, transformed_mask , retain_mask


    def move_and_inpaint_with_expansion_mask_3D(self, image, mask,depth_map, transforms, FX, FY,object_only=True,inpainter=None, mode=None,
                                             dilate_kernel_size=15, inp_prompt=None,target_mask=None,splatting_radius = 0.015,
                                            splatting_tau = 0.0,splatting_points_per_pixel = 30):

        if isinstance(image, Image.Image):
            image = np.array(image)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask = mask > 128

        if target_mask.ndim == 3 and target_mask.shape[2] == 3:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
            # source_mask =source_mask > 128

        mask = mask.astype(bool)
        target_mask = target_mask.astype(bool)
        """ OLD VERSION
                transformed_image_target_only, sparse_transformed_target_mask, transformed_depth = Integrated3DTransformAndInpaint(image,depth_map,
                                                                                                               transforms,
                                                                                                               FX, FY,
                                                                                                               source_mask,object_only,
                                                                                                               inpaint=False)
        _, sparse_transformed_expand_mask, _   = Integrated3DTransformAndInpaint(image,depth_map,transforms, FX, FY, mask,object_only,inpaint=False)

        # dilation_target = self.dilate_mask(sparse_transformed_target_mask, dilate_factor=15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closed_target = cv2.morphologyEx(sparse_transformed_target_mask, cv2.MORPH_CLOSE,kernel)
        dilated_closed_target = self.dilate_mask(cv2.morphologyEx(sparse_transformed_expand_mask, cv2.MORPH_CLOSE,kernel),5)
        closed_target = (closed_target >128).astype(bool)
        sparse_target = (sparse_transformed_target_mask>128).astype(bool)
        dilated_closed_target = (dilated_closed_target>128).astype(bool)
        # mask: original expansion
        # closed_target:   transformed closed target
        # sparse_trans.. : transformed sparse target
        mask = (mask>128).astype(bool)
        full_repair = ((dilated_closed_target | mask) & ~sparse_target).astype(np.uint8)*255
        # inner_repair = (closed_target &~sparse_target).astype(np.uint8)*255
        # outer_repair = ((mask | dilated_closed_target)&~closed_target).astype(np.uint8)*255
        blended_background = ~(mask | dilated_closed_target)
        image_inner_sparse = np.where(blended_background[:,:,None],image,transformed_image_target_only)
        # image_inner_full = cv2.inpaint(image_inner_sparse, inner_repair, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        image_inner_full = inpainter(Image.fromarray(image_inner_sparse), Image.fromarray(full_repair))#lama inpainting inner
        # image_with_hole =  image_inner_sparse
        # image_with_hole = np.where(blended_background[:,:,None],image,image_inner_full) #vis next inpaint area
        to_inpaint_img = image_inner_full
        # repair_mask = full_repair
        repair_mask =  ((dilated_closed_target | mask) & ~closed_target).astype(np.uint8)*255
        image_with_hole = np.where( ((dilated_closed_target | mask) & ~closed_target)[:, :, None], 0, image_inner_full)
        # else:
        #     repair_mask = (mask.astype(np.uint8) * 255)  #(1)
        #     # repair_mask = self.dilate_mask(repair_mask, dilate_factor=dilate_kernel_size) already expanision and dilation
        #     image_with_hole = np.where(repair_mask[:, :, None], 0, image).astype(np.uint8)  # for visualization use
        #     to_inpaint_img = image
        if mode == 1:
            # lama inpainting
            # to_inpaint_img = Image.fromarray(to_inpaint_img)
            inpaint_mask = Image.fromarray(repair_mask)
            inpainted_image = inpainter(to_inpaint_img, inpaint_mask)
        elif mode == 2:
            # sd inpainting
            # to_inpaint_img = Image.fromarray(to_inpaint_img)
            inpaint_mask = Image.fromarray(repair_mask)
            # print(f'img:{to_inpaint_img.size}')
            # print(f'msk:{repair_mask.size}')
            inpainted_image = self.sd_inpainter(prompt=inp_prompt, image=to_inpaint_img, mask_image=inpaint_mask).images[
                0]

        if inpainted_image.size != to_inpaint_img.size:
            print(f'inpainted image {inpainted_image.size} -> original size {to_inpaint_img.size}')
            inpainted_image = inpainted_image.resize(to_inpaint_img.size)
            retain_mask = ~(repair_mask.astype(bool))
            final_image = np.where(retain_mask[:, :, None], image_inner_full, inpainted_image)

        # if inpaint_mode==0:
        #     final_image = np.where(transformed_mask[:, :, None], new_image, inpainted_image)
        # else:
        # final_image = inpainted_image
        #mask retain
        retain_mask =  dict()
        retain_mask['obj_region'] = closed_target
        retain_mask['ori_expansion'] = mask
        """
        transformed_image, transformed_mask = IntegratedP3DTransRasterBlendingFull(image, depth_map, transforms, FX, FY,
                                                                               target_mask, object_only,
                                                                               splatting_radius=splatting_radius,
                                                                               splatting_tau=splatting_tau,
                                                                               splatting_points_per_pixel=splatting_points_per_pixel,
                                                                               return_mask=True,
                                                                               device=self.device)


        #mask bool
        # MORPH_OPEN transformed target mask to suppress noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        transformed_mask =  cv2.morphologyEx(transformed_mask, cv2.MORPH_OPEN, kernel)
        transformed_mask = (transformed_mask > 128).astype(bool)
        repair_mask = (mask & ~transformed_mask)
        ori_image_back_ground = np.where(mask[:, :, None], 0, image).astype(np.uint8)
        new_image = np.where(transformed_mask[:, :, None], transformed_image, ori_image_back_ground)#with repair area to be black
        image_with_hole = new_image
        coarse_repaired = inpainter(Image.fromarray(new_image), Image.fromarray(repair_mask.astype(np.uint8)*255))#lama inpainting filling the black regions

        to_inpaint_img = coarse_repaired

        if mode == 1:
            semantic_repaired = to_inpaint_img
        elif mode == 2:
            inpaint_mask = Image.fromarray(repair_mask.astype(np.uint8)*255)
            semantic_repaired = self.sd_inpainter(prompt=inp_prompt, image=to_inpaint_img, mask_image=inpaint_mask).images[0]

        if semantic_repaired.size != to_inpaint_img.size:
            print(f'inpainted image {semantic_repaired.size} -> original size {to_inpaint_img.size}')
            semantic_repaired = semantic_repaired.resize(to_inpaint_img.size)
            #mask retain in region only repairing
            retain_mask = ~repair_mask
            final_image = np.where(retain_mask[:, :, None], coarse_repaired, semantic_repaired)

        #mask retain
        retain_mask =  dict()
        retain_mask['obj_region'] = transformed_mask
        retain_mask['ori_expansion'] = mask

        return final_image, image_with_hole, transformed_mask , retain_mask


    def move_and_inpaint_with_expansion_mask(self, image, mask, dx, dy,inpainter=None,mode=None,dilate_kernel_size=15,inp_prompt=None,
                                             resize_scale=None,rotation_angle=None,flip_horizontal=False, flip_vertical=False):

        if isinstance(image, Image.Image):
            image = np.array(image)
        if inp_prompt is None:
            inp_prompt = ""
        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask > 128

        mask = mask.astype(bool)

        # 获取图像尺寸
        height, width = image.shape[:2]

        # 创建移动后的图像和掩码的存储
        shifted_image = np.zeros_like(image)
        shifted_mask = np.zeros_like(mask)

        # 计算移动后的新坐标，确保不会超出边界
        for y in range(height):
            for x in range(width):
                new_x = x + dx
                new_y = y + dy

                # 确保新的坐标在图像边界内
                if 0 <= new_x < width and 0 <= new_y < height:
                    shifted_image[new_y, new_x] = image[y, x]
                    shifted_mask[new_y, new_x] = mask[y, x]

        transformed_mask  = shifted_mask
        transformed_image = shifted_image
        y_indices, x_indices = np.where(shifted_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = np.min(y_indices), np.max(y_indices)
            left, right = np.min(x_indices), np.max(x_indices)

            mask_roi = shifted_mask[top:bottom + 1, left:right + 1]
            image_roi = shifted_image[top:bottom + 1, left:right + 1]
            print(f'mask_roi.shape:{mask_roi.shape}')

            # 计算 mask 区域的中心
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2

        if resize_scale is not None:
            new_height = int(mask_roi.shape[0] * resize_scale)
            new_width = int(mask_roi.shape[1] * resize_scale)
            mask_roi = cv2.resize(mask_roi.astype(np.uint8), (new_width, new_height),
                                  interpolation=cv2.INTER_NEAREST)
            image_roi = cv2.resize(image_roi, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


            # 确定新的边界框
            new_height, new_width = mask_roi.shape
            print(f'new_h:{new_height},new_w:{new_width}')
            new_top = max(0, int(mask_center_y - new_height / 2))
            new_bottom = min(height, new_top + new_height)
            new_left = max(0, int(mask_center_x - new_width / 2))
            new_right = min(width, new_left + new_width)
            mask_roi = mask_roi[:new_bottom-new_top,:new_right-new_left]
            image_roi = image_roi[:new_bottom-new_top,:new_right-new_left,:]
            if flip_horizontal:
                mask_roi = np.flip(mask_roi, axis=1)
                image_roi = np.flip(image_roi, axis=1)

            if flip_vertical:
                mask_roi = np.flip(mask_roi, axis=0)
                image_roi = np.flip(image_roi, axis=0)
        #     # 将变换后的区域放回原始图像和掩码
            transformed_image = np.zeros_like(image)
            transformed_mask = np.zeros_like(mask)
            transformed_mask[new_top:new_bottom, new_left:new_right] = mask_roi.astype(bool)
            transformed_image[transformed_mask[:,:,None].repeat(3,axis=2)] = image_roi[mask_roi[:,:,None].repeat(3,axis=2).astype(bool)]
        if rotation_angle is not None:
            # 计算旋转后需要的padding尺寸
            print(f'before padding shape:{mask_roi.shape}')

            diag_length = int(np.ceil(np.sqrt((new_right - new_left + 1) ** 2 + (new_bottom - new_top + 1) ** 2)))
            pad_w = (diag_length - (new_right - new_left + 1)) // 2
            pad_h = (diag_length - (new_bottom - new_top + 1)) // 2
            print(f'l{diag_length},w{pad_w},h{pad_h}')
            # 对掩码和图像区域进行填充
            mask_roi_padded = np.pad(mask_roi, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant',
                                     constant_values=0)
            print(f'after padding shape:{mask_roi_padded.shape}')
            image_roi_padded = np.pad(image_roi,  ((pad_h, pad_h), (pad_w, pad_w),(0, 0)), mode='constant',
                                      constant_values=255)

            rotation_matrix = cv2.getRotationMatrix2D((diag_length // 2, diag_length // 2), -rotation_angle, 1.0)
            rotated_mask_roi = cv2.warpAffine(mask_roi_padded.astype(np.uint8), rotation_matrix,
                                              (diag_length, diag_length), flags=cv2.INTER_NEAREST)
            rotated_image_roi = cv2.warpAffine(image_roi_padded, rotation_matrix, (diag_length, diag_length),
                                               flags=cv2.INTER_LINEAR)

            # 计算旋转后新边界框的顶点
            new_top = max(0, new_top - pad_h)
            new_bottom = min(height, new_bottom + pad_h)
            new_left = max(0, new_left - pad_w)
            new_right = min(width, new_right + pad_w)
            print(f't:{new_top},b{new_bottom},l:{new_left},r:{new_right}')
            print(f'rotated_mask_roi:{rotated_mask_roi[:new_bottom-new_top,:new_right-new_left].shape}')
            print(f'transform_mask_match:{transformed_mask[new_top:new_bottom, new_left:new_right].shape}')
        rotated_image_roi = rotated_image_roi[:new_bottom-new_top,:new_right-new_left,:]
        rotated_mask_roi = rotated_mask_roi[:new_bottom-new_top,:new_right-new_left]
        # 更新变换后的图像和掩码
        transformed_mask[new_top:new_bottom, new_left:new_right] = rotated_mask_roi.astype(bool)
        transformed_image[transformed_mask[:, :, None].repeat(3, axis=2)] = rotated_image_roi[
            rotated_mask_roi[:, :, None].repeat(3, axis=2).astype(bool)]

        # 创建一个新的图像，物体移动到新位置
        new_image = np.where(transformed_mask[:, :, None], transformed_image, image)

        #how to define inpaint mask and image
        #(1) original image + original mask
        #(2) new_moved_image + original mask - (original mask & shifted mask)
        #seems like (2) is more suitable for latest inpainting models

        repair_mask = (mask & ~transformed_mask).astype(np.uint8) * 255 #(2)
        repair_mask = self.dilate_mask(repair_mask, dilate_factor=dilate_kernel_size)
        image_with_hole = np.where(repair_mask[:, :, None], 0, new_image).astype(np.uint8)  # for visualization use
        to_inpaint_img = new_image
        # elif mode == 1:
        #     repair_mask = (mask.astype(np.uint8) * 255)  #(1)
        #     repair_mask = self.dilate_mask(repair_mask, dilate_factor=dilate_kernel_size)
        #     image_with_hole = np.where(repair_mask[:, :, None], 0, image).astype(np.uint8)  # for visualization use
        #     to_inpaint_img = image
        if mode == 1:
            #lama inpainting
            to_inpaint_img = Image.fromarray(to_inpaint_img)
            repair_mask = Image.fromarray(repair_mask)
            inpainted_image = inpainter(to_inpaint_img, repair_mask)
        elif mode == 2:
            #sd inpainting
            to_inpaint_img = Image.fromarray(to_inpaint_img)
            repair_mask = Image.fromarray(repair_mask)
            # print(f'img:{to_inpaint_img.size}')
            # print(f'msk:{repair_mask.size}')
            inpainted_image = self.sd_inpainter(prompt=inp_prompt, image=to_inpaint_img, mask_image=repair_mask).images[0]


        if inpainted_image.size != to_inpaint_img.size:
            print(f'inpainted image {inpainted_image.size} -> original size {to_inpaint_img.size}')
            inpainted_image = inpainted_image.resize(to_inpaint_img.size)

        # if mode==1:
        #     final_image = np.where(shifted_mask[:, :, None], new_image, inpainted_image)
        # elif mode==2:
        final_image = inpainted_image

        return final_image,image_with_hole,transformed_mask
    def move_and_inpaint(self, image, mask, dx, dy,inpainter=None,mode=None,dilate_kernel_size=15,inp_mask=None,inp_prompt=None,move_with_diy_mask=False):
        if move_with_diy_mask:
            mask = inp_mask
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask > 128

        mask = mask.astype(bool)

        # 获取图像尺寸
        height, width = image.shape[:2]

        # 创建移动后的图像和掩码的存储
        shifted_image = np.zeros_like(image)
        shifted_mask = np.zeros_like(mask)

        # 计算移动后的新坐标，确保不会超出边界
        for y in range(height):
            for x in range(width):
                # new_x = (x + dx) % width
                # new_y = (y + dy) % height
                new_x = (x + dx)
                new_y = (y + dy)

                # 使用取模操作可以选择性地允许或禁止环绕效果
                if 0 <= new_x < width and 0 <= new_y < height:
                    shifted_image[new_y, new_x] = image[y, x]
                    shifted_mask[new_y, new_x] = mask[y, x]

        # 创建一个新的图像，物体移动到新位置
        new_image = np.where(shifted_mask[:, :, None], shifted_image, image)



        # inp_mask = []
        if len(inp_mask) == 0:
            # # 创建修复掩码，只覆盖原始物体位置
            repair_mask = (mask.astype(np.uint8) * 255)


            # dilate mask to avoid unmasked edge effect implemented from inpaint anything
            # just dilation the same as my method

            repair_mask = self.dilate_mask(repair_mask, dilate_factor=dilate_kernel_size)

            # my process version
            # iterations = 2  # 腐蚀次数
            # erode_amount = 1  # 腐蚀量
            # dilate_amount = 15  # 膨胀量
            # repair_mask = self.erode_dilate_mask(repair_mask, erode_amount=erode_amount, dilate_amount=dilate_amount , iterations=iterations)

            # vis Masked image bellow
        else:
            print(inp_mask.shape)
            repair_mask = inp_mask
            # print(repair_mask.shape)
        image_with_hole = np.where(repair_mask[:, :, None], 0, image).astype(np.uint8)
        if mode == 0:
            inpainted_image = cv2.inpaint(image, repair_mask, 3, cv2.INPAINT_TELEA)
            # inpainted_image = cv2.inpaint(image_with_hole, repair_mask, 3, cv2.INPAINT_NS)
            # 将修复后的背景与新图像合并
            # 计算需要从修复后的图像中填充的区域：原始位置且非新位置
            # fill_mask = mask & ~shifted_mask
            final_image = np.where(shifted_mask[:, :, None], new_image , inpainted_image)
        elif mode ==1:
            #TODO: LaMa inpainting & refinement
            # Input formats: np.ndarray or PIL.Image.Image.
            # (3 channel input image & 1 channel binary mask image where pixels with 255 will be inpainted).
            # Output format: PIL.Image.Image
            # mask 没必要太精确，对边缘进行侵蚀
            #lama inpainteing
            inpainted_image = inpainter(image, repair_mask)
            # 计算需要从修复后的图像中填充的区域：原始位置且非新位置
            # fill_mask = mask & ~shifted_mask
            final_image = np.where(shifted_mask[:, :, None], new_image,inpainted_image)
        elif mode==2: #Stable diffusion inpainting
            # prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            # image and mask_image should be PIL images.
            # The mask structure is white for inpainting and black for keeping as is
            # image = pipe(prompt=inp_prompt, image=image, mask_image=mask_image).images[0]
            if inp_prompt is None:
                inp_prompt = ""
            image = Image.fromarray(image)
            repair_mask = Image.fromarray(repair_mask)
            inpainted_image = self.sd_inpainter(prompt=inp_prompt, image=image, mask_image=repair_mask).images[0]
            if inpainted_image.size != image.size:
                inpainted_image = inpainted_image.resize(image.size)
            final_image = np.where(shifted_mask[:, :, None], new_image, inpainted_image)
        return final_image,image_with_hole

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            latents = self.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
        return latents
    def ddim_inv(self, latent, prompt, emb_im=None,ddim_num_steps=50):
        ddim_inv = DDIMInversion(model=self, NUM_DDIM_STEPS=ddim_num_steps)
        ddim_latents = ddim_inv.invert(ddim_latents=latent.unsqueeze(2), prompt=prompt, emb_im=emb_im)
        return ddim_latents

    def run_my_Baseline(self, original_image, mask, prompt,
                        seed, selected_points, guidance_scale, num_step, max_resolution, mode, dilate_kernel_size, start_step, mask_ref = None, eta=0,move_with_diy_mask=False,
                        ):
        seed_everything(seed)
        # energy_scale = energy_scale * 1e3
        # get mask shifting values
        # print(mask_ref.shape)
        # storage=True
        # if storage:
        #     if isinstance(mask, np.ndarray):
        #         mask_store = Image.fromarray(mask)
        #     path = "/data/Hszhu/DragonDiffusion/examples/masks/"
        #     i = 0
        #     name =f'{i}.png'
        #     while os.path.exists(os.path.join(path,name)):
        #         i += 1
        #         name = f'{i}.png'
        #     mask_store.save(os.path.join(path,name))

        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution * max_resolution)
        print(img.shape) #768
        print(input_scale)
        if input_scale != 1:
            mask,_ = resize_numpy_image(mask, max_resolution * max_resolution)
            if len(mask_ref)>1:
                mask_ref  = resize_numpy_image(mask_ref, max_resolution * max_resolution)[0]
        x = []
        y = []
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx % 2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        dx = x_cur[0] - x[0]
        dy = y_cur[0] - y[0]
        #resize process
        dx = int(dx * input_scale)
        dy = int(dy * input_scale)
        # copy-paste-inpaint to get initial img
        img_preprocess,inpaint_mask = self.move_and_inpaint(img, mask, dx, dy,self.inpainter,mode,dilate_kernel_size,mask_ref,prompt,move_with_diy_mask)#1 for lama 2 for sd
        # got latent
        img = img_preprocess
        if isinstance(img,np.ndarray):
            img = Image.fromarray(img)
        # version = '1' #version 2 is not properly implemented yet
        final_im,noised_img = \
            self.DDIM(
                img,
                prompt,
                prompt,
                num_steps=num_step,
                start_step=start_step,
                guidance_scale=guidance_scale,
                version = 0,
                eta = eta,
            )



        return [img_preprocess], [final_im] ,[noised_img],[inpaint_mask]#iterable image for gallery

    def get_depth_V2(self, image_raw,base_depth=0.1,min_z=10,max_z=255):
        # h, w = image_raw.shape[:2]
        #raw_img = cv2.imread('your/image/path') #RGB img is enough ndarray
        depth= self.depth_anything.infer_image(image_raw) #ndarray H W back
        # depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        # GeoDiffuser Processor
        # depth = depth.max() - depth  # Negating depth as relative depth estimators assign high values to close objects. You can also try 1/depth (inverse depth, but we found this to work better prima facie)
        # depth = depth + depth.max() * base_depth  # This helps in reducing depth smearing where translate_factor is between 0 to 1.
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255  # Normalizes from 0-1.
        depth = depth.max() - depth  # Negating depth as relative depth estimators assign high values to close objects. You can also try 1/depth (inverse depth, but we found this to work better prima facie)
        depth = depth + depth.max() * base_depth  # This helps in reducing depth smearing where translate_factor is between 0 to 1.
        # depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalizes from 0-1.
        # modified by clawer
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * (max_z - min_z) + min_z
        return depth

    def get_depth(self,image_raw):
        h, w = image_raw.shape[:2]
        image = image_raw / 255
        image = self.transform({'image': image})['image']
        # Reshape transformed image to (H, W, C) for plotting
        # image_plot = np.transpose(image, (1, 2, 0))
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        base_depth = 0.1
        # depth_unit = 255
        # EPISILON = 1e-8
        with torch.no_grad():
            depth = self.depth_anything(image)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        # GeoDiffuser Processor
        depth = depth.max() - depth  # Negating depth as relative depth estimators assign high values to close objects. You can also try 1/depth (inverse depth, but we found this to work better prima facie)
        depth = depth + depth.max() * base_depth  # This helps in reducing depth smearing where translate_factor is between 0 to 1.
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255  # Normalizes from 0-1.
        return depth.cpu().numpy().astype(np.uint8)
    def run_my_Baseline_full_3D(self, original_image, mask, prompt,INP_prompt,
                        seed, guidance_scale, num_step, max_resolution, mode, dilate_kernel_size, start_step,tx,ty,tz,rx,ry,rz,sx,sy,sz, mask_ref = None, eta=0, use_mask_expansion=True,
                        standard_drawing=True,contrast_beta=1.67,exp_mask_type=0,strong_inpaint=True,cross_enhance=False,
                             mask_threshold=0.1,mask_threshold_target=0.1,blending_alpha=0.5,splatting_radius=0.015,splatting_tau = 0.0,splatting_points_per_pixel = 30,focal_length=1080,
                        ):
        exp_mask_target = ['INV','FOR','BOTH']
        target_mask_type = exp_mask_target[int(exp_mask_type)]
        seed_everything(seed)
        transforms = [tx, ty, tz, rx, ry, rz, sx, sy, sz]
        print(f'trans {transforms}')
        #select and resize
        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution * max_resolution)

        # depth = self.get_depth(img)
        depth = self.get_depth_V2(img,base_depth = 0.1,max_z = 255,min_z = 20)
        depth_plot = ((depth-depth.min())/(depth.max()-depth.min())*255).astype(np.uint8)
        depth_plot = np.repeat(depth_plot[..., np.newaxis], 3, axis=-1)
        depth_plot = Image.fromarray(depth_plot)
        depth_map = depth


        print(img.shape) #768
        print(input_scale)
        FINAL_WIDTH = img.shape[1]
        FINAL_HEIGHT = img.shape[0]
        # FX = FINAL_WIDTH * 0.6
        # FY = FINAL_HEIGHT * 0.6
        F = focal_length
        FX = F
        FY = F

        if standard_drawing: #box only input
            mask_to_use = mask
            if input_scale != 1:
                mask_to_use, _ = resize_numpy_image(mask_to_use, max_resolution * max_resolution)
            if strong_inpaint:
                target_mask = mask_to_use
            mask_to_use = self.dilate_mask(mask_to_use, dilate_kernel_size) * 255 #dilate for better expansion mask
            return_target = False #which designed for casual draw input target/expanded mask

        else:
            mask_to_use = mask_ref
            if len(mask_ref) > 1:
                mask_to_use, _ = resize_numpy_image(mask_to_use, max_resolution * max_resolution)
                return_target = False
                if strong_inpaint :
                    if cross_enhance:#use box mask to retain
                        target_mask, _ = resize_numpy_image(mask, max_resolution * max_resolution)
                        target_mask = np.array(self.dilate_mask(target_mask, dilate_kernel_size) * 255)
                    else: #use strict expansion mask to retain
                        return_target = True
        #mask expansion
        if return_target:
            self.controller.contrast_beta = contrast_beta * 100
            target_mask = self.gradio_mask_expansion_func(img=img, mask=mask_to_use, prompt="",
                                                          guidance_scale=guidance_scale, num_step=5,
                                                          eta=eta, roi_expansion=True,
                                                          mask_threshold=mask_threshold, post_process='hard',
                                                          )  # ndarray mask
        if use_mask_expansion:
            self.controller.contrast_beta = contrast_beta
            expand_mask= self.gradio_mask_expansion_func(img=img, mask=mask_to_use, prompt="",
                                                         guidance_scale=guidance_scale, num_step=5,
                                                         eta=eta,roi_expansion=True,
                                                         mask_threshold=mask_threshold,post_process='hard',) #ndarray mask
        else:
            expand_mask = mask_to_use

        # copy-paste-inpaint to get initial img
        # if strong_inpaint:
        img_preprocess, inpaint_mask, shifted_mask, retain_mask = self.move_and_inpaint_with_expansion_mask_3D(img, expand_mask,depth_map, transforms, FX,FY,True,
                                                                                                               self.inpainter,mode,dilate_kernel_size,INP_prompt,target_mask,splatting_radius,
                                                                                                               splatting_tau,splatting_points_per_pixel)

        # else:
        #     img_preprocess,inpaint_mask,shifted_mask = self.move_and_inpaint_with_expansion_mask(img, expand_mask, dx, dy,self.inpainter,mode,dilate_kernel_size,INP_prompt,resize_scale,rotation_angle,flip_horizontal,flip_vertical)
        #     retain_mask = None
        img = img_preprocess
        if isinstance(img,np.ndarray):
            img = Image.fromarray(img)

        #prepare target expansion mask
        if retain_mask is not None:
            shifted_mask = retain_mask['obj_region']
        shifted_mask = self.prepare_controller_ref_mask(shifted_mask)

        final_im,noised_img,candidate_mask,retain_region = \
            self.DDIM_DDPM_MASK(
                img,
                prompt,
                prompt,
                num_steps=num_step,
                start_step=start_step,
                guidance_scale=guidance_scale,
                version = 0,
                eta = eta,
                roi_expansion=True,mask_threshold=mask_threshold_target, post_process='hard',mask=shifted_mask,target_mask_type=target_mask_type,
                retain_mask = retain_mask, dilate_kernel_size=dilate_kernel_size,blending_alpha=blending_alpha
            )
        return [img_preprocess], [final_im] ,[noised_img],[inpaint_mask],[expand_mask],[candidate_mask],[retain_region],[depth_plot]#iterable image for gallery



    def run_my_Baseline_full(self, original_image, mask, prompt,INP_prompt,
                        seed, selected_points, guidance_scale, num_step, max_resolution, mode, dilate_kernel_size, start_step, mask_ref = None, eta=0, use_mask_expansion=True,
                        standard_drawing=True,contrast_beta=1.67,exp_mask_type=0,resize_scale=1.0,rotation_angle=None,strong_inpaint=True,flip_horizontal=False, flip_vertical=False,cross_enhance=False,
                             mask_threshold=0.1,mask_threshold_target=0.1,blending_alpha=0.5,
                        ):
        exp_mask_target = ['INV','FOR','BOTH']
        target_mask_type = exp_mask_target[int(exp_mask_type)]
        seed_everything(seed)
        # energy_scale = energy_scale * 1e3
        # get mask shifting values
        # print(mask_ref.shape)
        # storage=True
        # if storage:
        #     if isinstance(mask, np.ndarray):
        #         mask_store = Image.fromarray(mask)
        #     path = "/data/Hszhu/DragonDiffusion/examples/masks/"
        #     i = 0
        #     name =f'{i}.png'
        #     while os.path.exists(os.path.join(path,name)):
        #         i += 1
        #         name = f'{i}.png'
        #     mask_store.save(os.path.join(path,name))

        #select and resize
        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution * max_resolution)
        print(img.shape) #768
        print(input_scale)
        if standard_drawing: #box only input
            mask_to_use = mask
            if input_scale != 1:
                mask_to_use, _ = resize_numpy_image(mask_to_use, max_resolution * max_resolution)
            if strong_inpaint:
                target_mask = mask_to_use
            mask_to_use = self.dilate_mask(mask_to_use, dilate_kernel_size) * 255 #dilate for better expansion mask
            return_target = False

        else:
            mask_to_use = mask_ref
            if len(mask_ref) > 1:
                mask_to_use, _ = resize_numpy_image(mask_to_use, max_resolution * max_resolution)
                return_target = False
                if strong_inpaint :
                    if cross_enhance:#use box mask to retain
                        target_mask, _ = resize_numpy_image(mask, max_resolution * max_resolution)
                        target_mask = np.array(self.dilate_mask(target_mask, dilate_kernel_size) * 255)
                    else: #use strict expansion mask to retain
                        return_target = True



        #get move
        x = []
        y = []
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx % 2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        #IF ERROR is Raised , check whether you select start and end moving points
        dx = x_cur[0] - x[0]
        dy = y_cur[0] - y[0]
        dx = int(dx * input_scale)
        dy = int(dy * input_scale)
        #mask expansion
        if return_target:
            self.controller.contrast_beta = contrast_beta * 100
            target_mask = self.gradio_mask_expansion_func(img=img, mask=mask_to_use, prompt="",
                                                          guidance_scale=guidance_scale, num_step=5,
                                                          eta=eta, roi_expansion=True,
                                                          mask_threshold=mask_threshold, post_process='hard',
                                                          )  # ndarray mask
        if use_mask_expansion:
            self.controller.contrast_beta = contrast_beta
            expand_mask= self.gradio_mask_expansion_func(img=img, mask=mask_to_use, prompt="",
                                                         guidance_scale=guidance_scale, num_step=5,
                                                         eta=eta,roi_expansion=True,
                                                         mask_threshold=mask_threshold,post_process='hard',) #ndarray mask
        else:
            expand_mask = mask_to_use

        # copy-paste-inpaint to get initial img
        # if strong_inpaint:
        img_preprocess, inpaint_mask, shifted_mask, retain_mask = self.move_and_inpaint_with_expansion_mask_new(img, expand_mask, dx, dy, self.inpainter,mode,dilate_kernel_size,INP_prompt, resize_scale,rotation_angle,target_mask,flip_horizontal,flip_vertical)
        # else:
        #     img_preprocess,inpaint_mask,shifted_mask = self.move_and_inpaint_with_expansion_mask(img, expand_mask, dx, dy,self.inpainter,mode,dilate_kernel_size,INP_prompt,resize_scale,rotation_angle,flip_horizontal,flip_vertical)
        #     retain_mask = None
        img = img_preprocess
        if isinstance(img,np.ndarray):
            img = Image.fromarray(img)

        #prepare target expansion mask
        if retain_mask is not None:
            shifted_mask = retain_mask['obj_region']
        shifted_mask = self.prepare_controller_ref_mask(shifted_mask)

        final_im,noised_img,candidate_mask,retain_region = \
            self.DDIM_DDPM_MASK(
                img,
                prompt,
                prompt,
                num_steps=num_step,
                start_step=start_step,
                guidance_scale=guidance_scale,
                version = 0,
                eta = eta,
                roi_expansion=True,mask_threshold=mask_threshold_target, post_process='hard',mask=shifted_mask,target_mask_type=target_mask_type,
                retain_mask = retain_mask, dilate_kernel_size=dilate_kernel_size,blending_alpha=blending_alpha
            )
        return [img_preprocess], [final_im] ,[noised_img],[inpaint_mask],[expand_mask],[candidate_mask],[retain_region]#iterable image for gallery
    def normalize_expansion_mask(self,mask,exp_mask,roi_expansion):
        if roi_expansion:
            candidate_mask = torch.ones_like(exp_mask)
            expansion_loc = mask < 125
            average_expansion_masks_of_interested = exp_mask[expansion_loc]
            ma, mi = average_expansion_masks_of_interested.max(), average_expansion_masks_of_interested.min()
            average_expansion_masks_norm = (average_expansion_masks_of_interested - mi) / (ma - mi)
            candidate_mask[expansion_loc] = average_expansion_masks_norm
        else:
            ma, mi = exp_mask.max(), exp_mask.min()
            average_expansion_masks_norm = (exp_mask - mi) / (ma - mi)
            candidate_mask = average_expansion_masks_norm
        return  candidate_mask

    def bfs_distance_transform(self, expand_mask_bool, edge_pixels):
        # 初始化距离映射为无穷大
        distance_map = torch.full_like(expand_mask_bool, float('inf'), dtype=torch.float32)

        # 将边缘像素的距离设为0
        queue = [(x.item(), y.item()) for x, y in torch.nonzero(edge_pixels, as_tuple=False)]
        for x, y in queue:
            distance_map[x, y] = 0

        # 使用BFS计算距离，仅对expand_mask_bool为True的点计算距离
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.pop(0)
            current_distance = distance_map[x, y]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < expand_mask_bool.shape[0] and 0 <= ny < expand_mask_bool.shape[1]:
                    if expand_mask_bool[nx, ny] and distance_map[nx, ny] > current_distance + 1:
                        distance_map[nx, ny] = current_distance + 1
                        queue.append((nx, ny))

        return distance_map

    def generate_dynamic_threshold_function(self,mask_threshold, control_value, control_point, distance_range=(0, 3),
                                            device='cpu'):
        max_thr = 1.0
        control_point = torch.tensor(control_point, dtype=torch.float32, device=device)
        distance_end = torch.tensor(distance_range[1], dtype=torch.float32, device=device)
        value = torch.tensor((control_value - mask_threshold) / (max_thr - mask_threshold), dtype=torch.float32,
                             device=device)

        # Recalculate scale based on control_value and control_point
        scale = torch.log(value) / (control_point - distance_end)

        # Return the dynamic threshold function
        def dynamic_threshold_function(relative_distance):
            relative_distance = torch.tensor(relative_distance, dtype=torch.float32, device=device)
            # dynamic_thr = mask_threshold + (1 - mask_threshold) * torch.exp(relative_distance - 1)
            return mask_threshold + (max_thr - mask_threshold) * torch.exp(relative_distance * scale - 1) / torch.exp(
                distance_end * scale - 1)

        return dynamic_threshold_function

    def plot_bfs_distance_map(self, distance_map, save_path=None,img_name=None,title=None):
        # Ensure the distance_map is on the CPU for plotting
        save_path = os.path.join(save_path,img_name+'.png')
        distance_map = distance_map.cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.imshow(distance_map, cmap='viridis')
        plt.colorbar(label='BFS Distance')
        if title is None:
            plt.title('BFS Distance Map')
        else:
            plt.title(f'{title}')
        if save_path:
            plt.savefig(save_path)
            plt.show()
            plt.close()
        else:
            plt.show()

    import torch.nn.functional as F

    def get_structuring_element(self, kernel_size):
        # 创建一个正方形的结构元素
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
        # 将kernel移到和mask相同的设备上
        kernel = kernel.to(self.device)
        return kernel

    def opening_operation(self,mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 先腐蚀
        eroded = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        eroded = (eroded == torch.max(eroded)).float()
        # 后膨胀
        opened = F.conv2d(eroded.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        opened = (opened > 0).float()
        return opened

    def erode_operation(self, mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 腐蚀
        eroded = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        eroded = (eroded == torch.max(eroded)).float()

        return eroded
    def dilation_operation(self, mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 膨胀
        opened = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        opened = (opened > 0).float()
        return opened
    def closing_operation(self,mask, kernel_size=5):
        mask=mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 先膨胀
        dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        dilated = (dilated > 0).float()
        # 后腐蚀
        closed = F.conv2d(dilated.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        closed = (closed == torch.max(closed)).float()
        return closed

    def view(self,mask, title='Mask',name=None):
        """
        显示输入的mask图像

        参数:
        mask (torch.Tensor): 要显示的mask图像，类型应为torch.bool或torch.float32
        title (str): 图像标题
        """
        # 确保输入的mask是float类型以便于显示
        if isinstance(mask,np.ndarray):
            mask_new = mask
        else:
            mask_new = mask.float()
            mask_new = mask_new.detach().cpu()
            mask_new = mask_new.numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(mask_new, cmap='gray')
        plt.title(title)
        plt.axis('off')  # 去掉坐标轴
        plt.savefig(name+'.png')
        # plt.show()
    def get_self_adaptive_init_thr(self,obj_mask,strict_threshold,adp_k):
        valid_region = (obj_mask > 0).sum()
        invalid_region = (obj_mask == 0).sum()
        rate = valid_region / (valid_region+invalid_region)
        print(f'objmask occupy rate:{rate}')
        #物体越大，相关像素理应越多,阈值可相对降低，反之物体小，没必要这么多像素阈值提高一些
        # threshold = strict_threshold * (1-rate)
        threshold = strict_threshold * torch.exp(-adp_k * rate)

        return threshold


    def Mask_post_process(self,expand_mask, ref_mask, mask_threshold,type,control_value=0.15,control_point=1,adp_k=5,use_contrast=True,contrast_beta=1.67):

        ref_mask_bool = ref_mask > 0
        # 找到原始mask的边缘像素
        edge_pixels = self.find_edge_pixels(ref_mask_bool)
        edge_coords = torch.nonzero(edge_pixels, as_tuple=False)

        #expansion mask process hard / soft
        # if type=='soft':
        #     # adaptive_mask_threshold = self.get_self_adaptive_init_thr(ref_mask,mask_threshold,adp_k)
        #     # print(f'adaptive_thr : {adaptive_mask_threshold}')
        #     # expand_mask_bool = expand_mask > adaptive_mask_threshold #初筛
        #     # mask_threshold = adaptive_mask_threshold
        #     expand_mask_bool = expand_mask > mask_threshold  # 初筛
        # elif type=='hard':
        expand_mask_bool = expand_mask > mask_threshold #初筛

        #腐蚀
        expand_mask_bool = self.erode_operation(expand_mask_bool, kernel_size=10)
        expand_regions = expand_mask_bool.bool() & ~ref_mask_bool.bool()
        if type == 'soft':#计算所有expansion mask 位置到 ref mask位置的BFS距离，依据距离定阈值

            # 计算BFS距离映射
            distance_map = self.bfs_distance_transform(expand_regions,edge_pixels)
            distance_map[ref_mask_bool] = 0
            # 计算物体边界最大距离-表示物体大小
            # max_edge_distance = torch.max(torch.cdist(edge_coords.float(), edge_coords.float(), p=1))
            # distance_map /= max_edge_distance #norm
            valid_distances = distance_map[torch.isfinite(distance_map)]
            max_distance = valid_distances.max() if valid_distances.numel() > 0 else 1.0
            distance_map /= max_distance
            #TODO: 添加一个plot函数方便我看BFS距离图
            self.plot_bfs_distance_map(distance_map,self.BFS_SAVE_PATH,self.image_name,'BFS-DISTANCE')


            # 动态阈值调整
            # valid_distances = distance_map[torch.isfinite(distance_map)]
            # max_distance = valid_distances.max() if valid_distances.numel() > 0 else 1.0
            distance_range = (0, 1)
            # if control_point is None:
            #     control_point = max_distance
            dynamic_function = self.generate_dynamic_threshold_function(mask_threshold,control_value,control_point,distance_range,device=expand_mask_bool.device)
            dynamic_thr = dynamic_function(relative_distance=distance_map)
            self.plot_bfs_distance_map(dynamic_thr, self.BFS_SAVE_PATH, self.image_name+'_thr','THR')
            refined_mask_init = expand_mask > dynamic_thr
            expand_regions = refined_mask_init.bool() & ~ref_mask_bool.bool()
            # 初始化访问标记 DFS搜索通路
            visited = set()

            # 从边缘像素开始进行DFS
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for coord in edge_coords:
                x, y = coord[0].item(), coord[1].item()
                self.dfs(x, y, expand_regions, visited, directions)

            # 构建refined mask
            refined_mask = torch.zeros_like(expand_regions, dtype=torch.bool)
            for (x, y) in visited:
                refined_mask[x, y] = True

            # 确保包含原始mask的像素
            refined_mask |= ref_mask_bool



        # 不连通的位置直接去掉
        elif type == 'hard':
            # 初始化访问标记 DFS搜索通路
            visited = set()

            # 从边缘像素开始进行DFS
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for coord in edge_coords:
                x, y = coord[0].item(), coord[1].item()
                self.dfs(x, y, expand_regions, visited, directions)

            # 构建refined mask
            refined_mask = torch.zeros_like(expand_regions, dtype=torch.bool)
            for (x, y) in visited:
                refined_mask[x, y] = True

            # 确保包含原始mask的像素
            refined_mask |= ref_mask_bool


        refined_mask = self.dilation_operation(refined_mask, kernel_size=15)

        return refined_mask.to(torch.uint8) * 255


    def find_edge_pixels(self,mask):
        # 使用binary_dilation找到边缘
        kernel = torch.tensor([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], device=mask.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        dilated_mask = F.conv2d(mask, kernel, padding=1) > 0
        # 确保dilated_mask也是布尔类型
        dilated_mask = dilated_mask.bool()
        edge_pixels = dilated_mask & ~mask.bool()
        return edge_pixels.squeeze(0).squeeze(0)

    def is_valid(self,x, y, mask):
        # 检查像素是否在mask范围内且值为True
        return 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y]

    def dfs(self,x, y, mask, visited, directions):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if self.is_valid(nx, ny, mask) and (nx, ny) not in visited:
                    stack.append((nx, ny))

    def contrast_operation(self,mask,contrast_beta,clamp=True,min_v=0,max_v=1,dim=-1):
        if dim is not None:
            mean_value = mask.mean(dim=dim)[:,:,None]
        else:
            mean_value = mask.mean()

        #increase varience
        contrasted_mask = (mask - mean_value) * contrast_beta + mean_value
        del mean_value
        if clamp:
            return contrasted_mask.clamp(min=min_v,max=max_v)
        return contrasted_mask
    def prepare_controller_ref_mask(self,mask):
        #ndarray mask -> Tensor
        if mask.ndim == 3:
            mask = mask[:,:,0]
        mask = torch.Tensor(mask).to(self.device)
        self.controller.obj_mask = mask
        return mask

    def fetch_expansion_mask_from_store(self,expansion_masks,mask,roi_expansion,post_process,mask_threshold):
        #Tensor masks -> 0-255 cpu ndarray mask
        #expansion_masks = [ddim] or [ddpm] or [ddim,ddpm]
        step_masks = []
        for exp_msks in expansion_masks: #exp_msks is a dict contains of different steps masks Tensors
            step_masks.extend([v for i, v in exp_msks.items()])
        # TODO  exponential moving average refine / weighted sum
        average_expansion_masks = sum(step_masks) / len(step_masks)
        print(mask.shape)
        print(average_expansion_masks.shape)
        self.view(mask,name='mask')
        norm_exp_mask = self.normalize_expansion_mask(mask, average_expansion_masks, roi_expansion, )
        if post_process is not None:
            assert post_process in ['hard', 'soft'], f'not implement method: {post_process}'
            # post process based on distance and init thr
            norm_exp_mask = self.Mask_post_process(norm_exp_mask, mask, mask_threshold, post_process, )
        else:
            norm_exp_mask = norm_exp_mask > mask_threshold
            norm_exp_mask = norm_exp_mask.to(torch.uint8) * 255

        return norm_exp_mask.detach().cpu().numpy()

    def gradio_mask_expansion_func(self, img, mask, prompt,
                         guidance_scale, num_step, eta=0,roi_expansion=True,
                         mask_threshold=0.1,post_process='hard',
                        ):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        #reference mask prepare
        mask = self.prepare_controller_ref_mask(mask)
        _ = \
            self.MY_DDIM_INV(
                img,
                prompt,
                num_steps=num_step,
                guidance_scale=guidance_scale,
                eta = eta
            )
        expansion_masks = self.controller.expansion_mask_store #expansion mask & average of up mid down resized corresponded self attention maps
        self.controller.expansion_mask_store = {} #reset for next image
        # step_masks = [v for i, v in expansion_masks.items()]
        candidate_mask = self.fetch_expansion_mask_from_store([expansion_masks],mask,roi_expansion,post_process,mask_threshold)
        return candidate_mask

    @torch.no_grad()
    def DDIM_inversion_mask_expansion_func(self, img, mask, prompt,
                         guidance_scale, num_step, eta=0,roi_expansion=True,
                         mask_threshold=0.1,post_process='hard',use_mask_expansion=True,
                        ):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        #reference mask prepare
        mask = self.prepare_controller_ref_mask(mask)
        inverted_latents = \
            self.MY_DDIM_INV(
                img,
                prompt,
                num_steps=num_step,
                guidance_scale=guidance_scale,
                eta = eta
            )
        if not use_mask_expansion:
            self.controller.expansion_mask_store = {}  # reset for next image
            # torch.cuda.empty_cache()  # 清理显存
            return mask.detach().cpu().numpy(),inverted_latents
        expansion_masks = self.controller.expansion_mask_store #expansion mask & average of up mid down resized corresponded self attention maps
        self.controller.expansion_mask_store = {} #reset for next image
        # step_masks = [v for i, v in expansion_masks.items()]
        candidate_mask = self.fetch_expansion_mask_from_store([expansion_masks],mask,roi_expansion,post_process,mask_threshold)
        # 清理显存
        del expansion_masks
        return candidate_mask,inverted_latents

    @torch.no_grad()
    def mask_expansion_with_ddim_inv(self, original_image, mask, prompt,
                        seed, guidance_scale, num_step, max_resolution, eta=0,controller=None,roi_expansion=True,mask_dilation=False,dilation_kernel_size=15,
                                     maintain_step_mask=False,mask_threshold=0.1,post_process=None,control_value=0.15,control_point=1,adp_k=5,
                                     use_contrast=False,contrast_beta=1.67,
                        ):
        seed_everything(seed)
        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution * max_resolution)
        print(img.shape)  # 768
        print(input_scale)
        if input_scale != 1:
            mask, _ = resize_numpy_image(mask, max_resolution * max_resolution)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        # 将掩码转换为灰度并应用阈值


        if mask_dilation:
            mask = mask.astype(bool)
            mask = self.dilate_mask(mask,dilation_kernel_size)*255
        mask = torch.Tensor(mask).to(self.device)
        controller.obj_mask = mask
        inverted_latent = \
            self.MY_DDIM_INV(
                img,
                prompt,
                num_steps=num_step,
                guidance_scale=guidance_scale,
                eta=eta,
            )
        expansion_masks = controller.expansion_mask_store #expansion mask & average of up mid down resized corresponded self attention maps
        controller.expansion_mask_store = {} #reset for next image
        if maintain_step_mask:
            candidate_mask ={}
            for k,v in expansion_masks.items():
                average_expansion_masks_per_step = v
                norm_exp_mask= self.normalize_expansion_mask(mask,average_expansion_masks_per_step,roi_expansion)
                if use_contrast:
                    norm_exp_mask = self.contrast_operation(norm_exp_mask, contrast_beta,clamp=True,min_v=0,max_v=1)
                if post_process is not None:
                    # post process based on distance and init thr
                    norm_exp_mask = self.Mask_post_process(norm_exp_mask, mask, mask_threshold, post_process,
                                                           control_value, control_point,adp_k,)
                else:
                    norm_exp_mask = norm_exp_mask > mask_threshold
                candidate_mask[k] = norm_exp_mask
        else:
            # step_masks = [v for i, v in expansion_masks.items()]
            step_masks = [v for i,v in expansion_masks.items()]
            #TODO  exponential moving average refine / weighted sum
            average_expansion_masks = sum(step_masks)/len(step_masks)
            norm_exp_mask = self.normalize_expansion_mask(mask,average_expansion_masks,roi_expansion,)
            if use_contrast:
                #use clamp for mask after normalized
                norm_exp_mask = self.contrast_operation(norm_exp_mask, contrast_beta,clamp=True,min_v=0,max_v=1,dim=None)
            if post_process is not None:
                assert  post_process in ['hard','soft'],f'not implement method: {post_process}'
                # post process based on distance and init thr
                norm_exp_mask = self.Mask_post_process(norm_exp_mask, mask, mask_threshold, post_process,
                                                       control_value, control_point,adp_k,)
            else:
                norm_exp_mask = norm_exp_mask > mask_threshold
                norm_exp_mask = norm_exp_mask.to(torch.uint8) * 255

            candidate_mask = norm_exp_mask



        return inverted_latent,candidate_mask,mask,img
    #forward
    @torch.no_grad()
    def sample(self,
               prompt,
               start_step=0,
               start_latents=None,
               guidance_scale=3.5,
               num_inference_steps=30,
               num_images_per_prompt=1,
               do_classifier_free_guidance=True,
               negative_prompt="",
               eta=0,
               ):

        # Encode prompt
        text_embeddings = self._encode_prompt(
            prompt, self.device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Set num inference steps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Create a random starting point if we don't have one already
        if start_latents is None:
            start_latents = torch.randn(1, 4, 64, 64, device=self.device)
            start_latents *= self.scheduler.init_noise_sigma

        latents = start_latents.clone()

        for i in tqdm(range(start_step, num_inference_steps)):

            t = self.scheduler.timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Normally we'd rely on the scheduler to handle the update step:
            # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample


            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, eta=eta,).prev_sample

            # Instead, let's do it ourselves:
            # prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
            # alpha_t = self.scheduler.alphas_cumprod[t.item()]
            # alpha_t_prev = self.scheduler.alphas_cumprod[prev_t]
            # predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            # direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
            # latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

        # Post-processing
        images = self.decode_latents(latents)
        images = self.numpy_to_pil(images)

        return images
    ## Inversion
    @torch.no_grad()
    def invert(
            self,
            start_latents,
            prompt,
            guidance_scale=3.5,
            num_inference_steps=80,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
    ):

        # Encode prompt
        text_embeddings = self._encode_prompt(
            prompt, self.device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Latents are now the specified start latents
        latents = start_latents.clone()

        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Set num inference steps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)

        for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue

            t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
            next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.scheduler.alphas_cumprod[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                    1 - alpha_t_next
            ).sqrt() * noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.cat(intermediate_latents)

    @torch.no_grad()
    def invert_with_attn_map_store(
            self,
            start_latents,
            prompt,
            guidance_scale=3.5,
            num_inference_steps=80,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
    ):

        # Encode prompt
        text_embeddings = self._encode_prompt(
            prompt, self.device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Latents are now the specified start latents
        latents = start_latents.clone()

        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []
        self_attn_maps = []

        # Set num inference steps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)

        for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue

            t = timesteps[i]

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
            next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = self.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.scheduler.alphas_cumprod[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                    1 - alpha_t_next
            ).sqrt() * noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.cat(intermediate_latents)

    def DDIM(self,input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5,version=0,eta=0):
        ##input_image original PIL 512,512
        #VERSION ONE : DDIM Inversion official code in huggingface
        #VERSION TWO : DDIM inversion code in dragon utils
        with torch.no_grad():
            if version == 0:
                latent = self.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(self.device) * 2 - 1)
                l = 0.18215 * latent.latent_dist.sample()
                inverted_latents = self.invert(l, input_image_prompt, num_inference_steps=num_steps,guidance_scale=guidance_scale,)

            elif version == 1:
                #Dragon diffusion ddim inversion code,but with bug in self.unet forward function while they use diffenrent unet
                #bother to check ,just use official ddim inversion code with guidance
                img_tensor = (PILToTensor()(input_image) / 255.0 - 0.5) * 2
                img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
                latent = self.image2latent(img_tensor)
                inverted_latents = self.ddim_inv(latent=latent, prompt=input_image_prompt,ddim_num_steps=num_steps)


            start_latents = inverted_latents[-(start_step + 1)][None]
            noised_image = self.decode_latents(start_latents).squeeze(0)
            # print(noised_image.shape)
            noised_image = self.numpy_to_pil(noised_image)[0]
            # print(noised_image.size)
            # print(noised_image.shape)
            final_im = self.sample(
                edit_prompt,
                start_latents=start_latents,
                start_step=start_step,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                eta=eta,
            )[0]
            # print(final_im.size)
        return final_im , noised_image

    def alpha_blend(self,img1, img2, alpha):
        """
        对两个图像进行 alpha 混合
        :param img1: 第一个输入图像（0-255 范围的 numpy 数组）
        :param img2: 第二个输入图像（0-255 范围的 numpy 数组）
        :param alpha: alpha 值（0-1 之间的浮点数）
        :return: 混合后的图像
        """
        blended_img = (1 - alpha) * img2 + alpha * img1
        return blended_img.astype(np.uint8)
    def DDIM_DDPM_MASK(self, input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5,
             version=0, eta=0, roi_expansion=True,mask_threshold=0.1, post_process='hard',mask=None,target_mask_type='FOR',retain_mask=None,dilate_kernel_size=15,
                       blending_alpha=0.5):
        ##input_image original PIL 512,512
        # VERSION ONE : DDIM Inversion official code in huggingface
        # VERSION TWO : DDIM inversion code in dragon utils
        assert target_mask_type in ['INV','FOR','BOTH'],f'{target_mask_type} is not implemented,please check'
        with torch.no_grad():
            if version == 0:
                latent = self.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(self.device) * 2 - 1)
                l = 0.18215 * latent.latent_dist.sample()
                # inverted_latents = self.invert(l, input_image_prompt, num_inference_steps=num_steps,
                #                                guidance_scale=guidance_scale, )
                inverted_latents = self.invert(l, "", num_inference_steps=num_steps,
                                               guidance_scale=0.0, )

            candidate_mask = None
            #get ddim inv expansion target mask
            ddim_inv_expansion_masks = self.controller.expansion_mask_store  # expansion mask & average of up mid down resized corresponded self attention maps
            self.controller.expansion_mask_store = {}


            start_latents = inverted_latents[-(start_step + 1)][None]
            noised_image = self.decode_latents(start_latents).squeeze(0)
            # print(noised_image.shape)
            noised_image = self.numpy_to_pil(noised_image)[0]
            # print(noised_image.size)
            # print(noised_image.shape)
            final_im = self.sample(
                edit_prompt,
                start_latents=start_latents,
                start_step=start_step,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                eta=eta,
            )[0]

            # get ddpm forward expansion target mask
            ddpm_for_expansion_masks = self.controller.expansion_mask_store  # expansion mask & average of up mid down resized corresponded self attention maps
            self.controller.expansion_mask_store = {}  # reset for next image

            if target_mask_type == 'INV':
                candidate_mask = self.fetch_expansion_mask_from_store([ddim_inv_expansion_masks], mask, roi_expansion,post_process, mask_threshold)
            elif target_mask_type == 'FOR':
                candidate_mask = self.fetch_expansion_mask_from_store([ddpm_for_expansion_masks], mask, roi_expansion,post_process, mask_threshold)
            elif target_mask_type == 'BOTH':
                candidate_mask = self.fetch_expansion_mask_from_store([ddim_inv_expansion_masks,ddpm_for_expansion_masks], mask, roi_expansion,post_process, mask_threshold)

            # retain_mask = None
            if retain_mask is not None:
                # blending with CPI img to retain consistent part

                target_expansion = (candidate_mask>128).astype(bool)
                source_expansion = retain_mask['ori_expansion']
                before_obj_region = retain_mask['obj_region']

                retain_region_mask = ~(~(source_expansion | target_expansion) | before_obj_region)
                retain_region_mask = source_expansion | target_expansion
                # print(retain_region_mask)
                # self.view(retain_region_mask.astype(np.uint8) * 255, name="/data/Hszhu/DragonDiffusion/retain_region_mask")

                #blend with dilation like BrushNet
                retain_region = self.dilate_mask(retain_region_mask.astype(np.uint8), dilate_factor=dilate_kernel_size)[:,:,None]
                # retain_region = retain_region_mask.astype(np.uint8)[:,:,None]
                # print(retain_region.shape)
                input_image = np.array(input_image)
                ori_final_im = final_im
                final_im = np.array(final_im)
                pixel_wise_alpha = retain_region * blending_alpha
                final_im = self.alpha_blend(final_im,input_image,pixel_wise_alpha)
                print(final_im.dtype)
                if isinstance(final_im, np.ndarray):
                    final_im = Image.fromarray(final_im)
                final_im.save('blended_img.png')




                return ori_final_im, noised_image, candidate_mask, retain_region[:,:,0] * 255




        return final_im, noised_image, candidate_mask ,candidate_mask

    def forward_unet_features(self, z, t, encoder_hidden_states, h_feature=None, layer_idx=[0], interp_res_h=256, interp_res_w=256,):
        unet_output, all_intermediate_features,  copy_downblock = self.unet(
            z,
            t,
            h_sample = h_feature,
            copy = True,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True
            )
        all_return_features = []

        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode='bilinear')
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)

        h_feature = all_intermediate_features[0]
        # h_feature = copy_downblock[2]

        # h_feature = F.interpolate(h_feature, (interp_res_h, interp_res_w), mode='bilinear')

        return unet_output, return_features, h_feature







    def MY_DDIM_INV(self, input_image, input_image_prompt, num_steps=100, guidance_scale=3.5,
              eta=0):
        ##input_image original PIL 512,512
        # VERSION ONE : DDIM Inversion official code in huggingface
        # VERSION TWO : DDIM inversion code in dragon utils
        do_classifier_free_guidance = guidance_scale > 1
        with torch.no_grad():
            latent = self.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(self.device) * 2 - 1)
            l = 0.18215 * latent.latent_dist.sample()
            inverted_latents = self.invert_with_attn_map_store(l, input_image_prompt, num_inference_steps=num_steps,
                                           guidance_scale=guidance_scale,do_classifier_free_guidance=do_classifier_free_guidance ) #same with original function attention store was used in hook format
        return inverted_latents


class ClawerModel_v2(StableDiffusionPipeline):
    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)

    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def ctrl_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            mask,
            eta: float = 0.0,
            generator=None,
    ):
        """
        Predict the sample of the next step in the denoise process with eta control.

        Args:
            model_output (torch.FloatTensor): direct output from learned diffusion model.
            timestep (int): current discrete timestep in the diffusion chain.
            x (torch.FloatTensor): current instance of sample being created by diffusion process.
            eta (float): weight of noise for added noise in diffusion step. Default is 0.0.
            generator: random number generator.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: previous sample and predicted original sample.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        variance = self._get_variance(timestep, prev_timestep).to(self.device)
        std_dev_t = eta * variance ** (0.5)
        if model_output.shape[0] ==2: #reference stream
            #batch_0:LOCAL DDPM
            #batch_1:DDIM
            std_dev_t = torch.cat((std_dev_t[None,],torch.zeros_like(std_dev_t)[None,]))[:,None,None,None]
            mask = mask.repeat(1,4,1,1)
            mask = torch.cat((mask,torch.ones_like(mask)))

        if mask is not None:# local ddpm
            pred_dir_mask = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * model_output * mask #ddpm masked
            pred_dir = (1 - alpha_prod_t_prev ) ** (0.5) * model_output * (1-mask) + pred_dir_mask #ddim unmasked
        else: #full ddpm
            pred_dir = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * model_output
        x_prev = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir

        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            # batch_0:LOCAL DDPM
            # batch_1:DDIM
            if mask is None:
                variance = std_dev_t * variance_noise
            else:
                variance = std_dev_t * variance_noise * mask

            x_prev = x_prev + variance

        return x_prev, pred_x0

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance


    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = self.device
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.cuda())[0]
        return text_embeddings

    # get all intermediate features and then do bilinear interpolation
    # return features in the layer_idx list
    def forward_unet_features(self, z, t, encoder_hidden_states, h_feature=None, layer_idx=[0], interp_res_h=256, interp_res_w=256):
        unet_output, all_intermediate_features,  copy_downblock = self.unet(
            z,
            t,
            h_sample = h_feature,
            copy = True,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True
            )

        all_return_features = []

        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode='bilinear')
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)

        h_feature = all_intermediate_features[0]
        # h_feature = copy_downblock[2]

        # h_feature = F.interpolate(h_feature, (interp_res_h, interp_res_w), mode='bilinear')

        return unet_output, return_features, h_feature

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        prompt_embeds=None, # whether text embedding is directly provided.
        h_feature=None,
        batch_size=2,
        end_step=None,
        height=512,
        width=512,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        return_intermediates=False,
        gen_img=False,
        **kwds):
        DEVICE = self.device

        if prompt_embeds is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size

            # text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        else:
            batch_size = prompt_embeds.shape[0]
            text_embeddings = prompt_embeds
        print("input text embeddings :", text_embeddings.shape)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.vae.dtype)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]

        if gen_img:
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue

                if guidance_scale > 1.:
                    model_inputs = torch.cat([latents] * 2)
                else:
                    model_inputs = latents

                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
                    # predict the noise
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                if guidance_scale > 1.0:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                # compute the previous noise sample x_t -> x_t-1
                # YUJUN: right now, the only difference between step here and step in scheduler
                # is that scheduler version would clamp pred_x0 between [-1,1]
                # don't know if that's gonna have huge impact
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents_list.append(latents)

        else:
            h_feature = torch.cat([h_feature] * 2)

            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):

                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue


                if guidance_scale > 1.:
                    model_inputs = torch.cat([latents] * 2)
                    h_feature_inputs = torch.cat([h_feature] * 2)
                else:
                    model_inputs = latents
                    # h_feature_inputs = h_feature
                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
                # predict the noise
                if guidance_scale > 1:
                    if i < 50-end_step:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,
                                               h_sample=h_feature_inputs)
                    else:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                else:
                    if i < 50-end_step:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings, h_sample=h_feature)
                        print("i+: ", i)
                    else:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
                        print("i: ", i)
                if guidance_scale > 1.0:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)


                # compute the previous noise sample x_t -> x_t-1
                # YUJUN: right now, the only difference between step here and step in scheduler
                # is that scheduler version would clamp pred_x0 between [-1,1]
                # don't know if that's gonna have huge impact
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents_list.append(latents)
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image
    @torch.no_grad()
    def forward_sampling(
        self,
        prompt,
        prompt_embeds=None, # whether text embedding is directly provided.
        h_feature=None,
        batch_size=2,
        end_step=None,
        height=512,
        width=512,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        return_intermediates=False,
        gen_img=False,
        eta=0.0,
        **kwds):
        DEVICE = self.device
        start_time = time.time()
        if prompt_embeds is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size

            # text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        else:
            batch_size = prompt_embeds.shape[0]
            text_embeddings = prompt_embeds
        print("input text embeddings :", text_embeddings.shape)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.vae.dtype)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]

        if gen_img:
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue

                if guidance_scale > 1.:
                    model_inputs = torch.cat([latents] * 2)
                else:
                    model_inputs = latents

                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
                    # predict the noise
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,)

                if guidance_scale > 1.0:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                # compute the previous noise sample x_t -> x_t-1
                # YUJUN: right now, the only difference between step here and step in scheduler
                # is that scheduler version would clamp pred_x0 between [-1,1]
                # don't know if that's gonna have huge impact
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False,eta=eta)[0]
                latents_list.append(latents)
        else:
            # h_feature = torch.cat([h_feature] * 2)

            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):

                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue


                if guidance_scale > 1.:
                    model_inputs = torch.cat([latents] * 2)
                    h_feature_inputs = torch.cat([h_feature] * 2)
                else:
                    model_inputs = latents
                    # h_feature_inputs = h_feature
                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
                # predict the noise
                if guidance_scale > 1:
                    if i < 50-end_step:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,
                                               h_sample=h_feature_inputs)
                    else:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                else:
                    if i < 50-end_step:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings, h_sample=h_feature)
                        print("i+: ", i)
                    else:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
                        print("i: ", i)
                if guidance_scale > 1.0:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)


                # compute the previous noise sample x_t -> x_t-1
                # YUJUN: right now, the only difference between step here and step in scheduler
                # is that scheduler version would clamp pred_x0 between [-1,1]
                # don't know if that's gonna have huge impact
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False,eta=eta)[0]
                latents_list.append(latents)
        image = self.latent2image(latents, return_type="pt")
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算所花费的时间
        print(f"sampling time: {elapsed_time:.2f} seconds")
        if return_intermediates:
            return image, latents_list
        return image

    def forward_sampling_BG(
            self,
            prompt,
            prompt_embeds=None,  # whether text embedding is directly provided.
            refer_latents=None,
            batch_size=1,
            end_step=None,
            height=512,
            width=512,
            num_inference_steps=50,
            num_actual_inference_steps=None,
            guidance_scale=7.5,
            latents=None,
            unconditioning=None,
            neg_prompt=None,
            return_intermediates=False,
            eta=0.0,
            foreground_mask=None,
            obj_mask=None,
            local_var_reg=None,
            blending = True,
            feature_injection_allowed=True,
            feature_injection_timpstep_range=(900, 600),
            use_mtsa=True,
            **kwds):
        DEVICE = self.device
        assert guidance_scale > 1.0,'USING THIS MODULE CFG Must > 1.0'
        self.controller.use_cfg = True
        self.controller.share_attn = use_mtsa #allow SDSA
        self.controller.local_edit = True
        if prompt_embeds is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size

            # text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        else:
            batch_size = prompt_embeds.shape[0]
            text_embeddings = prompt_embeds
        print("input text embeddings :", text_embeddings.shape)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.vae.dtype)

        # unconditional embedding for classifier free guidance
        # if guidance_scale > 1.:
        if neg_prompt:
            uc_text = neg_prompt
        else:
            uc_text = ""
        unconditional_input = self.tokenizer(
            [uc_text] * batch_size,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]


        #original sample
        #TODO :EACH STEP :need h feature as input ,next step need new h feature from new t and new init latent
        assert  foreground_mask is not None,'FOR BG PRESERVATION foreground_mask should not be None'
        start_step = num_inference_steps - num_actual_inference_steps
        h_feature = None
        self.h_feature_cfg = True
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue
            timestep = t.detach().item()
            if timestep > feature_injection_timpstep_range[0] or timestep < feature_injection_timpstep_range[1]:
                self.controller.set_FI_forbid()
            else:
                if feature_injection_allowed:
                    print(f"Feature Injection is allowed at timestep={timestep}")
                    self.controller.set_FI_allow()
                else:
                    self.controller.set_FI_forbid()

            #TODO: BG preservation h feature
            # if i < 50 - end_step:
            #     self.controller.log_mask = False
            #     h_feature = self.prepare_h_feature(latents[0,None], t, prompt, BG_preservation=False,
            #                                        foreground_mask=foreground_mask, lr=0.1, lam=1, eta=1.0,
            #                                        refer_latent=refer_latents[i - start_step + 1],
            #                                        h_feature_input=h_feature,)
            #[edit,ref]
            ref_latent = refer_latents[i - start_step + 1][1]
            latents[1] = ref_latent
            if i<50 - end_step:
                self.controller.share_attn = use_mtsa  # allow SDSA
            else:
                self.controller.share_attn = False
                # h_feature = torch.cat([h_feature] * 2)
            # if guidance_scale > 1.:
            with torch.no_grad():

                model_inputs = torch.cat([latents] * 2)
                # h_feature_inputs = torch.cat([h_feature] * 2)
                h_feature_inputs = h_feature
                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
                # predict the noise
                # if guidance_scale > 1:
                self.controller.log_mask = False
                if i < 50 - end_step:
                    noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,
                                           h_sample=h_feature_inputs)
                else:
                    noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                if not blending:
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                else:
                    local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * obj_mask
                    noise_pred = noise_pred_uncon + local_text_guidance
                # compute the previous noise sample x_t -> x_t-1
                # YUJUN: right now, the only difference between step here and step in scheduler
                # is that scheduler version would clamp pred_x0 between [-1,1]
                # don't know if that's gonna have huge impact
                # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, eta=eta)[0]
                # full_mask = torch.ones_like(obj_mask)
                latents = self.ctrl_step(noise_pred, t,  latents,local_var_reg, eta=eta)[0]
                latents_list.append(latents)
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image

    @torch.no_grad()
    def Expansion_invert(
            self,
            image: torch.Tensor,
            assist_prompt,
            num_inference_steps=50,
            num_actual_inference_steps=None,
            **kwds):

        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = self.device
        batch_size = image.shape[0]
        assert batch_size==1,">1 bs not implemented yet"
        # if isinstance(assist_prompt, list):
        #     if batch_size == 1:
        image = image.expand(len(assist_prompt), -1, -1, -1)
        prompt = assist_prompt

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        # print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)

        # max_length = text_input.input_ids.shape[-1]
        unconditional_input = self.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
        self.controller.use_cfg = True
        self.controller.bidirectional_loc = True
        # interative sampling

        self.scheduler.set_timesteps(num_inference_steps)
        # DIFT_STEP = int(261/1000*num_inference_steps)
        # t_dift = self.scheduler.timesteps[num_inference_steps-DIFT_STEP-1]
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if num_actual_inference_steps is not None and i > num_actual_inference_steps:
                continue
            elif i==len(self.scheduler.timesteps)-1:
                self.controller.last_step = True

            model_inputs = torch.cat([latents] * 2)
            # t= t_dift
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncon
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
        return latents
    @torch.no_grad()
    def invert(
            self,
            image: torch.Tensor,
            prompt,
            num_inference_steps=50,
            num_actual_inference_steps=None,
            guidance_scale=7.5,
            eta=0.0,
            return_intermediates=False,
            **kwds):

        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = self.device
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            self.controller.use_cfg = True
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
        print(f'leng latent inv {len(latents_list)}')
        print(f'shape latent:{latents.shape}')
        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents
    def dilate_mask(self,mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def erode_mask(self,mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.erode(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def get_attention_scores(self, query, key, attention_mask=None,use_softmax=True):
        dtype = query.dtype

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if not use_softmax:
            return attention_scores,dtype

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_continuous_transformed_mask(self, mask, x_shift, y_shift, resize_scale=1.0, rotation_angle=0,
                                            flip_horizontal=False, flip_vertical=False, split_stage_num=10):

        transformed_mask_list = []
        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask > 128

        # Continuous editing is not designed for flip operation,
        # since it is simple, inpaining can solve this promblem
        # regardless of this for now
        ## 检查是否需要水平翻转
        # if flip_horizontal:
        #     transformed_mask = cv2.flip(transformed_mask, 1)
        #
        # # 检查是否需要垂直翻转
        # if flip_vertical:
        #     transformed_mask = cv2.flip(transformed_mask, 0)

        # 获取图像尺寸
        height, width = mask.shape[:2]
        y_indices, x_indices = np.where(mask.astype(bool))
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = np.min(y_indices), np.max(y_indices)
            left, right = np.min(x_indices), np.max(x_indices)
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2

        rotation_angle_step = rotation_angle / split_stage_num
        resize_scale_step = resize_scale ** (1 / split_stage_num)
        dx = x_shift / split_stage_num
        dy = y_shift / split_stage_num

        transformed_mask_temp = mask
        transformed_mask_list.append(transformed_mask_temp)
        for step in range(split_stage_num):
            # single step matrix for current center
            transformation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle_step,
                                                            resize_scale_step)
            transformation_matrix[0, 2] += dx
            transformation_matrix[1, 2] += dy
            # print(f'step:{step} matrix:{transformation_matrix}')
            transformed_mask_temp = cv2.warpAffine(transformed_mask_temp, transformation_matrix, (width, height),
                                                   flags=cv2.INTER_NEAREST)
            transformed_mask_list.append(transformed_mask_temp)
            mask_center_x += dx
            mask_center_y += dy
        # output_dir = 'temp_dir_vis'
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # for idx, mask in enumerate(transformed_mask_list):
        #     plt.figure()
        #     plt.imshow(mask, cmap='gray')
        #     plt.title(f'Step {idx}')
        #     plt.axis('off')
        #     plt.savefig(os.path.join(output_dir, f'mask_step_{idx}.png'))
        #     plt.close()
        return transformed_mask_list

    def temp_view_img(self,image: Image.Image, title: str = None) -> None:
        # PIL -> ndarray OR ndarray->PIL->ndarray
        if not isinstance(image, Image.Image):#ndarray
            # image_array = Image.fromarray(image).convert('RGB')
            image_array = image
        else:#PIL
            if image.mode != 'RGB':
                image.convert('RGB')
            image_array = np.array(image)


        plt.imshow(image_array)
        if title is not None:
            plt.title(title)
        plt.axis('off')  # Hide the axis
        plt.show()
    def replace_with_SV3D_targets_inpainting(self,ori_img,trans_img,ori_mask,trans_mask,target_mask,mode,inpainter,inp_prompt=None,
                                             ):

        #inpaint & replace
        if isinstance(ori_img, Image.Image):
            ori_img = np.array(ori_img)
        if isinstance(trans_img, Image.Image):
            trans_img = np.array(trans_img)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        if ori_mask.ndim == 3 and ori_mask.shape[2] == 3:
            ori_mask = cv2.cvtColor(ori_mask, cv2.COLOR_BGR2GRAY)
            # mask = mask > 128
        if trans_mask.ndim == 3 and trans_mask.shape[2] == 3:
            trans_mask = cv2.cvtColor(trans_mask, cv2.COLOR_BGR2GRAY)
            # source_mask =source_mask > 128



        ori_mask = (ori_mask > 0).astype(bool)
        ori_image_back_ground = np.where(ori_mask[:, :, None], 0, ori_img).astype(np.uint8)
        image_with_hole = ori_image_back_ground
        coarse_repaired = np.array(inpainter(Image.fromarray(ori_image_back_ground), Image.fromarray(
            ori_mask.astype(np.uint8) * 255)))  # lama inpainting filling the black regions
        if mode != 1:
            inpaint_mask = Image.fromarray(ori_mask.astype(np.uint8) * 255)
            sd_to_inpaint_img = Image.fromarray(coarse_repaired)
            # print(f'SD inpainting Processing:')
            semantic_repaired = \
            self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask).images[0]
            semantic_repaired = np.array(semantic_repaired)
        else:
            semantic_repaired = coarse_repaired

        if semantic_repaired.shape != image_with_hole.shape:
            print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
            h, w = image_with_hole.shape[:2]
            semantic_repaired = cv2.resize(semantic_repaired, (w, h), interpolation=cv2.INTER_LANCZOS4)


        x, y, w, h = cv2.boundingRect(target_mask)
        cent_h,cent_w = y+h//2,x+w//2
        x, y, w, h = cv2.boundingRect(trans_mask)
        repl_trans_mask = np.zeros_like(target_mask)
        repl_trans_img = np.zeros_like(ori_img)
        repl_trans_mask[
        cent_h - h // 2: cent_h - h // 2 + h,
        cent_w - w // 2: cent_w - w // 2 + w,
        ] = trans_mask[y : y + h, x : x + w]
        repl_trans_img[
        cent_h - h // 2: cent_h - h // 2 + h,
        cent_w - w // 2: cent_w - w // 2 + w,
        ] = trans_img[y : y + h, x : x + w]
        repl_trans_mask =(repl_trans_mask>0).astype(bool)
        final_image = np.where(repl_trans_mask[:, :, None], repl_trans_img, semantic_repaired)

        return final_image, image_with_hole,repl_trans_mask.astype(np.uint8)*255

    def move_and_inpaint_with_expansion_mask_3D(self, image, mask, depth_map, transforms, FX, FY, object_only=True,
                                                inpainter=None, mode=None,
                                                dilate_kernel_size=15, inp_prompt=None, target_mask=None,
                                                splatting_radius=0.015,
                                                splatting_tau=0.0, splatting_points_per_pixel=30):

        if isinstance(image, Image.Image):
            image = np.array(image)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask = mask > 128

        if target_mask.ndim == 3 and target_mask.shape[2] == 3:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
            # source_mask =source_mask > 128

        mask = mask.astype(bool)
        target_mask = target_mask.astype(bool)

        transformed_image, transformed_mask = IntegratedP3DTransRasterBlendingFull(image, depth_map, transforms, FX, FY,
                                                                                   target_mask, object_only,
                                                                                   splatting_radius=splatting_radius,
                                                                                   splatting_tau=splatting_tau,
                                                                                   splatting_points_per_pixel=splatting_points_per_pixel,
                                                                                   return_mask=True,
                                                                                   device=self.device)

        # mask bool
        # MORPH_OPEN transformed target mask to suppress noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        transformed_mask = cv2.morphologyEx(transformed_mask, cv2.MORPH_OPEN, kernel)
        transformed_mask = (transformed_mask > 128).astype(bool)
        # repair_mask = (mask & ~transformed_mask)
        # ori_image_back_ground = np.where(mask[:, :, None], 0, image).astype(np.uint8)
        # new_image = np.where(transformed_mask[:, :, None], transformed_image,
        #                      ori_image_back_ground)  # with repair area to be black
        # image_with_hole = new_image
        # coarse_repaired = inpainter(Image.fromarray(new_image), Image.fromarray(
        #     repair_mask.astype(np.uint8) * 255))  # lama inpainting filling the black regions
        #
        # to_inpaint_img = coarse_repaired
        #
        # if mode == 1:
        #     semantic_repaired = to_inpaint_img
        # elif mode == 2:
        #     inpaint_mask = Image.fromarray(repair_mask.astype(np.uint8) * 255)
        #     print(f'SD inpainting Processing:')
        #     semantic_repaired = \
        #     self.sd_inpainter(prompt=inp_prompt, image=to_inpaint_img, mask_image=inpaint_mask).images[0]
        #
        # if semantic_repaired.size != to_inpaint_img.size:
        #     print(f'inpainted image {semantic_repaired.size} -> original size {to_inpaint_img.size}')
        #     semantic_repaired = semantic_repaired.resize(to_inpaint_img.size)
        # # mask retain in region only repairing
        # retain_mask = ~repair_mask
        # final_image = np.where(retain_mask[:, :, None], semantic_repaired, semantic_repaired)
        # final_image = np.where(retain_mask[:, :, None], coarse_repaired, semantic_repaired)
        ori_image_back_ground = np.where(mask[:, :, None], 0, image).astype(np.uint8)
        image_with_hole = ori_image_back_ground
        coarse_repaired = np.array(inpainter(Image.fromarray(ori_image_back_ground), Image.fromarray(
            mask.astype(np.uint8) * 255)))  # lama inpainting filling the black regions
        if mode != 1:
            inpaint_mask = Image.fromarray(mask.astype(np.uint8) * 255)
            sd_to_inpaint_img = Image.fromarray(coarse_repaired)
            # print(f'SD inpainting Processing:')
            semantic_repaired = \
                self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask).images[0]
            semantic_repaired = np.array(semantic_repaired)
        else:
            semantic_repaired = coarse_repaired

        if semantic_repaired.shape != image_with_hole.shape:
            print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
            h, w = image_with_hole.shape[:2]
            semantic_repaired = cv2.resize(semantic_repaired, (w, h), interpolation=cv2.INTER_LANCZOS4)

        final_image = np.where(transformed_mask[:, :, None], transformed_image, semantic_repaired)

        return final_image, image_with_hole, transformed_mask

    def get_depth_V2(self, image_raw,base_depth=0.1,min_z=10,max_z=255):
        # h, w = image_raw.shape[:2]
        #raw_img = cv2.imread('your/image/path') #RGB img is enough ndarray
        depth= self.depth_anything.infer_image(image_raw) #ndarray H W back
        # depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        # GeoDiffuser Processor
        # depth = depth.max() - depth  # Negating depth as relative depth estimators assign high values to close objects. You can also try 1/depth (inverse depth, but we found this to work better prima facie)
        # depth = depth + depth.max() * base_depth  # This helps in reducing depth smearing where translate_factor is between 0 to 1.
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255  # Normalizes from 0-1.
        depth = depth.max() - depth  # Negating depth as relative depth estimators assign high values to close objects. You can also try 1/depth (inverse depth, but we found this to work better prima facie)
        depth = depth + depth.max() * base_depth  # This helps in reducing depth smearing where translate_factor is between 0 to 1.
        # depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalizes from 0-1.
        # modified by clawer
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * (max_z - min_z) + min_z
        return depth

    def Magic_Editing_Baseline_3D(self, original_image, mask, prompt, INP_prompt,
                                seed, guidance_scale, num_step, max_resolution, mode, dilate_kernel_size, start_step,
                                tx, ty, tz, rx, ry, rz, sx, sy, sz, mask_ref=None, eta=0, use_mask_expansion=True,
                                standard_drawing=True, contrast_beta=1.67, strong_inpaint=True,
                                cross_enhance=False,
                                mask_threshold=0.1, mask_threshold_target=0.1, blending_alpha=0.5,
                                splatting_radius=0.015, splatting_tau=0.0, splatting_points_per_pixel=30,
                                focal_length=1080,end_step=10,feature_injection=True,FI_range=(900,680),sim_thr=0.5,DIFT_LAYER_IDX = [0,1,2,3],use_sdsa=True,
                                ):
        seed_everything(seed)
        transforms = [tx, ty, tz, rx, ry, rz, sx, sy, sz]
        print(f'trans {transforms}')
        # select and resize
        print(f'mask shape: {mask.shape}')
        cv2.imwrite("mask.png",mask)
        #remove background
        mask_BG = (mask == 0).astype(bool)
        image_fg = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_fg[mask_BG]=255
        cv2.imwrite('FG_IMG.png',image_fg)


        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution * max_resolution)

        # depth = self.get_depth(img)
        depth = self.get_depth_V2(img, base_depth=0.1, max_z=255, min_z=20)
        depth_plot = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_plot = np.repeat(depth_plot[..., np.newaxis], 3, axis=-1)
        depth_plot = Image.fromarray(depth_plot)
        depth_map = depth

        # print(img.shape)  # 768
        # print(input_scale)
        FINAL_WIDTH = img.shape[1]
        FINAL_HEIGHT = img.shape[0]
        # FX = FINAL_WIDTH * 0.6
        # FY = FINAL_HEIGHT * 0.6
        F = focal_length
        FX = F
        FY = F

        if standard_drawing:  # box only input
            mask_to_use = mask
            if input_scale != 1:
                mask_to_use, _ = resize_numpy_image(mask_to_use, max_resolution * max_resolution)
            if strong_inpaint:# whether retain , sd inpaint needed
                target_mask = mask_to_use
            mask_to_use = self.dilate_mask(mask_to_use, dilate_kernel_size) * 255  # dilate for better expansion mask
            return_target = False  # which designed for casual draw input target/expanded mask

        else:
            mask_to_use = mask_ref
            if len(mask_ref) > 1:
                mask_to_use, _ = resize_numpy_image(mask_to_use, max_resolution * max_resolution)
                return_target = False
                if strong_inpaint:
                    if cross_enhance:  # use box mask to retain
                        target_mask, _ = resize_numpy_image(mask, max_resolution * max_resolution)
                        target_mask = np.array(self.dilate_mask(target_mask, dilate_kernel_size//2) * 255)
                    else:  # use strict expansion mask to retain
                        return_target = True

        # mask expansion
        if return_target:
            self.controller.contrast_beta = contrast_beta * 100
            target_mask, _ = self.DDIM_inversion_func(img=img, mask=mask_to_use, prompt="",
                                                      guidance_scale=1, num_step=10,
                                                      start_step=2,
                                                      roi_expansion=True,
                                                      mask_threshold=mask_threshold,
                                                      post_process='hard', )
        if use_mask_expansion:
            self.controller.contrast_beta = contrast_beta
            expand_mask, _ = self.DDIM_inversion_func(img=img, mask=mask_to_use, prompt="",
                                                      guidance_scale=1, num_step=10,
                                                      start_step=2,
                                                      roi_expansion=True,
                                                      mask_threshold=mask_threshold,
                                                      post_process='hard', )
        else:
            expand_mask = mask_to_use

        img_preprocess, inpaint_mask_vis, shifted_mask  = self.move_and_inpaint_with_expansion_mask_3D(img,expand_mask,depth_map,
                                                                                                    transforms,FX, FY,True, self.inpainter,
                                                                                                    mode, dilate_kernel_size, INP_prompt,
                                                                                                    target_mask, splatting_radius,splatting_tau,
                                                                                                    splatting_points_per_pixel)
        mask_to_use = shifted_mask * 255
        # mask_to_use = self.dilate_mask(shifted_mask, dilate_kernel_size) * 255  # dilate for better expansion mask
        ori_img = img
        img = img_preprocess
        if isinstance(img, np.ndarray):
            img_preprocess = Image.fromarray(img_preprocess)
        self.controller.contrast_beta = contrast_beta
        expand_shift_mask, inverted_latent = self.DDIM_inversion_func(img=img, mask=mask_to_use, prompt="",
                                                                      guidance_scale=1, num_step=num_step, start_step=start_step,
                                                                      roi_expansion=True,
                                                                      mask_threshold=mask_threshold_target,
                                                                      post_process='hard',
                                                                      use_mask_expansion=False, ref_img=ori_img)  # ndarray mask


        edit_gen_image,refer_gen_image, target_mask, = self.Details_Preserving_regeneration(img, inverted_latent, prompt, expand_shift_mask, target_mask, num_steps=num_step, start_step=start_step,
                                                                                            end_step=end_step, guidance_scale=guidance_scale, eta=eta, roi_expansion=True,
                                                                                            mask_threshold=mask_threshold_target, post_process='hard', feature_injection=feature_injection,
                                                                                            FI_range=FI_range, sim_thr=sim_thr, dilate_kernel_size=dilate_kernel_size, DIFT_LAYER_IDX=DIFT_LAYER_IDX,
                                                                                            use_sdsa=use_sdsa, ref_img=ori_img)
        # source_mask = Image.fromarray(source_mask)
        inpaint_mask_vis = Image.fromarray(inpaint_mask_vis)
        return  [edit_gen_image],[refer_gen_image],[img_preprocess],[inpaint_mask_vis],[target_mask],[depth_plot]
    def get_mask_from_rembg(self,trans_img,size=None,need_mask=True):
        if isinstance(trans_img,np.ndarray):
            trans_img = cv2.cvtColor(trans_img,cv2.COLOR_RGB2BGR)
            trans_img = Image.fromarray(trans_img) #PIL img
        if need_mask:
            if size is not None:
                trans_img.thumbnail(size, Image.Resampling.LANCZOS)
                print(f'resized_img shape:{trans_img.size}')

            return_img = np.array(trans_img.copy())
            return_img = cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB)
            trans_img = remove(trans_img.convert("RGBA"), alpha_matting=True)
            image_arr = np.array(trans_img)
            in_w, in_h = image_arr.shape[:2]
            _, trans_mask = cv2.threshold(
                np.array(trans_img.split()[-1]), 128, 255, cv2.THRESH_BINARY
            )
            return trans_mask,in_w,return_img
        else:#resize pipe
            assert size is not None,"for resize pipe,size should be given"
            trans_img.thumbnail(size, Image.Resampling.LANCZOS)
            return_img = np.array(trans_img.copy())
            return_img = cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB)
            in_w, in_h = return_img.shape[:2]
            return return_img, in_w
    def Magic_Editing_Baseline_SV3D(self, original_image, transformed_image, prompt, INP_prompt,
                                  seed, guidance_scale, num_step, max_resolution, mode, dilate_kernel_size, start_step,
                                  eta=0, use_mask_expansion=True,contrast_beta=1.67, mask_threshold=0.1, mask_threshold_target=0.1, end_step=10,
                                feature_injection=True, FI_range=(900, 680),sim_thr=0.5, DIFT_LAYER_IDX=[0, 1, 2, 3], use_sdsa=True,select_mask=None,
                                  ):
        seed_everything(seed)

        img = original_image
        trans_img = transformed_image
        if select_mask is None:
            print(f'generating original image mask')
            mask,in_w,img = self.get_mask_from_rembg(img,size=[max_resolution,max_resolution])
            trans_img = cv2.resize(trans_img,(in_w,in_w))
            print(f'generating transformed image mask')
            trans_mask,_,trans_img = self.get_mask_from_rembg(trans_img)
        else:#resize + mask resize
            print(f'input selected mask : Yes')
            img, in_w = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution],need_mask=False)
            w,h = img.shape[:2]
            mask = cv2.resize(select_mask,(h,w))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            trans_img = cv2.resize(trans_img, (in_w, in_w))
            print(f'generating transformed image mask')
            trans_mask, _, trans_img = self.get_mask_from_rembg(trans_img)


        target_mask = mask
        mask_to_use = self.dilate_mask(mask, dilate_kernel_size)  # dilate for better expansion mask

        # mask expansion
        if use_mask_expansion:
            self.controller.contrast_beta = contrast_beta #numpy input
            expand_mask, _ = self.DDIM_inversion_func(img=img, mask=mask_to_use, prompt="",
                                                      guidance_scale=1, num_step=10,
                                                      start_step=2,
                                                      roi_expansion=True,
                                                      mask_threshold=mask_threshold,
                                                      post_process='hard', )
        else:
            expand_mask = mask_to_use

        img_preprocess, inpaint_mask_vis,shifted_mask = self.replace_with_SV3D_targets_inpainting(img, trans_img,expand_mask,trans_mask,target_mask,mode,
                                                                                                      self.inpainter,INP_prompt,)
        mask_to_use = shifted_mask
        # mask_to_use = self.dilate_mask(shifted_mask, dilate_kernel_size) * 255  # dilate for better expansion mask
        ori_img = img
        img = img_preprocess
        if isinstance(img, np.ndarray):
            img_preprocess = Image.fromarray(img_preprocess)
        self.controller.contrast_beta = contrast_beta
        expand_shift_mask, inverted_latent = self.DDIM_inversion_func(img=img, mask=mask_to_use,
                                                                      prompt="",
                                                                      guidance_scale=1,
                                                                      num_step=num_step,
                                                                      start_step=start_step,
                                                                      roi_expansion=True,
                                                                      mask_threshold=mask_threshold_target,
                                                                      post_process='hard',
                                                                      use_mask_expansion=False,
                                                                      ref_img=ori_img)  # ndarray mask

        edit_gen_image, refer_gen_image, target_mask, = self.Details_Preserving_regeneration(img, inverted_latent, prompt,
                                                                                             expand_shift_mask, target_mask,
                                                                                             num_steps=num_step, start_step=start_step,
                                                                                             end_step=end_step,
                                                                                             guidance_scale=guidance_scale, eta=eta,
                                                                                             roi_expansion=True,
                                                                                             mask_threshold=mask_threshold_target,
                                                                                             post_process='hard',
                                                                                             feature_injection=feature_injection,
                                                                                             FI_range=FI_range, sim_thr=sim_thr,
                                                                                             dilate_kernel_size=dilate_kernel_size,
                                                                                             DIFT_LAYER_IDX=DIFT_LAYER_IDX,
                                                                                             use_sdsa=use_sdsa, ref_img=ori_img)
        # source_mask = Image.fromarray(source_mask)
        inpaint_mask_vis = Image.fromarray(inpaint_mask_vis)
        return [edit_gen_image], [refer_gen_image], [img_preprocess], [inpaint_mask_vis], [target_mask]
    def get_matching_score(self,candidate_sim,pos_len):
        positive_sim = candidate_sim[:, :pos_len].sum(dim=-1)
        negative_sim = candidate_sim[:, pos_len:].sum(dim=-1)
        matching_score = positive_sim - negative_sim * 2
        return matching_score
    @torch.no_grad()
    def sd_inpaint_results_filter(self,img_lists,mask,class_text,inp_prompt):
        #[ndarray,ndarray,.....]
        neg_text_list = [class_text] + ['object','person','texts',]
        pos_text_list = ['empty scene','background']
        pos_len = len(pos_text_list)
        # 通过mask区域的boundary box裁剪图像
        preprocessed_imgs = self.crop_image_with_mask(img_lists, mask)
        # preprocessed_imgs = self.pre_process_with_mask(img_lists, mask)
        img_lst = np.array(img_lists)
        # image = self.clip_process(cropped_img).unsqueeze(0).to(self.device)
        stack_images = torch.stack([self.clip_process(self.numpy_to_pil(crop_img)[0]).to(self.device) for crop_img in preprocessed_imgs])

        # 对每个词进行编码
        text_tokens_neg = clip.tokenize(neg_text_list).to(self.device)
        text_tokens_pos = clip.tokenize(pos_text_list).to(self.device)
        text_tokens = torch.cat([text_tokens_pos,text_tokens_neg])

        # 计算图像的特征向量
        image_features = self.clip.encode_image(stack_images)
        # Calculate the standard deviation of the image embeddings to measure semantic diversity
        # embeddings_std = torch.std(image_features[1:], dim=0).mean()
        # #uncertainty filter
        # if embeddings_std.item() >0.20:
        #     # print(f'choose lama inpainting results')
        #     # return img_lists[0]

        # 计算每个词的特征向量
        text_features = self.clip.encode_text(text_tokens)
        similarities = torch.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
        before_similarities = similarities[-1,:]#ori img sim
        similarities = similarities[:-1,:]
        # class filter
        valid_idx_cls = similarities.argmax(dim=1)<pos_len
        #discrepency filter
        pre_matching_score = self.get_matching_score(before_similarities[None,],pos_len)
        matching_score = self.get_matching_score(similarities,pos_len)
        valid_idx_dis = matching_score > pre_matching_score
        valid_idx = valid_idx_cls & valid_idx_dis
        if valid_idx.sum()==0:
            print(f'choose lama inpainting results')
            return img_lists[0]
        # valid_idx = valid_idx_cls & valid_idx_dis
        valid_index = np.array([idx for idx, i in enumerate(valid_idx) if i])
        candidate_img = img_lst[valid_index]
        matching_score_valid = matching_score[valid_idx]
        final_match_indices =  torch.argmax(matching_score_valid, dim=0).item()
        if valid_idx[0] and final_match_indices == 0:#default lama always good, but not real enough
            print(f'choose lama inpainting results')
        else:
            print(f'choose sd inpainting results')

        return candidate_img[int(final_match_indices)]


    def move_and_inpaint(self, ori_img, exp_mask, dx, dy, inpainter=None, mode=None,
                         inp_prompt=None,inp_prompt_negative=None,obj_text=None, resize_scale=1.0, rotation_angle=0, source_mask=None,
                         flip_horizontal=False, flip_vertical=False):

        # inpaint & replace
        if isinstance(ori_img, Image.Image):
            ori_img = np.array(ori_img)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        if exp_mask.ndim == 3 and exp_mask.shape[2] == 3:
            exp_mask = cv2.cvtColor(exp_mask, cv2.COLOR_BGR2GRAY)
        if source_mask.ndim == 3 and source_mask.shape[2] == 3:
            source_mask = cv2.cvtColor(source_mask, cv2.COLOR_BGR2GRAY)


        #prepare background
        # box mask do not influence sd inpaint
        # exp_mask = self.box_mask(exp_mask)
        ori_exp_mask = exp_mask
        exp_mask = (exp_mask > 0).astype(bool)
        ori_image_back_ground = np.where(exp_mask[:, :, None], 0, ori_img)
        image_with_hole = ori_image_back_ground
        # print(f'ori_shape:{ori_image_back_ground.shape}')
        # coarse_repaired = np.array(inpainter(Image.fromarray(ori_image_back_ground), Image.fromarray(
        #     exp_mask.astype(np.uint8) * 255)))  # lama inpainting filling the black regions

        coarse_repaired = inpainter(ori_image_back_ground, exp_mask.astype(np.uint8) * 255,)

        if mode != 1:
            inpaint_mask = Image.fromarray(exp_mask.astype(np.uint8) * 255)
            sd_to_inpaint_img = Image.fromarray(coarse_repaired)
            # print(f'SD inpainting Processing:')
            semantic_repaired = \
            self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask,guidance_scale=7.5,eta=1.0,
                              num_inference_steps=10,negative_prompt=inp_prompt_negative,num_images_per_prompt=10).images
            #resize
            semantic_repaired_new = []
            h, w = image_with_hole.shape[:2]
            for sd_inp_res in semantic_repaired:
                sd_inp_res = np.array(sd_inp_res)
                if sd_inp_res.shape != image_with_hole.shape:
                    # print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
                    sd_inp_res = cv2.resize(sd_inp_res, (w, h), interpolation=cv2.INTER_LANCZOS4)
                semantic_repaired_new.append(sd_inp_res)
            #Filter
            semantic_repaired_new.insert(0, coarse_repaired)
            semantic_repaired_new.append(ori_img)
            semantic_repaired = self.sd_inpaint_results_filter(semantic_repaired_new,ori_exp_mask,obj_text,inp_prompt)
        else:
            semantic_repaired = coarse_repaired


        # Prepare foreground
        height, width = ori_img.shape[:2]
        y_indices, x_indices = np.where(source_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = np.min(y_indices), np.max(y_indices)
            left, right = np.min(x_indices), np.max(x_indices)
            # mask_roi = mask[top:bottom + 1, left:right + 1]
            # image_roi = image[top:bottom + 1, left:right + 1]
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2


        rotation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle, resize_scale)
        rotation_matrix[0, 2] += dx
        rotation_matrix[1, 2] += dy

        transformed_image = cv2.warpAffine(ori_img, rotation_matrix, (width, height))
        transformed_mask_exp = cv2.warpAffine(exp_mask.astype(np.uint8), rotation_matrix, (width, height),
                                              flags=cv2.INTER_NEAREST).astype(bool)
        transformed_mask = cv2.warpAffine(source_mask.astype(np.uint8), rotation_matrix, (width, height),
                                          flags=cv2.INTER_NEAREST).astype(bool)


        # 检查是否需要水平翻转
        if flip_horizontal:
            transformed_image = cv2.flip(transformed_image, 1)
            transformed_mask = cv2.flip(transformed_mask, 1)
            transformed_mask_exp = cv2.flip(transformed_mask_exp, 1)


        # 检查是否需要垂直翻转
        if flip_vertical:
            transformed_image = cv2.flip(transformed_image, 0)
            transformed_mask = cv2.flip(transformed_mask, 0)
            transformed_mask_exp = cv2.flip(transformed_mask_exp, 0)


        ddpm_region = transformed_mask_exp *( 1-transformed_mask)
        final_image = np.where(transformed_mask_exp[:, :, None], transformed_image, semantic_repaired) #move with expansion pixels but inpaint
        return final_image, image_with_hole,semantic_repaired,transformed_mask,ddpm_region


    def Magic_Editing_Baseline_2D(self, original_image,prompt, INP_prompt,selected_points,
                                  seed, guidance_scale, num_step, max_resolution, mode, start_step, resize_scale,
                                  rotation_angle, flip_horizontal,flip_vertical,eta=0, use_mask_expansion=True,expansion_step=4,contrast_beta=1.67, end_step=10,
                                feature_injection=True, FI_range=(900, 680),sim_thr=0.5, DIFT_LAYER_IDX=[0, 1, 2, 3], use_mtsa=True,select_mask=None,assist_prompt="",lora_path="",
                                  ):
        if isinstance(assist_prompt,str):
            assist_prompt = [item.strip() for item in assist_prompt.split(',')]
        # set lora
        if lora_path == "":
            print("applying default parameters")
            self.unet.set_default_attn_processor()
        else:
            print("applying lora: " + lora_path)
            self.unet.load_attn_procs(lora_path)
        seed_everything(seed)
        img = original_image

        cv2.imwrite("mask.png",select_mask)

        if select_mask is None:
            print(f'generating original image mask')
            orw,orh = img.shape[:2]
            mask, in_w, img = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution])
            input_scale = in_w / orw
            print(f'input scale:{input_scale}')
        else:  # resize + mask resize
            print(f'input selected mask : Yes')
            orw, orh = img.shape[:2]
            img, in_w = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution], need_mask=False)
            input_scale = in_w / orw
            print(f'input scale:{input_scale}')
            w, h = img.shape[:2]
            mask = cv2.resize(select_mask, (h, w))
            if mask.ndim == 3:
                mask = mask[:,:,0]
        assert mask.shape[:2] == img.shape[:2],f'mask shape{mask.shape} while img shape{img.shape}'
        print(selected_points)
        # get move
        x = []
        y = []
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx % 2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        # IF ERROR is Raised , check whether you select start and end moving points
        dx = x_cur[0] - x[0]
        dy = y_cur[0] - y[0]
        dx = int(dx * input_scale)
        dy = int(dy * input_scale)

        source_mask = mask
        mask_to_use = mask

        #Mask guided prompt locating with CLIP Scores
        guidance_text = self.ClipWordMatching(img=img, mask=mask_to_use, full_prompt=prompt)
        print(f'located guidance prompt is "{guidance_text}"')
        # Prompt guided iterative mask expansion for source img
        if use_mask_expansion:
            print(f'mask expansion is allowed')
            self.controller.contrast_beta = contrast_beta  # numpy input
            expand_mask = self.Prompt_guided_mask_expansion_func(img=img, mask=mask_to_use,assist_prompt=assist_prompt,
                                                                num_step=expansion_step, start_step=1,)


        else:
            print(f'mask expansion is Forbiden')
            expand_mask = mask_to_use
        INP_prompt_negative = f'object,{guidance_text},shadow'
        # expand_mask_vis = self.numpy_to_pil(expand_mask)[0]
        img_preprocess, inpaint_mask_vis,inp_results,shifted_mask,ddpm_region = self.move_and_inpaint(img,expand_mask,dx, dy,self.inpainter, mode, INP_prompt,INP_prompt_negative,guidance_text, resize_scale,  rotation_angle,
                                                                                                                source_mask, flip_horizontal, flip_vertical)

        mask_to_use = shifted_mask
        # mask_to_use = self.dilate_mask(shifted_mask, dilate_kernel_size) * 255
        ori_img = img
        img = img_preprocess
        if isinstance(img, np.ndarray):
            img_preprocess = Image.fromarray(img_preprocess)#VIS
        inpaint_mask_vis = Image.fromarray(inpaint_mask_vis)
        inp_results = Image.fromarray(inp_results)
        self.controller.contrast_beta = contrast_beta
        #DDIM INVERSION
        shifted_mask, inverted_latent = self.DDIM_inversion_func(img=img, mask=mask_to_use,
                                                                      prompt="",
                                                                      num_step=num_step,
                                                                      start_step=start_step,
                                                                      ref_img=ori_img)  # ndarray mask

        edit_gen_image, refer_gen_image, = self.Details_Preserving_regeneration(img, inverted_latent,guidance_text,
                                                                                             shifted_mask, source_mask,
                                                                                             num_steps=num_step, start_step=start_step,
                                                                                             end_step=end_step,
                                                                                             guidance_scale=guidance_scale, eta=eta,
                                                                                             feature_injection=feature_injection,
                                                                                             FI_range=FI_range, sim_thr=sim_thr,
                                                                                             DIFT_LAYER_IDX=DIFT_LAYER_IDX,
                                                                                             use_mtsa=use_mtsa, ref_img=ori_img, ddpm_region=ddpm_region)
        if False:
            # prompt guided iterative mask expansion for target img
            print(f'target mask expansion is allowed')
            self.controller.contrast_beta = contrast_beta  # numpy input
            target_mask = self.Prompt_guided_mask_expansion_func(img=edit_gen_image, mask=shifted_mask,
                                                                    assist_prompt=assist_prompt,
                                                                    num_step=expansion_step,
                                                                    start_step=1,)
        else:
            target_mask = shifted_mask
        refer_gen_image = self.numpy_to_pil(refer_gen_image)[0]
        edit_gen_image = self.numpy_to_pil(edit_gen_image)[0]


        return [edit_gen_image], [refer_gen_image], [img_preprocess], [inpaint_mask_vis], [target_mask],[inp_results]
    def normalize_expansion_mask(self,mask,exp_mask,roi_expansion):
        if roi_expansion:
            candidate_mask = torch.ones_like(exp_mask)
            expansion_loc = mask < 125
            average_expansion_masks_of_interested = exp_mask[expansion_loc]
            ma, mi = average_expansion_masks_of_interested.max(), average_expansion_masks_of_interested.min()
            average_expansion_masks_norm = (average_expansion_masks_of_interested - mi) / (ma - mi)
            candidate_mask[expansion_loc] = average_expansion_masks_norm
        else:
            ma, mi = exp_mask.max(), exp_mask.min()
            average_expansion_masks_norm = (exp_mask - mi) / (ma - mi)
            candidate_mask = average_expansion_masks_norm
        return  candidate_mask

    def bfs_distance_transform(self, expand_mask_bool, edge_pixels):
        # 初始化距离映射为无穷大
        distance_map = torch.full_like(expand_mask_bool, float('inf'), dtype=torch.float32)

        # 将边缘像素的距离设为0
        queue = [(x.item(), y.item()) for x, y in torch.nonzero(edge_pixels, as_tuple=False)]
        for x, y in queue:
            distance_map[x, y] = 0

        # 使用BFS计算距离，仅对expand_mask_bool为True的点计算距离
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.pop(0)
            current_distance = distance_map[x, y]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < expand_mask_bool.shape[0] and 0 <= ny < expand_mask_bool.shape[1]:
                    if expand_mask_bool[nx, ny] and distance_map[nx, ny] > current_distance + 1:
                        distance_map[nx, ny] = current_distance + 1
                        queue.append((nx, ny))

        return distance_map

    def generate_dynamic_threshold_function(self,mask_threshold, control_value, control_point, distance_range=(0, 3),
                                            device='cpu'):
        max_thr = 1.0
        control_point = torch.tensor(control_point, dtype=torch.float32, device=device)
        distance_end = torch.tensor(distance_range[1], dtype=torch.float32, device=device)
        value = torch.tensor((control_value - mask_threshold) / (max_thr - mask_threshold), dtype=torch.float32,
                             device=device)

        # Recalculate scale based on control_value and control_point
        scale = torch.log(value) / (control_point - distance_end)

        # Return the dynamic threshold function
        def dynamic_threshold_function(relative_distance):
            relative_distance = torch.tensor(relative_distance, dtype=torch.float32, device=device)
            # dynamic_thr = mask_threshold + (1 - mask_threshold) * torch.exp(relative_distance - 1)
            return mask_threshold + (max_thr - mask_threshold) * torch.exp(relative_distance * scale - 1) / torch.exp(
                distance_end * scale - 1)

        return dynamic_threshold_function

    def plot_bfs_distance_map(self, distance_map, save_path=None,img_name=None,title=None):
        # Ensure the distance_map is on the CPU for plotting
        save_path = os.path.join(save_path,img_name+'.png')
        distance_map = distance_map.cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.imshow(distance_map, cmap='viridis')
        plt.colorbar(label='BFS Distance')
        if title is None:
            plt.title('BFS Distance Map')
        else:
            plt.title(f'{title}')
        if save_path:
            plt.savefig(save_path)
            plt.show()
            plt.close()
        else:
            plt.show()

    import torch.nn.functional as F
    def preprocess_image(self,image,
                         device):
        image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
        image = rearrange(image, "h w c -> 1 c h w")
        image = image.to(device)
        return image
    def get_structuring_element(self, kernel_size):
        # 创建一个正方形的结构元素
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
        # 将kernel移到和mask相同的设备上
        kernel = kernel.to(self.device)
        return kernel
    def opening_operation(self,mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 先腐蚀
        eroded = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        eroded = (eroded == torch.max(eroded)).float()
        # 后膨胀
        opened = F.conv2d(eroded.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        opened = (opened > 0).float()
        return opened

    def erode_operation(self, mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 腐蚀
        eroded = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        eroded = (eroded == torch.max(eroded)).float()

        return eroded
    def dilation_operation(self, mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 膨胀
        opened = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        opened = (opened > 0).float()
        return opened
    def closing_operation(self,mask, kernel_size=5):
        mask=mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 先膨胀
        dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        dilated = (dilated > 0).float()
        # 后腐蚀
        closed = F.conv2d(dilated.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        closed = (closed == torch.max(closed)).float()
        return closed

    def temp_view(self,mask, title='Mask',name=None):
        """
        显示输入的mask图像

        参数:
        mask (torch.Tensor): 要显示的mask图像，类型应为torch.bool或torch.float32
        title (str): 图像标题
        """
        # 确保输入的mask是float类型以便于显示
        if isinstance(mask,np.ndarray):
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
    def get_self_adaptive_init_thr(self,obj_mask,strict_threshold,adp_k):
        valid_region = (obj_mask > 0).sum()
        invalid_region = (obj_mask == 0).sum()
        rate = valid_region / (valid_region+invalid_region)
        print(f'objmask occupy rate:{rate}')
        #物体越大，相关像素理应越多,阈值可相对降低，反之物体小，没必要这么多像素阈值提高一些
        # threshold = strict_threshold * (1-rate)
        threshold = strict_threshold * torch.exp(-adp_k * rate)

        return threshold

    def Mask_post_process(self, binary_mask, ref_mask, type):

        ref_mask_bool = ref_mask > 0
        # 找到原始mask的边缘像素
        edge_pixels = self.find_edge_pixels(ref_mask_bool)
        edge_coords = torch.nonzero(edge_pixels, as_tuple=False)

        expand_mask_bool = binary_mask

        # 腐蚀
        expand_mask_bool = self.erode_operation(expand_mask_bool, kernel_size=10)
        expand_regions = expand_mask_bool.bool() & ~ref_mask_bool.bool()

        # 不连通的位置直接去掉
        if type == 'hard':
            # 初始化访问标记 DFS搜索通路
            visited = set()

            # 从边缘像素开始进行DFS
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for coord in edge_coords:
                x, y = coord[0].item(), coord[1].item()
                self.dfs(x, y, expand_regions, visited, directions)

            # 构建refined mask
            refined_mask = torch.zeros_like(expand_regions, dtype=torch.bool)
            for (x, y) in visited:
                refined_mask[x, y] = True

            # 确保包含原始mask的像素
            refined_mask |= ref_mask_bool

        refined_mask = self.dilation_operation(refined_mask, kernel_size=15)

        return refined_mask.to(torch.uint8) * 255


    def find_edge_pixels(self,mask):
        # 使用binary_dilation找到边缘
        kernel = torch.tensor([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], device=mask.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        dilated_mask = F.conv2d(mask, kernel, padding=1) > 0
        # 确保dilated_mask也是布尔类型
        dilated_mask = dilated_mask.bool()
        edge_pixels = dilated_mask & ~mask.bool()
        return edge_pixels.squeeze(0).squeeze(0)

    def is_valid(self,x, y, mask):
        # 检查像素是否在mask范围内且值为True
        return 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y]

    def dfs(self,x, y, mask, visited, directions):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if self.is_valid(nx, ny, mask) and (nx, ny) not in visited:
                    stack.append((nx, ny))

    def contrast_operation(self,mask,contrast_beta,clamp=True,min_v=0,max_v=1,dim=-1):
        if dim is not None:
            mean_value = mask.mean(dim=dim)[:,:,None]
        else:
            mean_value = mask.mean()

        #increase varience
        contrasted_mask = (mask - mean_value) * contrast_beta + mean_value
        del mean_value
        if clamp:
            return contrasted_mask.clamp(min=min_v,max=max_v)
        return contrasted_mask
    def prepare_controller_ref_mask(self,mask,use_mask_expansion=True):
        #ndarray mask -> Tensor
        if mask.ndim == 3:
            mask = mask[:,:,0]
        mask = torch.Tensor(mask).to(self.device)
        if use_mask_expansion:
            self.controller.obj_mask = mask
            self.controller.log_mask = True
        return mask
    def otsu_thresholding_torch(self,image):

        hist = torch.histc(image.float(), bins=256, min=0, max=255).to(image.device)


        pixel_sum = torch.sum(hist)
        weight1 = torch.cumsum(hist, 0)
        weight2 = pixel_sum - weight1

        bin_edges = torch.arange(256).float().to(image.device)


        mean1 = torch.cumsum(hist * bin_edges, 0) / (weight1 + (weight1 == 0))
        mean2 = (torch.cumsum(hist.flip(0) * bin_edges.flip(0), 0) / (weight2.flip(0) + (weight2.flip(0) == 0))).flip(0)


        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        threshold = torch.argmax(inter_class_variance)
        # print(f'threshold:{threshold/255}')


        binary_image = torch.where(image <= threshold, 0, 255)
        return binary_image.type(torch.uint8)

    def mask_with_otsu_pytorch(self,tensor: torch.Tensor):
        image = (tensor * 255).to(torch.uint8)

        binary_image = self.otsu_thresholding_torch(image)
        return (binary_image / 255).to(tensor.dtype)
    def normalize_and_binarize(self, mask_logits, ref_mask, roi_expansion, mask_threshold, otsu=False):
        norm_exp_mask = self.normalize_expansion_mask(ref_mask, mask_logits, roi_expansion, )
        if otsu:
            binary_mask = self.mask_with_otsu_pytorch(norm_exp_mask)
        else:
            norm_exp_mask = norm_exp_mask > mask_threshold
            binary_mask = norm_exp_mask.to(mask_logits.dtype)
        return binary_mask
    def fetch_expansion_mask_from_store(self, expansion_masks, mask, roi_expansion, post_process, mask_threshold,
                                        otsu=False):
        binary_exp_mask = self.normalize_and_binarize(expansion_masks, mask, roi_expansion, mask_threshold, otsu)
        if post_process is not None:
            assert post_process in ['hard'], f'not implement method: {post_process}'
            # post process based on distance and init thr
            final_mask = self.Mask_post_process(binary_exp_mask, mask, post_process, )
        else:
            final_mask = binary_exp_mask * 255
        return final_mask.detach().cpu().numpy()

    def gradio_mask_expansion_func(self, img, mask, prompt,
                         guidance_scale, num_step, eta=0,roi_expansion=True,
                         mask_threshold=0.1,post_process='hard',
                        ):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        #reference mask prepare
        mask = self.prepare_controller_ref_mask(mask)
        _ = \
            self.MY_DDIM_INV(
                img,
                prompt,
                num_steps=num_step,
                guidance_scale=guidance_scale,
                eta = eta
            )
        expansion_masks = self.controller.expansion_mask_on_the_fly / self.controller.step_num #expansion mask & average of up mid down resized corresponded self attention maps
        # self.controller.expansion_mask_store = {} #reset for next image
        # step_masks = [v for i, v in expansion_masks.items()]
        candidate_mask = self.fetch_expansion_mask_from_store(expansion_masks,mask,roi_expansion,post_process,mask_threshold)
        self.controller.reset()
        return candidate_mask


    @torch.no_grad()
    def ClipWordMatching(self, img, mask, full_prompt):
        nlp = spacy.load("en_core_web_sm")

        # 使用 spaCy 处理句子
        doc = nlp(full_prompt)
        omit_list = ['photo']

        # 提取名词，排除指定的词
        words = [token.text for token in doc if token.pos_ == "NOUN" and token.text not in omit_list]

        # 通过mask区域的boundary box裁剪图像
        cropped_img = self.crop_image_with_mask(img, mask)
        image = self.clip_process(self.numpy_to_pil(cropped_img)[0]).unsqueeze(0).to(self.device)

        # 对每个词进行编码
        text_tokens = clip.tokenize(words).to(self.device)

        # 计算图像的特征向量
        image_features = self.clip.encode_image(image)

        # 计算每个词的特征向量
        text_features = self.clip.encode_text(text_tokens)

        # 计算相似度
        similarities = torch.cosine_similarity(image_features, text_features)

        # 找到最匹配的词的索引
        best_match_index = torch.argmax(similarities).item()

        # 最匹配的词
        best_match_word = words[best_match_index]

        return best_match_word
    def box_mask(self,mask):
        new_mask = np.zeros_like(mask)
        x, y, w, h = cv2.boundingRect(mask)
        new_mask[y:y + h, x:x + w] = 1
        return new_mask

    def pre_process_with_mask(self, img_list, mask):
        processed_imgs = []

        # 1. Apply dilation to the mask
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

        # 2. Blend the edges of the mask using Gaussian blur
        mask_blurred = cv2.GaussianBlur(dilated_mask, (21, 21), 0)
        mask_blurred = mask_blurred / 255.0  # Normalize mask to [0, 1]

        for img_np in img_list:
            # 3. Calculate the average color of the image
            average_color = img_np.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
            avg_color_img = np.ones_like(img_np) * average_color

            # 4. Integrate the masked area with the average color using the blurred mask
            mask_expanded = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2)  # Expand mask to match image channels
            img_np = img_np * mask_expanded + avg_color_img * (1 - mask_expanded)

            # Clip the result to ensure it's within [0, 255] and convert to uint8
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            processed_imgs.append(img_np)

        return processed_imgs
    def crop_image_with_mask(self, img, mask):
        # 使用cv2.boundingRect获取mask的边界框
        x, y, w, h = cv2.boundingRect(mask)

        # 裁剪图像
        if isinstance(img,List):
            cropped_img = [im[y:y + h, x:x + w] for im in img ]
        else:
            cropped_img = img[y:y + h, x:x + w]
        return cropped_img
    @torch.no_grad()
    def Prompt_guided_mask_expansion_func(self, img, mask, assist_prompt,
                                          num_step, start_step=0,
                                          use_mask_expansion=True,
                                          ):
        source_image = self.preprocess_image(img, self.device)
        # reference mask prepare
        x, y, w, h = cv2.boundingRect(self.dilate_mask(mask,30))
        local_focus = np.zeros_like(mask)
        local_focus[y: y + h, x: x + w] = 255
        mask = self.prepare_controller_ref_mask(mask, use_mask_expansion)
        local_focus = self.prepare_controller_ref_mask(local_focus,False)
        self.controller.local_focus_box = local_focus
        if len(assist_prompt) ==1 and assist_prompt[0]=="":
            assist_prompt = None
        #ddim inv
        latents = self.Expansion_invert(
            source_image,
            assist_prompt,
            num_inference_steps=num_step,
            num_actual_inference_steps=num_step - start_step,
        )
        del latents
        self.controller.reset()  # reset for next image
        candidate_mask = self.controller.obj_mask.detach().cpu().numpy()
        candidate_mask = self.erode_mask(candidate_mask,15)
        return candidate_mask
    @torch.no_grad()
    def DDIM_inversion_func(self, img, mask, prompt,
                            num_step, start_step=0, ref_img=None, ):

        source_image = self.preprocess_image(img, self.device)
        if ref_img is not None:
            ref_image = self.preprocess_image(ref_img, self.device)
            source_image = torch.cat((source_image,ref_image))
        # reference mask prepare
        mask = self.prepare_controller_ref_mask(mask,False)

        latents, latents_list = \
            self.invert(
                source_image,
                prompt,
                guidance_scale=1.0,
                num_inference_steps=num_step,
                num_actual_inference_steps=num_step-start_step,
                return_intermediates=True,
            )

        self.controller.reset() #clear
        return mask.detach().cpu().numpy(), latents_list

    @torch.no_grad()
    def get_mask_center(self,mask):
        y_indices, x_indices = torch.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom =  torch.min(y_indices),  torch.max(y_indices)
            left, right =  torch.min(x_indices),  torch.max(x_indices)
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2
        return mask_center_x.item(),mask_center_y.item()

    def manual_interpolate(self,init_code, non_intersect_mask, union_mask):
        """
        Manually interpolate values for non-intersecting regions using nearby non-intersecting features.

        Parameters:
        init_code (torch.Tensor): The initial code tensor of shape (1, C, H, W).
        non_intersect_mask (torch.Tensor): Mask indicating non-intersecting regions, shape (1, 1, H, W).
        union_mask (torch.Tensor): Mask indicating the union of current and next masks, shape (1, 1, H, W).

        Returns:
        torch.Tensor: The interpolated code tensor.
        """
        batch_size, channels, height, width = init_code.shape
        print(f'processing nums : {non_intersect_mask.sum()}')
        filled_code = init_code.clone()
        start_time = time.time()  # 开始计时
        # Iterate over each pixel in the non_intersect_mask
        for y in range(height):
            for x in range(width):
                if non_intersect_mask[0, 0, y, x] > 0:  # This pixel is in the non_intersect region
                    # Find nearby non-intersecting features for interpolation
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width and union_mask[0, 0, ny, nx] == 0:
                                neighbors.append(init_code[0, :, ny, nx])

                    if neighbors:
                        # Average the features from the neighbors
                        interpolated_value = torch.stack(neighbors, dim=0).mean(dim=0)
                        filled_code[0, :, y, x] = interpolated_value
        end_time = time.time()  # 结束计时
        print(f"Manual interpolation completed in {end_time - start_time:.4f} seconds")
        return filled_code


    def edit_init_code(self, init_code, theta, current_mask, next_mask):
        batch_size, channels, height, width = init_code.shape

        current_mask = F.interpolate(current_mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width),
                                     mode='nearest').squeeze(0).squeeze(0)
        next_mask = F.interpolate(next_mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width),
                                  mode='nearest').squeeze(0).squeeze(0)

        moved_init_code = wrapAffine_tensor(init_code, theta, (width, height), mode='bilinear').unsqueeze(0).clone()

        intersection_mask = (current_mask.bool() & next_mask.bool()).float()
        non_intersect_current_mask = (current_mask.bool() & ~intersection_mask.bool()).float()

        kernel_size = 3
        feature_weight = [[0.7, 1.0, 0.7, ],
                          [1.0, 0.0, 1.0, ],
                          [0.7, 1.0, 0.7, ]]
        # feature_weight = [[0.5, 0.8, 0.8, 0.8, 0.5],
        #                   [0.8, 1.0, 1.0, 1.0, 0.8],
        #                   [0.8, 1.0, 0.0, 1.0, 0.8],
        #                   [0.8, 1.0, 1.0, 1.0, 0.8],
        #                   [0.5, 0.8, 0.8, 0.8, 0.5]]

        partial_conv = PartialConvInterpolation(kernel_size, channels, feature_weight).to(self.device)
        inpaint_mask = current_mask.unsqueeze(0).repeat(1,channels, 1, 1)  # 为指定通道重复掩码
        inpaint_mask[inpaint_mask > 0 ] = 1
        interpolated_original = partial_conv(init_code, 1 - inpaint_mask,non_intersect_current_mask, 2)  # 1 means valid ,0 means to be repaired
        # interpolated_original = tensor_inpaint_fmm(init_code, current_mask)

        next_mask = next_mask.unsqueeze(0).repeat(channels, 1, 1).unsqueeze(0)
        non_intersect_current_mask = non_intersect_current_mask.unsqueeze(0).repeat(channels, 1, 1).unsqueeze(0)
        result_code_0 = torch.where(next_mask > 0, moved_init_code, init_code)
        result_code = torch.where(non_intersect_current_mask > 0, interpolated_original, result_code_0)

        return result_code
    def forward_unet_features_simple(self, z, t, encoder_hidden_states, h_feature=None):
        unet_output, all_intermediate_features,  copy_downblock = self.unet(
            z,
            t,
            h_sample = h_feature,
            copy = True,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True
            )
        h_feature = all_intermediate_features[0]

        # h_feature = copy_downblock[2]
        # h_feature = F.interpolate(h_feature, (interp_res_h, interp_res_w), mode='bilinear')

        return unet_output, h_feature

    @torch.no_grad()
    def prepare_DIFT_INJ_IDX(self,DIFT_LATENTS,t,LAYER_IDX=[0,1,2,3],cos_threshold=0.5,use_one_step=False,source_image=None,ref_img=None,ensemble_size=1):
        if not use_one_step:
            #DDIM INV BASED
            edit_prompt = ""
            text_emb = self.get_text_embeddings(edit_prompt).detach()
            text_emb = torch.cat([text_emb , text_emb], dim=0)

            with torch.no_grad():
                _, all_intermediate_features, copy_downblock = self.unet(
                    DIFT_LATENTS,
                    t,
                    h_sample=None,
                    copy=True,
                    encoder_hidden_states=text_emb,
                    return_intermediates=True
                )
            del copy_downblock,_
            feature_map_layers = {}

            for layer in LAYER_IDX:
                feat = all_intermediate_features[layer]
                feature_map_layers[layer] = calculate_cosine_similarity_between_batches(feat,cos_threshold=cos_threshold)

            self.controller.correspondence_map = feature_map_layers
            del all_intermediate_features,feature_map_layers
        else:
            #ONE STEP ADD NOISE BASED
            assert source_image is not None,"given source img for one step FI"
            assert ref_img is not None, "given ref img for one step FI"
            source_image = self.preprocess_image(source_image, self.device)
            ref_img = self.preprocess_image(ref_img, self.device)
            batch_img_tensor = [source_image,ref_img]
            edit_prompt = ""
            text_emb = self.get_text_embeddings(edit_prompt).detach()# [1, 77, dim]
            text_emb = text_emb.repeat(ensemble_size, 1, 1)
            feature_map_layers = {}
            unet_ft =[]
            for img_tensor in batch_img_tensor: #1,C,H,W
                img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w
                with torch.no_grad():
                    latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
                    noise = torch.randn_like(latents).to(self.device)
                    latents_noisy = self.scheduler.add_noise(latents, noise, t)
                    _, all_intermediate_features, copy_downblock = self.unet(
                        latents_noisy,
                        t,
                        h_sample=None,
                        copy=True,
                        encoder_hidden_states=text_emb,
                        return_intermediates=True
                    )
                    unet_ft.append(all_intermediate_features)
                    del copy_downblock, _

            for layer in LAYER_IDX:
                edit_feat = unet_ft[0][layer].mean(0, keepdim=True)
                ref_feat = unet_ft[1][layer].mean(0, keepdim=True)
                feat = torch.cat((edit_feat,ref_feat),dim=0)
                feature_map_layers[layer] = calculate_cosine_similarity_between_batches(feat,
                                                                                        cos_threshold=cos_threshold)

            self.controller.correspondence_map = feature_map_layers
            del all_intermediate_features, feature_map_layers,unet_ft,feat,edit_feat,ref_feat














    def prepare_h_feature(self, init_code, t, edit_prompt, BG_preservation=True, foreground_mask=None, lr=0.01, lam=0.1,
                          eta=0.0, refer_latent=None,h_feature_input=None,):
        # Encode prompt
        # refer_latent = None
        # h_feature_input = None
        #only need edit_stream h_feature
        uc_text = ""
        # edit_prompt = edit_prompt[0]
        edit_prompt = ""
        text_emb = self.get_text_embeddings(edit_prompt).detach()
        uncon_text_emb = self.get_text_embeddings(uc_text).detach()
        text_emb = torch.cat([uncon_text_emb, text_emb], dim=0)#cfg text emb
        init_code = torch.cat([init_code, init_code], dim=0)
        if refer_latent is None:
            with torch.no_grad():
                if h_feature_input is not None and not BG_preservation:
                    h_feature = h_feature_input
                else:
                    unet_output, h_feature = self.forward_unet_features_simple(init_code, t, encoder_hidden_states=text_emb,h_feature=h_feature_input)
                    x_prev_0, _ = self.step(unet_output, t, init_code)
        else:
            with torch.no_grad():
                if h_feature_input is not None and not BG_preservation:
                    h_feature = h_feature_input
                else:
                    _, h_feature = self.forward_unet_features_simple(init_code, t, encoder_hidden_states=text_emb,h_feature=h_feature_input)
            x_prev_0 = refer_latent[0].unsqueeze(0)
        if BG_preservation:
            assert foreground_mask is not None, "For BG preservation, foreground mask should be given"
            h_feature.requires_grad_(True)
            optimizer = torch.optim.Adam([h_feature], lr=lr)
            scaler = torch.cuda.amp.GradScaler()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                unet_output, h_feature = self.forward_unet_features_simple(init_code, t,encoder_hidden_states=text_emb,h_feature=h_feature)
                x_prev_updated, _ = self.step(unet_output, t, init_code,None,eta)
                loss = lam * ((x_prev_updated - x_prev_0) * (1.0 - foreground_mask)).abs().sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return h_feature

    @torch.no_grad()
    def prepare_various_mask(self, shifted_mask, ori_mask, sup_res_w, sup_res_h, init_code,ddpm_region):
        shifted_mask_tensor_dilated = self.prepare_tensor_mask(self.dilate_mask(ddpm_region,15),sup_res_w,sup_res_h)
        shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
        ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
        foreground_latent_region = shifted_mask_tensor_dilated + ori_mask_tensor
        foreground_latent_region[foreground_latent_region > 0.0] = 1
        local_variance_reg = (1 - shifted_mask_tensor) * shifted_mask_tensor_dilated
        shifted_mask_tensor[shifted_mask_tensor > 0.0] = 1
        latent_foreground_masks = F.interpolate(foreground_latent_region.unsqueeze(0).unsqueeze(0),
                                                (init_code.shape[2], init_code.shape[3]),
                                                mode='nearest').squeeze(0, 1)
        obj_dilation_mask = F.interpolate(shifted_mask_tensor.unsqueeze(0).unsqueeze(0),
                                                (init_code.shape[2], init_code.shape[3]),
                                                mode='nearest').squeeze(0, 1)
        local_variance_reg = F.interpolate(local_variance_reg.unsqueeze(0).unsqueeze(0),
                                          (init_code.shape[2], init_code.shape[3]),
                                          mode='nearest').squeeze(0, 1)
        return latent_foreground_masks,obj_dilation_mask,local_variance_reg

    def prepare_tensor_mask(self,mask,sup_res_w,sup_res_h):
        # mask interpolation
        if mask.ndim == 3:
            mask = mask[:,:,0]
        mask_tensor_shifted = torch.tensor(mask, device=self.device)
        mask_tensor_shifted = mask_tensor_shifted.unsqueeze(0).unsqueeze(0)
        transformed_masks_tensor = F.interpolate(mask_tensor_shifted, (sup_res_h, sup_res_w), mode="nearest").squeeze(0, 1)
        transformed_masks_tensor[transformed_masks_tensor > 0.0] = 1

        return transformed_masks_tensor

    def Details_Preserving_regeneration(self, source_image, inverted_latents, edit_prompt, shifted_mask, ori_mask,
                                        num_steps=100, start_step=30, end_step=10,
                                        guidance_scale=3.5, eta=1,
                                        feature_injection=True, FI_range=(900,680), sim_thr=0.5, DIFT_LAYER_IDX = [0,1,2,3],
                                        use_mtsa=True, ref_img=None, ddpm_region=None):

        """
        latent vis
        # noised_image = self.decode_latents(start_latents).squeeze(0)
        # # print(noised_image.shape)
        # noised_image = self.numpy_to_pil(noised_image)[0]
        # print(noised_image.size)
        # print(noised_image.shape)
        """

        start_latents = inverted_latents[-1]  #[ori,35 steps latents:50->15]
        init_code_orig = deepcopy(start_latents)
        #PREPARE DIFT INJ IDX
        DIFT_STEP = int(261/1000*num_steps)
        DIFT_latents = inverted_latents[DIFT_STEP]  #time step 261 , 261/1000 = 13/50
        t_dift = self.scheduler.timesteps[num_steps-DIFT_STEP-1]
        self.prepare_DIFT_INJ_IDX(DIFT_latents,t_dift,DIFT_LAYER_IDX,sim_thr,False,source_image,ref_img,8)

        # PREPARE H FEATURE
        # t = self.scheduler.timesteps[start_step]
        # unet_feature_idx = [3]

        full_h, full_w = source_image.shape[:2]
        print(f'full_h:{full_h};full_w:{full_w}')
        sup_res_h = int(0.5 * full_h)
        sup_res_w = int(0.5 * full_w)

        foreground_mask,object_mask,local_var_reg = self.prepare_various_mask(shifted_mask, ori_mask, sup_res_w, sup_res_h, init_code_orig,ddpm_region)
        # object_mask = self.dilate_mask(object_mask,15)
        # h_feature = self.prepare_h_feature(init_code_orig,t,edit_prompt,BG_preservation=False,foreground_mask=foreground_mask,lr=0.1,lam=0.1,eta=1.0)



        # noised_optimized_image = self.decode_latents(init_code_orig).squeeze(0)
        # noised_optimized_image = self.numpy_to_pil(noised_optimized_image)[0]

        mask = self.prepare_controller_ref_mask(shifted_mask,True)
        SDSA_REF_MASK = self.prepare_controller_ref_mask(ori_mask,False)
        SDSA_REF_MASK = F.interpolate(SDSA_REF_MASK.unsqueeze(0).unsqueeze(0),
                                                (init_code_orig.shape[2], init_code_orig.shape[3]),
                                                mode='nearest').squeeze(0, 1)


        self.controller.SDSA_REF_MASK = SDSA_REF_MASK
        self.controller.reset()
        # refer_gen_image = edit_ori_image[0] #vis ddim results
        self.controller.log_mask = False
        refer_latents_ori = inverted_latents[::-1]
        edit_gen_image,ref_gen_image = self.forward_sampling_BG(
            prompt=[edit_prompt,""],
            # refer_latents=refer_latents_list,
            refer_latents=refer_latents_ori,
            end_step=end_step,
            batch_size=2,
            latents = init_code_orig,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            num_actual_inference_steps=num_steps - start_step,
            eta=eta,
            foreground_mask=foreground_mask,
            obj_mask=object_mask,
            local_var_reg=local_var_reg ,
            feature_injection_allowed = feature_injection,
            feature_injection_timpstep_range= FI_range,
            use_mtsa=use_mtsa,
        )
        self.controller.reset()
        refer_gen_image = ref_gen_image.permute(1, 2, 0).detach().cpu().numpy()
        edit_gen_image = edit_gen_image.permute(1, 2, 0).detach().cpu().numpy()
        return edit_gen_image,refer_gen_image
    def ReggioRecurrentEdit(self,source_image, inverted_latents,  edit_prompt,  expand_mask,x_shift,y_shift,resize_scale,rotation_angle,motion_split_steps,num_steps=100, start_step=30,end_step=10, guidance_scale=3.5, eta=0,
                   roi_expansion=True,mask_threshold=0.1, post_process='hard',max_times=10,sim_thr=0.7,lr=0.01,lam=0.1,):


        """
        latent vis
        # noised_image = self.decode_latents(start_latents).squeeze(0)
        # # print(noised_image.shape)
        # noised_image = self.numpy_to_pil(noised_image)[0]
        # print(noised_image.size)
        # print(noised_image.shape)
        """

        start_latents = inverted_latents[-1]
        init_code_orig = deepcopy(start_latents)
        t = self.scheduler.timesteps[start_step]
        unet_feature_idx = [3]




        full_h, full_w = source_image.shape[:2]
        print(f'full_h:{full_h};full_w:{full_w}')
        sup_res_h = int(0.5 * full_h)
        sup_res_w = int(0.5 * full_w)


        #mask interpolation
        mask_tensor = torch.tensor(expand_mask,device=self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        transformed_masks_tensor = F.interpolate(mask_tensor, (sup_res_h, sup_res_w), mode="nearest")
        x_shift = int(x_shift * 0.5)
        y_shift = int(y_shift * 0.5)

        #TODO: 1-step optimization
        self.controller.log_mask = False
        updated_init_code, h_feature, final_traj_mask = self.Recurrent_diffusion_updates_2D(edit_prompt,start_latents, t, transformed_masks_tensor,x_shift,y_shift,resize_scale,rotation_angle,motion_split_steps,
                                                                                 unet_feature_idx=unet_feature_idx,lam=lam,lr=lr,
                                                                                 sup_res_w = sup_res_w, sup_res_h = sup_res_h,max_times=max_times,sim_thr=sim_thr)

        with torch.no_grad():
            noised_optimized_image = self.decode_latents(updated_init_code).squeeze(0)
            noised_optimized_image = self.numpy_to_pil(noised_optimized_image)[0]
            #updated_init_code == init_code == start code
            # print(f'init_code == start_code : {torch.all(start_latents == updated_init_code)}')

            mask = self.prepare_controller_ref_mask(final_traj_mask)
            self.controller.expansion_mask_store = {}
            self.controller.log_mask = True

            ori_gen_image,edit_gen_image = self.forward_sampling(
                prompt=edit_prompt,
                h_feature=h_feature,
                end_step=end_step,
                batch_size=2,
                latents=torch.cat([updated_init_code, updated_init_code], dim=0), #same
                # latents=torch.cat([updated_init_code, updated_init_code], dim=0),
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                num_actual_inference_steps=num_steps-start_step,
            )
            ori_gen_image =  self.numpy_to_pil(ori_gen_image.permute(1,2,0).detach().cpu().numpy())[0]
            edit_gen_image = self.numpy_to_pil(edit_gen_image.permute(1,2,0).detach().cpu().numpy())[0]

            # 从意义上来说，如果ddpm forward 的作用是优化细节，其实mask不如直接用final mask
            # 但是对于需要利用后续先验进一步修复的case，或者循环每一步forward 进行优化的case 都可以重新获得mask
            # 所以先保留这部分代码
            # get ddpm forward expansion target mask
            ddpm_for_expansion_masks = self.controller.expansion_mask_store  # expansion mask & average of up mid down resized corresponded self attention maps
            self.controller.expansion_mask_store = {}  # reset for next image
            candidate_mask = self.fetch_expansion_mask_from_store([ddpm_for_expansion_masks], mask, roi_expansion,post_process, mask_threshold)



        return ori_gen_image, edit_gen_image,noised_optimized_image,candidate_mask

    def wrapAffine_tensor(self, tensor, theta, dsize, mode='bilinear', padding_mode='zeros', align_corners=False,
                          border_value=0):
        """
        对给定的张量进行仿射变换，仿照 cv2.warpAffine 的功能。

        参数：
        - tensor: 要变换的张量，形状为 (C, H, W) 或 (H, W)
        - theta: 2x3 变换矩阵，形状为 (2, 3)
        - dsize: 输出尺寸 (width, height)
        - mode: 插值方法，可选 'bilinear', 'nearest', 'bicubic'
        - padding_mode: 边界填充模式，可选 'zeros', 'border', 'reflection'
        - align_corners: 是否对齐角点，默认为 False
        - border_value: 填充值

        返回：
        - transformed_tensor: 变换后的张量，形状与输入张量相同
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # 生成变换的 grid
        grid = F.affine_grid(theta.unsqueeze(0), [tensor.size(0), tensor.size(1), dsize[1], dsize[0]],
                             align_corners=align_corners)

        # 进行 grid 采样
        output = F.grid_sample(tensor, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        transformed_tensor = output.squeeze(0)

        # 使用填充值填充边界
        if padding_mode == 'zeros' and border_value != 0:
            mask = (grid.abs() > 1).any(dim=-1, keepdim=True)
            transformed_tensor[mask] = border_value

        return transformed_tensor

    def weighted_loss(self,F0_region, F1_region, topk=10):
        """
        Compute the weighted L1 loss where the weights are determined by the cosine similarity.
        The topk positions with the smallest similarity have the highest weights.

        Parameters:
        F0_region (torch.Tensor): Reference feature tensor of shape [batch_size, channels, height, width].
        F1_region (torch.Tensor): Edited feature tensor of shape [batch_size, channels, height, width].
        topk (int): The number of top positions with the smallest similarity to assign higher weights.

        Returns:
        torch.Tensor: The computed weighted L1 loss.
        """
        # Compute L1 loss without reduction
        l1_diff = F.l1_loss(F0_region, F1_region, reduction='none').mean(dim=1)

        # Compute cosine similarity
        similarity = F.cosine_similarity(F0_region, F1_region, dim=1)

        inverse_sim = 1- similarity
        norm_inv_sim = (inverse_sim-inverse_sim.min())/(inverse_sim.max()-inverse_sim.min())
        # topk_values, topk_indices = torch.topk(1-similarity_flat, topk, dim=1)  # Use -similarity to get smallest
        #
        # # Create a weight tensor
        # weights = torch.ones_like(similarity_flat)
        #
        # # Assign higher weights to the topk smallest similarity positions
        # for i in range(weights.size(0)):
        #     weights[i, topk_indices[i]] = 10.0  # Assign a higher weight, e.g., 10.0

        # Reshape weights to match the original similarity shape
        weights =  norm_inv_sim * 2

        # Apply weights to the L1 loss
        weighted_l1_diff = weights * l1_diff

        # Compute the final weighted loss
        weighted_loss = torch.mean(weighted_l1_diff)

        return weighted_loss
    @torch.no_grad()
    def reach_similarity_condition(self, F1_region,target_feature,threshold):
        # current_feature = F1[:,:,trajectory_current_mask.bool()]
        current_feature = F1_region
        current_feature_flattened = current_feature.view(current_feature.size(0), -1)
        target_feature_flattened = target_feature.view(target_feature.size(0), -1)

        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(current_feature_flattened, target_feature_flattened, dim=1)
        print(f'current_similarity:{cosine_similarity.item()}')
        return cosine_similarity > threshold
    def Recurrent_diffusion_updates_2D(self,
                                 prompt,
                                 init_code,
                                 t,
                                 masks_tensor,
                                 x_shift,y_shift,resize_scale,rotation_angle,
                                 motion_split_steps,
                                 unet_feature_idx=[3],
                                 sup_res_h=256,
                                 sup_res_w=256,
                                 lam=0.1, lr=0.01,max_times = 10,sim_thr = 0.92 ):
        # iteratively optimize in one step and refine serveral times for each step to assure succefully edit each step
        print(f'mask_tensor.shape:{masks_tensor.shape}')
        # Encode prompt
        text_emb = self.get_text_embeddings(prompt).detach()
        with torch.no_grad():
            unet_output, F0, h_feature = self.forward_unet_features(init_code, t, encoder_hidden_states=text_emb,
                                                                    layer_idx=unet_feature_idx,
                                                                    interp_res_h=sup_res_h,
                                                                    interp_res_w=sup_res_w)
            x_prev_0, _ = self.step(unet_output, t, init_code)




        # prepare for point tracking and background regularization
        trajectory_masks_sup = copy.deepcopy(masks_tensor).squeeze(0,1)

        height , width = trajectory_masks_sup.shape[:2]
        rotation_angle_step = rotation_angle / motion_split_steps
        resize_scale_step = resize_scale ** (1 / motion_split_steps)
        dx = x_shift / motion_split_steps
        dy = y_shift / motion_split_steps

        # h_features = []


        trajectory_current_mask = trajectory_masks_sup
        current_feature_like = F0
        # return init_code, h_feature, trajectory_masks_sup

        step_idx = 0
        try_times = 1
        while step_idx <= motion_split_steps:
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                # copy_h = copy.deepcopy(h_feature)
                # h_features.append(copy_h)

                if (step_idx == 0) or (try_times > max_times) or self.reach_similarity_condition(F1_region, F0_region ,sim_thr):
                    #if init or last step finish -> move on to next step
                    step_idx += 1
                    try_times = 1
                    if step_idx > motion_split_steps:
                        break
                    # copy_h = copy.deepcopy(h_feature)
                    # h_features.append(copy_h)
                    # get mask center
                    mask_center_x, mask_center_y = self.get_mask_center(trajectory_current_mask) # relocate mask center every step,
                    # allowing tracking real mask center update

                    # get transformation matrix
                    transformation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle_step,
                                                                    resize_scale_step)
                    transformation_matrix[0, 2] += dx
                    transformation_matrix[1, 2] += dy
                    # cv2 rotation matrix -> Tensor adaptive transformation theta
                    theta = torch.tensor(param2theta(transformation_matrix, width, height), dtype=F0.dtype,
                                         device=self.device)
                    trajectory_current_mask = trajectory_current_mask.to(F0.dtype)
                    trajectory_next_mask = wrapAffine_tensor(trajectory_current_mask, theta, (width, height), mode='nearest')[0]
                    foreground_latent_region = trajectory_next_mask + trajectory_current_mask
                    foreground_latent_region[foreground_latent_region > 0.0] = 1
                    latent_trajectory_masks = F.interpolate(foreground_latent_region.unsqueeze(0).unsqueeze(0),
                                                            (init_code.shape[2], init_code.shape[3]),
                                                            mode='nearest').squeeze(0, 1)
                    # init_code = self.edit_init_code(init_code, theta, trajectory_current_mask, trajectory_next_mask)
                    # h_feature = self.edit_init_code(h_feature, theta, trajectory_current_mask, trajectory_next_mask)
                    h_feature = h_feature.detach().requires_grad_(True)
                    # return init_code, h_feature, trajectory_next_mask
                    # prepare optimizable init_code and optimizer
                    # h_feature.requires_grad_(True)
                    optimizer = torch.optim.Adam([h_feature], lr=lr)
                    # prepare amp scaler for mixed-precision training
                    scaler = torch.cuda.amp.GradScaler()

                    unet_output, F1, h_feature = self.forward_unet_features(init_code, t,
                                                                            encoder_hidden_states=text_emb,
                                                                            h_feature=h_feature,
                                                                            layer_idx=unet_feature_idx,
                                                                            interp_res_h=sup_res_h,
                                                                            interp_res_w=sup_res_w)
                    x_prev_updated, _ = self.step(unet_output, t, init_code)


                    trajectory_next_feature_like = wrapAffine_tensor(current_feature_like,theta,(width, height), mode='bilinear').unsqueeze(0)


                    F0_region = trajectory_next_feature_like[:,:,trajectory_next_mask.bool()].detach() # F0_region update
                    F1_region = F1[:,:,trajectory_next_mask.bool()]
                    trajectory_current_mask = trajectory_next_mask
                    current_feature_like = trajectory_next_feature_like
                else:
                    unet_output, F1, h_feature = self.forward_unet_features(init_code, t,
                                                                            encoder_hidden_states=text_emb,
                                                                            h_feature=h_feature,
                                                                            layer_idx=unet_feature_idx,
                                                                            interp_res_h=sup_res_h,
                                                                            interp_res_w=sup_res_w)
                    x_prev_updated, _ = self.step(unet_output, t, init_code)
                    #recurrent refine on this step
                    try_times += 1
                    F1_region = F1[:,:,trajectory_current_mask.bool()] # reattain F1_region from new F1

                #update every step every try
                assert F0_region.shape == F1_region.shape,"The shapes of F0_region and F1_region do not match for L1 loss calculation."
                edit_loss = F.l1_loss(F0_region, F1_region)
                # edit_loss = self.weighted_loss(F0_region,F1_region,topk=1000)
                BG_loss = lam * ((x_prev_updated - x_prev_0) * (1.0 - latent_trajectory_masks)).abs().sum()
                loss  = edit_loss + BG_loss
                print(f'handling step:{step_idx}/{motion_split_steps} , trying times:{try_times}')
                print(f'Edit_loss: {edit_loss.item()},BG_loss: {BG_loss.item()} Total_loss: {loss.item()}')
                # print(f'BG_loss: {BG_loss.item()}')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()




        # return init_code, h_feature, h_features
        # return init_code, h_feature, h_features, trajectory_current_mask
        return init_code, h_feature,

class AutoPipeReggio(StableDiffusionPipeline):
    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)

    def inv_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            eta=0.,
            verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def ctrl_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            mask,
            eta: float = 0.0,
            generator=None,
    ):
        """
        Predict the sample of the next step in the denoise process with eta control.

        Args:
            model_output (torch.FloatTensor): direct output from learned diffusion model.
            timestep (int): current discrete timestep in the diffusion chain.
            x (torch.FloatTensor): current instance of sample being created by diffusion process.
            eta (float): weight of noise for added noise in diffusion step. Default is 0.0.
            generator: random number generator.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: previous sample and predicted original sample.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        variance = self._get_variance(timestep, prev_timestep).to(self.device)
        std_dev_t = eta * variance ** (0.5)
        if model_output.shape[0] == 2:  # reference stream
            # batch_0:LOCAL DDPM
            # batch_1:DDIM
            std_dev_t = torch.cat((std_dev_t[None,], torch.zeros_like(std_dev_t)[None,]))[:, None, None, None]
            mask = mask.repeat(1, 4, 1, 1)
            mask = torch.cat((mask, torch.ones_like(mask)))

        if mask is not None:  # local ddpm
            pred_dir_mask = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (
                0.5) * model_output * mask  # ddpm masked
            pred_dir = (1 - alpha_prod_t_prev) ** (0.5) * model_output * (
                        1 - mask) + pred_dir_mask  # ddim unmasked
        else:  # full ddpm
            pred_dir = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * model_output
        x_prev = alpha_prod_t_prev ** 0.5 * pred_x0 + pred_dir

        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
            )
            # batch_0:LOCAL DDPM
            # batch_1:DDIM
            if mask is None:
                variance = std_dev_t * variance_noise
            else:
                variance = std_dev_t * variance_noise * mask

            x_prev = x_prev + variance

        return x_prev, pred_x0

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = self.device
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.cuda())[0]
        return text_embeddings

    # get all intermediate features and then do bilinear interpolation
    # return features in the layer_idx list
    def forward_unet_features(self, z, t, encoder_hidden_states, h_feature=None, layer_idx=[0],
                              interp_res_h=256, interp_res_w=256):
        unet_output, all_intermediate_features, copy_downblock = self.unet(
            z,
            t,
            h_sample=h_feature,
            copy=True,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True
        )

        all_return_features = []

        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode='bilinear')
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)

        h_feature = all_intermediate_features[0]
        # h_feature = copy_downblock[2]

        # h_feature = F.interpolate(h_feature, (interp_res_h, interp_res_w), mode='bilinear')

        return unet_output, return_features, h_feature

    @torch.no_grad()
    def __call__(
            self,
            prompt,
            prompt_embeds=None,  # whether text embedding is directly provided.
            h_feature=None,
            batch_size=2,
            end_step=None,
            height=512,
            width=512,
            num_inference_steps=50,
            num_actual_inference_steps=None,
            guidance_scale=7.5,
            latents=None,
            unconditioning=None,
            neg_prompt=None,
            return_intermediates=False,
            gen_img=False,
            **kwds):
        DEVICE = self.device

        if prompt_embeds is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size

            # text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        else:
            batch_size = prompt_embeds.shape[0]
            text_embeddings = prompt_embeds
        print("input text embeddings :", text_embeddings.shape)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.vae.dtype)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]

        if gen_img:
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue

                if guidance_scale > 1.:
                    model_inputs = torch.cat([latents] * 2)
                else:
                    model_inputs = latents

                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat(
                        [unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
                    # predict the noise
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                if guidance_scale > 1.0:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                # compute the previous noise sample x_t -> x_t-1
                # YUJUN: right now, the only difference between step here and step in scheduler
                # is that scheduler version would clamp pred_x0 between [-1,1]
                # don't know if that's gonna have huge impact
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents_list.append(latents)

        else:
            h_feature = torch.cat([h_feature] * 2)

            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):

                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue

                if guidance_scale > 1.:
                    model_inputs = torch.cat([latents] * 2)
                    h_feature_inputs = torch.cat([h_feature] * 2)
                else:
                    model_inputs = latents
                    # h_feature_inputs = h_feature
                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat(
                        [unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
                # predict the noise
                if guidance_scale > 1:
                    if i < 50 - end_step:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,
                                               h_sample=h_feature_inputs)
                    else:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                else:
                    if i < 50 - end_step:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,
                                               h_sample=h_feature)
                        print("i+: ", i)
                    else:
                        noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
                        print("i: ", i)
                if guidance_scale > 1.0:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

                # compute the previous noise sample x_t -> x_t-1
                # YUJUN: right now, the only difference between step here and step in scheduler
                # is that scheduler version would clamp pred_x0 between [-1,1]
                # don't know if that's gonna have huge impact
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents_list.append(latents)
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image

    def forward_sampling_BG(
            self,
            prompt,
            prompt_embeds=None,  # whether text embedding is directly provided.
            refer_latents=None,
            batch_size=1,
            end_step=None,
            height=512,
            width=512,
            num_inference_steps=50,
            num_actual_inference_steps=None,
            guidance_scale=7.5,
            latents=None,
            unconditioning=None,
            neg_prompt=None,
            return_intermediates=False,
            eta=0.0,
            foreground_mask=None,
            obj_mask=None,
            local_var_reg=None,
            blending=True,
            feature_injection_allowed=True,
            feature_injection_timpstep_range=(900, 600),
            use_mtsa=True,
            **kwds):
        DEVICE = self.device
        assert guidance_scale > 1.0, 'USING THIS MODULE CFG Must > 1.0'
        self.controller.use_cfg = True
        self.controller.share_attn = use_mtsa  # allow SDSA
        self.controller.local_edit = True
        if prompt_embeds is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size

            # text embeddings
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        else:
            batch_size = prompt_embeds.shape[0]
            text_embeddings = prompt_embeds
        print("input text embeddings :", text_embeddings.shape)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.vae.dtype)

        # unconditional embedding for classifier free guidance
        # if guidance_scale > 1.:
        if neg_prompt:
            uc_text = neg_prompt
        else:
            uc_text = ""
        unconditional_input = self.tokenizer(
            [uc_text] * batch_size,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]

        # original sample
        # TODO :EACH STEP :need h feature as input ,next step need new h feature from new t and new init latent
        assert foreground_mask is not None, 'FOR BG PRESERVATION foreground_mask should not be None'
        start_step = num_inference_steps - num_actual_inference_steps
        h_feature = None
        self.h_feature_cfg = True
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue
            timestep = t.detach().item()
            if timestep > feature_injection_timpstep_range[0] or timestep < feature_injection_timpstep_range[1]:
                self.controller.set_FI_forbid()
            else:
                if feature_injection_allowed:
                    print(f"Feature Injection is allowed at timestep={timestep}")
                    self.controller.set_FI_allow()
                else:
                    self.controller.set_FI_forbid()

            # TODO: BG preservation h feature
            # if i < 50 - end_step:
            #     self.controller.log_mask = False
            #     h_feature = self.prepare_h_feature(latents[0,None], t, prompt, BG_preservation=False,
            #                                        foreground_mask=foreground_mask, lr=0.1, lam=1, eta=1.0,
            #                                        refer_latent=refer_latents[i - start_step + 1],
            #                                        h_feature_input=h_feature,)
            # [edit,ref]
            ref_latent = refer_latents[i - start_step + 1][1]
            latents[1] = ref_latent
            if i < 50 - end_step:
                self.controller.share_attn = use_mtsa  # allow SDSA
            else:
                self.controller.share_attn = False
                # h_feature = torch.cat([h_feature] * 2)
            # if guidance_scale > 1.:
            with torch.no_grad():

                model_inputs = torch.cat([latents] * 2)
                # h_feature_inputs = torch.cat([h_feature] * 2)
                h_feature_inputs = h_feature
                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat(
                        [unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
                # predict the noise
                # if guidance_scale > 1:
                self.controller.log_mask = False
                if i < 50 - end_step:
                    noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,
                                           h_sample=h_feature_inputs)
                else:
                    noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                if not blending:
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                else:
                    local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * obj_mask
                    noise_pred = noise_pred_uncon + local_text_guidance
                # compute the previous noise sample x_t -> x_t-1
                # YUJUN: right now, the only difference between step here and step in scheduler
                # is that scheduler version would clamp pred_x0 between [-1,1]
                # don't know if that's gonna have huge impact
                # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, eta=eta)[0]
                # full_mask = torch.ones_like(obj_mask)
                latents = self.ctrl_step(noise_pred, t, latents, local_var_reg, eta=eta)[0]
                latents_list.append(latents)
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image

    @torch.no_grad()
    def Expansion_invert(
            self,
            image: torch.Tensor,
            assist_prompt,
            num_inference_steps=50,
            num_actual_inference_steps=None,
            **kwds):

        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = self.device
        batch_size = image.shape[0]
        assert batch_size == 1, ">1 bs not implemented yet"
        # if isinstance(assist_prompt, list):
        #     if batch_size == 1:
        image = image.expand(len(assist_prompt), -1, -1, -1)
        prompt = assist_prompt

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        # print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)

        # max_length = text_input.input_ids.shape[-1]
        unconditional_input = self.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
        text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
        self.controller.use_cfg = True
        self.controller.bidirectional_loc = True
        # interative sampling

        self.scheduler.set_timesteps(num_inference_steps)
        # DIFT_STEP = int(261/1000*num_inference_steps)
        # t_dift = self.scheduler.timesteps[num_inference_steps-DIFT_STEP-1]
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        # for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
        for i, t in enumerate(reversed(self.scheduler.timesteps)):
            if num_actual_inference_steps is not None and i > num_actual_inference_steps:
                continue
            elif i == len(self.scheduler.timesteps) - 1:
                self.controller.last_step = True

            model_inputs = torch.cat([latents] * 2)
            # t= t_dift
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncon
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def invert(
            self,
            image: torch.Tensor,
            prompt,
            num_inference_steps=50,
            num_actual_inference_steps=None,
            guidance_scale=7.5,
            eta=0.0,
            return_intermediates=False,
            **kwds):

        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = self.device
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            self.controller.use_cfg = True
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue

            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
        print(f'leng latent inv {len(latents_list)}')
        print(f'shape latent:{latents.shape}')
        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents

    def dilate_mask(self, mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def erode_mask(self, mask, dilate_factor=15):
        mask = mask.astype(np.uint8)
        mask = cv2.erode(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def get_attention_scores(self, query, key, attention_mask=None, use_softmax=True):
        dtype = query.dtype

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if not use_softmax:
            return attention_scores, dtype

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_continuous_transformed_mask(self, mask, x_shift, y_shift, resize_scale=1.0, rotation_angle=0,
                                            flip_horizontal=False, flip_vertical=False, split_stage_num=10):

        transformed_mask_list = []
        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = mask > 128

        # Continuous editing is not designed for flip operation,
        # since it is simple, inpaining can solve this promblem
        # regardless of this for now
        ## 检查是否需要水平翻转
        # if flip_horizontal:
        #     transformed_mask = cv2.flip(transformed_mask, 1)
        #
        # # 检查是否需要垂直翻转
        # if flip_vertical:
        #     transformed_mask = cv2.flip(transformed_mask, 0)

        # 获取图像尺寸
        height, width = mask.shape[:2]
        y_indices, x_indices = np.where(mask.astype(bool))
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = np.min(y_indices), np.max(y_indices)
            left, right = np.min(x_indices), np.max(x_indices)
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2

        rotation_angle_step = rotation_angle / split_stage_num
        resize_scale_step = resize_scale ** (1 / split_stage_num)
        dx = x_shift / split_stage_num
        dy = y_shift / split_stage_num

        transformed_mask_temp = mask
        transformed_mask_list.append(transformed_mask_temp)
        for step in range(split_stage_num):
            # single step matrix for current center
            transformation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y),
                                                            -rotation_angle_step,
                                                            resize_scale_step)
            transformation_matrix[0, 2] += dx
            transformation_matrix[1, 2] += dy
            # print(f'step:{step} matrix:{transformation_matrix}')
            transformed_mask_temp = cv2.warpAffine(transformed_mask_temp, transformation_matrix,
                                                   (width, height),
                                                   flags=cv2.INTER_NEAREST)
            transformed_mask_list.append(transformed_mask_temp)
            mask_center_x += dx
            mask_center_y += dy
        # output_dir = 'temp_dir_vis'
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # for idx, mask in enumerate(transformed_mask_list):
        #     plt.figure()
        #     plt.imshow(mask, cmap='gray')
        #     plt.title(f'Step {idx}')
        #     plt.axis('off')
        #     plt.savefig(os.path.join(output_dir, f'mask_step_{idx}.png'))
        #     plt.close()
        return transformed_mask_list

    def temp_view_img(self, image: Image.Image, title: str = None) -> None:
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

    def replace_with_SV3D_targets_inpainting(self, ori_img, trans_img, ori_mask, trans_mask, target_mask, mode,
                                             inpainter, inp_prompt=None,
                                             ):

        # inpaint & replace
        if isinstance(ori_img, Image.Image):
            ori_img = np.array(ori_img)
        if isinstance(trans_img, Image.Image):
            trans_img = np.array(trans_img)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        if ori_mask.ndim == 3 and ori_mask.shape[2] == 3:
            ori_mask = cv2.cvtColor(ori_mask, cv2.COLOR_BGR2GRAY)
            # mask = mask > 128
        if trans_mask.ndim == 3 and trans_mask.shape[2] == 3:
            trans_mask = cv2.cvtColor(trans_mask, cv2.COLOR_BGR2GRAY)
            # source_mask =source_mask > 128

        ori_mask = (ori_mask > 0).astype(bool)
        ori_image_back_ground = np.where(ori_mask[:, :, None], 0, ori_img).astype(np.uint8)
        image_with_hole = ori_image_back_ground
        coarse_repaired = np.array(inpainter(Image.fromarray(ori_image_back_ground), Image.fromarray(
            ori_mask.astype(np.uint8) * 255)))  # lama inpainting filling the black regions
        if mode != 1:
            inpaint_mask = Image.fromarray(ori_mask.astype(np.uint8) * 255)
            sd_to_inpaint_img = Image.fromarray(coarse_repaired)
            # print(f'SD inpainting Processing:')
            semantic_repaired = \
                self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask).images[0]
            semantic_repaired = np.array(semantic_repaired)
        else:
            semantic_repaired = coarse_repaired

        if semantic_repaired.shape != image_with_hole.shape:
            print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
            h, w = image_with_hole.shape[:2]
            semantic_repaired = cv2.resize(semantic_repaired, (w, h), interpolation=cv2.INTER_LANCZOS4)

        x, y, w, h = cv2.boundingRect(target_mask)
        cent_h, cent_w = y + h // 2, x + w // 2
        x, y, w, h = cv2.boundingRect(trans_mask)
        repl_trans_mask = np.zeros_like(target_mask)
        repl_trans_img = np.zeros_like(ori_img)
        repl_trans_mask[
        cent_h - h // 2: cent_h - h // 2 + h,
        cent_w - w // 2: cent_w - w // 2 + w,
        ] = trans_mask[y: y + h, x: x + w]
        repl_trans_img[
        cent_h - h // 2: cent_h - h // 2 + h,
        cent_w - w // 2: cent_w - w // 2 + w,
        ] = trans_img[y: y + h, x: x + w]
        repl_trans_mask = (repl_trans_mask > 0).astype(bool)
        final_image = np.where(repl_trans_mask[:, :, None], repl_trans_img, semantic_repaired)

        return final_image, image_with_hole, repl_trans_mask.astype(np.uint8) * 255

    def move_and_inpaint_with_expansion_mask_3D(self, image, mask, depth_map, transforms, FX, FY,
                                                object_only=True,
                                                inpainter=None, mode=None,
                                                dilate_kernel_size=15, inp_prompt=None, target_mask=None,
                                                splatting_radius=0.015,
                                                splatting_tau=0.0, splatting_points_per_pixel=30):

        if isinstance(image, Image.Image):
            image = np.array(image)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        # 将掩码转换为灰度并应用阈值
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask = mask > 128

        if target_mask.ndim == 3 and target_mask.shape[2] == 3:
            target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
            # source_mask =source_mask > 128

        mask = mask.astype(bool)
        target_mask = target_mask.astype(bool)

        transformed_image, transformed_mask = IntegratedP3DTransRasterBlendingFull(image, depth_map, transforms,
                                                                                   FX, FY,
                                                                                   target_mask, object_only,
                                                                                   splatting_radius=splatting_radius,
                                                                                   splatting_tau=splatting_tau,
                                                                                   splatting_points_per_pixel=splatting_points_per_pixel,
                                                                                   return_mask=True,
                                                                                   device=self.device)

        # mask bool
        # MORPH_OPEN transformed target mask to suppress noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        transformed_mask = cv2.morphologyEx(transformed_mask, cv2.MORPH_OPEN, kernel)
        transformed_mask = (transformed_mask > 128).astype(bool)
        # repair_mask = (mask & ~transformed_mask)
        # ori_image_back_ground = np.where(mask[:, :, None], 0, image).astype(np.uint8)
        # new_image = np.where(transformed_mask[:, :, None], transformed_image,
        #                      ori_image_back_ground)  # with repair area to be black
        # image_with_hole = new_image
        # coarse_repaired = inpainter(Image.fromarray(new_image), Image.fromarray(
        #     repair_mask.astype(np.uint8) * 255))  # lama inpainting filling the black regions
        #
        # to_inpaint_img = coarse_repaired
        #
        # if mode == 1:
        #     semantic_repaired = to_inpaint_img
        # elif mode == 2:
        #     inpaint_mask = Image.fromarray(repair_mask.astype(np.uint8) * 255)
        #     print(f'SD inpainting Processing:')
        #     semantic_repaired = \
        #     self.sd_inpainter(prompt=inp_prompt, image=to_inpaint_img, mask_image=inpaint_mask).images[0]
        #
        # if semantic_repaired.size != to_inpaint_img.size:
        #     print(f'inpainted image {semantic_repaired.size} -> original size {to_inpaint_img.size}')
        #     semantic_repaired = semantic_repaired.resize(to_inpaint_img.size)
        # # mask retain in region only repairing
        # retain_mask = ~repair_mask
        # final_image = np.where(retain_mask[:, :, None], semantic_repaired, semantic_repaired)
        # final_image = np.where(retain_mask[:, :, None], coarse_repaired, semantic_repaired)
        ori_image_back_ground = np.where(mask[:, :, None], 0, image).astype(np.uint8)
        image_with_hole = ori_image_back_ground
        coarse_repaired = np.array(inpainter(Image.fromarray(ori_image_back_ground), Image.fromarray(
            mask.astype(np.uint8) * 255)))  # lama inpainting filling the black regions
        if mode != 1:
            inpaint_mask = Image.fromarray(mask.astype(np.uint8) * 255)
            sd_to_inpaint_img = Image.fromarray(coarse_repaired)
            # print(f'SD inpainting Processing:')
            semantic_repaired = \
                self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask).images[0]
            semantic_repaired = np.array(semantic_repaired)
        else:
            semantic_repaired = coarse_repaired

        if semantic_repaired.shape != image_with_hole.shape:
            print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
            h, w = image_with_hole.shape[:2]
            semantic_repaired = cv2.resize(semantic_repaired, (w, h), interpolation=cv2.INTER_LANCZOS4)

        final_image = np.where(transformed_mask[:, :, None], transformed_image, semantic_repaired)

        return final_image, image_with_hole, transformed_mask



    def get_mask_from_rembg(self, trans_img, size=None, need_mask=True):
        if isinstance(trans_img, np.ndarray):
            trans_img = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
            trans_img = Image.fromarray(trans_img)  # PIL img
        if need_mask:
            if size is not None:
                trans_img.thumbnail(size, Image.Resampling.LANCZOS)
                print(f'resized_img shape:{trans_img.size}')

            return_img = np.array(trans_img.copy())
            return_img = cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB)
            trans_img = remove(trans_img.convert("RGBA"), alpha_matting=True)
            image_arr = np.array(trans_img)
            in_w, in_h = image_arr.shape[:2]
            _, trans_mask = cv2.threshold(
                np.array(trans_img.split()[-1]), 128, 255, cv2.THRESH_BINARY
            )
            return trans_mask, in_w, return_img
        else:  # resize pipe
            assert size is not None, "for resize pipe,size should be given"
            trans_img.thumbnail(size, Image.Resampling.LANCZOS)
            return_img = np.array(trans_img.copy())
            return_img = cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB)
            in_w, in_h = return_img.shape[:2]
            return return_img, in_w

    def Magic_Editing_Baseline_SV3D(self, original_image, transformed_image, prompt, INP_prompt,
                                    seed, guidance_scale, num_step, max_resolution, mode, dilate_kernel_size,
                                    start_step,
                                    eta=0, use_mask_expansion=True, contrast_beta=1.67, mask_threshold=0.1,
                                    mask_threshold_target=0.1, end_step=10,
                                    feature_injection=True, FI_range=(900, 680), sim_thr=0.5,
                                    DIFT_LAYER_IDX=[0, 1, 2, 3], use_sdsa=True, select_mask=None,
                                    ):
        seed_everything(seed)

        img = original_image
        trans_img = transformed_image
        if select_mask is None:
            print(f'generating original image mask')
            mask, in_w, img = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution])
            trans_img = cv2.resize(trans_img, (in_w, in_w))
            print(f'generating transformed image mask')
            trans_mask, _, trans_img = self.get_mask_from_rembg(trans_img)
        else:  # resize + mask resize
            print(f'input selected mask : Yes')
            img, in_w = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution], need_mask=False)
            w, h = img.shape[:2]
            mask = cv2.resize(select_mask, (h, w))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            trans_img = cv2.resize(trans_img, (in_w, in_w))
            print(f'generating transformed image mask')
            trans_mask, _, trans_img = self.get_mask_from_rembg(trans_img)

        target_mask = mask
        mask_to_use = self.dilate_mask(mask, dilate_kernel_size)  # dilate for better expansion mask

        # mask expansion
        if use_mask_expansion:
            self.controller.contrast_beta = contrast_beta  # numpy input
            expand_mask, _ = self.DDIM_inversion_func(img=img, mask=mask_to_use, prompt="",
                                                      guidance_scale=1, num_step=10,
                                                      start_step=2,
                                                      roi_expansion=True,
                                                      mask_threshold=mask_threshold,
                                                      post_process='hard', )
        else:
            expand_mask = mask_to_use

        img_preprocess, inpaint_mask_vis, shifted_mask = self.replace_with_SV3D_targets_inpainting(img,
                                                                                                   trans_img,
                                                                                                   expand_mask,
                                                                                                   trans_mask,
                                                                                                   target_mask,
                                                                                                   mode,
                                                                                                   self.inpainter,
                                                                                                   INP_prompt, )
        mask_to_use = shifted_mask
        # mask_to_use = self.dilate_mask(shifted_mask, dilate_kernel_size) * 255  # dilate for better expansion mask
        ori_img = img
        img = img_preprocess
        if isinstance(img, np.ndarray):
            img_preprocess = Image.fromarray(img_preprocess)
        self.controller.contrast_beta = contrast_beta
        expand_shift_mask, inverted_latent = self.DDIM_inversion_func(img=img, mask=mask_to_use,
                                                                      prompt="",
                                                                      guidance_scale=1,
                                                                      num_step=num_step,
                                                                      start_step=start_step,
                                                                      roi_expansion=True,
                                                                      mask_threshold=mask_threshold_target,
                                                                      post_process='hard',
                                                                      use_mask_expansion=False,
                                                                      ref_img=ori_img)  # ndarray mask

        edit_gen_image, refer_gen_image, target_mask, = self.Details_Preserving_regeneration(img,
                                                                                             inverted_latent,
                                                                                             prompt,
                                                                                             expand_shift_mask,
                                                                                             target_mask,
                                                                                             num_steps=num_step,
                                                                                             start_step=start_step,
                                                                                             end_step=end_step,
                                                                                             guidance_scale=guidance_scale,
                                                                                             eta=eta,
                                                                                             roi_expansion=True,
                                                                                             mask_threshold=mask_threshold_target,
                                                                                             post_process='hard',
                                                                                             feature_injection=feature_injection,
                                                                                             FI_range=FI_range,
                                                                                             sim_thr=sim_thr,
                                                                                             dilate_kernel_size=dilate_kernel_size,
                                                                                             DIFT_LAYER_IDX=DIFT_LAYER_IDX,
                                                                                             use_sdsa=use_sdsa,
                                                                                             ref_img=ori_img)
        # source_mask = Image.fromarray(source_mask)
        inpaint_mask_vis = Image.fromarray(inpaint_mask_vis)
        return [edit_gen_image], [refer_gen_image], [img_preprocess], [inpaint_mask_vis], [target_mask]

    def softmax(self,x, temperature=1.0):
        # 计算每个元素的指数值，并进行温度调整
        exp_x = torch.exp(x / temperature)
        # 计算总和以进行归一化
        sum_exp_x = exp_x.sum(dim=-1, keepdim=True)
        # 返回归一化后的结果
        return exp_x / sum_exp_x

    def get_matching_score(self, candidate_sim, pos_len, temperature=0.2):
        """
        carefully designed
        """
        # candidate_sim = self.softmax(candidate_sim, temperature=temperature)
        positive_sim = candidate_sim[:, :pos_len].sum(dim=-1)
        negative_sim = candidate_sim[:, pos_len:]
        # # 对 negative_sim 进行手动实现的 Softmax 锐化处理
        negative_sim_sharpened = self.softmax(negative_sim, temperature=temperature)
        negative_sim_sum = negative_sim_sharpened.max(dim=-1)[0]
        # negative_sim_sum = negative_sim.sum()
        #
        # # 计算 matching_score
        matching_score = 1- negative_sim_sum + positive_sim

        return matching_score

    @torch.no_grad()
    def sd_inpaint_results_filter(self, img_lists, mask, class_text,):
        # [ndarray,ndarray,.....]
        neg_text_list = [class_text] + ['object', 'person', 'texts', ]
        # pos_text_list = ['background','empty scene']
        pos_text_list = ['empty scene']
        pos_len = len(pos_text_list)
        # 通过mask区域的boundary box裁剪图像
        preprocessed_imgs = self.crop_image_with_mask(img_lists, mask)
        # preprocessed_imgs = self.pre_process_with_mask(img_lists, mask)
        img_lst = np.array(img_lists)
        # image = self.clip_process(cropped_img).unsqueeze(0).to(self.device)
        stack_images = torch.stack(
            [self.clip_process(self.numpy_to_pil(crop_img)[0]).to(self.device) for crop_img in
             preprocessed_imgs])

        # 对每个词进行编码
        text_tokens_neg = clip.tokenize(neg_text_list).to(self.device)
        text_tokens_pos = clip.tokenize(pos_text_list).to(self.device)
        text_tokens = torch.cat([text_tokens_pos, text_tokens_neg])

        # 计算图像的特征向量
        image_features = self.clip.encode_image(stack_images)
        # Calculate the standard deviation of the image embeddings to measure semantic diversity
        # embeddings_std = torch.std(image_features[1:], dim=0).mean()
        # #uncertainty filter
        # if embeddings_std.item() >0.20:
        #     # print(f'choose lama inpainting results')
        #     # return img_lists[0]

        # 计算每个词的特征向量
        text_features = self.clip.encode_text(text_tokens)
        similarities = torch.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
        before_similarities = similarities[-1, :]  # ori img sim
        similarities = similarities[:-1, :]
        # class filter
        valid_idx_cls = similarities.argmax(dim=1) < pos_len
        # discrepency filter
        pre_matching_score = self.get_matching_score(before_similarities[None,], pos_len)
        matching_score = self.get_matching_score(similarities, pos_len)
        valid_idx_dis = matching_score > pre_matching_score
        valid_idx_not_black = torch.tensor([False in (p_im[0]==(255,255,255)) for p_im in preprocessed_imgs[:-1]],dtype=valid_idx_dis.dtype,device=self.device)
        valid_idx = valid_idx_cls & valid_idx_dis &valid_idx_not_black
        # valid_idx = valid_idx_dis & valid_idx_not_black
        # if valid_idx.sum() == 0:
        #     print(f'choose lama inpainting results')
        #     return img_lists[0]
        # valid_idx = valid_idx_cls & valid_idx_dis
        valid_index = np.array([idx for idx, i in enumerate(valid_idx) if i])
        try:
            candidate_img = img_lst[valid_index]
            matching_score_valid = matching_score[valid_idx]
            final_match_indices = torch.argmax(matching_score_valid, dim=0).item()
            # if valid_idx[0] and final_match_indices == 0:  # default lama always good, but not real enough
            #     print(f'choose lama inpainting results')
            # else:
            #     print(f'choose sd inpainting results')
            final_inpainting = candidate_img[int(final_match_indices)]
            # self.temp_view_img(final_inpainting)
            return img_lists[0],final_inpainting
        except:
            print(f'no results left after filter, thus choosing lama results')
            return img_lists[0],img_lists[0]

    @torch.no_grad()
    def sd_inpaint_results_caption_filter(self, img_lists, mask, class_text, ):

        def get_matching_score_tag(pos_features, neg_features, target_features):
            sim_pos = torch.matmul(target_features, pos_features.T).mean(dim=-1)
            sim_neg = torch.matmul(target_features, neg_features.T).mean(dim=-1)
            return sim_pos - sim_neg
        # [ndarray,ndarray,.....]
        neg_text_list = [class_text] + ['an object','an item', 'a person', 'texts',]
        pos_text_list = ['a background','an empty_scene','surroundings','environment', 'a scenery' ]
        base_format = 'a photo of'
        neg_text_list = [f'{base_format} {i}' for i in neg_text_list]
        pos_text_list = [f'{base_format} {i}' for i in pos_text_list]

        preprocessed_imgs = self.crop_image_with_mask(img_lists, mask) #crop is enough
        # preprocessed_imgs = self.pre_process_with_mask(img_lists, mask)
        # preprocessed_imgs = self.crop_image_with_mask(preprocessed_imgs,mask)
        img_lst = np.array(img_lists)
        pil_img_lst = \
            [preprocess_tag_pil_img(self.numpy_to_pil(crop_img)[0],self.device) for crop_img in preprocessed_imgs]
        # stack_images = torch.stack(
        #     [self.clip_process(crop_img_pil).to(self.device) for crop_img_pil in pil_img_lst])
        # Tag2Text
        res_tag2text_list = [inference_tag2text(image_pillow, self.tag2text, 'None')[2] for image_pillow in pil_img_lst]
        #TODO:思考一下如何控制tag2text生成文本数量，同时如何对存在object的单词进行惩罚，另外要防止幻觉该怎么做？
        # 对每个词进行编码
        text_feature_neg = self.clip.encode_text(clip.tokenize(neg_text_list).to(self.device))
        text_feature_pos =self.clip.encode_text( clip.tokenize(pos_text_list).to(self.device))
        #normalize and encode
        text_feature_neg = text_feature_neg /text_feature_neg.norm(dim=1, keepdim=True)
        text_feature_pos = text_feature_pos / text_feature_pos.norm(dim=1, keepdim=True)
        text_feature_can = self.clip.encode_text(clip.tokenize(res_tag2text_list).to(self.device))
        text_feature_can = text_feature_can / text_feature_can.norm(dim=1, keepdim=True)
        #TODO: multiple text candidates
        tag_sim_scores = get_matching_score_tag(pos_features=text_feature_pos,neg_features=text_feature_neg,target_features=text_feature_can)
        lama_tag_score = tag_sim_scores[0]
        ori_tag_score = tag_sim_scores[-1]
        sd_tag_scores = tag_sim_scores[1:-1]

        valid_idx_dis = sd_tag_scores > ori_tag_score
        #nsfw remove black results
        valid_idx_not_black = torch.tensor([False in (p_im[0] == (255, 255, 255)) for p_im in preprocessed_imgs[1:-1]],
                                           dtype=valid_idx_dis.dtype, device=self.device)
        valid_idx = valid_idx_not_black & valid_idx_dis
        valid_index = np.array([idx for idx, i in enumerate(valid_idx) if i])
        try:
            candidate_img = img_lst[valid_index]
            matching_score_valid = sd_tag_scores[valid_idx]
            res_tag2text_list_valid = np.array(res_tag2text_list[1:-1])[valid_index]
            final_match_indices = torch.argmax(matching_score_valid, dim=0).item()
            if matching_score_valid[final_match_indices] > max(0.02,lama_tag_score):#strong condition to decide whether reject sd results
                final_inpainting = candidate_img[int(final_match_indices)]
                final_inpainting_caption = res_tag2text_list_valid[int(final_match_indices)]
                print(f'choosing inpainting results with caption:"{final_inpainting_caption}"')
                # self.temp_view_img(final_inpainting)
                return final_inpainting
            else:
                print(f'1:no results left after filter, thus choosing lama results')
                return img_lists[0]
        except:
            print(f'0:no results left after filter, thus choosing lama results')
            return img_lists[0]

    def move_and_inpaint(self, ori_img, exp_mask, dx, dy, inpainter=None, mode=None,
                         inp_prompt=None, inp_prompt_negative=None, obj_text=None, resize_scale=1.0,
                         rotation_angle=0, source_mask=None,
                         flip_horizontal=False, flip_vertical=False):

        # inpaint & replace
        if isinstance(ori_img, Image.Image):
            ori_img = np.array(ori_img)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        if exp_mask.ndim == 3 and exp_mask.shape[2] == 3:
            exp_mask = cv2.cvtColor(exp_mask, cv2.COLOR_BGR2GRAY)
        if source_mask.ndim == 3 and source_mask.shape[2] == 3:
            source_mask = cv2.cvtColor(source_mask, cv2.COLOR_BGR2GRAY)

        # prepare background
        # box mask do not influence sd inpaint
        # exp_mask = self.box_mask(exp_mask)
        ori_exp_mask = exp_mask
        exp_mask = (exp_mask > 0).astype(bool)
        ori_image_back_ground = np.where(exp_mask[:, :, None], 0, ori_img)
        image_with_hole = ori_image_back_ground
        # print(f'ori_shape:{ori_image_back_ground.shape}')
        # coarse_repaired = np.array(inpainter(Image.fromarray(ori_image_back_ground), Image.fromarray(
        #     exp_mask.astype(np.uint8) * 255)))  # lama inpainting filling the black regions

        coarse_repaired = inpainter(ori_image_back_ground, exp_mask.astype(np.uint8) * 255, )

        if mode != 1:
            inpaint_mask = Image.fromarray(exp_mask.astype(np.uint8) * 255)
            sd_to_inpaint_img = Image.fromarray(coarse_repaired)
            # print(f'SD inpainting Processing:')
            semantic_repaired = \
                self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask,
                                  guidance_scale=7.5, eta=1.0,
                                  num_inference_steps=10, negative_prompt=inp_prompt_negative,
                                  num_images_per_prompt=10).images
            # resize
            semantic_repaired_new = []
            h, w = image_with_hole.shape[:2]
            for sd_inp_res in semantic_repaired:
                sd_inp_res = np.array(sd_inp_res)
                if sd_inp_res.shape != image_with_hole.shape:
                    # print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
                    sd_inp_res = cv2.resize(sd_inp_res, (w, h), interpolation=cv2.INTER_LANCZOS4)
                semantic_repaired_new.append(sd_inp_res)
            # Filter
            semantic_repaired_new.insert(0, coarse_repaired)
            semantic_repaired_new.append(ori_img)
            semantic_repaired = self.sd_inpaint_results_filter(semantic_repaired_new, ori_exp_mask, obj_text,
                                                               inp_prompt)
        else:
            semantic_repaired = coarse_repaired

        # Prepare foreground
        height, width = ori_img.shape[:2]
        y_indices, x_indices = np.where(source_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = np.min(y_indices), np.max(y_indices)
            left, right = np.min(x_indices), np.max(x_indices)
            # mask_roi = mask[top:bottom + 1, left:right + 1]
            # image_roi = image[top:bottom + 1, left:right + 1]
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2

        rotation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y), -rotation_angle, resize_scale)
        rotation_matrix[0, 2] += dx
        rotation_matrix[1, 2] += dy

        transformed_image = cv2.warpAffine(ori_img, rotation_matrix, (width, height))
        transformed_mask_exp = cv2.warpAffine(exp_mask.astype(np.uint8), rotation_matrix, (width, height),
                                              flags=cv2.INTER_NEAREST).astype(bool)
        transformed_mask = cv2.warpAffine(source_mask.astype(np.uint8), rotation_matrix, (width, height),
                                          flags=cv2.INTER_NEAREST).astype(bool)

        # 检查是否需要水平翻转
        if flip_horizontal:
            transformed_image = cv2.flip(transformed_image, 1)
            transformed_mask = cv2.flip(transformed_mask, 1)
            transformed_mask_exp = cv2.flip(transformed_mask_exp, 1)

        # 检查是否需要垂直翻转
        if flip_vertical:
            transformed_image = cv2.flip(transformed_image, 0)
            transformed_mask = cv2.flip(transformed_mask, 0)
            transformed_mask_exp = cv2.flip(transformed_mask_exp, 0)

        ddpm_region = transformed_mask_exp * (1 - transformed_mask)
        final_image = np.where(transformed_mask_exp[:, :, None], transformed_image,
                               semantic_repaired)  # move with expansion pixels but inpaint
        return final_image, image_with_hole, semantic_repaired, transformed_mask, ddpm_region

    def Reggio_inpainting_func(self, ori_img, exp_mask, inpainter=None,
                               inp_prompt=None, inp_prompt_negative=None, samples_per_time=10):
        # lama & SD
        if isinstance(ori_img, Image.Image):
            ori_img = np.array(ori_img)
        if inp_prompt is None:
            inp_prompt = 'a photo of a background, a photo of an empty place'
        if exp_mask.ndim == 3 and exp_mask.shape[2] == 3:
            exp_mask = cv2.cvtColor(exp_mask, cv2.COLOR_BGR2GRAY)

        # prepare background
        # box mask do not influence sd inpaint
        # exp_mask = self.box_mask(exp_mask)

        exp_mask = (exp_mask > 0).astype(bool)
        ori_image_back_ground = np.where(exp_mask[:, :, None], 0, ori_img)
        image_with_hole = ori_image_back_ground


        coarse_repaired = inpainter(ori_image_back_ground, exp_mask.astype(np.uint8) * 255, )#lama


        inpaint_mask = Image.fromarray(exp_mask.astype(np.uint8) * 255)
        sd_to_inpaint_img = Image.fromarray(coarse_repaired)
        # print(f'SD inpainting Processing:')
        semantic_repaired = \
            self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask,
                              guidance_scale=7.5, eta=1.0,
                              num_inference_steps=10, negative_prompt=inp_prompt_negative,
                              num_images_per_prompt=samples_per_time,).images
        # resize
        # semantic_repaired_new = [coarse_repaired]
        semantic_repaired_new = []
        h, w = image_with_hole.shape[:2]
        for sd_inp_res in semantic_repaired:
            sd_inp_res = np.array(sd_inp_res)
            if sd_inp_res.shape != image_with_hole.shape:
                # print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
                sd_inp_res = cv2.resize(sd_inp_res, (w, h), interpolation=cv2.INTER_LANCZOS4)
            semantic_repaired_new.append(sd_inp_res)
        return semantic_repaired_new,coarse_repaired
    def expansion_and_inpainting_func(self, img, mask_pools,label_list,max_resolution=768,
                                  seed=42,expansion_step=4, contrast_beta=1.67,max_try_times=10,samples_per_time=10,
                                  assist_prompt=""):
        # seed_everything(seed)
        np.random.seed(int(time.time()))
        random_seed = np.random.randint(0, 2 ** 32 - 1)
        my_seed_everything(random_seed)
        expansion_mask_list = []
        inpainting_results_list = []
        lama_inpaint_results_list = []
        if isinstance(assist_prompt, str):
            assist_prompt = [item.strip() for item in assist_prompt.split(',')]
        #resize to avoid memory promblem in expansion process
        img, in_w = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution], need_mask=False)
        w, h = img.shape[:2]
        expand_forbid_regions = np.zeros_like(mask_pools[0]).astype(np.uint8)
        for i in range(len(mask_pools)):
            msk = mask_pools[i].astype(np.uint8)
            msk[msk>0]=1
            expand_forbid_regions += msk
        expand_forbid_regions[expand_forbid_regions>0]=1

        for idx,select_mask in enumerate(mask_pools):
            obj_text = label_list[idx]
            mask = cv2.resize(select_mask, (h, w))
            if idx==0:
                expand_forbid_regions = cv2.resize(expand_forbid_regions, (h, w))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                    expand_forbid_regions = expand_forbid_regions[:,:,0]
            elif mask.ndim == 3:
                mask = mask[:, :, 0]

            self.controller.contrast_beta = contrast_beta  # numpy input
            print(f'idx:{idx} | proceeding mask expansion:')
            expand_mask = self.Prompt_guided_mask_expansion_func(img=img, mask=mask,expand_forbit_region = expand_forbid_regions,
                                                                 assist_prompt=assist_prompt,
                                                                 num_step=expansion_step, start_step=1, )
            expansion_mask_list.append(expand_mask)
            INP_prompt = 'a photo of a background, a photo of an empty place'
            INP_prompt_negative = f'object,{obj_text},shadow,text'
            inpainting_results_temp = []
            print(f'idx:{idx} | proceeding inpainting:')
            for try_time in range(max_try_times):
                np.random.seed(int(time.time()))
                random_seed = np.random.randint(0, 2 ** 32 - 1)
                my_seed_everything(random_seed)
                inpainting_result,lama_result = self.Reggio_inpainting_func(img, expand_mask, self.inpainter, INP_prompt, INP_prompt_negative, samples_per_time)
                inpainting_results_temp.extend(inpainting_result)
            # Filter
            inpainting_results_temp.insert(0,lama_result)
            inpainting_results_temp.append(img)
            best_sd_inp_result = self.sd_inpaint_results_caption_filter(inpainting_results_temp, expand_mask, obj_text)
            inpainting_results_list.append(best_sd_inp_result)
            lama_inpaint_results_list.append(lama_result)

        return expansion_mask_list,inpainting_results_list,lama_inpaint_results_list

    def preprocess_image(self, image,
                         device):
        image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
        image = rearrange(image, "h w c -> 1 c h w")
        image = image.to(device)
        return image




    def temp_view(self, mask, title='Mask', name=None):
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


    def prepare_controller_ref_mask(self, mask, use_mask_expansion=True):
        # ndarray mask -> Tensor
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = torch.Tensor(mask).to(self.device)
        if use_mask_expansion:
            self.controller.obj_mask = mask
            self.controller.log_mask = True
        return mask


    def gradio_mask_expansion_func(self, img, mask, prompt,
                                   guidance_scale, num_step, eta=0, roi_expansion=True,
                                   mask_threshold=0.1, post_process='hard',
                                   ):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # reference mask prepare
        mask = self.prepare_controller_ref_mask(mask)
        _ = \
            self.MY_DDIM_INV(
                img,
                prompt,
                num_steps=num_step,
                guidance_scale=guidance_scale,
                eta=eta
            )
        expansion_masks = self.controller.expansion_mask_on_the_fly / self.controller.step_num  # expansion mask & average of up mid down resized corresponded self attention maps
        # self.controller.expansion_mask_store = {} #reset for next image
        # step_masks = [v for i, v in expansion_masks.items()]
        candidate_mask = self.fetch_expansion_mask_from_store(expansion_masks, mask, roi_expansion,
                                                              post_process, mask_threshold)
        self.controller.reset()
        return candidate_mask

    @torch.no_grad()
    def ClipWordMatching(self, img, mask, full_prompt):
        nlp = spacy.load("en_core_web_sm")

        # 使用 spaCy 处理句子
        doc = nlp(full_prompt)
        omit_list = ['photo']

        # 提取名词，排除指定的词
        words = [token.text for token in doc if token.pos_ == "NOUN" and token.text not in omit_list]

        # 通过mask区域的boundary box裁剪图像
        cropped_img = self.crop_image_with_mask(img, mask)
        image = self.clip_process(self.numpy_to_pil(cropped_img)[0]).unsqueeze(0).to(self.device)

        # 对每个词进行编码
        text_tokens = clip.tokenize(words).to(self.device)

        # 计算图像的特征向量
        image_features = self.clip.encode_image(image)

        # 计算每个词的特征向量
        text_features = self.clip.encode_text(text_tokens)

        # 计算相似度
        similarities = torch.cosine_similarity(image_features, text_features)

        # 找到最匹配的词的索引
        best_match_index = torch.argmax(similarities).item()

        # 最匹配的词
        best_match_word = words[best_match_index]

        return best_match_word

    def box_mask(self, mask):
        new_mask = np.zeros_like(mask)
        x, y, w, h = cv2.boundingRect(mask)
        new_mask[y:y + h, x:x + w] = 1
        return new_mask

    def pre_process_with_mask(self, img_list, mask):
        processed_imgs = []

        # 1. Apply dilation to the mask
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

        # 2. Blend the edges of the mask using Gaussian blur
        mask_blurred = cv2.GaussianBlur(dilated_mask, (21, 21), 0)
        mask_blurred = mask_blurred / 255.0  # Normalize mask to [0, 1]

        for img_np in img_list:
            # 3. Calculate the average color of the image
            average_color = img_np.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
            avg_color_img = np.ones_like(img_np) * average_color

            # 4. Integrate the masked area with the average color using the blurred mask
            mask_expanded = np.repeat(mask_blurred[:, :, np.newaxis], 3,
                                      axis=2)  # Expand mask to match image channels
            img_np = img_np * mask_expanded + avg_color_img * (1 - mask_expanded)

            # Clip the result to ensure it's within [0, 255] and convert to uint8
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            processed_imgs.append(img_np)

        return processed_imgs

    def pre_process_with_mask_list(self, img, mask_list):
        processed_imgs = []
        for mask in mask_list:
            # 1. Apply dilation to the mask
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

            # 2. Blend the edges of the mask using Gaussian blur
            mask_blurred = cv2.GaussianBlur(dilated_mask, (21, 21), 0)
            mask_blurred = mask_blurred / 255.0  # Normalize mask to [0, 1]

            # 3. Calculate the average color of the image
            average_color = img.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
            avg_color_img = np.ones_like(img) * average_color

            # 4. Integrate the masked area with the average color using the blurred mask
            mask_expanded = np.repeat(mask_blurred[:, :, np.newaxis], 3,
                                      axis=2)  # Expand mask to match image channels
            img_np = img * mask_expanded + avg_color_img * (1 - mask_expanded)

            # Clip the result to ensure it's within [0, 255] and convert to uint8
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            processed_imgs.append(img_np)

        return processed_imgs

    def crop_image_with_mask(self, img, mask):
        # 使用cv2.boundingRect获取mask的边界框
        x, y, w, h = cv2.boundingRect(mask)

        # 裁剪图像
        if isinstance(img, List):
            cropped_img = [im[y:y + h, x:x + w] for im in img]
        else:
            cropped_img = img[y:y + h, x:x + w]
        return cropped_img

    @torch.no_grad()
    def Prompt_guided_mask_expansion_func(self, img, mask, assist_prompt,expand_forbit_region,
                                          num_step, start_step=0,
                                          use_mask_expansion=True,
                                          ):
        source_image = self.preprocess_image(img, self.device)
        # reference mask prepare
        x, y, w, h = cv2.boundingRect(self.dilate_mask(mask, 30))
        local_focus = np.zeros_like(mask)
        local_focus[y: y + h, x: x + w] = 255
        mask = self.prepare_controller_ref_mask(mask, use_mask_expansion)
        local_focus = self.prepare_controller_ref_mask(local_focus, False)
        forbit_exp = self.prepare_controller_ref_mask(expand_forbit_region,False)
        forbit_exp = torch.where(mask.bool(),0,forbit_exp)
        self.controller.local_focus_box = local_focus
        self.controller.forbit_expand_area = forbit_exp
        if len(assist_prompt) == 1 and assist_prompt[0] == "":
            assist_prompt = None
        self.controller.assist_len = len(assist_prompt)
        # ddim inv
        latents = self.Expansion_invert(
            source_image,
            assist_prompt,
            num_inference_steps=num_step,
            num_actual_inference_steps=num_step - start_step,
        )
        del latents
        self.controller.reset()  # reset for next image
        candidate_mask = self.controller.obj_mask.detach().cpu().numpy()
        candidate_mask = self.erode_mask(candidate_mask, 15)
        return candidate_mask

    @torch.no_grad()
    def DDIM_inversion_func(self, img, mask, prompt,
                            num_step, start_step=0, ref_img=None, ):

        source_image = self.preprocess_image(img, self.device)
        if ref_img is not None:
            ref_image = self.preprocess_image(ref_img, self.device)
            source_image = torch.cat((source_image, ref_image))
        # reference mask prepare
        mask = self.prepare_controller_ref_mask(mask, False)

        latents, latents_list = \
            self.invert(
                source_image,
                prompt,
                guidance_scale=1.0,
                num_inference_steps=num_step,
                num_actual_inference_steps=num_step - start_step,
                return_intermediates=True,
            )

        self.controller.reset()  # clear
        return mask.detach().cpu().numpy(), latents_list

    @torch.no_grad()
    def get_mask_center(self, mask):
        y_indices, x_indices = torch.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = torch.min(y_indices), torch.max(y_indices)
            left, right = torch.min(x_indices), torch.max(x_indices)
            mask_center_x, mask_center_y = (right + left) / 2, (top + bottom) / 2
        return mask_center_x.item(), mask_center_y.item()

    def manual_interpolate(self, init_code, non_intersect_mask, union_mask):
        """
        Manually interpolate values for non-intersecting regions using nearby non-intersecting features.

        Parameters:
        init_code (torch.Tensor): The initial code tensor of shape (1, C, H, W).
        non_intersect_mask (torch.Tensor): Mask indicating non-intersecting regions, shape (1, 1, H, W).
        union_mask (torch.Tensor): Mask indicating the union of current and next masks, shape (1, 1, H, W).

        Returns:
        torch.Tensor: The interpolated code tensor.
        """
        batch_size, channels, height, width = init_code.shape
        print(f'processing nums : {non_intersect_mask.sum()}')
        filled_code = init_code.clone()
        start_time = time.time()  # 开始计时
        # Iterate over each pixel in the non_intersect_mask
        for y in range(height):
            for x in range(width):
                if non_intersect_mask[0, 0, y, x] > 0:  # This pixel is in the non_intersect region
                    # Find nearby non-intersecting features for interpolation
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width and union_mask[0, 0, ny, nx] == 0:
                                neighbors.append(init_code[0, :, ny, nx])

                    if neighbors:
                        # Average the features from the neighbors
                        interpolated_value = torch.stack(neighbors, dim=0).mean(dim=0)
                        filled_code[0, :, y, x] = interpolated_value
        end_time = time.time()  # 结束计时
        print(f"Manual interpolation completed in {end_time - start_time:.4f} seconds")
        return filled_code

    def edit_init_code(self, init_code, theta, current_mask, next_mask):
        batch_size, channels, height, width = init_code.shape

        current_mask = F.interpolate(current_mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width),
                                     mode='nearest').squeeze(0).squeeze(0)
        next_mask = F.interpolate(next_mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width),
                                  mode='nearest').squeeze(0).squeeze(0)

        moved_init_code = wrapAffine_tensor(init_code, theta, (width, height), mode='bilinear').unsqueeze(
            0).clone()

        intersection_mask = (current_mask.bool() & next_mask.bool()).float()
        non_intersect_current_mask = (current_mask.bool() & ~intersection_mask.bool()).float()

        kernel_size = 3
        feature_weight = [[0.7, 1.0, 0.7, ],
                          [1.0, 0.0, 1.0, ],
                          [0.7, 1.0, 0.7, ]]
        # feature_weight = [[0.5, 0.8, 0.8, 0.8, 0.5],
        #                   [0.8, 1.0, 1.0, 1.0, 0.8],
        #                   [0.8, 1.0, 0.0, 1.0, 0.8],
        #                   [0.8, 1.0, 1.0, 1.0, 0.8],
        #                   [0.5, 0.8, 0.8, 0.8, 0.5]]

        partial_conv = PartialConvInterpolation(kernel_size, channels, feature_weight).to(self.device)
        inpaint_mask = current_mask.unsqueeze(0).repeat(1, channels, 1, 1)  # 为指定通道重复掩码
        inpaint_mask[inpaint_mask > 0] = 1
        interpolated_original = partial_conv(init_code, 1 - inpaint_mask, non_intersect_current_mask,
                                             2)  # 1 means valid ,0 means to be repaired
        # interpolated_original = tensor_inpaint_fmm(init_code, current_mask)

        next_mask = next_mask.unsqueeze(0).repeat(channels, 1, 1).unsqueeze(0)
        non_intersect_current_mask = non_intersect_current_mask.unsqueeze(0).repeat(channels, 1, 1).unsqueeze(0)
        result_code_0 = torch.where(next_mask > 0, moved_init_code, init_code)
        result_code = torch.where(non_intersect_current_mask > 0, interpolated_original, result_code_0)

        return result_code

    def forward_unet_features_simple(self, z, t, encoder_hidden_states, h_feature=None):
        unet_output, all_intermediate_features, copy_downblock = self.unet(
            z,
            t,
            h_sample=h_feature,
            copy=True,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True
        )
        h_feature = all_intermediate_features[0]

        # h_feature = copy_downblock[2]
        # h_feature = F.interpolate(h_feature, (interp_res_h, interp_res_w), mode='bilinear')

        return unet_output, h_feature

    @torch.no_grad()
    def prepare_DIFT_INJ_IDX(self, DIFT_LATENTS, t, LAYER_IDX=[0, 1, 2, 3], cos_threshold=0.5,
                             use_one_step=False, source_image=None, ref_img=None, ensemble_size=1):
        if not use_one_step:
            # DDIM INV BASED
            edit_prompt = ""
            text_emb = self.get_text_embeddings(edit_prompt).detach()
            text_emb = torch.cat([text_emb, text_emb], dim=0)

            with torch.no_grad():
                _, all_intermediate_features, copy_downblock = self.unet(
                    DIFT_LATENTS,
                    t,
                    h_sample=None,
                    copy=True,
                    encoder_hidden_states=text_emb,
                    return_intermediates=True
                )
            del copy_downblock, _
            feature_map_layers = {}

            for layer in LAYER_IDX:
                feat = all_intermediate_features[layer]
                feature_map_layers[layer] = calculate_cosine_similarity_between_batches(feat,
                                                                                        cos_threshold=cos_threshold)

            self.controller.correspondence_map = feature_map_layers
            del all_intermediate_features, feature_map_layers
        else:
            # ONE STEP ADD NOISE BASED
            assert source_image is not None, "given source img for one step FI"
            assert ref_img is not None, "given ref img for one step FI"
            source_image = self.preprocess_image(source_image, self.device)
            ref_img = self.preprocess_image(ref_img, self.device)
            batch_img_tensor = [source_image, ref_img]
            edit_prompt = ""
            text_emb = self.get_text_embeddings(edit_prompt).detach()  # [1, 77, dim]
            text_emb = text_emb.repeat(ensemble_size, 1, 1)
            feature_map_layers = {}
            unet_ft = []
            for img_tensor in batch_img_tensor:  # 1,C,H,W
                img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w
                with torch.no_grad():
                    latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
                    noise = torch.randn_like(latents).to(self.device)
                    latents_noisy = self.scheduler.add_noise(latents, noise, t)
                    _, all_intermediate_features, copy_downblock = self.unet(
                        latents_noisy,
                        t,
                        h_sample=None,
                        copy=True,
                        encoder_hidden_states=text_emb,
                        return_intermediates=True
                    )
                    unet_ft.append(all_intermediate_features)
                    del copy_downblock, _

            for layer in LAYER_IDX:
                edit_feat = unet_ft[0][layer].mean(0, keepdim=True)
                ref_feat = unet_ft[1][layer].mean(0, keepdim=True)
                feat = torch.cat((edit_feat, ref_feat), dim=0)
                feature_map_layers[layer] = calculate_cosine_similarity_between_batches(feat,
                                                                                        cos_threshold=cos_threshold)

            self.controller.correspondence_map = feature_map_layers
            del all_intermediate_features, feature_map_layers, unet_ft, feat, edit_feat, ref_feat

    def prepare_h_feature(self, init_code, t, edit_prompt, BG_preservation=True, foreground_mask=None, lr=0.01,
                          lam=0.1,
                          eta=0.0, refer_latent=None, h_feature_input=None, ):
        # Encode prompt
        # refer_latent = None
        # h_feature_input = None
        # only need edit_stream h_feature
        uc_text = ""
        # edit_prompt = edit_prompt[0]
        edit_prompt = ""
        text_emb = self.get_text_embeddings(edit_prompt).detach()
        uncon_text_emb = self.get_text_embeddings(uc_text).detach()
        text_emb = torch.cat([uncon_text_emb, text_emb], dim=0)  # cfg text emb
        init_code = torch.cat([init_code, init_code], dim=0)
        if refer_latent is None:
            with torch.no_grad():
                if h_feature_input is not None and not BG_preservation:
                    h_feature = h_feature_input
                else:
                    unet_output, h_feature = self.forward_unet_features_simple(init_code, t,
                                                                               encoder_hidden_states=text_emb,
                                                                               h_feature=h_feature_input)
                    x_prev_0, _ = self.step(unet_output, t, init_code)
        else:
            with torch.no_grad():
                if h_feature_input is not None and not BG_preservation:
                    h_feature = h_feature_input
                else:
                    _, h_feature = self.forward_unet_features_simple(init_code, t,
                                                                     encoder_hidden_states=text_emb,
                                                                     h_feature=h_feature_input)
            x_prev_0 = refer_latent[0].unsqueeze(0)
        if BG_preservation:
            assert foreground_mask is not None, "For BG preservation, foreground mask should be given"
            h_feature.requires_grad_(True)
            optimizer = torch.optim.Adam([h_feature], lr=lr)
            scaler = torch.cuda.amp.GradScaler()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                unet_output, h_feature = self.forward_unet_features_simple(init_code, t,
                                                                           encoder_hidden_states=text_emb,
                                                                           h_feature=h_feature)
                x_prev_updated, _ = self.step(unet_output, t, init_code, None, eta)
                loss = lam * ((x_prev_updated - x_prev_0) * (1.0 - foreground_mask)).abs().sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return h_feature

    @torch.no_grad()
    def prepare_various_mask(self, shifted_mask, ori_mask, sup_res_w, sup_res_h, init_code, ddpm_region):
        shifted_mask_tensor_dilated = self.prepare_tensor_mask(self.dilate_mask(ddpm_region, 15), sup_res_w,
                                                               sup_res_h)
        shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
        ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
        foreground_latent_region = shifted_mask_tensor_dilated + ori_mask_tensor
        foreground_latent_region[foreground_latent_region > 0.0] = 1
        local_variance_reg = (1 - shifted_mask_tensor) * shifted_mask_tensor_dilated
        shifted_mask_tensor[shifted_mask_tensor > 0.0] = 1
        latent_foreground_masks = F.interpolate(foreground_latent_region.unsqueeze(0).unsqueeze(0),
                                                (init_code.shape[2], init_code.shape[3]),
                                                mode='nearest').squeeze(0, 1)
        obj_dilation_mask = F.interpolate(shifted_mask_tensor.unsqueeze(0).unsqueeze(0),
                                          (init_code.shape[2], init_code.shape[3]),
                                          mode='nearest').squeeze(0, 1)
        local_variance_reg = F.interpolate(local_variance_reg.unsqueeze(0).unsqueeze(0),
                                           (init_code.shape[2], init_code.shape[3]),
                                           mode='nearest').squeeze(0, 1)
        return latent_foreground_masks, obj_dilation_mask, local_variance_reg

    def prepare_tensor_mask(self, mask, sup_res_w, sup_res_h):
        # mask interpolation
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_tensor_shifted = torch.tensor(mask, device=self.device)
        mask_tensor_shifted = mask_tensor_shifted.unsqueeze(0).unsqueeze(0)
        transformed_masks_tensor = F.interpolate(mask_tensor_shifted, (sup_res_h, sup_res_w),
                                                 mode="nearest").squeeze(0, 1)
        transformed_masks_tensor[transformed_masks_tensor > 0.0] = 1

        return transformed_masks_tensor

    def Details_Preserving_regeneration(self, source_image, inverted_latents, edit_prompt, shifted_mask,
                                        ori_mask,
                                        num_steps=100, start_step=30, end_step=10,
                                        guidance_scale=3.5, eta=1,
                                        feature_injection=True, FI_range=(900, 680), sim_thr=0.5,
                                        DIFT_LAYER_IDX=[0, 1, 2, 3],
                                        use_mtsa=True, ref_img=None, ddpm_region=None):

        """
        latent vis
        # noised_image = self.decode_latents(start_latents).squeeze(0)
        # # print(noised_image.shape)
        # noised_image = self.numpy_to_pil(noised_image)[0]
        # print(noised_image.size)
        # print(noised_image.shape)
        """

        start_latents = inverted_latents[-1]  # [ori,35 steps latents:50->15]
        init_code_orig = deepcopy(start_latents)
        # PREPARE DIFT INJ IDX
        DIFT_STEP = int(261 / 1000 * num_steps)
        DIFT_latents = inverted_latents[DIFT_STEP]  # time step 261 , 261/1000 = 13/50
        t_dift = self.scheduler.timesteps[num_steps - DIFT_STEP - 1]
        self.prepare_DIFT_INJ_IDX(DIFT_latents, t_dift, DIFT_LAYER_IDX, sim_thr, False, source_image, ref_img,
                                  8)

        # PREPARE H FEATURE
        # t = self.scheduler.timesteps[start_step]
        # unet_feature_idx = [3]

        full_h, full_w = source_image.shape[:2]
        print(f'full_h:{full_h};full_w:{full_w}')
        sup_res_h = int(0.5 * full_h)
        sup_res_w = int(0.5 * full_w)

        foreground_mask, object_mask, local_var_reg = self.prepare_various_mask(shifted_mask, ori_mask,
                                                                                sup_res_w, sup_res_h,
                                                                                init_code_orig, ddpm_region)
        # object_mask = self.dilate_mask(object_mask,15)
        # h_feature = self.prepare_h_feature(init_code_orig,t,edit_prompt,BG_preservation=False,foreground_mask=foreground_mask,lr=0.1,lam=0.1,eta=1.0)

        # noised_optimized_image = self.decode_latents(init_code_orig).squeeze(0)
        # noised_optimized_image = self.numpy_to_pil(noised_optimized_image)[0]

        mask = self.prepare_controller_ref_mask(shifted_mask, True)
        SDSA_REF_MASK = self.prepare_controller_ref_mask(ori_mask, False)
        SDSA_REF_MASK = F.interpolate(SDSA_REF_MASK.unsqueeze(0).unsqueeze(0),
                                      (init_code_orig.shape[2], init_code_orig.shape[3]),
                                      mode='nearest').squeeze(0, 1)

        self.controller.SDSA_REF_MASK = SDSA_REF_MASK
        self.controller.reset()
        # refer_gen_image = edit_ori_image[0] #vis ddim results
        self.controller.log_mask = False
        refer_latents_ori = inverted_latents[::-1]
        edit_gen_image, ref_gen_image = self.forward_sampling_BG(
            prompt=[edit_prompt, ""],
            # refer_latents=refer_latents_list,
            refer_latents=refer_latents_ori,
            end_step=end_step,
            batch_size=2,
            latents=init_code_orig,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            num_actual_inference_steps=num_steps - start_step,
            eta=eta,
            foreground_mask=foreground_mask,
            obj_mask=object_mask,
            local_var_reg=local_var_reg,
            feature_injection_allowed=feature_injection,
            feature_injection_timpstep_range=FI_range,
            use_mtsa=use_mtsa,
        )
        self.controller.reset()
        refer_gen_image = ref_gen_image.permute(1, 2, 0).detach().cpu().numpy()
        edit_gen_image = edit_gen_image.permute(1, 2, 0).detach().cpu().numpy()
        return edit_gen_image, refer_gen_image

    def ReggioRecurrentEdit(self, source_image, inverted_latents, edit_prompt, expand_mask, x_shift, y_shift,
                            resize_scale, rotation_angle, motion_split_steps, num_steps=100, start_step=30,
                            end_step=10, guidance_scale=3.5, eta=0,
                            roi_expansion=True, mask_threshold=0.1, post_process='hard', max_times=10,
                            sim_thr=0.7, lr=0.01, lam=0.1, ):

        """
        latent vis
        # noised_image = self.decode_latents(start_latents).squeeze(0)
        # # print(noised_image.shape)
        # noised_image = self.numpy_to_pil(noised_image)[0]
        # print(noised_image.size)
        # print(noised_image.shape)
        """

        start_latents = inverted_latents[-1]
        init_code_orig = deepcopy(start_latents)
        t = self.scheduler.timesteps[start_step]
        unet_feature_idx = [3]

        full_h, full_w = source_image.shape[:2]
        print(f'full_h:{full_h};full_w:{full_w}')
        sup_res_h = int(0.5 * full_h)
        sup_res_w = int(0.5 * full_w)

        # mask interpolation
        mask_tensor = torch.tensor(expand_mask, device=self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        transformed_masks_tensor = F.interpolate(mask_tensor, (sup_res_h, sup_res_w), mode="nearest")
        x_shift = int(x_shift * 0.5)
        y_shift = int(y_shift * 0.5)

        # TODO: 1-step optimization
        self.controller.log_mask = False
        updated_init_code, h_feature, final_traj_mask = self.Recurrent_diffusion_updates_2D(edit_prompt,
                                                                                            start_latents, t,
                                                                                            transformed_masks_tensor,
                                                                                            x_shift, y_shift,
                                                                                            resize_scale,
                                                                                            rotation_angle,
                                                                                            motion_split_steps,
                                                                                            unet_feature_idx=unet_feature_idx,
                                                                                            lam=lam, lr=lr,
                                                                                            sup_res_w=sup_res_w,
                                                                                            sup_res_h=sup_res_h,
                                                                                            max_times=max_times,
                                                                                            sim_thr=sim_thr)

        with torch.no_grad():
            noised_optimized_image = self.decode_latents(updated_init_code).squeeze(0)
            noised_optimized_image = self.numpy_to_pil(noised_optimized_image)[0]
            # updated_init_code == init_code == start code
            # print(f'init_code == start_code : {torch.all(start_latents == updated_init_code)}')

            mask = self.prepare_controller_ref_mask(final_traj_mask)
            self.controller.expansion_mask_store = {}
            self.controller.log_mask = True

            ori_gen_image, edit_gen_image = self.forward_sampling(
                prompt=edit_prompt,
                h_feature=h_feature,
                end_step=end_step,
                batch_size=2,
                latents=torch.cat([updated_init_code, updated_init_code], dim=0),  # same
                # latents=torch.cat([updated_init_code, updated_init_code], dim=0),
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                num_actual_inference_steps=num_steps - start_step,
            )
            ori_gen_image = self.numpy_to_pil(ori_gen_image.permute(1, 2, 0).detach().cpu().numpy())[0]
            edit_gen_image = self.numpy_to_pil(edit_gen_image.permute(1, 2, 0).detach().cpu().numpy())[0]

            # 从意义上来说，如果ddpm forward 的作用是优化细节，其实mask不如直接用final mask
            # 但是对于需要利用后续先验进一步修复的case，或者循环每一步forward 进行优化的case 都可以重新获得mask
            # 所以先保留这部分代码
            # get ddpm forward expansion target mask
            ddpm_for_expansion_masks = self.controller.expansion_mask_store  # expansion mask & average of up mid down resized corresponded self attention maps
            self.controller.expansion_mask_store = {}  # reset for next image
            candidate_mask = self.fetch_expansion_mask_from_store([ddpm_for_expansion_masks], mask,
                                                                  roi_expansion, post_process, mask_threshold)

        return ori_gen_image, edit_gen_image, noised_optimized_image, candidate_mask

    def wrapAffine_tensor(self, tensor, theta, dsize, mode='bilinear', padding_mode='zeros',
                          align_corners=False,
                          border_value=0):
        """
        对给定的张量进行仿射变换，仿照 cv2.warpAffine 的功能。

        参数：
        - tensor: 要变换的张量，形状为 (C, H, W) 或 (H, W)
        - theta: 2x3 变换矩阵，形状为 (2, 3)
        - dsize: 输出尺寸 (width, height)
        - mode: 插值方法，可选 'bilinear', 'nearest', 'bicubic'
        - padding_mode: 边界填充模式，可选 'zeros', 'border', 'reflection'
        - align_corners: 是否对齐角点，默认为 False
        - border_value: 填充值

        返回：
        - transformed_tensor: 变换后的张量，形状与输入张量相同
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # 生成变换的 grid
        grid = F.affine_grid(theta.unsqueeze(0), [tensor.size(0), tensor.size(1), dsize[1], dsize[0]],
                             align_corners=align_corners)

        # 进行 grid 采样
        output = F.grid_sample(tensor, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        transformed_tensor = output.squeeze(0)

        # 使用填充值填充边界
        if padding_mode == 'zeros' and border_value != 0:
            mask = (grid.abs() > 1).any(dim=-1, keepdim=True)
            transformed_tensor[mask] = border_value

        return transformed_tensor

    def weighted_loss(self, F0_region, F1_region, topk=10):
        """
        Compute the weighted L1 loss where the weights are determined by the cosine similarity.
        The topk positions with the smallest similarity have the highest weights.

        Parameters:
        F0_region (torch.Tensor): Reference feature tensor of shape [batch_size, channels, height, width].
        F1_region (torch.Tensor): Edited feature tensor of shape [batch_size, channels, height, width].
        topk (int): The number of top positions with the smallest similarity to assign higher weights.

        Returns:
        torch.Tensor: The computed weighted L1 loss.
        """
        # Compute L1 loss without reduction
        l1_diff = F.l1_loss(F0_region, F1_region, reduction='none').mean(dim=1)

        # Compute cosine similarity
        similarity = F.cosine_similarity(F0_region, F1_region, dim=1)

        inverse_sim = 1 - similarity
        norm_inv_sim = (inverse_sim - inverse_sim.min()) / (inverse_sim.max() - inverse_sim.min())
        # topk_values, topk_indices = torch.topk(1-similarity_flat, topk, dim=1)  # Use -similarity to get smallest
        #
        # # Create a weight tensor
        # weights = torch.ones_like(similarity_flat)
        #
        # # Assign higher weights to the topk smallest similarity positions
        # for i in range(weights.size(0)):
        #     weights[i, topk_indices[i]] = 10.0  # Assign a higher weight, e.g., 10.0

        # Reshape weights to match the original similarity shape
        weights = norm_inv_sim * 2

        # Apply weights to the L1 loss
        weighted_l1_diff = weights * l1_diff

        # Compute the final weighted loss
        weighted_loss = torch.mean(weighted_l1_diff)

        return weighted_loss

    @torch.no_grad()
    def reach_similarity_condition(self, F1_region, target_feature, threshold):
        # current_feature = F1[:,:,trajectory_current_mask.bool()]
        current_feature = F1_region
        current_feature_flattened = current_feature.view(current_feature.size(0), -1)
        target_feature_flattened = target_feature.view(target_feature.size(0), -1)

        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(current_feature_flattened, target_feature_flattened, dim=1)
        print(f'current_similarity:{cosine_similarity.item()}')
        return cosine_similarity > threshold

    def Recurrent_diffusion_updates_2D(self,
                                       prompt,
                                       init_code,
                                       t,
                                       masks_tensor,
                                       x_shift, y_shift, resize_scale, rotation_angle,
                                       motion_split_steps,
                                       unet_feature_idx=[3],
                                       sup_res_h=256,
                                       sup_res_w=256,
                                       lam=0.1, lr=0.01, max_times=10, sim_thr=0.92):
        # iteratively optimize in one step and refine serveral times for each step to assure succefully edit each step
        print(f'mask_tensor.shape:{masks_tensor.shape}')
        # Encode prompt
        text_emb = self.get_text_embeddings(prompt).detach()
        with torch.no_grad():
            unet_output, F0, h_feature = self.forward_unet_features(init_code, t,
                                                                    encoder_hidden_states=text_emb,
                                                                    layer_idx=unet_feature_idx,
                                                                    interp_res_h=sup_res_h,
                                                                    interp_res_w=sup_res_w)
            x_prev_0, _ = self.step(unet_output, t, init_code)

        # prepare for point tracking and background regularization
        trajectory_masks_sup = copy.deepcopy(masks_tensor).squeeze(0, 1)

        height, width = trajectory_masks_sup.shape[:2]
        rotation_angle_step = rotation_angle / motion_split_steps
        resize_scale_step = resize_scale ** (1 / motion_split_steps)
        dx = x_shift / motion_split_steps
        dy = y_shift / motion_split_steps

        # h_features = []

        trajectory_current_mask = trajectory_masks_sup
        current_feature_like = F0
        # return init_code, h_feature, trajectory_masks_sup

        step_idx = 0
        try_times = 1
        while step_idx <= motion_split_steps:
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                # copy_h = copy.deepcopy(h_feature)
                # h_features.append(copy_h)

                if (step_idx == 0) or (try_times > max_times) or self.reach_similarity_condition(F1_region,
                                                                                                 F0_region,
                                                                                                 sim_thr):
                    # if init or last step finish -> move on to next step
                    step_idx += 1
                    try_times = 1
                    if step_idx > motion_split_steps:
                        break
                    # copy_h = copy.deepcopy(h_feature)
                    # h_features.append(copy_h)
                    # get mask center
                    mask_center_x, mask_center_y = self.get_mask_center(
                        trajectory_current_mask)  # relocate mask center every step,
                    # allowing tracking real mask center update

                    # get transformation matrix
                    transformation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y),
                                                                    -rotation_angle_step,
                                                                    resize_scale_step)
                    transformation_matrix[0, 2] += dx
                    transformation_matrix[1, 2] += dy
                    # cv2 rotation matrix -> Tensor adaptive transformation theta
                    theta = torch.tensor(param2theta(transformation_matrix, width, height), dtype=F0.dtype,
                                         device=self.device)
                    trajectory_current_mask = trajectory_current_mask.to(F0.dtype)
                    trajectory_next_mask = \
                    wrapAffine_tensor(trajectory_current_mask, theta, (width, height), mode='nearest')[0]
                    foreground_latent_region = trajectory_next_mask + trajectory_current_mask
                    foreground_latent_region[foreground_latent_region > 0.0] = 1
                    latent_trajectory_masks = F.interpolate(foreground_latent_region.unsqueeze(0).unsqueeze(0),
                                                            (init_code.shape[2], init_code.shape[3]),
                                                            mode='nearest').squeeze(0, 1)
                    # init_code = self.edit_init_code(init_code, theta, trajectory_current_mask, trajectory_next_mask)
                    # h_feature = self.edit_init_code(h_feature, theta, trajectory_current_mask, trajectory_next_mask)
                    h_feature = h_feature.detach().requires_grad_(True)
                    # return init_code, h_feature, trajectory_next_mask
                    # prepare optimizable init_code and optimizer
                    # h_feature.requires_grad_(True)
                    optimizer = torch.optim.Adam([h_feature], lr=lr)
                    # prepare amp scaler for mixed-precision training
                    scaler = torch.cuda.amp.GradScaler()

                    unet_output, F1, h_feature = self.forward_unet_features(init_code, t,
                                                                            encoder_hidden_states=text_emb,
                                                                            h_feature=h_feature,
                                                                            layer_idx=unet_feature_idx,
                                                                            interp_res_h=sup_res_h,
                                                                            interp_res_w=sup_res_w)
                    x_prev_updated, _ = self.step(unet_output, t, init_code)

                    trajectory_next_feature_like = wrapAffine_tensor(current_feature_like, theta,
                                                                     (width, height),
                                                                     mode='bilinear').unsqueeze(0)

                    F0_region = trajectory_next_feature_like[:, :,
                                trajectory_next_mask.bool()].detach()  # F0_region update
                    F1_region = F1[:, :, trajectory_next_mask.bool()]
                    trajectory_current_mask = trajectory_next_mask
                    current_feature_like = trajectory_next_feature_like
                else:
                    unet_output, F1, h_feature = self.forward_unet_features(init_code, t,
                                                                            encoder_hidden_states=text_emb,
                                                                            h_feature=h_feature,
                                                                            layer_idx=unet_feature_idx,
                                                                            interp_res_h=sup_res_h,
                                                                            interp_res_w=sup_res_w)
                    x_prev_updated, _ = self.step(unet_output, t, init_code)
                    # recurrent refine on this step
                    try_times += 1
                    F1_region = F1[:, :, trajectory_current_mask.bool()]  # reattain F1_region from new F1

                # update every step every try
                assert F0_region.shape == F1_region.shape, "The shapes of F0_region and F1_region do not match for L1 loss calculation."
                edit_loss = F.l1_loss(F0_region, F1_region)
                # edit_loss = self.weighted_loss(F0_region,F1_region,topk=1000)
                BG_loss = lam * ((x_prev_updated - x_prev_0) * (1.0 - latent_trajectory_masks)).abs().sum()
                loss = edit_loss + BG_loss
                print(f'handling step:{step_idx}/{motion_split_steps} , trying times:{try_times}')
                print(f'Edit_loss: {edit_loss.item()},BG_loss: {BG_loss.item()} Total_loss: {loss.item()}')
                # print(f'BG_loss: {BG_loss.item()}')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # return init_code, h_feature, h_features
        # return init_code, h_feature, h_features, trajectory_current_mask
        return init_code, h_feature, trajectory_current_mask







