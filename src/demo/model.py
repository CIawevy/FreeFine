from src.models.dragondiff import DragonPipeline
from src.utils.utils import resize_numpy_image, split_ldm, process_move, process_drag_face, process_drag, process_appearance, process_paste
from src.utils.inversion import DDIMInversion
import os
import cv2
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

NUM_DDIM_STEPS = 50
SIZES = {
    0:4,
    1:2,
    2:1,
    3:1,
}

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
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
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
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

    def move_and_inpaint_with_expansion_mask_new(self, image, mask, dx, dy, inpainter=None, mode=None,
                                             dilate_kernel_size=15, inp_prompt=None,
                                             resize_scale=1.0, rotation_angle=0,target_mask=None,flip_horizontal=False,flip_vertical=False):

        if isinstance(image, Image.Image):
            image = np.array(image)
        if inp_prompt is None:
            inp_prompt = ""
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


        rotation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_x), -rotation_angle, resize_scale)
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
        repair_region = ((mask | transformed_mask) & ~transformed_target)
        repair_mask = repair_region.astype(np.uint8) * 255  # (2)
        # repair_mask = self.dilate_mask(repair_mask, dilate_factor=dilate_kernel_size)
        image_with_hole = np.where(repair_mask[:, :, None], 0, new_image).astype(np.uint8)  # for visualization use
        to_inpaint_img = new_image
        # elif mode == 1:
        #     repair_mask = (mask.astype(np.uint8) * 255)  #(1)
        #     repair_mask = self.dilate_mask(repair_mask, dilate_factor=dilate_kernel_size)
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

        if inpainted_image.size != to_inpaint_img.size:
            print(f'inpainted image {inpainted_image.size} -> original size {to_inpaint_img.size}')
            inpainted_image = inpainted_image.resize(to_inpaint_img.size)
            retain_mask = ~repair_region
            inpainted_image = np.where(retain_mask[:, :, None], new_image, inpainted_image)

        # if mode==1:
        #     final_image = np.where(shifted_mask[:, :, None], new_image, inpainted_image)
        # elif mode==2:
        final_image = inpainted_image
        #mask retain
        retain_mask =  dict()
        retain_mask['obj_region'] = transformed_target
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
    def run_my_Baseline_full(self, original_image, mask, prompt,INP_prompt,
                        seed, selected_points, guidance_scale, num_step, max_resolution, mode, dilate_kernel_size, start_step, mask_ref = None, eta=0, use_mask_expansion=True,
                        standard_drawing=True,contrast_beta=1.67,exp_mask_type=0,resize_scale=1.0,rotation_angle=None,strong_inpaint=False,flip_horizontal=False, flip_vertical=False
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
        if standard_drawing:
            mask_to_use = mask
            if input_scale != 1:
                mask_to_use, _ = resize_numpy_image(mask_to_use, max_resolution * max_resolution)
        else:
            mask_to_use = mask_ref
            if len(mask_ref) > 1:
                mask_to_use, _ = resize_numpy_image(mask_to_use, max_resolution * max_resolution)
                if strong_inpaint:
                    target_mask, _ = resize_numpy_image(mask, max_resolution * max_resolution)

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
        if use_mask_expansion:
            self.controller.contrast_beta = contrast_beta
            expand_mask= self.gradio_mask_expansion_func(img=img, mask=mask_to_use, prompt="",
                                                         guidance_scale=guidance_scale, num_step=num_step,
                                                         eta=eta,roi_expansion=True,
                                                         mask_threshold=0.1,post_process='hard',) #ndarray mask
        else:
            expand_mask = mask_to_use

        # copy-paste-inpaint to get initial img
        if strong_inpaint:
            img_preprocess, inpaint_mask, shifted_mask,retain_mask = self.move_and_inpaint_with_expansion_mask_new(img, expand_mask, dx, dy, self.inpainter,mode,dilate_kernel_size,INP_prompt, resize_scale,rotation_angle,target_mask,flip_horizontal,flip_vertical)
        else:
            img_preprocess,inpaint_mask,shifted_mask = self.move_and_inpaint_with_expansion_mask(img, expand_mask, dx, dy,self.inpainter,mode,dilate_kernel_size,INP_prompt,resize_scale,rotation_angle,flip_horizontal,flip_vertical)
            retain_mask = None
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
                roi_expansion=True,mask_threshold=0.1, post_process='hard',mask=shifted_mask,target_mask_type=target_mask_type,
                retain_mask = retain_mask, dilate_kernel_size=dilate_kernel_size,
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
            #TODO: fetch self attn map in Unet process
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

    def DDIM_DDPM_MASK(self, input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5,
             version=0, eta=0, roi_expansion=True,mask_threshold=0.1, post_process='hard',mask=None,target_mask_type='FOR',retain_mask=None,dilate_kernel_size=15):
        ##input_image original PIL 512,512
        # VERSION ONE : DDIM Inversion official code in huggingface
        # VERSION TWO : DDIM inversion code in dragon utils
        assert target_mask_type in ['INV','FOR','BOTH'],f'{target_mask_type} is not implemented,please check'
        with torch.no_grad():
            if version == 0:
                latent = self.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(self.device) * 2 - 1)
                l = 0.18215 * latent.latent_dist.sample()
                inverted_latents = self.invert(l, input_image_prompt, num_inference_steps=num_steps,
                                               guidance_scale=guidance_scale, )

            # elif version == 1:
            #     # Dragon diffusion ddim inversion code,but with bug in self.unet forward function while they use diffenrent unet
            #     # bother to check ,just use official ddim inversion code with guidance
            #     img_tensor = (PILToTensor()(input_image) / 255.0 - 0.5) * 2
            #     img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
            #     latent = self.image2latent(img_tensor)
            #     inverted_latents = self.ddim_inv(latent=latent, prompt=input_image_prompt, ddim_num_steps=num_steps)

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

            if retain_mask is not None:
                # blending with CPI img to retain consistent part

                target_expansion = (candidate_mask>128).astype(bool)
                source_expansion = retain_mask['ori_expansion']
                before_obj_region = retain_mask['obj_region']

                retain_region_mask = ~(~(source_expansion | target_expansion) | before_obj_region)
                # print(retain_region_mask)
                self.view(retain_region_mask.astype(np.uint8) * 255, name="/data/Hszhu/DragonDiffusion/retain_region_mask")

                #blend with dilation like BrushNet
                retain_region = self.dilate_mask(retain_region_mask.astype(np.uint8), dilate_factor=dilate_kernel_size)[:,:,None]
                # retain_region = retain_region_mask.astype(np.uint8)[:,:,None]
                # print(retain_region.shape)

                final_im = (1-retain_region)* input_image + retain_region * final_im

                return final_im, noised_image, candidate_mask, retain_region[:,:,0] * 255




        return final_im, noised_image, candidate_mask ,None

    def MY_DDIM_INV(self, input_image, input_image_prompt, num_steps=100, guidance_scale=3.5,
              eta=0):
        ##input_image original PIL 512,512
        # VERSION ONE : DDIM Inversion official code in huggingface
        # VERSION TWO : DDIM inversion code in dragon utils
        with torch.no_grad():
            latent = self.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(self.device) * 2 - 1)
            l = 0.18215 * latent.latent_dist.sample()
            inverted_latents = self.invert_with_attn_map_store(l, input_image_prompt, num_inference_steps=num_steps,
                                           guidance_scale=guidance_scale, ) #same with original function attention store was used in hook format
        return inverted_latents
    @torch.no_grad()
    def invert2(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        from fpe code
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        # exit()
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

        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents




