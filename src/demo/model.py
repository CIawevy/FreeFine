from PIL.Image import blend
from diffusers.utils.torch_utils import  randn_tensor
import sys
sys.path.append('/data/Hszhu/Reggio')
import os
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import cv2
import random
from pytorch_lightning import seed_everything
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from src.utils.attention import override_forward
from rembg import remove
from typing import Optional
from pytorch_lightning.utilities import rank_zero_warn
from diffusers import DDIMScheduler
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
class Latent2RGBPreviewer:
    """
    User-diy code for SD- Intermediate Visulaization --> GIF
    """
    def __init__(self, latent_rgb_factors, latent_rgb_factors_bias=None):
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = None
        if latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        if x0.ndim == 5:
            x0 = x0[0, :, 0]
        else:
            x0 = x0[0]

        latent_image = torch.nn.functional.linear(x0.movedim(0, -1), self.latent_rgb_factors,
                                                  bias=self.latent_rgb_factors_bias)
        return self.preview_to_image(latent_image)

    def preview_to_image(self, latent_image):
        latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                         .mul(0xFF)  # to 0..255
                         ).to(device="cpu", dtype=torch.uint8)
        return Image.fromarray(latents_ubyte.numpy())
class FreeFine():
    def __init__(self, pretrained_model_path='/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"', device=None):
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # base_sd path
        precision = torch.float16
        model = FreeFinePipeline.from_pretrained(pretrained_model_path, torch_dtype=precision).to(device)
        model.scheduler = DDIMScheduler.from_config(model.scheduler.config,)
        self.model = model
    def run_remove(self):
        pass
    def run_edit(self):
        pass
    def run_compose(self):
        pass
class FreeFinePipeline(StableDiffusionPipeline):

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
        variance = self._get_variance(timestep, prev_timestep).to(
        device=self.device,
        dtype=model_output.dtype  # 显式指定类型
    )
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

    # @torch.no_grad()
    # def image2latent(self, image):
    #     DEVICE = self.device
    #     if type(image) is Image:
    #         image = np.array(image)
    #         image = torch.from_numpy(image).float() / 127.5 - 1
    #         image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    #     # input image density range [-1, 1]
    #     latents = self.vae.encode(image)['latent_dist'].mean
    #     latents = latents * 0.18215
    #     return latents

    @torch.no_grad()
    def image2latent(self, image):
        """Convert input image to latents with strict type and shape checking"""
        # --- 参数校验 ---
        assert isinstance(image, (Image.Image, np.ndarray, torch.Tensor)), \
            f"Unsupported input type: {type(image)}"

        # --- 设备与类型配置 ---
        target_device = self.device
        target_dtype = next(self.vae.parameters()).dtype
        # print(f'type:{target_dtype}')
        # --- 统一转换流程 ---
        if isinstance(image, Image.Image):
            # PIL处理（保持原逻辑）
            image = np.array(image).astype(np.float32)
            image = image / 127.5 - 1.0  # [-1,1] 归一化
            tensor = torch.from_numpy(image)
        elif isinstance(image, np.ndarray):
            # numpy处理（显式归一化）
            assert image.dtype in [np.float32, np.uint8], \
                "Numpy array must be float32 [0,255] or uint8 [0,255]"
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0 * 2 - 1  # uint8->[-1,1]
            tensor = torch.from_numpy(image)
        else:  # torch.Tensor
            # Tensor处理（强制归一化）
            assert image.dtype in [torch.float32, torch.float16], \
                "Tensor must be float32/float16"
            if image.max() > 1.0 or image.min() < -1.0:
                print(f"[WARN] Tensor value range [{image.min():.2f}, {image.max():.2f}]")
            tensor = image.clone().detach()

        # --- 维度校验（保持原逻辑）---
        if tensor.ndim == 3:  # (H,W,C)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # -> (1,C,H,W)
        elif tensor.ndim == 4:  # (B,C,H,W)
            pass  # 保持原状
        else:
            raise ValueError(f"Invalid tensor shape: {tensor.shape}")

        # --- 类型对齐（关键修改）---
        tensor = tensor.to(device=target_device, dtype=target_dtype)

        # --- VAE编码（保持原逻辑）---
        latents = self.vae.encode(tensor)['latent_dist'].mean
        return latents * 0.18215  # 缩放因子保持一致
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


    def forward_sampling_compose( #Simple no text version
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
            end_scale=0.5,
            local_var_reg=None,
            local_edit_text=True,cfg_masks_tensor=None,
            share_attn=True,method_type=None,verbose=False,local_perturbation=True,
            **kwds):
        DEVICE = self.device
        self.method_type = method_type
        # assert not local_edit_text,'currently not support local edit text for img compositions'
        assert guidance_scale > 1.0, 'USING THIS MODULE CFG Must > 1.0'
        if share_attn:
            if self.method_type == 'tca':
                self.controller.use_tca = True
                self.controller.layer_idx = list(range(10, 16)) # for tca , follow mtsa start layer = 10 and only in decoder layer for SD-V15
                self.controller.method = 'tca'

            elif self.method_type =='mmsa' or self.method_type =='mmsa_es':# for mMsa , start layer = 10 and only in decoder layer for SD-V15
                self.controller.use_tca = True
                self.controller.layer_idx = list(range(10, 16))
                self.controller.method = 'mmsa'

            elif self.method_type == 'ssa':
                self.controller.use_style_align = True
                self.controller.method = 'ssa'

            elif self.method_type =='sdsa': # for sdsa use all the layers
                self.controller.use_style_align = True
                self.controller.method = 'sdsa'




        self.controller.use_cfg = True
        self.controller.local_edit = local_edit_text   #allow local structure guidance

        prompt.append("")
        self.controller.prompt_length = len(prompt)
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]

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
        if not verbose:
            print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        start_step = num_inference_steps - num_actual_inference_steps

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue
            timestep = t.detach().item()

            ref_latent = refer_latents[i - start_step + 1][1:]
            if latents.shape[0]>1:
                latents[1:] = ref_latent
            else:
                latents = torch.cat([latents,ref_latent])
            if self.method_type=='tca':
                self.controller.context_guidance = self.linear_param(i,start_step,end_step,num_inference_steps,end_scale=end_scale)
            elif self.method_type=='mmsa_es':
                if i >= end_step:
                    self.controller.use_tca = False

            with torch.no_grad():
                # model_inputs = torch.cat([latents] * 2)
                model_inputs = torch.cat([latents,latents[0][None,:,:,:]] ) #[edit ref ref ref ,con_edit]
                # h_feature_inputs = torch.cat([h_feature] * 2)

                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat(
                        [unconditioning[i].expand(*text_embeddings.shape), text_embeddings])

                self.controller.log_mask = False
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                noise_pred_uncon, noise_pred_con = noise_pred[0][None,:,:,:],noise_pred[-1][None,:,:,:]

                if not local_edit_text:
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                else:
                    local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * cfg_masks_tensor
                    noise_pred = noise_pred_uncon + local_text_guidance

                full_mask = torch.ones_like(local_var_reg)
                if not local_perturbation:
                    latents = self.ctrl_step(noise_pred, t, latents[0][None,:,:,:], full_mask, eta=eta)[0]
                else:
                    latents = self.ctrl_step(noise_pred, t, latents[0][None,:,:,:], local_var_reg, eta=eta)[0]
                latents_list.append(latents[0])
        image = self.latent2image(latents, return_type="pt")[0]
        if return_intermediates:
            return image, latents_list
        return image, None


    def linear_param(self,t, t1, t0, t2,end_scale=0.5):
        """
        输入 t 返回 h，满足：
        - h(t1) = 1
        - h(t0) = 0.5
        - h(t2) = 0
        """
        if t < t1 or t > t2:
            raise ValueError(f"t must be in [{t1}, {t2}]")

        if t <= t0:
            # 第一段：t1到t0，h从1降到0.5
            slope = (end_scale-1) / (t0 - t1)
            return 1 + slope * (t - t1)
        else:
            # 第二段：t0到t2，h从0.5降到0
            slope = -end_scale / (t2 - t0)
            return end_scale + slope * (t - t0)

    def proximal_guidance(
        self,
        t,
        latents,
        mask_edit,
        dtype,
        local_var_reg,
        prox_guidance=False,
        recon_t=400,
        recon_end=0,
        recon_lr=0.1,
        target_latent=None
    ):
        if mask_edit is not None and prox_guidance and (recon_t > recon_end and t < recon_t) or (recon_t < -recon_end and t > -recon_t):
            fix_mask = deepcopy(local_var_reg.repeat(1,4,1,1))
            mask_edit[0] = (mask_edit[0] + fix_mask).clamp(0, 1)
            recon_mask = 1 - mask_edit
            latents = latents - recon_lr * (latents - target_latent) * recon_mask
        return latents.to(dtype)
    def forward_sampling(
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
            end_scale=0.5,
            local_var_reg=None,
            completion_mask_cfg=None,
            local_edit_text=True,
            share_attn=True,method_type=None,verbose=False,local_perturbation=True,
            **kwds):
        DEVICE = self.device
        self.method_type = method_type

        assert guidance_scale > 1.0, 'USING THIS MODULE CFG Must > 1.0'
        if share_attn:
            if self.method_type == 'tca':
                self.controller.use_tca = True
                self.controller.layer_idx = list(range(10, 16))  # for tca , follow mmsa start layer = 10 and only in decoder layer SDV15
                self.controller.method = 'tca'

            elif self.method_type =='mmsa' or self.method_type =='mmsa_es':# for mMsa , start layer = 10 and only in decoder layer sdv15
                self.controller.use_tca = True
                self.controller.layer_idx = list(range(10, 16))
                self.controller.method = 'mmsa'

            elif self.method_type == 'ssa':
                self.controller.use_style_align = True
                self.controller.method = 'ssa'

            elif self.method_type =='sdsa': # for sdsa use all the layers
                self.controller.use_style_align = True
                self.controller.method = 'sdsa'




        self.controller.use_cfg = True
        self.controller.local_edit = local_edit_text   #allow local content specific generation attn

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
        if not verbose:
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
        if not verbose:
            print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        start_step = num_inference_steps - num_actual_inference_steps
        #

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue
            timestep = t.detach().item()

            ref_latent = refer_latents[i - start_step + 1][1]
            if latents.shape[0] > 1:
                latents[1:] = ref_latent
            else:
                latents = torch.cat([latents, ref_latent])
            if self.method_type=='tca':
                self.controller.context_guidance = self.linear_param(i,start_step,end_step,num_inference_steps,end_scale=end_scale)
            elif self.method_type=='mmsa_es':
                if i >= end_step:
                    self.controller.use_tca = False

            with torch.no_grad():
                model_inputs = torch.cat([latents] * 2)
                # h_feature_inputs = torch.cat([h_feature] * 2)

                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat(
                        [unconditioning[i].expand(*text_embeddings.shape), text_embeddings])

                self.controller.log_mask = False
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)

                if not local_edit_text:
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                else:
                    local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * completion_mask_cfg
                    noise_pred = noise_pred_uncon + local_text_guidance

                full_mask = torch.ones_like(local_var_reg)
                if not local_perturbation:
                    latents = self.ctrl_step(noise_pred, t, latents, full_mask, eta=eta)[0]
                else:
                    latents = self.ctrl_step(noise_pred, t, latents, local_var_reg, eta=eta)[0]
                latents_list.append(latents)
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image, None

    def prox_regularization(
            self,
            noise_pred_uncond,
            noise_pred_text,
            t,
            quantile=0.75,
            recon_t=400,
            dilate_radius=2,
    ):
        def dilate(image, kernel_size, stride=1, padding=0):
            """
            Perform dilation on a binary image using a square kernel.
            """
            # Ensure the image is binary
            assert image.max() <= 1 and image.min() >= 0

            # Get the maximum value in each neighborhood
            dilated_image = F.max_pool2d(image, kernel_size, stride, padding)

            return dilated_image
        score_delta = (noise_pred_text - noise_pred_uncond).float()
        threshold = score_delta.abs().quantile(quantile)
        if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
            mask_edit = (score_delta.abs() > threshold).float()
            if dilate_radius > 0:
                radius = int(dilate_radius)
                mask_edit = dilate(mask_edit.float(), kernel_size=2 * radius + 1, padding=radius)
        else:
            mask_edit = None
        return mask_edit


    def forward_sampling_background_gen(
            self,
            prompt,
            batch_size=1,
            end_step=None,
            height=512,
            width=512,
            num_inference_steps=50,
            num_actual_inference_steps=None,
            guidance_scale=7.5,
            latents=None,
            refer_latents=None,
            unconditioning=None,
            neg_prompt=None,
            return_intermediates=False,
            eta=0.0,
            local_var_reg=None,
            local_cfg_reg=None,
            local_text_edit=True,
            share_attn=True,method_type='tca',verbose=False,local_perturbation=True,end_scale=0.5,
            latent_blended=True, blend_range=(0, 40),
            **kwds):

        DEVICE = self.device
        assert guidance_scale > 1.0, 'USING THIS MODULE CFG Must > 1.0'
        self.method_type = method_type
        # self.controller.context_guidance = context_guidance
        if share_attn:
            if self.method_type == 'tca':
                self.controller.use_tca = True
                self.controller.layer_idx = list(range(10, 16))   # for tca , follow mmsa start layer = 10 and only in decoder layer for base sd-v1-5
                self.controller.method ='tca'

            elif self.method_type == 'mmsa' or method_type=='mmsa_es':
                self.controller.use_tca = True
                self.controller.layer_idx = list(
                    range(10, 16)) # for mmsa , start layer = 10 and only in decoder layer  for base sd-v1-5
                self.controller.method = 'mmsa'

            elif self.method_type == 'ssa':
                self.controller.use_style_align = True
                self.controller.method = 'ssa'


            elif self.method_type == 'sdsa':  # for sdsa use all the layers
                self.controller.use_style_align = True
                self.controller.method = 'sdsa'


        self.controller.use_cfg = True
        self.controller.local_edit = local_text_edit  # allow local cfg


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
        if not verbose:
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
        if not verbose:
            print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        start_step = num_inference_steps - num_actual_inference_steps

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue
            ref_latent = refer_latents[i - start_step]
            if latents.shape[0] > 1:
                latents = latents[0].unsqueeze(0)

            latents= torch.cat([latents,ref_latent],dim=0)
            if self.method_type == 'tca':
                self.controller.context_guidance = self.linear_param(i, start_step, end_step, num_inference_steps,end_scale=end_scale)
            elif self.method_type =='mmsa_es':
                if i >= end_step:
                    self.controller.use_tca = False

            with torch.no_grad():
                model_inputs = torch.cat([latents] * 2)
                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat(
                        [unconditioning[i].expand(*text_embeddings.shape), text_embeddings])

                self.controller.log_mask = False

                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                # mask_edit = self.prox_regularization(noise_pred_uncon,noise_pred_con,t)
                # if i < end_step:
                if not local_text_edit:
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                else:
                    local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * local_cfg_reg
                    noise_pred = noise_pred_uncon + local_text_guidance

                full_mask = torch.ones_like(local_var_reg,dtype=noise_pred.dtype)

                if not local_perturbation:
                    latents = self.ctrl_step(noise_pred, t, latents, full_mask, eta=eta)[0]
                else:
                    latents = self.ctrl_step(noise_pred, t, latents, local_var_reg, eta=eta)[0]
                # print(f'noise_pred:{latents.dtype}')
                # Proximal and blending tech
                # latents = self.proximal_guidance(
                #     t,
                #     latents,
                #     mask_edit,
                #     self.unet.dtype,
                #     local_var_reg,
                #     prox_guidance=True,
                #     target_latent=refer_latents[i - start_step+1],
                # )
                #
                # if latent_blended and blend_range[0]<= i <= blend_range[1]:
                #     latents[0] = local_var_reg.repeat(1, 4, 1, 1) * latents[0]+ (1 - local_var_reg.repeat(1, 4, 1, 1)) * latents[1]
                # Proximal and blending tech
                latents_list.append(latents[0])
        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image, None



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
            verbose=False,
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
        if not verbose:
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
        if not verbose:
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
            if not verbose:
                print(f'leng latent inv {len(latents_list)}')
                print(f'shape latent:{latents.shape}')
            if return_intermediates:
                # return the intermediate laters during inversion
                # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
                return latents, latents_list
            return latents
        else:
            for i, t in enumerate(reversed(self.scheduler.timesteps)):
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
            if not verbose:
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


    def temp_view_img(self, image: Image.Image, title: str = None) -> None:
        # 如果输入不是 PIL 图像，假设是 ndarray
        if not isinstance(image, Image.Image):
            image_array = image
        else:  # 如果是 PIL 图像，转换为 ndarray
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)

        # 去除图像白边
        def remove_white_border(image_array):
            # 转换为灰度图
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 240 以上视为白色
            coords = cv2.findNonZero(thresh)  # 非零像素坐标
            x, y, w, h = cv2.boundingRect(coords)  # 获取边界框
            cropped_image = image_array[y:y + h, x:x + w]  # 裁剪图像
            return cropped_image

        # 调用去白边函数
        image_array_no_border = remove_white_border(image_array)

        # 显示图像
        fig, ax = plt.subplots(figsize=(8, 8))  # 自定义画布大小
        ax.imshow(image_array_no_border)
        ax.axis('off')  # 关闭坐标轴

        if title is not None:
            ax.set_title(title)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除多余边距
        plt.tight_layout(pad=0)  # 紧凑布局
        plt.show()



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
    def mask_reduce_dim(self,mask):
        if mask.ndim==3:
            mask = mask[:,:,0]
        return mask

    def FreeFine_generation(self, ori_img, ori_mask, coarse_input, target_mask, guidance_text,
                            guidance_scale, eta, end_step=10, num_step=50, start_step=25,
                            share_attn=True, method_type='tca', local_text_edit=True, local_perturbation=True, verbose=True, return_ori=False, seed=42, draw_mask=None,
                            return_intermediates=False, use_auto_draw=False, cons_area=None, reduce_inp_artifacts=False, end_scale=0.5):
        assert method_type in ['tca','ssa','sdsa','mmsa','mmsa_es'],f"check method type f{method_type}, which is not in {['tca','ssa','sdsa','mmsa','mmsa_es']}"
        print(f'current type is {method_type}')
        seed_everything(seed)
        ori_mask = self.mask_reduce_dim(ori_mask)
        target_mask = self.mask_reduce_dim(target_mask)
        if draw_mask is not None:
            draw_mask = self.mask_reduce_dim(draw_mask)

        # DDIM INVERSION
        shifted_mask, inverted_latent = self.DDIM_inversion_func(img=coarse_input, mask=target_mask,
                                                                 prompt="",
                                                                 num_step=num_step,
                                                                 start_step=start_step,
                                                                 ref_img=ori_img,verbose=verbose)  # ndarray mask

        edit_gen_image, refer_gen_image,intermediates = self.Details_Preserving_regeneration(coarse_input, inverted_latent,
                                                                                             guidance_text,
                                                                                             target_mask, ori_mask, draw_mask,
                                                                                             num_steps=num_step,
                                                                                             start_step=start_step,
                                                                                             end_step=end_step,
                                                                                             guidance_scale=guidance_scale, eta=eta,
                                                                                             share_attn=share_attn, method_type=method_type,
                                                                                             verbose=verbose,
                                                                                             local_text_edit=local_text_edit, local_perturbation=local_perturbation,
                                                                                             return_intermediates=return_intermediates, cons_area = cons_area,
                                                                                             use_auto_draw=use_auto_draw, end_scale=end_scale,
                                                                                             reduce_inp_artifacts=reduce_inp_artifacts,
                                                                                             )
        if intermediates is not None:
            self.save_intermediate_images_and_gif_v2(intermediates)
        if not return_ori:
            return edit_gen_image
        return edit_gen_image,refer_gen_image

    def FreeFine_cross_image_composition(self, img_lists, ori_mask_lists, tgt_mask_lists, coarse_input, guidance_text_list,
                                         guidance_scale, eta, end_step=10, num_step=50, start_step=25,
                                         share_attn=True, method_type='tca', local_text_edit=True, local_perturbation=True, verbose=True, seed=42, draw_mask=None,
                                         return_intermediates=False, use_auto_draw=False, end_scale=0.5, dil_completion=False, dil_factor=15, appearance_transfer=False):
        assert method_type in ['tca','ssa','sdsa','mmsa','mmsa_es'],f"check method type f{method_type}, which is not in {['tca','ssa','sdsa','mmsa','mmsa_es']}"
        print(f'current type is {method_type}')
        #for image copostion
        #each part of the target image will query different latents in the pool
        seed_everything(seed)
        ori_mask_lists = [self.mask_reduce_dim(mask) for mask in ori_mask_lists]
        tgt_mask_lists = [self.mask_reduce_dim(mask) for mask in tgt_mask_lists]
        # draw_mask = self.mask_reduce_dim(draw_mask)

        # DDIM INVERSION
        inverted_latents = self.DDIM_inversion_func_compose(img=coarse_input,compose_imgs=img_lists,
                                                            prompt="",
                                                            num_step=num_step,
                                                            start_step=start_step,
                                                            verbose=verbose)  # ndarray mask

        edit_gen_image, intermediates = self.Details_Preserving_regeneration_compose(coarse_input, inverted_latents,
                                                                                guidance_text_list,
                                                                                ori_mask_lists, tgt_mask_lists, draw_mask,
                                                                                num_steps=num_step,
                                                                                start_step=start_step,
                                                                                end_step=end_step,dil_factor=dil_factor,
                                                                                guidance_scale=guidance_scale, eta=eta,
                                                                                share_attn=share_attn,method_type=method_type,
                                                                                verbose=verbose,dil_completion=dil_completion,
                                                                                local_text_edit=local_text_edit,local_perturbation=local_perturbation,
                                                                                return_intermediates=return_intermediates,
                                                                                use_auto_draw=use_auto_draw,end_scale=end_scale,appearance_transfer=appearance_transfer,
                                                                                )
        if intermediates is not None:
            self.save_intermediate_images_and_gif_v2(intermediates)
        return edit_gen_image

    def FreeFine_background_generation(self, ori_img, ori_mask, guidance_text,
                                       guidance_scale, eta, end_step=10, num_step=50, start_step=25,
                                       share_attn=True, method_type='tca', local_text_edit=True, local_perturbation=True, verbose=True, seed=42,
                                       return_intermediates=False, end_scale=0.5, latent_blended=False, blend_range=(0,40),):
        seed_everything(seed)
        ori_mask = self.mask_reduce_dim(ori_mask)
        # DDIM INVERSION
        _, inverted_latent = self.DDIM_inversion_func(img=ori_img, mask=ori_mask,
                                                                 prompt="",
                                                                 num_step=num_step,
                                                                 start_step=start_step,
                                                                 ref_img=None,verbose=verbose)  # ndarray mask

        edit_gen_image,intermediates = self.Details_Preserving_regeneration_background(ori_img,inverted_latent,
                                                                                    guidance_text,
                                                                                    ori_mask,
                                                                                    num_steps=num_step,
                                                                                    start_step=start_step,
                                                                                    end_step=end_step,
                                                                                    guidance_scale=guidance_scale, eta=eta,
                                                                                    share_attn=share_attn,method_type=method_type,
                                                                                    verbose=verbose,end_scale=end_scale,
                                                                                    local_text_edit=local_text_edit,local_perturbation=local_perturbation,
                                                                                    return_intermediates=return_intermediates,
                                                                                    latent_blended=latent_blended,
                                                                                    blend_range=blend_range,
                                                                                    )
        if intermediates is not None:
            self.save_intermediate_images_and_gif_v2(intermediates)

        return edit_gen_image


    def save_intermediate_images_and_gif(self, intermediates, output_folder="sd_steps_output",
                                         gif_name="sd_progress.gif", duration=200):
        """
        将 intermediate latents 转换为图像并保存到文件夹，同时生成 GIF
        """

        os.makedirs(output_folder, exist_ok=True)  # 确保目录存在
        images = []  # 存储 PIL 图像用于 GIF

        for idx, inter_latent in enumerate(intermediates):
            # 1. 将 latent 转换为图像
            inter_image = self.latent2image(inter_latent, return_type="pt")[0].permute(1, 2,
                                                                                    0).detach().cpu().numpy() * 255
            inter_image = inter_image.astype(np.uint8)

            # 2. 使用 PIL 进行保存
            img = Image.fromarray(inter_image)

            # 3. 在图像上绘制 Step 信息
            draw = ImageDraw.Draw(img)
            # 设置字体大小和字体
            # 根据图像大小设置字体大小
            font_size = img.width // 8  # 根据图片大小调整字体
            try:
                # 尝试加载常见字体 DejaVu Sans
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except IOError:
                # 如果字体加载失败，则使用默认字体，并手动指定一个合理的大小
                font = ImageFont.load_default()  # 加载默认字体（可能会比较小）
                print(f"DejaVu Sans font not found, using default font with size 50")
                font_size = 50  # 设置默认字体大小，确保文本大小不会太小

            text = f"Step = {idx}"  # 显示步骤
            text_position = (10, 10)  # 文字显示在左上角
            text_color = (255, 0, 0)  # 红色

            # 绘制文字
            draw.text(text_position, text, fill=text_color, font=font)
            # 4. 保存 PNG
            # img_path = os.path.join(output_folder, f"denoise_step_{idx:03d}.png")
            # img.save(img_path)
            images.append(img)


        # print(f"Saved {len(intermediates)} images in {output_folder}")

        # 生成 GIF
        if images:
            gif_path = os.path.join(output_folder, gif_name)
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
            print(f"GIF saved as {gif_path}")


    def save_intermediate_images_and_gif_v2(self, intermediates, output_folder="sd_steps_output",
                                         gif_name="sd_progress.gif", duration=200):
        """
        将 intermediate latents 转换为图像并保存到文件夹，同时生成 GIF
        """

        os.makedirs(output_folder, exist_ok=True)  # 确保目录存在
        images = []  # 存储 PIL 图像用于 GIF
        latent_rgb_factors = [
            [0.3512, 0.2297, 0.3227],
            [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721],
            [-0.2120, -0.2616, -0.7177]
        ]
        previewer = Latent2RGBPreviewer(latent_rgb_factors)
        for idx, inter_latent in enumerate(intermediates):
            # 1. 将 latent 转换为图像，使用 preview_to_image 来替代 VAE 解码
            inter_image = previewer.decode_latent_to_preview(inter_latent)



            # 2. 使用 PIL 进行保存
            img = inter_image.resize((512, 512))

            # 3. 在图像上绘制 Step 信息
            draw = ImageDraw.Draw(img)
            font_size = img.width // 8  # 根据图片大小调整字体
            try:
                # 尝试加载常见字体 DejaVu Sans
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except IOError:
                # 如果字体加载失败，则使用默认字体，并手动指定一个合理的大小
                font = ImageFont.load_default()  # 加载默认字体（可能会比较小）
                print(f"DejaVu Sans font not found, using default font with size 50")
                font_size = 50  # 设置默认字体大小，确保文本大小不会太小
            text = f"Step = {idx}"
            text_position = (10, 10)  # 文字显示在左上角
            text_color = (255, 0, 0)  # 红色
            draw.text(text_position, text, fill=text_color, font=font)

            # 4. 保存 PNG
            # img_path = os.path.join(output_folder, f"denoise_step_{idx:03d}.png")
            # img.save(img_path)
            images.append(img)

        # print(f"Saved {len(intermediates)} images in {output_folder}")

        # 生成 GIF
        if images:
            gif_path = os.path.join(output_folder, gif_name)
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
            print(f"GIF saved as {gif_path}")


    def visualize_images_column(self, image_list):
        n = len(image_list)
        rows = n  # 每列显示一张图，所以行数就是图像的数量
        cols = 1  # 只有一列

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

        # 如果返回的是单一的 Axes 对象，变成列表
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i in range(n):
            axes[i].imshow(image_list[i], cmap='gray')  # 显示图像，假设是灰度图
            axes[i].axis('off')  # 关闭坐标轴

        for i in range(n, len(axes)):  # 如果多余的子图没有被使用，隐藏它们
            axes[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)  # 设置子图之间的间距为0
        plt.margins(0, 0)  # 关闭所有边距
        plt.tight_layout(pad=0)  # 紧凑布局
        plt.show()

    def visualize_images(self, image_list):
        n = len(image_list)
        cols = 3  # 每行3张图
        rows = (n + cols - 1) // cols  # 计算需要多少行

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

        # 如果返回的是单一的 Axes 对象，变成列表
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i in range(n):
            axes[i].imshow(image_list[i], cmap='gray')  # 显示图像，假设是灰度图
            axes[i].axis('off')  # 关闭坐标轴

        for i in range(n, len(axes)):  # 如果多余的子图没有被使用，隐藏它们
            axes[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)  # 设置子图之间的间距为0
        plt.margins(0, 0)  # 关闭所有边距
        plt.tight_layout(pad=0)  # 紧凑布局
        plt.show()

    def save_mask(self,mask, dst_path):
        cv2.imwrite(dst_path, mask)  # 将mask保存为png图片 (注意：mask是二值图，乘以255以得到可见的结果)

    def save_img(self,img, dst_path):
        cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 将图片保存为png格式 (注意：需转换为BGR格式)



    def preprocess_image(self, image, device):
        image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
        # 替代 rearrange(image, "h w c -> 1 c h w")
        image = image.permute(2, 0, 1)  # 将通道维度放到前面 (h,w,c) -> (c,h,w)
        image = image.unsqueeze(0)  # 添加批次维度 (c,h,w) -> (1,c,h,w)
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
        plt.margins(0, 0)  # 关闭所有边距
        plt.tight_layout(pad=0)  # 紧凑布局
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





    def resize_img(self,trans_img, size=None, ):
        if isinstance(trans_img, np.ndarray):
            trans_img = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
            trans_img = Image.fromarray(trans_img)  # PIL img
        assert size is not None, "for resize pipe,size should be given"
        trans_img.thumbnail(size, Image.Resampling.LANCZOS)
        return_img = np.array(trans_img.copy())
        return_img = cv2.cvtColor(return_img, cv2.COLOR_BGR2RGB)
        return return_img
    @torch.no_grad()
    def DDIM_inversion_func(self, img, mask, prompt,
                            num_step, start_step=0, ref_img=None, verbose=False):

        source_image = self.preprocess_image(img, self.device)
        if ref_img is not None:
            ref_img = self.resize_img(ref_img, size=[512, 512])
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
                return_intermediates=True,verbose=verbose
            )

        self.controller.reset()  # clear
        return mask.detach().cpu().numpy(), latents_list

    @torch.no_grad()
    def DDIM_inversion_func_compose(self, img, compose_imgs, prompt,
                            num_step, start_step=0,  verbose=False):

        source_image = self.preprocess_image(img, self.device)
        for ref_img in compose_imgs:
            ref_img = self.resize_img(ref_img, size=[512, 512])
            ref_image = self.preprocess_image(ref_img, self.device)
            source_image = torch.cat((source_image, ref_image))


        latents, latents_list = \
            self.invert(
                source_image,
                prompt,
                guidance_scale=1.0,
                num_inference_steps=num_step,
                num_actual_inference_steps=num_step - start_step,
                return_intermediates=True, verbose=verbose
            )

        self.controller.reset()  # clear
        return  latents_list



    def prepare_surrounding_mask(self, shifted_mask, cons_area,rate=0.5):
        feasible_regions = 1 - cons_area
        # 确保输入是二值化的 Tensor（0 和 1）
        shifted_mask[shifted_mask>0] = 1

        # 1. 获取掩码的边界框 (bounding box)
        mask_indices = torch.nonzero(shifted_mask)
        if mask_indices.numel() == 0:
            return torch.zeros_like(shifted_mask)  # 如果没有找到非零值，返回全零掩码

        y_min, x_min = mask_indices.min(dim=0)[0]
        y_max, x_max = mask_indices.max(dim=0)[0]

        # 宽度和高度
        w = x_max - x_min
        h = y_max - y_min

        # 2. 上下左右移动 50% 的宽度和高度 (四个角)
        jitter_x = int(rate * w)
        jitter_y = int(rate * h)

        # 随机移动左上角和右下角
        new_x_min = max(0, x_min - jitter_x)
        new_y_min = max(0, y_min - jitter_y)
        new_x_max = min(x_max + jitter_x, shifted_mask.shape[1] - 1)
        new_y_max = min(y_max + jitter_y, shifted_mask.shape[0] - 1)

        # 3. 生成新的编辑区域 (覆盖 jitter 后的范围)
        local_edit_region = torch.zeros_like(shifted_mask)
        local_edit_region[new_y_min:new_y_max + 1, new_x_min:new_x_max + 1] = 1

        # 4. 与 feasible_domains 做与运算，确保区域在可行范围内
        final_mask = local_edit_region * feasible_regions * (1-shifted_mask)

        return final_mask




    @torch.no_grad()
    def prepare_various_mask(self, shifted_mask, ori_mask, draw_mask, sup_res_w, sup_res_h, init_code, verbose=False, use_auto_draw=False, cons_area=None,
                             reduce_inp_artifacts=False,
                             ):
        if not use_auto_draw:
            #draw mask is provided by the user
            #constrain area is used to avoid overlapping dilation with other existing objects
            if not reduce_inp_artifacts:
                #ddpm region == completion area
                shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
                ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
                flexible_region = self.prepare_tensor_mask(draw_mask, sup_res_w, sup_res_h) * (1-shifted_mask_tensor)

                fg_mask = flexible_region + shifted_mask_tensor #add completion area to target mask to form full mask
                fg_mask[fg_mask>0] = 1.0
                if not verbose:
                    self.temp_view(fg_mask.cpu().numpy(), 'TCA mask')
                complete_region_tensor = flexible_region
                local_var_region_tensor = flexible_region
                if not verbose:
                    self.temp_view(complete_region_tensor.cpu().numpy(), 'disturb region')
            else:
                #ddpm region = completion area + background inpainting blending area, to reduce artifacts
                assert cons_area is not None, 'for auto artifact expansion use cons area '
                dil_ori_mask = self.dilate_mask(ori_mask, 30)
                dil_mask_tensor = self.prepare_tensor_mask(dil_ori_mask, sup_res_w, sup_res_h)
                cons_area_tensor = self.prepare_tensor_mask(cons_area, sup_res_w, sup_res_h)
                shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
                ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
                flexible_region = self.prepare_tensor_mask(draw_mask, sup_res_w, sup_res_h) * (1 - shifted_mask_tensor)

                fg_mask = flexible_region + shifted_mask_tensor #add completion area to target mask to form full mask
                fg_mask[fg_mask > 0] = 1.0
                if not verbose:
                    self.temp_view(fg_mask.cpu().numpy(), 'TCA mask')
                complete_region_tensor = flexible_region
                local_var_region_tensor = (1 - cons_area_tensor) * (1 - shifted_mask_tensor) * dil_mask_tensor + flexible_region
                local_var_region_tensor[local_var_region_tensor>0] = 1
                if not verbose:
                    self.temp_view(complete_region_tensor.cpu().numpy(), 'disturb region')
        else:
            # draw mask is not provided, which means no need to consider structure completion
            if not reduce_inp_artifacts:
                assert cons_area is not None, 'for auto draw better use cons area '
                #ddpm region is a surrounding dilation boundary
                dil_tgt_mask = self.dilate_mask(shifted_mask, 15)
                dil_tgt_mask_tensor = self.prepare_tensor_mask(dil_tgt_mask, sup_res_w, sup_res_h)
                shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
                ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
                cons_area_tensor = self.prepare_tensor_mask(cons_area, sup_res_w, sup_res_h)
                fg_mask = shifted_mask_tensor #no need to add

                cons_area_tensor = cons_area_tensor - ori_mask_tensor
                complete_region_tensor = (1 - cons_area_tensor) * (1-shifted_mask_tensor) * dil_tgt_mask_tensor
                # complete_region_tensor = dil_tgt_mask_tensor
                # self.temp_view(complete_region_tensor.cpu().numpy())
                local_var_region_tensor = complete_region_tensor

            else:
                assert cons_area is not None, 'for auto draw better use cons area '
                dil_tgt_mask = self.dilate_mask(shifted_mask, 15)
                dil_ori_mask = self.dilate_mask(ori_mask, 30) #the same as background gen dilation factor
                dil_mask_tensor = self.prepare_tensor_mask(dil_ori_mask, sup_res_w, sup_res_h)
                dil_tgt_mask_tensor = self.prepare_tensor_mask(dil_tgt_mask, sup_res_w, sup_res_h)
                shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
                ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
                cons_area_tensor = self.prepare_tensor_mask(cons_area, sup_res_w, sup_res_h)
                fg_mask = shifted_mask_tensor

                cons_area_tensor = cons_area_tensor - ori_mask_tensor
                complete_region_tensor =  dil_mask_tensor + dil_tgt_mask_tensor
                complete_region_tensor[complete_region_tensor>0] = 1
                complete_region_tensor *= (1 - cons_area_tensor) * (1 - shifted_mask_tensor)
                local_var_region_tensor = complete_region_tensor

        complete_region_tensor = F.interpolate(complete_region_tensor.unsqueeze(0).unsqueeze(0),
                                              (init_code.shape[2], init_code.shape[3]),
                                              mode='nearest').squeeze(0, 1)
        local_var_region_tensor = F.interpolate(local_var_region_tensor.unsqueeze(0).unsqueeze(0),
                                               (init_code.shape[2], init_code.shape[3]),
                                               mode='nearest').squeeze(0, 1)
        return fg_mask,shifted_mask_tensor,ori_mask_tensor,complete_region_tensor,local_var_region_tensor

    @torch.no_grad()
    def prepare_composition_masks(self, ori_mask_lists, tgt_mask_lists, sup_res_w, sup_res_h, init_code, dil_completion=False,dil_factor=15,draw_mask=None,appearance_transfer=False):
        if appearance_transfer:
            tgt_masks_tensor, ori_masks_tensor = [], []
            for ori_mask in ori_mask_lists:
                ori_masks_tensor.append(self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h))

            local_perturbation_tensor = torch.zeros_like(ori_masks_tensor[0])
            for shifted_mask in tgt_mask_lists:
                dil_tgt_mask_tensor = self.prepare_tensor_mask(self.dilate_mask(shifted_mask, dil_factor), sup_res_w, sup_res_h)
                tgt_masks_tensor.append(dil_tgt_mask_tensor )
                local_perturbation_tensor += dil_tgt_mask_tensor


            local_perturbation_tensor[local_perturbation_tensor > 0] = 1


            tgt_masks_tensor.append(1 - local_perturbation_tensor)  # bg = 1- dil fg

            local_perturbation_tensor = F.interpolate(local_perturbation_tensor.unsqueeze(0).unsqueeze(0),
                                                      (init_code.shape[2], init_code.shape[3]),
                                                      mode='nearest').squeeze(0, 1)
            ori_masks_tensor = torch.cat([p[None, :, :] for p in ori_masks_tensor], dim=0)
            tgt_masks_tensor = torch.cat([p[None, :, :] for p in tgt_masks_tensor], dim=0)
            completion_mask_cfg = deepcopy(local_perturbation_tensor)
            return tgt_masks_tensor, ori_masks_tensor, local_perturbation_tensor, completion_mask_cfg
        if draw_mask is None:
            tgt_masks_tensor, ori_masks_tensor = [], []
            for ori_mask in ori_mask_lists:
                ori_masks_tensor.append(self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h))

            local_perturbation_tensor, fg_tensor= torch.zeros_like(ori_masks_tensor[0]),torch.zeros_like(ori_masks_tensor[0])
            for shifted_mask in tgt_mask_lists:
                # lp region is a surrounding dilation boundary
                dil_tgt_mask_tensor = self.prepare_tensor_mask(self.dilate_mask(shifted_mask, dil_factor), sup_res_w, sup_res_h)
                shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
                if not dil_completion:
                    tgt_masks_tensor.append(shifted_mask_tensor)

                else:
                    tgt_masks_tensor.append(dil_tgt_mask_tensor)#further use cons area to constrain dilation

                fg_tensor += shifted_mask_tensor
                local_perturbation_tensor +=  dil_tgt_mask_tensor

            fg_tensor[fg_tensor>0] = 1
            local_perturbation_tensor[local_perturbation_tensor>0] = 1

            if not dil_completion:
                tgt_masks_tensor.append(1-local_perturbation_tensor) #bg = 1- dil fg
            else:
                tgt_masks_tensor.append(1 - fg_tensor)  # bg = 1- dil fg
            local_perturbation_tensor  = local_perturbation_tensor*(1- fg_tensor) #bd
            local_perturbation_tensor = F.interpolate( local_perturbation_tensor .unsqueeze(0).unsqueeze(0),
                                                    (init_code.shape[2], init_code.shape[3]),
                                                    mode='nearest').squeeze(0, 1)
            ori_masks_tensor = torch.cat([p[None,:,:] for p in ori_masks_tensor],dim=0)
            tgt_masks_tensor = torch.cat([p[None,:,:] for p in tgt_masks_tensor],dim=0)
            if not dil_completion:
                completion_mask_cfg = torch.zeros_like(local_perturbation_tensor)
            else:
                completion_mask_cfg = deepcopy(local_perturbation_tensor)
            return tgt_masks_tensor,ori_masks_tensor, local_perturbation_tensor,completion_mask_cfg
        else:
            #draw_mask is a list align with target mask list
            tgt_masks_tensor, ori_masks_tensor = [], []
            for ori_mask in ori_mask_lists:
                ori_masks_tensor.append(self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h))

            local_perturbation_tensor, fg_tensor = torch.zeros_like(ori_masks_tensor[0]), torch.zeros_like(
                ori_masks_tensor[0])
            for i,shifted_mask in enumerate(tgt_mask_lists):
                cur_draw_region = self.prepare_tensor_mask(draw_mask[i], sup_res_w, sup_res_h)
                # lp region is a surrounding dilation boundary
                # dil_tgt_mask_tensor = self.prepare_tensor_mask(self.dilate_mask(shifted_mask, dil_factor), sup_res_w,
                #                                                sup_res_h)

                shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
                dil_tgt_mask_tensor = cur_draw_region + shifted_mask_tensor
                dil_tgt_mask_tensor[dil_tgt_mask_tensor > 0] = 1
                tgt_masks_tensor.append(dil_tgt_mask_tensor)  # further use cons area to constrain dilation

                fg_tensor += shifted_mask_tensor
                local_perturbation_tensor += dil_tgt_mask_tensor

            fg_tensor[fg_tensor > 0] = 1
            local_perturbation_tensor[local_perturbation_tensor > 0] = 1

            tgt_masks_tensor.append(1 - local_perturbation_tensor)  # bg = 1- dil fg
            local_perturbation_tensor = local_perturbation_tensor * (1 - fg_tensor)  # bd
            local_perturbation_tensor = F.interpolate(local_perturbation_tensor.unsqueeze(0).unsqueeze(0),
                                                      (init_code.shape[2], init_code.shape[3]),
                                                      mode='nearest').squeeze(0, 1)
            ori_masks_tensor = torch.cat([p[None, :, :] for p in ori_masks_tensor], dim=0)
            tgt_masks_tensor = torch.cat([p[None, :, :] for p in tgt_masks_tensor], dim=0)
            return tgt_masks_tensor, ori_masks_tensor, local_perturbation_tensor,local_perturbation_tensor
    @torch.no_grad()
    def prepare_mask_bggen(self,mask, sup_res_w, sup_res_h, init_code):

        mask_tensor = self.prepare_tensor_mask(mask, sup_res_w, sup_res_h)

        local_var_region_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0),
                                               (init_code.shape[2], init_code.shape[3]),
                                               mode='nearest').squeeze(0, 1)


        return mask_tensor,local_var_region_tensor

    def prepare_tensor_mask(self, mask, sup_res_w, sup_res_h,binary=True):
        # mask interpolation
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_tensor_shifted = torch.tensor(mask, device=self.device)
        m_dtype = mask_tensor_shifted.dtype
        mask_tensor_shifted = mask_tensor_shifted.unsqueeze(0).unsqueeze(0)
        transformed_masks_tensor = F.interpolate(mask_tensor_shifted, (sup_res_h, sup_res_w),
                                                 mode="nearest").squeeze(0, 1)
        if binary:
            transformed_masks_tensor[transformed_masks_tensor>0.0] = 1.0
            # transformed_masks_tensor= transformed_masks_tensor.float()
        else:
            norm_ = transformed_masks_tensor.max()
            transformed_masks_tensor = transformed_masks_tensor.float()  # 转换为浮动类型
            transformed_masks_tensor /= norm_

        return transformed_masks_tensor
    def Details_Preserving_regeneration(self, source_image, inverted_latents, edit_prompt, shifted_mask,
                                        ori_mask, draw_mask,
                                        num_steps=100, start_step=30, end_step=10,
                                        eta=1, guidance_scale=7.5,
                                        share_attn=True, method_type='tca', verbose=False,
                                        local_text_edit=True, local_perturbation=True,
                                        return_intermediates=False, use_auto_draw=False, cons_area=None, use_share_attention=False,
                                        reduce_inp_artifacts=False, end_scale=0.5):

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

        full_h, full_w = source_image.shape[:2]

        fg_retain_mask,fg_retain_mask_st2,fg_ref_mask,completion_mask_cfg,local_var_reg=self.prepare_various_mask(shifted_mask, ori_mask, draw_mask, full_h, full_w,
                                                                                                                  init_code_orig, verbose=verbose, use_auto_draw=use_auto_draw, cons_area=cons_area,
                                                                                                                  reduce_inp_artifacts= reduce_inp_artifacts)
        # mask = self.prepare_controller_ref_mask(shifted_mask, False)
        completion_mask_cfg = F.interpolate(completion_mask_cfg.unsqueeze(0).unsqueeze(0),
                                               (init_code_orig.shape[2], init_code_orig.shape[3]),
                                               mode='nearest').squeeze(0, 1)
        self.controller.fg_retain_mask = fg_retain_mask.to(self.device)
        self.controller.fg_retain_mask_st2 = fg_retain_mask_st2.to(self.device)#shift mask tensor
        self.controller.fg_ref_mask = fg_ref_mask.to(self.device)
        self.controller.local_edit_region = fg_retain_mask.to(self.device)

        self.controller.reset()
        self.controller.log_mask = False #forbid mask expansion

        refer_latents_ori = inverted_latents[::-1]

        gen_images,intermediates = self.forward_sampling(
            prompt=[edit_prompt, ""],
            refer_latents=refer_latents_ori,
            end_step=end_step,
            batch_size=2,
            latents=init_code_orig,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            num_actual_inference_steps=num_steps - start_step,
            eta=eta,
            completion_mask_cfg = completion_mask_cfg,
            local_var_reg=local_var_reg,
            share_attn=share_attn,method_type=method_type,verbose=verbose,blending=local_text_edit,local_perturbation=local_perturbation,
            return_intermediates=return_intermediates,
            use_share_attention=use_share_attention,end_scale=end_scale,
        )
        edit_gen_image, ref_gen_image = gen_images
        self.controller.reset()
        refer_gen_image = ref_gen_image.permute(1, 2, 0).detach().cpu().numpy()*255
        edit_gen_image = edit_gen_image.permute(1, 2, 0).detach().cpu().numpy()*255
        return edit_gen_image.astype(np.uint8), refer_gen_image.astype(np.uint8),intermediates
    def Details_Preserving_regeneration_compose(self, source_image, inverted_latents, edit_prompt_list,ori_mask_lists, tgt_mask_lists, draw_mask,
                                        num_steps=100, start_step=30, end_step=10,
                                        eta=1,guidance_scale=7.5,dil_completion=False,appearance_transfer=False,
                                        share_attn=True,method_type='tca',verbose=False,
                                        local_text_edit=True, local_perturbation=True,
                                        return_intermediates=False,use_share_attention=False,dil_factor=15,end_scale=0.5):

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

        full_h, full_w = source_image.shape[:2]

        tgt_masks_tensor,ori_masks_tensor, local_perturbation_tensor,completion_mask_cfg =self.prepare_composition_masks(ori_mask_lists, tgt_mask_lists, full_h, full_w,
                                       init_code_orig,dil_completion=dil_completion,dil_factor=dil_factor,draw_mask=draw_mask,appearance_transfer=appearance_transfer)

        self.controller.src_masks = ori_masks_tensor
        self.controller.tgt_masks = tgt_masks_tensor
        self.controller.reset()


        refer_latents_ori = inverted_latents[::-1]

        gen_images,intermediates = self.forward_sampling_compose(
            prompt=edit_prompt_list,
            refer_latents=refer_latents_ori,
            end_step=end_step,
            batch_size=init_code_orig.shape[0],
            latents=init_code_orig,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            num_actual_inference_steps=num_steps - start_step,
            eta=eta,
            local_var_reg=local_perturbation_tensor,local_text_edit=local_text_edit,cfg_masks_tensor = completion_mask_cfg,
            share_attn=share_attn,method_type=method_type,verbose=verbose,local_perturbation=local_perturbation,
            return_intermediates=return_intermediates,
            use_share_attention=use_share_attention,end_scale=end_scale,
        )
        edit_gen_image = gen_images
        self.controller.reset()
        edit_gen_image = edit_gen_image.permute(1, 2, 0).detach().cpu().numpy()*255
        return edit_gen_image.astype(np.uint8), intermediates
    def Details_Preserving_regeneration_background(self, ori_img,inverted_latents, edit_prompt,
                                        ori_mask,
                                        num_steps=100, start_step=30, end_step=10,
                                        guidance_scale=3.5, eta=1,
                                        verbose=False,
                                        local_text_edit=True, local_perturbation=True,end_scale=0.5,
                                        return_intermediates=False,share_attn=True,method_type='tca',latent_blended=True,blend_range=(0,40)):

        """
        latent vis
        # noised_image = self.decode_latents(start_latents).squeeze(0)
        # # print(noised_image.shape)
        # noised_image = self.numpy_to_pil(noised_image)[0]
        # print(noised_image.size)
        # print(noised_image.shape)
        """

        start_latents = inverted_latents[-1]  # [ori,35 steps latents:50->15]
        refer_latents_ori = inverted_latents[::-1]
        init_code_orig = deepcopy(start_latents)

        full_h, full_w = ori_img.shape[:2]


        #note that sheilding other objects to be retain has been already down outside this func
        mask_tensor,local_var_reg=self.prepare_mask_bggen(ori_mask,full_h, full_w, init_code_orig)

        self.controller.fg_retain_mask = mask_tensor.to(self.device)
        self.controller.local_edit_region = mask_tensor.to(self.device)

        self.controller.reset()


        gen_images,intermediates = self.forward_sampling_background_gen(
            prompt=[edit_prompt,""],
            end_step=end_step,
            batch_size=2,
            refer_latents = refer_latents_ori,
            latents=init_code_orig,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            num_actual_inference_steps=num_steps - start_step,
            eta=eta,
            local_cfg_reg = local_var_reg,
            local_var_reg=local_var_reg,
            share_attn=share_attn,method_type=method_type,verbose=verbose,local_text_edit=local_text_edit,local_perturbation=local_perturbation,
            return_intermediates=return_intermediates,end_scale=end_scale,latent_blended=latent_blended,blend_range=blend_range,

        )


        self.controller.reset()
        edit_gen_image = gen_images[0].permute(1, 2, 0).detach().cpu().numpy()*255
        return edit_gen_image.astype(np.uint8),intermediates














