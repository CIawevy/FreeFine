# from src.utils.geo_utils import IntegratedP3DTransRasterBlendingFull,param2theta,wrapAffine_tensor,PartialConvInterpolation,tensor_inpaint_fmm,calculate_cosine_similarity_between_batches
# from src.utils.geo_utils import calculate_cosine_similarity_between_batches
from diffusers.utils.torch_utils import  randn_tensor
import time
import sys
sys.path.append('/data/Hszhu/Reggio')
# from ram import inference_tag2text
from typing import List, Union
import clip
from einops import rearrange
import spacy
import os
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import cv2
import copy
import random
from pytorch_lightning import seed_everything
import numpy as np
from src.utils.mask_gaussian_utils import generate_gaussian_mask
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
NUM_DDIM_STEPS = 50
SIZES = {
    0:4,
    1:2,
    2:1,
    3:1,
}
import torchvision.transforms as TS


def calculate_cosine_similarity_between_batches(features, has_reference_image=False, cos_threshold=0.8):
    batch, channels, height, width = features.size()

    # 将特征展平为 (batch, L, C)
    features_flat = features.view(batch, channels, height * width).transpose(1, 2)

    # 归一化特征向量
    norms = features_flat.norm(dim=2, keepdim=True) + 1e-8  # 防止除以零
    normalized_features = features_flat / norms

    # 编辑图像特征和参考图像特征
    edit_features = normalized_features[0]
    ref_features = normalized_features[1]

    # 计算编辑图像每个位置与参考图像每个位置的余弦相似度
    cosine_similarity = torch.matmul(edit_features, ref_features.transpose(0, 1))  # (height * width, height * width)

    # 处理相似度阈值
    max_sim, max_indices = torch.max(cosine_similarity, dim=1)

    # 初始化输出张量
    # final_max_indices = torch.zeros((batch, height, width, 4), dtype=torch.long)
    final_max_indices = torch.empty((height, width, 2), dtype=torch.long).to(features.device)

    for h in range(height):
        for w in range(width):
            hw = h * width + w
            max_ref_hw = max_indices[hw]
            max_ref_h = max_ref_hw // width
            max_ref_w = max_ref_hw % width
            max_sim_value = max_sim[hw]

            # 应用相似度阈值
            if max_sim_value < cos_threshold:
                max_ref_h = -1
                max_ref_w = -1
                scaled_cos_sim = -1
            else:
                scaled_cos_sim = max_sim_value

            final_max_indices[h, w] = torch.tensor([max_ref_h, max_ref_w], dtype=features.dtype)
            # final_max_indices[h, w] = torch.tensor([max_ref_h, max_ref_w, scaled_cos_sim], dtype=features.dtype)
            # final_max_indices[b, h, w] = torch.tensor([max_b, max_h, max_w, cos_sim * 1000])

    return final_max_indices
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
class Latent2RGBPreviewer:
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
            # feature_injection_allowed=True,
            # feature_injection_timpstep_range=(900, 600),
            use_mtsa=True,verbose=False,local_ddpm=True,
            **kwds):
        DEVICE = self.device
        assert guidance_scale > 1.0, 'USING THIS MODULE CFG Must > 1.0'
        self.controller.use_cfg = True
        self.controller.share_attn = use_mtsa  # allow MTSA
        self.controller.local_edit = blending    #allow local cfg
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

        # original sample
        # TODO :EACH STEP :need h feature as input ,next step need new h feature from new t and new init latent
        # assert foreground_mask is not None, 'FOR BG PRESERVATION foreground_mask should not be None'
        start_step = num_inference_steps - num_actual_inference_steps
        h_feature = None
        self.h_feature_cfg = True
        if not verbose:
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue
                timestep = t.detach().item()
                # if timestep > feature_injection_timpstep_range[0] or timestep < feature_injection_timpstep_range[1]:
                self.controller.set_FI_forbid()
                # else:
                #     if feature_injection_allowed:
                #         if not verbose:
                #             print(f"Feature Injection is allowed at timestep={timestep}")
                #         self.controller.set_FI_allow()
                #     else:
                #         self.controller.set_FI_forbid()

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
                    # if i < 50 - end_step:
                    #     noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,
                    #                            h_sample=h_feature_inputs)
                    # else:
                    noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    if not blending:
                        noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                    else:
                        #modified
                        # local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * obj_mask
                        local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * foreground_mask
                        noise_pred = noise_pred_uncon + local_text_guidance
                    # compute the previous noise sample x_t -> x_t-1
                    # YUJUN: right now, the only difference between step here and step in scheduler
                    # is that scheduler version would clamp pred_x0 between [-1,1]
                    # don't know if that's gonna have huge impact
                    # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, eta=eta)[0]
                    full_mask = torch.ones_like(local_var_reg)
                    if not local_ddpm:
                        latents = self.ctrl_step(noise_pred, t, latents, full_mask, eta=eta)[0]
                    else:
                        latents = self.ctrl_step(noise_pred, t, latents, local_var_reg, eta=eta)[0]
                    latents_list.append(latents)
            image = self.latent2image(latents, return_type="pt")
            if return_intermediates:
                return image, latents_list
            return image
        else:
            for i, t in enumerate(self.scheduler.timesteps):
                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue
                timestep = t.detach().item()
                # if timestep > feature_injection_timpstep_range[0] or timestep < feature_injection_timpstep_range[1]:
                self.controller.set_FI_forbid()
                # else:
                #     if feature_injection_allowed:
                #         if not verbose:
                #             print(f"Feature Injection is allowed at timestep={timestep}")
                #         self.controller.set_FI_allow()
                #     else:
                #         self.controller.set_FI_forbid()

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
                    # if i < 50 - end_step:
                    #     noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings,
                    #                            h_sample=h_feature_inputs)
                    # else:
                    noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    if not blending:
                        noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                    else:
                        # modified
                        # local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * obj_mask
                        local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * foreground_mask
                        noise_pred = noise_pred_uncon + local_text_guidance
                    # compute the previous noise sample x_t -> x_t-1
                    # YUJUN: right now, the only difference between step here and step in scheduler
                    # is that scheduler version would clamp pred_x0 between [-1,1]
                    # don't know if that's gonna have huge impact
                    # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, eta=eta)[0]
                    full_mask = torch.ones_like(obj_mask)
                    if not local_ddpm:
                        latents = self.ctrl_step(noise_pred, t, latents, full_mask, eta=eta)[0]
                    else:
                        latents = self.ctrl_step(noise_pred, t, latents, local_var_reg, eta=eta)[0]
                    latents_list.append(latents)
            image = self.latent2image(latents, return_type="pt")
            if return_intermediates:
                return image, latents_list
            return image
    def forward_sampling_new(
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
            local_var_reg=None,
            foreground_mask=None,
            blending=True,
            use_mtsa=True,verbose=False,local_ddpm=True,context_guidance=1.0,
            **kwds):
        DEVICE = self.device
        assert guidance_scale > 1.0, 'USING THIS MODULE CFG Must > 1.0'
        self.controller.use_cfg = True
        self.controller.share_attn = use_mtsa  # allow MTSA
        self.controller.local_edit = blending    #allow local cfg
        self.controller.context_guidance = context_guidance
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

        # original sample
        # TODO :EACH STEP :need h feature as input ,next step need new h feature from new t and new init latent
        # assert foreground_mask is not None, 'FOR BG PRESERVATION foreground_mask should not be None'
        start_step = num_inference_steps - num_actual_inference_steps

        self.h_feature_cfg = True
        if not verbose:
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
                if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                    continue
                timestep = t.detach().item()
                self.controller.set_FI_forbid()

                ref_latent = refer_latents[i - start_step + 1][1]
                latents[1] = ref_latent

                if i < end_step:
                    self.controller.share_attn = use_mtsa  # allow SDSA
                    self.controller.stat = 'stage1'
                else:
                    # self.controller.share_attn = False
                    self.controller.stat = 'stage2'




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
                    # if i < end_step:
                    if not blending:
                        noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
                    else:
                        #modified
                        # local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * obj_mask
                        local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * foreground_mask
                        noise_pred = noise_pred_uncon + local_text_guidance
                    # else:
                    #     noise_pred = noise_pred_uncon
                    # TODO: OMIT text guided currently
                    # noise_pred = noise_pred_uncon

                    full_mask = torch.ones_like(local_var_reg)
                    if not local_ddpm:
                        latents = self.ctrl_step(noise_pred, t, latents, full_mask, eta=eta)[0]
                    else:
                        latents = self.ctrl_step(noise_pred, t, latents, local_var_reg, eta=eta)[0]
                    latents_list.append(latents)
            image = self.latent2image(latents, return_type="pt")
            if return_intermediates:
                return image, latents_list
            return image,None
        # else:
        #     for i, t in enumerate(self.scheduler.timesteps):
        #         if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
        #             continue
        #
        #         self.controller.set_FI_forbid()
        #
        #
        #
        #         ref_latent = refer_latents[i - start_step + 1][1]
        #         latents[1] = ref_latent
        #
        #         if i < end_step:
        #             # self.controller.share_attn = use_mtsa  # allow SDSA
        #             self.controller.local_edit = blending
        #         else:
        #             # self.controller.share_attn = False
        #             self.controller.local_edit = False
        #         # if guidance_scale > 1.:
        #         with torch.no_grad():
        #
        #             model_inputs = torch.cat([latents] * 2)
        #             # h_feature_inputs = torch.cat([h_feature] * 2)
        #             h_feature_inputs = h_feature
        #             if unconditioning is not None and isinstance(unconditioning, list):
        #                 _, text_embeddings = text_embeddings.chunk(2)
        #                 text_embeddings = torch.cat(
        #                     [unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
        #             # predict the noise
        #             # if guidance_scale > 1:
        #             self.controller.log_mask = False
        #             noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
        #
        #             noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
        #             if i < end_step:
        #                 if not blending:
        #                     noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
        #                 else:
        #                     # modified
        #                     # local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * obj_mask
        #                     local_text_guidance = guidance_scale * (noise_pred_con - noise_pred_uncon) * foreground_mask
        #                     noise_pred = noise_pred_uncon + local_text_guidance
        #             else:
        #                 noise_pred = noise_pred_uncon
        #             #TODO: NO TEXT GUIDE currently
        #             noise_pred = noise_pred_uncon
        #
        #             full_mask = torch.ones_like(local_var_reg)
        #             if not local_ddpm:
        #                 latents = self.ctrl_step(noise_pred, t, latents, full_mask, eta=eta)[0]
        #             else:
        #                 latents = self.ctrl_step(noise_pred, t, latents, local_var_reg, eta=eta)[0]
        #             # else:
        #             #     latents = self.ctrl_step(noise_pred, t, latents, local_var_reg, eta=eta)[0]
        #             latents_list.append(latents)
        #     image = self.latent2image(latents, return_type="pt")
        #     if return_intermediates:
        #         return image, latents_list
        #     return image,None

    def denoise(
            self,
            prompt,
            batch_size=1,
            height=512,
            width=512,
            num_inference_steps=50,
            start_step=15,
            guidance_scale=7.5,
            latents=None,
            eta=0.0,
            mask=None,
            local_ddpm=True,
            **kwds):
        DEVICE = self.device

        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size
                latents = torch.cat([latents]*batch_size,dim=0)

            # text embeddings
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

        text_embeddings = torch.cat([text_embeddings, text_embeddings], dim=0)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]


        for i, t in enumerate(self.scheduler.timesteps):
            if i < start_step:
                continue
            timestep = t.detach().item()
            with torch.no_grad():
                model_inputs = torch.cat([latents] * 2)
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)

                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

                # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, eta=eta)[0]
                # full_mask = torch.ones_like(obj_mask)
                if local_ddpm:
                    latents = self.ctrl_step(noise_pred, t, latents, mask, eta=eta)[0]
                else:
                    full_mask = torch.ones_like(mask)
                    latents = self.ctrl_step(noise_pred, t, latents, full_mask, eta=eta)[0]
                latents_list.append(latents)
        image = self.latent2image(latents, return_type="pt")
        # if return_intermediates:
        #     return image, latents_list
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

    # def move_and_inpaint_with_expansion_mask_3D(self, image, mask, depth_map, transforms, FX, FY,
    #                                             object_only=True,
    #                                             inpainter=None, mode=None,
    #                                             dilate_kernel_size=15, inp_prompt=None, target_mask=None,
    #                                             splatting_radius=0.015,
    #                                             splatting_tau=0.0, splatting_points_per_pixel=30):
    #
    #     if isinstance(image, Image.Image):
    #         image = np.array(image)
    #     if inp_prompt is None:
    #         inp_prompt = 'a photo of a background, a photo of an empty place'
    #     # 将掩码转换为灰度并应用阈值
    #     if mask.ndim == 3 and mask.shape[2] == 3:
    #         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #         # mask = mask > 128
    #
    #     if target_mask.ndim == 3 and target_mask.shape[2] == 3:
    #         target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
    #         # source_mask =source_mask > 128
    #
    #     mask = mask.astype(bool)
    #     target_mask = target_mask.astype(bool)
    #
    #     transformed_image, transformed_mask = IntegratedP3DTransRasterBlendingFull(image, depth_map, transforms,
    #                                                                                FX, FY,
    #                                                                                target_mask, object_only,
    #                                                                                splatting_radius=splatting_radius,
    #                                                                                splatting_tau=splatting_tau,
    #                                                                                splatting_points_per_pixel=splatting_points_per_pixel,
    #                                                                                return_mask=True,
    #                                                                                device=self.device)
    #
    #     # mask bool
    #     # MORPH_OPEN transformed target mask to suppress noise
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    #     transformed_mask = cv2.morphologyEx(transformed_mask, cv2.MORPH_OPEN, kernel)
    #     transformed_mask = (transformed_mask > 128).astype(bool)
    #     # repair_mask = (mask & ~transformed_mask)
    #     # ori_image_back_ground = np.where(mask[:, :, None], 0, image).astype(np.uint8)
    #     # new_image = np.where(transformed_mask[:, :, None], transformed_image,
    #     #                      ori_image_back_ground)  # with repair area to be black
    #     # image_with_hole = new_image
    #     # coarse_repaired = inpainter(Image.fromarray(new_image), Image.fromarray(
    #     #     repair_mask.astype(np.uint8) * 255))  # lama inpainting filling the black regions
    #     #
    #     # to_inpaint_img = coarse_repaired
    #     #
    #     # if mode == 1:
    #     #     semantic_repaired = to_inpaint_img
    #     # elif mode == 2:
    #     #     inpaint_mask = Image.fromarray(repair_mask.astype(np.uint8) * 255)
    #     #     print(f'SD inpainting Processing:')
    #     #     semantic_repaired = \
    #     #     self.sd_inpainter(prompt=inp_prompt, image=to_inpaint_img, mask_image=inpaint_mask).images[0]
    #     #
    #     # if semantic_repaired.size != to_inpaint_img.size:
    #     #     print(f'inpainted image {semantic_repaired.size} -> original size {to_inpaint_img.size}')
    #     #     semantic_repaired = semantic_repaired.resize(to_inpaint_img.size)
    #     # # mask retain in region only repairing
    #     # retain_mask = ~repair_mask
    #     # final_image = np.where(retain_mask[:, :, None], semantic_repaired, semantic_repaired)
    #     # final_image = np.where(retain_mask[:, :, None], coarse_repaired, semantic_repaired)
    #     ori_image_back_ground = np.where(mask[:, :, None], 0, image).astype(np.uint8)
    #     image_with_hole = ori_image_back_ground
    #     coarse_repaired = np.array(inpainter(Image.fromarray(ori_image_back_ground), Image.fromarray(
    #         mask.astype(np.uint8) * 255)))  # lama inpainting filling the black regions
    #     if mode != 1:
    #         inpaint_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    #         sd_to_inpaint_img = Image.fromarray(coarse_repaired)
    #         # print(f'SD inpainting Processing:')
    #         semantic_repaired = \
    #             self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask).images[0]
    #         semantic_repaired = np.array(semantic_repaired)
    #     else:
    #         semantic_repaired = coarse_repaired
    #
    #     if semantic_repaired.shape != image_with_hole.shape:
    #         print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
    #         h, w = image_with_hole.shape[:2]
    #         semantic_repaired = cv2.resize(semantic_repaired, (w, h), interpolation=cv2.INTER_LANCZOS4)
    #
    #     final_image = np.where(transformed_mask[:, :, None], transformed_image, semantic_repaired)
    #
    #     return final_image, image_with_hole, transformed_mask



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
    # def generated_refine_results(self, ori_img, ori_mask, coarse_input, target_mask,constrain_area,guidance_text,
    #                              guidance_scale, eta, contrast_beta=1.67, end_step=10, num_step=50, start_step=25,
    #                              feature_injection=True, FI_range=(900, 680), sim_thr=0.5, DIFT_LAYER_IDX=[0, 1, 2, 3],
    #                              use_mtsa=True,local_text_edit=True,local_ddpm=True, verbose=True , return_ori=False,):
    #     ori_mask = self.mask_reduce_dim(ori_mask)
    #     target_mask = self.mask_reduce_dim(target_mask)
    #     # ddpm_region_mask = self.mask_reduce_dim(ddpm_region_mask)
    #     self.controller.contrast_beta = contrast_beta
    #     # DDIM INVERSION
    #     shifted_mask, inverted_latent = self.DDIM_inversion_func(img=coarse_input, mask=target_mask,
    #                                                              prompt="",
    #                                                              num_step=num_step,
    #                                                              start_step=start_step,
    #                                                              ref_img=ori_img,verbose=verbose)  # ndarray mask
    #
    #     edit_gen_image, refer_gen_image, = self.Details_Preserving_regeneration(coarse_input, inverted_latent,
    #                                                                             guidance_text,
    #                                                                             target_mask, ori_mask,
    #                                                                             num_steps=num_step,
    #                                                                             start_step=start_step,
    #                                                                             end_step=end_step,
    #                                                                             guidance_scale=guidance_scale, eta=eta,
    #                                                                             feature_injection=feature_injection,
    #                                                                             FI_range=FI_range, sim_thr=sim_thr,
    #                                                                             DIFT_LAYER_IDX=DIFT_LAYER_IDX,
    #                                                                             use_mtsa=use_mtsa, ref_img=ori_img,
    #                                                                             verbose=verbose,
    #                                                                             local_text_edit=local_text_edit,local_ddpm=local_ddpm,cons_area= constrain_area)
    #     # assist_prompt = [obj_label]
    #     # expansion_step = 10
    #     # self.controller.contrast_beta = contrast_beta  # numpy input
    #
    #     # expand_target_mask = self.Prompt_guided_mask_expansion_func(img=edit_gen_image, mask=target_mask,
    #     #                                                      expand_forbit_region=constrain_area,
    #     #                                                      assist_prompt=assist_prompt,
    #     #                                                      num_step=expansion_step, start_step=1,sem_expansion=False,init_enhance=False,)
    #     # expand_target_mask[expand_target_mask>0] = 1
    #     # return edit_gen_image,expand_target_mask
    #     if not return_ori:
    #         return edit_gen_image
    #     return edit_gen_image,refer_gen_image
    def Reggio_refine_generation(self, ori_img, ori_mask, coarse_input, target_mask,guidance_text,
                                 guidance_scale, eta, contrast_beta=1.67, end_step=10, num_step=50, start_step=25,
                                 use_mtsa=True,local_text_edit=True,local_ddpm=True, verbose=True , return_ori=False,seed=42,draw_mask=None,use_gs=True, gs_scale=0.4,
                                 return_intermediates=False,context_guidance=1.0 ):
        seed_everything(seed)
        ori_mask = self.mask_reduce_dim(ori_mask)
        target_mask = self.mask_reduce_dim(target_mask)
        draw_mask = self.mask_reduce_dim(draw_mask)

        # ddpm_region_mask = self.mask_reduce_dim(ddpm_region_mask)
        self.controller.contrast_beta = contrast_beta
        # DDIM INVERSION
        shifted_mask, inverted_latent = self.DDIM_inversion_func(img=coarse_input, mask=target_mask,
                                                                 prompt="",
                                                                 num_step=num_step,
                                                                 start_step=start_step,
                                                                 ref_img=ori_img,verbose=verbose)  # ndarray mask

        edit_gen_image, refer_gen_image,intermediates = self.Details_Preserving_regeneration_v2(coarse_input, inverted_latent,
                                                                                guidance_text,
                                                                                target_mask, ori_mask, draw_mask,
                                                                                num_steps=num_step,
                                                                                start_step=start_step,
                                                                                end_step=end_step,
                                                                                guidance_scale=guidance_scale, eta=eta,
                                                                                use_mtsa=use_mtsa,
                                                                                verbose=verbose,
                                                                                local_text_edit=local_text_edit,local_ddpm=local_ddpm,
                                                                                use_gs=use_gs, gs_scale=gs_scale,
                                                                                return_intermediates=return_intermediates,
                                                                                context_guidance=context_guidance,
                                                                                )
        if intermediates is not None:
            self.save_intermediate_images_and_gif_v2(intermediates)
        if not return_ori:
            return edit_gen_image
        return edit_gen_image,refer_gen_image


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
    def Reggio_refine_generation_gs(self, ori_img, ori_mask, coarse_input, target_mask, constrain_area, guidance_text,
                                 guidance_scale, eta, contrast_beta=1.67, end_step=10, num_step=50, start_step=25,
                                 use_mtsa=True, local_text_edit=True, local_ddpm=True, verbose=True, return_ori=False,
                                 seed=42,use_gs=True, gs_scale=0.4,
                                  gs_bond_scale=0.4,):
        seed_everything(seed)
        ori_mask = self.mask_reduce_dim(ori_mask)
        target_mask = self.mask_reduce_dim(target_mask)

        # ddpm_region_mask = self.mask_reduce_dim(ddpm_region_mask)
        self.controller.contrast_beta = contrast_beta
        # DDIM INVERSION
        shifted_mask, inverted_latent = self.DDIM_inversion_func(img=coarse_input, mask=target_mask,
                                                                 prompt="",
                                                                 num_step=num_step,
                                                                 start_step=start_step,
                                                                 ref_img=ori_img, verbose=verbose)  # ndarray mask

        edit_gen_image, refer_gen_image, = self.Details_Preserving_regeneration(coarse_input, inverted_latent,
                                                                                   guidance_text,
                                                                                   target_mask, ori_mask,
                                                                                   num_steps=num_step,
                                                                                   start_step=start_step,
                                                                                   end_step=end_step,
                                                                                   guidance_scale=guidance_scale,
                                                                                   eta=eta,
                                                                                   use_mtsa=use_mtsa,
                                                                                   verbose=verbose,
                                                                                   local_text_edit=local_text_edit,
                                                                                   local_ddpm=local_ddpm,
                                                                                   cons_area=constrain_area,
                                                                                    use_gs=use_gs, gs_scale=gs_scale,
                                                                                    gs_bond_scale=gs_bond_scale,
                                                                                   )

        if not return_ori:
            return edit_gen_image
        return edit_gen_image, refer_gen_image
    # def Magic_Editing_Baseline_SV3D(self, original_image, transformed_image, prompt, INP_prompt,
    #                                 seed, guidance_scale, num_step, max_resolution, mode, dilate_kernel_size,
    #                                 start_step,
    #                                 eta=0, use_mask_expansion=True, contrast_beta=1.67, mask_threshold=0.1,
    #                                 mask_threshold_target=0.1, end_step=10,
    #                                 feature_injection=True, FI_range=(900, 680), sim_thr=0.5,
    #                                 DIFT_LAYER_IDX=[0, 1, 2, 3], use_sdsa=True, select_mask=None,
    #                                 ):
    #     seed_everything(seed)
    #
    #     img = original_image
    #     trans_img = transformed_image
    #     if select_mask is None:
    #         print(f'generating original image mask')
    #         mask, in_w, img = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution])
    #         trans_img = cv2.resize(trans_img, (in_w, in_w))
    #         print(f'generating transformed image mask')
    #         trans_mask, _, trans_img = self.get_mask_from_rembg(trans_img)
    #     else:  # resize + mask resize
    #         print(f'input selected mask : Yes')
    #         img, in_w = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution], need_mask=False)
    #         w, h = img.shape[:2]
    #         mask = cv2.resize(select_mask, (h, w))
    #         if mask.ndim == 3:
    #             mask = mask[:, :, 0]
    #         trans_img = cv2.resize(trans_img, (in_w, in_w))
    #         print(f'generating transformed image mask')
    #         trans_mask, _, trans_img = self.get_mask_from_rembg(trans_img)
    #
    #     target_mask = mask
    #     mask_to_use = self.dilate_mask(mask, dilate_kernel_size)  # dilate for better expansion mask
    #
    #     # mask expansion
    #     if use_mask_expansion:
    #         self.controller.contrast_beta = contrast_beta  # numpy input
    #         expand_mask, _ = self.DDIM_inversion_func(img=img, mask=mask_to_use, prompt="",
    #                                                   guidance_scale=1, num_step=10,
    #                                                   start_step=2,
    #                                                   roi_expansion=True,
    #                                                   mask_threshold=mask_threshold,
    #                                                   post_process='hard', )
    #     else:
    #         expand_mask = mask_to_use
    #
    #     img_preprocess, inpaint_mask_vis, shifted_mask = self.replace_with_SV3D_targets_inpainting(img,
    #                                                                                                trans_img,
    #                                                                                                expand_mask,
    #                                                                                                trans_mask,
    #                                                                                                target_mask,
    #                                                                                                mode,
    #                                                                                                self.inpainter,
    #                                                                                                INP_prompt, )
    #     mask_to_use = shifted_mask
    #     # mask_to_use = self.dilate_mask(shifted_mask, dilate_kernel_size) * 255  # dilate for better expansion mask
    #     ori_img = img
    #     img = img_preprocess
    #     if isinstance(img, np.ndarray):
    #         img_preprocess = Image.fromarray(img_preprocess)
    #     self.controller.contrast_beta = contrast_beta
    #     expand_shift_mask, inverted_latent = self.DDIM_inversion_func(img=img, mask=mask_to_use,
    #                                                                   prompt="",
    #                                                                   guidance_scale=1,
    #                                                                   num_step=num_step,
    #                                                                   start_step=start_step,
    #                                                                   roi_expansion=True,
    #                                                                   mask_threshold=mask_threshold_target,
    #                                                                   post_process='hard',
    #                                                                   use_mask_expansion=False,
    #                                                                   ref_img=ori_img)  # ndarray mask
    #
    #     edit_gen_image, refer_gen_image, target_mask, = self.Details_Preserving_regeneration(img,
    #                                                                                          inverted_latent,
    #                                                                                          prompt,
    #                                                                                          expand_shift_mask,
    #                                                                                          target_mask,
    #                                                                                          num_steps=num_step,
    #                                                                                          start_step=start_step,
    #                                                                                          end_step=end_step,
    #                                                                                          guidance_scale=guidance_scale,
    #                                                                                          eta=eta,
    #                                                                                          roi_expansion=True,
    #                                                                                          mask_threshold=mask_threshold_target,
    #                                                                                          post_process='hard',
    #                                                                                          feature_injection=feature_injection,
    #                                                                                          FI_range=FI_range,
    #                                                                                          sim_thr=sim_thr,
    #                                                                                          dilate_kernel_size=dilate_kernel_size,
    #                                                                                          DIFT_LAYER_IDX=DIFT_LAYER_IDX,
    #                                                                                          use_sdsa=use_sdsa,
    #                                                                                          ref_img=ori_img)
    #     # source_mask = Image.fromarray(source_mask)
    #     inpaint_mask_vis = Image.fromarray(inpaint_mask_vis)
    #     return [edit_gen_image], [refer_gen_image], [img_preprocess], [inpaint_mask_vis], [target_mask]
    #
    # def softmax(self,x, temperature=1.0):
    #     # 计算每个元素的指数值，并进行温度调整
    #     exp_x = torch.exp(x / temperature)
    #     # 计算总和以进行归一化
    #     sum_exp_x = exp_x.sum(dim=-1, keepdim=True)
    #     # 返回归一化后的结果
    #     return exp_x / sum_exp_x
    #
    # def get_matching_score(self, candidate_sim, pos_len, temperature=0.2):
    #     """
    #     carefully designed
    #     """
    #     # candidate_sim = self.softmax(candidate_sim, temperature=temperature)
    #     positive_sim = candidate_sim[:, :pos_len].sum(dim=-1)
    #     negative_sim = candidate_sim[:, pos_len:]
    #     # # 对 negative_sim 进行手动实现的 Softmax 锐化处理
    #     negative_sim_sharpened = self.softmax(negative_sim, temperature=temperature)
    #     negative_sim_sum = negative_sim_sharpened.max(dim=-1)[0]
    #     # negative_sim_sum = negative_sim.sum()
    #     #
    #     # # 计算 matching_score
    #     matching_score = 1- negative_sim_sum + positive_sim
    #
    #     return matching_score

    # @torch.no_grad()
    # def sd_inpaint_results_filter(self, img_lists, mask, class_text,):
    #     # [ndarray,ndarray,.....]
    #     neg_text_list = [class_text] + ['object', 'person', 'texts', ]
    #     # pos_text_list = ['background','empty scene']
    #     pos_text_list = ['empty scene']
    #     pos_len = len(pos_text_list)
    #     # 通过mask区域的boundary box裁剪图像
    #     preprocessed_imgs = self.crop_image_with_mask(img_lists, mask)
    #     # preprocessed_imgs = self.pre_process_with_mask(img_lists, mask)
    #     img_lst = np.array(img_lists)
    #     # image = self.clip_process(cropped_img).unsqueeze(0).to(self.device)
    #     stack_images = torch.stack(
    #         [self.clip_process(self.numpy_to_pil(crop_img)[0]).to(self.device) for crop_img in
    #          preprocessed_imgs])
    #
    #     # 对每个词进行编码
    #     text_tokens_neg = clip.tokenize(neg_text_list).to(self.device)
    #     text_tokens_pos = clip.tokenize(pos_text_list).to(self.device)
    #     text_tokens = torch.cat([text_tokens_pos, text_tokens_neg])
    #
    #     # 计算图像的特征向量
    #     image_features = self.clip.encode_image(stack_images)
    #     # Calculate the standard deviation of the image embeddings to measure semantic diversity
    #     # embeddings_std = torch.std(image_features[1:], dim=0).mean()
    #     # #uncertainty filter
    #     # if embeddings_std.item() >0.20:
    #     #     # print(f'choose lama inpainting results')
    #     #     # return img_lists[0]
    #
    #     # 计算每个词的特征向量
    #     text_features = self.clip.encode_text(text_tokens)
    #     similarities = torch.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
    #     before_similarities = similarities[-1, :]  # ori img sim
    #     similarities = similarities[:-1, :]
    #     # class filter
    #     valid_idx_cls = similarities.argmax(dim=1) < pos_len
    #     # discrepency filter
    #     pre_matching_score = self.get_matching_score(before_similarities[None,], pos_len)
    #     matching_score = self.get_matching_score(similarities, pos_len)
    #     valid_idx_dis = matching_score > pre_matching_score
    #     valid_idx_not_black = torch.tensor([False in (p_im[0]==(255,255,255)) for p_im in preprocessed_imgs[:-1]],dtype=valid_idx_dis.dtype,device=self.device)
    #     valid_idx = valid_idx_cls & valid_idx_dis &valid_idx_not_black
    #     # valid_idx = valid_idx_dis & valid_idx_not_black
    #     # if valid_idx.sum() == 0:
    #     #     print(f'choose lama inpainting results')
    #     #     return img_lists[0]
    #     # valid_idx = valid_idx_cls & valid_idx_dis
    #     valid_index = np.array([idx for idx, i in enumerate(valid_idx) if i])
    #     try:
    #         candidate_img = img_lst[valid_index]
    #         matching_score_valid = matching_score[valid_idx]
    #         final_match_indices = torch.argmax(matching_score_valid, dim=0).item()
    #         # if valid_idx[0] and final_match_indices == 0:  # default lama always good, but not real enough
    #         #     print(f'choose lama inpainting results')
    #         # else:
    #         #     print(f'choose sd inpainting results')
    #         final_inpainting = candidate_img[int(final_match_indices)]
    #         # self.temp_view_img(final_inpainting)
    #         return img_lists[0],final_inpainting
    #     except:
    #         print(f'no results left after filter, thus choosing lama results')
    #         return img_lists[0],img_lists[0]

    # @torch.no_grad()
    # def sd_inpaint_results_caption_filter(self, img_lists, mask, class_text, ):
    #
    #     def get_matching_score_tag(pos_features, neg_features, target_features):
    #         sim_pos = torch.matmul(target_features, pos_features.T).mean(dim=-1)
    #         sim_neg = torch.matmul(target_features, neg_features.T).mean(dim=-1)
    #         return sim_pos - sim_neg
    #     # [ndarray,ndarray,.....]
    #     neg_text_list = [class_text] + ['an object','an item', 'a person', 'texts',]
    #     pos_text_list = ['a background','an empty_scene','surroundings','environment', 'a scenery' ]
    #     base_format = 'a photo of'
    #     neg_text_list = [f'{base_format} {i}' for i in neg_text_list]
    #     pos_text_list = [f'{base_format} {i}' for i in pos_text_list]
    #
    #     preprocessed_imgs = self.crop_image_with_mask(img_lists, mask) #crop is enough
    #     # preprocessed_imgs = self.pre_process_with_mask(img_lists, mask)
    #     # preprocessed_imgs = self.crop_image_with_mask(preprocessed_imgs,mask)
    #     img_lst = np.array(img_lists)
    #     pil_img_lst = \
    #         [preprocess_tag_pil_img(self.numpy_to_pil(crop_img)[0],self.device) for crop_img in preprocessed_imgs]
    #     # stack_images = torch.stack(
    #     #     [self.clip_process(crop_img_pil).to(self.device) for crop_img_pil in pil_img_lst])
    #     # Tag2Text
    #     res_tag2text_list = [inference_tag2text(image_pillow, self.tag2text, 'None')[2] for image_pillow in pil_img_lst]
    #     #TODO:思考一下如何控制tag2text生成文本数量，同时如何对存在object的单词进行惩罚，另外要防止幻觉该怎么做？
    #     # 对每个词进行编码
    #     text_feature_neg = self.clip.encode_text(clip.tokenize(neg_text_list).to(self.device))
    #     text_feature_pos =self.clip.encode_text( clip.tokenize(pos_text_list).to(self.device))
    #     #normalize and encode
    #     text_feature_neg = text_feature_neg /text_feature_neg.norm(dim=1, keepdim=True)
    #     text_feature_pos = text_feature_pos / text_feature_pos.norm(dim=1, keepdim=True)
    #     text_feature_can = self.clip.encode_text(clip.tokenize(res_tag2text_list).to(self.device))
    #     text_feature_can = text_feature_can / text_feature_can.norm(dim=1, keepdim=True)
    #     #TODO: multiple text candidates
    #     tag_sim_scores = get_matching_score_tag(pos_features=text_feature_pos,neg_features=text_feature_neg,target_features=text_feature_can)
    #     lama_tag_score = tag_sim_scores[0]
    #     ori_tag_score = tag_sim_scores[-1]
    #     sd_tag_scores = tag_sim_scores[1:-1]
    #
    #     valid_idx_dis = sd_tag_scores > ori_tag_score
    #     #nsfw remove black results
    #     valid_idx_not_black = torch.tensor([False in (p_im[0] == (255, 255, 255)) for p_im in preprocessed_imgs[1:-1]],
    #                                        dtype=valid_idx_dis.dtype, device=self.device)
    #     valid_idx = valid_idx_not_black & valid_idx_dis
    #     valid_index = np.array([idx for idx, i in enumerate(valid_idx) if i])
    #     try:
    #         candidate_img = img_lst[valid_index]
    #         matching_score_valid = sd_tag_scores[valid_idx]
    #         res_tag2text_list_valid = np.array(res_tag2text_list[1:-1])[valid_index]
    #         final_match_indices = torch.argmax(matching_score_valid, dim=0).item()
    #         if matching_score_valid[final_match_indices] > max(0.02,lama_tag_score):#strong condition to decide whether reject sd results
    #             final_inpainting = candidate_img[int(final_match_indices)]
    #             final_inpainting_caption = res_tag2text_list_valid[int(final_match_indices)]
    #             print(f'choosing inpainting results with caption:"{final_inpainting_caption}"')
    #             # self.temp_view_img(final_inpainting)
    #             return final_inpainting
    #         else:
    #             print(f'1:no results left after filter, thus choosing lama results')
    #             return img_lists[0]
    #     except:
    #         print(f'0:no results left after filter, thus choosing lama results')
    #         return img_lists[0]

    @torch.no_grad()
    def sd_only_results_caption_filter(self, img_lists, mask, class_text, ):

        def get_matching_score_tag(pos_features, neg_features, target_features):
            sim_pos = torch.matmul(target_features, pos_features.T).mean(dim=-1)
            sim_neg = torch.matmul(target_features, neg_features.T).mean(dim=-1)
            return sim_pos - sim_neg

        # [ndarray,ndarray,.....]
        neg_text_list = [class_text] + ['an object', 'an item', 'a person', 'texts', ]
        pos_text_list = ['a background', 'an empty_scene', 'surroundings', 'environment', 'a scenery']
        base_format = 'a photo of'
        neg_text_list = [f'{base_format} {i}' for i in neg_text_list]
        pos_text_list = [f'{base_format} {i}' for i in pos_text_list]

        preprocessed_imgs = self.crop_image_with_mask(img_lists, mask)  # crop is enough
        # preprocessed_imgs = self.pre_process_with_mask(img_lists, mask)
        # preprocessed_imgs = self.crop_image_with_mask(preprocessed_imgs,mask)
        img_lst = np.array(img_lists)
        pil_img_lst = \
            [preprocess_tag_pil_img(self.numpy_to_pil(crop_img)[0], self.device) for crop_img in preprocessed_imgs]
        # stack_images = torch.stack(
        #     [self.clip_process(crop_img_pil).to(self.device) for crop_img_pil in pil_img_lst])
        # Tag2Text
        res_tag2text_list = [inference_tag2text(image_pillow, self.tag2text, 'None')[2] for image_pillow in pil_img_lst]
        # TODO:思考一下如何控制tag2text生成文本数量，同时如何对存在object的单词进行惩罚，另外要防止幻觉该怎么做？
        # 对每个词进行编码
        text_feature_neg = self.clip.encode_text(clip.tokenize(neg_text_list).to(self.device))
        text_feature_pos = self.clip.encode_text(clip.tokenize(pos_text_list).to(self.device))
        # normalize and encode
        text_feature_neg = text_feature_neg / text_feature_neg.norm(dim=1, keepdim=True)
        text_feature_pos = text_feature_pos / text_feature_pos.norm(dim=1, keepdim=True)
        text_feature_can = self.clip.encode_text(clip.tokenize(res_tag2text_list).to(self.device))
        text_feature_can = text_feature_can / text_feature_can.norm(dim=1, keepdim=True)
        # TODO: multiple text candidates
        tag_sim_scores = get_matching_score_tag(pos_features=text_feature_pos, neg_features=text_feature_neg,
                                                target_features=text_feature_can)

        sd_tag_scores = tag_sim_scores
        final_match_indices = torch.argmax(sd_tag_scores, dim=0).item()
        final_inpainting = img_lst[int(final_match_indices)]
        final_inpainting_caption = res_tag2text_list[int(final_match_indices)]
        print(f'choosing inpainting results with caption:"{final_inpainting_caption}"')
        # self.temp_view_img(final_inpainting)
        return final_inpainting

    @torch.no_grad()
    def multi_inpainting_results_filter(self, img_lists, mask, class_text, ):

        def get_matching_score_tag(pos_features, neg_features, target_features):
            sim_pos = torch.matmul(target_features, pos_features.T).mean(dim=-1)
            sim_neg = torch.matmul(target_features, neg_features.T).mean(dim=-1)
            return sim_pos - sim_neg

        # [ndarray,ndarray,.....]
        neg_text_list = [class_text] + ['an object', 'an item', 'a person', 'texts', ]
        pos_text_list = ['a background', 'an empty_scene', 'surroundings', 'environment', 'a scenery']
        base_format = 'a photo of'
        neg_text_list = [f'{base_format} {i}' for i in neg_text_list]
        pos_text_list = [f'{base_format} {i}' for i in pos_text_list]

        preprocessed_imgs = self.crop_image_with_mask(img_lists, mask)  # crop is enough
        # preprocessed_imgs = self.pre_process_with_mask(img_lists, mask)
        # preprocessed_imgs = self.crop_image_with_mask(preprocessed_imgs,mask)
        img_lst = np.array(img_lists)
        pil_img_lst = \
            [preprocess_tag_pil_img(self.numpy_to_pil(crop_img)[0], self.device) for crop_img in preprocessed_imgs]
        # stack_images = torch.stack(
        #     [self.clip_process(crop_img_pil).to(self.device) for crop_img_pil in pil_img_lst])
        # Tag2Text
        res_tag2text_list = [inference_tag2text(image_pillow, self.tag2text, 'None')[2] for image_pillow in pil_img_lst]
        # TODO:思考一下如何控制tag2text生成文本数量，同时如何对存在object的单词进行惩罚，另外要防止幻觉该怎么做？
        # 对每个词进行编码
        text_feature_neg = self.clip.encode_text(clip.tokenize(neg_text_list).to(self.device))
        text_feature_pos = self.clip.encode_text(clip.tokenize(pos_text_list).to(self.device))
        # normalize and encode
        text_feature_neg = text_feature_neg / text_feature_neg.norm(dim=1, keepdim=True)
        text_feature_pos = text_feature_pos / text_feature_pos.norm(dim=1, keepdim=True)
        text_feature_can = self.clip.encode_text(clip.tokenize(res_tag2text_list).to(self.device))
        text_feature_can = text_feature_can / text_feature_can.norm(dim=1, keepdim=True)
        # TODO: multiple text candidates
        tag_sim_scores = get_matching_score_tag(pos_features=text_feature_pos, neg_features=text_feature_neg,
                                                target_features=text_feature_can)

        sd_tag_scores = tag_sim_scores
        final_match_indices = torch.argmax(sd_tag_scores, dim=0).item()
        final_inpainting = img_lst[int(final_match_indices)]
        final_inpainting_caption = res_tag2text_list[int(final_match_indices)]
        print(f'choosing inpainting results with caption:"{final_inpainting_caption}"')
        # self.temp_view_img(final_inpainting)
        return final_inpainting


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

    # def Reggio_inpainting_func(self, ori_img, exp_mask, inpainter=None,
    #                            inp_prompt=None, inp_prompt_negative=None, samples_per_time=10,mode='lama_only'):
    #     # lama & SD
    #     if isinstance(ori_img, Image.Image):
    #         ori_img = np.array(ori_img)
    #     if inp_prompt is None:
    #         inp_prompt = 'a photo of a background, a photo of an empty place'
    #     if exp_mask.ndim == 3 and exp_mask.shape[2] == 3:
    #         exp_mask = cv2.cvtColor(exp_mask, cv2.COLOR_BGR2GRAY)
    #
    #     # prepare background
    #     # box mask do not influence sd inpaint
    #     # exp_mask = self.box_mask(exp_mask)
    #
    #     exp_mask = (exp_mask > 0).astype(bool)
    #     ori_image_back_ground = np.where(exp_mask[:, :, None], 0, ori_img)
    #     image_with_hole = ori_image_back_ground
    #
    #     if 'lama' in mode: #lama_only lama_sd sd
    #         coarse_repaired = inpainter(ori_image_back_ground, exp_mask.astype(np.uint8) * 255,)#lama
    #     else:
    #         coarse_repaired = ori_img # no black filled
    #
    #     if mode == 'lama_only':
    #         if coarse_repaired.shape[:2] != image_with_hole.shape[:2]:
    #             h, w = image_with_hole.shape[:2]
    #             coarse_repaired = cv2.resize(coarse_repaired, (w, h), interpolation=cv2.INTER_LANCZOS4)
    #         return coarse_repaired
    #     else:
    #         inpaint_mask = Image.fromarray(exp_mask.astype(np.uint8) * 255)
    #         sd_to_inpaint_img = Image.fromarray(coarse_repaired)
    #         # print(f'SD inpainting Processing:')
    #         semantic_repaired = \
    #             self.sd_inpainter(prompt=inp_prompt, image=sd_to_inpaint_img, mask_image=inpaint_mask,
    #                               guidance_scale=7.5, eta=1.0,
    #                               num_inference_steps=10, negative_prompt=inp_prompt_negative,
    #                               num_images_per_prompt=samples_per_time,).images
    #         # resize
    #         # semantic_repaired_new = [coarse_repaired]
    #         semantic_repaired_new = []
    #         h, w = image_with_hole.shape[:2]
    #         for sd_inp_res in semantic_repaired:
    #             sd_inp_res = np.array(sd_inp_res)
    #             if sd_inp_res.shape != image_with_hole.shape:
    #                 # print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
    #                 sd_inp_res = cv2.resize(sd_inp_res, (w, h), interpolation=cv2.INTER_LANCZOS4)
    #             semantic_repaired_new.append(sd_inp_res)
    #         return semantic_repaired_new
    # def sd_local_inpaint(self,image,mask_image,
    #                      guidance_scale=1.0,eta=1.0,num_inference_step=50,init_start_step=15,num_samples=1,local_ddpm=True):
    #     # DDIM INVERSION
    #     shifted_mask, inverted_latents = self.DDIM_inversion_func(img=image, mask=mask_image,
    #                                                              prompt="",
    #                                                              num_step=num_inference_step,
    #                                                              start_step=init_start_step,verbose=True
    #                                                              )  # ndarray mask
    #     start_latent = inverted_latents[-1]
    #     shifted_mask[shifted_mask > 0.0] = 1
    #     shifted_mask_tensor = torch.tensor(shifted_mask,device=start_latent.device,dtype=start_latent.dtype)
    #     local_var_region = F.interpolate(shifted_mask_tensor.unsqueeze(0).unsqueeze(0),
    #                                           (start_latent.shape[2], start_latent.shape[3]),
    #                                           mode='nearest').squeeze(0, 1)
    #     self.controller.reset()
    #     self.controller.log_mask = False
    #     gen_images = self.denoise(
    #         prompt="",
    #         batch_size=num_samples,
    #         latents=start_latent,
    #         guidance_scale=guidance_scale,
    #         num_inference_steps=num_inference_step,
    #         start_step=init_start_step,
    #         eta=eta,
    #         mask = local_var_region,
    #         local_ddpm=local_ddpm,
    #     )
    #     self.controller.reset()
    #     gen_images = gen_images.cpu().permute(0, 2, 3, 1).numpy()
    #     gen_images = (gen_images * 255).astype(np.uint8)
    #     return gen_images

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
    def inpaint_and_repaint_func(self, ori_img, exp_mask, inpainter=None,num_samples=1,eta=1.0,init_start_step=15,simple=False,k=21,sigma=0):

        # lama & SD local refinement
        if isinstance(ori_img, Image.Image):
            ori_img = np.array(ori_img)
        if exp_mask.ndim == 3 and exp_mask.shape[2] == 3:
            exp_mask = cv2.cvtColor(exp_mask, cv2.COLOR_BGR2GRAY)

        exp_mask = (exp_mask > 0).astype(bool)
        ori_image_back_ground = np.where(exp_mask[:, :, None], 0, ori_img)
        image_with_hole = ori_image_back_ground

        coarse_repaired = inpainter(ori_image_back_ground, exp_mask.astype(np.uint8) * 255,)#lama
        print(f'simple is {simple}')
        if not simple :
            self.temp_view_img(ori_img)
            self.temp_view_img(coarse_repaired)
            self.temp_view(exp_mask)
        # self.save_img(coarse_repaired, "/data/Hszhu/BrushNet/examples/inp_img.png")
        inpaint_mask = exp_mask.astype(np.uint8) * 255

        #TODO
        fine_grained_repaired = self.sd_local_inpaint(image=coarse_repaired,mask_image=inpaint_mask,
                                                      guidance_scale=1.0,eta=eta,num_inference_step=50,init_start_step=init_start_step,
                                                      num_samples=num_samples,local_ddpm=True)
        blended = True
        semantic_repaired_new = []
        h, w = image_with_hole.shape[:2]
        # self.visualize_images(fine_grained_repaired)
        for sd_repaint_res in fine_grained_repaired:
            # sd_repaint_res = sd_repaint_res
            if sd_repaint_res.shape != image_with_hole.shape:
                # print(f'inpainted image {semantic_repaired.shape} -> original size {image_with_hole.shape}')
                sd_repaint_res = cv2.resize(sd_repaint_res, (w, h), interpolation=cv2.INTER_LANCZOS4)
            if blended:
                image_np = sd_repaint_res
                mask_np = np.expand_dims(exp_mask, axis=2).repeat(3, axis=2).astype(np.uint8)

                # blur, you can adjust the parameters for better performance
                mask_blurred = cv2.GaussianBlur(mask_np * 255, (k, k), sigma) / 255
                mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

                image_pasted = ori_img * (1 - mask_np) + image_np * mask_np
                image_pasted = image_pasted.astype(image_np.dtype)

            semantic_repaired_new.append(image_pasted)

        vis_img_list = semantic_repaired_new
        self.visualize_images_column(vis_img_list)
        return semantic_repaired_new

    # def expansion_and_inpainting_func(self, img, mask_pools,label_list,max_resolution=768,
    #                               seed=42,expansion_step=4, contrast_beta=1.67,max_try_times=10,samples_per_time=10,
    #                               assist_prompt="",mode='lama_only',sem_expansion=False):
    #     """
    #     semantic expansion and sd inpainting
    #
    #     """
    #     assert mode in ['lama_only','sd','lama_sd']
    #     # seed_everything(seed)
    #     np.random.seed(int(time.time()))
    #     random_seed = np.random.randint(0, 2 ** 32 - 1)
    #     my_seed_everything(random_seed)
    #     expansion_mask_list = []
    #     inpainting_results_list = []
    #     # lama_inpaint_results_list = []
    #     if isinstance(assist_prompt, str):
    #         assist_prompt = [item.strip() for item in assist_prompt.split(',')]
    #     #resize to avoid memory promblem in expansion process
    #     img, in_w = self.get_mask_from_rembg(img, size=[max_resolution, max_resolution], need_mask=False)
    #     w, h = img.shape[:2]
    #     expand_forbid_regions = np.zeros_like(mask_pools[0]).astype(np.uint8)
    #     for i in range(len(mask_pools)):
    #         msk = mask_pools[i].astype(np.uint8)
    #         msk[msk>0]=1
    #         expand_forbid_regions += msk
    #     expand_forbid_regions[expand_forbid_regions>0]=1
    #
    #     for idx,select_mask in enumerate(mask_pools):
    #         obj_text = label_list[idx]
    #         mask = cv2.resize(select_mask, (h, w))
    #         if idx==0:
    #             expand_forbid_regions = cv2.resize(expand_forbid_regions, (h, w))
    #             if mask.ndim == 3:
    #                 mask = mask[:, :, 0]
    #                 expand_forbid_regions = expand_forbid_regions[:,:,0]
    #         elif mask.ndim == 3:
    #             mask = mask[:, :, 0]
    #
    #         self.controller.contrast_beta = contrast_beta  # numpy input
    #         print(f'idx:{idx} | proceeding mask expansion:')
    #         expand_mask = self.Simple_object_aware_dilation(mask=mask,
    #                                                         expand_forbit_region=expand_forbid_regions, )
    #         expand_mask = self.Prompt_guided_mask_expansion_func(img=img, mask= expand_mask,expand_forbit_region = expand_forbid_regions,
    #                                                              assist_prompt=assist_prompt,
    #                                                              num_step=expansion_step, start_step=1,sem_expansion=sem_expansion,init_enhance=True)
    #
    #         expansion_mask_list.append(expand_mask)
    #         INP_prompt = 'a photo of a background, a photo of an empty place'
    #         INP_prompt_negative = f'object,{obj_text},shadow,text'
    #         inpainting_results_temp = []
    #         print(f'idx:{idx} | proceeding inpainting:')
    #         if 'sd' not in mode:
    #             lama_result = self.Reggio_inpainting_func(img, expand_mask, self.inpainter, INP_prompt, INP_prompt_negative, samples_per_time,mode)
    #             inpainting_results_list.append(lama_result)
    #         else:
    #             for try_time in range(max_try_times):
    #                 np.random.seed(int(time.time()))
    #                 random_seed = np.random.randint(0, 2 ** 32 - 1)
    #                 my_seed_everything(random_seed)
    #                 # inpainting_result,lama_result = self.Reggio_inpainting_func(img, expand_mask, self.inpainter, INP_prompt,
    #                 #                                                 INP_prompt_negative, samples_per_time, mode)
    #                 inpainting_result = self.Reggio_inpainting_func(img, expand_mask, self.inpainter, INP_prompt, INP_prompt_negative, samples_per_time,mode)
    #                 inpainting_results_temp.extend(inpainting_result)
    #             # Filter
    #             # inpainting_results_temp.insert(0,lama_result)
    #             # inpainting_results_temp.append(img)
    #             # best_sd_inp_result = self.sd_inpaint_results_caption_filter(inpainting_results_temp, expand_mask, obj_text)
    #             best_sd_inp_result = self.sd_only_results_caption_filter(inpainting_results_temp, expand_mask,
    #                                                                         obj_text)
    #             inpainting_results_list.append(best_sd_inp_result)
    #             # lama_inpaint_results_list.append(lama_result)
    #
    #     return expansion_mask_list,inpainting_results_list
    def save_mask(self,mask, dst_path):
        cv2.imwrite(dst_path, mask)  # 将mask保存为png图片 (注意：mask是二值图，乘以255以得到可见的结果)

    def save_img(self,img, dst_path):
        cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 将图片保存为png格式 (注意：需转换为BGR格式)

    def expansion_and_inpainting_func_exp(self, img, mask_pools,max_resolution=768,
                                  seed=42,expansion_step=4, contrast_beta=1.67,samples_per_time=10,
                                  assist_prompt="",sem_expansion=True):
        """
        semantic expansion and sd inpainting

        """
        # seed_everything(seed)
        np.random.seed(int(time.time()))
        random_seed = np.random.randint(0, 2 ** 32 - 1)
        my_seed_everything(random_seed)
        expansion_mask_list = []
        inpainting_results_list = []
        # lama_inpaint_results_list = []
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
            expand_mask = self.Simple_object_aware_dilation(mask=mask,
                                                            expand_forbit_region=expand_forbid_regions, )
            expand_mask = self.Prompt_guided_mask_expansion_func(img=img, mask=expand_mask,
                                                                 expand_forbit_region=expand_forbid_regions,
                                                                 assist_prompt=assist_prompt,
                                                                 num_step=expansion_step, start_step=1,
                                                                 sem_expansion=sem_expansion, init_enhance=False)

            expansion_mask_list.append(expand_mask)
            # self.save_mask(expand_mask,"/data/Hszhu/BrushNet/examples/exp_mask.png")
            print(f'idx:{idx} | proceeding inpainting:')
            torch.cuda.synchronize()
            inpaint_result = self.inpaint_and_repaint_func(img, expand_mask, self.inpainter,samples_per_time,
                                                             eta=1.0,init_start_step=30,simple=False)
            inpainting_results_list.append(inpaint_result)

        return expansion_mask_list,inpainting_results_list
    def mask_completion(self, img, mask_pools,label_list,max_resolution=768,
                                  seed=42,expansion_step=4, contrast_beta=1.67,max_try_times=10,samples_per_time=10,
                                  assist_prompt="",mode='lama_only'):
        assert mode in ['lama_only','sd','lama_sd']
        # seed_everything(seed)
        np.random.seed(int(time.time()))
        random_seed = np.random.randint(0, 2 ** 32 - 1)
        my_seed_everything(random_seed)
        expansion_mask_list = []
        inpainting_results_list = []
        # lama_inpaint_results_list = []
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
            assist_prompt = [obj_text]
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
                                                                 num_step=expansion_step, start_step=1,sem_expansion=False,init_enhance=False)
            expansion_mask_list.append(expand_mask)

        return expansion_mask_list

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
    def Simple_object_aware_dilation(self, mask,expand_forbit_region,):

        # other_object_masks = np.where(mask, 0, expand_forbit_region)
        dilated_mask = self.dilate_mask(mask,30)
        # valid_dilated_mask = np.where(other_object_masks,0,dilated_mask)
        # return valid_dilated_mask
        return dilated_mask
    @torch.no_grad()
    def Prompt_guided_mask_expansion_func(self, img, mask, assist_prompt,expand_forbit_region,
                                          num_step, start_step=0,
                                          use_mask_expansion=True,sem_expansion=False,init_enhance=True,
                                          ):
        source_image = self.preprocess_image(img, self.device)
        # reference mask prepare

        # x, y, w, h = cv2.boundingRect(self.dilate_mask(mask, 30))
        # local_focus = np.zeros_like(mask)
        # local_focus[y: y + h, x: x + w] = 255
        mask = self.prepare_controller_ref_mask(mask, use_mask_expansion)
        local_focus = self.prepare_local_region(mask, self.prepare_controller_ref_mask(expand_forbit_region, False))
        # local_focus = self.prepare_controller_ref_mask(local_focus, False)
        forbit_exp = self.prepare_controller_ref_mask(expand_forbit_region,False)
        if not sem_expansion:
            forbit_exp = torch.where(mask.bool(),0,forbit_exp)
        else:#sem expansion should focus rather than self regions
            forbit_exp = forbit_exp
        self.controller.local_focus_box = local_focus
        self.controller.forbit_expand_area = forbit_exp
        if len(assist_prompt) == 1 and assist_prompt[0] == "":
            assist_prompt = None
        self.controller.assist_len = len(assist_prompt)
        if init_enhance:
            self.controller.is_locating = True
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



    def prepare_local_region(self, shifted_mask, cons_area):
        shifted_mask[shifted_mask>0] = 1
        cons_area[cons_area>0] = 1
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
        jitter_x = int(0.1 * w)
        jitter_y = int(0.1 * h)

        # 随机移动左上角和右下角
        new_x_min = max(0, x_min - jitter_x)
        new_y_min = max(0, y_min - jitter_y)
        new_x_max = min(x_max + jitter_x, shifted_mask.shape[1] - 1)
        new_y_max = min(y_max + jitter_y, shifted_mask.shape[0] - 1)

        # 3. 生成新的编辑区域 (覆盖 jitter 后的范围)
        local_edit_region = torch.zeros_like(shifted_mask)
        local_edit_region[new_y_min:new_y_max + 1, new_x_min:new_x_max + 1] = 1

        # 4. 与 feasible_domains 做与运算，确保区域在可行范围内
        final_mask = local_edit_region * feasible_regions

        return final_mask
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

    def prepare_surrounding_mask_gs(self, shifted_mask, cons_area,obj_mask):
        bool_gaussian_region = shifted_mask.clone()
        bool_gaussian_region[bool_gaussian_region>0] = 1
        final_mask = shifted_mask * (1 - cons_area) *(1-obj_mask)
        return final_mask

    @torch.no_grad()
    def prepare_various_mask(self, shifted_mask, ori_mask, sup_res_w, sup_res_h, init_code, cons_area):

        shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
        ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
        cons_area_tensor = self.prepare_tensor_mask(cons_area,sup_res_w,sup_res_h)

        self.temp_view(shifted_mask_tensor.cpu().numpy(), 'mtsa mask')
        cons_area_tensor = cons_area_tensor - ori_mask_tensor

        local_edit_background = self.prepare_surrounding_mask(shifted_mask_tensor, cons_area_tensor,rate=0.1)


        local_edit_background[local_edit_background > 0.0] = 1
        self.temp_view(local_edit_background.cpu().numpy(),'ddpm mask')
        local_edit_foreground = F.interpolate( shifted_mask_tensor.unsqueeze(0).unsqueeze(0),
                                  (shifted_mask.shape[0], shifted_mask.shape[1]),
                                           mode='nearest').squeeze(0, 1)
        shifted_mask_tensor = F.interpolate(shifted_mask_tensor.unsqueeze(0).unsqueeze(0),
                                          (init_code.shape[2], init_code.shape[3]),
                                          mode='nearest').squeeze(0, 1)
        local_edit_background = F.interpolate(local_edit_background.unsqueeze(0).unsqueeze(0),
                                           (init_code.shape[2], init_code.shape[3]),
                                           mode='nearest').squeeze(0, 1)

        return local_edit_foreground, shifted_mask_tensor, local_edit_background

    @torch.no_grad()
    def prepare_various_mask_new(self, shifted_mask,ori_mask,draw_mask, sup_res_w, sup_res_h, init_code,gs_scale=None,use_gs=True):
        if use_gs:
            shifted_mask_gaussian_tensor = self.prepare_tensor_mask(generate_gaussian_mask(shifted_mask, gs_scale), sup_res_w, sup_res_h,
                                                                    binary=False)
            shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
            ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
            flexible_region = self.prepare_tensor_mask(draw_mask, sup_res_w, sup_res_h) * (1-shifted_mask_tensor)

            fg_mask = flexible_region*shifted_mask_gaussian_tensor +(1-flexible_region)*shifted_mask_tensor
            self.temp_view(fg_mask.cpu().numpy(), 'MTSA mask')
            complete_region_tensor = flexible_region * shifted_mask_gaussian_tensor
            # complete_region_tensor = flexible_region
            self.temp_view(complete_region_tensor.cpu().numpy(), 'disturb region')
        else:

            shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
            ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
            flexible_region = self.prepare_tensor_mask(draw_mask, sup_res_w, sup_res_h) * (1-shifted_mask_tensor)

            fg_mask = flexible_region + shifted_mask_tensor
            fg_mask[fg_mask>0] = 1.0
            self.temp_view(fg_mask.cpu().numpy(), 'MTSA mask')
            complete_region_tensor = flexible_region
            self.temp_view(complete_region_tensor.cpu().numpy(), 'disturb region')

        complete_region_tensor = F.interpolate(complete_region_tensor.unsqueeze(0).unsqueeze(0),
                                              (init_code.shape[2], init_code.shape[3]),
                                              mode='nearest').squeeze(0, 1)
        return fg_mask,shifted_mask_tensor,ori_mask_tensor,complete_region_tensor
    @torch.no_grad()
    def prepare_various_mask_gs(self, shifted_mask, ori_mask, sup_res_w, sup_res_h, init_code, cons_area,gs_scale,gs_bond_scale):
        shifted_mask_gaussian = generate_gaussian_mask(shifted_mask, gs_scale)
        self.temp_view(shifted_mask_gaussian, 'mtsa_mask')
        shifted_mask_gaussian_tensor = self.prepare_tensor_mask(shifted_mask_gaussian, sup_res_w, sup_res_h,
                                                                binary=False)

        inv_shifted_mask_gaussian = generate_gaussian_mask(shifted_mask, gs_scale,inverted=True,gs_bond_scale=gs_bond_scale)
        self.temp_view(inv_shifted_mask_gaussian, 'Gaussian_bond')
        inv_shifted_mask_gaussian_tensor = self.prepare_tensor_mask(inv_shifted_mask_gaussian, sup_res_w, sup_res_h,
                                                                binary=False)
        shifted_mask_tensor = self.prepare_tensor_mask(shifted_mask, sup_res_w, sup_res_h)
        ori_mask_tensor = self.prepare_tensor_mask(ori_mask, sup_res_w, sup_res_h)
        cons_area_tensor = self.prepare_tensor_mask(cons_area, sup_res_w, sup_res_h)


        cons_area_tensor = cons_area_tensor - ori_mask_tensor
        # local_edit_background = self.prepare_surrounding_mask(shifted_mask_tensor, cons_area_tensor, rate=0.1)
        # local_edit_background[local_edit_background > 0.0] = 1
        local_edit_background = self.prepare_surrounding_mask_gs(inv_shifted_mask_gaussian_tensor, cons_area_tensor,shifted_mask_tensor)
        #
        # local_edit_background = inv_shifted_mask_gaussian_tensor
        self.temp_view(local_edit_background.cpu().numpy(), 'ddpm mask')


        local_edit_foreground = F.interpolate(shifted_mask_gaussian_tensor.unsqueeze(0).unsqueeze(0),
                                              (shifted_mask.shape[0], shifted_mask.shape[1]),
                                              mode='nearest').squeeze(0, 1)
        local_edit_background = F.interpolate(local_edit_background.unsqueeze(0).unsqueeze(0),
                                              (init_code.shape[2], init_code.shape[3]),
                                              mode='nearest').squeeze(0, 1)

        return local_edit_foreground, local_edit_foreground, local_edit_background,shifted_mask_gaussian

    def prepare_tensor_mask(self, mask, sup_res_w, sup_res_h,binary=True):
        # mask interpolation
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_tensor_shifted = torch.tensor(mask, device=self.device)
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
    def Details_Preserving_regeneration_v2(self, source_image, inverted_latents, edit_prompt, shifted_mask,
                                        ori_mask, draw_mask,
                                        num_steps=100, start_step=30, end_step=10,
                                        guidance_scale=3.5, eta=1,
                                        use_mtsa=True,verbose=False,
                                        local_text_edit=True, local_ddpm=True,use_gs=True, gs_scale=0.4,
                                        return_intermediates=False,context_guidance=1.0):

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

        full_h, full_w = source_image.shape[:2]
        # if not verbose:
        #     print(f'full_h:{full_h};full_w:{full_w}')
        # sup_res_h = int(0.5 * full_h)
        # sup_res_w = int(0.5 * full_w)

        fg_retain_mask,fg_retain_mask_st2,fg_ref_mask,local_var_reg=self.prepare_various_mask_new(shifted_mask,ori_mask,draw_mask,full_h, full_w,
                                                                                init_code_orig,gs_scale=gs_scale,use_gs=use_gs)
        # mask = self.prepare_controller_ref_mask(shifted_mask, False)
        fg_retain_mask_cfg = F.interpolate(local_var_reg.unsqueeze(0).unsqueeze(0),
                                               (init_code_orig.shape[2], init_code_orig.shape[3]),
                                               mode='nearest').squeeze(0, 1)
        self.controller.fg_retain_mask = fg_retain_mask.to(self.device)
        self.controller.fg_retain_mask_st2 = fg_retain_mask_st2.to(self.device)#shift mask tensor
        self.controller.fg_ref_mask = fg_ref_mask.to(self.device)
        self.controller.local_edit_region = fg_retain_mask.to(self.device)

        self.controller.reset()
        # refer_gen_image = edit_ori_image[0] #vis ddim results
        self.controller.log_mask = False #forbid mask expansion

        refer_latents_ori = inverted_latents[::-1]

        gen_images,intermediates = self.forward_sampling_new(
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
            foreground_mask = fg_retain_mask_cfg,
            local_var_reg=local_var_reg,
            use_mtsa=use_mtsa,verbose=verbose,blending=local_text_edit,local_ddpm=local_ddpm,
            return_intermediates=return_intermediates,context_guidance=context_guidance
        )
        edit_gen_image, ref_gen_image = gen_images
        self.controller.reset()
        refer_gen_image = ref_gen_image.permute(1, 2, 0).detach().cpu().numpy()*255
        edit_gen_image = edit_gen_image.permute(1, 2, 0).detach().cpu().numpy()*255
        return edit_gen_image.astype(np.uint8), refer_gen_image.astype(np.uint8),intermediates

    def Details_Preserving_regeneration(self, source_image, inverted_latents, edit_prompt, shifted_mask,
                                        ori_mask,
                                        num_steps=100, start_step=30, end_step=10,
                                        guidance_scale=3.5, eta=1,
                                        use_mtsa=True,verbose=False,
                                        local_text_edit=True, local_ddpm=True,cons_area = None,use_gs=True,gs_scale=0.7,gs_bond_scale=0.7):

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


        full_h, full_w = source_image.shape[:2]
        if not verbose:
            print(f'full_h:{full_h};full_w:{full_w}')
        sup_res_h = int(0.5 * full_h)
        sup_res_w = int(0.5 * full_w)
        if use_gs:
            foreground_mask, object_mask, local_var_reg, shifted_mask_gaussian = self.prepare_various_mask_gs(shifted_mask, ori_mask,
                                                                                    sup_res_w, sup_res_h,
                                                                                    init_code_orig,cons_area,gs_scale=gs_scale,gs_bond_scale=gs_bond_scale)
            mask = self.prepare_controller_ref_mask(shifted_mask_gaussian, True)
        else:
            foreground_mask, object_mask, local_var_reg=self.prepare_various_mask(shifted_mask, ori_mask,
                                                                                    sup_res_w, sup_res_h,
                                                                                    init_code_orig,cons_area)
            mask = self.prepare_controller_ref_mask(shifted_mask, True)
        SDSA_REF_MASK = self.prepare_controller_ref_mask(ori_mask, False)
        SDSA_REF_MASK = F.interpolate(SDSA_REF_MASK.unsqueeze(0).unsqueeze(0),
                                      (init_code_orig.shape[2], init_code_orig.shape[3]),
                                      mode='nearest').squeeze(0, 1)

        self.controller.SDSA_REF_MASK = SDSA_REF_MASK
        self.controller.local_edit_region = foreground_mask #text refine
        foreground_mask = F.interpolate(foreground_mask.unsqueeze(0).unsqueeze(0),
                                        (init_code_orig.shape[2], init_code_orig.shape[3]),
                                                mode='nearest').squeeze(0, 1)
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
            # feature_injection_allowed=feature_injection,
            # feature_injection_timpstep_range=FI_range,
            use_mtsa=use_mtsa,verbose=verbose,blending=local_text_edit,local_ddpm=local_ddpm,
        )
        self.controller.reset()
        refer_gen_image = ref_gen_image.permute(1, 2, 0).detach().cpu().numpy()*255
        edit_gen_image = edit_gen_image.permute(1, 2, 0).detach().cpu().numpy()*255
        return edit_gen_image.astype(np.uint8), refer_gen_image.astype(np.uint8)

    # def ReggioRecurrentEdit(self, source_image, inverted_latents, edit_prompt, expand_mask, x_shift, y_shift,
    #                         resize_scale, rotation_angle, motion_split_steps, num_steps=100, start_step=30,
    #                         end_step=10, guidance_scale=3.5, eta=0,
    #                         roi_expansion=True, mask_threshold=0.1, post_process='hard', max_times=10,
    #                         sim_thr=0.7, lr=0.01, lam=0.1, ):
    #
    #     """
    #     latent vis
    #     # noised_image = self.decode_latents(start_latents).squeeze(0)
    #     # # print(noised_image.shape)
    #     # noised_image = self.numpy_to_pil(noised_image)[0]
    #     # print(noised_image.size)
    #     # print(noised_image.shape)
    #     """
    #
    #     start_latents = inverted_latents[-1]
    #     init_code_orig = deepcopy(start_latents)
    #     t = self.scheduler.timesteps[start_step]
    #     unet_feature_idx = [3]
    #
    #     full_h, full_w = source_image.shape[:2]
    #     print(f'full_h:{full_h};full_w:{full_w}')
    #     sup_res_h = int(0.5 * full_h)
    #     sup_res_w = int(0.5 * full_w)
    #
    #     # mask interpolation
    #     mask_tensor = torch.tensor(expand_mask, device=self.device)
    #     mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    #     transformed_masks_tensor = F.interpolate(mask_tensor, (sup_res_h, sup_res_w), mode="nearest")
    #     x_shift = int(x_shift * 0.5)
    #     y_shift = int(y_shift * 0.5)
    #
    #     # TODO: 1-step optimization
    #     self.controller.log_mask = False
    #     updated_init_code, h_feature, final_traj_mask = self.Recurrent_diffusion_updates_2D(edit_prompt,
    #                                                                                         start_latents, t,
    #                                                                                         transformed_masks_tensor,
    #                                                                                         x_shift, y_shift,
    #                                                                                         resize_scale,
    #                                                                                         rotation_angle,
    #                                                                                         motion_split_steps,
    #                                                                                         unet_feature_idx=unet_feature_idx,
    #                                                                                         lam=lam, lr=lr,
    #                                                                                         sup_res_w=sup_res_w,
    #                                                                                         sup_res_h=sup_res_h,
    #                                                                                         max_times=max_times,
    #                                                                                         sim_thr=sim_thr)
    #
    #     with torch.no_grad():
    #         noised_optimized_image = self.decode_latents(updated_init_code).squeeze(0)
    #         noised_optimized_image = self.numpy_to_pil(noised_optimized_image)[0]
    #         # updated_init_code == init_code == start code
    #         # print(f'init_code == start_code : {torch.all(start_latents == updated_init_code)}')
    #
    #         mask = self.prepare_controller_ref_mask(final_traj_mask)
    #         self.controller.expansion_mask_store = {}
    #         self.controller.log_mask = True
    #
    #         ori_gen_image, edit_gen_image = self.forward_sampling(
    #             prompt=edit_prompt,
    #             h_feature=h_feature,
    #             end_step=end_step,
    #             batch_size=2,
    #             latents=torch.cat([updated_init_code, updated_init_code], dim=0),  # same
    #             # latents=torch.cat([updated_init_code, updated_init_code], dim=0),
    #             guidance_scale=guidance_scale,
    #             num_inference_steps=num_steps,
    #             num_actual_inference_steps=num_steps - start_step,
    #         )
    #         ori_gen_image = self.numpy_to_pil(ori_gen_image.permute(1, 2, 0).detach().cpu().numpy())[0]
    #         edit_gen_image = self.numpy_to_pil(edit_gen_image.permute(1, 2, 0).detach().cpu().numpy())[0]
    #
    #         # 从意义上来说，如果ddpm forward 的作用是优化细节，其实mask不如直接用final mask
    #         # 但是对于需要利用后续先验进一步修复的case，或者循环每一步forward 进行优化的case 都可以重新获得mask
    #         # 所以先保留这部分代码
    #         # get ddpm forward expansion target mask
    #         ddpm_for_expansion_masks = self.controller.expansion_mask_store  # expansion mask & average of up mid down resized corresponded self attention maps
    #         self.controller.expansion_mask_store = {}  # reset for next image
    #         candidate_mask = self.fetch_expansion_mask_from_store([ddpm_for_expansion_masks], mask,
    #                                                               roi_expansion, post_process, mask_threshold)
    #
    #     return ori_gen_image, edit_gen_image, noised_optimized_image, candidate_mask

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

    # def Recurrent_diffusion_updates_2D(self,
    #                                    prompt,
    #                                    init_code,
    #                                    t,
    #                                    masks_tensor,
    #                                    x_shift, y_shift, resize_scale, rotation_angle,
    #                                    motion_split_steps,
    #                                    unet_feature_idx=[3],
    #                                    sup_res_h=256,
    #                                    sup_res_w=256,
    #                                    lam=0.1, lr=0.01, max_times=10, sim_thr=0.92):
    #     # iteratively optimize in one step and refine serveral times for each step to assure succefully edit each step
    #     print(f'mask_tensor.shape:{masks_tensor.shape}')
    #     # Encode prompt
    #     text_emb = self.get_text_embeddings(prompt).detach()
    #     with torch.no_grad():
    #         unet_output, F0, h_feature = self.forward_unet_features(init_code, t,
    #                                                                 encoder_hidden_states=text_emb,
    #                                                                 layer_idx=unet_feature_idx,
    #                                                                 interp_res_h=sup_res_h,
    #                                                                 interp_res_w=sup_res_w)
    #         x_prev_0, _ = self.step(unet_output, t, init_code)
    #
    #     # prepare for point tracking and background regularization
    #     trajectory_masks_sup = copy.deepcopy(masks_tensor).squeeze(0, 1)
    #
    #     height, width = trajectory_masks_sup.shape[:2]
    #     rotation_angle_step = rotation_angle / motion_split_steps
    #     resize_scale_step = resize_scale ** (1 / motion_split_steps)
    #     dx = x_shift / motion_split_steps
    #     dy = y_shift / motion_split_steps
    #
    #     # h_features = []
    #
    #     trajectory_current_mask = trajectory_masks_sup
    #     current_feature_like = F0
    #     # return init_code, h_feature, trajectory_masks_sup
    #
    #     step_idx = 0
    #     try_times = 1
    #     while step_idx <= motion_split_steps:
    #         with torch.autocast(device_type='cuda', dtype=torch.float16):
    #
    #             # copy_h = copy.deepcopy(h_feature)
    #             # h_features.append(copy_h)
    #
    #             if (step_idx == 0) or (try_times > max_times) or self.reach_similarity_condition(F1_region,
    #                                                                                              F0_region,
    #                                                                                              sim_thr):
    #                 # if init or last step finish -> move on to next step
    #                 step_idx += 1
    #                 try_times = 1
    #                 if step_idx > motion_split_steps:
    #                     break
    #                 # copy_h = copy.deepcopy(h_feature)
    #                 # h_features.append(copy_h)
    #                 # get mask center
    #                 mask_center_x, mask_center_y = self.get_mask_center(
    #                     trajectory_current_mask)  # relocate mask center every step,
    #                 # allowing tracking real mask center update
    #
    #                 # get transformation matrix
    #                 transformation_matrix = cv2.getRotationMatrix2D((mask_center_x, mask_center_y),
    #                                                                 -rotation_angle_step,
    #                                                                 resize_scale_step)
    #                 transformation_matrix[0, 2] += dx
    #                 transformation_matrix[1, 2] += dy
    #                 # cv2 rotation matrix -> Tensor adaptive transformation theta
    #                 theta = torch.tensor(param2theta(transformation_matrix, width, height), dtype=F0.dtype,
    #                                      device=self.device)
    #                 trajectory_current_mask = trajectory_current_mask.to(F0.dtype)
    #                 trajectory_next_mask = \
    #                 wrapAffine_tensor(trajectory_current_mask, theta, (width, height), mode='nearest')[0]
    #                 foreground_latent_region = trajectory_next_mask + trajectory_current_mask
    #                 foreground_latent_region[foreground_latent_region > 0.0] = 1
    #                 latent_trajectory_masks = F.interpolate(foreground_latent_region.unsqueeze(0).unsqueeze(0),
    #                                                         (init_code.shape[2], init_code.shape[3]),
    #                                                         mode='nearest').squeeze(0, 1)
    #                 # init_code = self.edit_init_code(init_code, theta, trajectory_current_mask, trajectory_next_mask)
    #                 # h_feature = self.edit_init_code(h_feature, theta, trajectory_current_mask, trajectory_next_mask)
    #                 h_feature = h_feature.detach().requires_grad_(True)
    #                 # return init_code, h_feature, trajectory_next_mask
    #                 # prepare optimizable init_code and optimizer
    #                 # h_feature.requires_grad_(True)
    #                 optimizer = torch.optim.Adam([h_feature], lr=lr)
    #                 # prepare amp scaler for mixed-precision training
    #                 scaler = torch.cuda.amp.GradScaler()
    #
    #                 unet_output, F1, h_feature = self.forward_unet_features(init_code, t,
    #                                                                         encoder_hidden_states=text_emb,
    #                                                                         h_feature=h_feature,
    #                                                                         layer_idx=unet_feature_idx,
    #                                                                         interp_res_h=sup_res_h,
    #                                                                         interp_res_w=sup_res_w)
    #                 x_prev_updated, _ = self.step(unet_output, t, init_code)
    #
    #                 trajectory_next_feature_like = wrapAffine_tensor(current_feature_like, theta,
    #                                                                  (width, height),
    #                                                                  mode='bilinear').unsqueeze(0)
    #
    #                 F0_region = trajectory_next_feature_like[:, :,
    #                             trajectory_next_mask.bool()].detach()  # F0_region update
    #                 F1_region = F1[:, :, trajectory_next_mask.bool()]
    #                 trajectory_current_mask = trajectory_next_mask
    #                 current_feature_like = trajectory_next_feature_like
    #             else:
    #                 unet_output, F1, h_feature = self.forward_unet_features(init_code, t,
    #                                                                         encoder_hidden_states=text_emb,
    #                                                                         h_feature=h_feature,
    #                                                                         layer_idx=unet_feature_idx,
    #                                                                         interp_res_h=sup_res_h,
    #                                                                         interp_res_w=sup_res_w)
    #                 x_prev_updated, _ = self.step(unet_output, t, init_code)
    #                 # recurrent refine on this step
    #                 try_times += 1
    #                 F1_region = F1[:, :, trajectory_current_mask.bool()]  # reattain F1_region from new F1
    #
    #             # update every step every try
    #             assert F0_region.shape == F1_region.shape, "The shapes of F0_region and F1_region do not match for L1 loss calculation."
    #             edit_loss = F.l1_loss(F0_region, F1_region)
    #             # edit_loss = self.weighted_loss(F0_region,F1_region,topk=1000)
    #             BG_loss = lam * ((x_prev_updated - x_prev_0) * (1.0 - latent_trajectory_masks)).abs().sum()
    #             loss = edit_loss + BG_loss
    #             print(f'handling step:{step_idx}/{motion_split_steps} , trying times:{try_times}')
    #             print(f'Edit_loss: {edit_loss.item()},BG_loss: {BG_loss.item()} Total_loss: {loss.item()}')
    #             # print(f'BG_loss: {BG_loss.item()}')
    #
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         optimizer.zero_grad()
    #
    #     # return init_code, h_feature, h_features
    #     # return init_code, h_feature, h_features, trajectory_current_mask
    #     return init_code, h_feature, trajectory_current_mask







