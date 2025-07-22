import torch
from PIL import Image
from typing import Optional, Union, Tuple, List, Callable, Dict
import numpy as np
from diffusers import DDIMScheduler, DDIMInverseScheduler
from GeoDiffuser.utils.attention_processors import *


class NullInversion:
    def null_text_optimization_transforms_loss(self, t, latent_cur, cond_embeddings, uncond_embeddings, latent_prev, noise_pred_i, t_coords=None):
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        
        
        
        if t_coords is None:
            latent_cur_t, state = set_seed_apply_transform(latent_cur, fill = 0.0, affine = 1)
            latent_cur_t_1, state = set_seed_apply_transform(latent_cur, fill = 1.0, affine = 1)
            latent_prev_t, _ = set_seed_apply_transform(latent_prev, fill = 0.0, affine = 1)
        else:
        #             print(t_coords.shape, latent_cur.shape)
            s = latent_cur.shape[-1]
            t_coords_new = t_coords.permute(0, -1, 1, 2)
            t_coords_new = T.Resize(size=(s, s))(t_coords_new).permute(0, 2, 3, 1).type_as(latent_cur)
            latent_cur_t = warp_grid_edit(latent_cur, t_coords_new, padding_mode='zeros', align_corners=True, mode="bilinear")
            latent_cur_t_1 = warp_grid_edit(latent_cur, t_coords_new, padding_mode='reflection', align_corners=True, mode="bilinear")
            latent_prev_t = warp_grid_edit(latent_prev, t_coords_new, padding_mode='reflection', align_corners=True, mode="bilinear")
            #             assert False, "break"
        #             latent_cur_t = warp_grid_edit
        #         Get mask for content 
        mask_content = (torch.abs(latent_cur_t_1 - latent_cur_t) > 0.1) * 1.0
        latent_cur_t = latent_cur_t * mask_content + (1.0 - mask_content) * noise_pred_i * (1 - alpha_prod_t) ** 0.5
        with torch.no_grad():
            noise_pred_cond = self.get_noise_pred_single(latent_cur_t, t, cond_embeddings)

        noise_pred_uncond = self.get_noise_pred_single(latent_cur_t, t, uncond_embeddings)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents_prev_rec_t = self.prev_step(noise_pred, t, latent_cur_t)
        latent_prev_t = latent_prev_t * mask_content + (1.0 - mask_content) * noise_pred * (1 - alpha_prod_t) ** 0.5
        
        loss = nnf.mse_loss(latents_prev_rec_t, latent_prev_t)
        # print(controller.)
        # aggregate_attention(self.controller)
        return loss 
    
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        # torch.set_rng_state(STATE)
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        # torch.set_rng_state(STATE)
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        # torch.set_rng_state(STATE)
        latents = latents.detach() / self.model.vae.config.scaling_factor
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * self.model.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [self.uncond_text], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, latent_2 = None):

        self.model.unet.set_attn_processor(VanillaAttentionProcessor())
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if latent_2 is not None:
            latent = torch.cat([latent, latent_2], 0)

        all_latent = [latent]
        all_noise = [latent]
        latent = latent.clone().detach()

        # Change to include sampling v objective for SD 2.1
        inverse_scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

        # inverse_scheduler = DDIMInverseScheduler.from_pretrained(DIFFUSION_MODEL, subfolder='scheduler', beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=0)#, timestep_spacing = "leading")


        # timesteps, num_inference_steps = self.model.retrieve_timesteps(inverse_scheduler, self.num_ddim_steps, self.device, None)

        inverse_scheduler.set_timesteps(self.num_ddim_steps, device=self.device)
        timesteps = inverse_scheduler.timesteps
        # timesteps = self.model.scheduler.timesteps
        latents = latent
        # print(self.context.shape, "context shape")

        context_in = self.context

        if latent_2 is not None:
            uncond_e, cond_e = context_in.chunk(2)
            context_in = torch.cat([uncond_e, uncond_e, cond_e, cond_e], 0)
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)
        with self.model.progress_bar(total=self.num_ddim_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if self.progress_bar is not None:
                    self.progress_bar(i / self.num_ddim_steps, desc="Performing DDIM Inversion")

                

                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = inverse_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=context_in,
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = inverse_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # print(latents.mean())
                all_latent.append(latents.detach())
                all_noise.append(noise_pred_cond.detach())
                progress_bar.update()

        return all_latent, all_noise

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, image_2 = None):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        latent_2 = None
        if image_2 is not None:
            latent_2 = self.image2latent(image_2)

        ddim_latents, ddim_noise = self.ddim_loop(latent, latent_2)
        return image_rec, ddim_latents, ddim_noise

    def null_optimization(self, latents, num_inner_steps, epsilon, t_coords=None):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_ddim_steps)

        with torch.enable_grad():
            for i in range(self.num_ddim_steps):
                uncond_embeddings = uncond_embeddings.clone().detach()
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]
                t = self.model.scheduler.timesteps[i]
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                for j in range(num_inner_steps):

                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                    
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    
    #                 if 1:
                    # print("opt")
                    if t < -1:
                    # if t < int(0.5 * (1000)):
                        print(j)
                        loss = (nnf.mse_loss(latents_prev_rec, latent_prev) + self.null_text_optimization_transforms_loss(t, latent_cur, cond_embeddings, uncond_embeddings, latent_prev, noise_pred_cond, t_coords=t_coords)) / 2.0
                    else:
                        loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                    

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    bar.update()
                    if loss_item < epsilon + i * 2e-5:
                        break
                for j in range(j + 1, num_inner_steps):
                    bar.update()
                uncond_embeddings_list.append(uncond_embeddings[:1].detach())
                with torch.no_grad():
                    context = torch.cat([uncond_embeddings, cond_embeddings])
                    latent_cur = self.get_noise_pred(latent_cur, t, False, context)
            bar.close()
        return uncond_embeddings_list

    @torch.autocast("cuda", dtype=torch.half)
    def invert(self, image_gt, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False, t_coords = None, perform_inversion=True, image_2 = None):
        self.init_prompt(prompt)

        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents, ddim_noise = self.ddim_inversion(image_gt, image_2)

        # If perform null inversion
        # Null text optimization is not required for GeoDiffuser :)
        if perform_inversion:
            if verbose:
                print("Null-text optimization...")
            uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon, t_coords = t_coords)
        else:
            uncond_embeddings = None
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings, ddim_latents, ddim_noise
        
    
    def __init__(self, model, num_ddim_steps = 50, uncond_text = "", device="cuda:0", progress_bar = None, guidance_scale = 3.0):
        self.guidance_scale = guidance_scale
        self.progress_bar = progress_bar
        self.device = device
        self.num_ddim_steps = num_ddim_steps
        self.uncond_text = uncond_text
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None
        self.controller = None
        