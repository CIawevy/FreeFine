# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

from PIL import Image
import os
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image

from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

# model_path: path of the model
# image: input image, have not been pre-processed
# save_lora_path: the path to save the lora
# prompt: the user input prompt
# lora_step: number of lora training step
# lora_lr: learning rate of lora training
# lora_rank: the rank of lora
# save_interval: the frequency of saving lora checkpoints
def train_lora(image,
    prompt,
    model_path,
    vae_path,
    save_lora_path,
    lora_step,
    lora_lr,
    lora_batch_size,
    lora_rank,
    progress,
    save_interval=-1):
    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16'
    )
    set_seed(0)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    # initialize the model
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(
        model_path, subfolder="text_encoder", revision=None
    )
    if vae_path == "default":
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", revision=None
        )
    else:
        vae = AutoencoderKL.from_pretrained(vae_path)
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", revision=None
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    vae=vae,
                    unet=unet,
                    text_encoder=text_encoder,
                    scheduler=noise_scheduler,
                    torch_dtype=torch.float16)

    # set device and dtype
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    # Set correct lora layers
    unet_lora_parameters = []
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                rank=lora_rank
            )
        )
        attn_module.to_k.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                rank=lora_rank
            )
        )
        attn_module.to_v.set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                rank=lora_rank
            )
        )
        attn_module.to_out[0].set_lora_layer(
            LoRALinearLayer(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                rank=lora_rank,
            )
        )

        # Accumulate the LoRA params to optimize.
        unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
        unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            attn_module.add_k_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_k_proj.in_features,
                    out_features=attn_module.add_k_proj.out_features,
                    rank=args.rank,
                )
            )
            attn_module.add_v_proj.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.add_v_proj.in_features,
                    out_features=attn_module.add_v_proj.out_features,
                    rank=args.rank,
                )
            )
            unet_lora_parameters.extend(attn_module.add_k_proj.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.add_v_proj.lora_layer.parameters())


    # Optimizer creation
    params_to_optimize = (unet_lora_parameters)
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_step,
        num_cycles=1,
        power=1.0,
    )

    # prepare accelerator
    # unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    # optimizer = accelerator.prepare_optimizer(optimizer)
    # lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    if torch.cuda.device_count() > 1:
        unet,optimizer,lr_scheduler = accelerator.prepare(unet,optimizer,lr_scheduler)
    else:
        unet = unet.to(device)
   

    # initialize text embeddings
    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
        text_embedding = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=False
        )
        text_embedding = text_embedding.repeat(lora_batch_size, 1, 1)

    # initialize image transforms
    image_transforms_pil = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(512),
        ]
    )
    image_transforms_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    for step in progress.tqdm(range(lora_step), desc="training LoRA"):
        unet.train()
        image_batch = []
        image_pil_batch = []
        for _ in range(lora_batch_size):
            # first store pil image
            image_transformed = image_transforms_pil(Image.fromarray(image))
            image_pil_batch.append(image_transformed)            

            # then store tensor image
            image_transformed = image_transforms_tensor(image_transformed).to(device, dtype=torch.float16)
            image_transformed = image_transformed.unsqueeze(dim=0)
            image_batch.append(image_transformed)

        # repeat the image_transformed to enable multi-batch training
        image_batch = torch.cat(image_batch, dim=0)

        latents_dist = vae.encode(image_batch).latent_dist
        model_input = latents_dist.sample() * vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        # Predict the noise residual
        model_pred = unet(noisy_model_input,
                          timesteps,
                          text_embedding).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if save_interval > 0 and (step + 1) % save_interval == 0:
            save_lora_path_intermediate = os.path.join(save_lora_path, str(step+1))
            if not os.path.isdir(save_lora_path_intermediate):
                os.mkdir(save_lora_path_intermediate)
            # unet = unet.to(torch.float32)
            # unwrap_model is used to remove all special modules added when doing distributed training
            # so here, there is no need to call unwrap_model
            # unet_lora_layers = accelerator.unwrap_model(unet_lora_layers)
            unet_lora_layers = unet_lora_state_dict(unet)
            LoraLoaderMixin.save_lora_weights(
                save_directory=save_lora_path_intermediate,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=None,
            )
            # unet = unet.to(torch.float16)

    # save the trained lora
    # unet = unet.to(torch.float32)
    # unwrap_model is used to remove all special modules added when doing distributed training
    # so here, there is no need to call unwrap_model
    # unet_lora_layers = accelerator.unwrap_model(unet_lora_layers)
    unet_lora_layers = unet_lora_state_dict(unet)
    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_path,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
    )

    return
