import torch
from GeoDiffuser.utils.generic import log_args
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from GeoDiffuser.utils.attention_processors import *
from GeoDiffuser.utils.warp_utils import RasterizePointsXYsBlending

# USE_PEFT_BACKEND = False
# UNCOND_TEXT="pixelated, unclear, blurry, grainy"

# UNCOND_TEXT=""
# MY_TOKEN = ''

# # DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
# DIFFUSION_MODEL = "CompVis/stable-diffusion-v1-4"
# # DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-1-base"
# # DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-base"
# # DIFFUSION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


# LOW_RESOURCE = False 
# NUM_DDIM_STEPS = 50
# GUIDANCE_SCALE = 4.0
# MAX_NUM_WORDS = 77
# IMAGE_SIZE = 512
# SKIP_OPTIM_STEPS = 0
# SEED = 1234
# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# MODE="bilinear"
# SPLATTER = RasterizePointsXYsBlending()

# LDM_STABLE = None
# SCHEDULER = None
# TOKENIZER = None
# UNET_NAME = None
# PROGRESS_BAR = None


@torch.autocast("cuda", dtype=torch.half)
def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, transform_coords=None, use_cfg=True, return_noise = False):
 
    if use_cfg:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred_out = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    else:
        latents = latents#.to(memory_format=torch.channels_last)
        model.unet = model.unet#.to(memory_format=torch.channels_last)
        noise_pred_out = model.unet(latents, t, encoder_hidden_states=context)["sample"]

    latents_out = model.scheduler.step(noise_pred_out, t, latents, eta=0.0)["prev_sample"]
    latents_out = controller.step_callback(latents_out, transform_coords)
    SPLATTER.clear_cache()

    if return_noise:
        return latents_out, noise_pred_out

    return latents_out

@torch.autocast("cuda", dtype=torch.half)
def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

# @torch.no_grad()
def image2latent(image, model, mask = None, device = "cuda:0"):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            
            if mask is not None:
                
                if type(mask) is np.ndarray:
                    mask_in = torch.from_numpy(mask).float()
                    image = image * (mask_in[..., None] < 0.5)
                    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                else:
                    mask_in = mask
                    image = image.permute(2, 0, 1).unsqueeze(0).to(device).tile(mask.shape[0], 1, 1, 1)
                    image = image * (mask_in < 0.5).to(device)
                    
            else:
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                 
            
            latents = model.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents

def load_model(diffusion_model, unet_path = "", device = "cuda:0"):

    global UNET_NAME

    if (unet_path != "") and (unet_path != None):
        diffusion_model = unet_path
    data_type = torch.half
    # ldm_stable = StableDiffusionXLPipeline.from_pretrained(DIFFUSION_MODEL, torch_dtype=torch.half).to(device)
    ldm_stable = StableDiffusionPipeline.from_pretrained(diffusion_model, torch_dtype=torch.half).to(device)
    # print(ldm_stable.scheduler.config)
    # scheduler = DDIMScheduler.from_pretrained(DIFFUSION_MODEL, subfolder='scheduler', beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    # variant="fp16"
    ldm_stable.scheduler = scheduler
    
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # ldm_stable = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL, scheduler=scheduler).to(device)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer
    
    ldm_stable.unet.set_attn_processor(VanillaAttentionProcessor())
    
    if diffusion_model.split("-")[-2] == "v1":
        print("[INFO]: Using Updated vae")
        ldm_stable.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.half).to(device).eval()

    # ldm_stable.vae.set_attn_processor(diffusers.models.attention_processor.AttnProcessor())

    UNET_NAME = diffusion_model
    if unet_path != "":
        print("[INFO]: Loading UNET model from path: ", unet_path)
        ldm_stable.unet = UNet2DConditionModel.from_pretrained(
        unet_path, subfolder="unet", torch_dtype=data_type).to(device)
        UNET_NAME = unet_path
    # ldm_stable.unet.eval()    
    # ldm_stable.unet = torch.compile(ldm_stable.unet, mode = "reduce-overhead")
    ldm_stable.unet.eval()

    



    # print("compile done!")
    # print(ldm_stable.unet.config)
    # exit()
    
    return ldm_stable, tokenizer, scheduler