import torch
from GeoDiffuser.utils.ptp_utils import view_images
from GeoDiffuser.utils.generic import log_args

import random
import pytorch_lightning as pl
import xformers.ops as xops

import logging
from typing import Optional, Union, Tuple, List, Callable, Dict

from diffusers.models.attention_processor import USE_PEFT_BACKEND
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from GeoDiffuser.utils.warp_utils import RasterizePointsXYsBlending
from GeoDiffuser.utils import warp_utils, vis_utils
from GeoDiffuser.utils.inversion import NullInversion
from GeoDiffuser.utils.generic_torch import *
from GeoDiffuser.utils.diffusion import image2latent, latent2image, diffusion_step, load_model
# from GeoDiffuser.utils.attention_ import *
from GeoDiffuser.utils.attention_processors import *
from GeoDiffuser.utils.generic import *
from tqdm import tqdm
from GeoDiffuser.utils.optimization import _update_latent, adaptive_optimization_step_editing, adaptive_optimization_step_remover, adaptive_optimization_step_stitching
from GeoDiffuser.utils.image_processing import *

# USE_PEFT_BACKEND = False
# UNCOND_TEXT="pixelated, unclear, blurry, grainy"

UNCOND_TEXT=""
MY_TOKEN = ''

# DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
DIFFUSION_MODEL = "CompVis/stable-diffusion-v1-4"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-1-base"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-base"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"



LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 4.0
MAX_NUM_WORDS = 77
IMAGE_SIZE = 512
SKIP_OPTIM_STEPS = 0
SEED = 1234
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MODE="bilinear"
SPLATTER = RasterizePointsXYsBlending()

LDM_STABLE = None
SCHEDULER = None
TOKENIZER = None
UNET_NAME = None
PROGRESS_BAR = None
# DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
DIFFUSION_MODEL = "CompVis/stable-diffusion-v1-4"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-1-base"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-2-base"
# DIFFUSION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 20,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    transform_coordinates=None,
    mask_obj = None,
    optimize_steps = 0.2,
    latent_replace = 0.2,
    lr = 0.0,
    optimize_embeddings=False,
    optimize_latents = False,
    ddim_latents = None,
    ddim_noise = None,
    edit_type = "geometry_editor",
    fast_start_steps = 0.0,
    num_first_optim_steps = 5,
    use_adaptive_optimization = True,
    adain_latents_steps=0.95,
    use_optimizer = False,
    removal_loss_value_in = -1.5,
):

    global_loss_log_dict = {}

    global GUIDANCE_SCALE, SKIP_OPTIM_STEPS, PROGRESS_BAR

    skip_optim_steps = SKIP_OPTIM_STEPS
    print("Using guidance scale: ", GUIDANCE_SCALE, " skip optim steps: ", SKIP_OPTIM_STEPS)
    batch_size = len(prompt)
    register_attention_control_diffusers(model, controller, transform_coordinates)
    model.uet = torch.compile(model.unet, mode = "reduce-overhead")
    
    height = width = IMAGE_SIZE
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [UNCOND_TEXT] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent[:1], model, height, width, generator, batch_size)

    if edit_type == "geometry_stitch":
        # Add latent for stitched object
        latents[1:2] = ddim_latents[-1][1:2]
    

    # print(latent.shape, latent.shape, ddim_latents[-1].shape)
    model.scheduler.set_timesteps(num_inference_steps)

    for param in model.unet.parameters():
        param.requires_grad = False

    context_save = None
    optimizer = None
    # use_optimizer = True
    first_optim_complete = False
    l_eff = lr * skip_optim_steps * (50 / (NUM_DDIM_STEPS + 1e-8))
    scaler = None
    # scaler = torch.cuda.amp.GradScaler(growth_interval=1)
    pbar = tqdm(model.scheduler.timesteps[-start_time:])



    if transform_coordinates is not None:
        t_coords_m = reshape_transform_coords(transform_coordinates, in_mat_shape=controller.image_mask.shape).tile(controller.image_mask.shape[0], 1, 1, 1).type_as(text_embeddings)
        controller.mask_new_warped = binarize_tensor(warp_grid_edit(controller.image_mask[:, None], t_coords_m, padding_mode='zeros', align_corners=True, mode=MODE)).type_as(text_embeddings)
    
    for i, t in enumerate(pbar):

        # SPLATTER.clear_cache()
        SPLATTER.radius = max(1, int(SPLATTER.radius * 0.95))
        SPLATTER.tau = max(SPLATTER.tau * 0.8, 1e-1)
        SPLATTER.points_per_pixel = max(int(SPLATTER.points_per_pixel * 0.9), 4)
        # print(i)

        alpha_t = model.scheduler.timesteps[i]
        # print(alpha_t, i, t)
        alpha_t = model.scheduler.alphas_cumprod[alpha_t]
        beta_t = 1 - alpha_t
        # print(alpha_t, i, t)

        if uncond_embeddings_ is None:
        #             0, 1 are uncond embeddings and 2, 3 are text embeddings
        # edit only embedding 1
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
            
        
        if type(controller).__name__.startswith("AttentionGeometry"):
            # del controller.loss
            # controller.loss = 0.0

            clear_controller_loss(controller)
            # print(i, (fast_start_steps * len(model.scheduler.timesteps[-start_time:])), (len(model.scheduler.timesteps[-start_time:])), (fast_start_steps)) 

            
            if ((i < optimize_steps * len(model.scheduler.timesteps[-start_time:])) and (i % skip_optim_steps ==0) and (i >= (fast_start_steps * len(model.scheduler.timesteps[-start_time:])))):
                
                

                
                if not first_optim_complete and (fast_start_steps > 0.0):
                    num_optim_steps = num_first_optim_steps
                    first_optim_complete = True
                else:
                    num_optim_steps = 1


                best_loss = 1e+8
                best_latents = None
                best_context = None

                # if edit_type == "geometry_stitch":
                #     l_eff = l_eff * 0.98
                # # elif edit_type == "geometry_remover":
                # #     l_eff = 50 * lr * beta_t * skip_optim_steps * (50 / (NUM_DDIM_STEPS + 1e-8))
                # else:

                # if edit_type == "geometry_editor":
                #     l_eff = 50 * lr * beta_t * skip_optim_steps * (50 / (NUM_DDIM_STEPS + 1e-8))

                # else:
                l_eff = lr * (50 - i) * skip_optim_steps * (50 / (NUM_DDIM_STEPS + 1e-8))


                if edit_type == "geometry_stitch":
                    set_attn_processor_for_edit(model, coords_base=(0, 2), coords_edit=(2, 3), use_cfg=False)
                else:
                    set_attn_processor_for_edit(model, coords_base=(0, 1), coords_edit=(1, 2), use_cfg=False)
                    

                transform_coordinates = transform_coordinates.detach()
                # model.unet.zero_grad()
                latents_in = latents.detach().float().requires_grad_(True)
                orig_norm = norm_tensor(latents_in[-1:].detach()).item()

                if context_save is None:
                    context_in = context.detach().float().requires_grad_(True)
                else:
                    context_in = context_save.detach().float().requires_grad_(True)


                if optimizer is None:
                    optimizer = torch.optim.SGD([latents_in, context_in], lr = l_eff, momentum=0.9)
                else:
                    optimizer.param_groups[0]["params"] = [latents_in, context_in]
                optimizer.zero_grad(set_to_none=True)

                if not use_optimizer:
                    optimizer = None


                for opt_iter in range(num_optim_steps):

                    with torch.enable_grad():
                        for param in model.unet.parameters():
                            param.requires_grad = False


                        # model.unet = model.unet.eval()
                        # with torch.autocast(device_type="cuda", dtype=torch.float16):
                        # if 1:
                            # with torch.profiler.profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
                                
                            # with torch.autograd.profiler.profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_stack=True) as prof:
                        if edit_type == "geometry_stitch":
                            latent_opt, noise_opt = diffusion_step(model, controller, latents_in, context_in[3:], t, guidance_scale, low_resource=False, transform_coords=transform_coordinates, use_cfg=False, return_noise=True)
                        else:
                            latent_opt, noise_opt = diffusion_step(model, controller, latents_in, context_in[2:], t, guidance_scale, low_resource=False, transform_coords=transform_coordinates, use_cfg=False, return_noise=True)

                        # model.unet.zero_grad()
                        # sds_grad = torch.sum((noise_opt[-1:] - noise_opt[:1]).detach() * latents_in[-1:]) + 0 * torch.sum(context_in)
                        controller.loss = controller.loss#.detach() * 0.0 + 1.0 * sds_grad
                        if controller.loss.detach().item() < best_loss:
                            best_latents = latents_in
                            best_context = context_in
                            best_loss = controller.loss.detach().item()


                        # print(opt_iter, " ", best_loss)
                        pbar.set_description_str(desc = "Loss: %f" % best_loss)

                        if PROGRESS_BAR is not None:
                            PROGRESS_BAR(i / NUM_DDIM_STEPS, desc="Editing Optimization Loss: %f" % best_loss)

                        if edit_type == "geometry_stitch":
                            latents_in, context_new = _update_latent(latents_in, controller.loss, l_eff, controller.mask_new_warped[:1], context_in, scaler = scaler, optimizer = optimizer)
                        else:
                            latents_in, context_new = _update_latent(latents_in, controller.loss, l_eff, controller.mask_new_warped[:1], context_in, scaler = scaler, optimizer = optimizer)

                        
                        if num_optim_steps == 1:
                            best_latents = latents_in
                            best_context = context_new
                        
                        context_in = context_new.detach().float().requires_grad_(True)
                        latents_in = latents_in.detach().float().requires_grad_(True)


                        out_loss_log_dict = convert_loss_log_to_numpy(controller.loss_log_dict)
                        
                        if use_adaptive_optimization:
                            print("[INFO]: Using Adaptive Optimization")
                            if edit_type == "geometry_editor":
                                adaptive_optimization_step_editing(controller, i, skip_optim_steps, out_loss_log_dict, num_ddim_steps = NUM_DDIM_STEPS, removal_loss_value_in = removal_loss_value_in)
                            elif edit_type == "geometry_stitch":
                                adaptive_optimization_step_stitching(controller, i, skip_optim_steps, out_loss_log_dict, num_ddim_steps = NUM_DDIM_STEPS)
                            elif edit_type == "geometry_remover":
                                adaptive_optimization_step_remover(controller, i, skip_optim_steps, out_loss_log_dict, num_ddim_steps = NUM_DDIM_STEPS, removal_loss_value_in = removal_loss_value_in)
                        else:
                            print("[INFO]: Not Using Adaptive Optimization Results may not be optimal")


                                # pass 
                        # print(out_loss_log_dict)
                        global_loss_log_dict[i] = out_loss_log_dict
                        print(out_loss_log_dict["self"])
                        print(out_loss_log_dict["cross"])

                        # del controller.loss
                        # controller.loss = 0.0
                        clear_controller_loss(controller)
                        controller.cur_step -= 1

                        # latents_in = _update_latent(latents_in, controller.loss, lr * (50 - i))
                    
                    # if optimize_latents and (i < 0.6 * optimize_steps * len(model.scheduler.timesteps[-start_time:])) :
                if optimize_latents:
                    # print("Updating latents")
                    latents = best_latents.detach()
                    # Preserving norm
                    latents[-1:] = latents[-1:] * orig_norm / (norm_tensor(latents[-1:].detach()).item())
                    
                
                if best_context is not None and optimize_embeddings:
                    # print("Updating context embeddings")
                    context = best_context.detach()
                    context_save = context
                    # print(context.shape)

                pbar.set_description_str(desc = "Loss: %f" % best_loss)
                
                if PROGRESS_BAR is not None:
                    PROGRESS_BAR(i / NUM_DDIM_STEPS, desc="Editing Optimization Loss: %f" % best_loss)

                pbar.refresh()
                # pbar.set_postfix({'Loss': controller.loss.detach().item()})
            
                del latents_in
                del context_new
                del best_context
                del best_latents

                    
                with torch.no_grad():
                    if edit_type == "geometry_stitch":
                        set_attn_processor_for_edit(model, coords_base=(3, 5), coords_edit=(5, 6), use_cfg=True)
                    else:
                        set_attn_processor_for_edit(model, coords_base=(2, 3), coords_edit=(3, 4), use_cfg=True)

                    # with torch.no_grad():
                    #     latents[-1:] = adain(latents[-1:], latents[:1], dim = 1).detach()
                    
                    # with torch.no_grad():
                    #     latents[-1:] = adain_latents(latents[-1:], latents[:1]).detach()

                    latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, transform_coords=transform_coordinates)
                    
                
            elif (i < fast_start_steps * len(model.scheduler.timesteps[-start_time:])):
                pass
            else:

                if context_save is not None:
                    context = context_save
                # set_attn_processor_for_edit(model, coords_base=(2, 3), coords_edit=(3, 4), use_cfg=True, perform_edit=False)
                
                with torch.no_grad():
                    if edit_type == "geometry_stitch":
                        set_attn_processor_for_edit(model, coords_base=(3, 5), coords_edit=(5, 6), use_cfg=True)
                    else:
                        set_attn_processor_for_edit(model, coords_base=(2, 3), coords_edit=(3, 4), use_cfg=True)

                    latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, transform_coords=transform_coordinates)

                    # if (i < adain_latents_steps * len(model.scheduler.timesteps[-start_time:]) and i > optimize_steps * len(model.scheduler.timesteps[-start_time:])):
                    #     print("adain")
                    #     with torch.no_grad():
                    #         latents[-1:] = adain_latents(latents[-1:], latents[:1]).detach()
            
            if ddim_latents is not None:
                i_n = len(ddim_latents) - 2 - i
                latents = torch.cat([ddim_latents[i_n].type_as(latents.detach()), latents[-1:].detach()], 0)
                
            if PROGRESS_BAR is not None:
                PROGRESS_BAR(i / NUM_DDIM_STEPS, desc="Optimizing Latents")

            if not type(controller).__name__ == "AttentionGeometryRemover" and (not type(controller).__name__ == "AttentionGeometryStitchSingle") :
            
                with torch.no_grad():
                    # print("latent warp")
                    if (i < len(model.scheduler.timesteps[-start_time:]) * latent_replace and mask_obj is not None) or (i < len(model.scheduler.timesteps[-start_time:]) * fast_start_steps):
                        t_coords = reshape_transform_coords(transform_coordinates, in_mat_shape=latents[1:].shape).type_as(latents)
                        i_mask_transform = controller.mask_new_warped[:1].detach()
                        i_mask = (T.Resize(size=(latents.shape[-1], latents.shape[-1]))(i_mask_transform) > 0.5) * 1.0
                        i_mask = i_mask.type_as(latents)

                        
                        # if edit_type == "geometry_stitch":

                        if i < len(model.scheduler.timesteps[-start_time:]) * fast_start_steps:
                            latents[-1:] = latents[:1] * (1 - i_mask) + i_mask * warp_grid_edit(latents[-2:-1].detach().clone(), t_coords, padding_mode='zeros', align_corners=True, mode=MODE)
                            
                        else:
                            latents[-1:] = latents[-1:] * (1 - i_mask) + i_mask * warp_grid_edit(latents[-2:-1].detach().clone(), t_coords, padding_mode='zeros', align_corners=True, mode=MODE)
                        # print(transform_coordinates)
                        # print("Warping latents")
                        # else:
                            # latents[1:] = latents[1:] * (1 - i_mask) + i_mask * warp_grid_edit(latents[:1].detach().clone(), t_coords, padding_mode='zeros', align_corners=True, mode=MODE)

        else:
            
            latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False, transform_coords=transform_coordinates)
        

        # print(latents.shape)

        # exit()
    # print(latents[1], ddim_latents[0].shape)
    # print(ddim_latents[0].min(), ddim_latents[0].max())
    # print(latents[1].min(), latents[1].max())
    # latents[1] = (((latents[1] - latents[1].min()) / ((latents[1].max() - latents[1].min()) + 1e-8)) * (ddim_latents[0].max() - ddim_latents[0].min())) + ddim_latents[0].min()
    # print(latents[1].min(), latents[1].max())

    if return_type == 'image':
        image = latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent, global_loss_log_dict


# @torch.autocast("cuda", dtype=torch.half)
@log_args(logging.getLogger())
def perform_geometric_edit(image, depth, image_mask, transform_in, prompt = "", ldm_stable_model = None, tokenizer_model = None, scheduler_in = None, 
    cross_replace_steps = {'default_': 0.95},
    self_replace_steps = 0.95,
    optimize_steps = 0.6,
    lr = 0.03,
    latent_replace = 0.6,
    optimize_embeddings = True,
    optimize_latents = True,
    obj_edit_step = 1.0,
    perform_inversion = True,
    guidance_scale = 7.5,
    skip_optim_steps = 1,
    num_ddim_steps = 50,
    splatting_radius = 1.3,
    edit_type = "geometry_editor",
    image_stitch = None,
    progress = None,
    fast_start_steps = 0.0,
    num_first_optim_steps = 1,
    loss_weights_dict = None,
    return_loss_log_dict = False,
    splatting_tau = 1.0,
    splatting_points_per_pixel = 15,
    use_adaptive_optimization = True,
    return_attention_maps = False,
    unet_path = "",
    use_optimizer = True,
    removal_loss_value_in = -1.5,
    ):
    torch.cuda.empty_cache()

    global SEED

    

    print("[INFO]: Using prompt: \"", prompt, "\" ")
    print("[INFO]: torch state: ", torch.get_rng_state())

    rng_state = torch.get_rng_state()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(SEED)
    torch.cuda.manual_seed_all(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # torch.manual_seed(SEED)


    max_opt = max(self_replace_steps, cross_replace_steps['default_'])
    if optimize_steps > max_opt:
        optimize_steps = max_opt

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # torch.autograd.detect_anomaly(check_nan=False)
    # torch.autograd.profiler.profile(enabled=False)
    torch.backends.cudnn.benchmark = True
    # torch.use_deterministic_algorithms(True)

    SPLATTER.clear_cache()
    SPLATTER.radius = splatting_radius
    SPLATTER.tau = splatting_tau
    SPLATTER.points_per_pixel = splatting_points_per_pixel

    global TOKENIZER, LDM_STABLE, SCHEDULER, PROGRESS_BAR
    global tokenizer, ldm_stable, scheduler

    PROGRESS_BAR = progress
    global GUIDANCE_SCALE, SKIP_OPTIM_STEPS, NUM_DDIM_STEPS, UNET_NAME
    
    GUIDANCE_SCALE = guidance_scale
    SKIP_OPTIM_STEPS = skip_optim_steps
    NUM_DDIM_STEPS = num_ddim_steps




    image_mask = torch.tensor(image_mask).float()
    # d_path = "/home/ec2-user/SageMaker/test_sd/DPT/outputs/debug_outputs_4/"
    # transforms_path = d_path + "transform_coordinates.npy"
    t_coords_depth, p_image, projected_image_amodal_mask = vis_utils.get_transform_coordinates(image / 255.0, depth, image_mask, transform_in = transform_in, return_mesh=True)



    if edit_type == "geometry_stitch" or edit_type == "geometry_stitch_single":
        if image_stitch is None:
            raise Exception("[ERROR]: No image provided for Stitching")


        t_coords_depth = torch.tensor(t_coords_depth)
        image_warped = warp_grid_edit((torch.tensor(image_stitch[None]).permute(0, -1, 1, 2) / 255.0).float(), t_coords_depth[None].float(), padding_mode='zeros', align_corners=True, mode=MODE)

        p_image = (image_warped[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")

        print(image_mask.shape, image_mask.max(), image.shape, image.max())
        mask_warp = warp_grid_edit((torch.tensor(image_mask[None, None])).float(), t_coords_depth[None].float(), padding_mode='zeros', align_corners=True, mode=MODE)

        mask_warped = (mask_warp[0, 0].detach().cpu().numpy() > 0.5)[..., None] * 1.0

        stitched_image = (p_image * mask_warped + image * (1.0 - mask_warped)).astype("uint8")
        # plt.imsave("./test.png", stitched_image)
        # exit()

        if edit_type == "geometry_stitch_single":
            image = stitched_image
        # else:
        #     print("Setting depth to 0.5 and using stitched image as image_stitch")
        #     image_stitch = stitched_image
        #     image_mask = torch.tensor(mask_warped[..., 0]).type_as(image_mask)
        #     transform_in = torch.eye(4).type_as(transform_in)
        #     # As image is stitched we set transform to I

        #     t_coords_depth, p_image = vis_utils.get_transform_coordinates(image / 255.0, 0.5 * np.ones_like(depth), image_mask, transform_in = transform_in)
        #     t_coords_depth = torch.tensor(t_coords_depth)
        #     # print(image_mask.shape, image_mask.dtype, image_mask.max(), image_mask.min())
        #     # print(image_mask.shape, image_mask.dtype, image_mask.max(), image_mask.min())
            

    transform_coordinates = t_coords_depth[None]
    transform_coordinates=torch.tensor(transform_coordinates).detach()

        
        


    ldm_stable = LDM_STABLE
    tokenizer = TOKENIZER
    scheduler = SCHEDULER

    print(unet_path, UNET_NAME, scheduler_in)

    if (scheduler_in is not None) and ((unet_path == "") or (unet_path == UNET_NAME)):
        print("using pre loaded model.")
        ldm_stable = ldm_stable_model
        tokenizer = tokenizer_model
        scheduler = scheduler_in
        

    elif (ldm_stable is None) or (tokenizer is None) or (scheduler is None) or ((unet_path != "") and (unet_path != UNET_NAME)):
        ldm_stable, tokenizer, scheduler = load_model(diffusion_model = DIFFUSION_MODEL, unet_path = unet_path, device = DEVICE)
        UNET_NAME = DIFFUSION_MODEL
        if unet_path != "":
            UNET_NAME = unet_path

    

    else:
        print("No model loading required")

    LDM_STABLE = ldm_stable
    TOKENIZER = tokenizer
    SCHEDULER = scheduler    

    null_inversion = NullInversion(ldm_stable, num_ddim_steps=NUM_DDIM_STEPS, uncond_text=UNCOND_TEXT, device=DEVICE, progress_bar = PROGRESS_BAR, guidance_scale = GUIDANCE_SCALE)

    state = torch.get_rng_state()

    if edit_type == "geometry_stitch_single":
        image_stitch = None


    (image_gt, image_enc), x_t, uncond_embeddings, ddim_latents, ddim_noise = null_inversion.invert(image, prompt, offsets=(0,0,0,0), verbose=True, perform_inversion = perform_inversion, image_2 = image_stitch)
    print("inversion done!")






    # prompts = ["", "road"]
    if edit_type == "geometry_stitch":
        prompts = [prompt, prompt, prompt]
        if image_stitch is None:
            raise Exception("[ERROR]: ]No image provided for Stitching")
    else:
        prompts = [prompt, prompt]

    image_mask_torch = image_mask
    m_i_1 = image_mask_torch[None, None]
    # Params


    if edit_type == "geometry_editor":
        controller = AttentionGeometryEdit(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=None, local_blend=None, controller=None, image_mask=image_mask.detach().cpu().numpy(), empty_scale = 0.0, use_all = False, obj_edit_step = obj_edit_step, tokenizer = tokenizer, device = DEVICE, mode = MODE)
    elif edit_type == "geometry_remover":
        controller = AttentionGeometryRemover(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=None, local_blend=None, controller=None, image_mask=image_mask.detach().cpu().numpy(), empty_scale = 0.0, use_all = False, obj_edit_step = obj_edit_step, tokenizer = tokenizer, device = DEVICE, mode = MODE)
    
    elif edit_type == "geometry_stitch":
        controller = AttentionGeometryStitch(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=None, local_blend=None, controller=None, image_mask=image_mask.detach().cpu().numpy(), empty_scale = 0.0, use_all = False, obj_edit_step = obj_edit_step)
    elif edit_type == "geometry_stitch_single":
        controller = AttentionGeometryStitchSingle(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, equalizer=None, local_blend=None, controller=None, image_mask=image_mask.detach().cpu().numpy(), empty_scale = 0.0, use_all = False, obj_edit_step = obj_edit_step)
    else:
        raise NotImplementedError

    if return_attention_maps:
        print("[INFO]: Storing Attention Maps")
        controller.store_attention_maps = True

    
    print("[INFO]: Setting Amodal Mask", projected_image_amodal_mask.shape)
    
    controller.amodal_mask = torch_erode(torch.tensor(projected_image_amodal_mask))


    if loss_weights_dict is not None:
        controller.loss_weight_dict = loss_weights_dict
        controller.default_loss_weights = loss_weights_dict
        print("[INFO]: Found loss weights dictionary for: ", edit_type)
    # elif edit_type == "geometry_stitch":
    #     lwd = {"self": {"sim": sim_loss_w_self, "scale": scale_loss_w_self}, "cross": { "sim": sim_loss_w_cross, "scale": scale_loss_w_cross}}
    #     controller.loss_weight_dict = lwd


    torch.set_rng_state(state)

    print("performing edit")

    generator = torch.Generator("cuda")
    generator.manual_seed(SEED)
    images, _, global_loss_log_dict = run_and_display(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, 
    transform_coordinates=transform_coordinates, mask_obj = m_i_1, optimize_steps=optimize_steps, latent_replace=latent_replace, lr = lr, optimize_embeddings = optimize_embeddings, optimize_latents=optimize_latents, ddim_latents=ddim_latents, verbose=False, ddim_noise = ddim_noise, edit_type = edit_type, fast_start_steps = fast_start_steps, num_first_optim_steps = num_first_optim_steps, use_adaptive_optimization = use_adaptive_optimization, removal_loss_value_in = removal_loss_value_in)


    # print(len(images))
    # print(images[1].shape, image.shape, image.max(), image.min(), images[1].max(), images[1].min())


    edited_image = images[-1]
    if edit_type == "geometry_editor":
        
        t_coords_depth = torch.tensor(t_coords_depth)[None]
        image_warped = warp_grid_edit((torch.tensor(image[None]).permute(0, -1, 1, 2) / 255.0).float(), t_coords_depth.float(), padding_mode='zeros', align_corners=True, mode=MODE)
        
        
        p_image = (image_warped[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")


        # print(p_image.min(), p_image.max(), p_image.shape)
        # p_image_m = (p_image).astype("uint8")
        mask_edit = controller.mask_new_warped[0, 0].detach().cpu().numpy()
        # print(mask_edit.shape, image_mask.shape, mask_edit.max(), image_mask.max())
        mask_im = image_mask.detach().cpu().numpy()
        mask_changed = ((mask_edit + mask_im) > 0.5) * 1.0
        mask_wo_edit = ((np.ones_like(mask_changed) - (mask_changed)) > 0.5) * 1.0
        p_image_new = (mask_wo_edit[..., None] * image + mask_edit[..., None] * p_image).astype("uint8")
        mask_source = ((mask_edit + mask_wo_edit) > 0.5) * 1.0
        # mask_template = ((mask_im + mask_wo_edit) > 0.5) * 1.0
        # edited_image = match_histograms(edited_image, image, channel_axis=-1)

        # p_image_new is being used which is already a warped image
        edited_image = masked_histogram_matching(edited_image, p_image_new, mask_source, mask_source)
        # plt.imsave("./test.png", p_image_new)
    elif edit_type == "geometry_stitch":
        mask_edit = controller.mask_new_warped[0, 0].detach().cpu().numpy()
        edited_image = masked_histogram_matching(edited_image, image, 1.0 - mask_edit)
    elif edit_type == "geometry_remover":

        # image_mask_hist = torch.from_numpy(image_mask[None])
        # print(image_mask.shape, image_mask.max())
        # image_mask_hist = torch_dilate(image_mask[None, None], 5)[0, 0].detach().cpu().numpy()
        # print(image_mask_hist.shape, image_mask.shape, image_mask_hist.max(), image_mask.max())
        edited_image = masked_histogram_matching(edited_image, image, 1.0 - image_mask)

    images[-1] = edited_image
    print("edit done")

    ldm_stable.unet.set_attn_processor(VanillaAttentionProcessor())
    
    del null_inversion
    # torch.set_rng_state(rng_state)
    if return_loss_log_dict:
        
        if return_attention_maps:
            return images, global_loss_log_dict, controller.attention_store
        return images, global_loss_log_dict

    if return_attention_maps:
        return images, controller.attention_store
    return images


def run_and_display(ldm_stable, prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, transform_coordinates=None, mask_obj = None, optimize_steps = 0.0, latent_replace = 0.0, lr = 0.0, optimize_embeddings = False, optimize_latents= True, ddim_latents = None, ddim_noise = None, edit_type = "geometry_editor", fast_start_steps = 0.0, num_first_optim_steps = 1, use_adaptive_optimization = True, removal_loss_value_in = -1.5):

    images, x_t, global_loss_log_dict = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, transform_coordinates=transform_coordinates, mask_obj = mask_obj, optimize_steps = optimize_steps, latent_replace = latent_replace, lr = lr, optimize_embeddings = optimize_embeddings, optimize_latents = optimize_latents, ddim_latents = ddim_latents, ddim_noise = ddim_noise, edit_type = edit_type, fast_start_steps = fast_start_steps, num_first_optim_steps = num_first_optim_steps, use_adaptive_optimization = use_adaptive_optimization, removal_loss_value_in = removal_loss_value_in)
    if verbose:
        view_images(images)
    return images, x_t, global_loss_log_dict