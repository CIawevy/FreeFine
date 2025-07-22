from GeoDiffuser.utils.editor import perform_geometric_edit

import os
from GeoDiffuser.utils import vis_utils, image_processing, diffusion
import GeoDiffuser.utils.generic as io
from GeoDiffuser.utils.ui_utils import read_exp, list_exp_details, complete_path, check_if_exp_root
import matplotlib.pyplot as plt
from GeoDiffuser.segment_anything import SamPredictor, sam_model_registry
from GeoDiffuser.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from GeoDiffuser.depth_anything.dpt import DepthAnything
import logging

"""Compute depth maps for images in the input folder.
"""
import glob
import torch
import cv2
import argparse
import numpy as np
from PIL import Image
import pickle
from scipy.ndimage import maximum_filter

# from geometry

import GeoDiffuser.utils.generic 

from torchvision.transforms import Compose

from GeoDiffuser.dpt.models import DPTDepthModel
from GeoDiffuser.dpt.midas_net import MidasNet_large
from GeoDiffuser.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from GeoDiffuser.utils.depth_predictor import *
from GeoDiffuser.utils.generic import log_args


INITIALIZED_LOGGER = False


def save_dictionary(d, pkl_path):
    with open(pkl_path, 'wb') as file:
        pickle.dump(d, file)

def load_dictionary(pkl_path):
    with open(pkl_path, 'rb') as file:
        loaded_dict = pickle.load(file)

    return loaded_dict


def log_dictionary_to_file(log_dict, file_path):

    global INITIALIZED_LOGGER
    if INITIALIZED_LOGGER:

        new_log_file = file_path
        new_file_handler = logging.FileHandler(new_log_file, 'w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        new_file_handler.setFormatter(formatter)

        # Get the root logger and replace the existing handler
        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            if isinstance(hdlr, logging.FileHandler):
                logger.removeHandler(hdlr)
        logger.addHandler(new_file_handler)

    else:
        # Set up logging
        logging.basicConfig(level=logging.DEBUG, filename=file_path)
        INITIALIZED_LOGGER = True

    for i in log_dict:

        # Log the dictionary
        # logging.debug
        log_dict[i]["optimization_step"] = i
        logging.debug("optimization_step: %d" % i)
        logging.debug('logged_dict_self: %s', log_dict[i]["self"])
        logging.debug('logged_dict_cross: %s', log_dict[i]["cross"])
        logging.debug("num_layers: %d" % log_dict[i]["num_layers"])



def generate_output_plot_and_save(exp_dict):

    im_spacing = 20
    input_im = exp_dict["input_image_png"]
    transformed_im = exp_dict["transformed_image_png"]
    final_result = exp_dict["final_result"]
    num_plots = 2
    im_list = [input_im, final_result]
    text_list = ["Input Image", "Edited Image"]
    
    
    if transformed_im is not None:
        num_plots = 3
        im_list = [input_im, transformed_im, final_result]
        text_list = ["Input Image", "Target Edit", "Edited Image"]


    h, w, _ = input_im.shape
    new_w = (num_plots * w + im_spacing * (num_plots - 1))
    
    final_image = np.zeros((h, new_w, input_im.shape[-1]))

    # for 
    # if num_plots == 2:
    #     h, w, _ = input_im.shape 

    current_w = 0
    for i in range(num_plots):

        final_image[:, current_w:current_w + w] = im_list[i]
        current_w = current_w + im_spacing + w

        # plt.subplot(num_plots, i + 1, 1)
        # plt.imshow(im_list[i])
        # plt.axis('off')
        # plt.title(text_list[i])

    exp_plot = exp_dict["path_name"] + "experiment.png"
    final_image = final_image.astype("uint8")
    # print(final_image.min(), final_image.max())
    plt.imsave(exp_plot, final_image)
    # plt.savefig(exp_plot, bbox_inches='tight', dpi=300)

    return


def save_results(exp_dict, out_image, loss_dict, edit_type = "geometry_editor", step_store = None):

    log_dictionary_to_file(loss_dict, exp_dict["path_name"] + "loss.log")
    save_dictionary(loss_dict, exp_dict["path_name"] + "loss.pkl")

    if step_store is not None:
        for k in step_store:

            if isinstance(step_store[k], list):
                if len(step_store[k]) > 0:
                    for a_idx in range(len(step_store[k])):
                        step_store[k][a_idx] = step_store[k][a_idx].detach().cpu().numpy() 
        save_dictionary(step_store, exp_dict["path_name"] + "attention.pkl")
    # l_dict = load_dictionary(exp_dict["path_name"] + "loss.pkl")

    aspect_ratio = exp_dict["image_shape_npy"]
    out_image_resized = image_processing.resize_image(out_image, aspect_ratio)

    plt.imsave(exp_dict["path_name"] + "result_ls.png", out_image)
    plt.imsave(exp_dict["path_name"] + "resized_result_ls.png", out_image_resized)

    if edit_type == "geometry_remover":
        mask = maximum_filter(exp_dict["input_mask_png"][..., :1] / 255.0, 20)
        # print(mask.min(), mask.max(), mask.shape)
        transformed_image = exp_dict["input_image_png"] 
        transformed_image = transformed_image * (1.0 - mask) + mask * (0.5 * 255 + 0.5 * transformed_image) 
        transformed_image = transformed_image.astype("uint8")
        exp_dict["transformed_image_png"] = transformed_image
    
    for k in exp_dict:
        if k.split("_")[-1] == "png":
            if exp_dict[k] is None:
                continue
            im = image_processing.resize_image(exp_dict[k], aspect_ratio)
            if k == "input_mask_png" or k == "depth_png":
                plt.imsave(exp_dict["path_name"] + "resized_" + k + ".png", im, cmap="gray")
            else:
                plt.imsave(exp_dict["path_name"] + "resized_" + k + ".png", im)

            exp_dict[k] = im



    exp_dict["final_result"] = out_image_resized
    generate_output_plot_and_save(exp_dict)


def run_exp_on_folder_single(exp_folder, exp_type, ldm_stable, tokenizer, scheduler):

    print("Performing edit on: ", exp_folder, " with exp type: ", exp_type)
    # continue
    exp_folder = complete_path(exp_folder)
    exp_dict = read_exp(exp_folder)
    list_exp_details(exp_dict)

    # print("[INFO]:  to path: ", out_path)
    images, loss_dict, step_store = perform_exp(exp_dict, edit_type = exp_type, ldm_stable_model = ldm_stable, tokenizer_model = tokenizer, scheduler_in = scheduler)
    out_image = images[-1]

    save_results(exp_dict, out_image, loss_dict, exp_type, step_store=step_store)
    logging.shutdown()




@log_args(logging.getLogger())
def perform_exp(exp_dict, prompt = "", ldm_stable_model = None, tokenizer_model = None, scheduler_in = None, 
    cross_replace_steps = {'default_': 0.9},
    self_replace_steps = 0.9,
    optimize_steps = 0.85,
    lr = 0.03,
    latent_replace = 0.4,
    optimize_embeddings = True,
    optimize_latents = True,
    obj_edit_step = 1.0,
    perform_inversion = False,
    skip_optim_steps = 2,
    guidance_scale = 5.0,
    num_ddim_steps = 50,
    edit_type = "geometry_editor"):



    splatting_tau = 1.0
    splatting_points_per_pixel = 15
    splatting_radius = 1.3
    image = exp_dict["input_image_png"]
    image_mask = (exp_dict["input_mask_png"] / 255.0)[..., 0]
    depth = exp_dict["depth_npy"]
    transform_in = torch.tensor(exp_dict["transform_npy"]).float()

    image_stitch = None
    loss_weights_dict = None

    if edit_type == "geometry_stitch_single":
        image = exp_dict["background_image_png"]
        image_stitch = exp_dict["input_image_png"]
        loss_weights_dict = {"self": {"sim": 0.7, "scale": 40.0, "smoothness": 0.0, "sim_out": 0.95, "scale_out": 3.0},
                                "cross": {"sim": 0.5, "scale": 5, "smoothness": 0.0, "sim_out": 0.95, "scale_out": 0.1}}

    if edit_type == "geometry_stitch":
        lr = 0.03
        image = exp_dict["background_image_png"]
        image_stitch = exp_dict["input_image_png"]
        
        loss_weights_dict = {"self": {"sim": 20.0, "movement": 40.0, "smoothness": 2.0, "sim_out": 120.0, "movement_out": 40},
                                "cross": {"sim": 2.5, "movement": 5, "smoothness": 1.0, "sim_out": 30.0, "movement_out": 30}}
        
        latent_replace = 0.2

        splatting_tau = 1e-1
        splatting_radius = 1.0
        splatting_points_per_pixel = 30

        
        # loss_weights_dict = {"self": {"sim": 20.0, "movement": 40.0, "smoothness": 0.0, "sim_out": 1300.0, "movement_out": 0},
        #                         "cross": {"sim": 2.5, "movement": 5, "smoothness": 0.0, "sim_out": 350.0, "movement_out": 0}}
        
        # optimize_steps = 0.9
        
    if edit_type == "geometry_remover":
        guidance_scale = 5.0
        
        # loss_weights_dict = {"self": {"sim": 0.97, "scale": 75, "smoothness": 30.0},
        # "cross": {"sim": 0.92, "scale": 45.0, "smoothness": 15.0}}
        # loss_weights_dict = {"self": {"sim": 55, "removal": 2.6, "smoothness": 90.0},
        #                         "cross": {"sim": 45, "removal": 2.6, "smoothness": 60.0}}
        loss_weights_dict = {"self": {"sim": 55, "removal": 4.6, "smoothness": 30.0},
                                "cross": {"sim": 45, "removal": 4.6, "smoothness": 15.0}}
        # guidance_scale = 3.0

    if edit_type == "geometry_editor":
        # loss_weights_dict = {"self": {"sim": 55, "movement": 15.5, "removal": 1.6, "smoothness": 10.0},
        #                         "cross": {"sim": 45, "movement": 8.34, "removal": 1.6, "smoothness": 5.0}}
        # splatting_radius = 1.3
        # splatting_tau = 0.5
        # optimize_steps = 0.85

        # splatting_points_per_pixel = 15

        # Old
        # loss_weights_dict = {"self": {"sim": 55, "movement": 30.5, "removal": 2.6, "smoothness": 30.0, "amodal": 80.5},
        #                         "cross": {"sim": 45, "movement": 30.34, "removal": 2.6, "smoothness": 15.0, "amodal": 3.5}}

        # EC sub
        # loss_weights_dict = {"self": {"sim": 55, "movement": 15.5, "removal": 2.6, "smoothness": 30.0},
        #                 "cross": {"sim": 45, "movement": 8.34, "removal": 2.6, "smoothness": 15.0}}


        loss_weights_dict = {"self": {"sim": 55, "movement": 30.5, "removal": 2.6, "smoothness": 30.0, "amodal": 80.5},
                                "cross": {"sim": 45, "movement": 30.34, "removal": 2.6, "smoothness": 15.0, "amodal": 3.5}}


        # SD 2.1
        # loss_weights_dict = {"self": {"sim": 55, "movement": 30.5, "removal": 2.6, "smoothness": 30.0},
        #                         "cross": {"sim": 45, "movement": 25.34, "removal": 2.6, "smoothness": 15.0}}

        splatting_radius = 1.3
        splatting_tau = 1.0
        optimize_steps = 0.65
        latent_replace = 0.1
        splatting_points_per_pixel = 15
        guidance_scale = 3.0

        cross_replace_steps = {'default_': 0.95}
        self_replace_steps = 0.95
        obj_edit_step = 0.9
        # lr = 0.01


    use_adaptive_optimization = True
    step_store = None

    # print(loss_weights_dict)
    # print(transform_in)
    # logging.debug()
    # images, loss_dict, step_store = perform_geometric_edit(image, depth, image_mask, transform_in, prompt, ldm_stable_model=ldm_stable_model, tokenizer_model = tokenizer_model, scheduler_in = scheduler_in, cross_replace_steps =cross_replace_steps, self_replace_steps = self_replace_steps, optimize_steps = optimize_steps, lr = lr, latent_replace = latent_replace, optimize_embeddings = optimize_embeddings, optimize_latents = optimize_latents, obj_edit_step = obj_edit_step, perform_inversion = perform_inversion, skip_optim_steps = skip_optim_steps, guidance_scale = guidance_scale, num_ddim_steps = num_ddim_steps, edit_type=edit_type, image_stitch = image_stitch, fast_start_steps = 0.0, num_first_optim_steps = 1, return_loss_log_dict = True, loss_weights_dict = loss_weights_dict, splatting_radius = splatting_radius, splatting_tau = splatting_tau, splatting_points_per_pixel = splatting_points_per_pixel, use_adaptive_optimization = use_adaptive_optimization, return_attention_maps=False)

    images, loss_dict = perform_geometric_edit(image, depth, image_mask, transform_in, prompt, ldm_stable_model=ldm_stable_model, tokenizer_model = tokenizer_model, scheduler_in = scheduler_in, cross_replace_steps =cross_replace_steps, self_replace_steps = self_replace_steps, optimize_steps = optimize_steps, lr = lr, latent_replace = latent_replace, optimize_embeddings = optimize_embeddings, optimize_latents = optimize_latents, obj_edit_step = obj_edit_step, perform_inversion = perform_inversion, skip_optim_steps = skip_optim_steps, guidance_scale = guidance_scale, num_ddim_steps = num_ddim_steps, edit_type=edit_type, image_stitch = image_stitch, fast_start_steps = 0.0, num_first_optim_steps = 1, return_loss_log_dict = True, loss_weights_dict = loss_weights_dict, splatting_radius = splatting_radius, splatting_tau = splatting_tau, splatting_points_per_pixel = splatting_points_per_pixel, use_adaptive_optimization = use_adaptive_optimization, return_attention_maps=False)



    # exit()

    return images, loss_dict, step_store


if __name__=="__main__":

    ldm_stable, tokenizer, scheduler = None, None, None
    # ldm_stable, tokenizer, scheduler = diffusion.load_model()

    # exp_type = "geometry_stitch"
    # exp_root_folder = "./ui_outputs/stitching/"


    # exp_type = "geometry_remover"
    # exp_root_folder = "./ui_outputs/Removal_Ablation/"

    # exp_type = "geometry_editor"
    # exp_root_folder = "./ui_outputs/large_scale_study_optimizer_sd_test/"
    exp_root_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/GeometryDiffuser/dataset/large_scale_study_optimizer_sd_test"
    # exp_root_folder = "./ui_outputs/large_scale_study_all/large_scale_study_dataset_metrics_wacv_test"
    # exp_root_folder = "./ui_outputs/large_scale_study/"
    # exp_root_folder = "./ui_outputs/large_scale_study_xl/"
    # exp_root_folder = "./ui_outputs/metric_ablations/large_scale_study_dataset_metrics_2_0.6_attn_sharing/"    
    # exp_root_folder = "./ui_outputs/metric_ablations/large_scale_study_dataset_metrics_2_0.75_attn_sharing/"
    # exp_root_folder = "./ui_outputs/metric_ablations/large_scale_study_dataset_metrics_2_wo_adaptive_optimization/"
    # exp_root_folder = "./ui_outputs/teasers"

    # exp_root_folder = "./ui_outputs/inpainting/"

    # exp_root_folder = "./ui_outputs/editing/"
    # exp_root_folder = "./ui_outputs/rotation_2/"
    exp_root_folder = "/oscar/scratch/rsajnani/rsajnani/research/2023/GeometryDiffuser/GeometryDiffuser/ui_outputs/test/"

    folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    folder_list.sort()
    # exp_folder = "./ui_outputs/teasers/Mix/1/"
    # exp_folder = "./ui_outputs/large_scale_study_optimizer/Translation_2D/30/"

    # exp_type = "geometry_editor"
    # exp_folder = "./ui_outputs/large_scale_study_all/large_scale_study_dataset_metrics_wacv_test/Translation_2D/24/"
    # exp_folder = "./ui_output/check_2/Mix/1"
    # exp_folder = "/users/rsajnani/scratch/rsajnani/research/2023/test_sd/test_sd/prompt-to-prompt/ui_outputs/large_scale_study_all/visuals/Translation_2D/16/"
    # run_exp_on_folder_single(exp_folder, exp_type, ldm_stable, tokenizer, scheduler)
    # exit()

    # print(folder_list)

    print(folder_list)
    # if complete_path(exp_root_folder) + "Mix/" in folder_list:

    if check_if_exp_root(exp_root_folder):
        print("Root found!")
        root_folders_list = folder_list
    
    
        for f in root_folders_list:
            folder_list = glob.glob(complete_path(f) + "**/")
            folder_list.sort()

            exp_cat = f.split("/")[-2]
            print(f, exp_cat)
            if exp_cat == "Rotation_2D" or exp_cat == "Scaling":
                # print("Continue")
            # if exp_cat == "Removal" or exp_cat == "Rotation_2D":
                continue
            # print(f, folder_list, exp_cat)

            if exp_cat == "Removal":
                exp_type = "geometry_remover"
            else:
                exp_type = "geometry_editor" 

            # if exp_cat != "Translation_3D":
            #     continue
                
            # exit()
            for exp_folder in folder_list:
                run_exp_on_folder_single(exp_folder, exp_type, ldm_stable, tokenizer, scheduler)
                print("Completed: ", exp_folder)
    
            # exit()
    else:
        for exp_folder in folder_list:
            run_exp_on_folder_single(exp_folder, exp_type, ldm_stable, tokenizer, scheduler)


    exit()
