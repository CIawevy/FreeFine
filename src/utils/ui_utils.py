import gradio as gr
import cv2
import numpy as np
from GeoDiffuser.utils.depth_predictor import get_mask_prediction, get_monocular_depth, get_monocular_ZoeDepth, get_monocular_depth_anything, get_constant_depth, depth_corrector, get_mask_prediction_multiple_points
from GeoDiffuser.utils import vis_utils
import GeoDiffuser.utils.generic as io
import torch, os
import GeoDiffuser.utils.editor as ge
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import pyrealsense2 as rs
import argparse
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
from copy import deepcopy




SD_MODEL = None
TOKENIZER = None
SCHEDULER = None
DEPTH_ANYTHING_MODEL = None
MIDAS_DEPTH_MODEL = None


def count_folders(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

def create_folder(directory):
    os.makedirs(directory, exist_ok = True)

def complete_path(directory):
    return os.path.join(directory, "")

def read_image(im_path):

    im = plt.imread(im_path)
    # print(im.min(), im.max(), im.shape, im_path)

    if len(im.shape) == 3:
        im = im[..., :3]
    
    if im.max() <= 1.0:
        im = (im * 255.0).astype("uint8")
    # print(im.min(), im.max(), im.shape, im_path)
    return im


def save_exp(save_location_in, 
            input_img,
            input_depth,
            input_depth_vis,
            input_mask,
            transform_in,
            transformed_image = None,
            edited_image = None,
            background_image = None,
            h = 512,
            w = 512,
            exp_transform_type = "Mix",
            download_input = None,
            download_edit = None
            ):

    # print(input_img)
    # print(type(input_img))
    # print(input_img[0])
    # print(input_img[1])
    h = int(h)
    w = int(w)

    save_location = complete_path(save_location_in) + exp_transform_type
    create_folder(save_location)

    folder_num = count_folders(save_location)
    folder_num += 1

    save_folder = complete_path(complete_path(save_location) + str(folder_num))
    
    create_folder(save_folder)

    # print(type(input_img))
    # print(input_img.shape)

    plt.imsave(save_folder + "input_image.png", input_img)

    if transformed_image is not None:
        plt.imsave(save_folder + "transformed_image.png", transformed_image)
    if edited_image is not None:
        plt.imsave(save_folder + "result.png", edited_image)
    if background_image is not None:
        plt.imsave(save_folder + "background_image.png", background_image)
    
    if download_input is not None:
        plt.imsave(save_folder + "download_input.png", download_input)
    if download_edit is not None:
        plt.imsave(save_folder + "download_edit.png", download_edit)
    

    
    plt.imsave(save_folder + "input_mask.png", input_mask, cmap="gray")
    plt.imsave(save_folder + "depth.png", input_depth_vis, cmap = "gray")
    np.save(save_folder + "depth.npy", input_depth)
    np.save(save_folder + "transform.npy", transform_in)
    np.save(save_folder + "image_shape.npy", np.array([h, w]))
    # print("Saved everything")
    # return




def file_exists(f_path):
    return os.path.exists(f_path)

def read_exp(d_path):
    save_folder = complete_path(d_path)


    img_path = save_folder + "input_image.png"
    depth_path = save_folder + "depth.npy"
    mask_path = save_folder + "input_mask.png"
    bg_path = save_folder + "background_image.png"
    depth_vis_path = save_folder + "depth.png"
    transform_path = save_folder + "transform.npy"
    im_shape = save_folder + "image_shape.npy"
    
    transformed_image_path = save_folder + "transformed_image.png"
    result_path = save_folder + "result.png"
    result_ls_path = save_folder + "resized_result_ls.png"
    zero123_result_path = save_folder + "zero123/lama_followed_by_zero123_result.png"
    object_edit_result_path = save_folder + "object_edit/result_object_edit.png"
    dragon_diffusion_result_path = save_folder + "dragon_diffusion/result_dragon_diffusion.png"
    diffusion_handles_result_path = save_folder + "diffhandles/im_edited_diffhandles.png"
    freedrag_result_path = save_folder + "free_drag/result_free_drag_resized.png"
    resized_input_image = save_folder + "resized_input_image_png.png"
    resized_input_mask = save_folder + "resized_input_mask_png.png"
    
    all_paths = [img_path, depth_path, mask_path, bg_path, depth_vis_path, transform_path, transformed_image_path, result_path, im_shape, result_ls_path, zero123_result_path, resized_input_image, object_edit_result_path, resized_input_mask, dragon_diffusion_result_path, diffusion_handles_result_path, freedrag_result_path]
    
    out_dict = {}
    for f_name in all_paths:
        base_name = os.path.basename(f_name)
        key_name = base_name.split(".")[0]
        f_type = base_name.split(".")[1]
        
        if file_exists(f_name):
            if f_type == "png":
                out_dict[key_name + "_png"] = read_image(f_name)
            elif f_type == "npy":
                out_dict[key_name + "_npy"] = np.load(f_name)
        else:
            out_dict[key_name + "_" + f_type] = None
    if out_dict["image_shape_npy"] is None:
        out_dict["image_shape_npy"] = np.array([512, 512])
    out_dict["path_name"] = d_path
    return out_dict

def read_exp_ui(d_path):

    ed = read_exp(d_path)




    return ed["input_image_png"], ed["input_mask_png"], ed["background_image_png"], ed["depth_npy"], ed["depth_png"], ed["transform_npy"], ed["transformed_image_png"], ed["result_png"], ed["image_shape_npy"][0], ed["image_shape_npy"][1]




def list_exp_details(exp_dict):

    for k in exp_dict.keys():
        if exp_dict[k] is None:
            print(k, " None")
        elif k != "path_name":
            print(k, " ", exp_dict[k].shape, " ", exp_dict[k].min(), " ", exp_dict[k].max())
        elif k == "path_name":
            print(k, " ", exp_dict[k])


def depth_bilateral_filter(image, depth, sigma_color = 0.1, sigma_space=16, d=5):

    d_max = depth.max()
    # depth = d_max - depth

    print(sigma_color, sigma_space, d)

    im_in = (image / 255.0).astype("float32")
    if len(im_in.shape) == 3:
        d_in = np.concatenate([depth[..., None]] *3, -1 ).astype("float32")
    else:
        d_in = depth

    out = cv2.ximgproc.jointBilateralFilter(im_in, d_in, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    if len(im_in.shape) == 3:
        # d_out = d_max - out[..., 0]
        d_out = out[..., 0]
    else:
        # d_out = d_max - out
        d_out = out

    return d_out

def correct_depth(depth, im_mask, sigma_color = 0.1, sigma_space = 16, d = 5):

    # depth = depth_corrector(depth)    
    depth = depth_bilateral_filter(im_mask[..., 0], depth, sigma_color, sigma_space, d)



    return depth


def get_image_mask_from_contour(im_pick, contour):

    z = np.zeros(im_pick.shape).astype(np.uint8)
    c = np.array(contour)
    print(contour, z.shape, im_pick.shape, im_pick.dtype, c.shape, c.dtype)
    object_pick_mask = cv2.fillPoly(z, pts=[np.array(contour)], color=(255, 255, 255))
    # print(object_pick_mask.min(), object_pick_mask.max())
    object_pick_mask[object_pick_mask < 255.0 / 2.0] = 0.0
    object_pick_mask[object_pick_mask >= 255.0 / 2.0] = 255.0

    return object_pick_mask.astype("uint8")


def get_mask_armbench(input_img, data, point):

    w, h = point
    print(point)
    w = w / 512 * input_img.shape[1]
    h = h / 512 * input_img.shape[0]

    print(w, h)
    for k in data:
        mask = get_image_mask_from_contour(input_img, data[k]["boundary"])

        # plt.imsave("./test/" + k + ".jpg", mask)
        print(h, w, mask.max(), mask.min())
        if (mask[int(h), int(w), 0] >= (255.0 / 2)):
            print(data[k]["title"])
            return mask

    
    return 

def get_points(img,
               sel_pix,
               point_label,
               sam_model_path,
               evt: gr.SelectData):
    # collect the selected point
    sel_pix.append((point_label, evt.index))
    edit_img = deepcopy(img)
    # draw points

    input_points = []
    input_labels = []
    for point in sel_pix:
        w,h = point[1]
        input_points.append([float(w), float(h)])
        if point[0] == "Positive":
            cv2.circle(edit_img, tuple(point[1]), 10, (255, 0, 0), -1)
            input_labels.append(1)

        elif point[0] == "Negative":
            cv2.circle(edit_img, tuple(point[1]), 10, (0, 0, 255), -1)
            input_labels.append(0)

        else:
            raise ValueError("Unknown point type")

    if not isinstance(edit_img, np.ndarray):
        edit_img = np.array(edit_img)


    # input_img = np.array(Image.fromarray(img))
    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    mask_image = get_mask_prediction_multiple_points(edit_img, input_points, input_labels, model_path=sam_model_path)



    return edit_img, mask_image

def undo_point(img, sel_pix, sam_model_path):
    edit_img = deepcopy(img)

    input_points = []
    input_labels = []
    # draw points
    for point in sel_pix[:-1]:
        w,h = point[1]
        input_points.append([float(w), float(h)])

        if point[0] == "Positive":
            cv2.circle(edit_img, tuple(point[1]), 10, (255, 0, 0), -1)
            input_labels.append(1)

        elif point[0] == "Negative":
            cv2.circle(edit_img, tuple(point[1]), 10, (0, 0, 255), -1)
            input_labels.append(0)

        else:
            raise ValueError("Unknown point type")

    if not isinstance(edit_img, np.ndarray):
        edit_img = np.array(edit_img)

    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    mask_image = get_mask_prediction_multiple_points(edit_img, input_points, input_labels, model_path=sam_model_path)

    return edit_img, mask_image, sel_pix[:-1]

def get_mask(img,
               mask_image,
               sel_pix,
               sam_path,
               evt: gr.SelectData,
               segment_file = None,
               original_image = None,
               ):
    # collect the selected point
    # sel_pix = gr.State([])
    # input_img = cv2.resize(img.copy(), (512,512), interpolation=cv2.INTER_NEAREST)

    input_img = np.array(Image.fromarray(img).resize((512, 512)))
    # input_img = 
    sel_pix = []
    sel_pix.append(evt.index)
    # draw points
    # points = []
    point = sel_pix[0]

    w,h = point
    mask_image = None

    # if segment_file is not None:
    #     data = io.read_json(segment_file)

    #     mask_image = get_mask_armbench(original_image, data, (w, h))
        
    #     if mask_image is not None:
    #         mask_image = np.array(Image.fromarray(mask_image).resize((512, 512)))
    
    if mask_image is None:
        # print("Segment file is None")
        h_norm = h / 512
        w_norm = w / 512
        mask_image = get_mask_prediction(input_img, h_norm, w_norm, sam_path)

    
    return input_img, mask_image


def clear_transforms():

    translation_x = 0.0
    translation_y = 0.0
    translation_z = 0.0

    rotation_x = 0.0
    rotation_y = 0.0
    rotation_z = 0.0

    scale_x = 1.0
    scale_y = 1.0
    scale_z = 1.0
    

    return translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z, scale_x, scale_y, scale_z

def get_depth(input_img,
            depth_pt_path,
            depth_image,
            depth_image_vis,
            depth_model = "depth_anything",
            translate_factor=0.0,
            ):

    torch.cuda.empty_cache()


    if depth_model == "midas_depth":
        print("Using MIDAS DEPTH")
        depth_image = get_monocular_depth(input_img / 255.0, depth_pt_path, translate_factor = translate_factor)
    elif depth_model == "zoe_depth":
        print("Using ZOE DEPTH")
        depth_image = get_monocular_ZoeDepth(input_img)
    elif depth_model == "constant_depth":
        print("Using CONSTANT DEPTH")
        depth_image = get_constant_depth(input_img)
    else:
        print("Using DEPTH ANYTHING")
        depth_image = get_monocular_depth_anything(input_img, translate_factor = translate_factor)
    # print(depth_image.min())
    # print(depth_image.max())
    depth_image = depth_image.astype(np.float32)
    depth_image = depth_image.clip(min=0.0)

    depth_image_vis = (depth_image / (depth_image.max() + 1e-8) * 255.0).astype(np.uint8)
    # depth_image = get_monocular_ZoeDepth(image)

    return depth_image, depth_image_vis


def project_image(input_img, 
                    mask_image, 
                    depth_image,
                    transform_mask,
                    transform_in,
                    splatting_radius = 1.3, 
                    background_img = None,
                    return_transformed_mask = False,
                    splatting_tau = 1.0,
                    splatting_points_per_pixel = 15,
                    focal_length = 550,
                    remove_noise = False,
                    return_mask_projected_mesh = False,
                    duplicate = False,
                    ):

    use_softsplat = True

    t_coords_depth, p_image, mask_projected_mesh = vis_utils.get_transform_coordinates(input_img / 255.0, depth_image, mask_image[..., 0], transform_in = transform_in, use_softsplat = use_softsplat, focal_length = focal_length, return_mesh=True)
    t_coords_depth = torch.tensor(t_coords_depth)[None]

    # print("projected mask shape, out: ", mask_projected_mesh.shape, mask_projected_mesh.max())
    mask_projected_mesh = np.clip(mask_projected_mesh, 0, 1)
    # print(p_image.shape, p_image.min(), p_image.max())
    # p_image_np = p_image[0].permute(1, 2, 0).detach().cpu().numpy()
    # print(p_image_np.shape, p_image_np.max(), p_image_np.min())
    p_image = (p_image * 255.0).astype("uint8")

    # print(p_image.min(), p_image.max(), input_img.min(), input_img.max())
    t_coords_depth = t_coords_depth.detach().cpu()
    # print(t_coords_depth.shape)
    image_mask_warped = ge.warp_grid_edit(torch.tensor(mask_image[..., 0][None, None]).float(), t_coords_depth.float(), padding_mode='zeros', align_corners=True, mode=ge.MODE, splatting_radius = splatting_radius, splatting_tau = splatting_tau, splatting_points_per_pixel = splatting_points_per_pixel)

    image_warped = ge.warp_grid_edit((torch.tensor(input_img[None]).permute(0, -1, 1, 2) / 255.0).float(), t_coords_depth.float(), padding_mode='zeros', align_corners=True, mode=ge.MODE, splatting_radius = splatting_radius, splatting_tau = splatting_tau, splatting_points_per_pixel = splatting_points_per_pixel)

    p_image = (image_warped[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")


    if remove_noise:
        image_mask_warped = ((image_mask_warped / 255.0) > 0.5) * 1.0 
        image_mask_warped_hole_filled = ge.torch_dilate(ge.torch_erode(ge.torch_erode(ge.torch_dilate(image_mask_warped))))
        image_mask_warped = ge.torch_dilate(ge.torch_erode(image_mask_warped))
        # print(torch.sum(image_mask_warped), " sum mask warped")
        # print(image_mask_warped.max(), image_mask_warped.shape)
        image_mask_warped_hole_filled = image_mask_warped_hole_filled * 255.0
        image_mask_warped = image_mask_warped * 255.0

    transform_mask = image_mask_warped[0, 0].detach().cpu().numpy()
    transform_mask = transform_mask.astype(np.uint8)
    t_mask_1 = ((transform_mask[..., None] / 255.0) > 0.5) * 1.0

    # print(mask_image.shape, mask_image.max())
    b_img = input_img
    orange = np.array([255, 165, 0])
    original_mask = mask_image[..., :1] / 255.0
    # Coloring object removal region with orange
    # b_img = original_mask * (0.8 * orange[None, None] + 0.2 * b_img) + (1.0 - original_mask) * b_img
    if not duplicate:
        b_img = original_mask * orange[None, None] + (1.0 - original_mask) * b_img

    if remove_noise:
        t_mask_filled = image_mask_warped_hole_filled[0]

    if background_img is not None:
        b_img = background_img

    # exit()

    # print(t_mask_1.min(), t_mask_1.max())



    olive = np.array([128, 128, 0])
    mask_projected_m = mask_projected_mesh[0, 0][..., None]
    amodal_mask = (((mask_projected_m - t_mask_1) > 0.5) * 1.0)
    p_image_out = t_mask_1 * p_image + (1.0 - mask_projected_m) * b_img + amodal_mask * olive[None, None]
    # p_image_out[]

    p_image_out = p_image_out.astype(np.uint8)


    if return_transformed_mask:
        if return_mask_projected_mesh:
            return p_image_out, transform_in.detach().cpu().numpy(), t_mask_1, mask_projected_mesh

        return p_image_out, transform_in.detach().cpu().numpy(), t_mask_1

    if return_mask_projected_mesh:
        return p_image_out, transform_in.detach().cpu().numpy(), mask_projected_mesh
    return p_image_out, transform_in.detach().cpu().numpy()

def get_transformed_mask(input_img, 
                        mask_image, 
                        depth_image,
                        transform_mask,
                        translation_x, 
                        translation_y, 
                        translation_z, 
                        rotation_x, 
                        rotation_y, 
                        rotation_z,
                        transform_in,
                        splatting_radius, 
                        background_img = None,
                        scale_x = 1.0,
                        scale_y = 1.0,
                        scale_z = 1.0,
                        splatting_tau = 1.0,
                        splatting_points_per_pixel = 15,
                        focal_length = 550):

    use_softsplat = True
    torch.cuda.empty_cache()


    transform_in = torch.eye(4).float()

    
    transform_in = transform_in @ vis_utils.translateMatrix(translation_x, translation_y, translation_z).type_as(transform_in)
    

    if scale_x != 1.0:
        flip_t = torch.eye(4).type_as(transform_in)
        flip_t[0, 0] = scale_x
        transform_in = transform_in @ flip_t

    if scale_y != 1.0:
        flip_t = torch.eye(4).type_as(transform_in)
        flip_t[1, 1] = scale_y
        transform_in = transform_in @ flip_t

    if scale_z != 1.0:
        flip_t = torch.eye(4).type_as(transform_in)
        flip_t[2, 2] = scale_z
        transform_in = transform_in @ flip_t

    if rotation_x != 0.0:
        transform_in = transform_in @ vis_utils.rotateAxis(rotation_x, 0).type_as(transform_in)
    if rotation_y != 0.0:
        transform_in = transform_in @ vis_utils.rotateAxis(rotation_y, 1).type_as(transform_in)
    if rotation_z != 0.0:
        transform_in = transform_in @ vis_utils.rotateAxis(rotation_z, 2).type_as(transform_in)


    

    # print(background_img, " back image ")


    return project_image(input_img, mask_image, depth_image, transform_mask, transform_in, splatting_radius, background_img, splatting_tau = splatting_tau, splatting_points_per_pixel = splatting_points_per_pixel, focal_length=focal_length)

    



def load_global_model_variables():

    global SD_MODEL, TOKENIZER, SCHEDULER
    
    return SD_MODEL, TOKENIZER, SCHEDULER

def save_global_model_variables(ldm_stable_model, tokenizer_model, scheduler_in):

    global SD_MODEL, TOKENIZER, SCHEDULER
    
    if ldm_stable_model != None:
        print("Storing model!")
        SD_MODEL = ldm_stable_model
    if tokenizer_model != None:
        TOKENIZER = tokenizer_model
    if scheduler_in != None:
        SCHEDULER = scheduler_in



def get_edited_image(image, depth, image_mask, transform_in, edited_image, guidance_scale = 7.5, skip_steps = 1, num_ddim_steps = 50, lr = 0.03, cross_replace_steps = 0.97, self_replace_steps = 0.97, latent_replace = 0.6, optimize_steps = 0.95, splatting_radius = 1.3,
    movement_sim_loss_w_self = 0.74, movement_sim_loss_w_cross = 0.5, movement_loss_w_self = 6.5, movement_loss_w_cross = 3.34, movement_removal_loss_w_self = 4.34, movement_removal_loss_w_cross = 2.67, movement_smoothness_loss_w_self = 0.0, movement_smoothness_loss_w_cross = 0.0, amodal_loss_w_cross = 3.5, amodal_loss_w_self = 80.5, splatting_tau = 1.0, splatting_points_per_pixel = 15, prompt = "", diffusion_correction = 0.0, unet_path = "", removal_loss_value_in = -1.5,
    ldm_stable_model = None, 
    tokenizer_model = None, 
    scheduler_in = None, 
    optimize_embeddings = True,
    optimize_latents = True,
    perform_inversion = False, 
    progress=gr.Progress(track_tqdm=False)):

    torch.cuda.empty_cache()

    # # For diffusion correction


    loss_weights_dict = {
    "self":{"sim": movement_sim_loss_w_self, "movement": movement_loss_w_self, "smoothness": movement_smoothness_loss_w_self, "removal": movement_removal_loss_w_self, "amodal": amodal_loss_w_self},
    "cross": {"sim": movement_sim_loss_w_cross, "movement": movement_loss_w_cross, "smoothness": movement_smoothness_loss_w_cross, "removal": movement_removal_loss_w_cross, "amodal": amodal_loss_w_cross}}

    print(f"[INFO]: Using Removal Loss value: {removal_loss_value_in}")


    print(loss_weights_dict)
    cross_replace_steps = {'default_': cross_replace_steps}
    images = ge.perform_geometric_edit(image, depth, image_mask[..., 0] / 255.0, torch.tensor(transform_in).float(), prompt=prompt, ldm_stable_model=ldm_stable_model, tokenizer_model = tokenizer_model, scheduler_in = scheduler_in, cross_replace_steps =cross_replace_steps, self_replace_steps = self_replace_steps, optimize_steps = optimize_steps, lr = lr, latent_replace = latent_replace, optimize_embeddings = optimize_embeddings, optimize_latents = optimize_latents, obj_edit_step = 1.0 - diffusion_correction, perform_inversion = perform_inversion, guidance_scale = guidance_scale, skip_optim_steps = skip_steps, num_ddim_steps = num_ddim_steps, splatting_radius = splatting_radius, progress = progress, loss_weights_dict = loss_weights_dict, splatting_tau = splatting_tau, splatting_points_per_pixel = splatting_points_per_pixel, unet_path = unet_path, removal_loss_value_in = removal_loss_value_in)

    edited_image = images[-1]
    return edited_image


def inpaint_mask(image, image_mask, edited_image, guidance_scale = 7.5, skip_steps = 1, num_ddim_steps = 50, lr = 0.03, cross_replace_steps = 0.97, self_replace_steps = 0.97,latent_replace = 0.6, optimize_steps = 0.95, splatting_radius = 1.3, inpainting_sim_loss_w_self = 0.912, inpainting_sim_loss_w_cross = 0.909, inpainting_removal_loss_w_self = 66.67, inpainting_removal_loss_w_cross = 36.67, inpainting_smoothness_loss_w_self = 55, inpainting_smoothness_loss_w_cross = 33.34, unet_path = "", prompt = "", removal_loss_value_in = -1.5,
    ldm_stable_model = None, 
    tokenizer_model = None, 
    scheduler_in = None, 
    optimize_embeddings = True,
    optimize_latents = True,
    obj_edit_step = 1.0,
    perform_inversion = False, 
    progress=gr.Progress(track_tqdm=False)):
    torch.cuda.empty_cache()


    depth = np.ones_like(image)
    depth_image = np.ones_like(image)
    depth, _ = get_depth(image, "", depth, depth_image, depth_model = "constant_depth")
    transform_in = np.eye(4)

    # guidance_scale = 1.5

    print(f"[INFO]: Using Removal Loss value: {removal_loss_value_in}")

    loss_weights_dict = {
        "cross": {"sim": inpainting_sim_loss_w_cross, "removal": inpainting_removal_loss_w_cross, "smoothness": inpainting_smoothness_loss_w_cross}, 
        "self":{"sim": inpainting_sim_loss_w_self,"removal": inpainting_removal_loss_w_self, "smoothness": inpainting_smoothness_loss_w_self}}
    # lr = 0.008

    cross_replace_steps = {'default_': cross_replace_steps}
    images = ge.perform_geometric_edit(image, depth, image_mask[..., 0] / 255.0, torch.tensor(transform_in).float(), prompt, ldm_stable_model=ldm_stable_model, tokenizer_model = tokenizer_model, scheduler_in = scheduler_in, cross_replace_steps =cross_replace_steps, self_replace_steps = self_replace_steps, optimize_steps = optimize_steps, lr = lr, latent_replace = latent_replace, optimize_embeddings = optimize_embeddings, optimize_latents = optimize_latents, obj_edit_step = obj_edit_step, perform_inversion = perform_inversion, guidance_scale = guidance_scale, skip_optim_steps = skip_steps, num_ddim_steps = num_ddim_steps, splatting_radius = splatting_radius, edit_type = "geometry_remover", progress = progress, loss_weights_dict = loss_weights_dict, unet_path=unet_path, removal_loss_value_in=removal_loss_value_in)

    edited_image = images[-1]
    return edited_image


def get_stitched_image(image, image_background, depth, image_mask, transform_in, edited_image, guidance_scale = 7.5, skip_steps = 1, num_ddim_steps = 50, lr = 0.03, cross_replace_steps = 0.97, self_replace_steps = 0.97, latent_replace = 0.6, optimize_steps = 0.95, splatting_radius = 1.3, movement_loss_w_self = 37, movement_loss_w_cross = 3.24, sim_loss_w_self = 0.54, sim_loss_w_cross = 0.5, fast_start_steps = 0.0, num_first_optim_steps = 1, smoothness_loss_self = 20.0, smoothness_loss_cross = 20.0, stitching_sim_loss_out_w_self = 0.5, stitching_sim_loss_out_w_cross = 0.5, stitching_movement_loss_out_w_self = 0.0, stitching_movement_loss_out_w_cross = 0.0, splatting_tau = 1.0, splatting_points_per_pixel = 15,
    prompt = "", 
    ldm_stable_model = None, 
    tokenizer_model = None, 
    scheduler_in = None, 
    optimize_embeddings = True,
    optimize_latents = True,
    diffusion_correction = 0.0,
    perform_inversion = False, progress=gr.Progress(track_tqdm=False)):

    torch.cuda.empty_cache()


    cross_replace_steps = {'default_': cross_replace_steps}

    loss_weights_dict = {
        "cross": {"sim": sim_loss_w_self, "movement": movement_loss_w_cross, "smoothness":smoothness_loss_cross, "sim_out": stitching_sim_loss_out_w_cross, "movement_out":  stitching_movement_loss_out_w_cross}, 
        "self":{"sim": sim_loss_w_self, "movement": movement_loss_w_self, "smoothness": smoothness_loss_self, "sim_out": stitching_sim_loss_out_w_self, "movement_out": stitching_movement_loss_out_w_self}}

    print(loss_weights_dict)


    images = ge.perform_geometric_edit(image_background, depth, image_mask[..., 0] / 255.0, torch.tensor(transform_in).float(), prompt, ldm_stable_model=ldm_stable_model, tokenizer_model = tokenizer_model, scheduler_in = scheduler_in, cross_replace_steps =cross_replace_steps, self_replace_steps = self_replace_steps, optimize_steps = optimize_steps, lr = lr, latent_replace = latent_replace, optimize_embeddings = optimize_embeddings, optimize_latents = optimize_latents, obj_edit_step = 1.0 - diffusion_correction, perform_inversion = perform_inversion, guidance_scale = guidance_scale, skip_optim_steps = skip_steps, num_ddim_steps = num_ddim_steps, splatting_radius = splatting_radius, image_stitch = image, edit_type = "geometry_stitch", scale_loss_w_self = scale_loss_w_self, scale_loss_w_cross = scale_loss_w_cross, sim_loss_w_self = sim_loss_w_self, sim_loss_w_cross = sim_loss_w_cross, fast_start_steps = fast_start_steps,
    num_first_optim_steps = num_first_optim_steps, progress = progress, loss_weights_dict = loss_weights_dict, splatting_tau = splatting_tau, splatting_points_per_pixel = splatting_points_per_pixel)


    edited_image = images[-1]
    return edited_image


def correct_depth_new(d):

    d_max = d.max()
    d_out = d_max - d

    d_out = depth_corrector(d_out, "median")
    d_out = d_max - d_out
    return d_out


def save_transformed_mask_exp(exp_root_folder):



    folder_list = glob.glob(complete_path(exp_root_folder) + "**/")

    folder_list.sort()

    for exp_folder in folder_list:

        print("Performing edit on: ", exp_folder)
        # continue
        exp_folder = complete_path(exp_folder)
        exp_dict = read_exp(exp_folder)
        list_exp_details(exp_dict)
        depth = exp_dict["depth_npy"]

        p_image_out, transform_out, transformed_mask = project_image(exp_dict["input_image_png"], exp_dict["input_mask_png"], depth, exp_dict["transformed_image_png"], torch.tensor(exp_dict["transform_npy"]), 1.0, None, return_transformed_mask=True)

        out_path = exp_folder + "ours/"
        os.makedirs(out_path, exist_ok=True)
        print("Saving to: ", out_path + "transformed_mask_square.png")
        # print(transformed_mask.shape)
        plt.imsave(out_path + "transformed_mask_square.png", transformed_mask[..., 0], cmap = "gray")
        # exit()


# def depth_corrector_new(depth, filter_type = "gaussian"):

#     if filter_type == "median":
#         # print(depth.shape)
#         depth_corrected = medfilt(depth)
#     elif filter_type == "gaussian":
#         depth_corrected = gaussian_filter(depth, sigma = 0.75)

#     return depth_corrected`

def depth_scaler(depth, im_mask, scale = 0.5):

    
    im_mask = (im_mask / 255.0).astype("float32")[..., 0]
    d_out = np.copy(depth)
    # d_max = depth.max()
    # d_out = depth.max() - depth
    d_out[im_mask > 0.5] = d_out[im_mask > 0.5] * scale

    # d_out = d_max - d_out

    return d_out


def get_smoothness_loss(d):



    D_dh, D_dw = ge.gradient(d)

    loss = D_dh.abs().mean() + D_dw.abs().mean()

    return loss, D_dh.abs(), D_dw.abs()

def get_depth_smoothness_loss(d, m):


    # print(d.shape)
    smoothness_loss, dh, dw = get_smoothness_loss(d[None, None])

    # print(dh.shape, m.shape, dw.shape)
    loss = torch.sum(dh * m[:, :, 1:]) / (torch.sum(m[:, :, 1:]) + 1e-8)
    loss += torch.sum(dw * m[:, :, :, 1:]) / (torch.sum(m[:, :, :, 1:]) + 1e-8)
    
    
    # print(dh.shape, dw.shape, dh.max(), dw.max(), loss.item())

    return loss


def depth_smoothing(depth, mask, lr = 1.0):

    mask_in = (np.copy(mask[..., 0]) / 255.0).astype("float32")
    d = torch.tensor(depth)
    d.requires_grad=True
    m = (torch.tensor(mask_in)[None, None] > 0.5) * 1.0
    loss = get_depth_smoothness_loss(d, m)
    torch.autograd.backward(loss, inputs=[d], retain_graph=False)
    d_grad = torch.nan_to_num(d.grad, posinf=0.0, neginf=0.0, nan=0.0)

    d_out = d - lr * d_grad

    mask_in = (mask_in > 0.5) * 1.0
    d_out = d_out.detach().cpu().numpy()
    kernel = np.ones((3,3)).astype("float32")
    erosion = cv2.erode(mask_in , kernel,iterations = 1)

    # d_out_max = maximum_filter(d_out, 5)

    # d_out_max[erosion > 0.5] = d_out[erosion > 0.5]

    # d_inter = minimum_filter(d_out_max, 3)

    # boundary_region = (mask_in - erosion)

    # d_out[boundary_region > 0.5] = d_inter[boundary_region > 0.5]
    # # dilation = cv2.dilate(erosion, kernel,iterations = 1)


    # d_out_mask = d_out * mask_in

    # d_out_mask = cv2.dilate(d_out_mask, kernel, iterations= 1)

    # d_out = d_out_mask * mask_in + (1.0-mask_in) * d_out


    # d_out_med = medfilt(d_out)
    # print(d_out.shape)
    # exit()
    # d_out = medfilt(d_out, 5)
    return d_out.astype("float32")


def disparity_to_depth(depth, focal_length=550):

    # Assuming baseline to be 1
    return 512 / (np.clip(depth, 1, None))



def get_axis_points(scale = 1.0):

    points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3) * scale

    return points

# def project_points_to_3D(im, K, d):



def draw_axis(img, R, t, K, points, shift = None):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    # points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)

    # axisPoints[:1] = axisPoints[:1] / axisPoints[-1:]
    # axisPoints = K @ (R @ points.T + t[..., None])
    # axisPoints[:2] = axisPoints[:2] / axisPoints[-1:]

    # axisPoints = (axisPoints.T)[:, :2]
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    # axisPoints = axi
    # print(axisPoints[-1])

    if shift is not None:
        
        w_i, h_i = axisPoints[-1].ravel()
        # print(w_i)
        h_new = h_i / shift[0]
        w_new = w_i / shift[1]
        shift = np.array([w_new - w_i, h_new - h_i])
        # print(shift)
        axisPoints = axisPoints + shift[None]
    # print(axisPoints)
    axisPoints = axisPoints.astype("int")
    # axisPoints = np.concatenate(axisPoints[:, 1:], axisPoints[:, :1], -1)
    # axisPoints = 
    # print(axisPoints[-1].ravel(), axisPoints[1].ravel())
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    return img

def get_mask_center(mask):

    m = mask.copy()
    if m.max() > 1:
        m = m / 255.0
    if len(m.shape) > 2:
        m = m[..., 0]

    h_ind, w_ind = np.indices(m.shape)
    h_mean = h_ind[m > 0.5].mean()
    w_mean = w_ind[m > 0.5].mean()

    return w_mean, h_mean

def prepare_depth_for_projection(d, mask_in = None):

    depth = d.copy()
    if mask_in is not None:
        obj_mask = mask_in[..., 0] / 255.0
    else:
        obj_mask = None
        
    if np.sum(depth) == 0.5 * (depth.shape[0] * depth.shape[1]):
        depth = np.ones_like(depth) * 0.5

    else:
        # Normalize depth
        depth = depth / (depth.max() + 1e-8)

        depth[depth > 0.95] = 1000.0

    mask = (depth < 100.0) * 1.0

    if obj_mask is not None:
        mask = obj_mask * mask


    return depth, mask

def get_exp_types():
    
    exp_types = ["Removal", "Rotation_3D", "Rotation_2D", "Translation_3D", "Scaling", "Mix", "Translation_2D"]

    return exp_types

def check_if_exp_root(exp_root_folder, folder_list = None):

    if folder_list is None:    
        folder_list = glob.glob(complete_path(exp_root_folder) + "**/")
    
    exp_types = get_exp_types()


    for f in folder_list:
        # print(f.split("/"))
        if f.split("/")[-2] in exp_types:
            return True

    return False

if __name__=="__main__":



    d_path = "/oscar/scratch/rsajnani/rsajnani/research/2023/datasets/armbench-damiano/1/"

    im = io.read_image(d_path + "Pick.RGB.jpg")
    data = io.read_json(d_path + "labels.json")
    get_mask_armbench(im, data, (100, 100))


    ############################################################
    # parser = argparse.ArgumentParser(description="Read a directory path from the command line.")
    # parser.add_argument('--path', help="Specify the directory path", default="./ui_outputs/rotation_2")
    # args = parser.parse_args()

    # exp_root_folder = args.path
    # save_transformed_mask_exp(exp_root_folder)

    # exit()

    # ##########################################################
    # spat_filter = rs.spatial_filter() 
    # exp_dir = complete_path("./ui_outputs/rotation_2/1/")
    # exp = read_exp(exp_dir)

    # # input_img, mask_image, depth_image, transform_mask, transform_in, splatting_radius, background_img

    # splatting_radius = 1.3
    # splatting_tau = 1e-1
    # splatting_points_per_pixel = 15
    # depth = exp["depth_npy"]
    # im = exp["input_image_png"]
    # im_mask = exp["input_mask_png"]


    # print("running")
    # depth_new = depth

    # # depth_new = disparity_to_depth(depth)
    # # depth_new = depth_new.max() - depth_new


    # # depth_new = depth_smoothing(depth, im_mask, 1e7)
    # # depth_new = depth_bilateral_filter(im_mask[..., 0], depth, 100, 350, 5)

    # # depth_new = depth_scaler(depth, im_mask, 0.8)

    # # d_r = depth.max() - depth
    # # depth_new = 1/(d_r + 1e-8)
    # # depth_new = depth_new.max() - depth_new
    # print(np.max(depth_new[depth_new != np.inf]))
    # print(np.min(depth[(im_mask[..., 0] / 255.0) > 0.5]), np.max(depth[(im_mask[..., 0] / 255.0) > 0.5]))
    # print(np.min(depth_new[(im_mask[..., 0] / 255.0) > 0.5]), np.max(depth_new[(im_mask[..., 0] / 255.0) > 0.5]))
    # print(np.mean(depth_new[(im_mask[..., 0] / 255.0) > 0.5]), np.mean(depth[(im_mask[..., 0] / 255.0) > 0.5]))

    # print(np.sum(np.abs(depth_new - depth)))
    # # exit()
    # # exit()
    # # depth_new = spat_filter.process(depth)
    # # depth_new = correct_depth_new(exp["depth_npy"])

    # p_image_out, transform_out, mask_projected_mesh = project_image(exp["input_image_png"], exp["input_mask_png"], depth_new, exp["transformed_image_png"], torch.tensor(exp["transform_npy"]), splatting_radius, None, return_transformed_mask=False, splatting_tau=splatting_tau, splatting_points_per_pixel = splatting_points_per_pixel, remove_noise=True, return_mask_projected_mesh=True)

    # plt.imsave("./test/test_mask.png", mask_projected_mesh[0, 0], cmap="gray")
    # plt.imsave("./test/test_2.png", p_image_out)







    
