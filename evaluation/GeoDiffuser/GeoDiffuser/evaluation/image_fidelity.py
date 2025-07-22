import argparse
import os
from einops import rearrange
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as F
import lpips
import clip
import ui_utils
import glob
from GeoDiffuser.evaluation.dift_sd import SDFeaturizer
from pytorch_lightning import seed_everything
import cv2
import matplotlib.pyplot as plt
from GeoDiffuser.utils import vis_utils
import tqdm
import geometry_editor_updated as ge

# code adapted from: https://github.com/Yujun-Shi/DragDiffusion

# def read_image():
#     pass


def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float()[..., :3] / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def get_Matches(im1, im2, mask):


    im2 = np.array(Image.fromarray(im2).resize(Image.fromarray(im1).size, PIL.Image.BILINEAR))

    # orb = cv2.ORB_create()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1,None) #this finds keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(im2,None)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #create a bfMatcher object
    bf = cv2.BFMatcher() #create a bfMatcher object
    matches = bf.knnMatch(des1,des2, k=2) #Match descriptors
    # matches = sorted(matches, key = lambda x:x.distance) #sorts them in order of their distance - lowest distance first.

    matches_mask = []
    interest_points = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
        #     good.append([m])
        # for m in matches:
            # print(m.queryIdx, m.trainIdx)
            k_1 = kp1[m.queryIdx].pt
            k_2 = kp2[m.trainIdx].pt

            if mask[int(k_1[1]), int(k_1[0])] > 0.5:
                matches_mask.append(m)
                interest_points.append([int(k_1[1]), int(k_1[0])])




    if len(interest_points) == 0:
        print("NO INTEREST POINTS FOUND")
        interest_points = detect_interest_points(im1, mask)
    else:
        interest_points = np.array(interest_points)

    # img3 = cv2.drawMatches(im1,kp1,im2,kp2,matches_mask[:-1],None, flags=2) #helps us to draw the matches.
    # plt.imsave("./test/kps.png", img3)
    return interest_points

    # plt.imshow(img3)
    # plt.show()


def detect_interest_points(img, mask = None):

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    kp_new = []

    interest_points = []
    for k in kp:
        k_h, k_w = int(k.pt[1]), int(k.pt[0])
        if mask is not None:
            if mask[k_h, k_w] >= 0.5:
                interest_points.append([k_h, k_w])
                kp_new.append(k)

        else:
            interest_points.append([k_h, k_w])

    interest_points = np.array(interest_points)

    return interest_points
    # print(interest_points.shape)

    # print(kp[:2].pt)
    # compute the descriptors with ORB
    # kp, des = orb.compute(img, kp)

    # print(kp)
    # draw only keypoints location,not size and orientation
    # img2 = cv2.drawKeypoints(img, kp_new, None, color=(0,255,0), flags=0)
    
    # img2 = cv2.line(img2, (int(kp[0].pt[0]), int(kp[0].pt[1])), (int(kp[0].pt[0]), int(kp[0].pt[1])+ 30), (0,0,255), 3)

    # plt.imsave("./test/kps.png", img2)

def transform_coords_to_pixel_flow(t_coords):

    h, w = t_coords.shape[:2]
    X = (t_coords[..., 0] + 1) * (w - 1) / 2.0 # w
    Y = (t_coords[..., 1] + 1) * (h - 1) / 2.0 # h

    return np.stack([Y, X], -1)

def get_md_score_from_images(im_source, im_edit, transform_in, depth, mask, dift, prompt="", device="cuda", all_dist = [], t_coords = None, p_image = None, p_mask = None, max_points = 30):



    kps = get_Matches(im_source, im_edit, mask[..., 0] / 255.0)
    im_edit = np.array(Image.fromarray(im_edit).resize(Image.fromarray(im_source).size, PIL.Image.BILINEAR))


    # print(im_edit.max(), p_image.max(), " warp loss")
    warp_loss = np.sum(np.abs(im_edit / 255.0 - p_image / 255.0) * p_mask[..., None]) / (3 * np.sum(p_mask) + 1e-8)

    # print(warp_loss)

    source_image_tensor = preprocess_image(im_source, device)
    edited_image_tensor = preprocess_image(im_edit, device)

    if t_coords is None:
        t_coords, p_image = vis_utils.get_transform_coordinates(im_source, depth, obj_mask = mask[:, :, 0] / 255.0, transform_in = torch.tensor(transform_in))
    # else:
        # print("reusing t_coords")

    t_coords_pixels = transform_coords_to_pixel_flow(t_coords)
    # print(t_coords_pixels.shape, t_coords_pixels.max(), t_coords_pixels.min())

    # exit()
    H, W = source_image_tensor.shape[-2:]

    ft_source = dift.forward(source_image_tensor,
            prompt=prompt,
            t=261,
            up_ft_index=1,
            ensemble_size=8)
    ft_source = F.interpolate(ft_source, (H, W), mode='bilinear')

    ft_edited = dift.forward(edited_image_tensor,
            prompt=prompt,
            t=261,
            up_ft_index=1,
            ensemble_size=8)
    ft_edited = F.interpolate(ft_edited, (H, W), mode='bilinear')
    kps = kps[:max_points]

    # print(ft_source.shape)
    cos = torch.nn.CosineSimilarity(dim=1)
    # all_dist = []
    for k in kps:
        num_channel = ft_source.size(1)
        src_vec = ft_source[0, :, k[0], k[1]].view(1, num_channel, 1, 1)
        cos_map = cos(src_vec, ft_edited).cpu().numpy()[0]  # H, W
        max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape) # the matched row,col
        tp = torch.tensor(t_coords_pixels[k[0], k[1]])
        # calculate distance
        dist = (tp - torch.tensor(max_rc)).float().norm()
        # print(k[0], k[1], max_rc, tp, dist)
        all_dist.append(dist)
    # exit()
    return warp_loss

def get_loss_sim_scores_from_images(im_source, im_edit, loss_fn_alex, clip_model, clip_preprocess, device="cuda"):
    
    source_image_lpips = preprocess_image(im_source, device)
    edited_image_lpips = preprocess_image(im_edit, device)

    # compute LPIPS
    with torch.no_grad():
        source_image_224x224 = F.interpolate(source_image_lpips, (224,224), mode='bilinear')
        edited_image_224x224 = F.interpolate(edited_image_lpips, (224,224), mode='bilinear')
        cur_lpips_loss = loss_fn_alex(source_image_224x224, edited_image_224x224)

    # compute CLIP similarity
    source_image_clip = clip_preprocess(Image.fromarray(im_source)).unsqueeze(0).to(device)
    edited_image_clip = clip_preprocess(Image.fromarray(im_edit)).unsqueeze(0).to(device)

    with torch.no_grad():
        source_feature = clip_model.encode_image(source_image_clip)
        edited_feature = clip_model.encode_image(edited_image_clip)
        source_feature /= source_feature.norm(dim=-1, keepdim=True)
        edited_feature /= edited_feature.norm(dim=-1, keepdim=True)
        cur_clip_sim = (source_feature * edited_feature).sum()
        cur_clip_sim = cur_clip_sim.cpu().numpy()
        
    return  cur_lpips_loss.item(), cur_clip_sim

def get_loss_and_sim_score_folder(exp_folder, loss_fn_alex, clip_model, clip_preprocess, dift = None, device="cuda"):

    exp_folder = ui_utils.complete_path(exp_folder)
    exp_dict = ui_utils.read_exp(exp_folder)


    input_image = exp_dict["resized_input_image_png_png"]
    our_result = exp_dict["resized_result_ls_png"]
    zero123_result = exp_dict["lama_followed_by_zero123_result_png"]
    object_edit_result = exp_dict["result_object_edit_png"]
    dragon_diffusion_result = exp_dict["result_dragon_diffusion_png"]
    freedrag_result = exp_dict["result_free_drag_resized_png"]
    diffusion_handles_result = exp_dict["im_edited_diffhandles_png"]


    input_image_sq = exp_dict["input_image_png"]
    input_mask_sq = exp_dict["input_mask_png"] 
    depth = exp_dict["depth_npy"]
    transform_in = exp_dict["transform_npy"]

    output_result_dict = {}
    if our_result is not None:
        lpips_loss_ours, clip_sim_ours = get_loss_sim_scores_from_images(input_image, our_result, loss_fn_alex, clip_model, clip_preprocess, device)
        output_result_dict["ours"] = (lpips_loss_ours, clip_sim_ours)
    else:
        output_result_dict["ours"] = None


    if zero123_result is not None:
        lpips_loss_zero123, clip_sim_zero123 = get_loss_sim_scores_from_images(input_image, zero123_result, loss_fn_alex, clip_model, clip_preprocess, device)
        output_result_dict["zero123"] = (lpips_loss_zero123, clip_sim_zero123)
    else:
        output_result_dict["zero123"] = None

    if object_edit_result is not None:
        lpips_loss_object_edit, clip_sim_object_edit = get_loss_sim_scores_from_images(input_image, object_edit_result, loss_fn_alex, clip_model, clip_preprocess, device)
        output_result_dict["object_edit"] = (lpips_loss_object_edit, clip_sim_object_edit)

    else:
        output_result_dict["object_edit"] = None


    if dragon_diffusion_result is not None:
        lpips_loss_dragon_diffusion, clip_sim_dragon_diffusion = get_loss_sim_scores_from_images(input_image, dragon_diffusion_result, loss_fn_alex, clip_model, clip_preprocess, device)
        output_result_dict["dragon_diffusion"] = (lpips_loss_dragon_diffusion, clip_sim_dragon_diffusion)

    else:
        output_result_dict["dragon_diffusion"] = None
    

    if diffusion_handles_result is not None:
        print("[INFO]: Running Diffusion Handles, lpips and CLIP")

        lpips_loss_diffusion_handles, clip_sim_diffusion_handles = get_loss_sim_scores_from_images(input_image, diffusion_handles_result, loss_fn_alex, clip_model, clip_preprocess, device)
        output_result_dict["diffusion_handles"] = (lpips_loss_diffusion_handles, clip_sim_diffusion_handles)

    else:
        output_result_dict["diffusion_handles"] = None

    if freedrag_result is not None:
        print("[INFO]: Running FreeDrag, lpips and CLIP")
        
        lpips_loss_freedrag, clip_sim_freedrag = get_loss_sim_scores_from_images(input_image, freedrag_result, loss_fn_alex, clip_model, clip_preprocess, device)
        output_result_dict["freedrag"] = (lpips_loss_freedrag, clip_sim_freedrag)

    else:
        output_result_dict["freedrag"] = None

        
    return output_result_dict



def get_md_list_folder(exp_folder, dift, device):

    exp_folder = ui_utils.complete_path(exp_folder)
    exp_dict = ui_utils.read_exp(exp_folder)


    input_image = exp_dict["resized_input_image_png_png"]
    our_result = exp_dict["resized_result_ls_png"]
    zero123_result = exp_dict["lama_followed_by_zero123_result_png"]
    object_edit_result = exp_dict["result_object_edit_png"]
    dragon_diffusion_result = exp_dict["result_dragon_diffusion_png"]
    freedrag_result = exp_dict["result_free_drag_resized_png"]
    diffusion_handles_result = exp_dict["im_edited_diffhandles_png"]



    input_image_sq = exp_dict["input_image_png"]
    input_mask_sq = exp_dict["input_mask_png"] 
    depth = exp_dict["depth_npy"]
    transform_in = exp_dict["transform_npy"]

    all_dist_ours = []
    all_dist_zero123 = []
    all_dist_object_edit = []
    all_dist_dragon_diffusion = []
    all_dist_diffusion_handles = []
    all_dist_freedrag = []
    
    output_result_dict = {}

    t_coords, p_image = vis_utils.get_transform_coordinates(input_image_sq, depth, obj_mask = input_mask_sq[:, :, 0] / 255.0, transform_in = torch.tensor(transform_in))

    image_warped = ge.warp_grid_edit((torch.tensor(input_image_sq[None]).permute(0, -1, 1, 2) / 255.0).float(), torch.tensor(t_coords)[None].float(), padding_mode='zeros', align_corners=True, mode=ge.MODE, splatting_radius = 1.3, splatting_tau = 1.0, splatting_points_per_pixel = 15)

    p_image = (image_warped[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")

    image_mask_warped = ge.warp_grid_edit((torch.tensor(input_mask_sq[None]).permute(0, -1, 1, 2) / 255.0).float(), torch.tensor(t_coords)[None].float(), padding_mode='zeros', align_corners=True, mode=ge.MODE, splatting_radius = 1.3, splatting_tau = 1.0, splatting_points_per_pixel = 15)

    image_mask_warped = (image_mask_warped[0].permute(1, 2, 0).detach().cpu().numpy()[..., 0] > 0.5) * 1.0


    # # print(p_image.shape, p_image.max(), p_image.dtype)
    # plt.imsave("./test/projected_image_test.png", p_image)
    # plt.imsave("./test/projected_mask_test.png", image_mask_warped)
    # plt.imsave("./test/input_image_test.png", input_image_sq.astype("uint8"))

    # exit()
    if our_result is not None:        
        if "ours" not in output_result_dict:
            output_result_dict["ours"] = {}
        warp_loss_ours = get_md_score_from_images(input_image_sq, our_result, transform_in, depth, input_mask_sq, dift, prompt="", device=device, all_dist = all_dist_ours, t_coords = t_coords, p_image = p_image, p_mask = image_mask_warped)
        output_result_dict["ours"]["md"] = all_dist_ours
        output_result_dict["ours"]["warp"] = warp_loss_ours

    if zero123_result is not None:
        if "zero123" not in output_result_dict:
            output_result_dict["zero123"] = {}
        warp_loss_zero123 = get_md_score_from_images(input_image_sq, zero123_result, transform_in, depth, input_mask_sq, dift, prompt="", device=device, all_dist = all_dist_zero123, t_coords = t_coords, p_image = p_image, p_mask = image_mask_warped)
        output_result_dict["zero123"]["md"] = all_dist_zero123
        output_result_dict["zero123"]["warp"] = warp_loss_zero123


    if object_edit_result is not None:
        if "object_edit" not in output_result_dict:
            output_result_dict["object_edit"] = {}
        
        warp_loss_object_edit = get_md_score_from_images(input_image_sq, object_edit_result, transform_in, depth, input_mask_sq, dift, prompt="", device=device, all_dist = all_dist_object_edit, t_coords = t_coords, p_image = p_image, p_mask = image_mask_warped)
        output_result_dict["object_edit"]["md"] = all_dist_object_edit
        output_result_dict["object_edit"]["warp"] = warp_loss_object_edit

    if dragon_diffusion_result is not None:
        if "dragon_diffusion" not in output_result_dict:
            output_result_dict["dragon_diffusion"] = {}
        
        warp_loss_dragon_diffusion = get_md_score_from_images(input_image_sq, dragon_diffusion_result, transform_in, depth, input_mask_sq, dift, prompt="", device=device, all_dist = all_dist_dragon_diffusion, t_coords = t_coords, p_image = p_image, p_mask = image_mask_warped)
        output_result_dict["dragon_diffusion"]["md"] = all_dist_dragon_diffusion
        output_result_dict["dragon_diffusion"]["warp"] = warp_loss_dragon_diffusion

    if diffusion_handles_result is not None:

        print("[INFO]: Running Diffusion Handles")
        if "diffusion_handles" not in output_result_dict:
            output_result_dict["diffusion_handles"] = {}
        
        warp_loss_diffusion_handles = get_md_score_from_images(input_image_sq, diffusion_handles_result, transform_in, depth, input_mask_sq, dift, prompt="", device=device, all_dist = all_dist_diffusion_handles, t_coords = t_coords, p_image = p_image, p_mask = image_mask_warped)
        output_result_dict["diffusion_handles"]["md"] = all_dist_diffusion_handles
        output_result_dict["diffusion_handles"]["warp"] = warp_loss_diffusion_handles

    if freedrag_result is not None:
        print("[INFO]: Running FreeDrag")
        if "freedrag" not in output_result_dict:
            output_result_dict["freedrag"] = {}
        
        warp_loss_freedrag = get_md_score_from_images(input_image_sq, freedrag_result, transform_in, depth, input_mask_sq, dift, prompt="", device=device, all_dist = all_dist_freedrag, t_coords = t_coords, p_image = p_image, p_mask = image_mask_warped)
        output_result_dict["freedrag"]["md"] = all_dist_freedrag
        output_result_dict["freedrag"]["warp"] = warp_loss_freedrag

    return output_result_dict

def run_md_evaluation_on_exps(exp_root_folder, dift, evaluation_type):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    folder_list = glob.glob(ui_utils.complete_path(exp_root_folder) + "**/")
    folder_list.sort()

    print(folder_list)

    all_md = {}
    all_warp = {}

    
    if ui_utils.check_if_exp_root(exp_root_folder, folder_list):
        print("[INFO]: Exp root found !")
        root_folders_list = folder_list
    
        for f in tqdm.tqdm(root_folders_list):
            folder_list = glob.glob(ui_utils.complete_path(f) + "**/")
            folder_list.sort()
            exp_cat = f.split("/")[-2]
            if exp_cat == "Mix" or exp_cat == "Removal" or exp_cat == "Scaling" or exp_cat == "Rotation_2D":
                continue
            
            if evaluation_type == "2D":
                if exp_cat != "Translation_2D":
                    continue
            elif evaluation_type == "3D":
                if not (exp_cat == "Translation_3D" or exp_cat == "Rotation_3D"):
                    continue
            
            for exp_folder in tqdm.tqdm(folder_list):
                print(exp_folder)
                output_result_dict = get_md_list_folder(exp_folder, dift=dift, device=device)
                for key in output_result_dict:
                    if output_result_dict[key] is None:
                        continue
                    if key not in all_md:
                        all_md[key] = [] 
                        all_warp[key] = []                   
                    all_md[key].extend(output_result_dict[key]["md"])
                    all_warp[key].append(output_result_dict[key]["warp"])
    
    else:
        for exp_folder in tqdm.tqdm(folder_list):
            output_result_dict = get_md_list_folder(exp_folder, dift=dift, device=device)

            for key in output_result_dict:
                if output_result_dict[key] is None:
                    continue
                if key not in all_md:
                    all_md[key] = []    
                    all_warp[key] = []                   
                all_md[key].extend(output_result_dict[key]["md"])
                all_warp[key].append(output_result_dict[key]["warp"])



    print("[INFO]: Evaluation Type:", args.evaluation_type)

    for key in all_md:
        avg_md = np.mean(all_md[key])
        print(key, " avg. md: ", avg_md)

    for key in all_warp:
        avg_warp = np.mean(all_warp[key])
        print(key, " avg. warp: ", avg_warp)




def run_evaluation_on_exps(exp_root_folder, loss_fn_alex, clip_model, clip_preprocess, evaluation_type = "all"):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    folder_list = glob.glob(ui_utils.complete_path(exp_root_folder) + "**/")
    folder_list.sort()

    # print(folder_list)

    all_lpips = {}
    all_clip_sim = {}
    
    if ui_utils.check_if_exp_root(exp_root_folder, folder_list):
        print("[INFO]: Exp root found !")
        root_folders_list = folder_list
    
        for f in tqdm.tqdm(root_folders_list):
            folder_list = glob.glob(ui_utils.complete_path(f) + "**/")
            folder_list.sort()
            exp_cat = f.split("/")[-2]
            if exp_cat == "Mix" or exp_cat == "Removal" or exp_cat == "Scaling" or exp_cat == "Rotation_2D":
                continue

            if evaluation_type == "2D":
                if exp_cat != "Translation_2D":
                    continue
            elif evaluation_type == "3D":
                if not (exp_cat == "Translation_3D" or exp_cat == "Rotation_3D"):
                    continue

            
            for exp_folder in tqdm.tqdm(folder_list):
                output_result_dict = get_loss_and_sim_score_folder(exp_folder, loss_fn_alex, clip_model, clip_preprocess, device=device)

                for key in output_result_dict:
                    if output_result_dict[key] is None:
                        continue
                    if key not in all_lpips:
                        all_lpips[key] = []
                        all_clip_sim[key] = []
                    all_lpips[key].append(output_result_dict[key][0])
                    all_clip_sim[key].append(output_result_dict[key][1])

    
    else:
        for exp_folder in tqdm.tqdm(folder_list):
            output_result_dict = get_loss_and_sim_score_folder(exp_folder, loss_fn_alex, clip_model, clip_preprocess, device=device)

            for key in output_result_dict:
                if output_result_dict[key] is None:
                    continue
                if key not in all_lpips:
                    all_lpips[key] = []
                    all_clip_sim[key] = []
                all_lpips[key].append(output_result_dict[key][0])
                all_clip_sim[key].append(output_result_dict[key][1])


    print("[INFO]: Evaluation Type:", args.evaluation_type)
    for key in all_lpips:
        avg_lpips = np.mean(all_lpips[key])
        IF = 1.0 - avg_lpips
        avg_clip = np.mean(all_clip_sim[key])
        print(key, " lpips: ", avg_lpips, " IF: ", IF, " avg clip: ", avg_clip)



if __name__ == '__main__':

    seed_everything(42)

    


    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--exp_root',
        required=False, default = None)
    parser.add_argument("--metric", choices=["md", "lpips", "clip"], default="clip")
    parser.add_argument('--path_1',
        required=False)
    parser.add_argument('--path_2',
        required=False)

    parser.add_argument("--evaluation_type", choices=["2D", "3D", "all"], default="all")
    args = parser.parse_args()


    print("[INFO]: Evaluation Type:", args.evaluation_type)

    exp_dict = ui_utils.read_exp(args.exp_root)
    # detect_interest_points(exp_dict["input_image_png"], mask=exp_dict["input_mask_png"][..., 0] / 255.0)
    # get_Matches(exp_dict["input_image_png"], exp_dict["resized_result_ls_png"], mask=exp_dict["input_mask_png"][..., 0] / 255.0)
    # exit()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # lpip metric
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # load clip model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)



    if args.exp_root is not None:
        if args.metric == "lpips" or args.metric == "clip":
            print("[INFO]: Running LPIPS and CLIP Metric")
            run_evaluation_on_exps(args.exp_root, loss_fn_alex, clip_model, clip_preprocess, evaluation_type = args.evaluation_type)
        elif args.metric == "md":
            print("[INFO]: Running MD Metric")
            # load dift model
            dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')
            run_md_evaluation_on_exps(args.exp_root, dift = dift, evaluation_type = args.evaluation_type)
        exit()

    original_img_root = args.path_1


    # for target_root :
    all_lpips = []
    all_clip_sim = []
    for file_name in os.listdir(os.path.join(original_img_root, "")):
        if file_name == '.DS_Store':
            continue
        
        # print(file_name)
        # filename.split("_")

        # exit()
        source_image_path = os.path.join(args.path_1, file_name)
        dragged_image_path = os.path.join(args.path_2, file_name)

        # print(source_image_path, dragged_image_path)
        source_image_PIL = Image.open(source_image_path)
        dragged_image_PIL = Image.open(dragged_image_path)
        # print(source_image_PIL.size)
        # print(dragged_image_PIL.size)
        dragged_image_PIL = dragged_image_PIL.resize(source_image_PIL.size,PIL.Image.BILINEAR)

        source_image = preprocess_image(np.array(source_image_PIL), device)
        dragged_image = preprocess_image(np.array(dragged_image_PIL), device)

        # compute LPIP
        with torch.no_grad():
            source_image_224x224 = F.interpolate(source_image, (224,224), mode='bilinear')
            dragged_image_224x224 = F.interpolate(dragged_image, (224,224), mode='bilinear')
            cur_lpips = loss_fn_alex(source_image_224x224, dragged_image_224x224)
            all_lpips.append(cur_lpips.item())

        # compute CLIP similarity
        source_image_clip = clip_preprocess(source_image_PIL).unsqueeze(0).to(device)
        dragged_image_clip = clip_preprocess(dragged_image_PIL).unsqueeze(0).to(device)

        with torch.no_grad():
            source_feature = clip_model.encode_image(source_image_clip)
            dragged_feature = clip_model.encode_image(dragged_image_clip)
            source_feature /= source_feature.norm(dim=-1, keepdim=True)
            dragged_feature /= dragged_feature.norm(dim=-1, keepdim=True)
            cur_clip_sim = (source_feature * dragged_feature).sum()
            all_clip_sim.append(cur_clip_sim.cpu().numpy())
    # print(target_root)
    avg_lpips = np.mean(all_lpips)
    print('avg lpips: ', avg_lpips)
    print('IF: ', 1.0 - avg_lpips)
    print('avg clip sim', np.mean(all_clip_sim))