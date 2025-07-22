import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from .dift_sd import SDFeaturizer
from scipy.ndimage import center_of_mass


def parse_data(data, image_label):
    data_pair = []
    for image in data.values():
        instances = image["instances"]
        for instance in instances.values():
            for sample in instance.values():
                data_pair.append((sample["ori_img_path"], sample[image_label], sample["ori_mask_path"], sample["edit_param"], sample['obj_label']))
    return data_pair

def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

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

def get_Matches(im1, im2, mask):
    im2 = np.array(Image.fromarray(im2).resize(Image.fromarray(im1).size, Image.BILINEAR))

    #this finds keypoints and descriptors with SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    
    try:
        bf = cv2.BFMatcher() #create a bfMatcher object
        matches = bf.knnMatch(des1,des2, k=2) #Match descriptors

        matches_mask = []
        interest_points = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                k_1 = kp1[m.queryIdx].pt
                k_2 = kp2[m.trainIdx].pt

                if mask[int(k_1[1]), int(k_1[0])] > 0.5:
                    matches_mask.append(m)
                    interest_points.append([int(k_1[1]), int(k_1[0])])
    except (cv2.error, ValueError) as e:
        interest_points = []

    if len(interest_points) == 0:
        interest_points = detect_interest_points(im1, mask)
    else:
        interest_points = np.array(interest_points)

    return interest_points

def get_transform_coordinates(edit_param, size, mask, path_3D):
    if edit_param[0]!=0 or edit_param[1]!=0:
        assert all(x == 0 for x in edit_param[2:9]) == 0
        points = np.zeros((size[0], size[1], 2))
        for i in range(size[0]):
            for j in range(size[1]):
                points[i, j, 0] = i + edit_param[1]
                points[i, j, 1] = j + edit_param[0]
        return points
    elif edit_param[5]!=0 or edit_param[6]!=1:
        center = center_of_mass(mask)
        if edit_param[5]!=0:
            assert all(x == 0 for x in edit_param[6:9]) == 0
            matrix = cv2.getRotationMatrix2D(center, edit_param[5], scale=1.0)
        elif edit_param[6]!=1:
            assert edit_param[6] == edit_param[7]
            scale = edit_param[6]
            matrix = np.array([
                [scale, 0, (1 - scale) * center[0]],
                [0, scale, (1 - scale) * center[1]]
            ])
        y, x = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
        ones = np.ones_like(x)
        points = np.stack((x, y, ones), axis=-1).reshape(-1, 3)
        rotated_point = np.dot(points, matrix.T).reshape(size[0], size[1], 2)
        return rotated_point
    else:
        return np.load(path_3D)[..., ::-1].copy()
    

def calculate_md(data, image_label):
    print("-----MD-----")
    data_pairs = parse_data(data, image_label)
    dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')
    all_dist = []
    max_points = 30


    for data_pair in tqdm(data_pairs):
        s_img_path, t_img, s_mask, edit_param, prompt = data_pair
        model_name = t_img.split("/")[-4]
        path_3D = t_img.replace("zkl", "Hszhu").replace(model_name, "correspondence").replace(".png", ".npy")
        s_img = np.array(Image.open(s_img_path))
        t_img= Image.open(t_img).resize(s_img.shape[:-1], Image.BILINEAR)
        t_img = np.array(t_img)
        s_mask = Image.open(s_mask).resize(s_img.shape[:-1], Image.BILINEAR)
        s_mask = np.array(s_mask) / 255.0

        try:
            kps = get_Matches(s_img, t_img, s_mask)
        except ValueError as e:
            print(f"Error in get_Matches: {s_img_path}")
            continue

        source_image_tensor = preprocess_image(s_img, "cuda")
        edited_image_tensor = preprocess_image(t_img, "cuda")
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
        t_coords_pixels = get_transform_coordinates(edit_param, s_img.shape[:-1], s_mask, path_3D)
        cos = torch.nn.CosineSimilarity(dim=1)
        for k in kps:
            num_channel = ft_source.size(1)
            src_vec = ft_source[0, :, k[0], k[1]].view(1, num_channel, 1, 1)
            cos_map = cos(src_vec, ft_edited).cpu().numpy()[0]  # H, W
            max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape) # the matched row,col
            tp = t_coords_pixels[k[0], k[1]]
            
            # calculate distance
            tp = torch.tensor(tp)
            dist = (tp - torch.tensor(max_rc)).float().norm()
            all_dist.append(dist)

    md = torch.tensor(all_dist).mean().item()
    print(f"MD: {md}")
    return md
