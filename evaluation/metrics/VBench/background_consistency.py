import clip
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def parse_data(data, image_label):
    data_pair = []
    for image in data.values():
        instances = image["instances"]
        for instance in instances.values():
            for sample in instance.values():
                data_pair.append((sample["ori_img_path"], sample[image_label], sample["ori_mask_path"], sample["tgt_mask_path"]))
    return data_pair

def compute_background_consistency(inputs_path: list, clip_model, preprocess, device):
    inputs = [Image.open(i) for i in inputs_path]
    mask1 = np.array(inputs[2].resize(inputs[0].size))
    mask2 = np.array(inputs[3].resize(inputs[0].size))
    mask = mask1 + mask2
    mask_bool = (mask < 128).astype(np.uint8)
    # Image.fromarray(mask).save("mask.png")
    # exit()
    for i in range(2):
        inputs[i] = Image.fromarray(np.array(inputs[i]) * mask_bool[..., np.newaxis])
    images = (preprocess(inputs[0]), preprocess(inputs[1]))
    image_features = []
    for image in images:
        image = image.unsqueeze(0)
        image = image.to(device)
        image_feature = clip_model.encode_image(image)
        image_feature = F.normalize(image_feature, dim=-1, p=2)
        image_features.append(image_feature)
    return max(0.0, F.cosine_similarity(image_features[0], image_features[1]).item())

def calculate_bgc(data, image_label):
    print("-----Background Consistency-----")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load('ViT-B/32', device=device)

    data_pair = parse_data(data, image_label)
    bgc = 0.0
    for inputs in tqdm(data_pair):
        bgc += compute_background_consistency(inputs, clip_model, preprocess, device)
    bgc = bgc/len(data_pair)
    print(f"BGC: {bgc}")
    return bgc

