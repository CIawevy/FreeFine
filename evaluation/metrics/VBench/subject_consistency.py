import torch
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from .background_consistency import parse_data
from torchvision.transforms import ToTensor, Compose, Resize, Normalize


def compute_subject_consistency(inputs_path: list, dino_model, device):
    image_transform = Compose([
        Resize(size=224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    inputs = [Image.open(i) for i in inputs_path]
    for i in range(2):
        mask = inputs[i + 2].resize(inputs[i].size)
        mask_bool = (np.array(mask) > 128).astype(np.uint8)
        inputs[i] = Image.fromarray(np.array(inputs[i]) * mask_bool[..., np.newaxis])
    images = (image_transform(inputs[0]), image_transform(inputs[1]))
    image_features = []
    for image in images:
        image = image.unsqueeze(0)
        image = image.to(device)
        image_feature = dino_model(image)
        image_feature = F.normalize(image_feature, dim=-1, p=2)
        image_features.append(image_feature)
    return max(0.0, F.cosine_similarity(image_features[0], image_features[1]).item())

def calculate_subc(data, image_label):
    print("-----Subject Consistency-----")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device)

    data_pair = parse_data(data, image_label)
    subc = 0.0
    for inputs in tqdm(data_pair):
        subc += compute_subject_consistency(inputs, dino_model, device)
    subc = subc/len(data_pair)
    print(f"SUBC: {subc}")
    return subc
