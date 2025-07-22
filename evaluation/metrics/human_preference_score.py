import torch
import huggingface_hub
from PIL import Image
from tqdm import tqdm
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


def initialize_model():
    device="cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    return model, preprocess_val, tokenizer

def score(img_path, prompt, model, preprocess_val, tokenizer):
    device="cuda" if torch.cuda.is_available() else "cpu"
    result = []
    for one_img_path in img_path:
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            if isinstance(one_img_path, str):
                image = preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=device, non_blocking=True)
            elif isinstance(one_img_path, Image.Image):
                image = preprocess_val(one_img_path).unsqueeze(0).to(device=device, non_blocking=True)
            else:
                raise TypeError('The type of parameter img_path is illegal.')
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.amp.autocast("cuda"):
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        result.append(hps_score[0])
    return result

def calculate_hps(data, image_label):
    print("-----Human preference Score-----")
    model, preprocess_val, tokenizer = initialize_model()
    hps_score = 0.0
    number = 0
    for image in tqdm(data.values()):
        instances = image["instances"]
        prompt = image["4v_caption"]
        images = list()
        for instance in instances.values():
            for sample in instance.values():
                images.append(sample[image_label])
                number += 1
        if images:
            results = score(images, prompt, model, preprocess_val, tokenizer)
            hps_score += sum(results)
    hps_score = hps_score / number
    print(f"HPS: {hps_score}")
    return hps_score
