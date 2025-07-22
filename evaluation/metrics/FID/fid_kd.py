import os, random
import torch
from .mmd import compute_mmd
from .fid_score import get_activations, IMAGE_EXTENSIONS


def parse_data(data, image_label, real_root_path):
    real_data = []
    gen_data = []
    for image in data.values():
        instances = image["instances"]
        for instance in instances.values():
            for sample in instance.values():
                real_data.append(sample["ori_img_path"])
                gen_data.append(sample[image_label])
    real_data = os.listdir(real_root_path)
    real_data = [os.path.join(real_root_path, i) for i in real_data]
    return real_data, gen_data


def calculate_fid_kd(data, image_label, real_root_path):
    print("-----FID_KD-----")
    dims=768
    device="cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    files = parse_data(data, image_label, real_root_path)

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)

    pre1 = get_activations(files[0], model, batch_size, dims, device, num_workers)
    pre2 = get_activations(files[1], model, batch_size, dims, device, num_workers)
    fid_value = compute_mmd(pre1, pre2).mean()

    print(f"FID_KD: {fid_value}")
    return fid_value
