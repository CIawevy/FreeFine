import os, random
import torch
from .fid_score import calculate_activation_statistics, calculate_frechet_distance


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
    # print(len(real_data))
    # real_data = random.sample(real_data, k=2000)
    return real_data, gen_data


def calculate_fid_dino(data, image_label, real_root_path):
    print("-----FID_DINO-----")
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
    m1, s1 = calculate_activation_statistics(files[0], model, batch_size,
                                                   dims, device, num_workers)
    m2, s2 = calculate_activation_statistics(files[1], model, batch_size,
                                                   dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"FID_DINO: {fid_value}")
    return fid_value
