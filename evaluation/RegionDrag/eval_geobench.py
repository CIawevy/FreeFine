import os
import argparse
import torch
from tqdm import tqdm
import gradio as gr
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm

from region_utils.drag import drag, get_drag_data, get_meta_data
from region_utils.ui_utils import region_pair_to_pts
from region_utils.evaluator import DragEvaluator
import time


class GeoData(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(f"{data_dir}/annotations.json", "r") as f:
            data = json.load(f)
        
        self.data = data
        img_idx = list(data.keys())
        self.instance_idx = []

        for i in range(len(img_idx)):
            img = data[img_idx[i]]
            instances = img["instances"]
            for key in instances.keys():
                for sub_key in instances[key].keys():
                    self.instance_idx.append([img_idx[i],key,sub_key])
    
    def __len__(self):
        return len(self.instance_idx)
    
    def __getitem__(self, idx):
        inst = self.instance_idx[idx]
        inst = self.data[inst[0]]["instances"][inst[1]][inst[2]]
        orig_img_path = inst["ori_img_path"]
        orig_img = np.array(Image.open(orig_img_path).convert('RGB'))
        orig_mask_path = inst["ori_mask_path"]
        tgt_mask_path = inst["tgt_mask_path"]
        tgt_mask = Image.open(tgt_mask_path).convert("L")
        orig_mask = Image.open(orig_mask_path).convert("L").resize(tgt_mask.size,Image.LANCZOS)
        mask = ImageChops.lighter(orig_mask, tgt_mask)
        tgt_mask = ((np.array(tgt_mask) >127)*255).astype(np.uint8)
        orig_mask = ((np.array(orig_mask) >127)*255).astype(np.uint8)
        mask = (np.array(mask) >127)*255
        # mask = np.empty((orig_mask.size + tgt_mask.size,), dtype=orig_mask.dtype)
        # mask[0:-1:2] = orig_mask
        # mask[1::2] = tgt_mask
        edit_param = inst["edit_param"]
        base_path = "/work/nvme/bcgq/yimingg8/Geo-Bench-3D'/target_mask"
        store_dir = os.path.relpath(inst["tgt_mask_path"], base_path)
        store_dir = f"result_SC/{store_dir}"
        return orig_img,  f"image of {inst['obj_label']}",  tgt_mask, orig_mask, tgt_mask, edit_param, store_dir

# class transform(source, edit_param):
#     dx,dy,dz = edit_param[0], edit_param[1], edit_param[2]
#     rx, ry, rz = edit_param[3], edit_param[4], edit_param[5]
#     sx, sy, sz = edit_param[6], edit_param[7], edit_param[8]

#     coords = (source == 1).nonzero(as_tuple=False)
#     coords[:,0] += dy
#     coords[:,0] += dx
#     dim = source.shape[-1]
#     coords = coords.clamp(min=0,max=dim)

    


# Setting up the argument parser
parser = argparse.ArgumentParser(description='Run the drag operation.')
parser.add_argument('--data_dir', type=str, default='/work/nvme/bcgq/yimingg8/Geo-Bench-3D') # OR 'drag_data/dragbench-sr/'
args = parser.parse_args()


start_t = 0.5
end_t = 0.2
steps = 50
noise_scale = 1.0
seed = 42

dataset = GeoData(args.data_dir)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

time_total = 0
time_avg = 0
count = 0

for i, (ori_img, prompt, mask, source, target, edit_param, store_dir) in enumerate(tqdm(loader)):
    # import pdb
    # pdb.set_trace()
    # Image.fromarray(source.squeeze().numpy()).convert("RGB").save("mask.png")
    # Image.fromarray(target.squeeze().numpy()).convert("RGB").save("mask_tgt.png")
    # target = transform(source, edit_param)

    start = time.time()
    source, target = region_pair_to_pts(source.squeeze().numpy(), target.squeeze().numpy(), scale=1/8)


    

    mask = torch.ones_like(mask)
    # viz_ori = Image.fromarray(ori_img.squeeze().numpy())
    # viz_ori.save(f"ori_{i}.png")

    drag_data = {
        "ori_img":ori_img,
        "preview":ori_img,
        "prompt":prompt,
        "mask": mask,
        "source": source*8,
        "target": target*8
    }

    out_image = drag(drag_data, steps, start_t, end_t, noise_scale, seed, progress=gr.Progress())
    end = time.time()
    time_total += end - start
    count += 1
    time_avg = time_total / count
    print(f"Time taken: {time_avg} seconds")
    viz_out = Image.fromarray(out_image)
    # viz_out.save(f"out_{i}.png")
    
    # os.makedirs(os.path.dirname(store_dir[0]), exist_ok=True)
   
    # viz_out.save(store_dir[0])





# for data_path in tqdm(data_dirs):
#     # Region-based Inputs for Editing
#     drag_data = get_drag_data(data_path)
#     ori_image = drag_data['ori_image']
#     out_image = drag(drag_data, steps, start_t, end_t, noise_scale, seed, progress=gr.Progress())

#     # Point-based Inputs for Evaluation
#     meta_data_path = os.path.join(data_path, 'meta_data.pkl')
#     prompt, _, source, target = get_meta_data(meta_data_path)    

#     all_distances.append(evaluator.compute_distance(ori_image, out_image, source, target, method='sd', prompt=prompt))
#     all_lpips.append(evaluator.compute_lpips(ori_image, out_image))

