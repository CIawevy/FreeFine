
import json
import os
from PIL import Image

annotations_path = "/work/nvme/bcgq/yimingg8/geobench/annotations.json"
with open(annotations_path, "r") as f:
    data = json.load(f)

cache = []
cache_key = []
for image_idx in data.keys():
    edit_indices = data[image_idx]["instances"]
    reindex_path = f"/work/nvme/bcgq/yimingg8/dragdiffuser_2d/{image_idx}"
    os.makedirs(reindex_path, exist_ok=True)
    for edit_idx in edit_indices.keys():
        for sub_edit_idx in edit_indices[edit_idx].keys():
            path = None
            for potential_path in ("dragdiffuser_output_2D", "dragdiffuser_output_2D_2", "dragdiffuser_output_2D_3", "dragdiffuser_output_2D_4", "dragdiffuser_output_2D_5"):
                path = edit_indices[edit_idx][sub_edit_idx]["coarse_input_path"].replace("geobench", potential_path).replace("/coarse_img","")
                if os.path.exists(path):
                    break
            if path is None:
                print(f"path is None for {image_idx} {edit_idx} {sub_edit_idx}")
            else:
                img = Image.open(path)
                save_path = f"{reindex_path}/{path.split('/')[-2]}/{path.split('/')[-1]}"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)

        



