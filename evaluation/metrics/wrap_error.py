from tqdm import tqdm
from PIL import Image
import numpy as np

def  calculate_we(data, image_label):
    print("-----Wrap Error-----")
    wrap_e = 0.0
    num = 0
    for image in tqdm(data.values()):
        instances = image["instances"]
        for instance in instances.values():
            for sample in instance.values():
                image_paths = [sample["coarse_input_path"], sample[image_label], sample["tgt_mask_path"]]
                images = [np.array(Image.open(image_path))/255 for image_path in image_paths]
                mask = np.repeat(images[2][..., np.newaxis], 3, axis=2)
                wrap_e += np.sum(np.abs(images[0]*mask - images[1]*mask)) / mask.sum()
                # wrap_e += np.average(np.abs(images[0]*mask - images[1]*mask))
                num += 1
    wrap_e = wrap_e/num
    print(f"WRAP_E: {wrap_e}")
    return wrap_e
