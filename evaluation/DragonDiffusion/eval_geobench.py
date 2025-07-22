import os, random, json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/zkl/huggingface/"

import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import center_of_mass
from src.demo.model import DragonModels
from pytorch_lightning import seed_everything


def test_single(model, original_image_path, mask_path, prompt, edit_param):
    """
    original_image_path = "/data/Hszhu/dataset/Geo-Bench/source_img/1.png"
    mask_path = "/data/Hszhu/dataset/Geo-Bench/source_mask/1/0.png"
    prompt = ""
    edit_param = [0.0, -40.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    """
    w_edit = 4
    w_content = 6
    w_contrast = 0.2
    w_inpaint = 0.8
    seed = random.randint(0, 10000)
    guidance_scale = 4
    energy_scale = 0.5
    max_resolution = 512
    SDE_strength = 0.4
    ip_scale = 0.1

    rz = edit_param[5]
    original_image = np.array(Image.open(original_image_path))
    mask = np.array(Image.open(mask_path).resize(original_image.shape[:2]))
    mask_ref = np.zeros_like(mask)
    mask = np.expand_dims(mask, -1).repeat(3, axis=2)
    center = center_of_mass(mask)
    center = [int(round(center[0])), int(round(center[1]))]
    target = [int(center[0]+edit_param[0]), int(center[1]+edit_param[1])]
    selected_points = [center, target]
    assert edit_param[6] == edit_param[7]
    resize_scale = edit_param[6]

    result = model.run_move(original_image, mask, mask_ref, prompt, rz, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale)[0]
    image = Image.fromarray(result)
    return image

def split_data():
    json_path = "/data/Hszhu/dataset/Geo-Bench/annotations.json"
    data = json.load(open(json_path))
    data_s = [dict(), dict(), dict(), dict()]
    i = 0
    for k in  data.keys():
        data_s[i][k] = data[k]
        i += 1
        i %= 4
    for i in range(4):
        json.dump(data_s[i], open(f"{i}.json", "w"))

def main(file):
    seed_everything(42)
    pretrained_model_path = "runwayml/stable-diffusion-v1-5"
    model = DragonModels(pretrained_model_path=pretrained_model_path)

    data = json.load(open(file))

    start_time = time.time()
    num = 0
    for k in list(data.keys()):
        image = data[k]
        prompt = image["4v_caption"]
        instances = image["instances"]
        for instance_key in instances.keys():
            instance = instances[instance_key]
            for sample_key in instance.keys():
                sample = instance[sample_key]
                edit_param = sample["edit_param"]

                # assert edit_param[2] == 0
                # assert edit_param[3] == 0
                # assert edit_param[4] == 0
                # assert edit_param[8] == 1.0
                # save_path = sample['coarse_input_path'].replace("coarse_img", "dragon").replace("Hszhu", "zkl")
                # if os.path.exists(save_path):
                #     data[k]["instances"][instance_key][sample_key]["gen_img_path"] = save_path
                #     continue
                result = test_single(model, sample["ori_img_path"], sample["ori_mask_path"], prompt, edit_param)
                num += 1
                print(num)
                if num == 50:
                    end_time = time.time()
                    print(f"运行时间: {end_time - start_time:.2f} 秒")
                    exit()

                # dirname = os.path.dirname(save_path)
                # if not os.path.exists(dirname):
                #     os.makedirs(dirname)
                # result.save(save_path)
                # data[k]["instances"][instance_key][sample_key]["gen_img_path"] = save_path
    # json.dump(data, open("dragon_sc.json", "w"))
    

def check():
    data = json.load(open('0.json'))
    for k in tqdm(data.keys()):
        image = data[k]
        instances = image["instances"]
        for instance_key in instances.keys():
            instance = instances[instance_key]
            for sample_key in instance.keys():
                sample = instance[sample_key]
                edit_param = sample["edit_param"]
                assert edit_param[2] == 0
                assert edit_param[3] == 0
                assert edit_param[4] == 0
                assert edit_param[8] == 1.0
                save_path = sample['coarse_input_path'].replace("coarse_img", "dragon").replace("Hszhu", "zkl")
                if os.path.exists(save_path):
                    data[k]["instances"][instance_key][sample_key]["gen_img_path"] = save_path
                else:
                    print(save_path)


if __name__ == "__main__":
    main("/data/Hszhu/dataset/Geo-Bench/annotations.json")
