import ImageReward as RM
from tqdm import tqdm
import json


def calculate_irs(data, image_label, save = False):
    print("-----ImageReward Score-----")
    model = RM.load("ImageReward-v1.0")
    ir_score = 0.0
    number = 0
    save_data = dict()
    for image in tqdm(data.values()):
        instances = image["instances"]
        prompt = image["4v_caption"]
        images = list()
        for instance in instances.values():
            for sample in instance.values():
                images.append(sample[image_label])
                number += 1
        if images:
            results = model.score(prompt, images)
            if type(results) == list:
                ir_score += sum(results)
                if save:
                    for i in range(len(results)):
                        save_data[images[i]] = results[i]
            else:
                ir_score += results
                if save:
                    save_data[images[0]] = results
    ir_score = ir_score / number
    print(f"IRS: {ir_score}")
    if save:
        json.dump(save_data, open("IRS.json", "w"))
    return ir_score

if __name__ == "__main__":
    data = json.load(open("/data/Hszhu/dataset/Geo-Bench/generated_results.json"))
    calculate_irs(data, "gen_img_path", save = True)

