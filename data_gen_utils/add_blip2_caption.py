import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import os.path as osp
from tqdm import  tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image

class BLIP2():
    def __init__(self, ckpt_path):
        processor = AutoProcessor.from_pretrained(ckpt_path)
        model = Blip2ForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self.device)
        self.processor = processor
        self.model = model
    def __call__(self,image_path, *args, **kwargs):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

def save_json(data_dict, file_path):
    """
    将字典保存为 JSON 文件

    Args:
        data_dict (dict): 需要保存的字典
        file_path (str): JSON 文件的保存路径
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)


def load_json(file_path):
    """
    加载指定路径的JSON文件并返回数据。

    :param file_path: JSON文件的路径
    :return: 从JSON文件中加载的数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        print(f"文件格式错误: {file_path}")
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
    return None
blip2 = BLIP2("/data/Hszhu/prompt-to-prompt/blip2-opt-2.7b/")
def add_caption(i,dst_base):
    src_source = osp.join(dst_base, f"Subset_{i}/mask_tag_relabelled_lmm_v2_{i}.json")
    tgt_path = osp.join(dst_base, f"Subset_{i}/final_filtered_{i}.json")
    # tgt_path = osp.join(dst_base, f"Subset_{i}/mat_fooocus_inpainting_{i}.json")
    src_data = load_json(src_source)
    meta = load_json(tgt_path)
    new_meta = {}
    for da_n,da in tqdm(meta.items()):
        print(da_n)
        if 'instances' not in da:
            # del meta[da_n]
            continue
        new_meta[da_n] = da
        blip_caption = blip2(src_data[da_n]['src_img_path'] )
        # new_meta[da_n].update({"src_img_path": src_data[da_n]["src_img_path"]})
        new_meta[da_n].update({"blip2_caption": blip_caption})
    save_json(new_meta,tgt_path)
dst_base = "/data/Hszhu/dataset/PIE-Bench_v1"
for i in tqdm(range(4)):
    add_caption(i,dst_base)
print('finish')