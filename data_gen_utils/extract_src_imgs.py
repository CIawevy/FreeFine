from tqdm import  tqdm
import json
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import cv2



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

def temp_view_img(image, title: str = None) -> None:
    # PIL -> ndarray OR ndarray->PIL->ndarray
    if not isinstance(image, Image.Image):  # ndarray
        # image_array = Image.fromarray(image).convert('RGB')
        image_array = image
    else:  # PIL
        if image.mode != 'RGB':
            image.convert('RGB')
        image_array = np.array(image)

    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()
def read_img(image_path):
    img = cv2.imread(image_path)  # bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def extract(i, dst_base):
    file_name = os.path.join(dst_base, f"Subset_{i}/sgpt4v_captions.json")
    # file_name = os.path.join(dst_base, f"Subset_{i}/sgpt4v_captions.json")
    data = load_json(file_name )
    for da_n, da in tqdm(data.items()):
        if 'instances' not in da:
            print(f'skip {da_n} for empty case')
            continue
        src_img_path = da["src_img_path"]


        tgt_img_path = src_img_path.replace('srcs/', 'src_imgs/')
        os.makedirs(os.path.dirname(tgt_img_path), exist_ok=True)
        # 确保目标路径存在
        # 复制文件
        if src_img_path != tgt_img_path:
            shutil.copy(src_img_path,tgt_img_path)
            print(tgt_img_path)
        #更新数据中的路径
        data[da_n].update({"src_img_path": tgt_img_path})


    # 保存更新后的数据
    save_json(data, file_name )
# dst_base = "/data/Hszhu/dataset/SA-1B/"
dst_base = "/data/Hszhu/dataset/GRIT/"
for i in tqdm(range(4)):
    extract(i,dst_base)
print('finish')
