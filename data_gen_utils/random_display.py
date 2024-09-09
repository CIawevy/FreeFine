import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import textwrap  # 用于自动换行
import json
import numpy as np
# 加载JSON文件的函数
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


# 随机展示图像的函数
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def random_display_with_prompts(datadict, num_samples=5, save_path=None, title_fontsize=12, subtitle_fontsize=10,
                                wrap_width=30):
    """
    随机从datadict中抽取图像和实例进行可视化，同时显示对应的edit_prompt和原图的img_id。

    Args:
        datadict (dict): 包含图片和元数据的字典
        num_samples (int): 要展示的生成图像的数量
        save_path (str, optional): 如果传入，将可视化结果保存为图像文件的路径
        title_fontsize (int, optional): 原图标题字体大小
        subtitle_fontsize (int, optional): 生成图片的edit_prompt字体大小
        wrap_width (int, optional): 每行显示字符数，用于edit_prompt自动换行
    """
    # 从外层随机抽取一个条目
    random_entry = random.choice(list(datadict.keys()))
    instances = datadict[random_entry]["instances"]

    # img_id 是字典的键
    img_id = random_entry

    # 从实例中随机抽取一个实例
    random_instance = random.choice(list(instances.keys()))
    instance_data = instances[random_instance]

    # 获取原图路径和生成图片路径及对应的edit_prompt
    ori_img_path = instance_data["0"]["ori_img_path"]
    gen_img_data = [(instance_data[str(i)]["gen_img_path"], instance_data[str(i)]["edit_prompt"])
                    for i in range(len(instance_data))]

    # 随机选择 num_samples 张生成图片及其对应的edit_prompt
    sampled_gen_img_data = random.sample(gen_img_data, num_samples)

    # 加载原图和生成图像
    ori_img = mpimg.imread(ori_img_path)
    gen_imgs = [(mpimg.imread(img_path), prompt) for img_path, prompt in sampled_gen_img_data]

    # 创建图像显示网格
    fig, axs = plt.subplots(1, num_samples + 1, figsize=(25, 5))  # 调整为更宽的画布

    # 显示原图，并在标题中添加 img_id
    axs[0].imshow(ori_img)
    axs[0].set_title(f'Original Image (ID: {img_id})', fontsize=title_fontsize)  # 添加 img_id 到标题
    axs[0].axis('off')

    # 显示生成图像和对应的edit_prompt
    for i, (gen_img, prompt) in enumerate(gen_imgs):
        wrapped_prompt = "\n".join(textwrap.wrap(prompt, wrap_width))  # 自动换行
        axs[i + 1].imshow(gen_img)
        axs[i + 1].set_title(wrapped_prompt, fontsize=subtitle_fontsize)  # 使用自动换行后的edit_prompt作为标题
        axs[i + 1].axis('off')

    # 调整布局
    plt.tight_layout()

    # 如果传入了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # 展示结果
    plt.show()


# 加载数据
data = load_json("/data/Hszhu/dataset/PIE-Bench_v1/generated_dataset_full_pack.json")

# 如果数据成功加载，调用随机展示函数

random_display_with_prompts(data, num_samples=5, save_path="/data/Hszhu/dataset/demo/random_sample_output.png", title_fontsize=14, subtitle_fontsize=10, wrap_width=30)
