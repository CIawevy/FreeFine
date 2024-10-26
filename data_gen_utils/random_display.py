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




def temp_view(mask, title='Mask', name=None):
    """
    显示输入的mask图像

    参数:
    mask (torch.Tensor): 要显示的mask图像，类型应为torch.bool或torch.float32
    title (str): 图像标题
    """
    # 确保输入的mask是float类型以便于显示
    if isinstance(mask, np.ndarray):
        mask_new = mask
    else:
        mask_new = mask.float()
        mask_new = mask_new.detach().cpu()
        mask_new = mask_new.numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(mask_new, cmap='gray')
    plt.title(title)
    plt.axis('off')  # 去掉坐标轴
    # plt.savefig(name+'.png')
    plt.show()
import cv2
def display_all_samples(datadict, save_path=None, title_fontsize=14, subtitle_fontsize=10, wrap_width=30,
                        specific_key=None,id=None):
    """
    展示生成的图片，显示edit_prompt、原图img_id和original mask。
    """
    import random
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import textwrap
    from concurrent.futures import ThreadPoolExecutor

    # 选择展示的实例
    img_id = specific_key or random.choice(list(datadict.keys()))
    instances = datadict[img_id]["instances"]
    if not instances:
        print(f'no instance')
        return

    # 随机选择一个实例
    if id is None:
        id = random.choice(list(instances.keys()))
    instance_data = instances[id]
    ori_img_path = instance_data[id]["ori_img_path"]
    ori_mask_path = instance_data[id].get("ori_mask_path")  # 获取 original mask 路径
    gen_img_data = [(v["gen_img_path"], v["edit_prompt"]) for v in instance_data.values()]

    # 图片展示设置
    total_samples = len(gen_img_data)
    total_items = total_samples + 1  # 原图+生成图
    if ori_mask_path:
        total_items += 1  # 加上 original mask

    images_per_row = 6
    rows = (total_items + images_per_row - 1) // images_per_row  # 动态计算行数

    # 使用线程池并行加载图片
    def load_image(image_path):
        return mpimg.imread(image_path)

    image_paths = [ori_img_path] + [img_path for img_path, _ in gen_img_data]
    if ori_mask_path:
        image_paths.append(ori_mask_path)

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, image_paths))

    # 开始绘制图像
    fig, axs = plt.subplots(rows, images_per_row, figsize=(25, rows * 5))
    axs = axs.flatten()

    # 显示原图
    axs[0].imshow(images[0])
    axs[0].set_title(f'Original Image (ID: {img_id})', fontsize=title_fontsize)
    axs[0].axis('off')

    # 显示生成图像
    for i, (img, (img_path, prompt)) in enumerate(zip(images[1:1 + total_samples], gen_img_data)):
        axs[i + 1].imshow(img)
        axs[i + 1].set_title("\n".join(textwrap.wrap(prompt, wrap_width)), fontsize=subtitle_fontsize)
        axs[i + 1].axis('off')

    # 显示 original mask，如果有的话
    if ori_mask_path:
        axs[total_samples + 1].imshow(images[-1])  # Mask 是最后一张图
        axs[total_samples + 1].set_title("Original Mask", fontsize=title_fontsize)
        axs[total_samples + 1].axis('off')

    # 隐藏多余的子图
    for j in range(total_items, len(axs)):
        axs[j].axis('off')

    # 布局调整与保存
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


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
import os.path as osp
#ablation study on unseen generation
# base_path = "/data/Hszhu/dataset/PIE-Bench_v1/Gen_results/"
# base_path = "/data/Hszhu/dataset/PIE-Bench_v1/Gen_results_local_ddpm/"
# base_path = "/data/Hszhu/dataset/PIE-Bench_v1/Gen_results_local_text_only/"
# base_path = "/data/Hszhu/dataset/PIE-Bench_v1/Gen_results_mtsa_only/"

# base_path = "/data/Hszhu/dataset/PIE-Bench_v1/Subset_lama_sd"
base_path = "/data/Hszhu/dataset/exp/"
# 加载数据
data = load_json(osp.join(base_path,'generated_dataset_full_pack_exp.json'))
# data = load_json(osp.join(base_path,'generated_dataset_full_pack_unseen.json'))
# keys=['114', '654', '250', '692', '142', '228', '104', '25', '281', '754']
# 如果数据成功加载，调用随机展示函数
keys_list = list(data.keys())
print(keys_list)
# random_display_with_prompts(data, num_samples=5, save_path="/data/Hszhu/dataset/PIE-Bench_v1/Subset_lama/random_sample_output.png", title_fontsize=14, subtitle_fontsize=10, wrap_width=30)
display_all_samples(data,specific_key='0',id='1', save_path=osp.join(base_path,"random_sample_output.png"))
