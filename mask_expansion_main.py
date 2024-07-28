from src.demo.download import download_all
# download_all()
from diffusers import  UNet2DConditionModel
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from simple_lama_inpainting import SimpleLama
from src.demo.demo import create_my_demo
from src.demo.model import ClawerModels
from src.unet.unet_2d_condition import DragonUNet2DConditionModel
import torch
import cv2
import gradio as gr
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDIMPipeline,StableDiffusionInpaintPipeline
from pathlib import Path
from src.utils.attention import AttentionStore,register_attention_control,Mask_Expansion_SELF_ATTN
import numpy as np
import matplotlib.pyplot as plt
# main demo
# pretrained_model_path = "runwayml/stable-diffusion-v1-5"
import torch.nn.functional as F




def get_image_paths_from_dir(directory):
    """
    读取指定目录下的所有图片文件，形成一个列表。

    参数:
    directory (str): 图片所在的目录路径。

    返回:
    List[str]: 图片文件路径的列表。
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))

    return image_paths
def get_images_and_masks(image_dir, mask_dir):
    """
    读取指定目录下的所有图片和掩码文件，返回包含图像和掩码的字典。

    参数:
    image_dir (str): 图像文件所在的目录。
    mask_dir (str): 掩码文件所在的目录。

    返回:
    Dict[str, Dict[str, np.ndarray]]: 包含图像及其掩码的字典。
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_paths = {}
    mask_paths = {}

    # 遍历图像目录并读取图像
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                base_name = os.path.splitext(file)[0]
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is not None:
                    image_paths[base_name] = image
                else:
                    print(f"Error loading image: {image_path}")

    # 遍历掩码目录并读取掩码
    for root, _, files in os.walk(mask_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                base_name = os.path.splitext(file)[0]
                mask_path = os.path.join(root, file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_paths[base_name] = mask
                else:
                    print(f"Error loading mask: {mask_path}")

    # 断言图像和掩码的基名一致
    assert set(image_paths.keys()) == set(mask_paths.keys()), "Mismatch between image and mask files"

    # 创建包含图像和掩码的字典
    input_data = {k: {"image": v, "mask": mask_paths[k]} for k, v in image_paths.items()}

    return input_data
import cv2
import numpy as np
import matplotlib.pyplot as plt

def opening_operation(mask, kernel_size=5):
    # 创建一个正方形的结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 开操作
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opening

def closing_operation(mask, kernel_size=5):
    # 创建一个正方形的结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 闭操作
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closing

def save_and_visualize_masks(img, expand_mask, original_mask, dst_path, k, name=None):
    """
    保存二值化后的扩展掩码，并将掩码可视化在输入图像上
    :param img: 输入的图像，与掩码大小一致
    :param expand_mask: 输入的扩展掩码张量
    :param original_mask: 输入的原始掩码张量
    :param mask_threshold: 二值化的阈值
    :param dst_path: 保存路径
    :param k: 文件名中的标识符
    """

    # 将 PyTorch 张量转换为 NumPy 数组
    binary_expand_mask_np = expand_mask.detach().cpu().numpy().astype(np.uint8)
    original_mask_np = original_mask.detach().cpu().numpy().astype(np.uint8)
    img = np.array(img)

    # 保存二值化的扩展掩码到目标路径
    # save_path = os.path.join(dst_path, f"{k}_expanded.png")
    # cv2.imwrite(save_path, binary_expand_mask_np)
    # print(f"Saved expanded mask to {save_path}")

    # 将掩码调整为与图像相同的通道数
    if len(img.shape) == 3 and img.shape[2] == 3:  # 彩色图像
        if len(original_mask_np.shape) == 2:  # 单通道掩码
            original_mask_np = cv2.cvtColor(original_mask_np, cv2.COLOR_GRAY2BGR)
        if len(binary_expand_mask_np.shape) == 2:  # 单通道掩码
            binary_expand_mask_np = cv2.cvtColor(binary_expand_mask_np, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 2:  # 灰度图像
        if len(original_mask_np.shape) == 3 and original_mask_np.shape[2] == 3:
            original_mask_np = cv2.cvtColor(original_mask_np, cv2.COLOR_BGR2GRAY)
        if len(binary_expand_mask_np.shape) == 3 and binary_expand_mask_np.shape[2] == 3:
            binary_expand_mask_np = cv2.cvtColor(binary_expand_mask_np, cv2.COLOR_BGR2GRAY)

    # 可视化扩展掩码和原始掩码在输入图像上
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img_with_original_mask = cv2.addWeighted(img, 0.5, original_mask_np, 0.5, 0)
    img_with_expanded_mask = cv2.addWeighted(img, 0.5, binary_expand_mask_np, 0.5, 0)

    # 在这里进行通道顺序调整，确保合并的时候通道顺序正确
    img_with_original_mask = cv2.cvtColor(img_with_original_mask, cv2.COLOR_BGR2RGB)
    img_with_expanded_mask = cv2.cvtColor(img_with_expanded_mask, cv2.COLOR_BGR2RGB)

    axes[0].imshow(img_with_original_mask)
    axes[0].set_title('Original Mask on Image')
    axes[0].axis('off')

    axes[1].imshow(img_with_expanded_mask)
    axes[1].set_title('Expanded Mask on Image')
    axes[1].axis('off')

    # 保存可视化结果
    if name is not None:
        visualization_path = os.path.join(dst_path, f"{k}_vis_{name}.png")
    else:
        visualization_path = os.path.join(dst_path, f"{k}_visualization.png")
    plt.savefig(visualization_path)
    print(f"Saved visualization to {visualization_path}")
    plt.show()
    plt.close(fig)




if __name__ == "__main__":
    #Load model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # pretrained_inpaint_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-inpainting/"
    # sd_inpainter = StableDiffusionInpaintPipeline.from_pretrained(
    #     pretrained_inpaint_model_path,
    #     revision="fp16",
    #     torch_dtype=torch.float16,
    # ).to(device)
    pretrained_model_path = "/data/Hszhu/prompt-to-prompt/stable-diffusion-v1-5/"
    precision = torch.float32
    model = ClawerModels.from_pretrained(pretrained_model_path, torch_dtype=precision).to(device)
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config, )
    # model.inpainter = SimpleLama()
    # model.sd_inpainter = sd_inpainter
    model.guidance_scale = 3.5
    model.num_step = 50
    model.max_resolution = 768
    model.eta = 0.0


    # controller = AttentionStore()
    controller = Mask_Expansion_SELF_ATTN()
    controller.contrast_beta = 1.67
    controller.use_contrast = True

    # register_attention_control_new(model, controller)
    register_attention_control(model, controller)

    mask_threshold = 0.1

    control_value = 0.15
    control_point = 0.4

    roi_expansion = True

    mask_dilation = False #init input
    dilation_kernel_size = 15

    vis_each_step = False

    post_process = 'hard'
    adp_k=5
    contrast_beta = 1.67
    use_contrast = False


    #start!
    image_path = "/data/Hszhu/Reggio/examples/Expansion_Mask/CPIG/"
    mask_path = "/data/Hszhu/Reggio/examples/Expansion_Mask/masks/"
    dst_path = "/data/Hszhu/Reggio/examples/Expansion_Mask/hard_high_cut_off_contrast_both_1.67"
    # dst_path = "/data/Hszhu/DragonDiffusion/examples/Expansion_Mask/"
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    BFS_plot_save = os.path.join(dst_path,'BFS-distance')
    Path(BFS_plot_save).mkdir(parents=True,exist_ok=True)
    model.BFS_SAVE_PATH = BFS_plot_save
    prepare_input = get_images_and_masks(image_path,mask_path)
    keys = prepare_input.keys()
    inter_latents = []
    for k in sorted(keys):
        instance = prepare_input[k]
        img = instance['image']
        msk = instance['mask']
        model.image_name = k
        inverted_latent,expand_mask,resized_original_msk,resized_img = model.mask_expansion_with_ddim_inv(original_image=img, mask=msk, prompt="",seed=42,
                                                         guidance_scale=model.guidance_scale, num_step=model.num_step,
                                                         max_resolution=model.max_resolution,
                                                         eta=model.eta,controller=controller,roi_expansion=roi_expansion,
                                                         mask_dilation=mask_dilation,dilation_kernel_size=dilation_kernel_size,maintain_step_mask=vis_each_step,
                                                         mask_threshold=mask_threshold,post_process=post_process,
                                                         control_value=control_value, control_point=control_point,adp_k=adp_k,
                                                         use_contrast=use_contrast,contrast_beta=contrast_beta)

        if vis_each_step:
            for s,v in expand_mask.items():
                save_and_visualize_masks(img=resized_img,
                                         expand_mask=v,
                                         original_mask=resized_original_msk,
                                         dst_path=dst_path, k=k,name =s )
        else:
            save_and_visualize_masks(img=resized_img,
                                     expand_mask=expand_mask,
                                     original_mask=resized_original_msk,
                                     dst_path=dst_path, k=k,)

