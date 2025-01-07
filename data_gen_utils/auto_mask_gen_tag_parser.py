from typing import List
from PIL import Image
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import os.path as osp
import torch
import spacy
import sys

sys.path.append('/data/Hszhu/Reggio')
from torchvision.transforms import ToTensor
from groundingdino.util.inference import Model
from sam.efficient_sam import build_efficient_sam_vits
import nltk
import clip
import argparse
# import litellm
nltk.data.path.append("/data/Hszhu/prompt-to-prompt/nltk/")
# nltk.download('punkt', download_dir="/data/Hszhu/prompt-to-prompt/nltk/")
# nltk.download('punkt_tab', download_dir="/data/Hszhu/prompt-to-prompt/nltk/")
# nltk.download('averaged_perceptron_tagger_eng', download_dir="/data/Hszhu/prompt-to-prompt/nltk/")
# nltk.download('wordnet', download_dir="/data/Hszhu/prompt-to-prompt/nltk/")
def load_clip_on_the_main_Model(main_model,device):
    # 加载CLIP模型和处理器
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("ViT-L/14", device=device)
    main_model.clip = model
    main_model.clip_process = preprocess
    return main_model
# Tag2Text
# from ram.models import tag2text_caption
from ram.models import ram
from ram.models import tag2text
from ram import inference_tag2text
from ram import inference_ram
import torchvision
import torchvision.transforms as TS
from tqdm import  tqdm
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

def generate_tags(caption, split=',', max_tokens=100, model="gpt-3.5-turbo"):
    lemma = nltk.wordnet.WordNetLemmatizer()
    # if openai_key:
    #     prompt = [
    #         {
    #             'role': 'system',
    #             'content': 'Extract the unique nouns in the caption. Remove all the adjectives. ' + \
    #                        f'List the nouns in singular form. Split them by "{split} ". ' + \
    #                        f'Caption: {caption}.'
    #         }
    #     ]
    #     response = litellm.completion(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
    #     reply = response['choices'][0]['message']['content']
    #     # sometimes return with "noun: xxx, xxx, xxx"
    #     tags = reply.split(':')[-1].strip()
    # else:
    # Ensure necessary resources are downloaded
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    # nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])
    tags_list = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[0] == 'N']
    tags_lemma = [lemma.lemmatize(w) for w in tags_list]
    tags = ', '.join(map(str, tags_lemma))
    return tags
def save_masks(masks, dst_dir, da_name):
    # 创建子文件夹
    subfolder_path = os.path.join(dst_dir, da_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # 用于存储保存的mask路径
    mask_paths = []

    # 保存每个mask到子文件夹中
    for idx, mask in enumerate(masks):
        mask_path = os.path.join(subfolder_path, f"mask_{idx + 1}.png")
        cv2.imwrite(mask_path, mask * 255)  # 将mask保存为png图片 (注意：mask是二值图，乘以255以得到可见的结果)
        print(f"Saved mask {idx + 1} to {mask_path}")
        mask_paths.append(mask_path)

    return mask_paths

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union
    return iou

def is_subordinate(mask1, mask2,subordinate_thr):
    intersection = (mask1 & mask2).sum()
    if (intersection / mask2.sum())>subordinate_thr:
        return True
    return False
def mutual_any_is_subordinate_list(mask1, mask2_list,subordinate_thr):
    for mask2 in mask2_list:
        intersection = (mask1 & mask2).sum()
        if ((intersection / mask2.sum())>subordinate_thr) or ((intersection / mask1.sum())>subordinate_thr) :
            return True
    return False
def mutual_any_is_subordinate(mask1, mask2,subordinate_thr):
    intersection = (mask1 & mask2).sum()
    if ((intersection / mask2.sum())>subordinate_thr) or ((intersection / mask1.sum())>subordinate_thr) :
        return True
    return False
def sub_masks_filter_v2(detections, AUTOMATIC_CLASSES, subordinate_thr=0.8):
    xyxy = detections.xyxy
    class_id = detections.class_id
    masks = detections.mask
    confidences = detections.confidence

    keep_masks = []
    keep_confidences = []
    keep_ids = []
    keep_xyxy = []

    # 排序：按置信度从高到低排序
    sorted_idx = np.argsort(-confidences)
    masks = masks[sorted_idx]
    confidences = confidences[sorted_idx]
    class_id = class_id[sorted_idx]
    xyxy = xyxy[sorted_idx]

    while len(masks) > 0:
        # 选择当前置信度最高的掩码
        current_mask = masks[0]
        current_confidence = confidences[0]
        current_id = class_id[0]
        current_xyxy = xyxy[0]
        current_label = AUTOMATIC_CLASSES[current_id]

        keep_masks.append(current_mask)
        keep_confidences.append(current_confidence)
        keep_ids.append(current_id)
        keep_xyxy.append(current_xyxy)

        if len(masks) == 1:
            break

        keep_idx = []

        # 对剩余掩码进行从属关系判断
        for idx in range(1, len(masks)):
            judge_cls = AUTOMATIC_CLASSES[class_id[idx]]
            if not mutual_any_is_subordinate(current_mask, masks[idx], subordinate_thr):
                keep_idx.append(idx)
        #TODO:
        # 现在对于field 和 martial，因为前者置信度高保留前者，而后者又包含前者，这时需要考虑取舍
        # 如果要通过双向从属判断，我们得保证我们需要的物体首先置信度高，否则会出现保留t-shirt不保留man的问题
        # 验证其他case是否合理,目前暂时改为mutual从属,
        # 注意还受到分割掩码影响如birthday cake

        masks = masks[keep_idx]
        confidences = confidences[keep_idx]
        class_id = class_id[keep_idx]
        xyxy = xyxy[keep_idx]

    detections.xyxy = np.array(keep_xyxy)
    detections.mask = np.array(keep_masks)
    detections.class_id = np.array(keep_ids)
    detections.confidence = np.array(keep_confidences)

    return detections
def pre_process_with_mask_list(img, mask_list):
    processed_imgs = []
    for mask in mask_list:
        mask[mask>0] =1
        # 1. Apply dilation to the mask
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

        # 2. Blend the edges of the mask using Gaussian blur
        mask_blurred = cv2.GaussianBlur(dilated_mask, (21, 21), 0)
        # mask_blurred = mask_blurred / 255.0  # Normalize mask to [0, 1]

        # 3. Calculate the average color of the image
        average_color = img.mean(axis=(0, 1), keepdims=True).astype(np.uint8)
        avg_color_img = np.ones_like(img) * average_color

        # 4. Integrate the masked area with the average color using the blurred mask
        mask_expanded = np.repeat(mask_blurred[:, :, np.newaxis], 3,
                                  axis=2)  # Expand mask to match image channels
        img_np = img * mask_expanded + avg_color_img * (1 - mask_expanded)

        # Clip the result to ensure it's within [0, 255] and convert to uint8
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        processed_imgs.append(img_np)

    return processed_imgs

def crop_image_with_mask(img, mask):
    # 使用cv2.boundingRect获取mask的边界框
    x, y, w, h = cv2.boundingRect(mask)

    # 裁剪图像
    if isinstance(img, List):
        cropped_img = [im[y:y + h, x:x + w] for im in img]
    else:
        cropped_img = img[y:y + h, x:x + w]
    return cropped_img
def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
def compute_clip_score(model,img,current_mask, current_label):
    """
    计算与label的clip得分
    """
    with torch.no_grad():
        #crop img feat
        mask_list = [current_mask]
        # cropped_img_list = [model.crop_image_with_mask(img, mask) for mask in mask_list]
        processed_img_list =pre_process_with_mask_list(img, mask_list)
        cropped_img = crop_image_with_mask(processed_img_list[0], mask_list[0])
        processed_cropped_img = model.clip_process(numpy_to_pil(cropped_img)[0])
        image_features = model.clip.encode_image(processed_cropped_img.to(model.device).unsqueeze(0))



        label_tokens = clip.tokenize(current_label).to(model.device)
        label_features = model.clip.encode_text(label_tokens)


        #normalize
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        label_features = label_features / label_features.norm(dim=1, keepdim=True)

        sim_pos = torch.matmul(image_features, label_features.T).item()
        return sim_pos
def retain_largest_connected_component(mask):
    """
    仅保留mask中面积最大的连通组件，其余部分设置为0。

    参数:
    mask (numpy.ndarray): 输入的二值化掩码图像。

    返回:
    numpy.ndarray: 仅保留最大连通组件的掩码图像。
    """
    # 确保输入掩码是单通道的灰度图
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)
    mask[mask>0] = 1
    # 寻找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 如果没有连通组件，直接返回空图像
    if num_labels == 1:  # 只有背景
        return np.zeros_like(mask)

    # 获取所有连通组件的面积（排除背景，背景索引为0）
    areas = stats[1:, cv2.CC_STAT_AREA]  # 不包括背景的面积
    max_area_idx = np.argmax(areas) + 1  # 找到最大面积的组件索引（+1 因为跳过了背景）

    # 创建一个空白图像
    filtered_mask = np.zeros_like(mask)

    # 仅保留面积最大的组件
    filtered_mask[labels == max_area_idx] = 255

    return filtered_mask
def edge_objects_filter(mask, edge_thr=0.50):
    """
    过滤贴边的物体，通过计算mask在图像边缘方向上的像素数量与bounding box
    在另一个方向上的尺寸比例来判断是否去除。

    参数:
    mask (ndarray): 当前物体的mask。
    edge_thr (float): 边缘阈值，决定是否过滤物体的标准。

    返回:
    bool: 如果物体位于边缘且应被去除，则返回True；否则返回False。
    """
    if mask.sum() == 0:
        return False

    img_height, img_width = mask.shape

    # 获取mask的bounding box
    x_min, y_min, box_width, box_height = cv2.boundingRect(mask.astype(np.uint8))
    x_max = x_min + box_width
    y_max = y_min + box_height

    edge_y_count_left,edge_y_count_right,edge_x_count_top,edge_x_count_bottom=0,0,0,0
    x_overlap_stat ,y_overlap_stat=False,False
    # 如果bounding box左或右边缘与图像边缘重合，则计算y方向上的重合度
    if x_min == 0:
        edge_y_count_left = mask[y_min:y_max, 0].sum() if x_min == 0 else 0
        y_overlap_stat=True

    if  x_max == img_width:
        edge_y_count_right = mask[y_min:y_max, img_width-1].sum() if x_max == img_width else 0
        y_overlap_stat = True
    # 如果bounding box上或下边缘与图像边缘重合，则计算x方向上的重合度
    if y_min == 0 :
        edge_x_count_top = mask[0, x_min:x_max].sum() if y_min == 0 else 0
        x_overlap_stat = True
    if y_max == img_height:
        edge_x_count_bottom = mask[img_height-1, x_min:x_max].sum() if y_max == img_height else 0
        x_overlap_stat = True
    y_overlap_box = max(edge_y_count_left, edge_y_count_right) / box_height
    # x_overlap_box = max(edge_x_count_top, edge_x_count_bottom) / box_width
    y_overlap_img = max(edge_y_count_left, edge_y_count_right) /img_height if y_overlap_stat else 0
    x_overlap_img =max(edge_x_count_top, edge_x_count_bottom)/ img_width if x_overlap_stat else 0
    # final_stat = (x_overlap_img > edge_thr) or (y_overlap_img > edge_thr) or (y_overlap_box > edge_thr)
    final_stat = (x_overlap_img > edge_thr) or  (y_overlap_img > edge_thr)
    # 如果任一方向的重合与自身该边长比例超过阈值，则认为该物体是贴边物体,贴边程度与背景比大者直接去除,y轴大部分贴边物体不利于编辑也去除
    return final_stat
def tiny_filter(mask,tiny_thr):
    img_height, img_width = mask.shape
    image_size = img_width*img_height
    area_ratio = mask.sum() / image_size
    return area_ratio < tiny_thr


def Tiny_filter(detections, tiny_thr=0.10):
    xyxy = detections.xyxy
    class_id = detections.class_id
    masks = detections.mask.astype(np.uint8)
    masks[masks > 0] = 1
    confidences = detections.confidence

    keep_masks = []
    keep_confidences = []
    keep_ids = []
    keep_xyxy = []

    # all_pixel = mask.shape[0] * mask.shape[1]
    # mask_areas = np.sum(masks, axis=(1, 2)) / all_pixel
    # sorted_idx = np.argsort(-(mask_areas + confidences))
    # masks = masks[sorted_idx]
    # confidences = confidences[sorted_idx]
    # class_id = class_id[sorted_idx]
    # xyxy = xyxy[sorted_idx]

    while len(masks) > 0:
        current_mask = masks[0]
        current_confidence = confidences[0]
        current_id = class_id[0]
        current_xyxy = xyxy[0]

        # 仅保留较小物体的过滤逻辑
        if tiny_filter(current_mask, tiny_thr):
            masks = masks[1:]
            confidences = confidences[1:]
            class_id = class_id[1:]
            xyxy = xyxy[1:]
            continue

        keep_masks.append(current_mask)
        keep_confidences.append(current_confidence)
        keep_ids.append(current_id)
        keep_xyxy.append(current_xyxy)

        masks = masks[1:]
        confidences = confidences[1:]
        class_id = class_id[1:]
        xyxy = xyxy[1:]

    detections.xyxy = np.array(keep_xyxy)
    detections.mask = np.array(keep_masks)
    detections.class_id = np.array(keep_ids)
    detections.confidence = np.array(keep_confidences)

    return detections
def Clawer_masks_post_filter(model, img, detections, AUTOMATIC_CLASSES,mask, subordinate_thr=0.8, edge_thr=0.5, tiny_thr=0.05):
    xyxy = detections.xyxy
    class_id = detections.class_id
    masks = detections.mask.astype(np.uint8)
    masks[masks>0] = 1
    confidences = detections.confidence

    keep_masks = []
    keep_confidences = []
    keep_ids = []
    keep_xyxy = []
    #从属物体保留最大，因此从mask面积排序
    # sorted_idx = np.argsort(-confidences)
    all_pixel = mask.shape[0]*mask.shape[1]
    mask_areas = np.sum(masks, axis=(1, 2))/all_pixel
    sorted_idx = np.argsort(-(mask_areas+confidences))
    masks = masks[sorted_idx]
    confidences = confidences[sorted_idx]
    class_id = class_id[sorted_idx]
    xyxy = xyxy[sorted_idx]

    while len(masks) > 0:
        stat=True
        # 选择当前面积最大的掩码
        current_mask = masks[0]
        current_confidence = confidences[0]
        current_id = class_id[0]
        current_xyxy = xyxy[0]
        current_label = AUTOMATIC_CLASSES[current_id]
        # CLAWER Post Refine Filter构建
        # 1. 贴边物体去除（排除背景、不完全的物体、不完美的mask最有效的去除方式）此处用mask x，y贴边部分比mask box x，y的比值与阈值来判断
        # 2. 较小物体去除，可编辑性差，占整张图百分比阈值
        # 3. 反转可行解尝试，对每个text-mask 对 构建反转mask 对 额外扩充，基于clip score 与 text是否更好，选择还是保留物体
        # 4. 从属物体保留最大和置信度最高综合考虑的（可惜拿不到分割置信度）
        #.1贴边和较小物体去除
        if edge_objects_filter(current_mask,  edge_thr) or tiny_filter(current_mask, tiny_thr):
            stat=False
            # 2. 当原mask不通过时，反转可行解尝试，反转解需不贴边、不小、同时不与原本mask set的任何mask为从属关系
            inverted_mask = 1 - current_mask
            if not(edge_objects_filter(inverted_mask,  edge_thr) or tiny_filter(inverted_mask, tiny_thr) or mutual_any_is_subordinate_list(inverted_mask, masks, subordinate_thr)):
                original_clip_score = compute_clip_score(model,img,current_mask, current_label)
                inverted_clip_score = compute_clip_score(model,img,inverted_mask, current_label)
                if inverted_clip_score > original_clip_score:
                    current_mask = inverted_mask
                    stat=True

        if not stat:
            masks = masks[1:]
            confidences = confidences[1:]
            class_id = class_id[1:]
            xyxy = xyxy[1:]
            continue

        keep_masks.append(current_mask)
        keep_confidences.append(current_confidence)
        keep_ids.append(current_id)
        keep_xyxy.append(current_xyxy)

        if len(masks) == 1:
            break

        keep_idx = []

        # 对剩余掩码进行从属关系判断，从属物体去除
        for idx in range(1, len(masks)):
            judge_cls = AUTOMATIC_CLASSES[class_id[idx]]
            if not mutual_any_is_subordinate(current_mask, masks[idx], subordinate_thr):#由于mask从大到小判断，并排除背景，因此可以直接去除从属物体
                keep_idx.append(idx)


        masks = masks[keep_idx]
        confidences = confidences[keep_idx]
        class_id = class_id[keep_idx]
        xyxy = xyxy[keep_idx]

    detections.xyxy = np.array(keep_xyxy)
    detections.mask = np.array(keep_masks)
    detections.class_id = np.array(keep_ids)
    detections.confidence = np.array(keep_confidences)

    return detections
def sub_masks_filter(detections,AUTOMATIC_CLASSES,subordinate_thr=0.8):
    xyxy = detections.xyxy
    class_id = detections.class_id
    masks = detections.mask
    confidences = detections.confidence

    keep_masks = []
    keep_confidences = []
    keep_ids = []
    keep_xyxy = []

    # 排序：按置信度从高到低排序
    sorted_idx = np.argsort(-confidences)
    masks = masks[sorted_idx]
    confidences = confidences[sorted_idx]
    class_id = class_id[sorted_idx]
    xyxy = xyxy[sorted_idx]

    while len(masks) > 0:
        # 选择当前置信度最高的掩码
        keep_masks.append(masks[0])
        keep_confidences.append(confidences[0])
        keep_ids.append(class_id[0])
        keep_xyxy.append(xyxy[0])

        if len(masks) == 1:
            break

        # 计算其余掩码与当前掩码的IoU
        # ious = np.array([calculate_iou(masks[0], mask) for mask in masks[1:]])

        subordinate_flags = np.array([is_subordinate(masks[0], mask,subordinate_thr) for mask in masks[1:]])
        # 保留IoU低于阈值即没有从属关系的掩码
        # keep_idx = np.where((ious < iou_threshold) & (~subordinate_flags))[0] + 1  # +1 是因为跳过当前置信度最高的掩码
        keep_idx = np.where(~subordinate_flags)[0] + 1  # +1 是因为跳过当前置信度最高的掩码

        masks = masks[keep_idx]
        confidences = confidences[keep_idx]
        class_id = class_id[keep_idx]
        xyxy = xyxy[keep_idx]

    detections.xyxy = keep_xyxy
    detections.mask = keep_masks
    detections.class_id = keep_ids
    detections.confidence = keep_confidences

    return detections


# 过滤函数
def filter_words(nlp,word_list, ignore_list):
    filtered_words = []
    for word in word_list:
        doc = nlp(word)
        if doc[0].pos_ == 'NOUN' and word not in ignore_list:
            filtered_words.append(word)
    return filtered_words
def temp_view( mask, title='Mask', name=None):
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


def my_annotate_image(img, detections, labels):
    overlay = img.copy()
    output = img.copy()

    for i, (box, label, mask, score) in enumerate(
            zip(detections.xyxy, labels, detections.mask, detections.confidence)):
        color = (0, 255, 0)  # 绿色作为mask的颜色
        mask = mask.astype(bool)

        overlay[mask] = color

        y, x = np.mean(np.argwhere(mask), axis=0).astype(int)

        text = f"{label} {score:.2f}"
        cv2.putText(output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

    return output
# def vis_masks(detections,image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # annotate image with detections
#     # box_annotator = sv.BoxAnnotator()
#     # mask_annotator = sv.MaskAnnotator()
#     labels = [
#         f"{AUTOMATIC_CLASSES[class_id]} {confidence:0.2f}"
#         for _, _, confidence, class_id, _, _
#         in detections]
#     # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
#     # visualize_rgb_image(Image.fromarray(annotated_image))
#     # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
#     # visualize_rgb_image(Image.fromarray(annotated_image))
#     annotated_image = my_annotate_image(image.copy(), detections,labels)
#     return annotated_image


def temp_view_img(image: Image.Image, title: str = None) -> None:
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
def read_img(image_path):
    img = cv2.imread(image_path)  # bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
import random
def split_data(data, num_splits, subset_num=None,seed=None):
    if seed is not None:
        random.seed(seed)
    data_keys = list(data.keys())

    # 如果需要从数据中随机抽取100个
    if subset_num is not None:
        data_keys = random.sample(data_keys, subset_num)  # 随机抽取subset_num个键
    # else:
    #     random.shuffle(data_keys)  # 随机打乱数据键

    chunk_size = len(data_keys) // num_splits
    data_parts = []

    for i in range(num_splits):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_splits - 1 else len(data_keys)
        data_part = {k: data[k] for k in data_keys[start_idx:end_idx]}
        data_parts.append(data_part)

    return data_parts

#BLIP
# blip2 = BLIP2("/data/Hszhu/prompt-to-prompt/blip2-opt-2.7b/")
#RAM
# ram_model = ram(pretrained=RAM_CHECKPOINT_PATH,
#                                         image_size=384,
#                                         vit='swin_l',
#                                         text_encoder_type='/data/Hszhu/prompt-to-prompt/bert-base-uncased')
# ram_model.eval()
# ram_model = ram_model.to(DEVICE)
import argparse
import os
import os.path as osp
import cv2
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import numpy as np



def efficient_sam_box_prompt_segment(image, pts_sampled, model):
    bbox = torch.reshape(torch.tensor(pts_sampled), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].cuda(),
        bbox.cuda(),
        bbox_labels.cuda(),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if curr_predicted_iou > max_predicted_iou or selected_mask_using_predicted_iou is None:
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou

def main(data_id, base_dir):
    dst_base = osp.join(base_dir, f'Subset_{data_id}')
    if osp.exists(osp.join(dst_base, f"packed_data_full_tag_{data_id}.json")):
        print(f'grounding for {data_id} already finish!')
        return
    # MAIN Process
    ckpt_base_dir = "/data/Hszhu/prompt-to-prompt/GroundingSAM_ckpts"
    # load models
    DEVICE =DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    GROUNDING_DINO_CONFIG_PATH = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = osp.join(ckpt_base_dir, "groundingdino_swint_ogc.pth")

    # SAM_ENCODER_VERSION = "vit_h"
    # SAM_CHECKPOINT_PATH = osp.join(ckpt_base_dir,"sam_vit_h_4b8939.pth")

    TAG2TEXT_CHECKPOINT_PATH = osp.join(ckpt_base_dir, "tag2text_swin_14m.pth")
    RAM_CHECKPOINT_PATH = osp.join(ckpt_base_dir, "ram_swin_large_14m.pth")

    TAG2TEXT_THRESHOLD = 0.64
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.7
    MAX_INSTANCES = 10

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    grounding_dino_model = load_clip_on_the_main_Model(grounding_dino_model, DEVICE)  # load GS with CLIP

    # Building SAM Model and SAM Predictor
    # sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    # sam_predictor = SamPredictor(sam)#TODO:CHECK 182 102 64 1 3 5 0 377 378
    efficientsam = build_efficient_sam_vits().to(DEVICE)

    # Tag2Text
    # initialize Tag2Text
    normalize = TS.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = TS.Compose(
        [
            TS.Resize((384, 384)),
            TS.ToTensor(),
            normalize
        ]
    )
    # 加载Spacy的语言模型
    # nlp = spacy.load('en_core_web_sm')
    # IGNORE_LIST = []
    DELETE_TAG_INDEX = []  # filter out attributes and action which are difficult to be grounded
    for idx in range(3012, 3429):
        DELETE_TAG_INDEX.append(idx)

    # tag2text
    tag2text_model = tag2text(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                              image_size=384,
                              vit='swin_b',
                              delete_tag_index=DELETE_TAG_INDEX,
                              text_encoder_type='/data/Hszhu/prompt-to-prompt/bert-base-uncased')
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    tag2text_model.threshold = TAG2TEXT_THRESHOLD
    tag2text_model.eval()
    tag2text_model = tag2text_model.to(DEVICE)
    omit_list = ['photo', 'eye', 'rock', 'dress', 'couple', 'wall','ear','sky']
    dst_dir_path = osp.join(dst_base, "masks_tag/")
    new_data = dict()
    dataset_json_file = osp.join(base_dir, "meta_data.json")
    data = load_json(dataset_json_file)  # load img paths
    data_parts = split_data(data, 8, seed=42, subset_num=80000)

    for da_n, da in tqdm(data_parts[data_id].items(), desc=f'procedding GroundingSAM Part:{data_id}'):
        try:
            new_data[da_n] = dict()
            # da = data_parts[data_id]['012987559'] #'018178356'
            SOURCE_IMAGE_PATH = da['img_path'] #for grit use img_path
            SOURCE_CAPTION = da['caption']
            image = cv2.imread(SOURCE_IMAGE_PATH)  # bgr
            image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # rgb

            image_pillow = image_pillow.resize((384, 384))
            image_pillow = transform(image_pillow).unsqueeze(0).to(DEVICE)

            # Tag2Text
            specified_tags = 'None'
            res_tag2text = inference_tag2text(image_pillow, tag2text_model, specified_tags)
            AUTOMATIC_CLASSES = res_tag2text[0].split(" | ")
            text_prompt = res_tag2text[0].replace(' |', ',')
            caption = res_tag2text[2]
            AUTOMATIC_CLASSES = [a for a in AUTOMATIC_CLASSES if a not in omit_list]

            # GroundingDino detect objects
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=AUTOMATIC_CLASSES,
                box_threshold=BOX_THRESHOLD,
                text_threshold=BOX_THRESHOLD
            )

            # NMS post process
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                IOU_THRESHOLD
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            # Efficient SAM segmentation
            result_masks = []
            for box in detections.xyxy:
                mask = efficient_sam_box_prompt_segment(image, box, efficientsam)
                result_masks.append(retain_largest_connected_component(mask))

            detections.mask = np.array(result_masks)

            # detections = clawer_masks_post_filter(grounding_dino_model, image, detections, AUTOMATIC_CLASSES,mask,
            #                                       subordinate_thr=0.8, edge_thr=0.7, tiny_thr=0.01)
            detections = Tiny_filter(detections,tiny_thr=0.10)

            labels = [
                f"{AUTOMATIC_CLASSES[class_id]}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            if len(labels) == 0 or len(labels) > MAX_INSTANCES:
                print(f'skip for {len(labels)} in img_id:{da_n}')
                continue

            annotated_image = my_annotate_image(image.copy(), detections, labels)
            mask_path = save_masks(detections.mask, dst_dir_path, da_n)



            cv2.imwrite(os.path.join(osp.dirname(mask_path[0]), f"anotated_img.png"), annotated_image)

            instances = dict()
            instances['obj_label'] = []
            instances['mask_path'] = []
            for i in range(len(detections.mask)):
                instances['obj_label'].append(labels[i])
                instances['mask_path'].append(mask_path[i])

            new_data[da_n]['instances'] = instances
            new_data[da_n]['caption'] = SOURCE_CAPTION
            new_data[da_n]['src_img_path'] = SOURCE_IMAGE_PATH
        except:
            print('skip error case')
            continue

    save_json(new_data, osp.join(dst_base, f"packed_data_full_tag_{data_id}.json"))

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="GroundingSAM processing script")
    parser.add_argument('--data_id', type=int, required=True, help="Data ID to process")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory path for dataset")
    # parser.add_argument('--gpu_id', type=int, default=0, help="Specify the GPU to use. Default is GPU 0")

    args = parser.parse_args()




    # 在需要时使用 device
    # model.to(device)
    # tensor = tensor.to(device)

    # 调用主逻辑并传入设备
    main(args.data_id, args.base_dir)

