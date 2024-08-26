import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from data_gen_utils.data import PIE_data_preprocessor
import cv2

#载入数据集,模型等
#1.img/caption like laion
#2.img/caption box/exp_text like GRIT
#3.img box/mask using dino to get label and mask
#into dict ,waiting for batch data generation
#注意多卡load 逻辑，以及初始化尺寸设置
#TODO:1. firstly try toy model
#TODO:based on the dataset's format.implement different code
PIE_BASE_DIR = "/data/Hszhu/dataset/PIE-Bench_v1/"
PIE_data_preprocessor(PIE_BASE_DIR)
print('load ok')
#初始化处理
#1.detect and segment get all the object mask
#2.get related object description
#3.for each object do prompt guided mask expansion before inpainting
#4.for each case try inpainting method for all
#4.get the best inpainting results for each object-removal case,if use sd,must guaranteen that it is better than lama
#TODO:implement expansion and inpainting pipe today 8.11



#given img,obj_mask_exp,obj_text,inpaint img
#randomly sample motions from pool
#if 3D judge whether it is appropriate case(Optional)
#if 2D case,especially multi obj case,ensure appropriate sample motion degrees
# transform and paste back to inpaint img
#TODO:inplement it today 8.11

#gen results
#!!!make sure ddim works well for real image like human or car!!!! #TODO solve this problem maybe pretrained at firste
#actually is there any training-based editing arts?
#input coarse input and obj_mask with cfg_guidance prompt,using our pipe to generate results
#TODO:already done except for 并行化,输出与存储

