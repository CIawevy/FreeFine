import numpy as np
import cv2
from PIL import Image

src_mask = cv2.resize(cv2.imread("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/new/1_source_mask.png"),dsize=(512,512),interpolation=cv2.INTER_NEAREST)
tgt_mask = cv2.resize(cv2.imread("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/new/1_target_mask.png"),dsize=(512,512),interpolation=cv2.INTER_NEAREST)
src_img = cv2.resize(cv2.imread("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/new/1_souce_img.png"),dsize=(512,512),interpolation=cv2.INTER_LANCZOS4)
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
tgt_img = cv2.resize(cv2.imread("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/new/1_coarse_img.png"),dsize=(512,512),interpolation=cv2.INTER_LANCZOS4)
tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
src_bg_mask = 255-src_mask
tgt_bg_mask = 255-tgt_mask
Image.fromarray(src_bg_mask).save("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/new/1_source_bg_mask.png")
Image.fromarray(tgt_bg_mask).save("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/new/1_target_bg_mask.png")
masked_src_img_fg = np.where(src_mask,src_img,0)
masked_src_img_bg = np.where(src_mask,0,src_img)
Image.fromarray(masked_src_img_fg).save("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/new/1_source_fg_img.png")
Image.fromarray(masked_src_img_bg).save("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/new/1_source_bg_img.png")