#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置为hf的国内镜像网站

from huggingface_hub import snapshot_download
import os.path as osp

# 指定要下载的模型或数据集
# model_name = 'poloclub/diffusiondb'
model_name = 'bert-base-uncased'
local_dir = osp.join("/data/Hszhu/prompt-to-prompt/", model_name)

# while True 是为了防止断联
while True:
    try:
        snapshot_download(
            repo_id=model_name,
            repo_type='model',
            local_dir_use_symlinks=True,  # 在local-dir指定的目录中都是一些“链接文件”
            # allow_patterns=["images/*.zip"],  # 只下载images文件夹中的所有压缩包文件
            local_dir=local_dir,
            token="hf_hf_HezQIwHGXLSOQThxPnJGBBMVNduRokcISH",   # huggingface的token
            resume_download=True
        )
        break
    except Exception as e:
        print(f"下载过程中发生错误: {e}")
        pass
