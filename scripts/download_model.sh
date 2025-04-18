#!/bin/bash

# 设置环境变量
# 请将以下路径替换为您的 conda 初始化脚本路径
source /path/to/your/conda/etc/profile.d/conda.sh

# 请将以下环境名称替换为您的 conda 环境名称
conda activate FreeFine

# 如果您无法直接访问 Hugging Face，请取消注释以下行并设置镜像或代理
# export HF_ENDPOINT=https://hf-mirror.com  # 或者使用代理

# 设置循环次数，防止下载中断
NUM_ITERATIONS=20  # 根据需要调整循环次数

# 请将以下信息替换为您的 Hugging Face token 和本地路径
HF_TOKEN="your_huggingface_token" #IF NEEDED
LOCAL_DIR="your_local_path"
TARGET="stable-diffusion-v1-5/stable-diffusion-v1-5"
# 执行下载命令
for ((i = 1; i <= NUM_ITERATIONS; i++)); do
    echo "正在执行第 $i 次下载..."
    huggingface-cli download --token $HF_TOKEN --resume-download $TARGET --local-dir $LOCAL_DIR --local-dir-use-symlinks False --resume-download
done