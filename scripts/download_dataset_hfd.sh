#!/bin/bash


# 请将以下路径替换为您的 conda 初始化脚本路径
source /path/to/your/conda/etc/profile.d/conda.sh

# 请将以下环境名称替换为您的 conda 环境名称
conda activate FreeFine

export HF_ENDPOINT=https://hf-mirror.com
# 请将以下信息替换为您的 Hugging Face token 和本地路径
HF_TOKEN="your_huggingface_token" #IF NEEDED
LOCAL_DIR="your_local_path"
TARGET="CIawevy/GeoBenchMeta"


cd FreeFine/scripts
# 设置循环次数
NUM_ITERATIONS=1  # 这里设置为你需要的循环次数
# 执行下载命令
# 执行下载命令
for ((i = 1; i <= NUM_ITERATIONS; i++)); do
    echo "正在执行第 $i 次下载..."
    ./hfd.sh $TARGET --hf_token $HF_TOKEN --local-dir $LOCAL_DIR --tool aria2c --dataset
done
