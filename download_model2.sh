#!/bin/bash

# 设置环境变量
source /home/hszhu/anaconda3/etc/profile.d/conda.sh
conda activate Reggio
export HF_ENDPOINT=https://hf-mirror.com

# 设置循环次数
NUM_ITERATIONS=300  # 这里设置为你需要的循环次数

# 执行下载命令
for ((i = 1; i <= NUM_ITERATIONS; i++)); do
    echo "正在执行第 $i 次下载..."
    huggingface-cli download  --resume-download Comfy-Org/flux1-schnell  --local-dir "/data/Hszhu/prompt-to-prompt/Flux1-schnell-fp8-comfy" --local-dir-use-symlinks False --resume-download
done
