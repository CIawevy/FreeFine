#!/bin/bash

# 设置环境变量
source /home/hszhu/anaconda3/etc/profile.d/conda.sh
conda activate Reggio
export HF_ENDPOINT=https://hf-mirror.com

# 设置循环次数
NUM_ITERATIONS=300  # 这里设置为你需要的循环次数
# 执行下载命令
# 执行下载命令
for ((i = 1; i <= NUM_ITERATIONS; i++)); do
    echo "正在执行第 $i 次下载..."
    huggingface-cli download  --token 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 --resume-download Kijai/flux-fp8 --local-dir "/data/Hszhu/prompt-to-prompt/flux1-dev-fp8" --local-dir-use-symlinks False --resume-download
done
SG161222/Realistic_Vision_V4.0_noVAE