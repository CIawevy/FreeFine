#!/bin/bash

# 请将以下路径替换为您的 conda 初始化脚本路径
source /path/to/your/conda/etc/profile.d/conda.sh

conda activate FreeFine #use huggingface_hub

# 如果您无法直接访问 Hugging Face，请取消注释以下行并设置镜像或代理
#export HF_ENDPOINT=https://hf-mirror.com  # 或者使用代理

# 设置循环次数，防止下载中断
NUM_ITERATIONS=20  # 根据需要调整循环次数

# 请将以下信息替换为您的 Hugging Face token 
HF_TOKEN="your_huggingface_token" #IF NEEDED

# 统一设置本地 checkpoint 根目录（用户仅需修改此处）
CHECKPOINT_DIR="./checkpoints"  # 可替换为绝对路径，如 /data/FreeFine/checkpoints

# 创建统一目录（自动生成子目录）
mkdir -p "${CHECKPOINT_DIR}/stable-diffusion-v1-5"       # SD-15 模型目录
mkdir -p "${CHECKPOINT_DIR}/sv3d"        # SV3D 模型目录
mkdir -p "${CHECKPOINT_DIR}/depth-anything"  # Depth Anything 模型目录

#download SD-15 baseline model
TARGET1="stable-diffusion-v1-5/stable-diffusion-v1-5"
#download SV3D model（目标目录直接指向统一路径）
TARGET2="stabilityai/sv3d"

# 执行下载命令（统一路径）
for ((i = 1; i <= NUM_ITERATIONS; i++)); do
    echo "正在执行第 $i 次下载..."
    huggingface-cli download --token $HF_TOKEN --resume-download $TARGET1 \
        --local-dir "${CHECKPOINT_DIR}/sd-15" --local-dir-use-symlinks False
    huggingface-cli download --token $HF_TOKEN --resume-download $TARGET2 \
        --local-dir "${CHECKPOINT_DIR}/sv3d" --local-dir-use-symlinks False
done

# 移动 SV3D 模型到 generative-models/checkpoints目录
cd "$(dirname "$0")/.."  # 回到项目根目录
mkdir -p "generative-models/checkpoints"  # 确保目标目录存在
mv "${CHECKPOINT_DIR}/sv3d"/* "generative-models/checkpoints/"

#download depth anything model ckpt（目标目录指向统一路径）
cd "${CHECKPOINT_DIR}/depth-anything"
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
# wget https://hf-mirror.com/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth



echo "所有模型已下载至 ${CHECKPOINT_DIR}，SV3D 模型已同步至 generative-models/checkpoints"