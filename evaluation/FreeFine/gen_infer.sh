#!/bin/bash

# 传入的命令行参数
DATA_ID=$1
GPU_ID=$2
BASE_DIR_1="/data/Hszhu/dataset/PIE-Bench_v1/"
BASE_DIR_2="/data/Hszhu/dataset/Subjects200K/"


# 设置指定 GPU 为可见设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 激活 Conda 环境
source /home/hszhu/anaconda3/etc/profile.d/conda.sh

conda activate Reggio
##进入指定目录
cd /data/Hszhu/Reggio
#Reggio repainting
python data_gen_utils/repainting_parser_3d.py --data_id "$DATA_ID" --base_dir "$BASE_DIR_1"
python data_gen_utils/repainting_parser_3d.py --data_id "$DATA_ID" --base_dir "$BASE_DIR_2"