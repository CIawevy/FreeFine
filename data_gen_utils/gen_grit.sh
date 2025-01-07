#!/bin/bash

# 传入的命令行参数
DATA_ID=$1
BASE_DIR=$2
GPU_ID=$3

# 设置指定 GPU 为可见设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 激活 Conda 环境
source /home/hszhu/anaconda3/etc/profile.d/conda.sh

# 进入指定目录
#cd /data/Hszhu/Semantic-SAM

# GroundingSAM segmentation and Filter(abandon)
#python data_gen_utils/auto_mask_gen_tag_parser.py --data_id "$DATA_ID" --base_dir "$BASE_DIR"

#SemanticSAM RAM++
#conda activate ssm
##python auto_mask_gen_parser.py --data_id "$DATA_ID" --base_dir "$BASE_DIR"
#conda deactivate

#sharegpt4v captioning

#conda activate relabel
#python  relabeling_v2.py --data_id "$DATA_ID" --base_dir "$BASE_DIR"
#conda deactivate

#conda activate flask
#cd /data/Hszhu/Reggio/flask_app
#bash preprocess_before_ui.sh "$DATA_ID" "$BASE_DIR"
#conda deactivate

#Background inpainting and semantic expansion
#conda activate Comfy
#cd /data/Hszhu/comfyui-inpaint-nodes
#python infer_fooocus_mat.py --data_id "$DATA_ID" --base_dir "$BASE_DIR"
#conda deactivate

#环境变更
conda activate pt2
#进入指定目录
cd /data/Hszhu/generative-models/
#pt2 coarse edit
python scripts/sampling/coarse_editing_2d_3d_parser.py --data_id "$DATA_ID" --base_dir "$BASE_DIR"
conda deactivate


##环境变更
conda activate Reggio
#进入指定目录
cd /data/Hszhu/Reggio
#Reggio repainting
python data_gen_utils/repainting_parser.py --data_id "$DATA_ID" --base_dir "$BASE_DIR"
conda deactivate
