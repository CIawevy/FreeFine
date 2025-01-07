#!/bin/bash

# 传入的命令行参数
DATA_ID=$1
BASE_DIR=$2


python process_img_for_ui.py --data_id "$DATA_ID" --base_dir "$BASE_DIR"
rm -rf "${BASE_DIR}/Subset_${DATA_ID}/flask_preprocess_stat.json"
