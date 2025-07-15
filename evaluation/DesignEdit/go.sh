#!/bin/bash

# 要统计的目录路径
#target_dir="/data/Hszhu/dataset/Geo-Bench/Gen_results_DesignEdit/"
target_dir="/data/Hszhu/dataset/Geo-Bench/Gen_results_refine_caa/"

# 检查目录是否存在
if [ ! -d "$target_dir" ]; then
    echo "错误: 指定的目录 '$target_dir' 不存在。"
    exit 1
fi

# 统计子文件夹数量
count=$(find "$target_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "目录 '$target_dir' 下有 $count 个文件夹。"