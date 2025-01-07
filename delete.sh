#!/bin/bash

# 固定的环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

# 基础目录
#base_dir="/data/Hszhu/dataset/PIE-Bench_v1/"
base_dir="/data/Hszhu/dataset/Subjects200K/"
#base_dir="/data/Hszhu/dataset/GRIT/"

# droprate 和 output directory 的配置
declare -a idx=(0 1 2 3 4 5 6 7)  # 注意这里数组的定义，不要加逗号
for id in "${idx[@]}"; do
    delete_dir="${base_dir}Subset_${id}"  # 正确的字符串拼接
    rm -rf "${delete_dir}/mask_tag_relabelled_lmm_v2_${id}.json"
done

## 删除其他指定文件和目录
#rm -rf "${base_dir}progress_*"
#rm -rf "${base_dir}users"
#rm -rf "/data/Hszhu/dataset/sessions"
