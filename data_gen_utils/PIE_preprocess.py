import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from data_gen_utils.data import PIE_data_preprocessor
PIE_BASE_DIR = "/data/Hszhu/dataset/PIE-Bench_v1/"
PIE_data_preprocessor(PIE_BASE_DIR)
print('load ok')
