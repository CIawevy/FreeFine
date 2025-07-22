
# 激活 Conda 环境
source /home/hszhu/anaconda3/etc/profile.d/conda.sh
conda activate diffusionhandles
export CUDA_VISIBLE_DEVICES=4
cd /data/Hszhu/DiffusionHandles/

python eval_geobench3d.py  --data_id 0
python eval_geobench.py  --data_id 0

#after inference all the data splits, run the following command to merge json data
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node=8 --master-port $FREE_PORT collect_design.py


#inference time testing
# python test_time.py  --data_id 0

