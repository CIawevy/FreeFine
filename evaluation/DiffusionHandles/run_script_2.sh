
# 激活 Conda 环境
source /home/hszhu/anaconda3/etc/profile.d/conda.sh
conda activate diffusionhandles
export CUDA_VISIBLE_DEVICES=2
cd /data/Hszhu/DiffusionHandles/
#FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

#torchrun --nproc_per_node=8 --master-port $FREE_PORT eval_geobench_batch.py

python eval_geobench3d.py  --data_id 2
python eval_geobench.py  --data_id 2
#torchrun --nproc_per_node=8 --master-port $FREE_PORT eval_geobench_batch.py