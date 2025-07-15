
# 激活 Conda 环境
source /home/hszhu/anaconda3/etc/profile.d/conda.sh
conda activate Reggio

cd /data/Hszhu/FreeFine
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

#torchrun --nproc_per_node=8 --master-port $FREE_PORT eval_geobench_batch.py
#torchrun --nproc_per_node=7 --master-port $FREE_PORT FreeFine/freefine_batch_infer_3d_depth.py
torchrun --nproc_per_node=8 --master-port $FREE_PORT Eval/freefine_batch_infer_3d_depth_no_blend.py
#torchrun --nproc_per_node=8 --master-port $FREE_PORT eval_geobench_batch.py