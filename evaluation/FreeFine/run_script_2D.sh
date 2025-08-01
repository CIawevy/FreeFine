
# 激活 Conda 环境
source /home/hszhu/anaconda3/etc/profile.d/conda.sh #replace with your own
conda activate FreeFine

cd /data/Hszhu/FreeFine/evaluation/FreeFine/
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

#For 2D evaluation

#First run batch inference to generate background
torchrun --nproc_per_node=8 --master-port $FREE_PORT freefine_batch_infer_bggen_2d.py  #replace with your own path in the file
#then coarse edit and refinement are all in the following codes
torchrun --nproc_per_node=8 --master-port $FREE_PORT freefine_batch_infer_2d.py   #replace with your own path in the file



