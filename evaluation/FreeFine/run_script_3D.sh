
# 激活 Conda 环境
source /home/hszhu/anaconda3/etc/profile.d/conda.sh
conda activate FreeFine

cd /data/Hszhu/FreeFine/evaluation/FreeFine/
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

#For 3D evaluation

#Step 1: run batch inference to generate background
torchrun --nproc_per_node=8 --master-port $FREE_PORT freefine_batch_infer_bggen_3d.py  #replace with your own path in the file

#Step2 run depth-based 3D transform where you got corase_edit image and 3d_correspondence for MD metric calculation
conda deactivate 
conda activate GeoDiffuser
python get_3d_transform_correspondence.py

#Step 3: run batch inference to refine the final outputs
torchrun --nproc_per_node=8 --master-port $FREE_PORT Eval/freefine_batch_infer_3d_depth.py #3d