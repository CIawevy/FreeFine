source /home/hszhu/anaconda3/etc/profile.d/conda.sh

conda activate DesignEdit

cd /data/Hszhu/DesignEdit/


torchrun --nproc_per_node=6 geobench_eval.py