
# 📊 2D Eval 
## Steps to Run 2D Evaluation

### Method 1: Running Commands Manually
Before starting, ensure you have modified all relevant paths (e.g., checkpoint paths) in the Python scripts to your local paths.

- **Step 1: Batch inference to generate background**
```bash
conda activate FreeFine
cd FreeFine/evaluation/FreeFine/
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node=8 --master-port $FREE_PORT freefine_batch_infer_bggen_2d.py
```
- **Step 2: batch inference to coarsely edit and re-generate**
```bash
torchrun --nproc_per_node=8 --master-port $FREE_PORT freefine_batch_infer_2d.py
```

### Method 2: Using the Script
Before running the script, make sure to modify all relevant paths in `run_script_2D.sh` to your local paths. Then execute the following command:
```bash
bash run_script_2D.sh
```

# 🧊 3D Eval
## Steps to Run 3D Evaluation

### Default Step: Run batch inference for final 3D refinement
You can directly run this step using the pre-generated data in the `Geo-Bench-3D` folder to save time and resources.
```bash
conda activate FreeFine
cd FreeFine/evaluation/FreeFine/
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node=8 --master-port $FREE_PORT Eval/freefine_batch_infer_3d_depth.py
```
### Reproduce the full process (Optional)
If you wish to reproduce the entire process from scratch, follow these additional steps before the default step.
- **Step 1: Batch inference to generate background**
```bash
conda activate FreeFine
cd FreeFine/evaluation/FreeFine/
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc_per_node=8 --master-port $FREE_PORT freefine_batch_infer_bggen_3d.py
```
- **Step 2: Run depth-based 3D transform** :
This step is modified from Geodiffuser's code base. We use Depth Anything and PyTorch3D to estimate depth and directly transform the image based on that. After this step, we will get a coarse_edit_3d_img (which is sparse and troublesome) and the correspondence map for Mean - Distance (MD) calculation.
```bash
conda activate GeoDiffuser
python get_3d_transform_correspondence.py # Replace with your own checkpoint path and other parameters in the file
```
- **Step 3: batch inference to re-generate**
```bash
torchrun --nproc_per_node=8 --master-port $FREE_PORT freefine_batch_infer_2d.py
```



 ## 🛠️ Installation  
```bash
git clone https://github.com/CIawevy/FreeFine.git
cd FreeFine
conda create -n FreeFine python=3.10.13 -y
conda activate FreeFine
pip install -r requirements.txt 
```
- Install [Pytorch3D](https://github.com/facebookresearch/pytorch3d) (Optional for depth-based 3D-editing) 
```
pip install iopath>=0.1.10 -i https://pypi.org/simple
pip install --no-index --no-cache-dir git+https://github.com/facebookresearch/pytorch3d.git@stable -i https://pypi.org/simple
```

- Install [SV3D](https://github.com/Stability-AI/generative-models) (Optional for SV3D-based 3D-editing)  

```bash
# Set up SV3D environment 
cd generative-models 
conda create -n pt2 python=3.10.0 -y
conda activate pt2
pip3 install -r requirements/pt2.txt
pip3 install . #Install sgm
```


# ⏬ Download Models   
- Modify parameters (e.g., local paths, HF token) in `scripts/download_models.sh` as needed, then run the following command to download models:
```bash
bash scripts/download_models.sh
```




