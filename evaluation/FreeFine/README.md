
# 📊 2D Eval 
 We provide the scripts for evaluating GeoBench-2d and GeoBench-3d for FreeFine and all the Baselines. Please See [EVAL](./evaluation/README.md) for more details.

# 📊 3D Eval 
 We provide the scripts for evaluating GeoBench-2d and GeoBench-3d for FreeFine and all the Baselines. Please See [EVAL](./evaluation/README.md) for more details.

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




