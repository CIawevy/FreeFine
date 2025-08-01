
# **Notice**: 
Note: Code is still in maintenance. Releasing initial version.



# FreeFine: Training-Free Diffusion for Geometric Image Editing
<p align="left">
  <a href="https://github.com/CIawevy/FreeFine">
    <img
      src="https://img.shields.io/badge/FreeFine-Project%20Page-0A66C2?logo=safari&logoColor=white"
      alt="FreeFine Project Page"
    />
  </a>
  <a href="https://www.arxiv.org/abs/2507.23300">
    <img
      src="https://img.shields.io/badge/FreeFine-Paper-red?logo=arxiv&logoColor=red"
      alt="FreeFine Paper on arXiv"
    />
  </a>
  <a href="">
    <img
      src="https://img.shields.io/badge/FreeFine-Demo-blue?logo=googleplay&logoColor=blue"
      alt="FreeFine Demo"
    />
  </a>
  <a href="https://huggingface.co/datasets/CIawevy/GeoBench">
    <img 
        src="https://img.shields.io/badge/GeoBench-Dataset-orange?logo=huggingface&logoColor=yellow" 
        alt="GeoBench"
    />
  </a>
   <a href="https://huggingface.co/datasets/CIawevy/GeoBenchMeta">
    <img 
        src="https://img.shields.io/badge/GeoBenchMeta-Dataset-orange?logo=huggingface&logoColor=yellow" 
        alt="GeoBenchMeta"
    />
  </a>
</p>


![teaser](assets/teaser.png)




**Official Implementation of ICCV 2025 Accepted Paper** 
<!-- | [Project Page](https://github.com/CIawevy/FreeFine) | [arXiv Paper](https://arxiv.org/pdf/2507.23300) | [GeoBench Dataset](https://huggingface.co/datasets/CIawevy/GeoBench)  | [GeoBenchMeta Dataset](https://huggingface.co/datasets/CIawevy/GeoBenchMeta)   -->




## 🌟 Introduction  
>We present **FreeFine**, a novel framework for high-fidelity geometric image editing that enpowers users with both  **Object-centric Editing**(such as **Object Repositioning, Reorientation, and Reshaping** and **Fine-grained Partial Editing**, all while maintaining global coherence. Remarkably, our framework simultaneously achieves **Structure Completion**, **Object Removal**, **Appearance Transfer**, and **Multi-Image Composition** within a unified pipeline - all through efficient, training-free algorithms based on diffusion models.
>
>![Pipeline](assets/pipeline.png)


## 🔥 News
- **2025-08-10**: 🚀 **Scheduled Full Project Open-Source Release**  
  We’re gearing up to release the entire FreeFine ecosystem, with these key components currently in development:  
  - 📊 GeoBench benchmark dataset (2D/3D geometric editing scenarios)  
  - 📈 Evaluation code for quantitative performance testing  
  - ⚙️ Complete inference codebase for end-to-end editing pipelines  
  - 📓 Interactive Jupyter notebook demos (step-by-step tutorials)  
  - 🖥️ User-friendly Gradio interface (no-code visual editing)
- **2025-07-31**: Our [Arxiv Paper](https://www.arxiv.org/abs/2507.23300) is now available!  
- **2025-06-26**: 🎉FreeFine has been accepted to **ICCV 2025**! 🎉

## 📌 TODO
- [x] Arxiv Paper
- [x] Release Code
- [x] GeoBench benchmark dataset 
- [x] Evaluation code platform
- [ ] Jupyter notebook demos  
- [ ] Gradio interface demos
- [ ] Adapt to stronger baselines such as SDXL and DIT
  
  
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

# 📊 Eval
 We provide the scripts for evaluating GeoBench-2d and GeoBench-3d for FreeFine and all the Baselines. Please See [EVAL](./evaluation/README.md) for more details.




# 🚀 Quick Start 
## Run on Jupyter Notebooks
```bash
cd jupyter_demo
```
## Run On Web Interface 🚧
 ⚠️ The web interface is currently under construction. Once ready, start it with:
 ```bash
python app.py  
 ```



# 📚 Relate Repos
[1] <a href="https://github.com/MC-E/DragonDiffusion>DragonDiffusion">**DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models**</a></p>
[2] <a href=https://github.com/google/prompt-to-prompt>**PROMPT-TO-PROMPT IMAGE EDITING WITH CROSS-ATTENTION CONTROL**</a></p>
[3] <a href=https://github.com/Stability-AI/generative-models>**SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion**</a></p>
[4] <a href=https://github.com/LiheYoung/Depth-Anything>**Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**</a></p>


## 📜 Citation  
```bibtex
@inproceedings{freefine2025,
  title={Training-Free Diffusion for Geometric Image Editing}, 
  author={Zhu, Hanshen and Zhu, Zhen and Zhang, Kaile and Gong, Yiming and Liu, Yuliang and Bai, Xiang},
  booktitle={ICCV}, 
  year={2025}
}
