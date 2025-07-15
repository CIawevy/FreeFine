# FreeFine: Training-Free Diffusion for Geometric Image Editing


![teaser](assets/teaser.png)



**Official Implementation of ICCV 2025 Accepted Paper** | [Project Page](https://github.com/CIawevy/FreeFine) | [arXiv Paper]() | [GeoBench Dataset]()  

---

## 🌟 Introduction  
We present **FreeFine**, a novel framework for high-fidelity geometric image editing that enpowers users with both  **Object-centric Editing**(such as **Object Repositioning, Reorientation, and Reshaping** and **Fine-grained Partial Editing**, all while maintaining global coherence. Remarkably, our framework simultaneously achieves **Structure Completion**, **Object Removal**, **Appearance Transfer**, and **Multi-Image Composition** within a unified pipeline - all through efficient, training-free algorithms based on diffusion models.

Unlike existing diffusion-based editing methods that struggle with large/complex transformations, our approach splits the editing process into object transformation, source region inpainting, and target region refinement, supporting both 2D and 3D transformations.
![Pipeline](assets/pipeline.png)


## 📢 News & Updates  

**2025-07-23**  
📜 Paper [arXiv link](https://arxiv.org/abs/xxxx.xxxx) now available  


**2025-07-15**  
🚀 **Full Project Open-Source Release**  
We're thrilled to make the entire FreeFine ecosystem publicly available, including:  
- 📊 GeoBench benchmark dataset (2D/3D geometric editing scenarios)  
- 📈 Evaluation code for quantitative performance testing  
- ⚙️ Complete inference codebase for end-to-end editing pipelines  
- 📓 Interactive Jupyter notebook demos (step-by-step tutorials)  
- 🖥️ User-friendly Gradio interface (no-code visual editing)  


**2025-06-26**  
🏆 FreeFine accepted to **ICCV 2025**!  



[//]: # (# 🌟 Extended Applications)

[//]: # (Our algorithm’s versatility allows it to tackle diverse tasks beyond its core functionality. Here, we demonstrate its effectiveness in several challenging scenarios.)

[//]: # (### **Fine-grained Partial Editing**  )

[//]: # (![Fine-grained Partial Editing]&#40;assets/Partial-edit-0.png&#41;)

[//]: # (![Fine-grained Partial Editing]&#40;assets/Partial-edit-2.png&#41;)

[//]: # ()
[//]: # (### **Appearance Transfer**)

[//]: # (![APT]&#40;assets/Appearance-transfer-2.png&#41;)

[//]: # (![APT]&#40;assets/Appearance-transfer-0.png&#41;)

[//]: # ()
[//]: # (### **Cross-Image Composition**)

[//]: # (![CIC]&#40;assets/Cross-image-composition-0.png&#41;)

[//]: # (![CIC]&#40;assets/Cross-image-composition-2.png&#41;)
## 🛠️ Installation  

- Python >= 3.8 , PyTorch >= 2.0.1
```bash
git clone https://github.com/CIawevy/FreeFine.git
cd FreeFine
conda create -n FreeFine python=3.9.19 -y
conda activate FreeFine
pip install -r requirements.txt
```
- Install [SV3D](https://github.com/Stability-AI/generative-models) for 3D-editing 
```bash
# Set up SV3D environment (separate from main project)
cd generative-models 
conda create -n pt2 python=3.10.0 -y
conda activate pt2
pip3 install -r requirements/pt2.txt
pip3 install . #Install sgm
```
- Evaluation
```bash
cd generative-models 
conda create -n pt2 python=3.10.0 -y
conda activate pt2
pip3 install -r requirements/pt2.txt
pip3 install . #Install sgm
```
# ⏬ Download Models 
- Models will automatically download via Hugging Face's `diffusers` on first run. For offline use or custom installations:
```bash
bash scripts/download_models.sh
```
- You should download SV3D models from [SV3D Hugging Face Repository](https://huggingface.co/stabilityai/sv3d/tree/main)
```bash
#move them to the correct location after download
cd FreeFine/generative-models
mkdir -p checkpoints  # Creates directory if missing
mv /path/to/downloaded/sv3d/* checkpoints/
```



# 🚀 Quick Start 
Run On Web Interface
```bash
python app.py  # Launch Gradio UI  
```
Run on Jupyter Notebooks
```bash
cd jupyter_demo
```
Inference & Eval
```bash
python FreeFine/inference.py
```

# 📚 Relate Repos
[1] <a href="https://github.com/MC-E/DragonDiffusion>DragonDiffusion">DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models</a></p>
[2] <a href=https://github.com/google/prompt-to-prompt>PROMPT-TO-PROMPT IMAGE EDITING WITH CROSS-ATTENTION CONTROL</a></p>
[3] <a href=https://github.com/Stability-AI/generative-models>** SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion **</a></p>


## 📜 Citation  
```bibtex
@inproceedings{freefine2025,
  title={FreeFine: Training-Free Diffusion for Geometric Image Editing}, 
  author={Your Name and Coauthors},
  booktitle={ICCV}, 
  year={2025}
}


