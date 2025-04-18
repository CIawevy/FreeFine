# FreeFine: Training-Free Diffusion for Geometric Image Editing


![teaser](assets/teaser.png)



**Official Implementation of ICCV 2025 Submission** | [Project Page]() | [arXiv Paper]() | [GeoBench Dataset]()  

---

## 🌟 Introduction  
We present **FreeFine**, a novel framework for high-fidelity geometric image editing that enpowers users with both  **Object-centric Editing**(such as **Object Repositioning, Reorientation, and Reshaping** and **Fine-grained Partial Editing**, all while maintaining global coherence. Remarkably, our framework simultaneously achieves **Structure Completion**, **Object Removal**, **Appearance Transfer**, and **Multi-Image Composition** within a unified pipeline - all through efficient, training-free algorithms based on diffusion models.

Unlike existing diffusion-based editing methods that struggle with large/complex transformations, our approach splits the editing process into object transformation, source region inpainting, and target region refinement, supporting both 2D and 3D transformations.
![Pipeline](assets/pipeline.png)


## 📢 News & Updates  
**2025-07-15**  
🚀 Codebase released with:  
- Pre-trained models for all GeoBench scenarios  
- Jupyter notebook tutorials  
- Windows/Linux compatibility patches  

**2025-06-30**  
🏆 Accepted to **ICCV 2025**! Paper [arXiv link]() now available  

**2025-03-02**  
📊 Submited to **ICCV 2025**!


# 🌟 Extended Applications
Our algorithm’s versatility allows it to tackle diverse tasks beyond its core functionality. Here, we demonstrate its effectiveness in several challenging scenarios.
### **Fine-grained Partial Editing**  
![Fine-grained Partial Editing](assets/Partial-edit-0.png)
![Fine-grained Partial Editing](assets/Partial-edit-2.png)

### **Appearance Transfer**
![APT](assets/Appearance-transfer-2.png)
![APT](assets/Appearance-transfer-0.png)

### **Cross-Image Composition**
![CIC](assets/Cross-image-composition-0.png)
![CIC](assets/Cross-image-composition-2.png)
## 🛠️ Installation  

- Python >= 3.8 , PyTorch >= 2.0.1
```bash
git clone https://github.com/CIawevy/FreeFine.git
cd FreeFine
conda create -n FreeFine python=3.9.19 -y
conda activate FreeFine
pip install -r requirements.txt
```

# ⏬ Download Models 
All models will be automatically downloaded by using diffuser.(todo:sam download) You can also choose to download them locally through the following scripts
```bash
bash scripts/download_models.sh
```
# 🚀 Quick Start 
Run On Web Interface
```bash
python app.py  # Launch Gradio UI  
```
Run On Jupyter Notebooks
```bash
cd jupyter_demo
```
Inference & Eval
```bash
python Eval/inference.py
```

# 📚 Relate Repos
[1] <a href="https://github.com/MC-E/DragonDiffusion>DragonDiffusion">DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models</a>
[examples](examples)</p>
[2] <a href=https://github.com/google/prompt-to-prompt>PROMPT-TO-PROMPT IMAGE EDITING
 WITH CROSS-ATTENTION CONTROL</a>
</p>
[3] <a href=https://github.com/design-edit/DesignEdit>DesignEdit: Unify Spatial-Aware Image Editing via Training-free Inpainting with a Multi-Layered Latent Diffusion Framework</a>
</p>


## 📜 Citation  
```bibtex
@inproceedings{freefine2025,
  title={FreeFine: Training-Free Diffusion for Geometric Image Editing}, 
  author={Your Name and Coauthors},
  booktitle={ICCV}, 
  year={2025}
}


