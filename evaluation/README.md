# Evaluation
 ## Data Preparation
 ### GeoBench Dataset
- **Download**: We release two versions of our GeoBench dataset:
  - **[GeoBench](https://huggingface.co/datasets/CIawevy/GeoBench)**: This version is in Parquet format and is more convenient for data preview and loading.

  - **[GeoBenchMeta](https://huggingface.co/datasets/CIawevy/GeoBenchMeta)**: This version is currently more recommended as it aligns with our evaluation codebase.

- **Run the following command to download the `GeoBenchMeta` dataset with `huggingface_hub`**:
```bash
 bash /mnt/bn/ocr-doc-nas/zhuhanshen/iccv/FreeFine/scripts/download_dataset.sh
```
- **Run the following command to download the `GeoBenchMeta` dataset with `aria2` and `hfd.sh`**:
```bash
chmod a+x FreeFine/scripts/hfd.sh
sudo apt update
sudo apt install aria2
bash FreeFine/scripts/download_dataset_hfd.sh
```
- **Easy loading `GeoBench`**
```bash
from datasets import load_dataset
dataset = load_dataset("CIawevy/GeoBench")
 ```
 - **Data Structure**: The GeoBenchMeta dataset has the following directory structure:
 ```
  GeoBenchMeta/
  ├── annotation_2d.json            # 2D task annotation file
  ├── annotation_3d.json            # 3D task annotation file
  ├── annotation_sc.json            # Structure Completion task annotation file
  ├── Geo-Bench-2D/            
  │   ├── coarse_img/          
  │   ├── inp_mask_vis/        
  │   ├── source_img/          
  │   ├── source_img_full_v2/       # Path to real images for FID calculation.
  │   ├── source_mask/         
  │   └── target_mask/          
  ├── Geo-Bench-3D/           
  │   ├── coarse3d_depth_anything/  # 3D depth-based coarse edit results
  │   ├── coarse_img_sv3d/          # SV3D-based coarse edit results
  │   ├── correspondence/           # 3D Correspondence-map for Mean-Distance metric
  │   ├── inp_mask_vis/        
  │   ├── md_mask/                  # auto draw_mask for depth-based edit
  │   ├── mesh_mask/                # target_mask for depth-based edit
  │   ├── source_img/          
  │   ├── source_mask/         
  │   └── target_mask/              # target_mask for SV3D-based edit
  └── Geo-Bench-SC/            
        ├── coarse_img/        
        ├── draw_mask/              # manually draw mask for the missing part
        ├── draw_mask_vis/     
        ├── inp_mask_vis/      
        ├── source_img/        
        ├── source_mask/       
        └── target_mask/       
 ```
  - **Usage**: The annotation files (e.g., `annotation_2d.json`) contain metadata such as edit prompts, edit parameters, image paths, mask paths... These metadata are used in the inference process of baselines.


 ## Baselines
 To use the baselines, follow these steps:
 1. Navigate to the subfolders of each method.
 2. Install the required environment by following the instructions provided in each subfolder.
 3. For inference code, refer to the `run_script.sh` file in each subfolder.

 ## Evaluation
 ### 1. Install Evaluation Environment
```bash
conda create -n metric python==3.11.11
conda activate metric
cd FreeFine/evaluation/metrics/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# If you are missing the bpe_simple_vocab_16e6.txt.gz file, please download it from the following link.
wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
```
 ### 2. Evaluate Results
 After performing inference and saving the generated results in JSON format, run the following steps to evaluate the results:
 ```bash
cd FreeFine/evaluation/metrics/

# For 2D Evaluation
python main.py \
    --path generate_results_2d_freefine.json \
    --use_relative_path \
    --base_dir /data/Hszhu/GeoBenchMeta

# For 3D Evaluation
python main.py \
    --path generate_results_3d_freefine.json \
    --3d \
    --use_relative_path \
    --base_dir /data/Hszhu/GeoBenchMeta
 ```

 <!-- ### Parameter Explanation
  - **`--path`** (required): Path to the input data JSON file (required), pointing to the JSON file containing generated results (e.g., `GeoBenchMeta/Geo-Bench-2D/generated_results_freefine.json`).
  - **`--level`** (default=0): Editing intensity level to test (integer), used to filter cases with specific editing strengths (1: lightly, 2: moderately, 3: heavily; default 0 means no filtering, i.e., all cases).
  - **`--task`** (default='100111111'): Flag string to control which metrics to compute (9-digit string, where each digit corresponds to a metric: 1 for compute, 0 for skip). Order: `[FID, IRS, HPS, BGC, SUBC, WRAP_E, MD, FID_DINO, FID_KD]`.
  - **`--image_label`** (default="gen_img_path"): Key name of the generated image path in the JSON file (e.g., if the generated image path is stored in the `"generated_image"` field, set to `--image_label generated_image`).
  - **`--no_rotate`** (flag): Whether to exclude rotation cases (adding this parameter indicates excluding rotation-edited cases).
  - **`--mesh`** (flag): Whether to use mesh masks (adding this parameter replaces `target_mask` with `mesh_mask` and adjusts the `coarse_input_path`).
  - **`--fid_path`** (default="GeoBenchMeta/Geo-Bench-2D/source_img_full_v2"): Path to real images used for FID calculation (default points to the full original image directory of GeoBench). -->
## Parameter Explanation

| Argument              | Default                          | Description                                                                 |
|-----------------------|----------------------------------|-----------------------------------------------------------------------------|
| `path`              | (required)                       | Path to the input JSON file containing generated results (e.g., `GeoBenchMeta/Geo-Bench-2D/generated_results_2d_freefine.json`). |
| `level`             | `0`                              | Edit level to test (0=All, 1=Easy, 2=Medium, 3=Hard).                       |
| `task`              | `'100111111'`                    | 9-digit string to enable metrics (1=compute, 0=skip). Order: FID, IRS, HPS, BGC, SUBC, WRAP_E, MD, FID_DINO, FID_KD. |
| `gen_img_key`       | `"gen_img_path"`                 | JSON key where generated image paths are stored. |
| `3d`                | (flag)                           | Use 3D mesh-based masks for 3D evaluation (replaces `target_mask` with `mesh_mask` and adjusts `coarse_input_path`). |
| `fid_path`          | `/data/Hszhu/GeoBenchMeta/Geo-Bench-2D/source_img_full_v2` | Path to real images for FID calculation. |
| `use_relative_path` | (flag)                           | Convert relative paths in the JSON to absolute paths using `--base_dir`.   |
| `base_dir`          | `/data/Hszhu/GeoBenchMeta`       | Base directory for relative path conversion (required if `--use_relative_path` is enabled). |