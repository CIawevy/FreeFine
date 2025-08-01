# Evaluation

# **Notice**: Note: Code is still in maintenance...
 ## Data Preparation
 ### GeoBench Dataset
- **Download**: We release two versions of our GeoBench dataset:
  - **[GeoBench](https://huggingface.co/datasets/CIawevy/GeoBench)**: This version is in Parquet format and is more convenient for data preview and loading.

  - **[GeoBenchMeta](https://huggingface.co/datasets/CIawevy/GeoBenchMeta)**: This version is currently more recommended as it aligns with our evaluation codebase.

- **Run the following command to download the `GeoBenchMeta` dataset with `huggingface-hub`**:
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
 # GeoBench/Meta
 # ├── annotations.json
 # ├── source_img/
 # │   ├── *.png
 # ├── source_mask/
 # │   ├── */
 # │       ├── *.png
 # ├── target_mask/
 # │   ├── */
 # │   ├── */
 # │       ├── *.png
 # ├── coarse_img/
 # │   ├── */
 # │   ├── */
 # │       ├── *.png
 ```
 - **Usage**: The `annotations.json` file contains metadata about the images, including prompts, edit parameters, and paths to source and target masks. You can use this information to perform various editing tasks.

 ## Baselines
 To use the baselines, follow these steps:
 1. Navigate to the subfolders of each method.
 2. Install the required environment by following the instructions provided in each subfolder.
 3. For inference code, refer to the `run_script.sh` file in each subfolder.

 ## Evaluation
 ### 1. Install Evaluation Environment
 ```bash
 # # Example installation commands, modify according to your actual requirements
 # conda create -n eval_env python=3.8
 # conda activate eval_env
 # pip install -r requirements.txt
 ```
 ### 2. Evaluate Results
 After performing inference and saving the generated results in JSON format, follow these steps to evaluate the results:
 1. Navigate to the `metric` subfolder.
 2. Run the following command:
 ```bash
 # python main.py
 ```

 ### Parameter Explanation
 - **`--model`**: Path to the model checkpoint file. Used in the flow calculation process to specify which pre - trained model to load.
 - **`--path`**: Path to the dataset used for evaluation. This parameter is required when evaluating the performance of the model on a specific dataset.
 - **`--small`**: A flag to indicate whether to use a small model. If set, a smaller and more lightweight model will be used.
 - **`--mixed_precision`**: A flag to enable mixed - precision training or inference. This can significantly reduce memory usage and speed up the process.
 - **`--alternate_corr`**: A flag to use an efficient correlation implementation. This can improve the computational efficiency of the model.