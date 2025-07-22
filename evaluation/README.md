# Evaluation

# **Notice**: The evaluation code is currently under development. We kindly ask for your patience as we work towards its completion.
 ## Data Preparation
 ### GeoBench Dataset
 - **Download**: You can download the GeoBench dataset from the official source [provide the official download link here if available]. Alternatively, you can quickly access it through the Hugging Face dataset. Use the following code snippet to load the dataset:
 ```python
 # from datasets import load_dataset
 # dataset = load_dataset('your-geo-bench-dataset-name-on-huggingface')
 ```
 - **Data Structure**: The GeoBench dataset has the following directory structure:
 ```
 # Geo-Bench/
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