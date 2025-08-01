import json, argparse

from FID.fid import calculate_fid
from FID.fid_dino import calculate_fid_dino
from FID.fid_kd import calculate_fid_kd
from image_reward import calculate_irs
from human_preference_score import calculate_hps
from VBench.background_consistency import calculate_bgc
from VBench.subject_consistency import calculate_subc
from wrap_error import calculate_we
from MD.mean_distance import calculate_md
import os

degrees = {
    1: {
        "description": ["lightly", "slightly", "gently", "mildly"]
    },
    2: {
        "description": ["moderately", "markedly", "appreciably"]
    },
    3: {
        "description": ["heavily", "intensely", "significantly", "strongly"]
    }
}
def classify_edit_prompt(edit_prompt):
    for level, data in degrees.items():
        for description in data['description']:
            if description in edit_prompt.lower():  # 转为小写匹配
                return level
    raise ValueError(f"No Level found for {edit_prompt}")

def parse_data_level(data, level):
    for da_n, da in data.items():
        ins_data = da['instances']
        for ins_id, ins in ins_data.items():
            pop_item = []
            for case_id, gt_data in ins.items():
                edit_prompt = gt_data.get('edit_prompt', '')
                if classify_edit_prompt(edit_prompt) != level:
                    pop_item.append(case_id)
            for p in pop_item:
                ins.pop(p)
    return data

def parse_data_rotate(data):
    for da_n, da in data.items():
        ins_data = da['instances']
        for ins_id, ins in ins_data.items():
            pop_item = []
            for case_id, gt_data in ins.items():
                edit_param = gt_data.get('edit_param', '')
                if edit_param[5]!=0:
                    pop_item.append(case_id)
            for p in pop_item:
                ins.pop(p)
    return data

def parse_data_3d(data):
    for da_n, da in data.items():
        ins_data = da['instances']
        for ins_id, ins in ins_data.items():
            for case_id, gt_data in ins.items():
                data[da_n]['instances'][ins_id][case_id]['tgt_mask_path'] = gt_data['target_mask_0']
                    
                data[da_n]['instances'][ins_id][case_id]['coarse_input_path'] = gt_data['coarse_input_path_0']
                    
    return data
def make_absolute_path(data, base_dir, gen_img_key):
    for da_n, da in data.items():
        ins_data = da['instances']
        for ins_id, ins in ins_data.items():
            for case_id, gt_data in ins.items():
                path_keys = [
                    "ori_img_path", 
                    "coarse_input_path", 
                    "ori_mask_path", 
                    "tgt_mask_path", 
                    gen_img_key  
                ]
                for key in path_keys:
                    if key in gt_data:
                        gt_data[key] = os.path.join(base_dir, gt_data[key])
                        
    return data
def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path", required=True, help="Path to the input JSON file containing generated results")  
    parser.add_argument("--level", default=0, type=int, help="Edit level to test (0=All, 1=Easy, 2=Medium, 3=Hard)")  
    parser.add_argument("--task", default='100111111', type=str, help="9-digit string to enable metrics (1=compute, 0=skip). Order: FID, IRS, HPS, BGC, SUBC, WRAP_E, MD, FID_DINO, FID_KD")  
    parser.add_argument("--gen_img_key", default="gen_img_path", help="JSON key for generated image paths")
    parser.add_argument("--no_rotate", action='store_true', help="Exclude rotation-related edit cases")  
    parser.add_argument("--3d", action='store_true', help="Use 3D mesh-based masks for 3D evaluation")  
    parser.add_argument("--fid_path", default="/data/Hszhu/dataset/Geo-Bench/source_img_full_v2", help="Path to real images for FID calculation") 
    parser.add_argument("--use_relative_path", action='store_true', help="Convert relative paths to absolute paths using --base_dir")
    parser.add_argument("--base_dir", default="/data/Hszhu/GeoBenchMeta", help="Base directory for relative path conversion (required if --use_relative_path is enabled)")
    args = parser.parse_args()

    file_path = args.path
    image_label = args.gen_img_key
    data = json.load(open(file_path))
    level = args.level
    if level:
        data = parse_data_level(data, level)
    if args.no_rotate:
        data = parse_data_rotate(data)
    if args.3d: 
        data = parse_data_3d(data)   
    
    if args.use_relative_path:
        assert args.base_dir is not None, "--base_dir must be specified when --use_relative_path is enabled"
        data = make_absolute_path(data, args.base_dir, image_label)  

    result = {}
    if int(args.task[0]):
        fid_value = calculate_fid(data, image_label, args.fid_path)
        result['FID'] = fid_value
    if int(args.task[1]):
        irs_value = calculate_irs(data, image_label)
        result['IRS'] = irs_value
    if int(args.task[2]):
        hps_value = calculate_hps(data, image_label)
        result['HPS'] = hps_value
    if int(args.task[3]):
        bgs_value = calculate_bgc(data, image_label)
        result['BGC'] = bgs_value
    if int(args.task[4]):
        subs_value = calculate_subc(data, image_label)
        result['SUBC'] = subs_value
    if int(args.task[5]):
        we_value = calculate_we(data, image_label)
        result['WRAP_E'] = we_value
    if int(args.task[6]):
        md_value = calculate_md(data, image_label)
        result['MD'] = md_value
    if int(args.task[7]):
        dino_value = calculate_fid_dino(data, image_label, args.fid_path)
        result['FID_DINO'] = dino_value
    if int(args.task[8]):
        KD_value = calculate_fid_kd(data, image_label, args.fid_path)
        result['FID_KD'] = KD_value
    
    print("-----Result-----")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
