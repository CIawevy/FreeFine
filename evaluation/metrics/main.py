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

def parse_data_mesh(data):
    for da_n, da in data.items():
        ins_data = da['instances']
        for ins_id, ins in ins_data.items():
            for case_id, gt_data in ins.items():
                data[da_n]['instances'][ins_id][case_id]['tgt_mask_path'] = \
                    gt_data['tgt_mask_path'].replace("target_mask", "mesh_mask")
                data[da_n]['instances'][ins_id][case_id]['coarse_input_path'] = \
                    gt_data['coarse_input_path'].replace("coarse_img", "coarse3d_depth_anything")
    return data

def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--path", required=True, help="输入数据json的路径")
    parser.add_argument("--level", default=0, type=int, help="待测试的编辑程度")
    parser.add_argument("--task", default='100111111', type=str, help="具体计算哪几个指标")
    parser.add_argument("--image_label", default="gen_img_path", help="生成图片路径对应的key")
    parser.add_argument("--no_rotate", action='store_true', help="是否排除掉旋转")
    parser.add_argument("--mesh", action='store_true', help="是否使用mesh的mask")
    parser.add_argument("--fid_path", default="/data/Hszhu/dataset/Geo-Bench/source_img_full_v2", help="计算FID时的真实图片路径")
    args = parser.parse_args()

    file_path = args.path
    image_label = args.image_label
    data = json.load(open(file_path))
    level = args.level
    if level:
        data = parse_data_level(data, level)
    if args.no_rotate:
        data = parse_data_rotate(data)
    if args.mesh:
        data = parse_data_mesh(data)

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
