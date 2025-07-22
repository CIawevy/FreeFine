import json
def load_json(file_path):
    """
    加载指定路径的JSON文件并返回数据。

    :param file_path: JSON文件的路径
    :return: 从JSON文件中加载的数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        print(f"文件格式错误: {file_path}")
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
    return None

def save_json(data_dict, file_path):
    """
    将字典保存为 JSON 文件

    Args:
        data_dict (dict): 需要保存的字典
        file_path (str): JSON 文件的保存路径
    """
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)

def merge_json(json_list):
    data_dict = dict()
    for json_file in json_list:
        i = 0
        da = load_json(json_file)
        for  k,v in da.items():
            data_dict[k] = v
        print(f'merge json name {json_file} , length {i}')
    save_json(data_dict,"/data/Hszhu/dataset/PIE-Bench_v1/coarse_input_full_pack.json")

# json_path_list = [ f"/data/Hszhu/dataset/PIE-Bench_v1/packed_data_full_EXP_INP_{i-1}.json" for i in range(1,9)]
json_path_list = [ f"/data/Hszhu/dataset/PIE-Bench_v1/coarse_input_full_pack_{i}.json" for i in range(3)]

merge_json = merge_json(json_path_list)
data = load_json("/data/Hszhu/dataset/PIE-Bench_v1/coarse_input_full_pack.json")
keys = data.keys()

valid_id = [k for k,v in data.items()]
print(f'full length:{len(valid_id)}')

