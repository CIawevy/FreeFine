import torch
import pickle
from util.io import read_pickle, complete_path, create_folder
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def chunk_attention_by_layer(attention_dict):

    layer_dict = {}

    for k in attention_dict:
        if not k.startswith("length_"):
            if len(attention_dict[k]) > 0:
                num_attention_layers = attention_dict["length_" + k]
                layer_dict[k] = {}
                for a_idx in range(num_attention_layers):
                    layer_dict[k][a_idx] = attention_dict[k][a_idx::num_attention_layers]
                    print(len(attention_dict[k][a_idx::num_attention_layers]))

    print(layer_dict.keys())
    return layer_dict


def prepare_attention_for_visualization(layer_dict, from_where, is_cross, select, res=16, max_com=10, im_size=256):
    location = f"{from_where}_{'cross' if is_cross else 'self'}"

    out_dict = {}
    out_dict[location] = []
    for l_idx in layer_dict[location]:
        time_maps = []
        # print(len(layer_dict[location][l_idx]))
        for a_idx in range(len(layer_dict[location][l_idx])):
            # print(a_idx)
            images = []
            # print(layer_dict[location][l_idx])
            attn_map = layer_dict[location][l_idx][a_idx][select]
            res = int(np.sqrt(attn_map.shape[-1]))
            u, s, vh = np.linalg.svd((attn_map - np.mean(attn_map, axis=0, keepdims=True)).T)
            for i in range(max_com):
                image = vh[i].reshape(res, res)
                image = 255 * image / image.max()
                image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
                image = Image.fromarray(image).resize((im_size, im_size))
                image = np.array(image)
                images.append(image)

                # print(image.shape, image.max(), image.min(), image.mean())
            time_maps.append(images)

        out_dict[location].append(time_maps)
        # break
        

    return out_dict

def prepare_cross_attention_for_visualization(layer_dict):
    
    return

def save_maps(att_dict, save_path):

    p = complete_path(save_path)
    create_folder(save_path)
    for k in att_dict:
        
        for layer in range(len(att_dict[k])):
            for t in range(len(att_dict[k][layer])):
                for comp in range(len(att_dict[k][layer][t])):
                    d_path = p + k + "/" + str(layer) + "/" + str(comp) + "/"
                    print(d_path, t)
                    create_folder(d_path)
                    # Image.fromarray(att_dict[k][layer][t][comp]).save(d_path + "%d.png" % t)
                    # print(att_dict[k][layer][t][comp].shape)
                    plt.imsave(d_path + "%d.png" % t, att_dict[k][layer][t][comp][..., 0], cmap="viridis")

def save_attention_maps(attention_dict, save_path, from_where="up", is_cross=False, select=0):
    
    layer_dict = chunk_attention_by_layer(attention_dict)
    out_dict = prepare_attention_for_visualization(layer_dict, from_where, is_cross, select)

    save_maps(out_dict, save_path)

    pass


def parse_args():

    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--exp_root')
    parser.add_argument("--save", default="./utils/attention_maps")
    args = parser.parse_args()
    return args
    
    
if __name__ == "__main__":

    args = parse_args()
    attn_dict = read_pickle(complete_path(args.exp_root) + "attention.pkl")
    save_attention_maps(attn_dict, args.save)
    


    pass