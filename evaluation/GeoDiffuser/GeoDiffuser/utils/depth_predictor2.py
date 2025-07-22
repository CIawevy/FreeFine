from GeoDiffuser.utils.editor import *
# from geometry_editor import *
import os
from GeoDiffuser.utils import vis_utils
import GeoDiffuser.utils.generic as io
import matplotlib.pyplot as plt
from GeoDiffuser.segment_anything import SamPredictor, sam_model_registry
from GeoDiffuser.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from GeoDiffuser.depth_anything.dpt import DepthAnything


"""Compute depth maps for images in the input folder.
"""
import glob
import torch
import cv2
import argparse
import numpy as np
from PIL import Image

# from geometry

from GeoDiffuser.utils.generic import load_256

from torchvision.transforms import Compose

from GeoDiffuser.dpt.models import DPTDepthModel
from GeoDiffuser.dpt.midas_net import MidasNet_large
from GeoDiffuser.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter

from GeoDiffuser.zoedepth.models.builder import build_model
from GeoDiffuser.zoedepth.utils.config import get_config


SAM_MODEL = None
DEPTH_ANYTHING_MODEL = None
ZOEDEPTH_MODEL = None

def depth_corrector(depth, filter_type = "gaussian"):
    

    if filter_type == "median":
        # print(depth.shape)
        depth_corrected = medfilt(depth)
    elif filter_type == "gaussian":
        depth_corrected = gaussian_filter(depth, sigma = 0.75)

    return depth_corrected


def get_mask_prediction_multiple_points(image, input_points, input_labels, model_path = "/home/ec2-user/SageMaker/test_sd/segment-anything/weights/sam_vit_h_4b8939.pth"):

    global SAM_MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if SAM_MODEL is None:
        sam = sam_model_registry["vit_h"](checkpoint=model_path).to(device)
        SAM_MODEL = sam
    else:
        sam = SAM_MODEL
    predictor = SamPredictor(sam)
    


    predictor.set_image(image)


    # input_points = np.array([[image.shape[1] * w, image.shape[0] * h]])
    # input_label = np.array([1])
    masks, _, _ = predictor.predict(input_points, input_labels)

    image_mask = masks[-1] * 1.0

    return image_mask

def get_mask_prediction(image, h, w, model_path = "/home/ec2-user/SageMaker/test_sd/segment-anything/weights/sam_vit_h_4b8939.pth"):

    global SAM_MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if SAM_MODEL is None:
        sam = sam_model_registry["vit_h"](checkpoint=model_path).to(device)
        SAM_MODEL = sam
    else:
        sam = SAM_MODEL
    predictor = SamPredictor(sam)
    


    predictor.set_image(image)
    h = float(h)
    w = float(w)

    input_points = np.array([[image.shape[1] * w, image.shape[0] * h]])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(input_points, input_label)

    image_mask = masks[-1] * 1.0

    return image_mask


def get_monocular_depth(image, model_path, model_type="dpt_large", optimize=True, translate_factor = 0.0):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # # get input
    # img_names = glob.glob(os.path.join(input_path, "*"))
    # # print(img_names)
    # num_images = len(img_names)

    # # create output folder
    # os.makedirs(output_path, exist_ok=True)

    img = image

    print("start processing")
    # for ind, img_name in enumerate(img_names):
    #     if os.path.isdir(img_name):
    #         continue

    #     print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
    #     # input

    #     img = util.io.read_image(img_name)

    #     if args.kitti_crop is True:
    #         height, width, _ = img.shape
    #         top = height - 352
    #         left = (width - 1216) // 2
    #         img = img[top : top + 352, left : left + 1216, :]

    img_input = transform({"image": image})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        if model_type == "dpt_hybrid_kitti":
            prediction *= 256

        if model_type == "dpt_hybrid_nyu":
            prediction *= 1000.0

    # filename = os.path.join(
    #     output_path, os.path.splitext(os.path.basename(img_name))[0]
    # )
    # util.io.write_depth(filename, prediction, bits=2, absolute_depth=args.absolute_depth)

    print("finished")
    # return io.process_depth(prediction)

    depth = prediction

    # # Converts from relative to absolute depth. This works better than using 1/depth
    # depth = depth.max() - depth
    
    # Pushes object farther off to reduce smearing
    depth = depth + depth.max() * translate_factor
    return prediction


def get_monocular_ZoeDepth(image, depth_model = "ZoeD_NK"):

    h, w, _ = image.shape

    global ZOEDEPTH_MODEL


    device = "cuda" if torch.cuda.is_available() else "cpu"
    pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
    if ZOEDEPTH_MODEL is None:
        conf = get_config("zoedepth_nk", "infer")
        conf["pretrained_resource"] = None
        # exit()
        model_zoe_nk = build_model(conf)

        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
        # repo = "isl-org/ZoeDepth"
        # model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=False)

        pretrained_dict = torch.hub.load_state_dict_from_url('https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt', map_location='cpu')
        model_zoe_nk.load_state_dict(pretrained_dict['model'], strict=False)
        for b in model_zoe_nk.core.core.pretrained.model.blocks:
            b.drop_path = torch.nn.Identity()

        # model_zoe_nk = model_zoe_nk.to("cuda")

        # pil_image = Image.from_numpy()


        zoe = model_zoe_nk.to(device)
        ZOEDEPTH_MODEL = zoe
    else:
        zoe = ZOEDEPTH_MODEL

    with torch.no_grad():
        depth = zoe.infer_pil(image, output_type="tensor")#.detach().cpu().numpy() 

    # print(depth.shape, h, w)

    # depth = torch.nn.functional.interpolate(depth[None, None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    
    # depth = depth.max() - depth
    return depth.detach().cpu().numpy()
    
def get_constant_depth(image):
    return np.ones_like(image[..., 0]) * 0.5

@torch.no_grad()
def get_monocular_depth_anything(image, encoder="vitl", translate_factor=0.1 ,local_rank=None):
    DEVICE = 'cuda'
    global DEPTH_ANYTHING_MODEL
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    if DEPTH_ANYTHING_MODEL is None:
        # depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)
        # DEPTH_ANYTHING_MODEL = depth_anything
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }

        # encoder = 'vitl'  # or 'vitb', 'vits'
        depth_anything = DepthAnything(model_configs[encoder])
        depth_anything.load_state_dict \
            (torch.load(f'/data/Hszhu/prompt-to-prompt/depth-anything/depth_anything_{encoder}14.pth'))
        depth_anything.to(DEVICE).eval()
        DEPTH_ANYTHING_MODEL = depth_anything

    else:
        depth_anything = DEPTH_ANYTHING_MODEL

    image = image.astype("uint8") / 255.0

    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    depth = depth_anything(image)

    depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]


    # Converts from relative to absolute depth. This works better than using 1/depth
    depth = depth.max() - depth

    # Pushes object farther off to reduce smearing
    depth = depth + depth.max() * translate_factor

    return depth.detach().cpu().numpy()
@torch.no_grad()
def get_monocular_depth_anything_old(image, encoder = "vitl", translate_factor=0.1):

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    global DEPTH_ANYTHING_MODEL
    transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])


    if DEPTH_ANYTHING_MODEL is None:
        depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)
        DEPTH_ANYTHING_MODEL = depth_anything

    else:
        depth_anything = DEPTH_ANYTHING_MODEL


    image = image.astype("uint8") / 255.0

    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    depth = depth_anything(image)

    depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    # Converts from relative to absolute depth. This works better than using 1/depth
    depth = depth.max() - depth

    # Pushes object farther off to reduce smearing
    depth = depth + depth.max() * translate_factor

    return depth.cpu().numpy()


def chain_transform_editor(image, depth, image_mask, transform_in, prompt = "", ldm_stable_model = None, tokenizer_model = None, scheduler_in = None, 
    cross_replace_steps = {'default_': 0.97},
    self_replace_steps = 0.97,
    optimize_steps = 0.95,
    lr = 0.03,
    latent_replace = 0.2,
    optimize_embeddings = True,
    optimize_latents = True,
    obj_edit_step = 1.0,
    perform_inversion = False,
    skip_optim_steps = 2,
    guidance_scale = 5.0,
    num_ddim_steps = 50,
    edit_type = "geometry_editor",
    image_stitch = None):


    # print("movement loss 2 check###################################")

    images = perform_geometric_edit(image, depth, image_mask, transform_in, prompt, ldm_stable_model=ldm_stable_model, tokenizer_model = tokenizer_model, scheduler_in = scheduler_in, cross_replace_steps =cross_replace_steps, self_replace_steps = self_replace_steps, optimize_steps = optimize_steps, lr = lr, latent_replace = latent_replace, optimize_embeddings = optimize_embeddings, optimize_latents = optimize_latents, obj_edit_step = obj_edit_step, perform_inversion = perform_inversion, skip_optim_steps = skip_optim_steps, guidance_scale = guidance_scale, num_ddim_steps = num_ddim_steps, edit_type=edit_type, image_stitch = image_stitch)


    return images


if __name__=="__main__":

    print("check0.95softneweposepflslsmall###################################")




    use_softsplat = True
    print("halflatent0.07splattingcheck: ", use_softsplat)
    
    # depth_pt_path = "/home/ec2-user/SageMaker/weights/dpt/dpt_large-midas-2f21e586.pt"
    # d_path = "/home/ec2-user/SageMaker/data/epfl-cars/epfl-gims08/tripod-seq/"
    # d_path = "/home/ec2-user/SageMaker/data/diff-handles"

    # depth_pt_path = "/home/ec2-user/SageMaker/weights/dpt/dpt_large-midas-2f21e586.pt"
    # d_path = "/home/ec2-user/SageMaker/data/diff-handles"
    # sam_path = "/home/ec2-user/SageMaker/test_sd/segment-anything/weights/sam_vit_h_4b8939.pth"

    depth_pt_path = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/DPT/weights/dpt_large-midas-2f21e586.pt"
    d_path = "./example_images/"
    sam_path = "/oscar/scratch/rsajnani/rsajnani/research/2023/test_sd/test_sd/segment-anything/weights/sam_vit_h_4b8939.pth"
    
    # d_seg_path = "/home/ec2-user/SageMaker/data/epfl_segmented/"
    # d_depth_path = "/home/ec2-user/SageMaker/data/epfl-depths/"
    d_path = os.path.join(d_path, "")
    # d_seg_path = os.path.join(d_seg_path, "")
    # d_depth_path = os.path.join(d_depth_path, "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_name = "image_foreground.png"
    img_bg_path = "image_background.png"
    # img_name = "tripod_seq_05_067.jpg"

    image_path = d_path + img_name
    image_bg_path = d_path + img_bg_path
    # segmentation_path = d_seg_path + "mask_" + img_name #"mask_tripod_seq_04_045.jpg"
    # depth_path = d_depth_path + img_name.split(".")[0] + ".pfm"
    # depth = vis_utils.read_depth(depth_path, (IMAGE_SIZE, IMAGE_SIZE))

    # # print(depth.dtype)
    # print(depth.dtype, depth.min(), depth.max())
    image = load_256(image_path)
    image_bg = load_256(image_bg_path)
    # image_mask = (load_256(segmentation_path)[..., 0] / 255.0 > 0.5) * 1.0

    output_dir = "./output_incremental/%s/" % img_name.split(".")[0]
    print(output_dir)
    output_dir = fix_dir_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ldm_stable, tokenizer, scheduler = load_model()




    # h = input("Height normalized")
    # w = input("Width normalized")

    h = 0.75
    w = 0.75

    image_mask = get_mask_prediction(image, h, w, model_path = sam_path)


    for i in range(5):

        
        print(image.dtype, image_mask.dtype, image.shape, image_mask.shape)
        print(image.max(), image.min(), image_mask.max(), image_mask.min())

        
        plt.imsave(output_dir + "input_nt_%d_chain.png" % i, image)
        plt.imsave(output_dir + "input_nt_bg_%d_chain.png" % i, image_bg)
        plt.imsave(output_dir + "mask_nt_%d_chain.png" % i, image_mask, cmap="gray")

        prompt = ""

        # transform_in = vis_utils.rotateAxis(0.001, 1)
        transform_in = torch.eye(4).float()
        transform_in = transform_in @ vis_utils.translateMatrix(0.1, 0.0, 0).type_as(transform_in)

        print(transform_in)
        # transfor

        # print(image.max(), image.min())
        # depth_zoe = get_monocular_depth(image / 255.0, depth_pt_path)
        # depth_zoe = get_monocular_depth_anything(image)
        depth_zoe = get_constant_depth(image)
        # depth_zoe = get_monocular_ZoeDepth(image).astype(np.float32)
        # depth_zoe = depth_zoe.astype(np.float32)
        # depth_zoe = depth_zoe.clip(min=0.0)
        depth = depth_zoe


        # depth = depth_corrector(depth)
        
        # exit()



        

        print(depth.dtype, depth.min(), depth.max())

        t_coords_depth, p_image = vis_utils.get_transform_coordinates(image / 255.0, depth, image_mask, transform_in = transform_in, use_softsplat = use_softsplat)
        t_coords_depth = torch.tensor(t_coords_depth)[None]
        # exit()

        t_coords_depth = t_coords_depth.detach().cpu()
        # print(t_coords_depth.shape)
        image_mask_warped = warp_grid_edit(torch.tensor(image_mask[None, None]).float(), t_coords_depth.float(), padding_mode='zeros', align_corners=True, mode=MODE)

        image_mask_warped_save = image_mask_warped[0, 0].detach().cpu().numpy()
        plt.imsave(output_dir + "mask_nt_expected_%d_chain.png" % i, image_mask_warped_save, cmap='gray')


        # print(image_mask_warped.dtype, image_mask_warped.shape, image_mask_warped.max(), image_mask_warped.min())
        # exit()




        # if i % 2 == 0:

        #     guidance_scale = 7.5
        # else:
        #     guidance_scale = 2.0

        # guidance_scale = 1.5

        guidance_scale = 7.5
        images = chain_transform_editor(image_bg, depth, image_mask, transform_in, prompt, ldm_stable_model = ldm_stable, tokenizer_model = tokenizer, scheduler_in = scheduler, guidance_scale = guidance_scale, image_stitch=image, edit_type="geometry_stitch")
        # images = chain_transform_editor(image, depth, image_mask, transform_in, prompt, ldm_stable_model = ldm_stable, tokenizer_model = tokenizer, scheduler_in = scheduler, guidance_scale = guidance_scale, image_stitch=None, edit_type = "geometry_remover")
        # images = chain_transform_editor(image, depth, image_mask, transform_in, prompt, ldm_stable_model = ldm_stable, tokenizer_model = tokenizer, scheduler_in = scheduler, guidance_scale = guidance_scale, image_stitch=None, edit_type = "geometry_editor")

        # print(images, len(images))

        

        plt.imsave(output_dir + "output_nt_%d_chain.png" % i, images[-1])


        image = images[-1]
        image_mask = image_mask_warped[0, 0].detach().cpu().numpy()

        h, w = np.indices(image_mask.shape)

        h = np.mean(h[image_mask > 0.5])
        w = np.mean(w[image_mask > 0.5])

        h = h / (image_mask.shape[0] + 1e-8)
        w = w / (image_mask.shape[1] + 1e-8)
        exit()

        image_mask = get_mask_prediction(image, h, w, model_path = sam_path)


    # return