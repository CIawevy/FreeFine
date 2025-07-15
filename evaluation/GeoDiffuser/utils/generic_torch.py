
import torch
import torchvision.transforms as T
import os
import torch.nn.functional as F
import numpy as np
from GeoDiffuser.utils.generic import *
import numbers, math
from torch import nn



class GaussianSmoothing(nn.Module):
    """
    # Borrowed from: https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/utils/gaussian_smoothing.py
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels = 1, kernel_size = 3, sigma=None, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim

        if sigma is None:
            sigma = (kernel_size[0] // 2 * 2 / 6.0)    
        
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim


        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.kernel_size = kernel_size
        self.register_buffer('weight', kernel)
        self.groups = channels

        # print(self.weight.shape)
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input_tensor):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input_tensor, weight=self.weight.type_as(input_tensor), groups=self.groups, padding=self.kernel_size[0] // 2)


def norm_tensor(A, eps = 1e-12):
    return torch.sqrt(torch.sum(A * A) + eps)


def cosine_sim(a, b, dim_l = -1):
    # a - b, f, h*w, d
    # b - b, f, h*w, d
    # sim - b, f, h*w

    a_n = torch.nn.functional.normalize(a, dim=dim_l)
    b_n = torch.nn.functional.normalize(b, dim=dim_l)
    

    sim = torch.sum(a_n * b_n, dim_l)

    return sim


def set_seed_apply_transform(im, state = None, fill = 0.0, affine = 1):
    #     https://stackoverflow.com/questions/65447992/pytorch-how-to-apply-the-same-random-transformation-to-multiple-image
    if state is not None:
        torch.set_rng_state(state)
    
    state = torch.get_rng_state()
    
    if affine:
        im_out = T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.3), scale=(0.8, 0.95), fill = fill)(im)
    else:
        im_out = T.RandomPerspective(distortion_scale=0.6, p=1.0, fill = fill)(im)
    
    
    return im_out, state



def binarize_tensor(t, thresh = 0.5):
    
    return (t > thresh) * 1.0

class CoordinateDistances:
    "Get Coordinate Distances and store for faster editing"
    def __init__(self):
        self.coord_distance_dict = {}
        self.theta = torch.eye(3)[:2][None]
    
    def get_coord_distance(self, size, device="cuda"):
        if size not in self.coord_distance_dict:
            grid = torch.nn.functional.affine_grid(self.theta, (1, 1, size, size), align_corners=None) # 1, H, W, 2
            d = grid.reshape(1, -1, 2)
            dist = torch.sqrt(torch.sum(torch.square(d[:, :, None] - d[:, None]), -1) + 1e-12) # 1, hw, hw

            self.coord_distance_dict[size] = dist.to(device)

        return self.coord_distance_dict[size]


GAUSSIAN_FEATURE_SMOOTHER = GaussianSmoothing(dim=2, kernel_size=5)

def smooth_attention_features(features):
    # features - b, h, n^2, D

    b, h_heads, n, D = features.shape
    im_size = int(np.sqrt(n))
    features_in = features.permute(0, 1, -1, 2).reshape(-1, 1, im_size, im_size) # B, 1, h, h
    features_out = GAUSSIAN_FEATURE_SMOOTHER(features_in)
    features_out = features_out.reshape(b, h_heads, D, n).permute(0, 1, -1, 2)

    return features_out

def reshape_transform_coords(transform_coords, in_mat = None, in_mat_shape = None):
    
    if in_mat is not None:
        s_z = in_mat.shape[-1]
    else:
        s_z = in_mat_shape[-1]
        

    # if s_z < transform_coords.shape[1]:
    #     t_coords = torch.nn.functional.interpolate(transform_coords.float().permute(0, -1, 1, 2), (s_z, s_z), mode="area").permute(0, 2, 3, 1).to(transform_coords.device).half()
    # else:

    smoothness_size = int(transform_coords.shape[1] / s_z)

    if smoothness_size % 2 == 0:
        smoothness_size = smoothness_size + 1

    if smoothness_size > 7:
        smoothness_size = 7
    elif smoothness_size < 3:
        smoothness_size = 3


    # print("Smoothing transform coords ", smoothness_size)
    # print(transform_coords.device)
    # t_coords_in = GaussianSmoothing(dim=2, channels = 3, kernel_size=smoothness_size)(transform_coords.permute(0, -1, 1, 2).to("cuda"))
    # t_coords_in = transform_coords

    # t_coords = T.Resize(size=(s_z, s_z), antialias=False, interpolation=T.InterpolationMode.BILINEAR)(t_coords_in).permute(0, 2, 3, 1).to(transform_coords.device)
    t_coords = T.Resize(size=(s_z, s_z), antialias=False, interpolation=T.InterpolationMode.BILINEAR)(transform_coords.permute(0, -1, 1, 2)).permute(0, 2, 3, 1).to(transform_coords.device)
    return t_coords


def reshape_attention_mask(mask, in_mat = None, in_mat_shape = None):
    
    s_m = int((mask.shape[-1]))
    
    if in_mat is not None:
        s_z = in_mat.shape[-1]
    else:
        s_z = in_mat_shape[-1]
        
    m = torch.reshape(mask, (mask.shape[0], mask.shape[1], s_m, s_m))

    # if m.shape[-1] > s_z:
    #     m = torch.nn.functional.interpolate(m.float(), (s_z, s_z), mode="area").half()
    # else:

    # print(f"resizing {s_m} to {s_z}")
    m = T.Resize(size=(s_z, s_z), antialias=False, interpolation=T.InterpolationMode.BILINEAR)(m)
    
    return m


def torch_erode(A, kernel=3):

    k = torch.ones(1, 1, kernel, kernel).type_as(A)

    k_sum = torch.sum(k)
    conv_out = torch.nn.functional.conv2d(A, k, padding=kernel // 2)

    conv_mask = (conv_out == k_sum) * 1.0
    out_f = torch.ones_like(A) * conv_mask
    # print(torch.sum(out_f), "sum")

    return out_f

def torch_dilate(A, kernel=3):

    k = torch.ones(1, 1, kernel, kernel).type_as(A)

    k_sum = 1
    conv_out = torch.nn.functional.conv2d(A, k, padding=kernel // 2)

    conv_mask = (conv_out >= k_sum) * 1.0
    out_f = torch.ones_like(A) * conv_mask
    # print(torch.sum(out_f), "sum")


    return out_f

def calc_mean_std(feat, eps: float = 1e-5, dim = -2):
    feat_std = (feat.var(dim=dim, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=dim, keepdims=True)
    return feat_mean, feat_std


def adain(feat, feat_2, dim=-2):
    feat_mean, feat_std = calc_mean_std(feat, dim=dim)
    feat_2_mean, feat_2_std = calc_mean_std(feat_2, dim=dim)

    # print("feat 1: ", feat_mean, feat_std)
    # print("feat 2: ", feat_2_mean,.shape feat_2_std.shape)
    # feat_style_mean = expand_first(feat_mean)
    # feat_style_std = expand_first(feat_std)
    feat_1 = (feat - feat_mean) / feat_std
    feat_1 = feat_1 * feat_2_std + feat_2_mean
    return feat_1

def adain_latents(latent, latent_ref):

    b, f, h, w = latent.shape
    latent_in = latent.reshape(latent.shape[0], f, -1)
    latent_ref_in = latent_ref.reshape(latent_ref.shape[0], f, -1)

    latent_adain = adain(latent_in, latent_ref_in, dim=-1).reshape(b, f, h, w)

    
    return latent_adain

def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, 4, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  4, height // 8, width // 8).to(model.device)
    return latent, latents


