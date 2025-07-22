import torch

from GeoDiffuser.utils.generic_torch import *

def get_correlation_loss_stitch(correlation, mask_zero, mask_one, mask_warped_one):


    # print(correlation.shape, mask_zero.shape, mask_one.shape, mask_warped_one.shape)
    f = correlation.shape[0]

    correlation_inpaint = correlation * mask_zero[..., 0]
    correlation_wo_edit = correlation * mask_one[..., 0]

    
    p_correlation_inpaint = torch.max(correlation_inpaint, -1).values * mask_warped_one[0, :1, :, 0]
    p_correlation_wo_edit = torch.max(correlation_wo_edit, -1).values * mask_warped_one[0, :1, :, 0]

    loss_correlation = torch.sum(-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4)) / (torch.sum(mask_warped_one) * f + 1e-8)

    return loss_correlation

def gradient(r_out):

    D_dw = r_out[:, :, :, 1:] - r_out[:, :, :, :-1]
    D_dh = r_out[:, :, 1:, :] - r_out[:, :, :-1, :]
    return D_dh, D_dw


def get_smoothness_loss(replace_out):

    b, f, hw, d = replace_out.shape

    h = int(np.sqrt(hw))
    w = h
    r_out = replace_out.reshape(b, f, h, w, d)

    D_dh, D_dw = gradient(r_out)

    loss = D_dh.abs().mean() + D_dw.abs().mean()

    return loss, D_dh.abs(), D_dw.abs()