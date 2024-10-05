import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist


def get_padding(orig_H, orig_W):
    """ returns how the input of shape (orig_H, orig_W) should be padded
        this ensures that both H and W are divisible by 32
    """
    if orig_W % 32 == 0:
        l = 0
        r = 0
    else:
        new_W = 32 * ((orig_W // 32) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 32 == 0:
        t = 0
        b = 0
    else:
        new_H = 32 * ((orig_H // 32) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b

def pad_input(img, intrins, lrtb=(0,0,0,0)):
    """ pad input image
        img should be a torch tensor of shape (B, 3, H, W)
        intrins should be a torch tensor of shape (B, 3, 3)
    """
    l, r, t, b = lrtb
    if l+r+t+b != 0:
        pad_value_R = (0 - 0.485) / 0.229
        pad_value_G = (0 - 0.456) / 0.224
        pad_value_B = (0 - 0.406) / 0.225

        img_R = F.pad(img[:,0:1,:,:], (l, r, t, b), mode="constant", value=pad_value_R)
        img_G = F.pad(img[:,1:2,:,:], (l, r, t, b), mode="constant", value=pad_value_G)
        img_B = F.pad(img[:,2:3,:,:], (l, r, t, b), mode="constant", value=pad_value_B)

        img = torch.cat([img_R, img_G, img_B], dim=1)

        if intrins is not None:
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t
    return img, intrins

def compute_normal_error(pred_norm, gt_norm):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    return pred_error

def compute_normal_metrics(total_normal_errors):
    """ compute surface normal metrics (used for benchmarking)
        NOTE: total_normal_errors should be a 1D torch tensor of errors in degrees
    """
    total_normal_errors = total_normal_errors.detach().cpu().numpy()
    num_pixels = total_normal_errors.shape[0]

    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / num_pixels),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / num_pixels),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / num_pixels),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / num_pixels),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / num_pixels),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / num_pixels)
    }
    return metrics