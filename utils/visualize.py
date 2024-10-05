import cv2
import numpy as np

import torch

from matplotlib import cm
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('root')



def tensor_to_numpy(tensor_in):
    """ torch tensor to numpy array
    """
    if tensor_in is not None:
        if tensor_in.ndim == 3:
            # (C, H, W) -> (H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(1, 2, 0).numpy()
        elif tensor_in.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(0, 2, 3, 1).numpy()
        else:
            raise Exception('invalid tensor size')
    return tensor_in

# def unnormalize(img_in, img_stats={'mean': [0.485, 0.456, 0.406], 
#                                     'std': [0.229, 0.224, 0.225]}):
def unnormalize(img_in, img_stats={'mean': [0.5,0.5,0.5], 'std': [0.5,0.5,0.5]}):
    """ unnormalize input image
    """
    if torch.is_tensor(img_in):
        img_in = tensor_to_numpy(img_in)

    img_out = np.zeros_like(img_in)
    for ich in range(3):
        img_out[..., ich] = img_in[..., ich] * img_stats['std'][ich]
        img_out[..., ich] += img_stats['mean'][ich]
    img_out = (img_out * 255.0).astype(np.uint8)
    return img_out

def normal_to_rgb(normal, normal_mask=None):
    """ surface normal map to RGB
        (used for visualization)

        NOTE: x, y, z are mapped to R, G, B
        NOTE: [-1, 1] are mapped to [0, 255]
    """
    if torch.is_tensor(normal):
        normal = tensor_to_numpy(normal)
        normal_mask = tensor_to_numpy(normal_mask)

    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm < 1e-12] = 1e-12
    normal = normal / normal_norm

    normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)
    if normal_mask is not None:
        normal_rgb = normal_rgb * normal_mask     # (B, H, W, 3)
    return normal_rgb

def kappa_to_alpha(pred_kappa, to_numpy=True):
    """ Confidence kappa to uncertainty alpha
        Assuming AngMF distribution (introduced in https://arxiv.org/abs/2109.09881)
    """
    if torch.is_tensor(pred_kappa) and to_numpy:
        pred_kappa = tensor_to_numpy(pred_kappa)

    if torch.is_tensor(pred_kappa):
        alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((torch.exp(- pred_kappa * np.pi) * np.pi) / (1 + torch.exp(- pred_kappa * np.pi)))
        alpha = torch.rad2deg(alpha)
    else:
        alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
                + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
        alpha = np.degrees(alpha)

    return alpha


def visualize_normal(target_dir, prefixs, img, pred_norm, pred_kappa,
                        gt_norm, gt_norm_mask, pred_error, num_vis=-1):
    """ visualize normal
    """
    error_max = 60.0

    img = tensor_to_numpy(img)                      # (B, H, W, 3)
    pred_norm = tensor_to_numpy(pred_norm)          # (B, H, W, 3)
    pred_kappa = tensor_to_numpy(pred_kappa)        # (B, H, W, 1)
    gt_norm = tensor_to_numpy(gt_norm)              # (B, H, W, 3)
    gt_norm_mask = tensor_to_numpy(gt_norm_mask)    # (B, H, W, 1)
    pred_error = tensor_to_numpy(pred_error)        # (B, H, W, 1)

    num_vis = len(prefixs) if num_vis == -1 else num_vis
    for i in range(num_vis):
        # img
        img_ = unnormalize(img[i, ...])
        target_path = '%s/%s_img.png' % (target_dir, prefixs[i])
        plt.imsave(target_path, img_)

        # pred_norm 
        target_path = '%s/%s_norm.png' % (target_dir, prefixs[i])
        plt.imsave(target_path, normal_to_rgb(pred_norm[i, ...]))

        # pred_kappa
        if pred_kappa is not None:
            pred_alpha = kappa_to_alpha(pred_kappa[i, :, :, 0])
            target_path = '%s/%s_pred_alpha.png' % (target_dir, prefixs[i])
            plt.imsave(target_path, pred_alpha, vmin=0.0, vmax=error_max, cmap='jet')

        # gt_norm, pred_error
        if gt_norm is not None:
            target_path = '%s/%s_gt.png' % (target_dir, prefixs[i])
            plt.imsave(target_path, normal_to_rgb(gt_norm[i, ...], gt_norm_mask[i, ...]))

            E = pred_error[i, :, :, 0] * gt_norm_mask[i, :, :, 0]
            target_path = '%s/%s_pred_error.png' % (target_dir, prefixs[i])
            plt.imsave(target_path, E, vmin=0, vmax=error_max, cmap='jet')
