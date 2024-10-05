
import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor, resize, InterpolationMode
from tqdm.auto import tqdm
import cv2

from .dataset_depth import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from .dataset_normal.normal_dataloader import *
from .util import metric
from .util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from .util.metric import MetricTracker
from .util import normal_utils

from utils.image_utils import concatenate_images, colorize_depth_map, resize_max_res
import utils.visualize as vis_utils

eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
    # "pixel_mean",
    # "pixel_var"
]

def parser_agrs():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory of depth predictions",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    # LS depth alignment
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()
    return args

# Referred to Marigold
def evaluation_depth(output_dir, dataset_config, base_data_dir, eval_mode, pred_suffix="", alignment="least_square", alignment_max_res=None, prediction_dir=None, gen_prediction=None, pipeline=None, save_pred=False, save_pred_vis=False):
    '''
    if eval_mode == "load_prediction": assert prediction_dir is not None
    elif eval_mode == "generate_prediction": assert gen_prediction is not None and pipeline is not None
    '''
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)
    processing_res = cfg_data.get('processing_res',None)
    alignment_max_res = cfg_data.get('alignment_max_res', None)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Per-sample metric file head --------------------
    per_sample_filename = os.path.join(output_dir, "per_sample_metrics.csv")
    # write title
    with open(per_sample_filename, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")

    if save_pred_vis:
        save_vis_path = os.path.join(output_dir, "vis")
        os.makedirs(save_vis_path, exist_ok=True)

    # -------------------- Evaluate --------------------
    for data in tqdm(dataloader, desc=f"Evaluating {cfg_data.name}"):
        # GT data
        depth_raw_ts = data["depth_raw_linear"].squeeze()
        valid_mask_ts = data["valid_mask_raw"].squeeze()
        rgb_name = data["rgb_relative_path"][0]

        depth_raw = depth_raw_ts.numpy()
        valid_mask = valid_mask_ts.numpy()

        depth_raw_ts = depth_raw_ts.to(device)
        valid_mask_ts = valid_mask_ts.to(device)

        # Get predictions
        rgb_basename = os.path.basename(rgb_name)
        pred_basename = get_pred_name(
            rgb_basename, dataset.name_mode, suffix=pred_suffix
        )
        pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)
        if eval_mode == "load_prediction":
            pred_path = os.path.join(prediction_dir, pred_name)
            depth_pred = np.load(pred_path)
            if not os.path.exists(pred_path):
                logging.warn(f"Can't find prediction: {pred_path}")
                continue

        elif eval_mode == "generate_prediction":
            # resize to processing_res
            input_size = data["rgb_int"].shape
            if processing_res is not None:
                input_rgb =  resize_max_res(
                    data["rgb_int"],
                    max_edge_resolution=processing_res,
                    # resample_method=resample_method,
                )
            else:
                input_rgb = data["rgb_int"]
            depth_pred = gen_prediction(input_rgb, pipeline)
            # resize to original res
            if processing_res is not None:
                depth_pred = torch.tensor(depth_pred).unsqueeze(0).unsqueeze(0)
                # depth_pred = resize(depth_pred, input_size[-2:], InterpolationMode.NEAREST, antialias=True, )
                depth_pred = resize(depth_pred, input_size[-2:], antialias=True, )
                depth_pred = depth_pred.squeeze().numpy()
            

            if save_pred:
                # save_npy
                npy_dir = os.path.join(prediction_dir, 'pred_npy', cfg_data.name)
                scene_dir = os.path.join(npy_dir, os.path.dirname(rgb_name))
                if not os.path.exists(scene_dir):
                    os.makedirs(scene_dir)
                pred_basename = get_pred_name(
                    rgb_basename, dataset.name_mode, suffix=".npy"
                )
                save_to = os.path.join(scene_dir, pred_basename)
                if os.path.exists(save_to):
                    logging.warning(f"Existing file: '{save_to}' will be overwritten")
                np.save(save_to, depth_pred)

                # save_color
                color_dir = os.path.join(prediction_dir, 'pred_color',  cfg_data.name)
                scene_dir = os.path.join(color_dir, os.path.dirname(rgb_name))
                if not os.path.exists(scene_dir):
                    os.makedirs(scene_dir)
                pred_basename = get_pred_name(
                    rgb_basename, dataset.name_mode, suffix=".png"
                )
                save_to = os.path.join(scene_dir, pred_basename)
                if os.path.exists(save_to):
                    logging.warning(f"Existing file: '{save_to}' will be overwritten")
                depth_colored = colorize_depth_map(depth_pred)
                depth_colored.save(save_to)



        # Align with GT using least square
        if "least_square" == alignment:
            depth_pred, scale, shift = align_depth_least_square(
                gt_arr=depth_raw,
                pred_arr=depth_pred,
                valid_mask_arr=valid_mask,
                return_scale_shift=True,
                max_resolution=alignment_max_res,
            )
        elif "least_square_disparity" == alignment:
            # convert GT depth -> GT disparity
            gt_disparity, gt_non_neg_mask = depth2disparity(
                depth=depth_raw, return_mask=True
            )
            # LS alignment in disparity space
            pred_non_neg_mask = depth_pred > 0
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

            disparity_pred, scale, shift = align_depth_least_square(
                gt_arr=gt_disparity,
                pred_arr=depth_pred,
                valid_mask_arr=valid_nonnegative_mask,
                return_scale_shift=True,
                max_resolution=alignment_max_res,
            )
            # convert to depth
            disparity_pred = np.clip(
                disparity_pred, a_min=1e-3, a_max=None
            )  # avoid 0 disparity
            depth_pred = disparity2depth(disparity_pred)

        # Clip to dataset min max
        depth_pred = np.clip(
            depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
        )

        # clip to d > 0 for evaluation
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

        # Evaluate (using CUDA if available)
        sample_metric = []
        depth_pred_ts = torch.from_numpy(depth_pred).to(device)
        if save_pred_vis:
            depth_pred_vis = colorize_depth_map(depth_pred_ts.cpu())
            save_path = os.path.join(save_vis_path, f"{pred_name.replace('/', '_')}.png")
            depth_pred_vis.save(save_path)

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
            sample_metric.append(_metric.__str__())
            metric_tracker.update(_metric_name, _metric)

        # Save per-sample metric
        with open(per_sample_filename, "a+") as f:
            f.write(pred_name + ",")
            f.write(",".join(sample_metric))
            f.write("\n")

    # -------------------- Save metrics to file --------------------
    eval_text = f"Evaluation metrics:\n\
    of predictions: {prediction_dir}\n\
    on dataset: {dataset.disp_name}\n\
    with samples in: {dataset.filename_ls_path}\n"

    eval_text += f"min_depth = {dataset.min_depth}\n"
    eval_text += f"max_depth = {dataset.max_depth}\n"

    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )

    metrics_filename = "eval_metrics"
    if alignment:
        metrics_filename += f"-{alignment}"
    metrics_filename += ".txt"

    _save_to = os.path.join(output_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
        logging.info(f"Evaluation metrics saved to {_save_to}")
    
    return metric_tracker

# Referred to DSINE
def evaluation_normal(eval_dir, base_data_dir, dataset_split_path, eval_mode="generate_prediction", gen_prediction=None, pipeline=None, prediction_dir=None, 
    eval_datasets=[('nyuv2', 'test'), ('scannet', 'test'), ('ibims', 'ibims'), ('sintel', 'sintel')],
    save_pred_vis=False
    ):
    '''
    if eval_mode == "load_prediction": assert prediction_dir is not None
    elif eval_mode == "generate_prediction": assert gen_prediction is not None and pipeline is not None
    '''
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device('cuda')
    metric_results = {}
    for dataset_name, split in eval_datasets:
        loader = TestLoader(base_data_dir, dataset_split_path, dataset_name_test=dataset_name, test_split=split)
        test_loader = loader.data
        
        results_dir = None
        total_normal_errors = None

        output_dir = os.path.join(eval_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        if save_pred_vis:
            results_dir = os.path.join(output_dir, "vis")
            os.makedirs(results_dir, exist_ok=True)
            print(f"Saving visualizations to {results_dir}")

        for data_dict in tqdm(test_loader):
            #↓↓↓↓
            #NOTE: forward pass
            img = data_dict['img'].to(device)
            scene_names = data_dict['scene_name']
            img_names = data_dict['img_name']
            intrins = data_dict['intrins'].to(device)

            # pad input
            _, _, orig_H, orig_W = img.shape
            lrtb = normal_utils.get_padding(orig_H, orig_W)
            img, intrins = normal_utils.pad_input(img, intrins, lrtb)

            # forward pass
            # pred_list = model(img, intrins=intrins, mode='test')
            # norm_out = pred_list[-1] # [1, 3, h, w]
            if eval_mode == "load_prediction":
                pred_path = os.path.join(prediction_dir, dataset_name, f'{scene_names[0]}_{img_names[0]}_norm.png')
                norm_out = cv2.cvtColor(cv2.imread(pred_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                norm_out = (norm_out.astype(np.float32) / 255.0) * 2.0 - 1.0 # np.array([h,w,3])
                norm_out = torch.tensor(norm_out).permute(2,0,1).unsqueeze(0).to(device) # torch.tensor([1, 3, h, w])

            elif eval_mode == "generate_prediction":
                norm_out = gen_prediction(img, pipeline) # [1, 3, h, w]

            # crop the padded part
            norm_out = norm_out[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

            pred_norm, pred_kappa = norm_out[:, :3, :, :], norm_out[:, 3:, :, :]
            pred_kappa = None if pred_kappa.size(1) == 0 else pred_kappa
            #↑↑↑↑

            if 'normal' in data_dict.keys():
                gt_norm = data_dict['normal'].to(device)
                gt_norm_mask = data_dict['normal_mask'].to(device)
                # # resize gt_norm to original size
                # pred_norm = resize(pred_norm, (gt_norm.shape[-2], gt_norm.shape[-1]), antialias=True)
                # # import torchvision; torchvision.utils.save_image(pred_norm, 'pred_norm.png')
                # # import torchvision; torchvision.utils.save_image(gt_norm, 'gt_norm.png')
                # # import torchvision; torchvision.utils.save_image(gt_norm_mask.float(), 'gt_norm_mask.png')
                # # breakpoint()

                pred_error = normal_utils.compute_normal_error(pred_norm, gt_norm)
                if total_normal_errors is None:
                    total_normal_errors = pred_error[gt_norm_mask]
                else:
                    total_normal_errors = torch.cat((total_normal_errors, pred_error[gt_norm_mask]), dim=0)

            if results_dir is not None:
                prefixs = ['%s_%s' % (i,j) for (i,j) in zip(scene_names, img_names)]
                vis_utils.visualize_normal(results_dir, prefixs, img, pred_norm, pred_kappa,
                                        gt_norm, gt_norm_mask, pred_error)

        metrics = None
        if total_normal_errors is not None:
            metrics = normal_utils.compute_normal_metrics(total_normal_errors)
            print("Dataset: ", dataset_name)
            print("mean median rmse 5 7.5 11.25 22.5 30")
            print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
                metrics['mean'], metrics['median'], metrics['rmse'],
                metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

        metric_results[dataset_name] = metrics
        # -------------------- Save metrics to file --------------------
        eval_text = f"Evaluation metrics:\n\
        on dataset: {dataset_name}\n\
        with samples in: {loader.test_samples.split_path}\n"

        eval_text += tabulate(
        [metrics.keys(), metrics.values()]
        )

        _save_to = os.path.join(output_dir, "eval_metrics.txt")
        with open(_save_to, "w+") as f:
            f.write(eval_text)
            logging.info(f"Evaluation metrics saved to {_save_to}")

        
    return metric_results