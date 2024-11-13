import logging
import argparse
import os

from contextlib import nullcontext
import torch
from diffusers.utils import check_min_version

from pipeline import LotusGPipeline, LotusDPipeline
from utils.seed_all import seed_all
from evaluation.evaluation import evaluation_depth, evaluation_normal

check_min_version('0.28.0.dev0')

def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run Lotus..."
    )
    # model settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The used prediction_type. ",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="regression", # "generation"
        help="Whether to use the generation or regression pipeline."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="depth", # "normal"
    )
    parser.add_argument(
        "--disparity",
        action="store_true",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # inference settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--base_test_data_dir",
        type=str,
        default="datasets/eval/"
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    
    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run evaluation...")

    args = parse_args()

    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output dir = {args.output_dir}")

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Model --------------------
    if args.mode == 'generation':
        pipeline = LotusGPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
    elif args.mode == 'regression':
        pipeline = LotusDPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
    logging.info(f"Successfully loading pipeline from {args.pretrained_model_name_or_path}.")

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    def gen_depth(rgb_in, pipe, prompt="", num_inference_steps=50):
        if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(pipe.device.type)

        with autocast_ctx:
            rgb_input = rgb_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

            pred_depth = pipe(
                            rgb_in=rgb_input, 
                            prompt=prompt, 
                            num_inference_steps=num_inference_steps,
                            output_type='np',
                            timesteps=[args.timestep],
                            task_emb=task_emb, 
                            ).images[0]
            pred_depth = pred_depth.mean(axis=-1) # [0,1]
        return pred_depth

    def gen_normal(img, pipe, prompt="", num_inference_steps=50):
        if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(pipe.device.type)

        with autocast_ctx:
            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

            pred_normal = pipe(
                            rgb_in=img, # [-1,1] 
                            prompt=prompt, 
                            num_inference_steps=num_inference_steps,
                            output_type='pt',
                            timesteps=[args.timestep],
                            task_emb=task_emb,
                            ).images[0] # [0,1], (3,h,w)
            pred_normal = (pred_normal*2-1.0).unsqueeze(0) # [-1,1], (1,3,h,w)
        return pred_normal

    # -------------------- Evaluation --------------------
    with torch.no_grad():
        if args.task_name == 'depth':
            test_data_dir = os.path.join(args.base_test_data_dir, args.task_name)
            test_depth_dataset_configs = {
                "nyuv2": "configs/data_nyu_test.yaml", 
                "kitti": "configs/data_kitti_eigen_test.yaml",
                "scannet": "configs/data_scannet_val.yaml",
                "eth3d": "configs/data_eth3d.yaml",
            }
            for dataset_name, config_path in test_depth_dataset_configs.items():
                eval_dir = os.path.join(args.output_dir, args.task_name, dataset_name)
                test_dataset_config = os.path.join(test_data_dir, config_path)
                alignment_type = "least_square_disparity" if args.disparity else "least_square"
                metric_tracker = evaluation_depth(eval_dir, test_dataset_config, test_data_dir, eval_mode="generate_prediction",
                                                  gen_prediction=gen_depth, pipeline=pipeline, alignment=alignment_type)
                print(dataset_name,',', 'abs_relative_difference: ', metric_tracker.result()['abs_relative_difference'], 'delta1_acc: ', metric_tracker.result()['delta1_acc'])
        elif args.task_name == 'normal':
            test_data_dir = os.path.join(args.base_test_data_dir, args.task_name)
            dataset_split_path = "evaluation/dataset_normal"
            eval_datasets = [('nyuv2', 'test'), ('scannet', 'test'), ('ibims', 'ibims'), ('sintel', 'sintel')]
            eval_dir = os.path.join(args.output_dir, args.task_name)
            evaluation_normal(eval_dir, test_data_dir, dataset_split_path, eval_mode="generate_prediction", 
                              gen_prediction=gen_normal, pipeline=pipeline, eval_datasets=eval_datasets)
        else:
            raise ValueError(f"Not support predicting {args.task_name} yet. ")
        
        print('==> Evaluation is done. \n==> Results saved to:', args.output_dir)


if __name__ == '__main__':
    main()
