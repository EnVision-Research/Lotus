import logging
import os
import argparse
from pathlib import Path
from PIL import Image
from contextlib import nullcontext

import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.utils import check_min_version

from pipeline import LotusGPipeline, LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.seed_all import seed_all

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
        "--input_dir", type=str, required=True, help="Input directory."
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run inference...")

    args = parse_args()

    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output dir = {args.output_dir}")

    output_dir_color = os.path.join(args.output_dir, f'{args.task_name}_vis')
    output_dir_npy = os.path.join(args.output_dir, f'{args.task_name}')
    if not os.path.exists(output_dir_color): os.makedirs(output_dir_color)
    if not os.path.exists(output_dir_npy): os.makedirs(output_dir_npy)

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32
    
    # processing_res
    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Data --------------------
    root_dir = Path(args.input_dir)
    test_images = list(root_dir.rglob('*.png')) + list(root_dir.rglob('*.jpg'))
    test_images = sorted(test_images)
    print('==> There are', len(test_images), 'images for validation.')
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
    logging.info(f"processing_res = {processing_res or pipeline.default_processing_resolution}")

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i in tqdm(range(len(test_images))):
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(pipeline.device.type)
            with autocast_ctx:
                # Preprocess validation image
                test_image = Image.open(test_images[i]).convert('RGB')
                test_image = np.array(test_image).astype(np.float32)
                test_image = torch.tensor(test_image).permute(2,0,1).unsqueeze(0)
                test_image = test_image / 127.5 - 1.0 
                test_image = test_image.to(device)

                task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

                # Run
                pred = pipeline(
                    rgb_in=test_image, 
                    prompt='', 
                    num_inference_steps=1, 
                    generator=generator, 
                    # guidance_scale=0,
                    output_type='np',
                    timesteps=[args.timestep],
                    task_emb=task_emb,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    resample_method=resample_method,
                    ).images[0]

                # Post-process the prediction
                save_file_name = os.path.basename(test_images[i])[:-4]
                if args.task_name == 'depth':
                    output_npy = pred.mean(axis=-1)
                    output_color = colorize_depth_map(output_npy, reverse_color=args.disparity)
                else:
                    output_npy = pred
                    output_color = Image.fromarray((output_npy * 255).astype(np.uint8))

                output_color.save(os.path.join(output_dir_color, f'{save_file_name}.png'))
                np.save(os.path.join(output_dir_npy, f'{save_file_name}.npy'), output_npy)

            torch.cuda.empty_cache()
            
    print('==> Inference is done. \n==> Results saved to:', args.output_dir)


if __name__ == '__main__':
    main()
