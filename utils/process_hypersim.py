import pandas as pd
import os
import shutil
import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm

def has_nan(hdf5_path):
    """
    Check if the HDF5 file contains any NaN values.
    """
    with h5py.File(hdf5_path, 'r') as file:
        for key in file.keys():
            if np.isnan(file[key][:]).any():
                return True
    return False

def copy_images_and_depths(df, src_path, trg_path, filter_nan=False):
    """
    Copy images and depth files that do not contain NaN values in depth data.
    """
    print(f"Filtering NaN values: {filter_nan}")
    metadata = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images and depths"):
        image_file_name = f"frame.{row['frame_id']:04d}.tonemap.jpg"
        depth_file_name = f"frame.{row['frame_id']:04d}.depth_meters.hdf5"
        normal_file_name = f"frame.{row['frame_id']:04d}.normal_cam.hdf5"
        # normal_file_name_bump = f"frame.{row['frame_id']:04d}.normal_bump_cam.hdf5"
        
        src_image_path = os.path.join(src_path, row['scene_name'], 'images', f'scene_{row["camera_name"]}_final_preview', image_file_name)
        src_depth_path = os.path.join(src_path, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', depth_file_name)
        src_normal_path = os.path.join(src_path, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', normal_file_name)
        # src_normal_bump_path = os.path.join(src_path, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', normal_file_name_bump)
        
        do_copy = True
        if filter_nan:
            if has_nan(src_depth_path):  # Check if the depth file contains NaN
                do_copy = False

        if do_copy:  # Check if the depth file contains NaN
            trg_image_path = os.path.join(trg_path, row['scene_name'], 'images', f'scene_{row["camera_name"]}_final_preview', image_file_name)
            trg_depth_path = os.path.join(trg_path, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', depth_file_name)
            trg_normal_path = os.path.join(trg_path, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', normal_file_name)
            # trg_normal_bump_path = os.path.join(trg_path, row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', normal_file_name_bump)

            os.makedirs(os.path.dirname(trg_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(trg_depth_path), exist_ok=True)
            os.makedirs(os.path.dirname(trg_normal_path), exist_ok=True)
            # os.makedirs(os.path.dirname(trg_normal_bump_path), exist_ok=True)
            
            shutil.copy(src_image_path, trg_image_path)
            shutil.copy(src_depth_path, trg_depth_path)
            shutil.copy(src_normal_path, trg_normal_path)
            # shutil.copy(src_normal_bump_path, trg_normal_bump_path)
            
            metadata.append({
                "file_name": os.path.join(row['scene_name'], 'images', f'scene_{row["camera_name"]}_final_preview', image_file_name), 
                "depth": os.path.join(row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', depth_file_name),
                "normal_cam": os.path.join(row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', normal_file_name),
                # "normal_bump_cam": os.path.join(row['scene_name'], 'images', f'scene_{row["camera_name"]}_geometry_hdf5', normal_file_name_bump),
                })
            # print(f"Copied {src_image_path} and {src_depth_path} to {trg_image_path} and {trg_depth_path}")
            # print(f"Copied {src_normal_path} and {src_normal_bump_path} to {trg_normal_path} and {trg_normal_bump_path}")

    return metadata

def save_metadata(metadata, trg_path):
    """
    Save metadata to a JSONL file.
    """
    metadata_path = os.path.join(trg_path, 'metadata.jsonl')
    with open(metadata_path, 'w') as f:
        for meta in metadata:
            json_line = json.dumps(meta)
            f.write(json_line + '\n')
    print(f"Metadata saved to {metadata_path}")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--src_path", required=True)
    parser.add_argument("--trg_path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--filter_nan", action="store_true")
    return parser.parse_args()

'''
Example Usage:
    python utils/process_hypersim.py \
    --csv_path=datasets/hypersim_raw/metadata_images_split_scene_v1.csv \
    --src_path=datasets/hypersim_raw/downloads \
    --trg_path=datasets/hypersim_filtered \
    --split='train' \
    --filter_nan
'''

if __name__ == "__main__":
    args = parse_args()

    # Load the CSV file
    data = pd.read_csv(args.csv_path)

    # Filter the data for images that are included in public release and are part of the target split
    split_data = data[(data['included_in_public_release'] == True) & (data['split_partition_name'] == args.split)]

    # Prepare paths
    src_base_path = args.src_path
    trg_base_path = os.path.join(args.trg_path, args.split)

    # Process and copy files
    metadata = copy_images_and_depths(split_data, src_base_path, trg_base_path, args.filter_nan)

    # Save metadata
    save_metadata(metadata, trg_base_path)
