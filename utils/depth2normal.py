from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm

from utils.projection import intrins_to_intrins_inv, get_cam_coords
from utils.visualize import normal_to_rgb
from utils.d2n.plane_svd import Depth2normal as d2n_svd

def count_data(root_dir, scenes):
    count = 0
    # Walk through all directories and files in the root directory
    for scene in scenes:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Filter files that end with the given suffix
            for filename in filenames:
                if filename.startswith("depth") and f'Scene{scene}' in dirpath:
                    count += 1
    return count

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--depth_min", type=float, default=1e-3)
    parser.add_argument("--depth_max", type=float, default=80.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--scenes", type=str, nargs="+",
                        default=['01', '02', '06', '18', '20'])
    
    return parser.parse_args()

if __name__ == "__main__":
    
    args = args_parser()
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    
    data_path = args.data_path
    depth_min = args.depth_min
    depth_max = args.depth_max
    batch_size = args.batch_size

    # scenes = ['01', '02', '06', '18', '20']
    scenes = args.scenes
    print(f"Processing scenes: {scenes}. ")

    num_data = count_data(data_path, scenes)
    print(f'Total number of data: {num_data}. ')

    conditions = [
            '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone',
            'fog', 'morning', 'overcast', 'rain', 'sunset'
        ]
    cameras = ['0', '1']
    # SceneX/Y/frames/depth/Camera_Z/depth_%05d.png
    # SceneX/Y/intrinsic.txt
    
    pbar = tqdm(total=num_data)

    for scene in scenes:
        for cond in conditions:
            intrinsic_file = os.path.join(data_path,f'Scene{scene}/{cond}/intrinsic.txt')
            with open(intrinsic_file, 'r') as file:
                lines = file.readlines()
                frames_x_cameras = len(lines)-1
                intrinsics = np.zeros((frames_x_cameras//len(cameras), len(cameras),4))
                for line in lines[1:]:
                    line = line.strip().split(" ")
                    frame_id = int(line[0])
                    camera_id = int(line[1])
                    k_00 = float(line[2])
                    k_11 = float(line[3])
                    k_02 = float(line[4])
                    k_12 = float(line[5])
                    intrinsics[frame_id][camera_id] = np.array([k_00, k_11, k_02, k_12])
            
            for cam in cameras:
                depth_dir = os.path.join(data_path, f'Scene{scene}/{cond}/frames/depth/Camera_{cam}')
                depth_files = os.listdir(depth_dir)

                num_batch = len(depth_files) // batch_size
                batches = [batch_size]*num_batch
                res = len(depth_files) % batch_size
                if res > 0:
                    num_batch += 1
                    batches += [res]

                for idx_b, batch in enumerate(batches):
                    
                    depth_batch = []
                    intrins_inv_batch = []
                    depth_path_batch = []

                    for i in range(batch):
                        depth_file_idx = idx_b * batch_size + i
                        depth_file = depth_files[depth_file_idx]

                        depth_path = os.path.join(depth_dir, depth_file)
                        depth_path_batch.append(depth_path)

                        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        depth = depth / 100 # cm -> m
                        depth = depth[:, :, None]                                       # (H, W, 1)
                        depth = torch.from_numpy(depth).permute(2, 0, 1).unsqueeze(0)   # (1, 1, H, W)
                        depth = depth.to(device)
                        depth_batch.append(depth)

                        frame_id = int(depth_file.split(".")[0].split("_")[-1])
                        k_00 = intrinsics[frame_id][int(cam)][0]
                        k_11 = intrinsics[frame_id][int(cam)][1]
                        k_02 = intrinsics[frame_id][int(cam)][2]
                        k_12 = intrinsics[frame_id][int(cam)][3]
                        intrins = np.array([
                            [k_00, 0, k_02],
                            [0, k_11, k_12],
                            [0,  0,    1  ]
                            ])
                        intrins_inv = intrins_to_intrins_inv(intrins)
                        intrins_inv = torch.from_numpy(intrins_inv).unsqueeze(0).to(device)
                        intrins_inv_batch.append(intrins_inv)
                    
                    depth_batch = torch.cat(depth_batch, dim=0)
                    intrins_inv_batch = torch.cat(intrins_inv_batch, dim=0)
                    
                    points = get_cam_coords(intrins_inv_batch, depth_batch)

                    with torch.no_grad():
                        D2N = d2n_svd(d_min=depth_min, d_max=depth_max, k=5, d=1, gamma=0.05, min_nghbr=4)
                        normal, valid_mask = D2N(points)
                    normal_rgb = normal_to_rgb(normal, valid_mask)
                    for i in range(batch):
                        norm_rgb = normal_rgb[i]
                        norm_rgb = Image.fromarray(norm_rgb, "RGB")

                        norm_rgb_save_path = depth_path_batch[i].replace("depth",'normal')
                        os.makedirs(os.path.dirname(norm_rgb_save_path), exist_ok=True)
                        norm_rgb.save(norm_rgb_save_path)

                    pbar.update(batch)
    pbar.close()