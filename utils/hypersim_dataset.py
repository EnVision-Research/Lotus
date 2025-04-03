from datasets import Dataset as Dataset_hf
import os
import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import random
from torchvision import transforms

def hypersim_distance_to_depth(npyDistance):
    intWidth=1024
    intHeight=768
    fltFocal=886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal

    return npyDepth

def creat_uv_mesh(H, W):
    y, x = np.meshgrid(np.arange(0, H, dtype=np.float64), np.arange(0, W, dtype=np.float64), indexing='ij')
    meshgrid = np.stack((x,y))
    ones = np.ones((1,H*W), dtype=np.float64)
    xy = meshgrid.reshape(2, -1)
    return np.concatenate([xy, ones], axis=0)

# Some Hypersim normals are not properly oriented towards the camera.
    # The align_normals and creat_uv_mesh functions are from GeoWizard
    # https://github.com/fuxiao0719/GeoWizard/blob/5ff496579c6be35d9d86fe4d0760a6b5e6ba25c5/geowizard/training/dataloader/file_io.py#L79
def align_normals(normal, depth, K, H, W):
    '''
    Orientation of surface normals in hypersim is not always consistent
    see https://github.com/apple/ml-hypersim/issues/26
    '''
    # inv K
    K = np.array([[K[0],    0, K[2]], 
                    [   0, K[1], K[3]], 
                    [   0,    0,    1]])
    inv_K = np.linalg.inv(K)
    # reprojection depth to camera points
    xy = creat_uv_mesh(H, W)
    points = np.matmul(inv_K[:3, :3], xy).reshape(3, H, W)
    points = depth * points
    points = points.transpose((1,2,0))
    # align normal
    orient_mask = np.sum(normal * points, axis=2) < 0
    normal[orient_mask] *= -1
    return normal 

class HypersimImageDepthNormalTransform:
    def __init__(self, size, random_flip, norm_type, truncnorm_min=0.02, align_cam_normal=False) -> None:
        self.size = size
        self.random_flip = random_flip
        self.norm_type = norm_type
        self.truncnorm_min = truncnorm_min
        self.truncnorm_max = 1 - truncnorm_min
        self.d_max = 65
        self.align_cam_normal = align_cam_normal
    
    def to_tensor_and_resize_normal(self, normal):
        # to tensor
        normal = torch.from_numpy(normal).permute(2,0,1).unsqueeze(0)
        # resize
        normal = F.interpolate(normal, size=self.size, mode='nearest').squeeze()
        # shape = 3 * 768 * 1024
        return normal

    def __call__(self, image, depth, normal):
        # convert the inward normals to outward normals
        normal[:,:,0] *= -1
        if self.align_cam_normal:
            # align normal towards camera
            H, W = normal.shape[:2]
            normal = align_normals(normal, depth, [886.81, 886.81, W/2, H/2], H, W)

        # resize
        image = transforms.functional.resize(image, self.size, interpolation=Image.BILINEAR)
        
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=self.size, mode='nearest').squeeze()

        normal = self.to_tensor_and_resize_normal(normal)
        
        # random flip
        if self.random_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            depth = torch.flip(depth, [-1])
            normal = torch.flip(normal, [-1])
            normal[0,:,:] = - normal[0,:,:] # Flip x-component of normal map

        # to_tensor and normalize
        # image
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5], [0.5])(image)

        # depth
        if self.norm_type == 'instnorm':
            dmin = depth.min()
            dmax = depth.max()
            depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == 'truncnorm':
            # refer to Marigold
            dmin = torch.quantile(depth, self.truncnorm_min)
            dmax = torch.quantile(depth, self.truncnorm_max)
            depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == 'perscene_norm':
            depth_norm = ((depth / self.d_max) - 0.5 ) * 2.0
        elif self.norm_type == "disparity":
            disparity = 1 / depth
            disparity_min = disparity.min()
            disparity_max = disparity.max()
            disparity_norm = ((disparity - disparity_min)/(disparity_max - disparity_min + 1e-5) - 0.5) * 2
            depth_norm = disparity_norm
        elif self.norm_type == "trunc_disparity":
            disparity = 1 / depth
            disparity_min = torch.quantile(disparity, self.truncnorm_min)
            disparity_max = torch.quantile(disparity, self.truncnorm_max)
            disparity_norm = ((disparity - disparity_min)/(disparity_max - disparity_min + 1e-5) - 0.5) * 2
            depth_norm = disparity_norm
        else:
            raise TypeError(f"Not supported normalization type: {self.norm_type}. ")
        
        depth_norm = depth_norm.clip(-1,1)
        depth = depth_norm.unsqueeze(0).repeat(3,1,1)

        # normal
        normal = normal.clip(-1, 1)

        return image, depth, normal

def get_hypersim_dataset_depth_normal(data_dir, resolution, random_flip, norm_type, truncnorm_min, align_cam_normal=False, split='train'):
    split_dir = os.path.join(data_dir, split)
    # load data and construct the dataset
    data_dict = {
        "image": [],
        "depth": [],
        "normal": []
    }
    for root, dirs, files in os.walk(split_dir):
        for file in files:
            if file.endswith("tonemap.jpg"): 
                image_path = os.path.join(root, file)
                depth_path = image_path.replace("final_preview", "geometry_hdf5").replace("tonemap.jpg", "depth_meters.hdf5")
                normal_path = image_path.replace("final_preview", "geometry_hdf5").replace("tonemap.jpg", "normal_cam.hdf5")
                data_dict["image"].append(image_path)
                data_dict["depth"].append(depth_path)
                data_dict["normal"].append(normal_path)
    dataset = Dataset_hf.from_dict(data_dict)
    
    # define dataset transform
    column_names = dataset.column_names

    image_column = column_names[0]
    depth_column = column_names[1]
    normal_colum = column_names[2]

    w, h = Image.open(dataset[0][image_column]).size
    if h > w:
        new_w = resolution
        new_h = int(resolution * h / w)
    else:
        new_h = resolution
        new_w = int(resolution * w / h)
    
    transforms = HypersimImageDepthNormalTransform((new_h, new_w), random_flip, norm_type, truncnorm_min, align_cam_normal)

    def preprocess_hypersim(examples):
        # convert image to RGB
        images = [Image.open(image).convert("RGB") for image in examples[image_column]]
        depths = [] # List[np.array()]
        normals = [] # List[np.array()]
        for depth_file in examples[depth_column]:
            # convert distance to depth
            depth_fd = h5py.File(depth_file, 'r')
            dist = np.array(depth_fd['dataset'])
            depths.append(hypersim_distance_to_depth(dist))
        for normal_file in examples[normal_colum]:
            normal_fd = h5py.File(normal_file, 'r')
            dist = np.array(normal_fd['dataset'])
            normals.append(dist)
        # transform image and annotation simutanueously. 
        examples["pixel_values"] = []
        examples["depth_values"] = []
        examples["normal_values"] = []

        for img, dep, nor in zip(images, depths, normals):
            train_img, train_dep, train_nor = transforms(img, dep, nor)
            examples["pixel_values"].append(train_img)
            examples["depth_values"].append(train_dep)
            examples["normal_values"].append(train_nor)
        
        return examples

    # define collate function
    def collate_fn_hypersim(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        depth_values = torch.stack([example["depth_values"] for example in examples])
        depth_values = depth_values.to(memory_format=torch.contiguous_format).float()

        normal_values = torch.stack([example["normal_values"] for example in examples])
        normal_values = normal_values.to(memory_format=torch.contiguous_format).float()

        image_paths = [example[image_column] for example in examples]
        depth_paths = [example[depth_column] for example in examples]
        normal_paths = [example[normal_colum] for example in examples]

        example_dict = {
            "pixel_values": pixel_values, 
            "depth_values": depth_values,
            "normal_values": normal_values, 
            "image_pathes": image_paths,
            "depth_paths": depth_paths,
            "normal_paths": normal_paths
        }

        return example_dict
    
    return dataset, preprocess_hypersim, collate_fn_hypersim
