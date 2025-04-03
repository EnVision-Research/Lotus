import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from torch.utils.data import Dataset


class VKITTITransform:
    '''Crop images to KITTI benchmark size and randomly flip. '''

    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216

    def __init__(self, random_flip):
        self.random_flip = random_flip

    def resize_image(self, image, target_width, target_height, interpolation):
        """Resizes the image and returns the new image and its dimensions."""
        current_width, current_height = image.size
        if current_height < target_height or current_width < target_width:
            # Calculate the scaling factor and new dimensions
            scaling_factor = max(target_height / current_height, target_width / current_width)
            new_width = int(current_width * scaling_factor)
            new_height = int(current_height * scaling_factor)
            image = TF.resize(image, (new_height, new_width), interpolation)
        return image

    def __call__(self, image, depth, normal):
        '''
        Input:
            image (PIL.Image)
            depth (torch.Tensor): (1,1,h,w)
            normal (PIL.Image)
        Output:
            image (PIL.Image)
            depth (torch.Tensor): (h,w)
            normal (PIL.Image)
        '''
        # Resize images if necessary
        image = self.resize_image(image, self.KB_CROP_WIDTH, self.KB_CROP_HEIGHT, InterpolationMode.BILINEAR)
        normal = self.resize_image(normal, self.KB_CROP_WIDTH, self.KB_CROP_HEIGHT, InterpolationMode.NEAREST)
        depth = F.interpolate(depth, size=(image.height, image.width), mode='nearest').squeeze()  # Adjust depth to new size

        # Crop to exact dimensions
        top = int(image.height - self.KB_CROP_HEIGHT)
        left = int((image.width - self.KB_CROP_WIDTH) / 2)
        image = image.crop((left, top, left+self.KB_CROP_WIDTH, top+self.KB_CROP_HEIGHT))
        normal = normal.crop((left, top, left+self.KB_CROP_WIDTH, top+self.KB_CROP_HEIGHT))
        depth = depth[top:top+self.KB_CROP_HEIGHT, left:left+self.KB_CROP_WIDTH]

        # Random horizontal flipping
        if self.random_flip and torch.rand(1) > 0.5:
            image = TF.hflip(image)
            depth = torch.flip(depth, [-1])  # Flip depth tensor on width dimension
            normal = TF.hflip(normal)
            normal = np.array(normal)
            normal_mask = np.any(normal != [0, 0, 0], axis=-1)
            normal[:, :, 0][normal_mask] = 255 - normal[:, :, 0][normal_mask]  # Flip x-component of normal map
            normal = Image.fromarray(normal)
        return image, depth, normal

class VKITTIDataset(Dataset):
    def __init__(self, root_dir, transform=None, norm_type='truncnorm', truncnorm_min=0.02):
        """
        Args:
            root_dir (string): Directory with all the images and depths.
            transform (callable, optional): Optional transform to be applied on a sample.
            norm_type (str, optional): Normalization type.
            truncnorm_min (float, optional): Minimum value for truncated normal distribution.
        """
        self.root_dir = root_dir
        self.transform = transform
        # self.scenes = ['01', '02', '06', '18', '20']
        self.scenes = ['02', '06', '18', '20']
        self.conditions = [
            '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone',
            'fog', 'morning', 'overcast', 'rain', 'sunset'
        ]
        self.cameras = ['0', '1']

        self.d_min = 1e-5
        self.d_max = 80

        self.norm_type = norm_type
        self.truncnorm_min = truncnorm_min
        self.truncnorm_max = 1 - truncnorm_min
        
        self.image_path=[]
        self.depth_path=[]
        self.normal_path=[]
        
        for scene in self.scenes:
            for cond in self.conditions:
                for cam in self.cameras:
                    image_dir = os.path.join(self.root_dir, f'Scene{scene}/{cond}/frames/rgb/Camera_{cam}')
                    images = os.listdir(image_dir)
                    for image in images:
                        img_path = os.path.join(image_dir, image)
                        self.image_path.append(img_path)
                        self.depth_path.append(img_path.replace("rgb","depth").replace(".jpg",".png"))
                        self.normal_path.append(img_path.replace("rgb","normal").replace(".jpg",".png"))
                    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        example = {}

        # Load data
        image_path = self.image_path[idx]
        depth_path = self.depth_path[idx]
        normal_path = self.normal_path[idx]
        example['image_path'] = image_path
        example['depth_path'] = depth_path
        example['normal_path'] = normal_path
        ## image
        image = Image.open(image_path).convert('RGB')
        ## depth
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth / 100 # cm -> m
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0) # (1, 1, h, w)
        ## normal
        normal = Image.open(normal_path).convert('RGB')

        # Transform data simultaneously if needed
        if self.transform:
            image, depth, normal = self.transform(image, depth, normal)
        
        # Get mask given depth
        depth = depth.unsqueeze(0) # (1, h, w)
        valid_mask_raw = self._get_valid_mask(depth).clone()
        sky_mask_raw = self._get_sky_mask(depth).clone()
        example['valid_mask_values'] = valid_mask_raw
        example['sky_mask_values'] = sky_mask_raw

        # Normalize
        ## image
        image_norm = np.array(image).astype(np.float32) / 127.5 - 1
        example['pixel_values'] = torch.from_numpy(image_norm).permute(2, 0, 1)
        ## depth
        valid_mask = valid_mask_raw & (depth > 0)
        if self.norm_type == 'instnorm':
            dmin = depth[valid_mask].min()
            dmax = depth[valid_mask].max()
            depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == 'truncnorm':
            # refer to Marigold
            dmin = torch.quantile(depth[valid_mask], self.truncnorm_min)
            dmax = torch.quantile(depth[valid_mask], self.truncnorm_max)
            depth_norm = ((depth - dmin)/(dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == 'perscene_norm':
            depth_norm = ((depth / self.d_max) - 0.5 ) * 2.0
        elif self.norm_type == "disparity":
            disparity = 1 / depth
            disparity_min = disparity[valid_mask].min()
            disparity_max = disparity[valid_mask].max()
            disparity_norm = ((disparity - disparity_min)/(disparity_max - disparity_min + 1e-5) - 0.5) * 2.0
            depth_norm = disparity_norm
        elif self.norm_type == "trunc_disparity":
            disparity = 1 / depth 
            disparity_min = torch.quantile(disparity[valid_mask], self.truncnorm_min)
            disparity_max = torch.quantile(disparity[valid_mask], self.truncnorm_max)
            disparity_norm = ((disparity - disparity_min)/(disparity_max - disparity_min + 1e-5) - 0.5) * 2.0
            depth_norm = disparity_norm
        else:
            raise TypeError(f"Not supported normalization type: {self.norm_type}. ")
        
        depth_norm = depth_norm.clip(-1,1)
        depth_norm = depth_norm.repeat(3,1,1) # (3, h, w)
        
        example['depth_values'] = depth_norm
        ## normal
        normal_norm = np.array(normal).astype(np.float32) / 127.5 - 1
        example['normal_values'] = torch.from_numpy(normal_norm).permute(2, 0, 1)

        return example
    
    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.d_min), (depth < self.d_max)
        ).bool()
        return valid_mask

    def _get_sky_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.d_min), (depth >= self.d_max)
        ).bool()
        return valid_mask

def collate_fn_vkitti(examples):
    image_pathes = [example['image_path'] for example in examples]
    depth_pathes = [example['depth_path'] for example in examples]
    normal_pathes = [example['normal_path'] for example in examples]

    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    depth_values = torch.stack([example['depth_values'] for example in examples])
    depth_values = depth_values.to(memory_format=torch.contiguous_format).float()

    normal_values = torch.stack([example["normal_values"] for example in examples])
    normal_values = normal_values.to(memory_format=torch.contiguous_format).float()

    valid_mask_values = torch.stack([example["valid_mask_values"] for example in examples])
    valid_mask_values = valid_mask_values.to(memory_format=torch.contiguous_format).float()

    sky_mask_values = torch.stack([example["sky_mask_values"] for example in examples])
    sky_mask_values = sky_mask_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "depth_values": depth_values,
        "normal_values": normal_values,
        "valid_mask_values": valid_mask_values,
        "sky_mask_values": sky_mask_values,
        "image_pathes": image_pathes,
        "depth_pathes": depth_pathes,
        "normal_pathes": normal_pathes
    }