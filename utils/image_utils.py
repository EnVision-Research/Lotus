from PIL import Image
import matplotlib
import numpy as np

from typing import List, Union
import PIL.Image

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

def concatenate_images(*image_lists):
    # Ensure at least one image list is provided
    if not image_lists or not image_lists[0]:
        raise ValueError("At least one non-empty image list must be provided")
    
    # Determine the maximum width of any single row and the total height
    max_width = 0
    total_height = 0
    row_widths = []
    row_heights = []

    # Compute dimensions for each row
    for image_list in image_lists:
        if image_list:  # Ensure the list is not empty
            width = sum(img.width for img in image_list)
            height = image_list[0].height  # Assuming all images in the list have the same height
            max_width = max(max_width, width)
            total_height += height
            row_widths.append(width)
            row_heights.append(height)
    
    # Create a new image to concatenate everything into
    new_image = Image.new('RGB', (max_width, total_height))
    
    # Concatenate each row of images
    y_offset = 0
    for i, image_list in enumerate(image_lists):
        x_offset = 0
        for img in image_list:
            new_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row_heights[i]  # Move the offset down to the next row
    
    return new_image


def colorize_depth_map(depth, mask=None, reverse_color=False):
    cm = matplotlib.colormaps["Spectral"]
    # normalize
    depth = ((depth - depth.min()) / (depth.max() - depth.min()))
    # colorize
    if reverse_color:
        img_colored_np = cm(1 - depth, bytes=False)[:, :, 0:3]  # Invert the depth values before applying colormap
    else:
        img_colored_np = cm(depth, bytes=False)[:, :, 0:3] # (h,w,3)

    depth_colored = (img_colored_np * 255).astype(np.uint8) 
    if mask is not None:
        masked_image = np.zeros_like(depth_colored)
        masked_image[mask.numpy()] = depth_colored[mask.numpy()]
        depth_colored_img = Image.fromarray(masked_image)
    else:
        depth_colored_img = Image.fromarray(depth_colored)
    return depth_colored_img


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img

def resize_back(
        img: Union[torch.Tensor, np.ndarray, PIL.Image.Image, List[PIL.Image.Image]],
        target_size: Union[int, tuple[int, int]],
        resample_method: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
) -> Union[torch.Tensor, np.ndarray, PIL.Image.Image, List[PIL.Image.Image]]:
    """
    Resize image to target size.

    Args:
        img (`Union[torch.Tensor, np.ndarray, PIL.Image.Image, List[PIL.Image.Image]]`):
            Image to be resized. 
        target_size (`Union[int, tuple[int, int]]`):
            Target size of the resized image.
        resample_method (`Union[InterpolationMode, int]`):
            Resampling method used to resize images. 

    Returns:
        `Union[torch.Tensor, np.ndarray, PIL.Image.Image, List[PIL.Image.Image]]`: Resized image.
    """
    if isinstance(img, torch.Tensor): # [B, C, H, W]
        resized_img = resize(img, target_size, resample_method, antialias=True)
    if isinstance(img, np.ndarray): # [B, H, W, C]
        # Conver to Torch.Tensor
        img = torch.tensor(img).permute(0, 3, 1, 2)
        resized_img = resize(img, target_size, resample_method, antialias=True)
        # Convert back to np.ndarray
        resized_img = resized_img.permute(0, 2, 3, 1).numpy()
    elif isinstance(img, PIL.Image.Image):
        target_size = (target_size[1], target_size[0])  # PIL uses (width, height)
        resized_img = img.resize(target_size, resample_method)
    elif isinstance(img, list) and all(isinstance(i, PIL.Image.Image) for i in img):
        target_size = (target_size[1], target_size[0])  # PIL uses (width, height)
        resized_img = [i.resize(target_size, resample_method) for i in img]
    return resized_img

def get_pil_resample_method(method_str: str) -> int:
    resample_method_dict = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "nearest": Image.NEAREST,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method

def get_tv_resample_method(method_str: str) -> InterpolationMode:
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST_EXACT,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method
