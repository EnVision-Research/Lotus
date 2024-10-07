# Prediction interface for Cog ⚙️
# https://cog.run/python


import os
import subprocess
import time
from PIL import Image
import numpy as np
import torch
from cog import BasePredictor, Input, Path, BaseModel

from pipeline import LotusGPipeline, LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.seed_all import seed_all


MODEL_CACHE = "model_cache"
MODEL_URL = f"https://weights.replicate.delivery/default/EnVision-Research/Lotus/{MODEL_CACHE}.tar"

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)


class ModelOutput(BaseModel):
    generative: Path
    discriminative: Path


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # dtype = torch.float16
        dtype = torch.float32

        self.pipe_g_depth = LotusGPipeline.from_pretrained(
            f"{MODEL_CACHE}/lotus-depth-g-v1-0",
            torch_dtype=dtype,
        )
        self.pipe_d_depth = LotusDPipeline.from_pretrained(
            f"{MODEL_CACHE}/lotus-depth-d-v1-1",
            torch_dtype=dtype,
        )
        self.pipe_g_normal = LotusGPipeline.from_pretrained(
            f"{MODEL_CACHE}/lotus-normal-g-v1-0",
            torch_dtype=dtype,
        )
        self.pipe_d_normal = LotusDPipeline.from_pretrained(
            f"{MODEL_CACHE}/lotus-normal-d-v1-0",
            torch_dtype=dtype,
        )

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        task: str = Input(
            description="Choose a task", choices=["depth", "normal"], default="depth"
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        device = "cuda:0"
        seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed)

        test_image = Image.open(str(image)).convert("RGB")

        # test_image = np.array(test_image).astype(np.float16)
        test_image = np.array(test_image)
        test_image = torch.tensor(test_image).permute(2, 0, 1).unsqueeze(0)
        test_image = test_image / 127.5 - 1.0
        test_image = test_image.to(device)

        task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(
            1, 1
        )

        pipeline_g = self.pipe_g_depth if task == "depth" else self.pipe_g_normal
        pipeline_g = pipeline_g.to(device)
        pipeline_g.set_progress_bar_config(disable=True)
        pipeline_g.enable_xformers_memory_efficient_attention()

        pipeline_d = self.pipe_d_depth if task == "depth" else self.pipe_d_normal
        pipeline_d = pipeline_d.to(device)
        pipeline_d.set_progress_bar_config(disable=True)
        pipeline_d.enable_xformers_memory_efficient_attention()

        pred_g = pipeline_g(
            rgb_in=test_image,
            prompt="",
            num_inference_steps=1,
            generator=generator,
            # guidance_scale=0,
            output_type="np",
            timesteps=[999],
            task_emb=task_emb,
        ).images[0]
        pred_d = pipeline_d(
            rgb_in=test_image,
            prompt="",
            num_inference_steps=1,
            generator=generator,
            # guidance_scale=0,
            output_type="np",
            timesteps=[999],
            task_emb=task_emb,
        ).images[0]

        if task == "depth":
            output_npy_g = pred_g.mean(axis=-1)
            output_color_g = colorize_depth_map(output_npy_g)
            output_npy_d = pred_d.mean(axis=-1)
            output_color_d = colorize_depth_map(output_npy_d)
        else:
            output_npy_g = pred_g
            output_color_g = Image.fromarray((output_npy_g * 255).astype(np.uint8))
            output_npy_d = pred_d
            output_color_d = Image.fromarray((output_npy_d * 255).astype(np.uint8))

        out_path_g = "/tmp/out_g.png"
        out_path_d = "/tmp/out_d.png"
        output_color_g.save(out_path_g)
        output_color_d.save(out_path_d)
        return ModelOutput(generative=Path(out_path_g), discriminative=Path(out_path_d))
