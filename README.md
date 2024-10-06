# <img src="assets/badges/lotus_icon.png" alt="lotus" style="height:1em; vertical-align:bottom;"/> Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction

[![Page](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://lotus3d.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2409.18124)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo%20(Depth)-yellow)](https://huggingface.co/spaces/haodongli/Lotus_Depth)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo%20(Normal)-yellow)](https://huggingface.co/spaces/haodongli/Lotus_Normal)

[Jing He](https://scholar.google.com/citations?hl=en&user=RsLS11MAAAAJ)<sup>1<span style="color:red;">&#10033;</span></sup>,
[Haodong Li](https://haodong-li.com/)<sup>1<span style="color:red;">&#10033;</span></sup>,
[Wei Yin](https://yvanyin.net/)<sup>2</sup>,
[Yixun Liang](https://yixunliang.github.io/)<sup>1</sup>,
[Leheng Li](https://len-li.github.io/)<sup>1</sup>,
[Kaiqiang Zhou]()<sup>3</sup>,
[Hongbo Zhang]()<sup>3</sup>,
[Bingbing Liu](https://scholar.google.com/citations?user=-rCulKwAAAAJ&hl=en)<sup>3</sup>,
[Ying-Cong Chen](https://www.yingcong.me/)<sup>1,4&#9993;</sup>

<span class="author-block"><sup>1</sup>HKUST(GZ)</span>
<span class="author-block"><sup>2</sup>University of Adelaide</span>
<span class="author-block"><sup>3</sup>Noah's Ark Lab</span>
<span class="author-block"><sup>4</sup>HKUST</span><br>
<span class="author-block">
    <sup style="color:red;">&#10033;</sup>**Both authors contributed equally (order randomized).**
    <sup>&#9993;</sup>Corresponding author.
</span>

![teaser](assets/badges/teaser_1.jpg)
![teaser](assets/badges/teaser_2.jpg)

We present **Lotus**, a diffusion-based visual foundation model for dense geometry prediction. With minimal training data, Lotus achieves SoTA performance in two key geometry perception tasks, i.e., zero-shot depth and normal estimation. "Avg. Rank" indicates the average ranking across all metrics, where lower values are better. Bar length represents the amount of training data used.

## 📢 News
- 2024-10-06: The demos are now available ([Depth](https://huggingface.co/spaces/haodongli/Lotus_Depth) & [Normal](https://huggingface.co/spaces/haodongli/Lotus_Normal)). Video depth/normal predictions are also supported. Please have a try! <br>
- 2024-10-05: The inference code is now available! <br>
- 2024-09-26: [Paper](https://arxiv.org/abs/2409.18124) released. <br>

## 🛠️ Setup
This installation was tested on: Ubuntu 20.04 LTS, Python 3.9, CUDA 12.3, NVIDIA A800-SXM4-80GB.  

1. Clone the repository (requires git):
```
git clone https://github.com/EnVision-Research/Lotus.git
cd Lotus
```

2. Install dependencies (requires conda):
```
conda create -n lotus python=3.9 -y
conda activate lotus
pip install -r requirements.txt 
```

## 🕹️ Usage
### Testing on your images
1. Place your images in a directory, for example, under `assets/in-the-wild_example` (where we have prepared several examples). 
2. Run the inference command: `bash infer.sh`. 

### Evaluation on benchmark datasets
1. Prepare benchmark datasets:
    - For **depth** estimation, you can download the [evaluation datasets (depth)](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/) by the following commands (referred to [Marigold](https://github.com/prs-eth/Marigold?tab=readme-ov-file#-evaluation-on-test-datasets-))：
    ```
    cd datasets/eval/depth/
    
    wget -r -np -nH --cut-dirs=4 -R "index.html*" -P . https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
    ```
    - For **normal** estimation, you can download the  [evaluation datasets (normal)](https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm?usp=drive_link) (`dsine_eval.zip`) into the path `datasets/eval/normal/` and unzip it (referred to [DSINE](https://github.com/baegwangbin/DSINE?tab=readme-ov-file#getting-started)). 

2. Run the evaluation command: `bash eval.sh`

### Choose your model
We offer four models in total, here are the corresponding configurations:
|CHECKPOINT_DIR |TASK_NAME |MODE |
|:--:|:--:|:--:|
| `jingheya/lotus-depth-g-v1-0`| `depth`| `generation`|
| `jingheya/lotus-depth-d-v1-0`| `depth`|`regression` |
| `jingheya/lotus-normal-g-v1-0`|`normal` | `generation`|
| `jingheya/lotus-normal-d-v1-0`|`normal` |`regression` |

## 🎓 Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex
@article{he2024lotus,
    title={Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction},
    author={He, Jing and Li, Haodong and Yin, Wei and Liang, Yixun and Li, Leheng and Zhou, Kaiqiang and Liu, Hongbo and Liu, Bingbing and Chen, Ying-Cong},
    journal={arXiv preprint arXiv:2409.18124},
    year={2024}
}
```
