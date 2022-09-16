
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

import compressai

import numpy as np 

from compressai.zoo import image_models as pretrained_models

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB") # <PIL.Image.Image image mode=RGB size=768x512 at 0x7FF85D64CF10>
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, x, name_img):
    x = x.unsqueeze(0) # x是4维的

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p # new_h = h = 512 先//p向下取整
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


    out_enc = model.compress(x_padded) # shape:[8,12]

    out_dec = model.decompress(out_enc["strings"], out_enc["shape"]) # 最终out_dec中只有['x_hat']一个变量 len=1,4维

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    # 输出图片
    print(out_dec["x_hat"].shape)
    img=out_dec["x_hat"]
    img_save = img[0]*255 #涉及到所有的元素 都x255
    img_save = np.transpose(img_save.cpu().detach().numpy(), (1, 2, 0))
    img_save = img_save[:, :, 0:3]
    img_save = np.array(img_save, dtype=np.uint8)
    if np.ndim(img_save) > 3:
        assert img_save.shape[0] == 1
        img_save = img_save[0]

    img_save = Image.fromarray(img_save, "RGB")
    img_save.save(f"/workspace/kyz/dataset/compressed/test_urban2/{name_img}")

    return {
        "psnr": psnr(x, out_dec["x_hat"]),
    }

def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def eval_model(model, filepaths, entropy_estimation=False, half=False):
    device = next(model.parameters()).device # type = CPU 跟cuda=false有关？
    metrics = defaultdict(float)
    for f in filepaths:             # f对应的某路径下的png图片
        name_img=f.split("/")[-1]
        x = read_image(f).to(device)
        if not entropy_estimation:
            rv = inference(model, x, name_img) # rv就是包含psnr ssim bpp等的字典      
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():

    parser = argparse.ArgumentParser(description="Evaluate a model on an image dataset.", add_help=True)

    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)
    # 调试所用
    args.source = "pretrained"
    args.dataset = "/workspace/kyz/dataset/original/test/urban2"
    args.qualities = (3,)
    args.metric = "mse"
    args.entropy_estimation = False
    args.half = False
    args.cuda = True
    args.architecture = "bmshj2018-hyperprior"
    args.entropy_coder = compressai.available_entropy_coders()[0]


    filepaths = collect_images(args.dataset)

    compressai.set_entropy_coder(args.entropy_coder)

    runs = sorted(args.qualities)
    opts = (args.architecture, args.metric)
    load_func = load_pretrained

    results = defaultdict(list)
    for run in runs:
        model = load_func(*opts, run) # 加载预训练模型 opts包括 architecture 和 metric
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half) # 读入图片
        for k, v in metrics.items():
            results[k].append(v)
    output = {
        "name": args.architecture,
        "results": results,
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main(sys.argv[1:])
