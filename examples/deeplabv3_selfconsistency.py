"""Integration example: segrobust on a real pretrained segmentation model.

Model: torchvision deeplabv3_resnet50 (COCO-pretrained, VOC 21 classes).
Images: 4 real photos fetched from COCO val2017 at runtime (never committed
        to this repository).
Labels: the model's OWN clean predictions (pseudo-labels). This measures
        prediction stability under corruption (self-consistency), NOT
        ground-truth accuracy. clean_miou is 1.0 by construction; the
        signal is how far corrupt mIoU falls below it.

Requires: pip install torch torchvision  (CPU wheels are fine)
Runtime: ~6 minutes on CPU.
"""
import glob
import os
import urllib.request

import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from segrobust import evaluate

COCO_IDS = ["000000039769", "000000000139", "000000000285", "000000000632"]
CACHE = os.path.join(os.path.dirname(__file__), "_coco_cache")


def fetch_images():
    os.makedirs(CACHE, exist_ok=True)
    paths = []
    for cid in COCO_IDS:
        p = os.path.join(CACHE, f"{cid}.jpg")
        if not os.path.exists(p):
            url = f"http://images.cocodataset.org/val2017/{cid}.jpg"
            urllib.request.urlretrieve(url, p)
        paths.append(p)
    images = []
    for p in sorted(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((384, 384))  # keep CPU runtime reasonable
        images.append(np.asarray(im, dtype=np.uint8))
    return images


def main():
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).eval()
    pre = weights.transforms()

    @torch.no_grad()
    def model_fn(u8):
        h, w0 = u8.shape[:2]
        t = pre(torch.from_numpy(u8.copy()).permute(2, 0, 1).float()
                / 255.0).unsqueeze(0)
        out = model(t)["out"][0].argmax(0)
        # VOC eval transforms resize to 520; restore input resolution
        out = torch.nn.functional.interpolate(
            out[None, None].float(), size=(h, w0), mode="nearest")[0, 0]
        return out.numpy().astype(np.int64)

    images = fetch_images()
    pseudo_labels = [model_fn(im) for im in images]

    res = evaluate(model_fn, images, pseudo_labels, num_classes=21,
                   corruptions=["gaussian_noise", "defocus_blur",
                                "fog", "frost"],
                   severities=(1, 3, 5), seed=0,
                   progress=lambda m: print(" ", m))

    print("clean_miou (self-consistency):", round(res["clean_miou"], 4))
    print("mean_corrupt_miou:", round(res["mean_corrupt_miou"], 4))
    print("relative_drop:", round(res["relative_drop"], 4))
    for name, sev in res["per_corruption"].items():
        print(f"  {name}:", {s: round(v, 4) for s, v in sev.items()})


if __name__ == "__main__":
    main()
