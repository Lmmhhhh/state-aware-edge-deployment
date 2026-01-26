import argparse
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

APP_ROOT = Path(__file__).resolve().parents[2] 
DATA_ROOT = APP_ROOT / "_data"

def pick_random_image(img_dir: Path, seed: int) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    random.seed(seed)
    return random.choice(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing images. If omitted, uses edgebench/_data/image_classification/coco2017_val/images",
    )
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    # default data dir: <edgebench_root>/_data/image_classification/coco2017_val/images
    edge_root = Path(__file__).resolve().parents[3]
    default_dir = DATA_ROOT / "image_classification" /"coco2017_val" / "images"
    img_dir = Path(args.data_dir) if args.data_dir else default_dir

    print("Executing image-classification-mnv3-single (RPi)...")
    start_time = time.time()

    # model
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    # choose 1 image
    img_path = pick_random_image(img_dir, seed=args.seed)
    input_image = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(input_image).unsqueeze(0)

    # inference
    with torch.no_grad():
        out = model(input_tensor)

    probs = torch.nn.functional.softmax(out, dim=1)[0]
    top_prob, top_idx = torch.max(probs, dim=0)

    categories = None
    try:
        categories = weights.meta.get("categories", None)
    except Exception:
        categories = None
    label = categories[int(top_idx)] if categories else f"class_{int(top_idx)}"

    checksum = float(out[0, int(top_idx)].item())

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"image: {img_path.name}")
    print(f"top1: {label} ({float(top_prob.item())*100:.2f}%)")
    print(f"checksum(logit_top1): {checksum:.6f}")
    print("The function has executed successfully in {:.2f} seconds.".format(elapsed))


if __name__ == "__main__":
    main()