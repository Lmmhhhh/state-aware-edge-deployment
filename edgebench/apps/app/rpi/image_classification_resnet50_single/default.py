import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

APP_ROOT = Path(__file__).resolve().parents[2]  # .../edgebench/apps/app
DEFAULT_IMG_DIR = APP_ROOT / "_data" / "image_classification" / "coco2017_val" / "images"

def pick_fixed_image(img_dir: Path) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None, help=f"Default: {DEFAULT_IMG_DIR}")
    args = ap.parse_args()

    img_dir = Path(args.data_dir) if args.data_dir else DEFAULT_IMG_DIR

    print("[workload=image_classification][model=resnet50][mode=single][device=rpi] start")
    t0 = time.perf_counter()

    # model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    # choose 1 image
    img_path = pick_fixed_image(img_dir)
    x = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        out = model(x)

    probs = torch.nn.functional.softmax(out, dim=1)[0]
    top_prob, top_idx = torch.max(probs, dim=0)

    categories = weights.meta.get("categories", None)
    label = categories[int(top_idx)] if categories else f"class_{int(top_idx)}"
    checksum = float(out[0, int(top_idx)].item())

    t1 = time.perf_counter()
    inner_total_s = t1 - t0

    print(
        f"[workload=image_classification][model=resnet50][mode=single][device=rpi] "
        f"image={img_path.name} top1={label} prob_pct={float(top_prob)*100:.2f} "
        f"checksum_logit_top1={checksum:.6f} inner_total_s={inner_total_s:.6f}"
    )

if __name__ == "__main__":
    main()