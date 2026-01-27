import argparse
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

APP_ROOT = Path(__file__).resolve().parents[2]  # .../edgebench/apps/app
DEFAULT_IMG_DIR = APP_ROOT / "_data" / "image_classification" / "coco2017_val" / "images"

def list_images(img_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    return files

def pick_seed_random_subset(files, max_images: int, seed: int):
    if max_images == 0:
        return files 
    if max_images < 0:
        raise ValueError("--max_images must be >= 0")
    if max_images >= len(files):
        return files
    rng = random.Random(seed)
    return rng.sample(files, k=max_images)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None, help=f"Default: {DEFAULT_IMG_DIR}")
    ap.add_argument(
        "--max_images",
        type=int,
        default=128,
        help="0 means use all images. Otherwise, select N images by seed-random sampling.",
    )
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    img_dir = Path(args.data_dir) if args.data_dir else DEFAULT_IMG_DIR

    print("[workload=image_classification][model=mobilenetv3_large][mode=batch][device=rpi] start")

    t0 = time.perf_counter()

    # model load
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta.get("categories", None)

    t_load_done = time.perf_counter()

    all_images = list_images(img_dir)
    images = pick_seed_random_subset(all_images, max_images=args.max_images, seed=args.seed)

    t_infer0 = time.perf_counter()

    last_label = "NA"
    last_prob_pct = 0.0
    last_checksum = float("nan")

    with torch.inference_mode():
        for p in images:
            x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
            out = model(x)

            probs = torch.softmax(out, dim=1)[0]
            top_prob, top_idx = probs.max(dim=0)

            last_label = categories[int(top_idx)] if categories else f"class_{int(top_idx)}"
            last_prob_pct = float(top_prob) * 100.0
            last_checksum = float(out[0, int(top_idx)].item())

    t_infer1 = time.perf_counter()

    t1 = time.perf_counter()
    inner_load_s = t_load_done - t0
    inner_infer_s = t_infer1 - t_infer0
    inner_total_s = t1 - t0

    print(
        f"[workload=image_classification][model=mobilenetv3_large][mode=batch][device=rpi] "
        f"seed={args.seed} num_images={len(images)} last_top1={last_label} last_prob_pct={last_prob_pct:.2f} "
        f"last_checksum_logit_top1={last_checksum:.6f} "
        f"inner_load_s={inner_load_s:.6f} inner_infer_s={inner_infer_s:.6f} inner_total_s={inner_total_s:.6f}"
    )

if __name__ == "__main__":
    main()