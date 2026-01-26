import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

APP_ROOT = Path(__file__).resolve().parents[2] 
DATA_ROOT = APP_ROOT / "_data"

def list_images(img_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing images. If omitted, uses edgebench/_data/image_classification/coco2017_val/images",
    )
    ap.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="0 means use all images. Otherwise, process first N images.",
    )
    args = ap.parse_args()

    # default data dir: <edgebench_root>/_data/image_classification/coco2017_val/images
    default_dir = DATA_ROOT / "image_classification" /"coco2017_val" / "images"
    img_dir = Path(args.data_dir) if args.data_dir else default_dir

    print("Executing image-classification-mnv3-batch (RPi)...")
    start_time = time.time()

    # model load
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.eval()
    preprocess = weights.transforms()

    load_done = time.time()

    images = list_images(img_dir)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    categories = None
    try:
        categories = weights.meta.get("categories", None)
    except Exception:
        categories = None

    # inference loop
    infer_start = time.time()
    last_top1 = ("NA", 0.0)
    last_checksum = float("nan")

    with torch.no_grad():
        for p in images:
            img = Image.open(p).convert("RGB")
            x = preprocess(img).unsqueeze(0)
            out = model(x)

            probs = torch.nn.functional.softmax(out, dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)
            label = categories[int(top_idx)] if categories else f"class_{int(top_idx)}"

            last_top1 = (label, float(top_prob.item()) * 100.0)
            last_checksum = float(out[0, int(top_idx)].item())

    infer_done = time.time()
    end_time = infer_done

    load_time = load_done - start_time
    infer_time = infer_done - infer_start
    total_time = end_time - start_time

    print(f"num_images={len(images)}")
    print(f"last_top1: {last_top1[0]} ({last_top1[1]:.2f}%)")
    print(f"last_checksum(logit_top1): {last_checksum:.6f}")
    print(f"{load_time:.4f} , Load time , {infer_time:.4f} , Infer time , {total_time:.4f} , Total time")
    print("Function executed in {:.2f} seconds.".format(total_time))


if __name__ == "__main__":
    main()