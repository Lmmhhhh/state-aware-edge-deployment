import time
from pathlib import Path
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[2]  # .../edgebench/apps/app
IMG_DIR = APP_ROOT / "_data" / "image_classification" / "coco2017_val" / "images"

MAX_IMAGES = 128
RESIZE_W = 400
RESIZE_H = 400
OUT_DIR = APP_ROOT / "_data" / "image_processing_pillow" / "ResizedImages"          
OUT_PREFIX = "resized_"
SAVE_OUTPUT = True

def list_images_fixed(img_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    return files[:MAX_IMAGES]

def main():
    print(f"[workload=image_processing_pillow][op=resize][device=rpi] start")
    t0 = time.perf_counter()

    images = list_images_fixed(IMG_DIR)
    out_dir = OUT_DIR
    if SAVE_OUTPUT:
        out_dir.mkdir(parents=True, exist_ok=True)

    checksum = 0

    for p in images:
        im = Image.open(p).convert("RGB")
        resized = im.resize((RESIZE_W, RESIZE_H))

        if SAVE_OUTPUT:
            out_path = out_dir / f"{OUT_PREFIX}{p.name}"
            resized.save(out_path)

        r, g, b = resized.getpixel((0, 0))
        checksum = (checksum + r + g + b) % 1000000007

    t1 = time.perf_counter()
    inner_total_s = t1 - t0

    print(
        f"[workload=image_processing_pillow][op=resize][device=rpi] "
        f"num_images={len(images)} size={RESIZE_W}x{RESIZE_H} "
        f"save_output={int(SAVE_OUTPUT)} checksum={checksum} inner_total_s={inner_total_s:.6f}"
    )


if __name__ == "__main__":
    main()