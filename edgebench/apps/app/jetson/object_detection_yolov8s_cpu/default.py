import random
import time 
from pathlib import Path

SEED = 1
MODEL_NAME = "yolov8s.pt"

APP_ROOT = Path(__file__).resolve().parents[2]  # .../edgebench/apps/app
DEFAULT_IMG_DIR = APP_ROOT / "_data" / "image_classification" / "coco2017_val" / "images"

def pick_random_image(img_dir: Path, seed: int) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    random.seed(seed)
    return random.choice(files)

def main():
    from ultralytics import YOLO

    img_dir = DEFAULT_IMG_DIR

    print("[workload=object_detection][model=yolov8s][device=jetson][mode=cpu] start")
    t0 = time.perf_counter()

    # model load
    model = YOLO(MODEL_NAME)

    # choose 1 image
    img_path = pick_random_image(img_dir, seed=SEED)

    # inference
    # verbose=False 
    results = model.predict(source=str(img_path), device="cpu", verbose=False)

    r0 = results[0]
    n_boxes = 0
    checksum = 0.0
    try:
        boxes = r0.boxes
        n_boxes = int(len(boxes))
        if n_boxes > 0:
            xyxy0 = boxes.xyxy[0].tolist()
            checksum = float(sum(xyxy0))
    except Exception:
        pass

    t1 = time.perf_counter()
    inner_elapsed_s = t1 - t0

    print(
        "[workload=object_detection][model=yolov8s][device=jetson][mode=cpu] "
        f"image={img_path.name} n_boxes={n_boxes} checksum={checksum:.6f} "
        f"inner_elapsed_s={inner_elapsed_s:.6f}"
    )

if __name__ == "__main__":
    main()