import argparse
import time
from pathlib import Path
from statistics import median

import torch
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG_DIR = APP_ROOT / "edgebench" /"apps" / "app" / "_data" / "image_classification" / "coco2017_val" / "images"

def pick_fixed_image(img_dir: Path) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    return files[0]

def percentile(xs, q: float) -> float:
    """q: 0~100"""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)

def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None, help=f"Default: {DEFAULT_IMG_DIR}")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--include_preprocess", action="store_true",
                    help="if set, include preprocess time in per-iter latency")
    args = ap.parse_args()

    img_dir = Path(args.data_dir) if args.data_dir else DEFAULT_IMG_DIR
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    print(f"[workload=image_classification][model=mobilenetv3_large] device={device} warmup={args.warmup} repeat={args.repeat}")

    # ---- load model (NOT timed) ----
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights).to(device)
    model.eval()

    preprocess = weights.transforms()

    # fixed image (NOT timed)
    img_path = pick_fixed_image(img_dir)
    img = Image.open(img_path).convert("RGB")

    # Precompute tensor if preprocess time excluded
    if not args.include_preprocess:
        x0 = preprocess(img).unsqueeze(0).to(device)

    # ---- warmup (NOT recorded) ----
    with torch.inference_mode():
        for _ in range(args.warmup):
            if args.include_preprocess:
                x = preprocess(img).unsqueeze(0).to(device)
            else:
                x = x0
            sync_if_cuda(device)
            _ = model(x)
            sync_if_cuda(device)

    # ---- measure ----
    lat_ms = []
    last_out = None

    with torch.inference_mode():
        for i in range(args.repeat):
            if args.include_preprocess:
                sync_if_cuda(device)
                t0 = time.perf_counter()
                x = preprocess(img).unsqueeze(0).to(device)
                sync_if_cuda(device)
                out = model(x)
                sync_if_cuda(device)
                t1 = time.perf_counter()
            else:
                x = x0
                sync_if_cuda(device)
                t0 = time.perf_counter()
                out = model(x)
                sync_if_cuda(device)
                t1 = time.perf_counter()

            last_out = out
            lat_ms.append((t1 - t0) * 1000.0)

    # sanity info from last output (NOT important for timing)
    probs = torch.nn.functional.softmax(last_out, dim=1)[0]
    top_prob, top_idx = torch.max(probs, dim=0)
    categories = weights.meta.get("categories", None)
    label = categories[int(top_idx)] if categories else f"class_{int(top_idx)}"
    checksum = float(last_out[0, int(top_idx)].item())

    p50 = percentile(lat_ms, 50)
    p95 = percentile(lat_ms, 95)
    mn = min(lat_ms)
    mx = max(lat_ms)
    avg = sum(lat_ms) / len(lat_ms)

    print(
        f"[result] image={img_path.name} top1={label} prob_pct={float(top_prob)*100:.2f} "
        f"checksum_logit_top1={checksum:.6f} "
        f"lat_ms: avg={avg:.3f} p50={p50:.3f} p95={p95:.3f} min={mn:.3f} max={mx:.3f} "
        f"include_preprocess={args.include_preprocess}"
    )

    # optionally dump raw latencies (runner가 파싱하기 편하게)
    for i, v in enumerate(lat_ms):
        print(f"[latency_ms] iter={i} value={v:.3f}")

if __name__ == "__main__":
    main()
