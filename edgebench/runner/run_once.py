import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def is_jetson():
    return shutil.which("tegrastats") is not None


def start_tegrastats(out_path: str, interval_ms: int = 1000):
    # tegrastats 는 옵션이 기기/버전에 따라 다를 수 있어 안정적으로 "파이프로 잘라서" 저장
    f = open(out_path, "w", buffering=1)
    p = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                         stdout=f, stderr=subprocess.STDOUT, text=True)
    return p, f


def start_psutil_sampler(out_path: str, interval_s: float = 1.0):
    f = open(out_path, "w", buffering=1)
    f.write("ts,cpu_util_pct,mem_util_pct\n")
    while True:
        # 이 함수는 subprocess로 띄워서 돌릴 거라 여기서는 안 씀
        time.sleep(interval_s)


def run_psutil_loop(out_path: str, duration_s: float, interval_s: float = 1.0):
    if psutil is None:
        return

    with open(out_path, "w", buffering=1) as f:
        f.write("ts,cpu_util_pct,mem_util_pct\n")
        t_end = time.time() + duration_s
        while time.time() < t_end:
            ts = time.time()
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            f.write(f"{ts:.3f},{cpu:.2f},{mem:.2f}\n")
            time.sleep(interval_s)


def append_run_csv(csv_path: str, row: dict, fieldnames: list[str]):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload_id", required=True)           # e.g., matrix_mul
    ap.add_argument("--workload_category", default="cpu")     # cpu/mem/io/net/gpu
    ap.add_argument("--variant", required=True)               # low/medium/high
    ap.add_argument("--cmd", nargs="+", required=True)        # 실행 커맨드 전체
    ap.add_argument("--input_size", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gpu_used", type=int, default=0)
    ap.add_argument("--model_size_mb", type=float, default=-1.0)
    ap.add_argument("--precision", default="NA")              # FP32/FP16/INT8/NA
    ap.add_argument("--flops", type=float, default=-1.0)

    ap.add_argument("--state_interval_s", type=float, default=1.0)
    args = ap.parse_args()

    run_id = f"{args.workload_id}-{args.variant}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    EDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # .../edgebench
    base_dir = os.path.join(EDGE_ROOT, "logs")                                  # .../edgebench/logs
    raw_stdout_dir = os.path.join(base_dir, "raw", "stdout")
    raw_tegrastats_dir = os.path.join(base_dir, "raw", "tegrastats")
    raw_psutil_dir = os.path.join(base_dir, "raw", "psutil")
    runs_csv = os.path.join(base_dir, "runs", "runs.csv")

    for d in [raw_stdout_dir, raw_tegrastats_dir, raw_psutil_dir, os.path.dirname(runs_csv)]:
        ensure_dir(d)

    # ---- timestamps
    t0 = time.time()
    meta = {
        "run_id": run_id,
        "timestamp_start": now_iso(),
        "t0": t0,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cmd": args.cmd,
        "inputs": {
            "workload_id": args.workload_id,
            "variant": args.variant,
            "workload_category": args.workload_category,
            "input_size": args.input_size,
            "batch_size": args.batch_size,
            "gpu_used": args.gpu_used,
            "model_size_mb": args.model_size_mb,
            "precision": args.precision,
            "flops": args.flops,
        },
        "paths": {},
    }

    # ---- start state collectors
    tegra_proc = None
    tegra_f = None
    tegra_path = os.path.join(raw_tegrastats_dir, f"{run_id}.log")
    psutil_path = os.path.join(raw_psutil_dir, f"{run_id}.csv")
    stdout_path = os.path.join(raw_stdout_dir, f"{run_id}.log")

    meta["paths"]["stdout"] = stdout_path
    meta["paths"]["tegrastats"] = tegra_path if is_jetson() else None
    meta["paths"]["psutil"] = psutil_path if psutil is not None else None
    meta_path = os.path.join(base_dir, "runs", f"{run_id}.meta.json")
    meta["paths"]["meta"] = meta_path

    # tegrastats는 실행 시간 동안만 켜두고, psutil은 종료 후 duration으로 재생성(간단/안전)
    if is_jetson():
        tegra_proc, tegra_f = start_tegrastats(tegra_path, interval_ms=int(args.state_interval_s * 1000))

    # ---- run workload, capture stdout/stderr
    with open(stdout_path, "w", buffering=1) as out:
        out.write(f"[edgebench] run_id={run_id}\n")
        out.write(f"[edgebench] start={meta['timestamp_start']}\n")
        out.write(f"[edgebench] cmd={' '.join(args.cmd)}\n\n")

        p = subprocess.Popen(args.cmd, stdout=out, stderr=subprocess.STDOUT, text=True)
        ret = p.wait()

    t1 = time.time()
    meta["timestamp_end"] = now_iso()
    meta["t1"] = t1
    meta["return_code"] = ret

    # ---- stop collectors
    if tegra_proc is not None:
        try:
            tegra_proc.terminate()
        except Exception:
            pass
        try:
            tegra_f.close()
        except Exception:
            pass

    # ---- psutil sampling for the exact duration (post-hoc)
    duration_s = max(0.001, t1 - t0)
    if psutil is not None:
        run_psutil_loop(psutil_path, duration_s=duration_s, interval_s=args.state_interval_s)

    # ---- write meta json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ---- per-run CSV row (집계 최소 버전: latency_ms만 먼저)
    latency_ms = (t1 - t0) * 1000.0

    row = {
        "run_id": run_id,
        "workload_id": args.workload_id,
        "variant": args.variant,
        "workload_category": args.workload_category,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "gpu_used": args.gpu_used,
        "model_size_mb": args.model_size_mb,
        "precision": args.precision,
        "flops": args.flops,
        "latency_ms": f"{latency_ms:.3f}",
        "stdout_path": stdout_path,
        "tegrastats_path": tegra_path if is_jetson() else "NA",
        "psutil_path": psutil_path if psutil is not None else "NA",
        "return_code": ret,
    }

    fieldnames = list(row.keys())
    append_run_csv(runs_csv, row, fieldnames)

    print(f"[edgebench] done run_id={run_id} latency_ms={latency_ms:.3f} ret={ret}")
    print(f"[edgebench] wrote: {runs_csv}")
    print(f"[edgebench] raw: {stdout_path}")


if __name__ == "__main__":
    main()