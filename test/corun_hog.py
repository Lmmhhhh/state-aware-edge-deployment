import argparse
import os
import time
import threading
import torch

def set_torch_threads(n: int):
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, n // 2))
    except Exception:
        pass

def hog_matmul(stop_event: threading.Event, n: int, dtype: torch.dtype):
    # 큰 matmul로 메모리/연산 압박 (CPU)
    a = torch.randn((n, n), dtype=dtype)
    b = torch.randn((n, n), dtype=dtype)
    while not stop_event.is_set():
        _ = a @ b  # noqa: F841

def hog_fft(stop_event: threading.Event, n: int, dtype: torch.dtype):
    # FFT로 메모리/캐시 패턴 다르게 압박 (CPU)
    x = torch.randn((n,), dtype=dtype)
    while not stop_event.is_set():
        _ = torch.fft.rfft(x)  # noqa: F841

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["matmul", "fft"], default="matmul")
    ap.add_argument("--workers", type=int, default=2, help="number of threads running hog")
    ap.add_argument("--n", type=int, default=2048, help="matmul matrix size or fft vector size")
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("--duration_s", type=float, default=40.0)
    ap.add_argument("--torch_threads", type=int, default=4, help="torch CPU threads")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.float64
    set_torch_threads(args.torch_threads)

    stop_event = threading.Event()
    threads = []

    fn = hog_matmul if args.mode == "matmul" else hog_fft

    print(f"[corun_hog] mode={args.mode} workers={args.workers} n={args.n} dtype={args.dtype} "
          f"torch_threads={args.torch_threads} duration_s={args.duration_s}")

    for i in range(args.workers):
        t = threading.Thread(target=fn, args=(stop_event, args.n, dtype), daemon=True)
        t.start()
        threads.append(t)

    try:
        time.sleep(args.duration_s)
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        print("[corun_hog] done")

if __name__ == "__main__":
    main()
