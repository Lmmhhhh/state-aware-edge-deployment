# cpu_load.py
import argparse
import math
import os
import time
from multiprocessing import Process, Event

def burn(stop: Event, intensity: float):
    """
    intensity: 0.0~1.0  (1.0=최대한 태움)
    duty-cycle로 과열/불안정 방지.
    """
    x = 0.0001
    period = 0.05  # 50ms
    busy = max(0.0, min(1.0, intensity)) * period
    idle = period - busy

    while not stop.is_set():
        t0 = time.perf_counter()
        # busy loop
        while (time.perf_counter() - t0) < busy:
            x = math.sin(x) * math.cos(x) + 0.000001
        if idle > 0:
            time.sleep(idle)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None, help="Default: cpu_count()-1 (min 1)")
    ap.add_argument("--intensity", type=float, default=0.7, help="0.0~1.0, default 0.7")
    ap.add_argument("--duration_s", type=float, default=60.0, help="how long to run")
    args = ap.parse_args()

    cpu = os.cpu_count() or 2
    workers = args.workers if args.workers is not None else max(1, cpu - 1)

    stop = Event()
    procs = [Process(target=burn, args=(stop, args.intensity), daemon=True) for _ in range(workers)]
    for p in procs:
        p.start()

    print(f"[cpu_load] workers={workers} intensity={args.intensity} duration_s={args.duration_s}")
    try:
        time.sleep(args.duration_s)
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=1.0)
        print("[cpu_load] done")

if __name__ == "__main__":
    main()
