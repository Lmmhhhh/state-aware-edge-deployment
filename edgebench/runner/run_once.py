import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


psutil_import_error: Optional[str] = None
try:
    import psutil  # type: ignore
except ImportError as e:
    psutil = None
    psutil_import_error = f"{type(e).__name__}: {e}"


# ---------------------------
# utils
# ---------------------------
def now_iso() -> str: #현재 시간
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: str): #디렉터리 검사
    os.makedirs(p, exist_ok=True)


def is_jetson_runtime() -> bool: #tegrastats 명령어 있으면 jetson 디바이스
    return shutil.which("tegrastats") is not None


def resolve_device(user_device: str) -> str:
    # user_device: auto | jetson | rpi
    if user_device != "auto":
        return user_device
    return "jetson" if is_jetson_runtime() else "rpi"


def build_workload_script(edge_root: str, tier: str, device: str, workload_id: str, variant: str) -> str:
    # edgebench/apps/{tier}/{device}/{workload_id}/{variant}.py
    return os.path.join(edge_root, "apps", tier, device, workload_id, f"{variant}.py")


def exc_info_dict(e: BaseException) -> Dict[str, Any]: #예외 기록
    return {
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc(),
    }


# ---------------------------
# runner logger (thread-safe)
# ---------------------------
class RunnerLogger:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._f = open(path, "w", buffering=1)

    def _write(self, level: str, msg: str):
        ts = now_iso()
        with self._lock:
            self._f.write(f"[{ts}] [{level}] {msg}\n")

    def info(self, msg: str):
        self._write("INFO", msg)

    def warn(self, msg: str):
        self._write("WARN", msg)

    def error(self, msg: str):
        self._write("ERROR", msg)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# ---------------------------
# collectors
# ---------------------------
def psutil_snapshot() -> Optional[Dict[str, Any]]:
    if psutil is None:
        return None
    try:
        # cpu_percent first call warms up elsewhere, but this is fine for snapshot
        return {
            "ts_epoch": time.time(),
            "cpu_util_pct": float(psutil.cpu_percent(interval=None)),
            "mem_util_pct": float(psutil.virtual_memory().percent),
        }
    except Exception:
        return None


def psutil_sampler(out_path: str, interval_s: float, stop_evt: threading.Event,
                   logger: RunnerLogger, errors: List[Dict[str, Any]]):
    """
    Sample during run. Errors are logged to runner.log and appended to meta["errors"].
    """
    if psutil is None:
        logger.warn(f"psutil not available. sampler skipped. import_error={psutil_import_error}")
        return

    try:
        # warm-up to avoid initial spike
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        with open(out_path, "w", buffering=1) as f:
            f.write("ts_epoch,cpu_util_pct,mem_util_pct\n")
            while not stop_evt.is_set():
                ts = time.time()
                try:
                    cpu = psutil.cpu_percent(interval=None)
                except Exception as e:
                    cpu = float("nan")
                    info = exc_info_dict(e)
                    info["where"] = "psutil.cpu_percent"
                    errors.append(info)
                    logger.error(f"psutil cpu_percent failed: {info['error_type']}: {info['error_message']}")

                try:
                    mem = psutil.virtual_memory().percent
                except Exception as e:
                    mem = float("nan")
                    info = exc_info_dict(e)
                    info["where"] = "psutil.virtual_memory"
                    errors.append(info)
                    logger.error(f"psutil virtual_memory failed: {info['error_type']}: {info['error_message']}")

                f.write(f"{ts:.3f},{cpu:.2f},{mem:.2f}\n")
                stop_evt.wait(interval_s)

    except Exception as e:
        info = exc_info_dict(e)
        info["where"] = "psutil_sampler_thread"
        errors.append(info)
        logger.error(f"psutil sampler thread crashed: {info['error_type']}: {info['error_message']}")


def start_tegrastats_with_ts(out_path: str, interval_ms: int,
                            logger: RunnerLogger, errors: List[Dict[str, Any]]):
    """
    Write tegrastats with ts_epoch:
      <ts_epoch>\t<tegrastats raw line>
    """
    p = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stop_evt = threading.Event()

    def _pump():
        try:
            assert p.stdout is not None
            with open(out_path, "w", buffering=1) as f:
                for line in p.stdout:
                    if stop_evt.is_set():
                        break
                    ts = time.time()
                    f.write(f"{ts:.3f}\t{line.rstrip()}\n")
        except Exception as e:
            info = exc_info_dict(e)
            info["where"] = "tegrastats_pump_thread"
            errors.append(info)
            logger.error(f"tegrastats pump crashed: {info['error_type']}: {info['error_message']}")

    th = threading.Thread(target=_pump, daemon=True)
    th.start()
    return p, stop_evt, th


# ---------------------------
# aggregation helpers
# ---------------------------
def percentile(values: List[float], q: float) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return float("nan")
    vals.sort()
    if len(vals) == 1:
        return vals[0]
    idx = int(math.ceil(q * (len(vals) - 1)))
    return vals[idx]


def aggregate_psutil_csv(psutil_path: str) -> Dict[str, Any]:
    if (not psutil_path) or (psutil_path == "NA") or (not os.path.exists(psutil_path)):
        return {
            "psutil_sample_count": 0,
            "cpu_util_avg": "NA",
            "mem_util_avg": "NA",
            "mem_util_p95": "NA",
        }

    cpu_vals: List[float] = []
    mem_vals: List[float] = []
    n = 0

    with open(psutil_path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                cpu = float(parts[1])
                mem = float(parts[2])
            except Exception:
                continue
            cpu_vals.append(cpu)
            mem_vals.append(mem)
            n += 1

    if n == 0:
        return {
            "psutil_sample_count": 0,
            "cpu_util_avg": "NA",
            "mem_util_avg": "NA",
            "mem_util_p95": "NA",
        }

    cpu_avg = sum(cpu_vals) / n
    mem_avg = sum(mem_vals) / n
    mem_p95 = percentile(mem_vals, 0.95)

    return {
        "psutil_sample_count": n,
        "cpu_util_avg": f"{cpu_avg:.3f}",
        "mem_util_avg": f"{mem_avg:.3f}",
        "mem_util_p95": f"{mem_p95:.3f}",
    }


POWER_PATTERNS = [
    ("VIN_SYS_5V", re.compile(r"VIN\s+SYS_5V\s+(\d+)mW(?:/(\d+)mW)?")),
    ("VDD_IN", re.compile(r"\bVDD_IN\s+(\d+)mW(?:/(\d+)mW)?")),
    ("POM_5V_IN",  re.compile(r"\bPOM_5V_IN\s+(\d+)mW(?:/(\d+)mW)?")),
]

TEMP_CPU_PAT = re.compile(r"\bcpu@([0-9.]+)C")
TEMP_GPU_PAT = re.compile(r"\bgpu@([0-9.]+)C|\bGPU@([0-9.]+)C")


def parse_power_mw(raw_line: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    for name, pat in POWER_PATTERNS:
        m = pat.search(raw_line)
        if m:
            cur = float(m.group(1))
            avg = float(m.group(2)) if m.group(2) else None
            return cur, avg, name
    return None, None, None


def aggregate_tegrastats(tegrastats_path: str, t0_epoch: float, t1_epoch: float) -> Dict[str, Any]:
    if (not tegrastats_path) or (tegrastats_path == "NA") or (not os.path.exists(tegrastats_path)):
        return {
            "tegra_sample_count": 0,
            "power_source": "NA",
            "energy_j": "NA",
            "avg_power_w_e2e": "NA",
            "avg_power_w_measured": "NA",
            "temp_cpu_max_c": "NA",
            "temp_gpu_max_c": "NA",
        }

    samples: List[Tuple[float, float]] = []  # (ts, power_w)
    temp_cpu_max: Optional[float] = None
    temp_gpu_max: Optional[float] = None
    power_source: Optional[str] = None

    with open(tegrastats_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            ts_s, raw = line.split("\t", 1)
            try:
                ts = float(ts_s)
            except Exception:
                continue

            if ts < t0_epoch or ts > t1_epoch:
                continue

            cur_mw, avg_mw_field, src = parse_power_mw(raw)
            if cur_mw is not None:
                samples.append((ts, cur_mw / 1000.0))
                power_source = power_source or src

            m1 = TEMP_CPU_PAT.search(raw)
            if m1:
                try:
                    v = float(m1.group(1))
                    temp_cpu_max = v if (temp_cpu_max is None or v > temp_cpu_max) else temp_cpu_max
                except Exception:
                    pass

            m2 = TEMP_GPU_PAT.search(raw)
            if m2:
                try:
                    v = float(m2.group(1) or m2.group(2))
                    temp_gpu_max = v if (temp_gpu_max is None or v > temp_gpu_max) else temp_gpu_max
                except Exception:
                    pass

    if len(samples) < 2:
        return {
            "tegra_sample_count": len(samples),
            "power_source": power_source or "NA",
            "energy_j": "NA",
            "avg_power_w_e2e": "NA",
            "avg_power_w_measured": "NA",
            "temp_cpu_max_c": f"{temp_cpu_max:.3f}" if temp_cpu_max is not None else "NA",
            "temp_gpu_max_c": f"{temp_gpu_max:.3f}" if temp_gpu_max is not None else "NA",
        }

    samples.sort(key=lambda x: x[0])

    # energy_j = trapezoid integration: Σ (p[k] + p[k+1]) / 2 * Δt
    energy_j = 0.0
    for i in range(len(samples) - 1):
        ts0, p0 = samples[i]
        ts1, p1 = samples[i + 1]
        dt = ts1 - ts0
        if dt <= 0:
            continue
        energy_j += (p0 + p1) * 0.5 * dt

    e2e_duration_s = max(1e-6, (t1_epoch - t0_epoch))  # runner 기준
    ts_first = samples[0][0]
    ts_last = samples[-1][0]
    measured_duration_s = max(1e-6, (ts_last - ts_first))  # tegrastats 기준

    avg_power_w_e2e = energy_j / e2e_duration_s
    avg_power_w_measured = energy_j / measured_duration_s

    return {
        "tegra_sample_count": len(samples),
        "power_source": power_source or "NA",
        "energy_j": f"{energy_j:.6f}",
        "avg_power_w_e2e": f"{avg_power_w_e2e:.6f}",
        "avg_power_w_measured": f"{avg_power_w_measured:.6f}",
        "temp_cpu_max_c": f"{temp_cpu_max:.3f}" if temp_cpu_max is not None else "NA",
        "temp_gpu_max_c": f"{temp_gpu_max:.3f}" if temp_gpu_max is not None else "NA",
    }

def append_run_csv(csv_path: str, row: Dict[str, Any], fieldnames: List[str]):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # core (single, reproducible mode)
    ap.add_argument("--device", default="auto", choices=["auto", "jetson", "rpi"])
    ap.add_argument("--tier", default="micro", choices=["micro", "app"])
    ap.add_argument("--workload_id", required=True)
    ap.add_argument("--variant", required=True)  # low/medium/high

    # workload meta (kept for schema / ML features)
    ap.add_argument("--workload_category", default="cpu")
    ap.add_argument("--input_size", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gpu_used", type=int, default=0)
    ap.add_argument("--model_size_mb", type=float, default=-1.0)
    ap.add_argument("--precision", default="NA")
    ap.add_argument("--flops", type=float, default=-1.0)

    # sampling
    ap.add_argument("--state_interval_s", type=float, default=1.0)

    # pass-through args to workload
    # usage: ... -- --arg1 v1 --arg2 v2
    ap.add_argument("script_args", nargs=argparse.REMAINDER)

    args = ap.parse_args()

    EDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    device = resolve_device(args.device)

    run_id = f"{args.workload_id}-{args.variant}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    base_dir = os.path.join(EDGE_ROOT, "logs")
    raw_stdout_dir = os.path.join(base_dir, "raw", "stdout")
    raw_tegrastats_dir = os.path.join(base_dir, "raw", "tegrastats")
    raw_psutil_dir = os.path.join(base_dir, "raw", "psutil")
    runs_dir = os.path.join(base_dir, "runs")
    runs_meta_dir = os.path.json(runs_dir, "meta")
    runs_log_dir = os.path.json(runs_dir, "log")
    runs_csv = os.path.join(runs_dir, "runs.csv")

    for d in [raw_stdout_dir, raw_tegrastats_dir, raw_psutil_dir, runs_dir, runs_meta_dir, runs_log_dir]:
        ensure_dir(d)

    stdout_path = os.path.join(raw_stdout_dir, f"{run_id}.log")
    tegra_path = os.path.join(raw_tegrastats_dir, f"{run_id}.log")
    psutil_path = os.path.join(raw_psutil_dir, f"{run_id}.csv")
    meta_path = os.path.join(runs_meta_dir, f"{run_id}.meta.json")
    runner_log_path = os.path.join(runs_log_dir, f"{run_id}.runner.log")

    logger = RunnerLogger(runner_log_path)
    meta_errors: List[Dict[str, Any]] = []

    logger.info(f"run_id={run_id}")
    logger.info(f"device={device} tier={args.tier}")
    if psutil is None:
        logger.warn(f"psutil import failed: {psutil_import_error}")

    # determine workload script path by convention (no --cmd)
    script_path = build_workload_script(
        EDGE_ROOT, args.tier, device, args.workload_id, args.variant
    )
    if not os.path.exists(script_path):
        e = FileNotFoundError(
            f"workload script missing: {script_path} "
            f"(expected: edgebench/apps/{args.tier}/{device}/{args.workload_id}/{args.variant}.py)"
        )
        meta_errors.append({**exc_info_dict(e), "where": "build_workload_script"})
        logger.error(str(e))
        raise e

    extra = args.script_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd = ["python", script_path] + extra

    logger.info(f"script_path={script_path}")
    logger.info(f"cmd={' '.join(cmd)}")

    # timestamps
    t0_epoch = time.time()
    t0_mono = time.perf_counter()

    # meta
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp_start": now_iso(),
        "t0_epoch": t0_epoch,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device": device,
        "tier": args.tier,
        "script_path": script_path,
        "cmd": cmd,
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
        "paths": {
            "stdout": stdout_path,
            "tegrastats": tegra_path if (device == "jetson" and is_jetson_runtime()) else None,
            "psutil": psutil_path if psutil is not None else None,
            "meta": meta_path,
            "runner_log": runner_log_path,
        },
        "psutil_import_error": psutil_import_error,
        "errors": meta_errors,
    }

    # optional before snapshot (meta only)
    meta["state_before"] = psutil_snapshot()

    # start collectors
    tegra_proc = None
    tegra_stop = None
    tegra_thread = None

    if device == "jetson" and is_jetson_runtime():
        try:
            tegra_proc, tegra_stop, tegra_thread = start_tegrastats_with_ts(
                tegra_path,
                interval_ms=int(args.state_interval_s * 1000),
                logger=logger,
                errors=meta_errors,
            )
            logger.info("tegrastats started")
        except Exception as e:
            info = exc_info_dict(e)
            info["where"] = "start_tegrastats_with_ts"
            meta_errors.append(info)
            logger.error(f"failed to start tegrastats: {info['error_type']}: {info['error_message']}")

    ps_stop = threading.Event()
    ps_thread = None
    if psutil is not None:
        try:
            ps_thread = threading.Thread(
                target=psutil_sampler,
                args=(psutil_path, args.state_interval_s, ps_stop, logger, meta_errors),
                daemon=True,
            )
            ps_thread.start()
            logger.info("psutil sampler started")
        except Exception as e:
            info = exc_info_dict(e)
            info["where"] = "start_psutil_sampler_thread"
            meta_errors.append(info)
            logger.error(f"failed to start psutil sampler: {info['error_type']}: {info['error_message']}")

    ret = -1
    workload_exc: Optional[Dict[str, Any]] = None

    try:
        with open(stdout_path, "w", buffering=1) as out:
            out.write(f"[edgebench] run_id={run_id}\n")
            out.write(f"[edgebench] start={meta['timestamp_start']}\n")
            out.write(f"[edgebench] device={device} tier={args.tier}\n")
            out.write(f"[edgebench] script={script_path}\n")
            out.write(f"[edgebench] cmd={' '.join(cmd)}\n\n")

            p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, text=True)
            ret = p.wait()

    except Exception as e:
        workload_exc = exc_info_dict(e)
        workload_exc["where"] = "subprocess_run_workload"
        meta_errors.append(workload_exc)
        logger.error(f"workload execution failed: {workload_exc['error_type']}: {workload_exc['error_message']}")

    finally:
        # after snapshot (meta only)
        meta["state_after"] = psutil_snapshot()

        # stop collectors
        try:
            ps_stop.set()
            if ps_thread is not None:
                ps_thread.join(timeout=3.0)
            logger.info("psutil sampler stopped")
        except Exception as e:
            info = exc_info_dict(e)
            info["where"] = "stop_psutil_sampler"
            meta_errors.append(info)
            logger.error(f"failed stopping psutil sampler: {info['error_type']}: {info['error_message']}")

        try:
            if tegra_stop is not None:
                tegra_stop.set()
            if tegra_proc is not None:
                tegra_proc.terminate()
            if tegra_thread is not None:
                tegra_thread.join(timeout=3.0)
            logger.info("tegrastats stopped")
        except Exception as e:
            info = exc_info_dict(e)
            info["where"] = "stop_tegrastats"
            meta_errors.append(info)
            logger.error(f"failed stopping tegrastats: {info['error_type']}: {info['error_message']}")

    # end timestamps
    t1_epoch = time.time()
    t1_mono = time.perf_counter()

    meta["timestamp_end"] = now_iso()
    meta["t1_epoch"] = t1_epoch
    meta["return_code"] = ret

    duration_s = max(1e-6, (t1_mono - t0_mono))
    latency_ms = duration_s * 1000.0

    # aggregates -> per-run values
    ps_agg = aggregate_psutil_csv(psutil_path if psutil is not None else "NA")

    tegra_agg = {
        "tegra_sample_count": 0,
        "power_source": "NA",
        "avg_power_w": "NA",
        "energy_j": "NA",
        "temp_cpu_max_c": "NA",
        "temp_gpu_max_c": "NA",
    }
    if device == "jetson" and is_jetson_runtime():
        try:
            tegra_agg = aggregate_tegrastats(tegrastats_path=tegra_path, t0_epoch=t0_epoch, t1_epoch=t1_epoch)
        except Exception as e:
            info = exc_info_dict(e)
            info["where"] = "aggregate_tegrastats"
            meta_errors.append(info)
            logger.error(f"tegrastats aggregate failed: {info['error_type']}: {info['error_message']}")

    status = "success" if (ret == 0 and workload_exc is None) else "fail"

    meta["status"] = status
    meta["latency_ms"] = float(f"{latency_ms:.3f}")
    meta["aggregates"] = {"psutil": ps_agg, "tegrastats": tegra_agg}

    # write meta
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # per-run CSV row (fixed-ish schema)
    row = {
        "run_id": run_id,
        "device": device,
        "tier": args.tier,
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

        "energy_j": tegra_agg.get("energy_j", "NA"),
        "avg_power_w__avg": tegra_agg.get("avg_power_w_e2e", "NA"),
        "avg_power_w__measured": tegra_agg.get("avg_power_w_measured", "NA"),

        "cpu_util__avg": ps_agg["cpu_util_avg"],
        "cpu_util__max": ps_agg["cpu_util_max"],
        "mem_util__avg": ps_agg["mem_util_avg"],
        "mem_util__p95": ps_agg["mem_util_p95"],

        "temp_cpu_max_c": tegra_agg["temp_cpu_max_c"],
        "temp_gpu_max_c": tegra_agg["temp_gpu_max_c"],

        "status": status,
        "return_code": ret,
        "psutil_sample_count": ps_agg["psutil_sample_count"],
        "tegra_sample_count": tegra_agg["tegra_sample_count"],
        "power_source": tegra_agg["power_source"],
        "error_type": workload_exc["error_type"] if workload_exc else "NA",

        "stdout_path": stdout_path,
        "psutil_path": psutil_path if psutil is not None else "NA",
        "tegrastats_path": tegra_path if (device == "jetson" and is_jetson_runtime()) else "NA",
        "meta_path": meta_path,
        "runner_log_path": runner_log_path,
    }

    append_run_csv(runs_csv, row, list(row.keys()))

    logger.info(f"done status={status} latency_ms={latency_ms:.3f} ret={ret}")
    logger.info(f"wrote runs.csv: {runs_csv}")
    logger.info(f"stdout: {stdout_path}")
    logger.close()

    print(f"[edgebench] done run_id={run_id} status={status} latency_ms={latency_ms:.3f} ret={ret}")
    print(f"[edgebench] wrote: {runs_csv}")
    print(f"[edgebench] raw: {stdout_path}")
    print(f"[edgebench] runner log: {runner_log_path}")

if __name__ == "__main__":
    main()
