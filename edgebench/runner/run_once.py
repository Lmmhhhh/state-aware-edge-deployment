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
def now_iso() -> str:  # 현재 시간
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def is_jetson_runtime() -> bool:
    return shutil.which("tegrastats") is not None


def resolve_device(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    return "jetson" if is_jetson_runtime() else "rpi"


def build_workload_script(edge_root: str, tier: str, device: str, workload_id: str, variant: str) -> str:
    # edgebench/apps/{tier}/{device}/{workload_id}/{variant}.py
    return os.path.join(edge_root, "apps", tier, device, workload_id, f"{variant}.py")


def exc_info_dict(e: BaseException) -> Dict[str, Any]:
    return {
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc(),
    }


def read_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.readline().strip()
    except Exception:
        return None


def get_cpu_governor() -> str:
    """
    Prefer sysfs scaling_governor if present.
    Return "NA" if not available.
    """
    # common path
    candidates = [
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
    ]
    for p in candidates:
        v = read_first_line(p)
        if v:
            return v
    return "NA"


def get_throttle_flag_rpi() -> str:
    """
    Raspberry Pi throttling flag from `vcgencmd get_throttled`.
    Returns "NA" if vcgencmd unavailable.
    """
    if shutil.which("vcgencmd") is None:
        return "NA"
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, stderr=subprocess.STDOUT).strip()
        m = re.search(r"0x([0-9a-fA-F]+)", out)
        if not m:
            return "NA"
        val = int(m.group(1), 16)
        return "1" if val != 0 else "0"
    except Exception:
        return "NA"


def get_power_mode_jetson() -> str:
    """
    Jetson power/performance mode via nvpmodel.
    Returns "NA" if not available.
    """
    if shutil.which("nvpmodel") is None:
        return "NA"
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.STDOUT)
        out = out.strip()
        for line in out.splitlines():
            if "Power Mode" in line or "NV Power Mode" in line:
                return line.strip()
        return out.splitlines()[0].strip() if out else "NA"
    except Exception:
        return "NA"


# ---------------------------
# runner logger
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
        return {
            "ts_epoch": time.time(),
            "cpu_util_pct": float(psutil.cpu_percent(interval=None)),
            "mem_util_pct": float(psutil.virtual_memory().percent),
        }
    except Exception:
        return None


def psutil_sampler(
    out_path: str,
    interval_s: float,
    stop_evt: threading.Event,
    logger: RunnerLogger,
    errors: List[Dict[str, Any]],
    workload_pid: Optional[int] = None,
):
    """
    Sample during run.
    Writes:
      ts_epoch,cpu_util_pct,mem_util_pct,cpu_freq_mhz,proc_rss_mb
    """
    if psutil is None:
        logger.warn(f"psutil not available. sampler skipped. import_error={psutil_import_error}")
        return

    proc = None
    if workload_pid is not None:
        try:
            proc = psutil.Process(workload_pid)
        except Exception:
            proc = None

    try:
        # warm-up to avoid initial spike
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        with open(out_path, "w", buffering=1) as f:
            f.write("ts_epoch,cpu_util_pct,mem_util_pct,cpu_freq_mhz,proc_rss_mb\n")
            while not stop_evt.is_set():
                ts = time.time()

                # cpu util
                try:
                    cpu = float(psutil.cpu_percent(interval=None))
                except Exception as e:
                    cpu = float("nan")
                    info = exc_info_dict(e)
                    info["where"] = "psutil.cpu_percent"
                    errors.append(info)
                    logger.error(f"psutil cpu_percent failed: {info['error_type']}: {info['error_message']}")

                # mem util
                try:
                    mem = float(psutil.virtual_memory().percent)
                except Exception as e:
                    mem = float("nan")
                    info = exc_info_dict(e)
                    info["where"] = "psutil.virtual_memory"
                    errors.append(info)
                    logger.error(f"psutil virtual_memory failed: {info['error_type']}: {info['error_message']}")

                # cpu freq
                try:
                    cf = psutil.cpu_freq()
                    cpu_freq_mhz = float(cf.current) if cf and cf.current is not None else float("nan")
                except Exception:
                    cpu_freq_mhz = float("nan")

                # proc rss
                proc_rss_mb = float("nan")
                if proc is not None:
                    try:
                        rss = proc.memory_info().rss
                        for ch in proc.children(recursive=True):
                            rss += ch.memory_info().rss
                        proc_rss_mb = float(rss) / (1024.0 * 1024.0)
                    except Exception:
                        proc_rss_mb = float("nan")

                f.write(f"{ts:.3f},{cpu:.2f},{mem:.2f},{cpu_freq_mhz:.2f},{proc_rss_mb:.3f}\n")
                stop_evt.wait(interval_s)

    except Exception as e:
        info = exc_info_dict(e)
        info["where"] = "psutil_sampler_thread"
        errors.append(info)
        logger.error(f"psutil sampler thread crashed: {info['error_type']}: {info['error_message']}")


def start_tegrastats_with_ts(out_path: str, interval_ms: int, logger: RunnerLogger, errors: List[Dict[str, Any]]):
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
    """
    Exposes:
      cpu_util_pct__avg, cpu_util_pct__max,
      mem_util_pct__avg, mem_util_pct__p95,
      cpu_freq_mhz__avg,
      memory_footprint_peak_mb (=proc_rss_mb peak)
    """
    if (not psutil_path) or (psutil_path == "NA") or (not os.path.exists(psutil_path)):
        return {
            "psutil_sample_count": 0,
            "cpu_util_pct__avg": "NA",
            "cpu_util_pct__max": "NA",
            "mem_util_pct__avg": "NA",
            "mem_util_pct__p95": "NA",
            "cpu_freq_mhz__avg": "NA",
            "memory_footprint_peak_mb": "NA",
        }

    cpu_vals: List[float] = []
    mem_vals: List[float] = []
    freq_vals: List[float] = []
    rss_vals: List[float] = []
    n = 0

    with open(psutil_path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                cpu = float(parts[1])
                mem = float(parts[2])
                freq = float(parts[3])
                rss = float(parts[4])
            except Exception:
                continue
            cpu_vals.append(cpu)
            mem_vals.append(mem)
            if freq == freq:
                freq_vals.append(freq)
            if rss == rss:
                rss_vals.append(rss)
            n += 1

    if n == 0:
        return {
            "psutil_sample_count": 0,
            "cpu_util_pct__avg": "NA",
            "cpu_util_pct__max": "NA",
            "mem_util_pct__avg": "NA",
            "mem_util_pct__p95": "NA",
            "cpu_freq_mhz__avg": "NA",
            "memory_footprint_peak_mb": "NA",
        }

    cpu_avg = sum(cpu_vals) / n
    cpu_max = max(cpu_vals)
    mem_avg = sum(mem_vals) / n
    mem_p95 = percentile(mem_vals, 0.95)

    cpu_freq_avg = (sum(freq_vals) / len(freq_vals)) if freq_vals else float("nan")
    rss_peak = max(rss_vals) if rss_vals else float("nan")

    return {
        "psutil_sample_count": n,
        "cpu_util_pct__avg": f"{cpu_avg:.3f}",
        "cpu_util_pct__max": f"{cpu_max:.3f}",
        "mem_util_pct__avg": f"{mem_avg:.3f}",
        "mem_util_pct__p95": f"{mem_p95:.3f}",
        "cpu_freq_mhz__avg": f"{cpu_freq_avg:.3f}" if cpu_freq_avg == cpu_freq_avg else "NA",
        "memory_footprint_peak_mb": f"{rss_peak:.3f}" if rss_peak == rss_peak else "NA",
    }


POWER_PATTERNS = [
    ("VIN_SYS_5V", re.compile(r"VIN\s+SYS_5V\s+(\d+)mW(?:/(\d+)mW)?")),
    ("POM_5V_IN", re.compile(r"\bPOM_5V_IN\s+(\d+)mW(?:/(\d+)mW)?")),
    ("VDD_IN", re.compile(r"\bVDD_IN\s+(\d+)mW(?:/(\d+)mW)?")),
]

TEMP_CPU_PAT = re.compile(r"\bcpu@([0-9.]+)C")
TEMP_GPU_PAT = re.compile(r"\bgpu@([0-9.]+)C|\bGPU@([0-9.]+)C")

# GR3D_FREQ 0% or GR3D_FREQ 99% ... sometimes "GR3D_FREQ 0%@..." -> capture leading %
GPU_UTIL_PAT = re.compile(r"\bGR3D_FREQ\s+(\d+)%")

def parse_power_mw(raw_line: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    for name, pat in POWER_PATTERNS:
        m = pat.search(raw_line)
        if m:
            cur = float(m.group(1))
            avg = float(m.group(2)) if m.group(2) else None
            return cur, avg, name
    return None, None, None


def aggregate_tegrastats(tegrastats_path: str, t0_epoch: float, t1_epoch: float) -> Dict[str, Any]:
    """
    Exposes:
      energy_j, avg_power_w__e2e, avg_power_w__measured,
      temp_cpu_c__max, temp_gpu_c__max,
      gpu_util_pct__avg (from GR3D_FREQ %)
    """
    if (not tegrastats_path) or (tegrastats_path == "NA") or (not os.path.exists(tegrastats_path)):
        return {
            "tegra_sample_count": 0,
            "power_source": "NA",
            "energy_j": "NA",
            "avg_power_w__e2e": "NA",
            "avg_power_w__measured": "NA",
            "temp_cpu_c__max": "NA",
            "temp_gpu_c__max": "NA",
            "gpu_util_pct__avg": "NA",
        }

    samples: List[Tuple[float, float]] = []  
    temp_cpu_max: Optional[float] = None
    temp_gpu_max: Optional[float] = None
    power_source: Optional[str] = None

    gpu_utils: List[float] = []

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

            cur_mw, _, src = parse_power_mw(raw)
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

            mg = GPU_UTIL_PAT.search(raw)
            if mg:
                try:
                    gpu_utils.append(float(mg.group(1)))
                except Exception:
                    pass

    # gpu util avg
    gpu_util_avg = (sum(gpu_utils) / len(gpu_utils)) if gpu_utils else float("nan")

    if len(samples) < 2:
        return {
            "tegra_sample_count": len(samples),
            "power_source": power_source or "NA",
            "energy_j": "NA",
            "avg_power_w__e2e": "NA",
            "avg_power_w__measured": "NA",
            "temp_cpu_c__max": f"{temp_cpu_max:.3f}" if temp_cpu_max is not None else "NA",
            "temp_gpu_c__max": f"{temp_gpu_max:.3f}" if temp_gpu_max is not None else "NA",
            "gpu_util_pct__avg": f"{gpu_util_avg:.3f}" if gpu_util_avg == gpu_util_avg else "NA",
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

    avg_power_w__e2e = energy_j / e2e_duration_s
    avg_power_w__measured = energy_j / measured_duration_s

    return {
        "tegra_sample_count": len(samples),
        "power_source": power_source or "NA",
        "energy_j": f"{energy_j:.6f}",
        "avg_power_w__e2e": f"{avg_power_w__e2e:.6f}",
        "avg_power_w__measured": f"{avg_power_w__measured:.6f}",
        "temp_cpu_c__max": f"{temp_cpu_max:.3f}" if temp_cpu_max is not None else "NA",
        "temp_gpu_c__max": f"{temp_gpu_max:.3f}" if temp_gpu_max is not None else "NA",
        "gpu_util_pct__avg": f"{gpu_util_avg:.3f}" if gpu_util_avg == gpu_util_avg else "NA",
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

    # device-level state / experiment knobs
    ap.add_argument("--co_run_count", type=int, default=1) 

    # sampling
    ap.add_argument("--state_interval_s", type=float, default=1.0)

    # pass-through args to workload
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
    runs_meta_dir = os.path.join(runs_dir, "meta")
    runs_log_dir = os.path.join(runs_dir, "log")
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

    # device-level states 
    cpu_governor = get_cpu_governor()
    throttle_flag = get_throttle_flag_rpi() if device == "rpi" else "NA"
    power_mode = get_power_mode_jetson() if device == "jetson" else "NA"

    logger.info(f"cpu_governor={cpu_governor}")
    logger.info(f"throttle_flag={throttle_flag}")
    logger.info(f"power_mode={power_mode}")
    logger.info(f"co_run_count={args.co_run_count}")

    # determine workload script path by convention
    script_path = build_workload_script(EDGE_ROOT, args.tier, device, args.workload_id, args.variant)
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
    cmd = [sys.executable, script_path] + extra

    logger.info(f"script_path={script_path}")
    logger.info(f"cmd={' '.join(cmd)}")

    # timestamps
    t0_epoch = time.time()
    t0_mono = time.perf_counter()

    t1_epoch: Optional[float] = None
    t1_mono: Optional[float] = None

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
        "device_state": {
            "co_run_count": args.co_run_count,
            "power_mode": power_mode,
            "cpu_governor": cpu_governor,
            "throttle_flag": throttle_flag,
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

    # psutil sampler
    ps_stop = threading.Event()
    ps_thread = None

    ret = -1
    workload_exc: Optional[Dict[str, Any]] = None
    workload_pid: Optional[int] = None

    try:
        with open(stdout_path, "w", buffering=1) as out:
            out.write(f"[edgebench] run_id={run_id}\n")
            out.write(f"[edgebench] start={meta['timestamp_start']}\n")
            out.write(f"[edgebench] device={device} tier={args.tier}\n")
            out.write(f"[edgebench] script={script_path}\n")
            out.write(f"[edgebench] cmd={' '.join(cmd)}\n\n")

            p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, text=True)
            workload_pid = p.pid

            # start psutil sampler now
            if psutil is not None:
                try:
                    ps_thread = threading.Thread(
                        target=psutil_sampler,
                        args=(psutil_path, args.state_interval_s, ps_stop, logger, meta_errors, workload_pid),
                        daemon=True,
                    )
                    ps_thread.start()
                    logger.info(f"psutil sampler started (pid={workload_pid})")
                except Exception as e:
                    info = exc_info_dict(e)
                    info["where"] = "start_psutil_sampler_thread"
                    meta_errors.append(info)
                    logger.error(f"failed to start psutil sampler: {info['error_type']}: {info['error_message']}")

            ret = p.wait()

            t1_epoch = time.time()
            t1_mono = time.perf_counter()

    except Exception as e:
        workload_exc = exc_info_dict(e)
        workload_exc["where"] = "subprocess_run_workload"
        meta_errors.append(workload_exc)
        logger.error(f"workload execution failed: {workload_exc['error_type']}: {workload_exc['error_message']}")

        t1_epoch = t1_epoch or time.time()
        t1_mono = t1_mono or time.perf_counter()

    finally:
        # after snapshot
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

    t1_epoch = t1_epoch or time.time()
    t1_mono = t1_mono or time.perf_counter()

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
        "energy_j": "NA",
        "avg_power_w__e2e": "NA",
        "avg_power_w__measured": "NA",
        "temp_cpu_c__max": "NA",
        "temp_gpu_c__max": "NA",
        "gpu_util_pct__avg": "NA",
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

    row = {
        "run_id": run_id,
        "status": status,
        "return_code": ret,
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
        "memory_footprint_peak_mb": ps_agg.get("memory_footprint_peak_mb", "NA"),

    
        "cpu_util_pct__avg": ps_agg.get("cpu_util_pct__avg", "NA"),
        "cpu_util_pct__max": ps_agg.get("cpu_util_pct__max", "NA"),
        "mem_util_pct__avg": ps_agg.get("mem_util_pct__avg", "NA"),
        "mem_util_pct__p95": ps_agg.get("mem_util_pct__p95", "NA"),

        "temp_cpu_c__max": tegra_agg.get("temp_cpu_c__max", "NA"),
        "gpu_util_pct__avg": tegra_agg.get("gpu_util_pct__avg", "NA"),
        "temp_gpu_c__max": tegra_agg.get("temp_gpu_c__max", "NA"),

        "co_run_count": str(args.co_run_count),

        "power_mode": power_mode,
        "cpu_governor": cpu_governor,
        "throttle_flag": throttle_flag,
        "cpu_freq_mhz__avg": ps_agg.get("cpu_freq_mhz__avg", "NA"),

        "latency_ms": f"{latency_ms:.3f}",
        "energy_j": tegra_agg.get("energy_j", "NA"),
        "avg_power_w__avg": tegra_agg.get("avg_power_w__e2e", "NA"),
        "avg_power_w__measured": tegra_agg.get("avg_power_w__measured", "NA"),
        "power_source": tegra_agg.get("power_source", "NA"),
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