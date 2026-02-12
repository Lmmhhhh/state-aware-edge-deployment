
import argparse
import csv
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Optional, Tuple

# -----------------------------
# helpers
# -----------------------------
def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

# -----------------------------
# tegrastats power sampling
# -----------------------------
@dataclass
class PowerSample:
    t: float  # perf_counter seconds
    p_w: float

class TegraStatsSampler:
    """
    Samples power from tegrastats output during workload execution.
    Parses a chosen rail like POM_5V_IN or VDD_IN. Many Jetsons report:
      "POM_5V_IN 4066/4066"
    in mW. We take the first number before '/' and convert to W.
    """
    def __init__(self, interval_ms: int = 100, rail: str = "auto"):
        self.interval_ms = interval_ms
        self.rail = rail  # "auto" or exact label
        self.proc: Optional[subprocess.Popen] = None
        self.samples: List[PowerSample] = []
        self._buf: List[str] = []

        # Example patterns:
        # "POM_5V_IN 4066/4066"
        # "VDD_IN 5800/5800"
        # Sometimes there are units, sometimes not; assume mW if value is big.
        self._rail_pat = re.compile(r"(?P<label>[A-Z0-9_]+)\s+(?P<v1>\d+)(?:/(?P<v2>\d+))?")

    def start(self):
        cmd = ["tegrastats", f"--interval", str(self.interval_ms)]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

    def stop_and_collect(self, timeout_s: float = 1.0):
        if not self.proc:
            return
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout_s)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

        out = ""
        if self.proc.stdout:
            try:
                out = self.proc.stdout.read()
            except Exception:
                out = ""

        # Parse samples line-by-line; use timestamp when parsing as "now"
        # (best-effort; for energy integration, relative timing is more important)
        for line in out.splitlines():
            p = self._parse_power_w(line)
            if p is not None:
                self.samples.append(PowerSample(time.perf_counter(), p))

    def _parse_power_w(self, line: str) -> Optional[float]:
        # Find all "LABEL n/n" tokens and pick a rail.
        matches = list(self._rail_pat.finditer(line))
        if not matches:
            return None

        candidates: List[Tuple[str, int]] = []
        for m in matches:
            label = m.group("label")
            v1 = int(m.group("v1"))
            # Heuristic: tegrastats power rails are often named POM_*, VDD_*
            if label.startswith("POM_") or label.startswith("VDD_"):
                candidates.append((label, v1))

        if not candidates:
            return None

        chosen_label, chosen_val = None, None

        if self.rail != "auto":
            for lab, val in candidates:
                if lab == self.rail:
                    chosen_label, chosen_val = lab, val
                    break
            if chosen_val is None:
                return None
        else:
            # Auto priority: POM_5V_IN -> VDD_IN -> first candidate
            priority = ["POM_5V_IN", "VDD_IN"]
            for pr in priority:
                for lab, val in candidates:
                    if lab == pr:
                        chosen_label, chosen_val = lab, val
                        break
                if chosen_val is not None:
                    break
            if chosen_val is None:
                chosen_label, chosen_val = candidates[0]

        # Convert to Watts.
        # tegrastats rails are typically in mW. If value is small (<200), it might already be W.
        if chosen_val >= 200:
            return chosen_val / 1000.0
        return float(chosen_val)

def integrate_energy_j(samples: List[PowerSample]) -> Tuple[float, float]:
    """
    Returns (avg_power_w, energy_j) using trapezoidal integration over time.
    If timing is too sparse, falls back to mean power * duration.
    """
    if len(samples) < 2:
        return float("nan"), float("nan")

    # Sort by time
    xs = sorted(samples, key=lambda s: s.t)
    t0, t1 = xs[0].t, xs[-1].t
    dur = max(1e-9, t1 - t0)

    # Trapezoid
    e = 0.0
    for i in range(1, len(xs)):
        dt = xs[i].t - xs[i-1].t
        if dt <= 0:
            continue
        e += 0.5 * (xs[i].p_w + xs[i-1].p_w) * dt

    avg_p = e / dur
    return avg_p, e

# -----------------------------
# runner
# -----------------------------
@dataclass
class RunRow:
    run_idx: int
    ok: bool
    return_code: int
    latency_ms: float
    avg_power_w: float
    energy_j: float
    stdout_path: str
    stderr_path: str
    power_log_path: str

def run_workload_with_power(
    cmd: List[str],
    cwd: Optional[str],
    timeout_s: Optional[float],
    stdout_path: Path,
    stderr_path: Path,
    power_backend: str,
    tegra_interval_ms: int,
    tegra_rail: str,
    power_log_path: Path,
) -> Tuple[bool, int, float, float, float]:
    """
    Returns: ok, rc, latency_ms, avg_power_w, energy_j
    """
    sampler: Optional[TegraStatsSampler] = None

    if power_backend == "tegrastats":
        sampler = TegraStatsSampler(interval_ms=tegra_interval_ms, rail=tegra_rail)
        sampler.start()

    t0 = time.perf_counter()
    ok, rc = False, 0
    try:
        with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
            cp = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=out_f,
                stderr=err_f,
                timeout=timeout_s,
                check=False,
            )
        ok = (cp.returncode == 0)
        rc = cp.returncode
    except subprocess.TimeoutExpired:
        ok = False
        rc = 124
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    avg_p, e_j = float("nan"), float("nan")

    if sampler is not None:
        sampler.stop_and_collect()
        # Save raw samples as CSV for debugging
        with power_log_path.open("w", encoding="utf-8") as f:
            f.write("t,p_w\n")
            for s in sampler.samples:
                f.write(f"{s.t:.6f},{s.p_w:.6f}\n")
        avg_p, e_j = integrate_energy_j(sampler.samples)

    return ok, rc, latency_ms, avg_p, e_j

def main():
    ap = argparse.ArgumentParser(description="Measure latency + power (Jetson tegrastats) and write CSV.")
    ap.add_argument("--device", default="auto", help="csv 기록용: auto|jetson|rpi")
    ap.add_argument("--workload_id", required=True, help="예: micro/matmul, app/yolov8s")
    ap.add_argument("--variant", default="default", help="예: low|medium|high|cpu|cuda")
    ap.add_argument("--entry", required=True, help="실행할 엔트리 파일 경로(.py)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout_s", type=float, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--cwd", default=None)

    ap.add_argument("--power_backend", choices=["none", "tegrastats"], default="none")
    ap.add_argument("--tegra_interval_ms", type=int, default=100)
    ap.add_argument("--tegra_rail", default="auto", help="auto or rail label like POM_5V_IN, VDD_IN")

    ap.add_argument("--out_dir", default="edgebench/logs/run")
    ap.add_argument("--csv", default="edgebench/logs/run/results.csv")
    args, unknown = ap.parse_known_args()

    out_dir = Path(args.out_dir) / f"{args.workload_id.replace('/','_')}-{args.variant}-{now_iso().replace(':','')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    entry = Path(args.entry)
    cmd = [args.python, str(entry)] + unknown

    # Warmup (not recorded)
    for i in range(args.warmup):
        sp = out_dir / f"warmup_{i+1}.stdout.log"
        ep = out_dir / f"warmup_{i+1}.stderr.log"
        pp = out_dir / f"warmup_{i+1}.power.csv"
        run_workload_with_power(
            cmd, args.cwd, args.timeout_s, sp, ep,
            args.power_backend, args.tegra_interval_ms, args.tegra_rail, pp
        )

    rows: List[RunRow] = []
    lat_ok: List[float] = []
    p_ok: List[float] = []
    e_ok: List[float] = []

    for i in range(1, args.runs + 1):
        sp = out_dir / f"run_{i}.stdout.log"
        ep = out_dir / f"run_{i}.stderr.log"
        pp = out_dir / f"run_{i}.power.csv"

        ok, rc, lat_ms, avg_p, e_j = run_workload_with_power(
            cmd, args.cwd, args.timeout_s, sp, ep,
            args.power_backend, args.tegra_interval_ms, args.tegra_rail, pp
        )

        rows.append(RunRow(i, ok, rc, lat_ms, avg_p, e_j, str(sp), str(ep), str(pp)))
        if ok:
            lat_ok.append(lat_ms)
            if not (avg_p != avg_p):  # not NaN
                p_ok.append(avg_p)
            if not (e_j != e_j):
                e_ok.append(e_j)

        print(f"[run {i}/{args.runs}] ok={ok} rc={rc} latency_ms={lat_ms:.2f} avg_power_w={avg_p:.3f} energy_j={e_j:.3f}")

    # Summary
    mean_lat = mean(lat_ok) if lat_ok else float("nan")
    std_lat = pstdev(lat_ok) if len(lat_ok) > 1 else 0.0
    p50_lat = percentile(lat_ok, 50)
    p95_lat = percentile(lat_ok, 95)

    mean_p = mean(p_ok) if p_ok else float("nan")
    mean_e = mean(e_ok) if e_ok else float("nan")

    # Append CSV
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "device", "workload_id", "variant", "run_dir",
                "run_idx", "ok", "return_code",
                "latency_ms", "avg_power_w", "energy_j",
                "stdout_path", "stderr_path", "power_log_path",
                "mean_latency_ms_ok", "std_latency_ms_ok", "p50_latency_ms_ok", "p95_latency_ms_ok",
                "mean_power_w_ok", "mean_energy_j_ok", "n_ok", "n_total"
            ])
        for r in rows:
            w.writerow([
                args.device, args.workload_id, args.variant, str(out_dir),
                r.run_idx, int(r.ok), r.return_code,
                f"{r.latency_ms:.3f}",
                f"{r.avg_power_w:.6f}" if not (r.avg_power_w != r.avg_power_w) else "",
                f"{r.energy_j:.6f}" if not (r.energy_j != r.energy_j) else "",
                r.stdout_path, r.stderr_path, r.power_log_path,
                "", "", "", "", "", "", "", ""
            ])

        # Summary row
        w.writerow([
            args.device, args.workload_id, args.variant, str(out_dir),
            "summary", "", "",
            "", "", "",
            "", "", "",
            f"{mean_lat:.3f}", f"{std_lat:.3f}", f"{p50_lat:.3f}", f"{p95_lat:.3f}",
            f"{mean_p:.6f}" if not (mean_p != mean_p) else "",
            f"{mean_e:.6f}" if not (mean_e != mean_e) else "",
            len(lat_ok), args.runs
        ])

    print("\n=== Summary (successful runs) ===")
    print(f"n_ok={len(lat_ok)}/{args.runs} mean_ms={mean_lat:.2f} p95_ms={p95_lat:.2f}")
    if args.power_backend == "tegrastats":
        print(f"mean_power_w={mean_p:.3f} mean_energy_j={mean_e:.3f}")
    print(f"csv: {csv_path}")
    print(f"run_dir: {out_dir}")

if __name__ == "__main__":
    main()
