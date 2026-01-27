import os
import re
import time
import subprocess


# ---- defaults
WRITE_INPUT = "/dev/zero"
WRITE_OUTPUT = "dd_output.bin"     # 현재 작업 디렉토리에 생성됨
WRITE_BS = "1024k"
WRITE_COUNT = "500"
WRITE_OFLAG = "dsync"              # 원본 벤치마크 유지

READ_INPUT = WRITE_OUTPUT
READ_OUTPUT = "/dev/null"
READ_BS = "1024k"
READ_COUNT = "10"


DD_LINE_RE = re.compile(r",\s*([0-9.]+)\s*s,\s*([0-9.]+)\s*([kMGT]?B/s)", re.IGNORECASE)


def run_dd(cmd: list[str]) -> tuple[int, str]:
    """
    dd는 진행/결과를 stderr로 많이 뿌린다.
    stderr를 캡처해서 반환.
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (p.stderr or "").strip()
    return p.returncode, out


def parse_dd_summary(stderr_text: str) -> dict:
    """
    dd stderr 마지막 결과 라인에서 time, throughput을 파싱.
    예:
      "104857600 bytes (105 MB, 100 MiB) copied, 0.123 s, 852 MB/s"
    """
    lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""

    m = DD_LINE_RE.search(last)
    if not m:
        return {
            "raw": last,
            "time_s": "NA",
            "throughput": "NA",
        }

    return {
        "raw": last,
        "time_s": float(m.group(1)),
        "throughput": f"{m.group(2)} {m.group(3)}",
    }


def main():
    t0 = time.perf_counter()
    # --- WRITE TEST
    write_cmd = [
        "dd",
        f"if={WRITE_INPUT}",
        f"of={WRITE_OUTPUT}",
        f"bs={WRITE_BS}",
        f"count={WRITE_COUNT}",
        f"oflag={WRITE_OFLAG}",
    ]

    rc_w, w_stderr = run_dd(write_cmd)
    w_sum = parse_dd_summary(w_stderr)

    # --- READ TEST
    read_cmd = [
        "dd",
        f"if={READ_INPUT}",
        f"of={READ_OUTPUT}",
        f"bs={READ_BS}",
        f"count={READ_COUNT}",
    ]

    rc_r, r_stderr = run_dd(read_cmd)
    r_sum = parse_dd_summary(r_stderr)

    t1 = time.perf_counter()
    inner_elapsed_s = t1 - t0

    status = "success" if (rc_w == 0 and rc_r == 0) else "fail"

    print(
        "[dd_cmd] "
        f"status={status} inner_elapsed_s={inner_elapsed_s:.4f} "
        f"write_time_s={w_sum['time_s']} write_bw={w_sum['throughput']} "
        f"read_time_s={r_sum['time_s']} read_bw={r_sum['throughput']} "
        f"write_rc={rc_w} read_rc={rc_r}"
    )

    print("[dd_cmd] write_raw:", w_sum["raw"])
    print("[dd_cmd] read_raw:", r_sum["raw"])

    try:
        if os.path.exists(WRITE_OUTPUT):
            os.remove(WRITE_OUTPUT)
    except Exception:
        pass

    if status != "success":
        if rc_w != 0:
            print("[dd_cmd] write_stderr:", w_stderr)
        if rc_r != 0:
            print("[dd_cmd] read_stderr:", r_stderr)


if __name__ == "__main__":
    main()