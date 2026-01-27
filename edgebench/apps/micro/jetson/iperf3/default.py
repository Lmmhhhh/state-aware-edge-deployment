import json
import os
import socket
import subprocess
import time

SERVER_HOST = os.getenv("EDGEBENCH_IPERF_HOST", "127.0.0.1")
PORT = int(os.getenv("EDGEBENCH_IPERF_PORT", "5201"))
DURATION_S = int(os.getenv("EDGEBENCH_IPERF_DURATION_S", "5"))
PARALLEL = int(os.getenv("EDGEBENCH_IPERF_PARALLEL", "1"))
REVERSE = os.getenv("EDGEBENCH_IPERF_REVERSE", "0") == "1"

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)

def _wait_port_open(port: int, timeout_s: float = 2.0, interval_s: float = 0.05) -> bool:
    # 로컬 서버 준비 확인(127.0.0.1:PORT)
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        try:
            import socket as _s
            with _s.create_connection(("127.0.0.1", port), timeout=0.2):
                return True
        except OSError:
            time.sleep(interval_s)
    return False

def main():
    inner_t0 = time.perf_counter()

    server_cmd = ["iperf3", "-s", "-1", "-p", str(PORT)]
    server = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    status = "success"
    sent_mbps = float("nan")
    recv_mbps = float("nan")
    sent_bytes = -1
    recv_bytes = -1
    err_msg = "NA"

    try:
        if not _wait_port_open(PORT, timeout_s=2.0):
            raise RuntimeError(f"iperf3 server not ready on 127.0.0.1:{PORT}")

        client_cmd = [
            "iperf3", "-c", SERVER_HOST,
            "-p", str(PORT),
            "-t", str(DURATION_S),
            "-P", str(PARALLEL),
            "-J",
        ]
        if REVERSE:
            client_cmd.append("-R")

        proc = _run(client_cmd)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "").strip())

        data = json.loads(proc.stdout)

        end = data.get("end", {})
        sum_sent = end.get("sum_sent", {}) or {}
        sum_recv = end.get("sum_received", {}) or {}

        sent_mbps = float(sum_sent.get("bits_per_second", float("nan"))) / 1e6
        recv_mbps = float(sum_recv.get("bits_per_second", float("nan"))) / 1e6
        sent_bytes = int(sum_sent.get("bytes", -1))
        recv_bytes = int(sum_recv.get("bytes", -1))

    except Exception as e:
        status = "fail"
        err_msg = str(e).replace("\n", " ")[:300]

    finally:
        try:
            server.terminate()
        except Exception:
            pass
        try:
            server.wait(timeout=1.0)
        except Exception:
            pass

    inner_elapsed_s = time.perf_counter() - inner_t0
    host = socket.gethostname()

    print(
        f"[iperf3] variant=default status={status} host={host} server={SERVER_HOST}:{PORT} "
        f"t={DURATION_S}s P={PARALLEL} reverse={int(REVERSE)} "
        f"sent_mbps={sent_mbps:.3f} recv_mbps={recv_mbps:.3f} "
        f"sent_bytes={sent_bytes} recv_bytes={recv_bytes} "
        f"inner_elapsed_s={inner_elapsed_s:.3f} err={err_msg}"
    )

if __name__ == "__main__":
    main()