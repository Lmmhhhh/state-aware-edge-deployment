import json
import subprocess
import time
import socket

# ---- params (필요하면 숫자만 조절)
SERVER_HOST = "127.0.0.1"   # 진짜 네트워크 측정이면 "상대 디바이스 IP"로 바꾸기
PORT = 5201
DURATION_S = 5              # iperf3 -t
PARALLEL = 1                # iperf3 -P
REVERSE = False             # True면 reverse(-R): 서버->클라 방향

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)

def main():
    t0 = time.perf_counter()

    # 1) 서버(one-off) 시작: 테스트 1회 끝나면 자동 종료
    server_cmd = ["iperf3", "-s", "-1", "-p", str(PORT)]
    server = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    try:
        # 서버 준비 시간을 아주 짧게 준다
        time.sleep(0.2)

        # 2) 클라이언트 실행(JSON)
        client_cmd = [
            "iperf3",
            "-c", SERVER_HOST,
            "-p", str(PORT),
            "-t", str(DURATION_S),
            "-P", str(PARALLEL),
            "-J",
        ]
        if REVERSE:
            client_cmd.append("-R")

        proc = _run(client_cmd)
        data = json.loads(proc.stdout)

        # 3) 결과 파싱(요약)
        end = data.get("end", {})
        sum_sent = end.get("sum_sent", {})
        sum_recv = end.get("sum_received", {})

        # bits_per_second -> Mbps
        sent_mbps = float(sum_sent.get("bits_per_second", float("nan"))) / 1e6
        recv_mbps = float(sum_recv.get("bits_per_second", float("nan"))) / 1e6
        sent_bytes = int(sum_sent.get("bytes", -1))
        recv_bytes = int(sum_recv.get("bytes", -1))

        t1 = time.perf_counter()
        elapsed_s = t1 - t0

        # device name(그냥 호스트네임 찍기)
        host = socket.gethostname()

        print(
            f"[iperf3] host={host} server={SERVER_HOST}:{PORT} "
            f"t={DURATION_S}s P={PARALLEL} reverse={int(REVERSE)} "
            f"sent_mbps={sent_mbps:.3f} recv_mbps={recv_mbps:.3f} "
            f"sent_bytes={sent_bytes} recv_bytes={recv_bytes} "
            f"elapsed_s={elapsed_s:.3f}"
        )

    finally:
        # server가 혹시 남아있으면 정리
        try:
            server.terminate()
        except Exception:
            pass
        try:
            server.wait(timeout=1.0)
        except Exception:
            pass

if __name__ == "__main__":
    main()