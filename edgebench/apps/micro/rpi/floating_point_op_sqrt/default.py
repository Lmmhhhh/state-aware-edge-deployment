import time as t
from math import sqrt

OUTER = 1_001
X_MAX = 30_000

def main():
    start = t.time()
    acc = 0.0 

    for _ in range(OUTER):
        for x in range(X_MAX):
            acc += sqrt(x)

    end = t.time()
    elapsed_s = end - start
    checksum = float(acc)

    print(
        f"[workload=floating_point_op_sqrt]"
        f"[device=rpi]"
        f" outer={OUTER} x_max={X_MAX} elapsed_s={elapsed_s:.6f} checksum={checksum:.6f}"
    )

if __name__ == "__main__":
    main()