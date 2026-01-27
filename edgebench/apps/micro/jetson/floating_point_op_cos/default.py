import time
from math import cos, pi

OUTER = 100_001
DEGREE_MAX = 360 

def main():
    start = time.perf_counter()

    acc = 0.0
    for _ in range(OUTER):
        for x in range(DEGREE_MAX + 1):
            acc += cos(x * pi / 180.0)

    inner_elapsed_s = time.perf_counter() - start
    checksum = float(acc)

    print(
        f"[workload=floating_point_op_cos]"
        f"outer={OUTER} x_max={X_MAX} inner_elapsed_s={inner_elapsed_s:.6f} checksum={checksum:.6f}"
    )

if __name__ == "__main__":
    main()