import time as t
from math import cos, pi

OUTER = 100_001
DEGREE_MAX = 360

def main():
    start = t.time()
    acc = 0.0 

    for _ in range(OUTER):
        for x in range(DEGREE_MAX + 1):
            acc += cos(x * pi / 180.0)

    end = t.time()
    elapsed_s = end - start
    checksum = float(acc)

    print(
        f"[workload=floating_point_op_cos]"
        f"[device=rpi]"
        f" outer={OUTER} degree_max={DEGREE_MAX} elapsed_s={elapsed_s:.6f} checksum={checksum:.6f}"
    )

if __name__ == "__main__":
    main()