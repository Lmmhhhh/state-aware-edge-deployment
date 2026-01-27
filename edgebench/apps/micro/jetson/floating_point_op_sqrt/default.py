import time
from math import sqrt

OUTER = 1_001
X_MAX = 1_000_000

def main():
    start = time.perf_counter()

    acc = 0.0
    for _ in range(OUTER):
        for x in range(X_MAX):
            acc += sqrt(x)

    inner_elapsed_s = time.perf_counter() - start
    checksum = float(acc)

    print(
        f"[floating_point_op_sqrt] variant=default "
        f"outer={OUTER} x_max={X_MAX} inner_elapsed_s={inner_elapsed_s:.6f} checksum={checksum:.6f}"
    )

if __name__ == "__main__":
    main()