import time
import numpy as np

SEED = 1
M = 1000
REPEAT = 100


def main():
    np.random.seed(SEED)

    print(f"[matrix_multiplication][low] init m={M} repeat={REPEAT}")

    matrix_1 = np.random.rand(M, M)
    matrix_2 = np.random.rand(M, M)

    start = time.perf_counter()
    out = None
    for _ in range(REPEAT):
        out = np.matmul(matrix_1, matrix_2)
    inner_compute_s = time.perf_counter() - start

    checksum = float(out[0, 0]) if out is not None else float("nan")

    print(f"[matrix_multiplication][low] inner_compute_time_s={inner_compute_s:.3f} checksum={checksum:.6f}")

if __name__ == "__main__":
    main()
