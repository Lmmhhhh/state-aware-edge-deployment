import time
import numpy as np

M = 450
REPEAT = 400
SEED = 1

def main():
    np.random.seed(SEED)

    matrix_1 = np.random.rand(M, M)
    matrix_2 = np.random.rand(M, M)

    start = time.time()
    out = None
    for _ in range(REPEAT):
        out = np.matmul(matrix_1, matrix_2)
    end = time.time()

    checksum = float(out[0, 0]) if out is not None else float("nan")

    print(f"[matrix_multiplication][low] m={M} repeat={REPEAT}")
    print(f"[matrix_multiplication][low] compute_time_s={end-start:.3f} checksum={checksum:.6f}")

if __name__ == "__main__":
    main()

