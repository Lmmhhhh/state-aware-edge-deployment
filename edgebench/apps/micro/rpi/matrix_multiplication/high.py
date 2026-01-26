import argparse
import time as t
import numpy as np

np.random.seed(1)

def run(save_output: bool = False, out_path: str = "result_matrix.txt"):
    main_start = t.time()

    # high: 450x450, 400 times
    m = 450
    n = 400

    matrix_1 = np.random.rand(m, m)
    matrix_2 = np.random.rand(m, m)

    start = t.time()
    result = None
    for _ in range(n):
        result = np.matmul(matrix_1, matrix_2)
    end = t.time()

    if save_output and result is not None:
        np.savetxt(out_path, result, fmt="%.4f", delimiter=" ",
                   header="The result of matrix multiplication is:")

    main_end = t.time()

    compute_sec = end - start
    total_sec = main_end - main_start

    print(f"compute_sec={compute_sec:.6f}")
    print(f"total_sec={total_sec:.6f}")

    return compute_sec, total_sec

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", action="store_true", help="save output matrix to file")
    p.add_argument("--out", default="result_matrix.txt", help="output path when --save is set")
    args = p.parse_args()
    run(save_output=args.save, out_path=args.out)

if __name__ == "__main__":
    main()
