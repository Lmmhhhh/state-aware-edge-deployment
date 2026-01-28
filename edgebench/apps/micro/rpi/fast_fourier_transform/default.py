import numpy as np

N = 10_000_000
REPEAT = 3
SEED = 1

def main():
    np.random.seed(SEED)

    x = np.random.rand(N)

    out = None
    for _ in range(REPEAT):
        out = np.fft.fft(x)

    checksum = float(np.abs(out[0])) if out is not None else float("nan")
    print(f"[fast_fourier_transform] n={N} repeat={REPEAT} checksum={checksum:.6f}")

if __name__ == "__main__":
    main()