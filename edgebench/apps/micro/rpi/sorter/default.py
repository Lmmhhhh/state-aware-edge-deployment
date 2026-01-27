import os
import time
import hashlib
import subprocess

SORT_ENV = {**os.environ, "LC_ALL": "C"}

DEFAULT_INPUT = os.path.join(os.path.dirname(__file__), "data.txt")

def run_sort_and_hash(input_path: str) -> tuple[float, str]:
    h = hashlib.sha256()

    start = time.perf_counter()
    p = subprocess.Popen(
        ["sort", input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False, 
        env=SORT_ENV,
    )

    assert p.stdout is not None
    for chunk in iter(lambda: p.stdout.read(1024 * 1024), b""):
        h.update(chunk)

    _, err = p.communicate()
    ret = p.returncode
    end = time.perf_counter()

    if ret != 0:
        msg = err.decode("utf-8", errors="replace") if err else ""
        raise RuntimeError(f"sort failed (ret={ret}): {msg}")

    return (end - start), h.hexdigest()

def main():
    input_path = DEFAULT_INPUT
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"missing input file: {input_path}")

    elapsed_s, checksum = run_sort_and_hash(input_path)

    print(f"[sorter] input=data.txt inner_elapsed_s={elapsed_s:.6f} checksum_sha256={checksum}")

if __name__ == "__main__":
    main()