import csv
import random
import time
from pathlib import Path
from textblob import TextBlob

MAX_SAMPLES = 512
SEED = 1

APP_ROOT = Path(__file__).resolve().parents[2]  # .../edgebench/apps/app
CSV_PATH = APP_ROOT / "_data" / "sentiment_analysis" / "twitter_validation.csv"

TEXT_COL_IDX = 3

def load_texts(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing dataset: {csv_path}")

    texts: list[str] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) <= TEXT_COL_IDX:
                continue
            text = (row[TEXT_COL_IDX] or "").strip()
            if text:
                texts.append(text)

    if not texts:
        raise ValueError("dataset has no usable text rows")
    return texts

def main():
    DEVICE = "rpi"  

    print(f"[workload=sentiment_analysis][src=twitter_validation.csv][device={DEVICE}] start")
    t0 = time.perf_counter()

    texts = load_texts(CSV_PATH)

    rnd = random.Random(SEED)
    if len(texts) >= MAX_SAMPLES:
        picked = rnd.sample(texts, MAX_SAMPLES)
    else:
        picked = texts[:] 

    pol_sum = 0.0
    subj_sum = 0.0
    n = 0

    for s in picked:
        sent = TextBlob(s).sentiment
        pol_sum += float(sent.polarity)
        subj_sum += float(sent.subjectivity)
        n += 1

    pol_avg = pol_sum / n if n else 0.0
    subj_avg = subj_sum / n if n else 0.0

    checksum = float(pol_avg + subj_avg)

    t1 = time.perf_counter()
    inner_total_s = t1 - t0

    print(
        f"[workload=sentiment_analysis][src=twitter_validation.csv][device={DEVICE}] "
        f"num_samples={n} seed={SEED} polarity_avg={pol_avg:.6f} subjectivity_avg={subj_avg:.6f} "
        f"checksum={checksum:.6f} inner_total_s={inner_total_s:.6f}"
    )

if __name__ == "__main__":
    main()