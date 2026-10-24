"""Near-duplicate check across train/valid/test (dHash, Hamming distance).

MD5 only catches byte-identical files; this catches same scene re-cropped/resized.
Usage: python src/check_near_duplicates.py [dataset_dir]
"""
import sys
from pathlib import Path

import numpy as np

from imghash import dhash, hamming

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/helmet_roboflow")
THRESHOLDS = (0, 2, 4, 8)


def load(split):
    files = sorted((root / split / "images").glob("*.*"))
    return files, np.stack([dhash(f) for f in files])


data = {s: load(s) for s in ("train", "valid", "test") if (root / s / "images").exists()}  # sets without a test split are fine
for s, (f, _) in data.items():
    print(f"{s}: {len(f)} images")

for src, dst in (("valid", "train"), ("test", "train"), ("test", "valid")):
    if src not in data or dst not in data:
        continue
    sf, sh = data[src]
    df, dh = data[dst]
    best = np.array([hamming(h, dh).min() for h in sh])
    parts = ", ".join(f"<={t}: {int((best <= t).sum())} ({(best <= t).mean():.1%})" for t in THRESHOLDS)
    print(f"{src} images with a near-twin in {dst} -> {parts}")
    if dst == "train":  # image lists without near-twins, to re-run yolo val on leak-free data
        keep = [str(f.resolve()) for f, b in zip(sf, best) if b > 8]
        (root / f"{src}_clean.txt").write_text("\n".join(keep) + "\n")
    if src == "valid" and dst == "train":
        for i in np.argsort(best)[:5]:
            j = hamming(sh[i], dh).argmin()
            print(f"  example d={best[i]}: {sf[i].name}  ~  {df[j].name}")
