"""Per-class box and image counts per split for a YOLO dataset dir (reads names from data.yaml).
Usage: python src/class_stats.py DATASET_DIR
"""
import sys
from collections import Counter
from pathlib import Path

import yaml

root = Path(sys.argv[1])
names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
for split in ("train", "valid", "val", "test"):
    lbl = root / split / "labels"  # Roboflow layout; Ultralytics layout is labels/<split>
    if not lbl.exists():
        lbl = root / "labels" / split
    if not lbl.exists():
        continue
    boxes, imgs, empty = Counter(), Counter(), 0
    files = list(lbl.glob("*.txt"))
    for f in files:
        ids = [int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()]
        empty += not ids
        boxes.update(ids)
        imgs.update(set(ids))
    print(f"{split}: {len(files)} label files, {empty} empty")
    for i, n in enumerate(names):
        print(f"  {i:>2} {n:<12} boxes={boxes[i]:<7} imgs={imgs[i]}")
