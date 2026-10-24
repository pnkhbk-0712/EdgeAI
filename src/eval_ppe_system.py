"""System-level PPE check: for each ground-truth person, is an item judged missing correctly?

mAP says how well boxes are found; the system decision is per person and item. For every GT person big
enough to be assessed, an item's GT status comes from the labels (positive box in its body region ->
present, explicit no_* box -> missing, neither -> skipped). The model's decision uses the same
association code as the demo (src/ppe_compliance.py), on GT person boxes (isolates the item logic from
person detection). Two ways to call "missing" are compared:
  explicit - only a detected no_* class counts (infer_absent=())
  absence  - no positive evidence for the item at all counts (infer_absent=all items)
Usage: python src/eval_ppe_system.py MODEL.pt [--data data/ppe_merged] [--split test] [--conf 0.3] [--imgsz 640]
"""
import argparse
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO

from ppe_compliance import ITEMS, MIN_PERSON_H_FRAC, _in_region, assess_person, norm_name

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--data", default="data/ppe_merged")
ap.add_argument("--split", default="test")
ap.add_argument("--conf", type=float, default=0.3)
ap.add_argument("--imgsz", type=int, default=640)
args = ap.parse_args()

root = Path(args.data)
names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
names = {int(k): norm_name(v) for k, v in names.items()}
model = YOLO(args.model)
items = [i for i in ITEMS if ITEMS[i][0] in names.values()]

stats = {s: {i: dict(tp=0, fp=0, fn=0, tn=0) for i in items} for s in ("explicit", "absence")}
gt_counts = {i: dict(present=0, missing=0) for i in items}

for img in sorted((root / args.split / "images").iterdir()):
    lf = root / args.split / "labels" / (img.stem + ".txt")
    W, H = Image.open(img).size
    gt = []
    for ln in lf.read_text().splitlines():
        c, x, y, w, h = ln.split()[:5]
        x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
        gt.append((names[int(c)], (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    persons = [b for n, b in gt if n == "person" and b[3] - b[1] >= MIN_PERSON_H_FRAC * H]
    if not persons:
        continue
    r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    dets = [(names[int(c)], tuple(b), float(s)) for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
            if names[int(c)] != "person"]
    pred = {"explicit": (), "absence": tuple(items)}
    for p in persons:
        status = {k: assess_person(p, dets, items, H, infer_absent=v) for k, v in pred.items()}
        for item in items:
            pos, neg, band = ITEMS[item]
            has_pos = any(n == pos and _in_region(p, b, band) for n, b in gt)
            has_neg = any(n == neg and _in_region(p, b, band) for n, b in gt)
            if has_pos == has_neg:  # no GT call, or contradictory -> skip
                continue
            gt_counts[item]["missing" if has_neg else "present"] += 1
            for k in pred:
                said_missing = status[k][item][0] == "missing"
                st = stats[k][item]
                key = ("tp" if said_missing else "fn") if has_neg else ("fp" if said_missing else "tn")
                st[key] += 1

print(f"model={args.model} split={args.split} conf>={args.conf} imgsz={args.imgsz}")
print(f"{'item':<8} {'GT present/missing':<19} | {'explicit (no_* class)':<28} | absence (no positive found)")
for item in items:
    g = gt_counts[item]
    cells = []
    for k in ("explicit", "absence"):
        s = stats[k][item]
        prec = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else float("nan")
        rec = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else float("nan")
        cells.append(f"P={prec:.2f} R={rec:.2f} (FP {s['fp']}, FN {s['fn']})")
    print(f"{item:<8} {g['present']:>4}/{g['missing']:<14} | {cells[0]:<28} | {cells[1]}")
