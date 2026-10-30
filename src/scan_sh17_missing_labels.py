"""Find SH17 train images where the current model sees a helmet/vest (conf>=0.6) that has no GT box (IoU<0.3).
Writes _tools/sh17_review/cands.json; the image stems were then saved to data/ppe_sources/sh17/exclude_unlabeled.txt
(builder skips them). Reviewed 24 samples by eye 2026-10-30: ~1/3 of helmet hits and ~4/5 of vest hits were real unlabeled PPE."""
import json, sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
m = YOLO("models/ppe_v1_best.pt")
NAMES = m.names  # 1 helmet, 3 vest
root = Path("data/ppe_merged/train")
imgs = sorted((root / "images").glob("sh17_*"))
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u else 0
out = []; cnt = {"helmet": 0, "vest": 0}; imgcnt = {"helmet": set(), "vest": set()}; gt_cnt = {"helmet": 0, "vest": 0}
for k in range(0, len(imgs), 16):
    for r in m.predict([str(p) for p in imgs[k:k+16]], conf=0.6, imgsz=640, verbose=False, device="cpu"):
        p = Path(r.path); h, w = r.orig_shape
        gt = [l.split() for l in (root / "labels" / (p.stem + ".txt")).read_text().split("\n") if l]
        for cls, nm in ((1, "helmet"), (3, "vest")):
            g = [[(float(x[1])-float(x[3])/2)*w, (float(x[2])-float(x[4])/2)*h, (float(x[1])+float(x[3])/2)*w, (float(x[2])+float(x[4])/2)*h] for x in gt if int(x[0]) == cls]
            gt_cnt[nm] += len(g)
            for b, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if int(c) == cls and max([iou(b, q) for q in g] + [0]) < 0.3:
                    cnt[nm] += 1; imgcnt[nm].add(p.name); out.append((p.name, nm, round(cf, 2), [round(v) for v in b]))
print("images scanned", len(imgs)); print("GT boxes", gt_cnt); print("unlabeled-but-detected boxes (conf>=0.6)", cnt)
print("images affected", {k: len(v) for k, v in imgcnt.items()})
json.dump(out, open(r"D:/DOANMINHHIEU/STUDIES/Ky9/01_Ai/_tools/sh17_review/cands.json", "w"))
