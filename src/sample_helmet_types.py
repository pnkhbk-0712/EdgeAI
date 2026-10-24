"""Sample N `helmet` boxes from train, auto-label colour, write numbered contact sheets for manual type review.
Usage: python src/sample_helmet_types.py OUT_DIR [N=120] [class_id=1]
"""
import colorsys
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

root = Path("data/helmet_roboflow/train")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
CLS = sys.argv[3] if len(sys.argv) > 3 else "1"
random.seed(0)


def colour(rgb):
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    if v < 0.25: return "black"
    if s < 0.18: return "white" if v > 0.6 else "grey"
    h *= 360
    if h < 15 or h >= 345: return "red"
    if h < 45: return "orange"
    if h < 70: return "yellow"
    if h < 170: return "green"
    if h < 260: return "blue"
    return "other"


boxes = []
for lf in sorted((root / "labels").glob("*.txt")):
    for ln in lf.read_text().splitlines():
        c, x, y, w, h = ln.split()[:5]
        if c == CLS:
            boxes.append((lf.stem, float(x), float(y), float(w), float(h)))
sample = random.sample(boxes, N * 2)  # oversample; skip tiny boxes below
crops, tally = [], Counter()
for stem, x, y, w, h in sample:
    if len(crops) == N: break
    img = next((root / "images").glob(stem + ".*"))
    im = Image.open(img).convert("RGB"); W, H = im.size
    x0, y0, x1, y1 = [int(v) for v in ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)]
    if x1 - x0 < 24 or y1 - y0 < 24: continue
    c = im.crop((x0, y0, x1, y1))
    a = np.asarray(c.resize((16, 16)))[2:10, 4:12].reshape(-1, 3)  # upper-centre = shell
    col = colour(tuple(np.median(a, axis=0)))
    tally[col] += 1
    crops.append((c.resize((120, 120)), col))
print(f"{len(crops)} crops, colour tally:", dict(tally.most_common()))
for s in range(0, len(crops), 30):
    sheet = Image.new("RGB", (6 * 120, 5 * 140), "white"); d = ImageDraw.Draw(sheet)
    for k, (c, col) in enumerate(crops[s:s + 30]):
        X, Y = (k % 6) * 120, (k // 6) * 140
        sheet.paste(c, (X, Y)); d.text((X + 2, Y + 122), f"#{s + k + 1} {col}", fill="black")
    sheet.save(out / f"sheet{s // 30 + 1}.png")
