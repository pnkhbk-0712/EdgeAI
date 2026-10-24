"""Cut every box of data/helmet_roles into per-class folders: OUT/<class>/<split>/<image>_<n>.jpg
Boxes smaller than MIN_PX on a side are skipped (unreadable). Usage: python src/export_role_crops.py [OUT]
"""
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

SRC = Path("data/helmet_roles")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/helmet_roles_by_class")
NAMES = ["head", "helmet_supervisor", "helmet_worker", "helmet_other"]
MIN_PX = 24

kept, skipped = Counter(), Counter()
for split in ("train", "valid", "test"):
    for n in NAMES:
        (OUT / n / split).mkdir(parents=True, exist_ok=True)
    for lf in sorted((SRC / split / "labels").glob("*.txt")):
        im = Image.open(next((SRC / split / "images").glob(lf.stem + ".*"))).convert("RGB")
        W, H = im.size
        for i, ln in enumerate(lf.read_text().splitlines()):
            c, x, y, w, h = ln.split()[:5]
            x, y, w, h = float(x), float(y), float(w), float(h)
            b = (max(int((x - w / 2) * W), 0), max(int((y - h / 2) * H), 0), min(int((x + w / 2) * W), W), min(int((y + h / 2) * H), H))
            if b[2] - b[0] < MIN_PX or b[3] - b[1] < MIN_PX:
                skipped[NAMES[int(c)]] += 1
                continue
            im.crop(b).save(OUT / NAMES[int(c)] / split / f"{lf.stem[:40]}_{i}.jpg", quality=92)
            kept[(NAMES[int(c)], split)] += 1
for n in NAMES:
    print(n, {s: kept[(n, s)] for s in ("train", "valid", "test")}, "skipped(<24px):", skipped[n])
