"""Cut every `helmet` box of the original dataset into per-colour folders:
data/helmet_roles_by_class/helmet_<colour>/<split>/. Colour from helmet_colour.py (hand-checked, see docs/DATASET_AUDIT_WEEK11.md).
valid/test use only images without a near-twin in train (valid_clean.txt / test_clean.txt).
The role (supervisor/worker/...) is NOT stored here: it is a per-site convention, kept in the report.
Colour logic: src/helmet_colour.py.
Usage: python src/export_color_crops.py
"""
from collections import Counter
from pathlib import Path

from PIL import Image

from helmet_colour import colour_of, region

SRC = Path("data/helmet_roboflow")
OUT = Path("data/helmet_roles_by_class")
MIN_PX = 24


counts = Counter()
for split in ("train", "valid", "test"):
    clean = None
    if split != "train":
        clean = {Path(p).stem for p in (SRC / f"{split}_clean.txt").read_text().split()}
    for lf in sorted((SRC / split / "labels").glob("*.txt")):
        if clean is not None and lf.stem not in clean:
            continue
        im = None
        for i, ln in enumerate(lf.read_text().splitlines()):
            c, x, y, w, h = ln.split()[:5]
            if c != "1":
                continue
            if im is None:
                im = Image.open(next((SRC / split / "images").glob(lf.stem + ".*"))).convert("RGB")
            W, H = im.size
            x, y, w, h = map(float, (x, y, w, h))
            b = (max(int((x - w / 2) * W), 0), max(int((y - h / 2) * H), 0), min(int((x + w / 2) * W), W), min(int((y + h / 2) * H), H))
            if b[2] - b[0] < MIN_PX or b[3] - b[1] < MIN_PX:
                continue
            col = colour_of(region(im, x, y, w, h))
            d = OUT / f"helmet_{col}" / split
            d.mkdir(parents=True, exist_ok=True)
            im.crop(b).save(d / f"{lf.stem[:40]}_{i}.jpg", quality=92)
            counts[(col, split)] += 1
for col in sorted({k[0] for k in counts}):
    print(f"helmet_{col}", {s: counts[(col, s)] for s in ("train", "valid", "test")})
