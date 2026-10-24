"""Rebuild the helmet dataset with role-by-helmet-colour classes (common site convention).

white -> helmet_supervisor | yellow, orange -> helmet_worker
blue, green, red, unclear -> helmet_other (visitor / specialist)   | head kept as-is | person dropped
Colour (src/helmet_colour.py) is a proxy for role, not a fact: no legal text fixes colour by role.
valid/test keep only images without a near-twin in train (valid_clean.txt / test_clean.txt,
made by check_near_duplicates.py). Usage: python src/build_role_dataset.py
"""
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from helmet_colour import colour_of, region

SRC = Path("data/helmet_roboflow")
DST = Path("data/helmet_roles")
NAMES = ["head", "helmet_supervisor", "helmet_worker", "helmet_other"]
ROLE = {"white": 1, "yellow": 2, "orange": 2, "blue": 3, "green": 3, "red": 3, "unclear": 3}


def clean_set(split):
    f = SRC / f"{split}_clean.txt"
    return {Path(p).stem for p in f.read_text().split()} if f.exists() else None


stats = {}
for split in ("train", "valid", "test"):
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)
    keep = clean_set(split) if split != "train" else None
    cnt, imgs = Counter(), Counter()
    for lf in sorted((SRC / split / "labels").glob("*.txt")):
        if keep is not None and lf.stem not in keep:
            continue
        img = next((SRC / split / "images").glob(lf.stem + ".*"))
        im = Image.open(img).convert("RGB")
        out, seen = [], set()
        for ln in lf.read_text().splitlines():
            c, *rest = ln.split()
            x, y, w, h = map(float, rest[:4])
            if c == "2":  # person dropped
                continue
            nc = 0 if c == "0" else ROLE[colour_of(region(im, x, y, w, h))]
            out.append(f"{nc} {' '.join(rest)}"); cnt[nc] += 1; seen.add(nc)
        if not out:
            continue
        (DST / split / "labels" / lf.name).write_text("\n".join(out) + "\n")
        d = DST / split / "images" / img.name
        if not d.exists():
            try: os.link(img, d)
            except OSError: shutil.copy2(img, d)
        for c in seen: imgs[c] += 1
    stats[split] = (cnt, imgs)
    print(split, "images:", len(list((DST / split / "images").glob("*.*"))), {NAMES[k]: f"{cnt[k]} boxes / {imgs[k]} imgs" for k in range(4)})

(DST / "data.yaml").write_text(
    f"path: {DST.resolve().as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames: {dict(enumerate(NAMES))}\n"
)
