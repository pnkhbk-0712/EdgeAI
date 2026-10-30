"""Shrink the SH17 images we need and write them in Ultralytics layout for build_ppe_dataset.py (2026-10-30).

SH17 (CC BY-NC-SA 4.0, educational/research use) ships ~14 GB of up-to-5760px photos. We keep only images that
contain gloves or glasses (the two weak classes), resize to MAX_SIDE, and keep the YOLO labels (normalised, so
unchanged). Class ids were verified against the VOC xml name counts of the first 300 files.
Usage: python src/prep_sh17.py
"""
from pathlib import Path

import yaml
from PIL import Image

SRC = Path("data/ppe_sources/sh17")
OUT = SRC / "raw"
NAMES = ["person", "ear", "ear-mufs", "face", "face-guard", "face-mask-medical", "foot", "tools", "glasses", "gloves",
         "helmet", "hands", "head", "medical-suit", "shoes", "safety-suit", "safety-vest"]
KEEP_IF_ANY = {8, 9}   # glasses, gloves
MAX_SIDE = 960

(OUT / "images" / "all").mkdir(parents=True, exist_ok=True)
(OUT / "labels" / "all").mkdir(parents=True, exist_ok=True)
n = 0
for lf in sorted((SRC / "labels").glob("*.txt")):
    ids = {int(l.split()[0]) for l in lf.read_text().splitlines() if l.strip()}
    if not ids & KEEP_IF_ANY:
        continue
    img = next((p for p in (SRC / "images").glob(lf.stem + ".*")), None)
    if img is None:
        continue
    with Image.open(img) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(OUT / "images" / "all" / (lf.stem + ".jpg"), quality=90)
    (OUT / "labels" / "all" / lf.name).write_text(lf.read_text())
    n += 1
(OUT / "data.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(NAMES))}), encoding="utf-8")
print(f"kept {n} images -> {OUT}")
