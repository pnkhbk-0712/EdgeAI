"""Crop the non-safety headwear classes of the external cap/hat set (CC BY 4.0, euc, Roboflow v3)
into data/helmet_roles_by_class/hat_not_safety_ext/<class>/<split>/. Boxes < MIN_PX are skipped.
Usage: python src/export_ext_hat_crops.py
"""
from collections import Counter
from pathlib import Path

from PIL import Image

SRC = Path("data/helmet_suplement/cap-hat.v3i.yolov8")
OUT = Path("data/helmet_roles_by_class/hat_not_safety_ext")
NAMES = ["bareface", "cap-hat", "face mask-bandana", "hoodie", "mask-full face mask", "motor helmet"]
KEEP = {"cap-hat", "hoodie", "motor helmet"}
MIN_PX = 24

raw, kept = Counter(), Counter()
for split in ("train", "valid"):
    for lf in sorted((SRC / split / "labels").glob("*.txt")):
        im = None
        for i, ln in enumerate(lf.read_text().splitlines()):
            p = ln.split()
            if len(p) < 5:
                continue
            name = NAMES[int(p[0])]
            raw[name] += 1
            if name not in KEEP:
                continue
            if im is None:
                im = Image.open(next((SRC / split / "images").glob(lf.stem + ".*"))).convert("RGB")
            W, H = im.size
            x, y, w, h = map(float, p[1:5])
            b = (max(int((x - w / 2) * W), 0), max(int((y - h / 2) * H), 0), min(int((x + w / 2) * W), W), min(int((y + h / 2) * H), H))
            if b[2] - b[0] < MIN_PX or b[3] - b[1] < MIN_PX:
                continue
            d = OUT / name.replace(" ", "_")  # NB: "motor helmet" crops were renamed by hand to motor_helmet_label_unreliable / split
            d.mkdir(parents=True, exist_ok=True)
            im.crop(b).save(d / f"{lf.stem[:40]}_{i}.jpg", quality=92)
            kept[name] += 1
print("boxes in set:", dict(raw))
print("cropped (>=24px):", dict(kept))
