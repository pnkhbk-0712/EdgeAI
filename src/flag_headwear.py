"""Flag `head` crops whose top region is not hair/skin-like (candidate non-bare heads: caps, hoods, other hats).
Heuristic only: dark caps look like black hair and are missed. Writes OUT/candidates.txt (paths).
Usage: python src/flag_headwear.py [BY_CLASS_DIR]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/helmet_roles_by_class")
THRESH = 0.5


def odd_fraction(path):
    a = np.asarray(Image.open(path).convert("HSV").resize((32, 32)), dtype=np.float32)
    h, s, v = a[..., 0] * 360 / 255, a[..., 1] / 255, a[..., 2] / 255
    h, s, v = h[:11, 4:28], s[:11, 4:28], v[:11, 4:28]  # top third, centre
    hair = v < 0.35
    skin = (h >= 5) & (h <= 45) & (s >= 0.15) & (s <= 0.75)
    grey = (s < 0.15) & (v < 0.7)
    return float((~(hair | skin | grey)).mean())


files = sorted((root / "head").rglob("*.jpg"))
flag = [f for f in files if odd_fraction(f) > THRESH]
(root / "head" / "candidates.txt").write_text("\n".join(str(f) for f in flag) + "\n")
print(len(files), "head crops,", len(flag), "flagged")
