"""Split data/helmet_roles_by_class/head into no_hat / hat_not_safety / head_mislabeled_helmet.
Indices below are positions in head/candidates.txt (src/flag_headwear.py output, thresh 0.5),
picked by eye from contact sheets. Everything not picked stays no_hat.
"""
import shutil
from pathlib import Path

root = Path("data/helmet_roles_by_class")
cand = [Path(p) for p in (root / "head" / "candidates.txt").read_text().split()]
HAT_NOT_SAFETY = {24, 27, 39, 66, 104, 110, 122, 133, 134, 135, 144, 146, 154, 155, 156, 170, 187, 214,
                  234, 241, 246, 253, 254, 255, 256, 265, 267, 284, 285, 294, 338}
MISLABELED_HELMET = {45, 103, 186, 239, 279}
picked = {cand[i]: "hat_not_safety" for i in HAT_NOT_SAFETY} | {cand[i]: "head_mislabeled_helmet" for i in MISLABELED_HELMET}

for f in sorted((root / "head").rglob("*.jpg")):
    split = f.parent.name
    dst = root / picked.get(f, "no_hat") / split
    dst.mkdir(parents=True, exist_ok=True)
    shutil.move(f, dst / f.name)
(root / "head" / "candidates.txt").unlink()
shutil.rmtree(root / "head")
for d in ("no_hat", "hat_not_safety", "head_mislabeled_helmet"):
    print(d, {s: len(list((root / d / s).glob("*.jpg"))) for s in ("train", "valid", "test") if (root / d / s).exists()})
