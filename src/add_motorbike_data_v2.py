"""Merge a second motorcycle source into data/ua_detrac_yolo/ (2026-09-18, overnight follow-up
to the class-imbalance finding -- more real motorcycle boxes, still real data only).

Source: Roboflow "car-and-motorcycle-detection-with-kaggle-dataset" (aliff-haikal-ssf6d),
version 14, CC BY 4.0, 2 classes: Car(0), motorcycle(1). Verified before merging: drew a
sample motorcycle box back onto its image -- correct, vehicle-only (no rider in the box,
unlike a rejected candidate this same session -- see docs/RUNLOG.md for why that one was
skipped). Only 33 motorcycle instances in this source (most images are unlabeled/background),
so this is a modest top-up, not a second major source -- said plainly, not oversold.

Only motorcycle boxes are taken (same reasoning as add_motorbike_data.py: this source's Car
class is dropped, UA-DETRAC already covers cars with a more consistent domain).
"""
import os
from pathlib import Path
from shutil import copyfile

from roboflow import Roboflow

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "ua_detrac_yolo"
SOURCE_CLASS_NAMES = ["Car", "motorcycle"]  # this source's own data.yaml order
OUR_MOTORCYCLE_CLASS_ID = 1
SPLIT_MAP = {"train": "train", "valid": "val", "test": "val"}


def merge_split(source_dir: Path, our_split: str):
    images_out = OUT_ROOT / "images" / our_split
    labels_out = OUT_ROOT / "labels" / our_split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    src_motorcycle_id = SOURCE_CLASS_NAMES.index("motorcycle")
    n_images, n_boxes = 0, 0

    for lf in sorted((source_dir / "labels").glob("*.txt")):
        keep = []
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if parts and int(parts[0]) == src_motorcycle_id:
                keep.append(" ".join([str(OUR_MOTORCYCLE_CLASS_ID), *parts[1:]]))
        if not keep:
            continue

        stem = lf.stem
        src_img = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = source_dir / "images" / f"{stem}{ext}"
            if candidate.exists():
                src_img = candidate
                break
        if src_img is None:
            continue

        out_stem = f"rfmoto2_{stem}"
        copyfile(src_img, images_out / f"{out_stem}{src_img.suffix}")
        (labels_out / f"{out_stem}.txt").write_text("\n".join(keep), encoding="utf-8")
        n_images += 1
        n_boxes += len(keep)

    return n_images, n_boxes


def main():
    if "ROBOFLOW_API_KEY" not in os.environ:
        raise SystemExit("Set ROBOFLOW_API_KEY before running this script.")
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    proj = rf.workspace("aliff-haikal-ssf6d").project(
        "car-and-motorcycle-detection-with-kaggle-dataset"
    )
    ds = proj.version(14).download("yolov8", location=str(ROOT / "data" / "car_moto_roboflow"))
    source_root = Path(ds.location)

    total_images, total_boxes = 0, 0
    for src_split, our_split in SPLIT_MAP.items():
        src_dir = source_root / src_split
        if not src_dir.exists():
            continue
        n_images, n_boxes = merge_split(src_dir, our_split)
        total_images += n_images
        total_boxes += n_boxes
        print(f"[{our_split}] +{n_images} images, +{n_boxes} motorcycle boxes (from {src_split})")

    print(f"\nTotal added: {total_images} images, {total_boxes} motorcycle boxes")


if __name__ == "__main__":
    main()
