"""Merge motorcycle examples into data/ua_detrac_yolo/ (UA-DETRAC has none -- Report Sec. VI.A).

Source: Kaggle "Vehicle Dataset for YOLO" (nadinpethiyagoda/vehicle-dataset-for-yolo),
3000 images, 6 classes already in YOLO format: car, threewheel, bus, truck, motorbike, van.
Verified for real before use: drew boxes back onto several sample images (motorbike class
index 4) -- correct boxes, mixed domain (some studio/for-sale-listing photos, some real
street scenes e.g. Sri Lanka tuk-tuk photos with a motorbike in the background).

Only the motorbike boxes are taken, remapped to OUR class id 1 -- car/bus/truck/van/threewheel
from this source are dropped, since UA-DETRAC already covers those classes with a larger,
more consistent (fixed traffic camera) domain; mixing in this source's very different-looking
car/bus/truck photos would more likely hurt than help. An image is only copied if it has at
least one motorbike box.

Known limitation, not hidden: most of these photos are close-up single-vehicle shots (product
photos, driveway photos), not the elevated traffic-camera angle the pilot site will actually
use. Real pilot-site motorcycle footage (once filmed) will be a better domain match than this
supplementary set -- this closes the "zero examples" gap, not the "right domain" gap.
"""
import os
from pathlib import Path
from shutil import copyfile

import kagglehub

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "ua_detrac_yolo"

MOTORBIKE_SOURCE_CLASS_ID = 4  # "motorbike" in this source's classes.txt
OUR_MOTORCYCLE_CLASS_ID = 1    # class id 1 in data/ua_detrac_yolo/data.yaml

SPLIT_MAP = {"train": "train", "valid": "val"}  # source split name -> our split name


def merge_split(source_dir: Path, our_split: str):
    images_out = OUT_ROOT / "images" / our_split
    labels_out = OUT_ROOT / "labels" / our_split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    label_files = sorted((source_dir / "labels").glob("*.txt"))
    n_images, n_boxes, n_missing_image = 0, 0, 0

    for lf in label_files:
        motorbike_lines = []
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            if int(parts[0]) == MOTORBIKE_SOURCE_CLASS_ID:
                motorbike_lines.append(" ".join([str(OUR_MOTORCYCLE_CLASS_ID), *parts[1:]]))
        if not motorbike_lines:
            continue

        stem = lf.stem
        # source images can be .jpg/.jpeg/.png with the same stem
        src_img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = source_dir / "images" / f"{stem}{ext}"
            if candidate.exists():
                src_img = candidate
                break
        if src_img is None:
            n_missing_image += 1
            continue

        out_stem = f"motorbike_{stem}"
        copyfile(src_img, images_out / f"{out_stem}{src_img.suffix.lower()}")
        (labels_out / f"{out_stem}.txt").write_text("\n".join(motorbike_lines), encoding="utf-8")
        n_images += 1
        n_boxes += len(motorbike_lines)

    return n_images, n_boxes, n_missing_image


def main():
    os.environ.setdefault(
        "KAGGLEHUB_CACHE",
        "D:/DOANMINHHIEU/STUDIES/Ky9/01_Ai/.cache/kagglehub",  # never let this default to C:
    )
    path = kagglehub.dataset_download("nadinpethiyagoda/vehicle-dataset-for-yolo")
    source_root = Path(path) / "vehicle dataset"
    if not source_root.exists():
        # dataset layout changed upstream -- fail loudly rather than silently doing nothing
        raise SystemExit(f"Expected folder not found: {source_root}")

    total_images, total_boxes = 0, 0
    for src_split, our_split in SPLIT_MAP.items():
        src_dir = source_root / src_split
        if not src_dir.exists():
            print(f"WARNING: {src_dir} not found, skipping")
            continue
        n_images, n_boxes, n_missing = merge_split(src_dir, our_split)
        total_images += n_images
        total_boxes += n_boxes
        print(f"[{our_split}] +{n_images} motorbike images, +{n_boxes} boxes "
              f"({n_missing} labels had no matching image file)")

    print(f"\nTotal added: {total_images} images, {total_boxes} motorcycle boxes")
    print("data.yaml unchanged (motorcycle was already reserved as class id 1) -- "
          "no re-generation needed, just re-run model.train() to pick up the new files.")


if __name__ == "__main__":
    main()
