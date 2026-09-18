"""Oversample motorcycle-containing images in the TRAIN split only.

Real finding (2026-09-18): car:motorcycle box ratio is 154:1 in the merged dataset
(107,889 vs 699 boxes). Left as-is, YOLO's loss can all but ignore the minority class
since car dominates the total loss regardless of motorcycle performance.

Fix: duplicate each motorcycle-containing TRAIN image+label OVERSAMPLE_FACTOR times
(as new filenames, not literal copies of the same file re-read -- avoids relying on
any training-time repeat-sampling feature and works with plain model.train()).
VAL is never touched, so evaluation still reflects the true, unbalanced real-world
distribution -- oversampling only changes what the optimizer sees, not how it's judged.
"""
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parent.parent
IMAGES_TRAIN = ROOT / "data" / "ua_detrac_yolo" / "images" / "train"
LABELS_TRAIN = ROOT / "data" / "ua_detrac_yolo" / "labels" / "train"

MOTORCYCLE_CLASS_ID = 1
OVERSAMPLE_FACTOR = 5  # 373 motorcycle train images -> ~1865 (154:1 -> ~31:1 for training)


def has_motorcycle(label_path: Path) -> bool:
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if parts and int(parts[0]) == MOTORCYCLE_CLASS_ID:
            return True
    return False


def main():
    # Exclude already-"_dup"-suffixed files so re-running this script (e.g. after merging a
    # new motorcycle source) only oversamples the NEW base images, not the previous run's
    # duplicates-of-duplicates. Idempotent by construction, not by accident.
    motorcycle_labels = [
        lf for lf in LABELS_TRAIN.glob("*.txt")
        if "_dup" not in lf.stem and has_motorcycle(lf)
    ]
    print(f"Found {len(motorcycle_labels)} motorcycle-containing BASE train images (excluding prior duplicates)")

    added = 0
    for lf in motorcycle_labels:
        stem = lf.stem
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = IMAGES_TRAIN / f"{stem}{ext}"
            if candidate.exists():
                img = candidate
                break
        if img is None:
            continue
        for i in range(1, OVERSAMPLE_FACTOR):  # i=1..4 -> 4 extra copies (5x total with original)
            new_stem = f"{stem}_dup{i}"
            copyfile(img, IMAGES_TRAIN / f"{new_stem}{img.suffix}")
            copyfile(lf, LABELS_TRAIN / f"{new_stem}.txt")
            added += 1

    print(f"Added {added} duplicate image+label pairs (factor={OVERSAMPLE_FACTOR})")
    print("Re-run the class-count check to see the new ratio.")


if __name__ == "__main__":
    main()
