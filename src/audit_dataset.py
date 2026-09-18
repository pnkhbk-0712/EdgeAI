"""Data-integrity audit for data/ua_detrac_yolo/ -- run this after any conversion/merge/
oversample step, before training. Checks what a "does it run" smoke-test won't catch:

- corrupt/unreadable images (cv2.imread returns None)
- orphan images (no matching label file) or orphan labels (no matching image)
- malformed label lines (wrong field count, non-numeric)
- out-of-range normalized coordinates, zero-size boxes, invalid class ids

Writes a full report to docs/audit_report.txt (git-ignored -- regenerate, don't diff it) and
prints a summary. Real bug this caught once already: a source filename with a non-ASCII
character got mangled during zip extraction on Windows, producing 5 unreadable images after
oversampling multiplied the one bad file -- see docs/RUNLOG.md, 2026-09-18.
"""
import os
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent / "data" / "ua_detrac_yolo"
REPORT_OUT = Path(__file__).resolve().parent.parent / "docs" / "audit_report.txt"
NAMES = ["car", "motorcycle", "bus", "truck"]


def main():
    issues = {
        "corrupt_images": [],
        "orphan_images": [],   # image with no label file
        "orphan_labels": [],   # label file with no image
        "empty_labels": 0,     # label file exists but has 0 boxes (background frame, not an error)
        "malformed_lines": [],
        "out_of_range": [],
        "zero_size_box": [],
        "bad_class_id": [],
    }

    for split in ["train", "val"]:
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split
        if not img_dir.exists():
            print(f"WARNING: {img_dir} not found, skipping {split}")
            continue

        img_files = {f.stem: f.name for f in img_dir.iterdir()}
        lbl_files = {f.stem: f.name for f in lbl_dir.iterdir()}
        img_stems, lbl_stems = set(img_files), set(lbl_files)

        for stem in img_stems - lbl_stems:
            issues["orphan_images"].append(f"{split}/{img_files[stem]}")
        for stem in lbl_stems - img_stems:
            issues["orphan_labels"].append(f"{split}/{lbl_files[stem]}")

        checked = 0
        for stem in img_stems & lbl_stems:
            img_path = img_dir / img_files[stem]
            lbl_path = lbl_dir / lbl_files[stem]

            img = cv2.imread(str(img_path))
            if img is None:
                issues["corrupt_images"].append(f"{split}/{img_files[stem]}")
                continue

            text = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                issues["empty_labels"] += 1
                checked += 1
                continue

            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    issues["malformed_lines"].append(f"{split}/{lbl_files[stem]}: {line!r}")
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    issues["malformed_lines"].append(f"{split}/{lbl_files[stem]}: {line!r}")
                    continue
                if cls < 0 or cls >= len(NAMES):
                    issues["bad_class_id"].append(f"{split}/{lbl_files[stem]}: class {cls}")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    issues["out_of_range"].append(f"{split}/{lbl_files[stem]}: {line!r}")
                if w <= 0 or h <= 0:
                    issues["zero_size_box"].append(f"{split}/{lbl_files[stem]}: {line!r}")
            checked += 1

        print(f"[{split}] checked {checked} image+label pairs")

    lines = []
    for key, val in issues.items():
        if isinstance(val, list):
            lines.append(f"{key}: {len(val)}")
            lines.extend(f"    {v}" for v in val[:20])
        else:
            lines.append(f"{key}: {val}")
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nFull report: {REPORT_OUT}")
    print("Summary:")
    for key, val in issues.items():
        print(f"  {key}: {len(val) if isinstance(val, list) else val}")


if __name__ == "__main__":
    main()
