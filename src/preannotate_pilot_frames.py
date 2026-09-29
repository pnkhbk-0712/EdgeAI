"""Auto pre-label the 606 extracted pilot frames using edgeai_v1_best.pt, so the team corrects
boxes in CVAT instead of drawing every one from scratch (2026-09-30).

This is NOT a substitute for real ground-truth labeling -- it's a speed-up. The model has known
gaps on this real footage (see docs/RUNLOG.md 2026-09-29 sanity check: missed a clearly parked
car), so every frame still needs a human pass in CVAT to fix missed/wrong/extra boxes, not just a
rubber stamp.

Output: one CVAT 1.1 (Pascal VOC-style XML per image, zipped as "CVAT for images 1.1") annotation
set per clip, written to data/pilot_frames/<tag>/annotations.xml -- CVAT's own import format
that accepts pre-existing boxes as an upload alongside the image set.

Class mapping matches the project's REAL label scheme (car=0/motorcycle=1/bus=2/truck=3), NOT
COCO's -- see docs/RUNLOG.md for why COCO's mapping doesn't apply here.
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "edgeai_v1_best.pt"
FRAMES_ROOT = Path(__file__).resolve().parent.parent / "data" / "pilot_frames"
CONF_THRESHOLD = 0.25  # low bar on purpose: cheaper for a human to delete a bad box in CVAT
                        # than to draw a missed one from scratch

CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

TAGS = ["nostop_nopark", "nostop", "nopark_even", "nopark_odd"]


def build_cvat_xml(image_results: list[tuple[str, int, int, list]]) -> str:
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>"]
    for idx, (name, w, h, boxes) in enumerate(image_results):
        parts.append(f'  <image id="{idx}" name="{escape(name)}" width="{w}" height="{h}">')
        for x1, y1, x2, y2, cls_id, conf in boxes:
            label = CLASS_NAMES.get(cls_id, f"class{cls_id}")
            parts.append(
                f'    <box label="{label}" xtl="{x1:.2f}" ytl="{y1:.2f}" '
                f'xbr="{x2:.2f}" ybr="{y2:.2f}" occluded="0" '
                f'z_order="0"><attribute name="model_conf">{conf:.3f}</attribute></box>'
            )
        parts.append("  </image>")
    parts.append("</annotations>")
    return "\n".join(parts)


def annotate_clip(model: YOLO, tag: str) -> int:
    clip_dir = FRAMES_ROOT / tag
    out_path = clip_dir / "annotations.xml"
    frame_paths = sorted(clip_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"[{tag}] no frames found in {clip_dir}, skipping")
        return 0

    image_results = []
    total_boxes = 0
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        result = model.predict(img, conf=CONF_THRESHOLD, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, cls_id, conf))
        image_results.append((fp.name, w, h, boxes))
        total_boxes += len(boxes)

    out_path.write_text(build_cvat_xml(image_results), encoding="utf-8")
    print(f"[{tag}] {len(frame_paths)} frames -> {total_boxes} draft boxes -> {out_path}")
    return total_boxes


def main():
    if not MODEL_PATH.exists():
        print(f"MISSING model: {MODEL_PATH}")
        sys.exit(1)

    model = YOLO(str(MODEL_PATH))
    grand_total = 0
    for tag in TAGS:
        grand_total += annotate_clip(model, tag)

    print(f"\nTotal draft boxes across all clips: {grand_total}")
    print("Next: in CVAT, create each clip's task, upload its image set, then Actions > Upload "
          "annotations > 'CVAT 1.1' format pointing at that clip's annotations.xml. Review and "
          "correct every frame -- these are draft boxes from a model with a known miss on real "
          "footage (docs/RUNLOG.md), not ground truth.")


if __name__ == "__main__":
    main()
