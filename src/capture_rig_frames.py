"""Capture still frames from the demo camera, with the zone polygons drawn on the live preview.

Two jobs in one tool (Hoang, rig setup):
  1. Zone calibration -- the preview shows HIGH_RISK / NORMAL polygons from helmet_zone_configs.py on the
     real camera view, so you can see whether they land on the rig (edit the *_FRAC lists, rerun).
  2. Test-set capture -- SPACE saves the clean frame (no overlay) as data/rig_frames/<tag>_<NNN>.jpg.
     The tag says which scenario was staged (see docs/RIG_BUILD_GUIDE.md); frames feed the held-out
     test set and are never trained on.
Usage: python src/capture_rig_frames.py --tag full_near [--source 0] [--out data/rig_frames]
Keys: SPACE = save frame, Q = quit.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from helmet_zone_configs import HIGH_RISK_ZONE_POLYGON_FRAC, NORMAL_ZONE_POLYGON_FRAC, scale_polygon

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True, help="scenario name, e.g. full_near, no_helmet_mid (letters/digits/_ only)")
ap.add_argument("--source", default="0", help="webcam index or video path")
ap.add_argument("--out", default="data/rig_frames")
args = ap.parse_args()
assert args.tag.replace("_", "").isalnum(), "--tag: letters, digits and _ only"

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
n = len(list(out.glob(f"{args.tag}_*.jpg")))  # continue numbering, never overwrite
cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
if not cap.isOpened():
    raise SystemExit(f"Cannot open source {args.source}")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    view = frame.copy()
    for frac, color in ((HIGH_RISK_ZONE_POLYGON_FRAC, (0, 0, 255)), (NORMAL_ZONE_POLYGON_FRAC, (0, 215, 255))):
        cv2.polylines(view, [np.array(scale_polygon(frac, w, h), np.int32)], True, color, 2)
    cv2.putText(view, f"{args.tag}  saved:{n}  {w}x{h}  SPACE=save Q=quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("rig capture", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        cv2.imwrite(str(out / f"{args.tag}_{n:03d}.jpg"), frame)
        n += 1
    elif k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
print(f"{n} frame(s) with tag '{args.tag}' in {out}")
