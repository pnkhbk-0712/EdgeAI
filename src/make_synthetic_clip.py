"""Regenerate the synthetic true-positive test clip (see demo_synthetic.py's docstring).

A few seconds of real approach footage from car-detection.mp4, followed by the last
real frame held static long enough to cross whatever DWELL_SECONDS demo_synthetic.py
is currently testing. Re-run this whenever the dwell threshold changes (e.g. the
2s -> 60s change from docs/ZONE_LABEL_DEFINITIONS.md), since the held duration must
stay longer than the threshold for the test to mean anything.
"""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "samples" / "car-detection.mp4"
OUT = ROOT / "data" / "samples" / "synthetic_parked_violation.mp4"

APPROACH_FRAMES = range(205, 220)  # real frames showing the car driving up
HOLD_SECONDS = 65.0                 # must exceed demo_synthetic.py's DWELL_SECONDS


def main():
    cap = cv2.VideoCapture(str(SOURCE))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.5
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT), fourcc, fps, (w, h))

    last_frame = None
    for idx in APPROACH_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            writer.write(frame)
            last_frame = frame
    cap.release()

    hold_frames = int(round(HOLD_SECONDS * fps))
    for _ in range(hold_frames):
        writer.write(last_frame)
    writer.release()

    total = len(list(APPROACH_FRAMES)) + hold_frames
    print(f"Wrote {OUT} -- {total} frames at {fps} fps (~{total/fps:.1f}s), "
          f"{hold_frames} of them held static ({HOLD_SECONDS}s)")


if __name__ == "__main__":
    main()
