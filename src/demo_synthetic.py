"""Synthetic true-positive check (NOT a real-world result).

`car-detection.mp4` and `person-bicycle-car-detection.mp4` were both checked
and contain no vehicle that actually stops -- every track is a vehicle
driving through frame (see docs/RUNLOG.md). To prove the dwell-time
violation trigger fires correctly through the REAL pipeline (not just the
isolated unit test in tests/test_zone.py), this script runs the same
detector + zone logic against `data/samples/synthetic_parked_violation.mp4`:
a few seconds of real approach footage from car-detection.mp4, followed by
the last real frame held static to simulate the vehicle stopping and
staying. The held portion is fabricated -- flagged clearly here and in the
report -- not a genuine parking event.
"""
from pathlib import Path

import cv2
from ultralytics import YOLO

from zone import RestrictedZone

ROOT = Path(__file__).resolve().parent.parent
VIDEO_IN = ROOT / "data" / "samples" / "synthetic_parked_violation.mp4"
VIDEO_OUT = ROOT / "data" / "samples" / "synthetic_parked_violation_annotated.mp4"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
ZONE_POLYGON = [(270, 0), (500, 0), (500, 230), (270, 230)]  # covers the held car's position
# Matches demo.py's real threshold (docs/ZONE_LABEL_DEFINITIONS.md Q2: 2s -> 60s).
# The clip's held portion must be regenerated longer than this -- see make_synthetic_clip.py.
DWELL_SECONDS = 60.0


def main():
    cap = cv2.VideoCapture(str(VIDEO_IN))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.5
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dwell_frames = max(1, int(round(DWELL_SECONDS * fps)))
    zone = RestrictedZone(ZONE_POLYGON, dwell_frames=dwell_frames)
    model = YOLO("yolov8n.pt")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (width, height))

    results_gen = model.track(source=str(VIDEO_IN), classes=list(VEHICLE_CLASSES.keys()),
                               persist=True, stream=True, verbose=False)
    frame_idx = 0
    violations = []
    for result in results_gen:
        frame = result.orig_img.copy()
        zone.draw(frame)
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for box, tid, cls_id, conf in zip(boxes.xyxy.cpu().numpy(),
                                                boxes.id.cpu().numpy().astype(int),
                                                boxes.cls.cpu().numpy().astype(int),
                                                boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                name = VEHICLE_CLASSES.get(cls_id, str(cls_id))
                inside, is_violation, newly_flagged = zone.update(int(tid), cx, cy)
                color = (0, 0, 255) if is_violation else (0, 200, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{name} #{tid} dwell={zone.track_dwell.get(int(tid),0)}"
                if is_violation:
                    label += " VIOLATION"
                cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if newly_flagged:
                    violations.append((frame_idx, name, float(conf)))
                    print(f"VIOLATION fired at frame {frame_idx} ({frame_idx/fps:.1f}s): "
                          f"{name} track #{tid}, confidence {conf:.2f}")
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nTotal frames: {frame_idx}, violations fired: {len(violations)}")
    print(f"Annotated video: {VIDEO_OUT}")


if __name__ == "__main__":
    main()
