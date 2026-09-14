"""Laptop MVP demo (Report Sec. II.B / X): pretrained YOLO + zone-overlap rule
running end-to-end on a recorded video, exactly as the report's MVP describes.

This intentionally uses YOLOv8n's stock COCO weights (no custom training yet --
that's Hieu's Day 3-4 task). COCO already includes car / motorcycle / bus /
truck, so this proves the detection + zone-overlap + event pipeline works
before any custom fine-tuning exists.
"""
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from zone import RestrictedZone

ROOT = Path(__file__).resolve().parent.parent
VIDEO_IN = ROOT / "data" / "samples" / "car-detection.mp4"
VIDEO_OUT = ROOT / "data" / "samples" / "car-detection_annotated.mp4"
EVENTS_OUT = ROOT / "data" / "samples" / "events.jsonl"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Restricted zone, based on inspecting sample frames from car-detection.mp4:
# the top-left stall a gray sedan occupies for the whole clip, designated
# "no parking" for this demo so the dwell-time trigger has something to catch.
# (The open middle aisle was tried first -- vehicles only drive through it,
# never stop, so it correctly produces zero violations; see docs/RUNLOG.md.)
ZONE_POLYGON = [(60, 0), (280, 0), (280, 270), (60, 270)]
DWELL_SECONDS = 2.0


def frame_timestamp(frame_idx, fps):
    total_seconds = frame_idx / fps
    m, s = divmod(int(total_seconds), 60)
    return f"{m:02d}:{s:02d}"


def main():
    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 12.5
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dwell_frames = max(1, int(round(DWELL_SECONDS * fps)))

    zone = RestrictedZone(ZONE_POLYGON, dwell_frames=dwell_frames)
    model = YOLO("yolov8n.pt")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (width, height))

    events = []
    frame_idx = 0
    t_start = time.time()
    detections_total = 0

    results_gen = model.track(
        source=str(VIDEO_IN),
        classes=list(VEHICLE_CLASSES.keys()),
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results_gen:
        frame = result.orig_img.copy()
        zone.draw(frame)

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id, cls_id, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy().astype(int),
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy(),
            ):
                detections_total += 1
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                vehicle_name = VEHICLE_CLASSES.get(cls_id, str(cls_id))

                inside, is_violation, newly_flagged = zone.update(int(track_id), cx, cy)
                color = (0, 0, 255) if is_violation else (0, 200, 0)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{vehicle_name} #{track_id} {conf:.2f}"
                if is_violation:
                    label += " VIOLATION"
                cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if newly_flagged:
                    event = {
                        "event": "illegal_parking",
                        "vehicle": vehicle_name,
                        "track_id": int(track_id),
                        "confidence": round(float(conf), 2),
                        "timestamp": frame_timestamp(frame_idx, fps),
                    }
                    events.append(event)
                    print("VIOLATION:", json.dumps(event))

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    EVENTS_OUT.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    print("\n--- Run summary ---")
    print(f"Frames processed:      {frame_idx}")
    print(f"Vehicle detections:    {detections_total}")
    print(f"Violations flagged:    {len(events)}")
    print(f"Wall-clock time:       {elapsed:.1f}s  ({frame_idx/elapsed:.1f} FPS on CPU)")
    print(f"Annotated video saved: {VIDEO_OUT}")
    print(f"Events log saved:      {EVENTS_OUT}")


if __name__ == "__main__":
    main()
