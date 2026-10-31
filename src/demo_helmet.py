"""Live PPE-compliance demo (helmet-only 2026-10-03; person-centric multi-item PPE 2026-10-24) -- same
architecture as the parking project's demo.py (YOLO track() + RestrictedZone + dwell debounce +
JSONL event log), applied to people and the PPE they wear.

2026-10-24 redesign (supervisor feedback: a helmet is one small part of PPE): the unit of
assessment is now a tracked *person*, not a head. Each person is checked for every PPE item the
loaded model can detect (helmet, vest, gloves, goggles -- see src/ppe_compliance.py) and the
items REQUIRED in a zone (helmet_zone_configs.REQUIRED_ITEMS) decide what counts as a violation
there. The old 3-class helmet model (head/helmet/person) still loads: "head" is aliased to
no_helmet and only the helmet item is judged. Needs a usable `person` class -- the 3-class helmet
model's person AP50 was ~0.04, so a PPE model trained with a real Person class is the intended one.

Two risk zones: the HIGH_RISK zone (the foam-scaffolding mockup) fires after a short dwell; the
NORMAL zone logs after a longer debounce. Dwell is tracked per (track_id, item), so a person
missing a vest and a helmet raises two independent events.

Monitoring (Report Step 7): periodic compliance snapshots (overall + per item) and a "no
detections for too long" failure flag are written to EVENTS_OUT alongside violation events, so a
review of that one file can distinguish "actually compliant" from "camera/model stopped working".

Per-(track, item) smoothing (2026-10-11, from a real webcam test of the helmet alone): motion blur
made the model flip-flop frame-to-frame on the same real head. A single frame is not trusted;
each (track, item)'s last CLASS_SMOOTHING_WINDOW assessed frames vote and ties lean "missing".

Privacy (Report Step 8): no frame is ever written to disk and no identity is attached to a track
id -- ids come from YOLO's tracker, reset every time this script starts, and are never linked to a
name. Only zone/item/timestamp/confidence numbers are logged. See PRIVACY_NOTE below.

Confidence-uncertain band + event retention (2026-10-13): a detection confidence inside
CONFIDENCE_UNCERTAIN_LOW/HIGH is treated as "missing" (missing a real violation is the costlier
failure) but the event is tagged uncertain so a reviewer can tell it from a clean flag. Events are
loaded, purged past EVENT_RETENTION_DAYS and appended to, not overwritten; each carries "logged_at".

Usage:
    python src/demo_helmet.py                  # live webcam (source 0)
    python src/demo_helmet.py --source path.mp4  # recorded video instead, for dry-running
    python src/demo_helmet.py --model models/ppe_v1_best.onnx  # once the PPE model is trained
"""
import argparse
import json
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from ultralytics import YOLO

from zone import RestrictedZone
from helmet_zone_configs import (
    HIGH_RISK_ZONE_POLYGON_FRAC, NORMAL_ZONE_POLYGON_FRAC, scale_polygon,
    HIGH_RISK_DWELL_SECONDS, NORMAL_DWELL_SECONDS, REQUIRED_ITEMS, INFER_ABSENT,
)
from ppe_compliance import ItemSmoother, assess_person, norm_name, split_dets, supported_items

ROOT = Path(__file__).resolve().parent.parent
EVENTS_OUT = ROOT / "data" / "samples" / "helmet_events.jsonl"
# ONNX fp32, not the raw .pt -- Report Step 4 (2026-10-10) measured ONNX +56.3% faster than
# PyTorch on CPU, with INT8 tried and rejected (slower despite smaller). Ultralytics' YOLO()
# loads .onnx the same way as .pt, no other code here needs to change.
DEFAULT_MODEL = ROOT / "models" / "helmet_v1_best.onnx"

ASSUMED_FPS = 15.0  # webcam fps varies by device; used only to convert dwell seconds -> frames
                     # before the real cap.get(CAP_PROP_FPS) is known for the actual source.

COMPLIANCE_WINDOW_SEC = 30  # rolling window for the live "% compliant" readout

# Report Step 7 -- Monitoring, Maintenance & Iteration
FAILURE_DETECTION_SEC = 60.0        # no detections of ANY kind this long -> flag, don't stay silent
COMPLIANCE_SNAPSHOT_INTERVAL_SEC = 30.0  # persist a compliance snapshot this often, not just on-screen

# Majority vote over this many of a (track, item)'s most recent assessed frames. 5 frames is short
# on purpose: dwell thresholds are 1-3 s, so smoothing must stay well under that.
CLASS_SMOOTHING_WINDOW = 5

# A detection confidence inside this band is not trusted as a clean call (see docstring). Symmetric
# around 0.5, not tuned against field data (none exists yet) -- a starting point, not an optimum.
CONFIDENCE_UNCERTAIN_LOW = 0.40
CONFIDENCE_UNCERTAIN_HIGH = 0.60

# Report Step 8 (Ethics) -- events older than this are dropped on startup. 30 days is a starting
# default, not matched to any real retention requirement; one line to change.
EVENT_RETENTION_DAYS = 30

PRIVACY_NOTE = (
    "Privacy note: no video frame is saved to disk; track ids reset every run and are never "
    "linked to an identity; only zone/item/timestamp/confidence events are logged to "
)


def frame_timestamp(frame_idx, fps):
    total_seconds = frame_idx / fps
    m, s = divmod(int(total_seconds), 60)
    return f"{m:02d}:{s:02d}"


def load_and_purge_events(path, retention_days):
    """Load events already logged from prior runs, dropping any older than retention_days.
    Returns the list to keep appending to for this run."""
    if not path.exists():
        return []
    cutoff = datetime.now() - timedelta(days=retention_days)
    kept = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            logged_at = datetime.fromisoformat(event["logged_at"])
        except (json.JSONDecodeError, KeyError, ValueError):
            # Events from before 2026-10-13 have no "logged_at" field -- keep them rather than
            # silently discarding older history just because the schema changed.
            kept.append(event)
            continue
        if logged_at >= cutoff:
            kept.append(event)
        else:
            dropped += 1
    if dropped:
        print(f"Retention: dropped {dropped} event(s) older than {retention_days} days.")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index (default 0) or video path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained PPE/helmet model path")
    parser.add_argument("--no-display", action="store_true", help="Don't open a preview window")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Train it first (see docs/HELMET_TRAINING.md) or pass --model to point at "
            "an existing checkpoint, e.g. models/yolov8n.pt for a quick untrained smoke test."
        )

    cap_probe = cv2.VideoCapture(source)
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or ASSUMED_FPS
    frame_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap_probe.release()
    print(f"Camera reports {frame_w}x{frame_h} @ {fps:.1f}fps -- scaling zone polygons to this.")

    zones = {
        "high_risk": RestrictedZone(
            scale_polygon(HIGH_RISK_ZONE_POLYGON_FRAC, frame_w, frame_h),
            dwell_frames=max(1, round(HIGH_RISK_DWELL_SECONDS * fps))),
        "normal": RestrictedZone(
            scale_polygon(NORMAL_ZONE_POLYGON_FRAC, frame_w, frame_h),
            dwell_frames=max(1, round(NORMAL_DWELL_SECONDS * fps))),
    }
    zone_colors = {"high_risk": (0, 0, 255), "normal": (0, 215, 255)}  # red / amber outlines

    # task="detect" avoids an ONNX-only warning (ONNX export drops the task metadata .pt keeps)
    model = YOLO(str(model_path), task="detect")
    names = {int(k): norm_name(v) for k, v in model.names.items()}
    if "person" not in names.values():
        raise SystemExit(f"Model classes {sorted(names.values())} have no 'person' class -- a "
                         "person-centric PPE check needs one.")
    can_see = supported_items(names.values())
    required = {z: [i for i in REQUIRED_ITEMS[z] if i in can_see] for z in zones}
    assessed = sorted({i for items in required.values() for i in items})
    print(f"Model detects PPE items {can_see}; required per zone: {required}")
    if not assessed:
        raise SystemExit("None of the required PPE items is detectable by this model.")

    print(PRIVACY_NOTE + str(EVENTS_OUT))

    events = load_and_purge_events(EVENTS_OUT, EVENT_RETENTION_DAYS)
    carried_over_count = len(events)  # events kept from prior runs -- excluded from this run's summary
    frame_idx = 0
    t_start = time.time()
    # rolling (timestamp, item, is_missing) samples for the live compliance-rate readout
    compliance_window = deque()
    smoother = ItemSmoother(CLASS_SMOOTHING_WINDOW)
    last_detection_time = time.time()
    last_snapshot_time = time.time()
    failure_flagged = False  # avoid re-flagging every frame while still silent

    def compliance_pct(samples):
        return round(100.0 * (1 - sum(1 for s in samples if s[2]) / len(samples)), 1) if samples else None

    results_gen = model.track(source=source, persist=True, stream=True, verbose=False)

    for result in results_gen:
        frame = result.orig_img.copy()
        for zname, zone in zones.items():
            zone.draw(frame, color=zone_colors[zname])

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            last_detection_time = time.time()
            failure_flagged = False
            xyxy = boxes.xyxy.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

            people, item_dets = [], []
            for box, tid, cid, conf in zip(xyxy, ids, clses, confs):
                name = names.get(int(cid), str(cid))
                if name == "person":
                    if tid is not None:
                        people.append((int(tid), box))
                else:
                    item_dets.append((name, box, float(conf)))
                    # thin box per detected item: green = PPE present, red = explicit "no_*"
                    c = (0, 0, 255) if name.startswith("no_") else (0, 200, 0)
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), c, 1)

            now = time.time()
            own_dets = split_dets([b for _, b in people], item_dets)
            for (tid, pbox), my_dets in zip(people, own_dets):
                px1, py1, px2, py2 = pbox
                cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
                status = assess_person(pbox, my_dets, assessed, result.orig_shape[0], INFER_ABSENT)
                missing, alerting, uncertain_any = [], set(), False
                for item, (st, conf) in status.items():
                    if st == "unknown":
                        continue
                    uncertain = conf is not None and CONFIDENCE_UNCERTAIN_LOW <= conf <= CONFIDENCE_UNCERTAIN_HIGH
                    raw_missing = st == "missing" or uncertain
                    is_missing = smoother.update((tid, item), raw_missing)
                    compliance_window.append((now, item, is_missing))
                    key = (tid, item)
                    if not is_missing:
                        for zone in zones.values():
                            zone.track_dwell.pop(key, None)  # compliant again -> dwell restarts
                        continue
                    missing.append(item)
                    uncertain_any |= uncertain
                    for zname, zone in zones.items():
                        if item not in required[zname]:
                            continue
                        _, violation, newly = zone.update(key, cx, cy)
                        if violation and zname == "high_risk":
                            alerting.add("HIGH-RISK")
                        elif violation:
                            alerting.add("logged")
                        if newly:
                            event = {
                                "event": "ppe_violation",
                                "missing_item": item,
                                "zone": zname,
                                "track_id": tid,
                                "confidence": round(conf, 2) if conf is not None else None,
                                "confidence_band": "uncertain" if uncertain else "normal",
                                "timestamp": frame_timestamp(frame_idx, fps),
                                "logged_at": datetime.now().isoformat(timespec="seconds"),
                                "reviewed": False,
                            }
                            events.append(event)
                            print("VIOLATION:", json.dumps(event))

                if not missing:
                    color, label = (0, 200, 0), f"person #{tid} OK"
                else:
                    color = (0, 0, 255) if "HIGH-RISK" in alerting else (0, 165, 255)
                    label = f"#{tid} NO {','.join(missing).upper()}"
                    if uncertain_any:
                        label += " [UNCERTAIN - review]"
                    if alerting:
                        label += " [" + "/".join(sorted(alerting)) + "]"
                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), color, 2)
                cv2.putText(frame, label, (int(px1), max(0, int(py1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # drop samples older than the rolling window and show a live compliance readout
        cutoff = time.time() - COMPLIANCE_WINDOW_SEC
        while compliance_window and compliance_window[0][0] < cutoff:
            compliance_window.popleft()
        pct = compliance_pct(compliance_window)
        if pct is not None:
            cv2.putText(frame, f"Compliance (last {COMPLIANCE_WINDOW_SEC}s): {pct:.0f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Report Step 7 -- failure/drift detection: distinguish "genuinely compliant" from
        # "camera or model stopped producing detections" instead of letting both look like silence.
        now = time.time()
        if not failure_flagged and (now - last_detection_time) > FAILURE_DETECTION_SEC:
            failure_flagged = True
            event = {
                "event": "monitoring_alert",
                "reason": "no_detections",
                "seconds_silent": round(now - last_detection_time, 1),
                "timestamp": frame_timestamp(frame_idx, fps),
                "logged_at": datetime.now().isoformat(timespec="seconds"),
            }
            events.append(event)
            print("MONITORING ALERT (no detections -- check camera/model, not necessarily compliant):",
                  json.dumps(event))

        # Report Step 7 -- persist periodic compliance snapshots (overall + per item).
        if now - last_snapshot_time >= COMPLIANCE_SNAPSHOT_INTERVAL_SEC:
            last_snapshot_time = now
            events.append({
                "event": "compliance_snapshot",
                "compliance_pct": compliance_pct(compliance_window),  # None = no samples, not 100%
                "by_item": {i: compliance_pct([s for s in compliance_window if s[1] == i]) for i in assessed},
                "sample_count": len(compliance_window),
                "timestamp": frame_timestamp(frame_idx, fps),
                "logged_at": datetime.now().isoformat(timespec="seconds"),
            })

        if not args.no_display:
            cv2.imshow("PPE compliance demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    EVENTS_OUT.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    violations = sum(1 for e in events[carried_over_count:] if e.get("event") == "ppe_violation")
    print("\n--- Run summary ---")
    print(f"Frames processed:    {frame_idx}")
    print(f"Violations flagged this run: {violations}  "
          f"(+{carried_over_count} events carried over from prior runs, {len(events)} total in log)")
    print(f"Wall-clock time:     {elapsed:.1f}s ({frame_idx/max(elapsed,1e-6):.1f} FPS)")
    print(f"Events log saved:    {EVENTS_OUT}")


if __name__ == "__main__":
    main()
