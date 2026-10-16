"""Live helmet-compliance demo (2026-10-03, monitoring/privacy pass 2026-10-08) -- same
architecture as the parking project's demo.py (YOLO track() + RestrictedZone + dwell debounce +
JSONL event log), applied to head/helmet/person instead of vehicles.

Two risk zones instead of one: a "head" detection (= a person without a helmet on) is checked
against both zones. The HIGH_RISK zone (the foam-scaffolding mockup) fires near-instantly; the
NORMAL zone just logs after a short debounce. A "helmet" detection is always drawn green and
never checked against either zone -- wearing a helmet is compliant regardless of which zone
you're standing in.

Monitoring (Report Step 7): periodic compliance snapshots and a "no detections for too long"
failure flag are both written to EVENTS_OUT alongside violation events, so a review of that one
file after a run can distinguish "actually compliant" from "camera/model stopped working" --
those look identical in a system that only logs violations.

Per-track class smoothing (2026-10-11, from a real webcam test): moving the head causes motion
blur, which made the model flip-flop frame-to-frame between "head" (no helmet) and "helmet" on
the exact same real head -- the live manifestation of the 17% head-vs-helmet confusion found in
the confusion matrix (docs/RUNLOG.md 2026-10-09). A single frame's classification is no longer
trusted directly; each track's last CLASS_SMOOTHING_WINDOW frames vote, and the majority decides
whether that track counts as "no helmet" this frame -- so a momentary blur-induced misread doesn't
flip a compliant person into a violation (or hide a real one) for one frame, without hiding the
raw per-frame confidence from the log.

Privacy (Report Step 8): no frame is ever written to disk and no identity is attached to a
track id -- ids come from YOLO's tracker, reset every time this script starts, and are never
linked to a name. Only zone/timestamp/confidence numbers are logged. See the printed notice at
startup and PRIVACY_NOTE below; this is a statement about what this code actually does, not just
a claim made in the report.

Confidence-uncertain band + event retention (2026-10-13, closes two named gaps rather than just
documenting them): (1) a raw per-frame classification confidence inside CONFIDENCE_UNCERTAIN_LOW/
HIGH is not trusted as a clean call either way -- it is treated as "no helmet" for the safety-first
reason the temporal smoothing above already uses (missing a real violation is the costlier
failure), but the resulting label/event is tagged uncertain=True so a human reviewing the log can
tell a low-confidence flag from a clean one, instead of every alert carrying equal weight. (2) a
real bug found while adding this: EVENTS_OUT was being overwritten on every run with only that
run's events, silently discarding all prior history -- there was no retention *policy* partly
because there was no retained history to apply one to. Fixed by loading, purging anything older
than EVENT_RETENTION_DAYS, and appending to the existing log instead of replacing it; every event
now also carries a real "logged_at" timestamp (previously only a relative mm:ss since script start,
which can't support a retention rule at all).

Usage:
    python src/demo_helmet.py                  # live webcam (source 0)
    python src/demo_helmet.py --source path.mp4  # recorded video instead, for dry-running
    python src/demo_helmet.py --model models/helmet_v1_best.pt  # once trained
"""
import argparse
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from ultralytics import YOLO

from zone import RestrictedZone
from helmet_zone_configs import (
    CLASS_NAMES, NO_HELMET_CLASS_ID,
    HIGH_RISK_ZONE_POLYGON_FRAC, NORMAL_ZONE_POLYGON_FRAC, scale_polygon,
    HIGH_RISK_DWELL_SECONDS, NORMAL_DWELL_SECONDS,
)

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

# Per-track head/helmet class smoothing (2026-10-11) -- majority vote over this many of a track's
# most recent frames, not the single current frame. 5 frames is short on purpose: this is a live
# demo with 1-3s dwell thresholds, so the smoothing window needs to stay well under that to not
# itself delay a real alert.
CLASS_SMOOTHING_WINDOW = 5

# A raw per-frame classification confidence inside this band is not trusted as a clean call
# either way (2026-10-13 -- see module docstring). Chosen as a symmetric band around 0.5 (the
# point a binary head/helmet confidence carries the least information); not yet tuned against
# real field data, since no field data exists yet -- a reasonable starting point, not a measured
# optimum, stated honestly rather than implied otherwise.
CONFIDENCE_UNCERTAIN_LOW = 0.40
CONFIDENCE_UNCERTAIN_HIGH = 0.60

# Report Step 8 (Ethics) -- events older than this are dropped on startup rather than kept with
# no policy at all. 30 days is a starting default, not yet matched to any real retention
# requirement (none exists yet for a classroom prototype) -- stated as a parameter precisely so
# it's one line to change once a real policy is decided, not buried in logic.
EVENT_RETENTION_DAYS = 30

PRIVACY_NOTE = (
    "Privacy note: no video frame is saved to disk; track ids reset every run and are never "
    "linked to an identity; only zone/timestamp/confidence events are logged to "
)


def frame_timestamp(frame_idx, fps):
    total_seconds = frame_idx / fps
    m, s = divmod(int(total_seconds), 60)
    return f"{m:02d}:{s:02d}"


def load_and_purge_events(path, retention_days):
    """Load events already logged from prior runs, dropping any older than retention_days.
    Returns the list to keep appending to for this run -- previously this project had no
    cross-run persistence at all (EVENTS_OUT was overwritten every run), so there was nothing
    for a retention policy to act on; this both adds the persistence and the policy together."""
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
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Trained helmet model path")
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

    high_risk = RestrictedZone(
        scale_polygon(HIGH_RISK_ZONE_POLYGON_FRAC, frame_w, frame_h),
        dwell_frames=max(1, round(HIGH_RISK_DWELL_SECONDS * fps)),
    )
    normal = RestrictedZone(
        scale_polygon(NORMAL_ZONE_POLYGON_FRAC, frame_w, frame_h),
        dwell_frames=max(1, round(NORMAL_DWELL_SECONDS * fps)),
    )

    # task="detect" avoids an ONNX-only warning (ONNX export drops the task metadata .pt keeps)
    model = YOLO(str(model_path), task="detect")

    print(PRIVACY_NOTE + str(EVENTS_OUT))

    events = load_and_purge_events(EVENTS_OUT, EVENT_RETENTION_DAYS)
    carried_over_count = len(events)  # events kept from prior runs -- excluded from this run's
                                       # own "violations flagged" summary so that number isn't
                                       # inflated by history the retention fix now preserves.
    frame_idx = 0
    t_start = time.time()
    # rolling (timestamp, is_no_helmet) samples for the live compliance-rate readout
    compliance_window = deque()
    # per-track_id rolling buffer of raw is_no_helmet reads, for the majority-vote smoothing above
    track_class_history = defaultdict(lambda: deque(maxlen=CLASS_SMOOTHING_WINDOW))
    last_detection_time = time.time()
    last_snapshot_time = time.time()
    failure_flagged = False  # avoid re-flagging every frame while still silent

    results_gen = model.track(source=source, persist=True, stream=True, verbose=False)

    for result in results_gen:
        frame = result.orig_img.copy()
        high_risk.draw(frame, color=(0, 0, 255))   # red outline = high-risk zone
        normal.draw(frame, color=(0, 215, 255))    # amber outline = normal zone

        boxes = result.boxes
        if boxes is not None and boxes.id is not None and len(boxes.id) > 0:
            last_detection_time = time.time()
            failure_flagged = False
        if boxes is not None and boxes.id is not None:
            for box, track_id, cls_id, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy().astype(int),
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                class_name = CLASS_NAMES.get(int(cls_id), str(cls_id))
                raw_is_no_helmet = int(cls_id) == NO_HELMET_CLASS_ID
                # A low-confidence raw call is not trusted either way -- biased toward "no
                # helmet" for the same safety-first reason the smoothing tie-break already uses,
                # but tagged so the event log can tell a clean flag from an uncertain one.
                is_uncertain_raw = CONFIDENCE_UNCERTAIN_LOW <= conf <= CONFIDENCE_UNCERTAIN_HIGH
                if is_uncertain_raw:
                    raw_is_no_helmet = True

                now = time.time()
                if class_name == "person":
                    # Person boxes are drawn but never themselves a violation -- the "head"
                    # class (no helmet detected on that head) is what we check against zones.
                    color = (255, 180, 0)
                    label = f"person #{track_id} {conf:.2f}"
                else:
                    # Smoothed, not raw: majority vote over this track's last few frames so one
                    # blurry frame doesn't flip the decision (module docstring, 2026-10-11).
                    history = track_class_history[int(track_id)]
                    history.append(raw_is_no_helmet)
                    is_no_helmet = sum(history) >= len(history) / 2  # ties lean "no helmet" -- safer

                    if not is_no_helmet:
                        color = (0, 200, 0)
                        label = f"helmet #{track_id} {conf:.2f}"
                        compliance_window.append((now, False))
                    else:
                        in_high, high_violation, high_new = high_risk.update(int(track_id), cx, cy)
                        in_normal, normal_violation, normal_new = normal.update(int(track_id), cx, cy)
                        is_violation = high_violation or normal_violation
                        color = (0, 165, 255) if is_uncertain_raw else ((0, 0, 255) if is_violation else (0, 165, 255))
                        label = f"NO HELMET #{track_id} {conf:.2f}"
                        if is_uncertain_raw:
                            label += " [UNCERTAIN - review]"
                        if high_violation:
                            label += " [HIGH-RISK ALERT]"
                        elif normal_violation:
                            label += " [logged]"

                        compliance_window.append((now, True))

                        for zone_name, newly in (("high_risk", high_new), ("normal", normal_new)):
                            if newly:
                                event = {
                                    "event": "no_helmet_violation",
                                    "zone": zone_name,
                                    "track_id": int(track_id),
                                    "confidence": round(float(conf), 2),
                                    "confidence_band": "uncertain" if is_uncertain_raw else "normal",
                                    "timestamp": frame_timestamp(frame_idx, fps),
                                    "logged_at": datetime.now().isoformat(timespec="seconds"),
                                    "reviewed": False,
                                }
                                events.append(event)
                                print("VIOLATION:", json.dumps(event))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # drop samples older than the rolling window and show a live compliance readout
        cutoff = time.time() - COMPLIANCE_WINDOW_SEC
        while compliance_window and compliance_window[0][0] < cutoff:
            compliance_window.popleft()
        if compliance_window:
            no_helmet_count = sum(1 for _, nh in compliance_window if nh)
            compliance_pct = 100.0 * (1 - no_helmet_count / len(compliance_window))
            cv2.putText(frame, f"Compliance (last {COMPLIANCE_WINDOW_SEC}s): {compliance_pct:.0f}%",
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

        # Report Step 7 -- persist periodic compliance snapshots, not just the on-screen readout,
        # so a session's compliance trend can be reviewed after the fact.
        if now - last_snapshot_time >= COMPLIANCE_SNAPSHOT_INTERVAL_SEC:
            last_snapshot_time = now
            if compliance_window:
                no_helmet_count = sum(1 for _, nh in compliance_window if nh)
                snap_pct = round(100.0 * (1 - no_helmet_count / len(compliance_window)), 1)
            else:
                snap_pct = None  # no samples in this window -- distinct from "100% compliant"
            events.append({
                "event": "compliance_snapshot",
                "compliance_pct": snap_pct,
                "sample_count": len(compliance_window),
                "timestamp": frame_timestamp(frame_idx, fps),
                "logged_at": datetime.now().isoformat(timespec="seconds"),
            })

        if not args.no_display:
            cv2.imshow("Helmet compliance demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    EVENTS_OUT.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    print("\n--- Run summary ---")
    print(f"Frames processed:    {frame_idx}")
    print(f"Violations flagged this run: {len(events) - carried_over_count}  "
          f"(+{carried_over_count} carried over from prior runs, {len(events)} total in log)")
    print(f"Wall-clock time:     {elapsed:.1f}s ({frame_idx/max(elapsed,1e-6):.1f} FPS)")
    print(f"Events log saved:    {EVENTS_OUT}")


if __name__ == "__main__":
    main()
