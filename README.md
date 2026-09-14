# Edge AI Illegal Parking Detection — Prototype

Code counterpart to `Edge_AI_Illegal_Parking_Detection_Report.docx`. Structure follows the
report's own sections so it's obvious which folder answers which part of the proposal.

```
prototype/
  data/
    samples/     sample video(s) used for the laptop MVP demo (Sec. II.B)
    raw/         (empty) drop real pilot-site footage / UA-DETRAC subset here — Hieu, Day 1-2
  src/
    zone.py      restricted-zone polygon + dwell-time violation logic (Sec. V / VII.D)
    demo.py      end-to-end MVP: pretrained YOLO + zone.py on a recorded video (Sec. II.B)
  tests/
    test_zone.py deterministic unit tests for the zone-overlap/dwell-time rule (Sec. VIII)
  models/        (empty) fine-tuned weights land here once Hieu trains on real data
  docs/
    RUNLOG.md    what's actually been run, with real measured numbers (not just targets)
```

## Setup

```
pip install -r requirements.txt
```

## Run the MVP demo

```
cd src
python demo.py
```

Downloads `yolov8n.pt` (pretrained COCO weights, no custom training yet) on first run,
processes `data/samples/car-detection.mp4`, writes an annotated video + `events.jsonl`,
and prints real latency/throughput numbers.

## Run the zone-logic tests

```
cd tests
python test_zone.py
```

## Status

See `docs/RUNLOG.md` for what has actually been verified so far, and the published
"Group 9 Work Split" tracker for day-by-day task ownership.
