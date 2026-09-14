# Run Log

Real results only — no projected/planned numbers in this file.

## 2026-09-14 — Environment + laptop MVP baseline (Hieu/Hanh Day 1)

**Environment:** OpenCV 4.10.0, Ultralytics 8.3.86, PyTorch 2.6.0 (CPU) — already installed,
no setup needed.

**Sample video:** `car-detection.mp4`, Intel IoT DevKit sample-videos (public, free to use for
development/testing), 768x432, 12.5 fps, 377 frames (~30s), top-down parking-lot camera.

**What was run:** `src/demo.py` — pretrained YOLOv8n (stock COCO weights, no fine-tuning yet)
with `model.track()` for per-vehicle IDs, feeding into `src/zone.py`'s polygon + dwell-time
rule.

**Real measured numbers (first run):**
- 58 vehicle detections (car/bus/truck) across 377 frames
- **18.4 FPS on CPU** — already above the ≥5 FPS target in Table III, on a pretrained model,
  before any optimization
- 0 violations flagged

**Why 0 violations, and why that's correct, not a bug:** printed the raw per-frame track
positions and confirmed every vehicle in this specific clip drives through the frame in under
~1 second — none actually stops. The zone+dwell-time logic is designed to *not* fire on a
pass-through vehicle (Report Sec. VIII.A test case), so zero violations on this footage is the
expected, correct output. Tried two zone placements (the open drive-aisle, then the top-left
stall) — same result both times, confirming it's the footage, not the zone placement.

**Follow-up:** added `tests/test_zone.py` — 5 deterministic tests against synthetic
coordinates, isolating the zone-overlap/dwell-time logic from footage availability:
1. no violation before the dwell threshold
2. violation fires exactly once, on the correct frame
3. pass-through vehicle never violates
4. leaving the zone resets the dwell counter
5. two vehicles tracked independently

All 5 passed.

**Next real test needed:** footage (or self-recorded pilot clip) with a vehicle that actually
stays put for 2+ seconds in the zone, to see a true-positive fire end-to-end. This is exactly
what Hieu's Day 1-2 pilot-site filming is for.

**Not yet done:** custom dataset (UA-DETRAC requires registration at
http://detrac-db.rit.albany.edu/ — someone needs to request access), fine-tuning, quantization,
edge-device deployment. Baseline above uses stock pretrained weights only.
