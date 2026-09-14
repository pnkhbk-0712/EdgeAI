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

**Not yet done (at time of first entry):** custom dataset, fine-tuning, quantization, edge-device
deployment.

---

## 2026-09-14 (later same day) — Full pipeline run: synthetic violation, export, quantization, training smoke-test

See `docs/IMPLEMENTATION_REPORT.md` for the full write-up. Summary of what changed since the
entry above:

- Checked a second real video (`person-bicycle-car-detection.mp4`, 647 frames) — same finding,
  no genuinely parked vehicle, longest track is a 53-frame diagonal drive-through.
- Built `synthetic_parked_violation.mp4` (real approach footage + held-static frame) and ran the
  **actual pipeline** (not just the unit test) against it: violation correctly fired at frame 29
  (2.3s), confirming the end-to-end system, not just the isolated logic, works on a true
  positive.
- Exported `yolov8n.pt` -> ONNX: 12.85 MB, 9.6s. Real full-video FPS: **16.5 (PyTorch) -> 29.0
  (ONNX)**, +76%.
- INT8 dynamic-quantized the ONNX model: 12.85 MB -> **3.5 MB** (-72.8%), but per-frame latency
  got *worse* (31ms fp32 vs 276ms INT8, single-frame ONNX Runtime timing) — a real, reportable
  negative result, not swept under the rug. Likely cause: dynamic quantization overhead without
  INT8 hardware acceleration on this generic x86 CPU; needs re-testing on the actual
  Jetson/Pi target.
- Ran a 3-epoch training smoke-test on Ultralytics' built-in `coco8` (8 generic images, not
  vehicle data) to prove `model.train(...)` runs cleanly end-to-end here: 13.4s, mAP50=0.888 on
  the toy set. Proves the training *pipeline*, not model quality — real vehicle data is still
  needed for a meaningful model.

**Still blocked on the team, not on code:** UA-DETRAC registration, pilot-site filming, and any
physical edge hardware (Raspberry Pi / Jetson) to deploy to or benchmark on.
