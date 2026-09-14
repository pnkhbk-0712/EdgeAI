# Implementation Report — Edge AI Illegal Parking Detection

**Date:** 2026-09-14
**Scope:** everything in this report was actually executed on this machine — no projected numbers,
no simulated results. Where something could *not* be run for real (custom dataset, physical
edge hardware), that is stated explicitly rather than estimated.

This report is the execution counterpart to `Edge_AI_Illegal_Parking_Detection_Report.docx`
(the proposal). Section references below (Sec. II, V, VI, VII, VIII) point back to it.

---

## 1. Environment

Already installed on this machine, no setup required:

| Package | Version |
|---|---|
| Python | 3.12.1 |
| OpenCV | 4.10.0 |
| Ultralytics (YOLOv8) | 8.3.86 |
| PyTorch | 2.6.0 (CPU only — no GPU in this environment) |
| ONNX Runtime | 1.30.0 |

Hardware: 13th Gen Intel Core i7-13620H, CPU inference only.

---

## 2. What was built

```
prototype/
  src/zone.py             restricted-zone polygon + dwell-time violation logic
  src/demo.py             end-to-end pipeline on a real public traffic video
  src/demo_synthetic.py   end-to-end pipeline on a synthetic parked-vehicle clip
  tests/test_zone.py      5 deterministic unit tests for the zone logic
  models/                 exported weights (.pt, .onnx, INT8 .onnx) + training smoke-test output
```

All code is committed to a local git repository (`git log` has 1 commit as of this report;
not yet pushed to a shared remote — that still needs the team's GitHub repo URL).

---

## 3. Detection + zone-overlap pipeline (Report Sec. II.B / V)

**Test videos used** (public, Intel IoT DevKit sample-videos, free for development use):

| Video | Resolution | FPS | Frames | Scene |
|---|---|---|---|---|
| `car-detection.mp4` | 768x432 | 12.5 | 377 | top-down parking-lot camera |
| `person-bicycle-car-detection.mp4` | 768x432 | 12.0 | 647 | mixed street traffic |

**Real result:** 58 vehicle detections (car/bus/truck) across `car-detection.mp4`, running the
pretrained YOLOv8n model (stock COCO weights — **no custom fine-tuning yet**) through
`model.track()` for per-vehicle IDs, feeding into the zone-overlap + dwell-time rule.

**Finding — zero violations on real footage, and why that's correct:** every tracked vehicle
in both videos was checked frame-by-frame, and none stays in one place for more than ~4.4
seconds while actually stationary — they are all vehicles driving through frame. The
zone+dwell-time logic is designed to *not* fire on a pass-through vehicle (this is Report
Sec. VIII.A's own "vehicle passing through without stopping" test case), so reporting zero
violations on this footage is the system behaving correctly, not a bug. Confirmed this by
printing raw per-frame track positions for the longest-lived tracks in both videos (see
`docs/RUNLOG.md`).

**Synthetic true-positive check:** since neither public clip contains a genuine parking event,
built `data/samples/synthetic_parked_violation.mp4` — a few seconds of real approach footage
followed by the last real frame held static (simulating the vehicle stopping), clearly labeled
as fabricated, not passed off as real. Running the *actual* pipeline (not the isolated unit
test) against it:

> **Violation fired at frame 29 (2.3 s), car, confidence 0.25** — matching the 2-second dwell
> threshold as designed.

This proves the full detection → tracking → zone-overlap → dwell-time → event pipeline fires
correctly end-to-end on a genuine violation, independent of the unit tests below.

**Zone-logic unit tests** (`tests/test_zone.py`, isolated from footage availability):

| Test | Result |
|---|---|
| No violation before the dwell threshold | PASS |
| Violation fires exactly once, on the correct frame | PASS |
| Pass-through vehicle never violates (Sec. VIII.A) | PASS |
| Leaving the zone resets the dwell counter | PASS |
| Multiple vehicles tracked independently (Sec. VIII.A) | PASS |

5/5 passed.

---

## 4. Model export & optimization (Report Sec. VII.C-D)

| Format | Size | Notes |
|---|---|---|
| PyTorch (.pt, fp32) | 6.5 MB | pretrained COCO weights |
| ONNX (fp32) | 12.85 MB | exported via `model.export(format="onnx")`, 9.6 s |
| ONNX (INT8, dynamic quantization) | **3.5 MB** | 72.8% smaller than fp32 ONNX; well under the <20 MB Table III target |

**Real throughput, full video (`car-detection.mp4`, 377 frames, CPU):**

| Runtime | FPS |
|---|---|
| PyTorch (.pt) | 16.5 |
| **ONNX Runtime (fp32)** | **29.0** (+76%) |
| ONNX Runtime (INT8, single-frame timing) | slower than fp32 (276 ms vs. 31 ms per frame) |

**Honest finding on quantization:** ONNX export alone nearly doubled CPU throughput — a clear,
real win. INT8 dynamic quantization shrank the model by 73% as expected, but made CPU inference
*slower*, not faster, in this environment. This is a known, documented limitation of dynamic
(as opposed to static, calibrated) quantization without dedicated INT8 hardware acceleration —
exactly the kind of hardware-dependent result Fig. 2's "re-measure before accepting" step exists
to catch. **Conclusion for the real target board:** ONNX export is worth keeping; INT8
quantization should be re-evaluated once the actual Raspberry Pi/Jetson hardware is available,
since ARM/Jetson INT8 paths (TensorRT, NNAPI) behave very differently from a generic x86 CPU.

---

## 5. Training pipeline smoke-test (Report Sec. VII.A-B)

**What this is and isn't:** the report's real training data (UA-DETRAC + supplementary
motorcycle images + pilot-site footage) has not been assembled yet — UA-DETRAC requires a
registration step at http://detrac-db.rit.albany.edu/ that needs a person to request access,
and the pilot footage needs the team to actually film the target site. Neither can be done from
this environment.

To still prove the *training pipeline itself* runs correctly — transfer learning from
pretrained weights, fine-tuning, validation, and checkpoint saving — a 3-epoch smoke-test was
run on Ultralytics' built-in `coco8` dataset (8 generic COCO images; **not vehicle-specific
data**, purely a pipeline check):

| Metric | Result |
|---|---|
| Training time | 13.4 s for 3 epochs, CPU |
| Final mAP50 (on the 8-image toy set) | 0.888 |
| Output | `models/smoketest/weights/best.pt` (6.5 MB) |

This confirms `model.train(data=..., epochs=..., ...)` executes cleanly end-to-end on this
machine with no missing dependencies — the real fine-tuning run in Report Sec. VII.B is a data
problem now, not a code or environment problem.

---

## 6. What is genuinely blocked (needs the team, not more code)

| Item | Why it can't be done here |
|---|---|
| Real vehicle dataset (UA-DETRAC + motorcycles + pilot footage) | UA-DETRAC needs a person to register; pilot footage needs someone to film the actual target site |
| Real fine-tuning on that data | Depends on the item above |
| TFLite export for Raspberry Pi | Not attempted yet — needs the TensorFlow toolchain; can be added once a real trained model exists |
| Deployment to actual Raspberry Pi / Jetson hardware | No physical board in this environment |
| Real accuracy/false-alarm numbers against Table III | Needs the real dataset above; current numbers are all pipeline/infrastructure metrics, not detection-quality metrics on real parking scenes |

---

## 7. Bottom line

Everything that can run without a physical camera, a physical board, or a dataset registration
step has been run for real, not simulated:

- End-to-end detection + zone-overlap + dwell-time pipeline: **working**, verified on 2 real
  videos + 1 synthetic true-positive check + 5 unit tests
- ONNX export: **working**, real +76% CPU throughput
- INT8 quantization: **working**, but a real negative result on latency — flagged for
  hardware-specific re-evaluation, not silently accepted
- Training pipeline: **working**, verified via smoke-test; blocked on real data, not code

The gap between this report and the proposal's Table III targets is now entirely a *data and
hardware acquisition* problem — exactly the two things flagged as team-owned, not code-owned,
in the "Group 9 Work Split" tracker.
