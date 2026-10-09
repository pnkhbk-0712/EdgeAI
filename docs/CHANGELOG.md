# Model & config changelog

One line per deployed model or zone-config version, per Report Step 7 ("Monitoring, Maintenance &
Iteration" -- versioning). `docs/RUNLOG.md` has the full narrative for each entry; this file is
just the quick-scan index of *what changed* and *what moved*, so a stale model running in a demo
can be spotted at a glance.

| Date | File | What changed | Metric |
|---|---|---|---|
| 2026-10-10 | `models/helmet_v1_best.onnx` | ONNX fp32 export of v1, adopted as the deploy format | 23.3ms/frame CPU (+56.3% vs PyTorch), see `docs/train_helmet/export_benchmark.md` |
| 2026-10-10 | `models/helmet_v1_best_int8.onnx` | INT8 dynamic quantization of v1 -- **tested, rejected, not deployed** | 34.0ms/frame, 46.1% slower than fp32 despite 72.6% smaller; same pattern as the parking project |
| 2026-10-09 | `helmet_v1b` (not integrated) | +20 epochs from v1's weights -- tested, no real change | All per-class AP50 within noise of v1; not adopted, see `docs/RUNLOG.md` 2026-10-09 |
| 2026-10-09 | `models/helmet_v1_best.pt` | First trained helmet model (YOLOv8n, 30/30 epochs, Hard Hat Workers Dataset) | mAP50 0.659 overall (head 0.963, helmet 0.982, **person 0.034** -- known gap, not used by zone logic, see `docs/RUNLOG.md` 2026-10-09) |
| 2026-10-03 | `src/helmet_zone_configs.py` | Initial draft zone polygons (HIGH_RISK/NORMAL) | Not yet calibrated against a real demo-rig frame |

**Rule:** a new model file is never swapped into `demo_helmet.py --model` for an actual
demo/class run without first being checked against a handful of sample frames (same "verify
before trusting" habit used throughout this project) and getting its own row here. Keep the
previous working `.pt` file rather than overwriting it, so pointing `--model` back at it is a
complete rollback.
