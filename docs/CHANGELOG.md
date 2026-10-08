# Model & config changelog

One line per deployed model or zone-config version, per Report Step 7 ("Monitoring, Maintenance &
Iteration" -- versioning). `docs/RUNLOG.md` has the full narrative for each entry; this file is
just the quick-scan index of *what changed* and *what moved*, so a stale model running in a demo
can be spotted at a glance.

| Date | File | What changed | Metric |
|---|---|---|---|
| (pending) | `models/helmet_v1_best.pt` | First trained helmet model (YOLOv8n, 30 epochs, Hard Hat Workers Dataset) | Not yet trained -- see `docs/PROJECT_REPORT.md` Step 3 |
| 2026-10-03 | `src/helmet_zone_configs.py` | Initial draft zone polygons (HIGH_RISK/NORMAL) | Not yet calibrated against a real demo-rig frame |

**Rule:** a new model file is never swapped into `demo_helmet.py --model` for an actual
demo/class run without first being checked against a handful of sample frames (same "verify
before trusting" habit used throughout this project) and getting its own row here. Keep the
previous working `.pt` file rather than overwriting it, so pointing `--model` back at it is a
complete rollback.
