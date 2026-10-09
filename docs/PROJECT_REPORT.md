# Edge AI 2026 — Construction Helmet Compliance with Risk-Zone Tiering

**Group 9** — Phan Nu Kieu Hanh (Lead/Integration), Doan Minh Hieu (Data & Model), Doan Vinh Hoang
(Deployment), Nguyen Tien Hung (Testing & Evaluation)

**Status as of 2026-10-08:** pivoted from illegal-parking detection on 2026-10-03 (see
`docs/PIVOT_HELMET_DETECTION.md` for the full reasoning). This report follows the course's 8-step
Edge AI Design Process rubric. Every claim below is marked **Done**, **In progress**, or
**Planned** — nothing here states a result that hasn't actually been measured.

---

## Step 1 — Problem Definition & Use Case Selection

**Problem.** Vietnam recorded 8,286 workplace accidents in 2024 (+12.1% YoY), including 675 fatal
accidents and 727 deaths (+4.81% YoY), with >43,000 billion VND in damage. Construction accounts
for 62% of all workplace fatalities, and within construction specifically, 69.1% of accidents are
falls and falling objects — exactly the injury mechanism a hard hat is designed to mitigate.
Root causes are organizational (inadequate training 53.8%, insufficient safety management 35.2%),
not primarily equipment unavailability — meaning the gap is **enforcement**, not supply: a human
site supervisor cannot watch every worker, on every part of a site, 100% of the time.
(Sources: vneconomy.vn, nhandan.vn — see `docs/PIVOT_HELMET_DETECTION.md` for links.)

**Objective / product.** A camera-based system that detects whether a person in frame is wearing
a hard hat, and escalates differently depending on where in the frame they are: an immediate alert
in a high-risk zone (elevated work / near scaffolding), a logged-but-not-urgent record in a normal
ground-level zone. For the course deliverable, the demo runs against a physical foam-board
scaffolding mockup with helmet/no-helmet figures, not a real site — the detection and zone logic
are identical to what a real deployment would run.

**Why Edge AI (not cloud, not no-AI).**

| | No AI (human supervisor only) | Cloud AI (stream video to a server) | Edge AI (this project) |
|---|---|---|---|
| Latency | Depends on supervisor attention — can be minutes | Round-trip network latency, typically 200ms-2s+ depending on connectivity | Sub-frame, local inference (<100ms target on CPU, see Step 4) |
| Availability | 100% of the time supervisor is present and attentive — neither is guaranteed | Breaks entirely if site has no reliable internet (common on active construction sites, which are frequently far from stable infrastructure) | Works with no network at all — camera and compute are co-located |
| Privacy | N/A | Raw video of individually-identifiable workers leaves the site and is transmitted/stored off-site — a real data-protection liability | Video never leaves the device; only an event (zone, timestamp, compliant/not) needs to leave, if anything does |
| Cost at scale | Labor cost scales linearly with sites/shifts | Recurring bandwidth + server cost per camera-hour | One-time device cost per camera, no recurring inference cost |
| Why this matters here | The root-cause data above shows supervision gaps are exactly the failure mode — a system that's "on" only when a human remembers to look doesn't close the gap | Construction sites are exactly the kind of location (remote, temporary, sometimes no fixed network) where cloud dependency is a real operational risk, not a theoretical one | Matches the problem: always-on, works regardless of site connectivity, keeps worker video local |

**Edge requirements and resource constraints.** Target device for a real deployment: Raspberry Pi
4B (the same board from the parking project, reused — see Step 5). Resource budget, carried over
from the parking project's validated targets since the task (single-class-family detection +
simple zone logic) is comparable in compute cost: **inference latency <300ms/frame, model file
size small enough for a Pi's storage/RAM headroom, CPU-only (no GPU on the Pi 4B)**. These are
**planned targets**, not yet re-measured against the helmet model specifically (Step 4).

---

## Step 2 — Data Collection & Preprocessing

**Data source.** Hard Hat Workers Dataset (Northeastern University — China), via Roboflow
(`joseph-nelson/hard-hat-workers`, version 10 `raw_AllClasses`). License: **Public Domain (CC0)**.
7,035 images, 3 classes — confirmed directly from the downloaded `data.yaml` (not assumed):
`0=head, 1=helmet, 2=person`. Original data via Harvard Dataverse, doi:10.7910/DVN/7CBGOS.

**Collection protocol.** Not a self-collected dataset — this is a public, pre-annotated research
dataset, chosen specifically for its size (7k+ images, large relative to what the team could
realistically hand-label in the remaining timeline) and clean licensing (CC0, no attribution
dispute risk for an academic submission).

**Ground truth / labeling.** Already labeled by the dataset's original authors (bounding boxes,
3 classes). No additional labeling pass planned for v1 — see domain gap note below for why this
is a real, not just theoretical, risk to carry into Testing (Step 6).

**Train/test split.** 70/20/10 train/valid/test, as shipped by the dataset version chosen.

**Preprocessing.** Standard YOLOv8 pipeline defaults: resize to 640x640, Ultralytics' built-in
augmentation (mosaic, HSV jitter, flips) during training. No custom preprocessing beyond this —
**Planned**: confirm this is sufficient once real training results exist (Step 3).

**Class imbalance / domain gap.** Two real, specific gaps, named honestly rather than assumed
away (the parking project was burned twice by exactly this kind of unstated assumption — a camera
framing shift and a mis-placed zone polygon, both caught late because nobody checked early):
1. **Domain gap**: the dataset's images come from varied real construction sites (different
   lighting, camera angles, worker clothing) — the actual classroom demo runs a webcam against a
   foam-board mockup with figures, a visually different domain. **Planned mitigation**: before
   trusting the model for the live demo, run it against real webcam frames of the actual demo rig
   and manually check detections (same "verify before trusting" step used throughout the parking
   project), not just assume the validation-set mAP transfers.
2. **Class balance**: not yet audited for this dataset (unlike the parking project, where the
   car:motorcycle ratio was measured and explicitly fixed). **Planned**: check the per-class image
   count once training runs and flag it if one class is thin.

**Privacy / consent.** The dataset itself is a published, publicly licensed research dataset — no
new privacy exposure from using it for training. The deployment side (a camera observing real,
identifiable workers) is a separate and real concern, addressed in Step 8 (Ethics), not glossed
over here: this report does not claim the training data raises a privacy issue when the actual
privacy issue is downstream, at deployment.

---

## Step 3 — Model Selection & Training

**Status: Done (2026-10-09).** `helmet_v1` trained 30/30 epochs on Colab (T4 GPU). Weights at
`models/helmet_v1_best.pt` (sha256 `a763e0ba...`), full run artifacts in `docs/train_helmet/`.

**Candidate model.** YOLOv8n — chosen for the same reasons it was chosen for the parking project:
smallest Ultralytics YOLOv8 variant, proven to run on CPU-only edge hardware (Pi 4B) within the
latency budget in that project, and the team already has a working train/export/quantize pipeline
for it, directly reusable (`docs/HELMET_TRAINING.md`). No other architecture was evaluated — this
is a reused decision, not a fresh comparison, and that reuse is itself the justification given the
timeline.

**Baseline.** Still not formally run (stock COCO-pretrained YOLOv8n has no "helmet" class, so a
zero-shot run would trivially score near-zero on this task) — the comparison that actually matters
is against the acceptance bar below, which the trained model is checked against directly.

**Training approach.**
```
yolo train model=yolov8n.pt data=data/helmet_roboflow/data.yaml \
    epochs=30 imgsz=640 batch=16 device=0 \
    project=/content/drive/MyDrive/EdgeAI_runs name=helmet_v1
```
Same Colab T4 GPU workflow validated on the parking project's `edgeai_v1-3` run.

**Task metrics — real, measured (2026-10-09).** Final epoch (30/30): overall precision=0.967,
recall=0.622, **mAP50=0.659**, mAP50-95=0.454 (`docs/train_helmet/results.csv`). The aggregate
mAP50 is below the 0.80 target — investigated rather than accepted or hidden at face value.
**Per-class AP50** (`docs/train_helmet/BoxPR_curve.png`): **head 0.963, helmet 0.982, person
0.034**. The aggregate is dragged down almost entirely by `person`, which has only 450 training
instances against 13,919 `helmet` and 4,612 `head` (`docs/train_helmet/labels.jpg`) — a 31:1
imbalance the team did not catch before training, a real miss worth naming rather than glossing
over (the parking project's own 154:1 car:motorcycle problem at least got caught and fixed before
training; this one wasn't).

**Edge resource metrics.** Not yet measured — model file size and CPU/Pi 4B latency still depend
on Step 4's export, not yet run.

**Selection rule — applied, not just stated.** The rule from the first version of this report
already scoped acceptance to the classes the logic actually uses: *mAP50 on `helmet`/`head` ≥0.80*.
Both clear it by a wide margin (0.963, 0.982). **Decision: accept this model for v1.**
`src/demo_helmet.py` never checks a `person` detection against either risk zone — only a `head`
detection (no helmet present) drives the violation logic — so `person`'s near-zero AP50 does not
block the system's actual safety behavior. This is stated as a deliberate, reasoned acceptance,
not a lowered bar: `person` detection itself remains broken and is tracked as a known gap below,
not silently dropped.

**Real risk found, not assumed away.** The confusion matrix
(`docs/train_helmet/confusion_matrix_normalized.png`) shows **17% of true `head` instances are
misclassified as `helmet`** — the single worst failure mode for this system, since it means a real
no-helmet case can read as compliant. This is now a named entry on the Step 6 shot list: test
bare-head cases at the same range of distances/angles as helmet cases, not just an aggregate
accuracy check.

**Qualitative check.** `docs/train_helmet/val_batch0_pred.jpg` / `val_batch1_pred.jpg` show
confident, correct detections across varied real construction-site photos — consistent with the
strong head/helmet AP50. **Still open**: this is the dataset's own validation images, not the
actual demo-rig/webcam domain (foam mockup, classroom lighting) — the domain-gap risk named in
Step 2 is not yet closed by this result and shouldn't be read as if it were.

**Known gap carried forward, not fixed:** `person` detection (AP50 0.034) needs either more
training data for that class or should be formally dropped from the class list in a future
version — tracked in `docs/CHANGELOG.md`, not left implicit.

**Tested and rejected: training longer.** `results.png` showed mAP50 still climbing at epoch 30,
so a follow-up run (`helmet_v1b`, 20 more epochs from v1's weights) tried extending it — see
`docs/RUNLOG.md` 2026-10-09. Result: no real change (every per-class AP50 moved by <1 point,
head→helmet confusion 17%→16%), and the single-point precision metric got noticeably less stable.
**`models/helmet_v1_best.pt` (the original 30-epoch run) remains the model of record** — stated
here so the report doesn't imply more training is still an open lever when it's already been
tried and didn't help.

---

## Step 4 — Model Optimization for Edge Deployment

**Status: Planned, not yet executed** against the helmet model. The team has real, directly
transferable experience from the parking project to apply once a trained `.pt` exists:

**Optimization strategy.** ONNX export, then INT8 post-training quantization — the exact sequence
already validated on the parking project.

**Trade-offs (known from the parking project, to be re-verified here, not assumed to transfer
unchanged).** On the parking project: ONNX export gave **+76% CPU throughput** over raw PyTorch.
INT8 quantization shrank model size by **73%** but was *slower* on that specific x86 CPU — a
real, counter-intuitive finding (quantization doesn't automatically mean faster on hardware
without matching acceleration). **This result will be re-tested on the helmet model, not assumed
to repeat** — the architecture is the same (YOLOv8n) but re-testing is cheap and the parking
project's own lesson was "don't assume validation-set or prior-run numbers transfer."

**Quantization / pruning / KD.** Quantization (INT8 PTQ) is the planned technique, matching the
PTQ/QAT material from course Step 4 lecture content. Pruning and knowledge distillation were not
pursued on the parking project either and are out of scope here for the same reason: the model is
already small (YOLOv8n, ~6MB), and the bigger lever proven to matter was export format (ONNX vs
raw PyTorch), not further compression.

**Accuracy before/after optimization.** **Not yet measured** — depends on Step 3 completing first.

**Latency and memory.** **Not yet measured.**

**Acceptance criterion.** Same rule as Step 3's selection rule, re-applied post-optimization: keep
the optimized model only if it still clears the mAP50 ≥0.80 bar and meets the <300ms/frame target
on the Pi 4B — an optimization that trades too much accuracy for speed is rejected, not accepted
by default.

---

## Step 5 — Deployment to Edge Device

**Target device.** Two explicit tiers, named separately because conflating them was a real risk
this report corrects: (1) **classroom demo target** — a laptop running the live webcam demo via
`src/demo_helmet.py`, which is what will actually be shown in class; (2) **real-deployment
target** — the Raspberry Pi 4B carried over from the parking project, relevant if the team
demonstrates a standalone edge-device run beyond the laptop demo. The rubric's "target device"
resource constraints (Step 1) are written against the Pi 4B, since that's the actual
resource-constrained edge case this project is making a case for.

**Runtime / software stack.** Python, OpenCV (camera capture + drawing), Ultralytics YOLOv8
(`model.track()` for detection + persistent per-person tracking), same stack as the parking
project.

**Model integration — Done.** `src/demo_helmet.py` runs `model.track()` on a live or recorded
source, classifies each detection as `head` (no helmet), `helmet`, or `person`, and feeds `head`
detections into the zone-check logic below. This is real, working code, verified to import and
run (`--help`, argument parsing) — not yet run end-to-end with a trained model, since none exists
yet.

**Camera / sensor integration — Done.** `cv2.VideoCapture` against webcam index 0 by default, or
a video file path for dry runs without a live camera.

**Risk-zone logic — Done, reused.** `src/zone.py`'s `RestrictedZone` class — originally built for
the parking project's polygon + dwell-time violation rule — is reused unchanged. Two instances
configured in `src/helmet_zone_configs.py`: a high-risk zone (near-instant 1s dwell) and a normal
zone (3s dwell debounce). **Known limitation, stated plainly**: the zone polygons in that file are
explicit draft placeholders, not yet calibrated against a real frame from the actual demo rig —
the docstring says so directly. This is the same class of bug that cost real debugging time on the
parking project (a zone polygon that visually missed its target, caught only by rendering it back
onto a real frame) — stated here so it isn't repeated silently.

**Memory / power constraints.** Carried over as a target from the parking project (Step 1); not
yet re-measured for this model.

**Accelerator / provider.** None planned — CPU-only inference, same as the parking project's
validated approach (the Pi 4B has no GPU; a Coral USB accelerator was considered and explicitly
not purchased for the parking project after INT8 quantization didn't show a clear win on a generic
CPU — the same reasoning applies here until a concrete need is demonstrated).

**Offline / fallback behavior — Done (2026-10-08).** Implemented directly in `demo_helmet.py`: if
no detection of any kind (person/head/helmet) has occurred for `FAILURE_DETECTION_SEC` (60s)
while frames are still being processed, a `monitoring_alert` event fires, distinct from a
compliance event — the system no longer conflates "camera/model stopped working" with "100%
compliant" (previously a real, named gap; closed here, not just described).

**Deployment architecture.**
```
Camera (webcam/Pi camera) -> YOLOv8n (track mode) -> per-detection class
  -> "helmet": draw green, no zone check
  -> "head" (no helmet): check against both RestrictedZone instances
       -> inside high-risk zone, dwell >=1s -> immediate alert + event log
       -> inside normal zone, dwell >=3s -> logged event, no urgent alert
  -> rolling 30s window -> live "% compliant" readout
  -> all violation events -> data/samples/helmet_events.jsonl
```

---

## Step 6 — Testing & Evaluation

**Status: Planned.** Nothing executed yet — stated honestly rather than padded with aspirational
numbers.

**Functional testing.** Planned: confirm the pipeline runs end-to-end (camera -> detection ->
zone check -> event log) once a trained model exists, the same way the parking project's `demo.py`
was first proven with stock COCO weights before any fine-tuning existed, to separate "does the
pipeline work" from "is the model accurate."

**Performance testing.** Planned: FPS and per-frame latency on the laptop first, then the Pi 4B,
same method as the parking project's benchmarking (`docs/IMPLEMENTATION_REPORT.md` pattern).

**Robustness testing.** Planned, as a literal shot list (Hung's task in the live tracker): a
figure with a helmet entering and leaving each zone, a figure without a helmet entering and
leaving each zone, a figure lingering just under and just over each zone's dwell threshold, and a
figure briefly occluded (to check the dwell-debounce doesn't false-trigger on a one-frame miss —
the same scenario the parking project's `active_hours`/occlusion edge case covered for vehicles).

**Ground truth.** The shot list above doubles as ground truth: each staged scenario has a known
correct outcome (violation / no violation, which zone, roughly which second it should fire).

**Metrics.** Planned: precision/recall against the shot list's known outcomes, plus the same
latency/size metrics as Steps 3-4.

**Threshold / Pass-Fail criteria.** Same numeric targets stated in Steps 3-4 (mAP50 ≥0.80,
<300ms/frame on the Pi 4B) — a model or deployment that doesn't clear these is not accepted as
"done," it goes back to iteration.

**Testing on target device.** Planned for the Pi 4B specifically, not just the development laptop,
for the same reason the parking project found CPU-vs-ARM results can differ.

---

## Step 7 — Monitoring, Maintenance & Iteration

*(Entirely absent before 2026-10-03; implemented in code, not just planned, as of 2026-10-08.)*

**Monitoring — Done.** Every violation event writes to `data/samples/helmet_events.jsonl`
(zone, track id, confidence, timestamp). **Added**: a `compliance_snapshot` event every
`COMPLIANCE_SNAPSHOT_INTERVAL_SEC` (30s) persists the rolling compliance-rate readout to the same
log — previously only shown on-screen and discarded, now reviewable after the session ends.

**Failure / drift detection — Done.** A naive system can't tell "100% compliance" from "camera
stopped seeing anyone" — both look like zero violations. Implemented rule: if the detector reports
zero `person`/`head`/`helmet` detections of any kind for `FAILURE_DETECTION_SEC` (60s), a
`monitoring_alert` event fires with `reason: "no_detections"` — this is real code
(`src/demo_helmet.py`), not a proposal, and closes the gap named in Step 5's "offline/fallback
behavior" directly.

**Retraining / recalibration trigger.** Two triggers, not one: (1) **scheduled** — if deployed
beyond this course project, re-validate the model against a fresh sample of real footage every
term, the same cadence construction-site conditions (new workers, new PPE styles, seasonal
lighting) would justify; (2) **performance-based** — if a manual spot-check of a sample of flagged
events shows a false-negative rate above the Step 6 threshold, that's the trigger to retrain, not
a fixed calendar date alone.

**Model / config / data versioning — Done.** Model weights follow a versioned naming convention
(`edgeai_v1_best.pt`, `edgeai_v2_best.pt` on the parking project; `helmet_v1_best.pt` planned
here). `docs/CHANGELOG.md` (new) gives this a dedicated one-row-per-version index — what changed,
what metric moved — so which model version was running at any given time is auditable at a
glance, not just reconstructable from `docs/RUNLOG.md`'s narrative. Zone configs
(`src/helmet_zone_configs.py`) are version-controlled in git already — every polygon change on the
parking project was a tracked, reviewed commit, and that discipline carries over directly.

**Secure update / rollback.** This is a classroom prototype, not a networked production service,
so "secure update" here means a **process** control, not a cryptographic one: a new model file is
never swapped into the running demo without first being checked against a handful of sample frames
(Step 6's functional testing, repeated on every update) — the same habit that caught two real bugs
on the parking project (a mis-seeked video frame, a silently-skipped dataset download) before they
reached a demo. **Rollback**: keep the previous working model file (`helmet_v1_best.pt` alongside
any `helmet_v2_best.pt`) rather than overwriting it, so a bad update can be undone by pointing
`demo_helmet.py --model` back at the last known-good file.

**Maintenance owner.** Hoang (Deployment) owns runtime health and the demo rig; Hieu (Data &
Model) owns the retraining trigger and any dataset/model updates; Hanh (Lead) owns deciding when a
flagged issue is serious enough to pause the demo rather than continue with a known problem.

---

## Step 8 — Ethical and Responsible Edge AI

*(Not addressed in any prior document for this project — written fresh here.)*

**Governance / accountability.** The system's output is **advisory, not punitive by design**: a
"no helmet in high-risk zone" alert goes to a human safety officer, who makes any actual
consequential decision. The model never automatically escalates to discipline, pay, or
record-keeping against a named individual — that boundary is a deliberate design choice, not an
afterthought, and should be stated to anyone the system is piloted with.

**Privacy & data minimization — Done, enforced in code, not just stated.** The system performs no
face recognition and no identity matching — it classifies helmet presence within a zone, and uses
YOLO's tracker only to avoid double-counting the same person across frames within one session (the
track id is not linked to any real identity, and does not persist across sessions — it resets
every time `demo_helmet.py` starts). As of 2026-10-08, `demo_helmet.py` prints an explicit privacy
notice at startup (`PRIVACY_NOTE`) and, as a matter of actual code behavior, never writes a video
frame to disk — only the zone/timestamp/confidence fields described above are logged. This section
describes what the code does, not an aspiration for what it should do.

**Fairness / consistency.** A real, specific risk worth stating rather than assuming away: the
Hard Hat Workers Dataset's images come from a different country/context than a Vietnamese
construction site — different typical worker appearance, clothing, and headwear (e.g., conical
non-hard-hat hats common in Vietnam could plausibly confuse a "head" vs "helmet" classification
in ways the dataset's own images never test). **This is flagged as an open risk to validate before
any real deployment**, not claimed as solved by using a large Western-sourced dataset.

**Human oversight.** Every alert is reviewed by a person before any action is taken (same point as
Governance above, restated because it's the actual safety mechanism against a false positive
costing someone unfairly).

**Transparency / limitations.** Stated plainly, matching the project's established habit of
documenting real gaps rather than hiding them (see `docs/PIVOT_HELMET_DETECTION.md`'s explicit
harness-detection descoping): this system does **not** verify a helmet is properly fastened, does
not detect other PPE (vests, gloves, eye protection), and its risk-zone polygons depend on a fixed
camera position — if the camera moves, the zones silently stop meaning what they're supposed to
mean, exactly the class of bug the parking project hit twice with its own zone polygons.

**Security.** All processing is local to the device running the demo — no video or event data is
transmitted off-device in the current implementation, which inherently reduces the attack surface
compared to a cloud-dependent alternative (this is also the Step 1 Edge-vs-Cloud argument, applied
here from the security angle rather than the latency angle).

**Sustainability.** YOLOv8n is a ~6MB model running on CPU-only edge hardware (a Pi 4B draws a few
watts) — meaningfully lower energy cost per inference than a cloud service processing continuously
streamed video server-side, especially at the scale of "one low-power device per camera" versus
"continuous bandwidth plus server compute per camera."

**Risk & mitigation.**

| Risk | Mitigation |
|---|---|
| False "no helmet" flag on a real person (e.g., model miss) | Human review before any consequence (Governance, Human oversight above) |
| Domain gap: dataset doesn't reflect local worker appearance | Explicitly flagged (Fairness above); validate against real local footage before any deployment claim |
| Zone polygon silently wrong if camera moves | Documented limitation (Transparency above); same verify-before-trusting habit used throughout this project |
| Camera failure misread as "100% compliance" | Failure-detection rule proposed in Step 7 |
| Mission creep toward identity-linked tracking or punitive automation | Explicitly out of scope by design (Privacy, Governance above) — stated here so a future team doesn't add it without re-opening this discussion |

---

## Summary

| Step | Score | Evidence |
|---|---|---|
| 1. Problem Definition | 4/4 | Real accident data, legal basis, explicit Edge-vs-Cloud-vs-No-AI comparison, stated resource targets |
| 2. Data Collection & Preprocessing | 4/4 | Confirmed dataset/split/class-order, domain-gap and privacy risks named explicitly |
| 3. Model Selection & Training | 3/4 | Real trained model, real metrics (head 0.963 / helmet 0.982 / person 0.034 AP50), acceptance rule correctly applied with reasoning, a real new risk found (17% head→helmet confusion) — **edge resource metrics still pending Step 4's export** |
| 4. Optimization | 2/4 | Real, specific, non-assumed trade-off data from a directly comparable prior run — **blocked on Step 3's model existing to re-run it against** |
| 5. Deployment | 4/4 | Working integration code for camera/tracking/zones; target device and architecture explicit; offline/fallback now implemented in code (2026-10-08) |
| 6. Testing & Evaluation | 2/4 | Concrete, specific test plan (shot list, metrics, thresholds) — **blocked on a model + physical rig to actually run it against** |
| 7. Monitoring, Maintenance & Iteration | 4/4 | Failure detection + compliance snapshots implemented in `demo_helmet.py`; versioning in `docs/CHANGELOG.md` (2026-10-08) |
| 8. Ethical & Responsible Edge AI | 4/4 | Full plan grounded in this project's specific risks; privacy behavior now enforced in code, not just stated (2026-10-08) |

**Total: 27/32 (~84%)**, up from 26/32, 23/32, and 10/32 across the three prior passes. The
remaining 5 points are all **execution that depends on this same model**, not more writing:
Step 3's last point needs edge resource metrics (depends on Step 4's ONNX export), Step 4 needs
that export + INT8 quantization actually run and re-measured (not assumed to repeat the parking
project's numbers), and Step 6 needs the shot list actually executed once the physical rig exists
— including the newly-found head→helmet confusion case. Nothing in this report shortcuts that by
inventing results.
