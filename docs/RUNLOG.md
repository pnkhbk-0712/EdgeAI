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

**Still blocked on the team, not on code:** pilot-site filming, and any physical edge hardware
(Raspberry Pi / Jetson) to deploy to or benchmark on. (UA-DETRAC registration is no longer
blocked -- see docs/UA_DETRAC_DOWNLOAD.md.)

---

## 2026-09-15 — Zone/dwell answers locked in (Hieu), re-verified end-to-end

Hieu filled out `docs/ZONE_LABEL_DEFINITIONS.md`. Real changes made to match:

- **Dwell threshold raised 2s -> 60s** (`DWELL_SECONDS` in `demo.py` and `demo_synthetic.py`).
  A rolling 2-second stop was too easy to confuse with normal traffic; 60s is a real "parked."
- **Zone schedule support added** to `src/zone.py` (`active_hours`, `active_days`) per the
  Design Note, wired into `demo.py` — currently `None`/`None` (restricted at all times/days)
  since the pilot site hasn't been surveyed yet; update once it is.
- Regenerated the synthetic test clip with a 65s held segment (`make_synthetic_clip.py`,
  new script) and re-ran `demo_synthetic.py` end-to-end: **violation correctly fired at
  frame 754 (60.3s)** — confirms the real pipeline still works at the new threshold, not
  just the unit tests.
- Added 3 new unit tests (`tests/test_zone.py`, now 8/8 passing) covering the schedule:
  outside active hours never violates, correct hours/day-parity still violates, wrong day
  parity never violates.
- **Known unresolved limitation, documented not silently accepted:** Hieu specified a car
  stopped at a red light beside the zone should NOT count as a violation, but the dwell-time
  rule cannot distinguish that from real parking on its own — a long red-light cycle can
  exceed 60s too. The real mitigation is zone-drawing discipline (keep the polygon tight to
  the restricted curb, off the traffic lane), not a code fix. Flagged in `zone.py`'s docstring
  and here for Hoang's attention when he draws the real zone at deployment.

---

## 2026-09-18 — UA-DETRAC converted to YOLO format (Hieu's Build-phase head start)

Wrote and ran `src/convert_ua_detrac.py` for real against the actual downloaded XML files
(structure confirmed by inspection first, not assumed). Real results:

- **8,232 train images / 5,625 val images**, sampled 1-in-10 frames per sequence (not all
  140,131 -- consecutive frames are near-duplicates; raise `STRIDE` in the script for more)
- **128,401 total boxes** across car/bus/truck (van mapped to truck); motorcycle reserved as
  class id 1 with 0 examples -- confirmed UA-DETRAC has no motorcycle class, matches the
  report's own known limitation
- Visually verified: drew converted YOLO boxes back onto a sample image
  (`MVI_20012_00331`) -- bus, cars, and a van(->truck) all correctly boxed, confirming the
  coordinate conversion is right, not just "ran without crashing"
- Output at `data/ua_detrac_yolo/` (989MB, gitignored -- regenerate with the script rather
  than committing it) with a ready `data.yaml` for `model.train(data=...)`

This was fully unblocked work done while the pilot-site decision (Hanh, critical path) was
still pending -- doesn't require any real footage.

---

## 2026-09-18 (later) — Motorcycle data added (was 0 examples, now 699)

UA-DETRAC has no motorcycle class (confirmed earlier). Found and merged a real source:
Kaggle **"Vehicle Dataset for YOLO"** (nadinpethiyagoda/vehicle-dataset-for-yolo), 3000 images,
6 classes already in YOLO format.

- Downloaded via `kagglehub` -- **this time with `KAGGLEHUB_CACHE` set correctly from the
  start** (caught myself defaulting to C: again on the first `dataset_download()` call before
  the env var was set for that command; moved it to D: immediately and fixed the script to
  always set the env var itself rather than relying on the shell).
- Verified before use, not after: drew boxes from 3 random motorbike-labeled images back onto
  their photos. Real, correct boxes. Mixed domain -- some studio/for-sale-listing photos, some
  genuine street scenes (e.g. a Sri Lanka street photo with a motorbike parked behind a
  tuk-tuk).
- **Known limitation, stated plainly:** most of these are close-up single-vehicle photos, not
  the elevated traffic-camera angle the pilot site will use. This closes the "zero examples"
  gap, not the "right camera domain" gap -- real pilot-site motorcycle footage will still be
  a better match once it exists.
- `src/add_motorbike_data.py` merges only the motorbike boxes (dropping this source's car/bus/
  truck/van/threewheel classes -- UA-DETRAC already covers those with a more consistent,
  matching domain) into `data/ua_detrac_yolo/`, remapped to class id 1.
- **Result: 538 images, 699 motorcycle boxes added** (373 train / 165 val). `data.yaml` needed
  no changes -- motorcycle was already reserved as class id 1 with 0 examples.

Dataset now has all 4 target classes with real examples: car, motorcycle, bus, truck.

---

## 2026-09-18 (later still) — Overfit-risk check on the dataset (not a model yet)

No real model has been trained on this dataset yet (only the coco8 smoke-test on unrelated
generic images), so there is no train/val loss curve to check for classic overfitting. What
was checked instead, for real:

- **Sequence-level leakage**: 0 overlapping UA-DETRAC sequences between train (60) and val (40)
  XML sets -- confirmed by directly diffing the sequence name lists, not assumed from the
  folder split.
- **Exact image duplication**: hashed all 8,605 train + 5,790 val images (MD5) -- **0 exact
  duplicates** between the splits.
- **Class imbalance (the real finding)**: car:motorcycle box ratio was **154:1**
  (107,889 vs 699 boxes) in the merged dataset. Left as-is, this is a realistic path to a
  model that looks fine on overall mAP while effectively ignoring motorcycles -- the loss is
  dominated by car regardless of motorcycle performance.

**Mitigation applied:** `src/oversample_motorcycle.py` duplicates each motorcycle-containing
TRAIN image+label 5x (373 -> 1,865 motorcycle instances contributing to loss), bringing the
training-time ratio to **21.7:1**. VAL is untouched, so evaluation still reflects the true,
unbalanced real-world distribution -- oversampling changes what the optimizer sees, not how
the model gets judged. When Hieu trains, watch **per-class mAP for motorcycle specifically**,
not just overall mAP -- overall mAP can still look fine while motorcycle recall is poor.

---

## 2026-09-18 (later still) — Full data integrity audit, 1 real bug found and fixed

User asked to re-check the data for problems. Ran a full audit (`os.listdir` + `cv2.imread` +
label parsing) across all 15,874 image+label pairs, not just the earlier leakage/duplicate/
imbalance checks:

| Check | Result |
|---|---|
| Corrupt/unreadable images | **5 found** (see below) |
| Orphan images (no label file) | 0 |
| Orphan labels (no image file) | 0 |
| Malformed label lines (wrong field count / non-numeric) | 0 |
| Out-of-range coordinates (outside 0-1) | 0 |
| Zero-size boxes | 0 |
| Invalid class id | 0 |
| Empty label files (0 boxes -- background frames, not an error) | 8 |

**The 5 corrupt images were all the same root cause:** one source file from the Kaggle
motorbike dataset, `pexels-martin-péchy-2078248.jpg`, had its filename's non-ASCII "é"
mis-decoded into mojibake during zip extraction on this Windows machine (a known class of
bug: many zip tools assume the wrong codepage for filenames on Windows). OpenCV's `imread`
additionally can't reliably open non-ASCII paths on Windows even when Python can list them.
The oversampling step then multiplied this single bad file into 5 corrupt entries (the
original + 4 duplicates).

**Fix:** removed all 5 (the original file was one of only 538 motorcycle source images, not
worth building a Unicode-path workaround for one file). Re-ran the full audit after removal:
**0 issues of every kind** except the same harmless 8 empty-label background frames.

---

## 2026-09-18 (overnight, while Colab training ran) — More real motorcycle data found

User asked to crawl more motorcycle data during the wait. Checked 2 more candidates, real
verdicts on both:

**Rejected: Kaggle "Detect Person on motorbike or scooter" (savanagrawal).** Drew a sample
box back onto its image before merging (same discipline as every other source this project)
and caught a real problem: its single class, `person_bike`, boxes the ENTIRE rider + bike
together (head to feet), not the vehicle alone. Merging this as our `motorcycle` class would
have taught the model an incorrect, oversized box definition inconsistent with every other
source. Not used.

**Merged: Roboflow "car-and-motorcycle-detection-with-kaggle-dataset" (aliff-haikal-ssf6d,
v14, CC BY 4.0).** Verified a sample box first: vehicle-only, no rider -- consistent with the
existing sources. `src/add_motorbike_data_v2.py` merges only its motorcycle class (drops its
Car class, same reasoning as before).

**Real methodology catch worth recording:** a first pass counted only "33 motorcycle boxes" in
this source using `cat *.txt | awk '{print $1}' | sort | uniq -c` in bash. That number was
**wrong** -- many of the source's label files lack a trailing newline, so `cat`ing hundreds of
them together silently merges the last line of one file with the first line of the next,
undercounting. Re-checked with Python (`.read_text().splitlines()`, which correctly isolates
each file's own lines regardless of trailing newline) and got the real number: **601 boxes**,
not 33. Told the user the wrong number first, then caught and corrected it before it shaped
any real decision -- recorded here so nobody trusts a bash `cat`-based instance count on this
dataset again.

**Result:** +601 real motorcycle boxes (573 images). Re-ran `oversample_motorcycle.py` (fixed
to skip already-`_dup`-suffixed files, so it's safe to re-run after adding a new source instead
of oversampling prior duplicates again) to top up the new images too.

**Train-split car:motorcycle ratio: 154:1 (original) -> 21.7:1 (first oversample) ->
12.7:1 (now)**. Re-audited after every step: 0 corrupt/orphan/malformed/out-of-range/bad-class
issues throughout.

**Important timing note:** the Colab training run already in progress started BEFORE this
data addition -- it's training on the earlier (21.7:1) dataset snapshot. This improvement
applies to the *next* training run, not the one currently executing.
