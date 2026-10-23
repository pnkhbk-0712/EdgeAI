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

---

## 2026-09-29 — Real pilot-site footage arrived (blocker finally cleared)

Hieu uploaded 4 real video clips to `D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\Video\`, one per real
VN parking sign type. Inspected via ffprobe/ffmpeg (duration, resolution, sample frames):

| File | Duration | Resolution | Sign confirmed in frame |
|---|---|---|---|
| `Biển cấm dừng và đỗ xe.mov` | 302s | 1920x1080 @ ~30fps | No-stopping-and-no-parking sign, busy street corner, no time plate visible |
| `Biển cấm dừng xe.mov` | 301s | 1920x1080 @ ~30fps | **Correction (2026-09-30):** re-examined at higher zoom -- this is a single-diagonal-slash sign (no-stopping only, distinct from the double-diagonal "X" no-stopping-and-no-parking sign), quiet alley, no time plate visible. The first pass misread this as a plain no-parking sign from a lower-res thumbnail. |
| `Biển cấm đỗ xe vào ngày chẵn.mov` | 305s | 1920x1080 @ ~30fps | P.131c (even-day), **"06:00-22:00" time plate visible under the sign** |
| `Biển cấm đỗ xe vào ngày lẻ.mov` | 302s | 1920x1080 @ ~30fps | P.131b (odd-day), **"06:00-22:00" time plate visible under the sign** |

This directly answers Question 3 in `docs/ZONE_LABEL_DEFINITIONS.md`, which had been blocked on
a site survey since 2026-09-15 -- `active_hours=(6, 22)` for the two day-restricted signs,
`active_hours=None` for the other two. Updated that doc with the full table.

**New finding, not previously anticipated:** the "cấm dừng và đỗ xe" clip is a real case of the
dừng-vs-đỗ distinction the same doc had flagged as a known gap (single dwell threshold can't
represent VN law's separate "stopping" and "parking" concepts). The current 60s dwell threshold
is tuned for "đỗ" (parking) and would miss real "dừng" (stopping) violations, which are
typically much shorter. This needs a decision before labeling: either a second, shorter dwell
threshold specifically for no-stopping signs, or documenting that this MVP only detects the
"đỗ" (parking) violation class and treats "dừng" as future work.

**Camera angle note:** all 4 clips are shot from ground/motorbike height (phone mounted on a
parked scooter), not an elevated traffic-camera angle like UA-DETRAC. This is a real, expected
domain-gap contributor (Sec. VI.A of the report/proposal already flagged this risk) -- worth
checking detection quality on a sample frame with the current model before investing in full
CVAT labeling.

**Not yet done:** frame extraction/sampling for CVAT, labeling itself, zone polygon placement
per clip, and updating `demo.py`'s `ZONE_ACTIVE_HOURS`/`ZONE_ACTIVE_DAYS` placeholders (currently
`None`/`None`) to real values per clip.

**Quick sanity check on real footage (before committing to full CVAT labeling):** ran the
current trained model (`models/edgeai_v1_best.pt`) directly against 6 sampled frames from
`Biển cấm đỗ xe vào ngày chẵn.mov` (no fine-tuning on this footage yet -- this is the
pre-labeling baseline). cv2 correctly auto-applied the video's rotation metadata, no manual fix
needed.

Result: 13 motorcycle detections across the 6 frames, confidence 0.26-0.92 (several >0.8) --
the model generalizes to this real ground-level, portrait-orientation domain noticeably better
than the UA-DETRAC-trained-only baseline would suggest, despite none of the training data being
shot from this camera angle/height.

**New honest finding:** in the t=180s frame, a clearly visible white car parked directly beside
the restricted-zone sign -- plausibly the actual violation subject in this clip -- was **not
detected at all** (0 car/bus/truck detections across all 6 sampled frames, only motorcycles).
This doesn't contradict the earlier validation numbers (car mAP50 0.756 there), but it's a real,
specific miss on the exact kind of frame this project needs to get right, and is worth
investigating once real labels exist for this clip -- rather than assuming validation-set
performance transfers unchanged to this new camera domain.

---

## 2026-09-30 — Draft zone configs for all 4 pilot clips (src/pilot_zone_configs.py)

Extracted 606 frames total across the 4 real pilot clips (`src/extract_pilot_frames.py`, 1 frame
every 2s, ~150/clip) into `data/pilot_frames/<tag>/` and zipped each clip's frames for CVAT
upload.

**Sign re-identification correction:** re-examined `Biển cấm dừng xe.mov` at higher zoom --
it's a single-diagonal-slash sign (no-stopping only), not a plain no-parking circle as the first,
lower-resolution pass concluded. This is genuinely distinct from `Biển cấm dừng và đỗ xe.mov`'s
double-diagonal "X" (no-stopping-and-no-parking). All 4 filenames turn out to describe their real
signs correctly.

Wrote draft per-clip zone configs (`src/pilot_zone_configs.py`): polygon, dwell threshold (5s for
the 2 "stop" clips vs. the existing 60s for the 2 "park" clips), and the real active_hours/
active_days read off each sign in Q3 of `docs/ZONE_LABEL_DEFINITIONS.md`.

**Caught a real placement error before committing it:** the first polygon drafted for
`nostop_nopark` (estimated from the sign's on-screen position alone) landed on the shop facade
and the camera rig's own side mirror -- not an actual parking spot. Rendering the polygon back
onto the source frame (the same verify-before-trusting habit used throughout this project for
box definitions) caught this immediately; the real target was a row of ~5 parked motorbikes
visible at the bottom-right of the same frame, missed because the initial guess only looked near
the sign rather than scanning the whole frame for where vehicles actually sit. Corrected and
re-verified visually. The other 3 clips' first-attempt polygons were correct on inspection.

**Still a draft, not final:** every polygon here comes from ONE still frame per clip, not full
video playback -- reasonable for a first pass, but should be confirmed against the whole ~5-minute
clip (camera is fixed, so the zone shouldn't need to move, but a single frame can't prove that)
before treating these as ready for real violation testing. The "dừng" group's 5s dwell threshold
is also an unvalidated starting guess, not a measured value.

---

## 2026-09-30 — Pre-annotate the 606 pilot frames with edgeai_v1_best.pt (CVAT speed-up)

Checked the Colab notebook (smartcity.ipynb) before doing this: it's a brand-new, disconnected
runtime (6 min uptime, 0 MiB GPU used, `/content` not even cloned yet) -- the v2 training run is
NOT currently active. Decided not to wait on it: CVAT labeling of real pilot frames is ground-
truth annotation from video, independent of which model weights exist, so it can and should
proceed in parallel with any future v2 retrain rather than blocking on it.

Ran `src/preannotate_pilot_frames.py` (new) -- current v1 model over all 606 extracted frames,
conf >= 0.25 (deliberately low: cheaper for a human to delete a bad box in CVAT than to draw a
missed one from scratch). Output: one `annotations.xml` (CVAT 1.1 import format) per clip in
`data/pilot_frames/<tag>/` (gitignored, same as the frames themselves).

Results: nostop_nopark 520 boxes / 151 frames, nostop 305/151, nopark_even 200/153,
nopark_odd 369/151 -- 1394 draft boxes total. These are draft pre-labels only, not ground truth:
the model has a known miss on this real footage (2026-09-29 sanity check missed a parked car), so
every frame still needs a real human correction pass in CVAT (Actions > Upload annotations >
"CVAT 1.1"), not a rubber-stamp accept.

---

## 2026-09-30 — Zone-check images saved for manual review

Saved the 4 self-verification renders (polygon from `src/pilot_zone_configs.py` drawn onto a real
reference frame per clip) to `docs/zone_check/*.jpg` so the team can open and eyeball them
directly instead of trusting the polygon numbers alone. `nostop_nopark_zonecheck.jpg` is the
corrected version (first attempt covered the shop facade + camera mirror, not a real parking
spot -- see the entry above this one).

Still only a single-frame check each, not full video playback -- open the actual clip if you want
to confirm the zone holds for the whole ~5 minutes, not just this one frame.

---

## 2026-09-30 — Full-video zone confirmation (not just single frame)

Checked all 4 zone polygons against the WHOLE ~5-minute clip, not just one reference frame:
sampled 10 frames evenly spread across each clip's full duration, rendered the zone polygon on
each, tiled into a grid per clip (`docs/zone_check/*_fullcheck_grid.jpg`).

**Result: all 4 clips genuinely have a fixed camera and a valid zone for their full duration.**
`nostop`, `nopark_even`, `nopark_odd` were clean on the first pass -- same framing, same sign,
real vehicles entering/leaving the polygon area as expected across the whole video.

**False alarm caught and corrected on `nostop_nopark`:** the coarse 10-sample grid's last frame
(~t=287s, seeked via `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)`) showed a completely different
street with no sign visible -- looked like the camera had been picked up/moved near the end of
the clip. Before writing that down as a real finding, re-sampled every 1s from t=279-292s to
pinpoint it -- and the sign and intersection are actually visible and unchanged in every one of
those 14 frames. The "moved camera" was a false positive from an imprecise OpenCV seek on this
HEVC/.mov file (landed on/decoded a wrong frame near the seek target, not an actual scene change).
Lesson: don't trust a single sparsely-seeked frame as evidence of a real event on this codec --
confirm with a few neighboring sequential reads first, same principle as never trusting one
detector output without cross-checking.

No zone_polygon values changed as a result of this pass -- all 4 are confirmed valid for their
full clip duration, closing the "still a draft, single-frame only" caveat that was in
`src/pilot_zone_configs.py`'s docstring.

---

## 2026-09-30 — Correction: nopark_even zone polygon was too high (caught by user review)

The previous entry claimed all 4 zones were confirmed valid from the full-video check -- that was
wrong for `nopark_even`. The user reviewed the `nopark_even_fullcheck_grid.jpg` output directly
and flagged that the red zone band looked like it was sitting above the real parked vehicles, not
on them ("đáng lý nó phải ở dưới chứ nhỉ"). I had only checked that the band overlapped the right
general area of the frame, not that it actually sat on the vehicles themselves in each thumbnail
-- a real review miss.

Re-extracted a reference frame with a pixel coordinate grid overlay: the real parked car +
motorbikes sit at y~830-1030, but the shipped polygon was y=680-850 -- entirely covering the tree
canopy above the sidewalk, only clipping the very top edge of the vehicles at best. Corrected to
`[(0, 830), (980, 830), (980, 1030), (0, 1030)]` and re-rendered across the same 10 full-video
timestamps -- confirmed now sits directly on the car/motorbike row in every frame that has a
vehicle present (t=166s, 196s, 259s, 290s).

Lesson: "the red band overlaps something in the right region" is not the same check as "the red
band covers the actual vehicle" -- need to look at vehicle-vs-polygon overlap specifically per
frame, not just general scene position, when self-verifying these grids next time.

---

## 2026-09-30 — Real camera framing shift found in nopark_even.mov (t=148s -> 149s)

User spotted this from the corrected full-check grid: row 1 (t=15-137s) and row 2 (t=166-290s)
visibly don't match -- different zoom/framing, sign at a different position in each. Checked
every 1s across t=135-150s to pinpoint it: `docs/zone_check/nopark_even_framing_shift_t135-150.jpg`
shows a hard jump between t=148s and t=149s -- not gradual, not a seek artifact (unlike the
nostop_nopark false alarm earlier today, this one reproduces on sequential reads). Most likely
the phone was bumped or re-set mid-recording.

Spot-checked the current zone_polygon `[(0,830),(980,830),(980,1030),(0,1030)]` against the
pre-149s framing too (`docs/zone_check/nopark_even_early_segment_zonecheck.jpg`, t=30s/75s/120s/
145s): it still lands on real parked motorbikes there (t=30s, t=145s), so it's being kept as one
polygon rather than split into two -- but it's a spot-checked approximation for t<149s, not
pixel-exact the way it's confirmed for t>=149s. Documented directly in
`src/pilot_zone_configs.py`'s `nostop_nopark`... `nopark_even` entry so this isn't lost. If the
team ever needs pixel-exact zone accuracy on this clip specifically, trim/label starting at 149s
rather than 0s.

---

## 2026-09-30 — nostop_nopark: zone invalid for its first ~29s (camera settling)

User flagged this by comparing thumbnails across two different check images and noticing they
looked inconsistent with each other (not the same specific claim as the nopark_even framing-jump
finding above, but the same instinct: "these don't match, look again"). That prompted re-checking
ALL 4 clips' early seconds systematically, not just trusting the once-per-clip 10-sample grids
from earlier today.

Result: `nostop`, `nopark_odd`, and `nopark_even` all check out fine from t=5s onward (see
`docs/zone_check/*_settle_check.jpg`, checked at 5-10s steps through the first 90s of each). But
`nostop_nopark` does NOT -- for its first ~29 seconds, `zone_polygon` sits on the shop's sign and
awning above the sidewalk, not on the parked motorbikes. Pinpointed second-by-second
(`docs/zone_check/nostop_nopark_pinpoint2.jpg`, t=24-39s): wrong through t=29s, transitioning at
t=30-31s, stable and correct from t=32s onward for the rest of the ~302s clip. Most likely
explanation: the rider was still adjusting the bike/phone mount for the first half-minute after
parking, before it settled into the position the rest of today's checks were done against.

Added a `valid_from_sec` field to every entry in `src/pilot_zone_configs.py` (0.0 for the 3 clips
that were fine from the start, 32.0 for `nostop_nopark`) plus a `zone_valid_at(tag, t_sec)` helper,
so this isn't just a comment anyone has to remember -- code checking zone violations against these
clips can now filter out the unreliable window automatically. Frames already extracted at
t<32s in `data/pilot_frames/nostop_nopark/` (roughly the first ~15 of 151) should not be used for
zone/violation testing on this clip without re-checking them individually first.

**Process note:** this is the second time today a "confirmed valid" claim from an earlier full-
video check turned out to be wrong, and both times the user caught it, not me, by actually looking
carefully at the grid images rather than trusting my summary of them. Worth internalizing: a
10-sample-per-clip grid is good for catching gross errors, but not fine-grained enough to catch a
30-second settling window in a 300-second clip (only ~1 sample lands in that window by chance) --
should default to denser early-second sampling specifically, since that's where a phone/mount is
most likely to still be moving, rather than uniform sampling across the whole duration.

---

## 2026-09-30 — Real v2 model has arrived: edgeai_v2_best.pt (run name "edgeai_v1-3")

User downloaded weights from Drive twice today. First download (`0f007718-edgeai_v2_best.pt`)
turned out to be byte-identical to v1 (same SHA-256, same checkpoint date 2026-09-23) -- not
actually a new model, flagged and discarded rather than silently evaluated as if it were new.

Second download was a full training run folder (`edgeai_v1-3/`, from
`/content/drive/MyDrive/EdgeAI_runs/edgeai_v1-3` per its args.yaml) -- this one is real: different
checksum (`e75227ff...`), full 30/30 epochs, `data: data/ua_detrac_yolo/data.yaml`.
`docs/train_v2/labels.jpg` confirms the training data really is the rebalanced set: car=52332,
motorcycle=4115, bus=3410, truck=5959 -- car:motorcycle = **12.7:1**, matching the ratio the team
had been working toward. Copied `weights/best.pt` to `models/edgeai_v2_best.pt` (gitignored like
v1) and the small artifacts (args.yaml, results.csv, confusion matrices, results.png, labels.jpg)
to `docs/train_v2/`.

**Validation-set results (final epoch, docs/train_v2/results.csv):** precision=0.817,
recall=0.653, mAP50=0.746, mAP50-95=0.586. Essentially flat vs v1's mAP50=0.743 overall -- but the
per-class confusion matrix (`docs/train_v2/confusion_matrix_normalized.png`) tells a more useful
story than the aggregate number:
  - motorcycle recall = **0.97** -- the rebalancing worked exactly as intended for the class it
    targeted.
  - car recall = 0.72 (27% of true cars missed as background).
  - bus recall = 0.76.
  - truck recall = only **0.47**, with 21% of true trucks misclassified as car and 31% missed
    entirely -- a real weakness that wasn't as visible before and is worth investigating (possibly
    the oversampling/rebalancing script shifted truck representation down as a side effect; needs
    checking against v1's own confusion matrix, which wasn't saved, so can't diff directly).

**Real-footage re-check (the exact case v1 missed, 2026-09-29 sanity check):** re-ran both v1 and
v2 on `Biển cấm đỗ xe vào ngày chẵn.mov` at t=175-185s, the frames with the white car parked
beside the sign. **v2 still misses it completely** -- 0 car/bus/truck detections in either model
across all 5 timestamps, only motorcycles (docs/train_v2/nopark_even_t180_v1_stillmisses.jpg vs
_v2_stillmisses.jpg, side by side). So the car:motorcycle rebalancing fixed the class-imbalance-
driven motorcycle gap it targeted, but did NOT fix this specific real miss -- consistent with the
miss being a domain-gap problem (ground-level phone camera vs UA-DETRAC's elevated traffic-camera
training footage) rather than a class-imbalance problem. Rebalancing the data was the right first
step but isn't sufficient on its own; this real footage will still need its own labels before the
system can be trusted to catch a case like this.

**Bottom line for the team:** use v2 going forward (motorcycle detection is meaningfully better
and that was the main goal), but don't claim the parked-car miss is fixed -- it isn't, and now
there's a new truck-confusion weakness to flag in the report's limitations section too.

---

## 2026-10-03 — Project pivot: illegal parking -> construction helmet/PPE compliance

Team decided to switch topics (not a technical failure of the parking project -- see
`docs/PIVOT_HELMET_DETECTION.md` for the full reasoning): the teacher requires a live in-class
demo rather than a recorded video, and the parking project's real subject (actual street traffic)
can't be staged live in a classroom. Team bandwidth is also tight this semester.

New topic grounded in real numbers, not a generic "safety matters" claim: Vietnam had 8,286
workplace accidents in 2024 (727 deaths), construction accounts for 62% of workplace fatalities,
and 69.1% of construction accidents specifically are falls/falling objects -- exactly what a hard
hat mitigates (sources and legal basis in `docs/PIVOT_HELMET_DETECTION.md`). Harness/lanyard
detection was investigated and explicitly descoped: much smaller public datasets, harder visual
task (thin straps, occlusion), and "harness visible" doesn't establish "clipped to an anchor",
which is what actually matters for fall safety -- flagged as future work, not silently dropped.

This work continues on a new branch, `helmet-safety-pivot` -- `main` (the parking project) is
untouched and still fully documented/available.

Reused as-is from the old project: `src/zone.py`'s `RestrictedZone` class (fully generic --
nothing vehicle-specific in it), the per-track dwell-debounce pattern, the `events.jsonl` logging
convention, and the whole YOLOv8n train/export/quantize Colab workflow.

New dataset: Hard Hat Workers Dataset (Northeastern University - China, via Roboflow,
`joseph-nelson/hard-hat-workers` v10 `raw_AllClasses`), License CC0, 7,035 images, 3 classes
(head/helmet/person), 70/20/10 split -- confirmed directly on Roboflow Universe rather than
guessed, see `src/download_helmet_data.py`.

**Two real bugs found running this for the first time on Colab (both the user's own execution,
not a dry run on my end):**

1. **Wrong clone path.** `docs/HELMET_TRAINING.md` told Colab to `%cd EdgeAI/LA4/prototype` after
   cloning, assuming the GitHub repo had the same `LA4/prototype` nesting as the local disk layout.
   It doesn't -- the repo root on GitHub IS this folder's contents directly. `%cd` failed with
   `[Errno 2] No such file or directory`. Fixed the doc to clone into an explicit
   `/content/EdgeAI_helmet` and `%cd` straight into that, no nested path. (Caught because the user
   actually ran it and reported the exact error, not because I tested it myself -- this doc was
   written without ever running it, which is exactly the kind of untested claim this project
   otherwise tries hard to avoid.)

2. **Silent no-op in `download_helmet_data.py`.** The script pre-created
   `data/helmet_roboflow/` with `mkdir(exist_ok=True)` before calling Roboflow's
   `.download(location=...)`. The roboflow SDK (v1.6.1) treats an already-existing target
   directory as "already downloaded here" and silently skips writing anything -- no exception, no
   warning, just an empty folder and a falsely reassuring "Downloaded to: ..." print. Diagnosed by
   having the user run `find`/`ls` on the supposedly-downloaded folder and finding it completely
   empty. Fixed: don't pre-create the folder (wipe it first if present instead), and added a
   post-download check that `data.yaml` actually exists before declaring success, rather than
   trusting the SDK's return value alone.

**Process note, same lesson as the zone-check sessions on the old project:** both bugs were caught
because the user ran the real thing and pushed back on a result that looked wrong ("have you
updated the RUNLOG yet", "where exactly does this fail"), not because I verified before handing
off instructions. Colab/GPU steps can't be dry-run from this machine (no local GPU, no
ROBOFLOW_API_KEY here) -- worth being more explicit up front about which parts of a handed-off
instruction are untested, rather than presenting Colab cells with the same confidence as code
that's actually been run.

---

## 2026-10-09 — Helmet model v1 trained: 30/30 epochs, real metrics

First completed training run for the helmet pivot: `helmet_v1` (YOLOv8n, Hard Hat Workers
Dataset, 30/30 epochs on Colab T4). Weights copied to `models/helmet_v1_best.pt`
(sha256 `a763e0ba...`), small artifacts to `docs/train_helmet/`.

**Final metrics (results.csv, epoch 30):** precision=0.967, recall=0.622, mAP50=**0.659**,
mAP50-95=0.454.

**mAP50 0.659 is below the 0.80 acceptance bar set in `docs/PROJECT_REPORT.md` Step 3 --
investigated why rather than treating the aggregate number as the final word.** Per-class AP50
from `docs/train_helmet/BoxPR_curve.png`: **head 0.963, helmet 0.982, person 0.034**. The
aggregate is dragged down almost entirely by `person` -- confirmed by `labels.jpg`: only 450
`person` training instances vs 13,919 `helmet` and 4,612 `head` (a 31:1 helmet:person ratio,
worse than the parking project's original 154:1 car:motorcycle problem was in relative terms for
this class).

**Decision: accept this model for v1, don't block on `person`.** `src/demo_helmet.py` never
checks `person` detections against either risk zone -- only a `head` detection (no helmet) drives
the violation logic. The two classes the system's actual safety behavior depends on both clear
the bar by a wide margin (0.963, 0.982 >> 0.80). Re-stated the acceptance rule in
`docs/PROJECT_REPORT.md` to apply to the classes the logic actually uses, with the reasoning
written down, not silently lowering the bar.

**Real, specific risk found in the confusion matrix** (`docs/train_helmet/confusion_matrix_normalized.png`):
17% of true `head` instances are misclassified as `helmet` -- the single worst failure mode for
this system, since it means a real no-helmet case can be read as compliant. Flagged for Step 6
testing once the physical rig exists: the shot list should specifically include bare-head cases at
the same distance/angle helmet cases are tested at, not just a generic accuracy check.

**Qualitative check** (`docs/train_helmet/val_batch0_pred.jpg`, `val_batch1_pred.jpg`): visually
confirms real, confident helmet/head detections across varied real construction-site images --
consistent with the strong per-class AP50. Still pending: checking the model against the actual
demo-rig/webcam domain (foam mockup, classroom lighting) once that rig exists, per the domain-gap
risk already flagged in Step 2 -- validation-set performance on the Hard Hat Workers Dataset's own
images is not the same claim as "works on this demo's camera."

**Not yet done:** ONNX export + INT8 quantization (Step 4), testing on the Pi 4B (Step 5/6).

---

## 2026-10-09 — My mistake: `resume=True` on a completed run silently trained garbage

Documented the previous entry's "resume training" plan with `model.train(resume=True, epochs=50)`
-- this was wrong, and it's on me, not caught before handing it to the team to run.

**What happened:** Ultralytics strips a checkpoint's optimizer/epoch state once a run finishes
*normally*. `helmet_v1`'s 30-epoch run completed cleanly, so its `last.pt`/`best.pt` were already
stripped -- `resume` only works on an *interrupted* run's checkpoint. Pointing `resume=True` at a
completed run's checkpoint printed a warning (`not a resumable training checkpoint ... Starting
new training instead`) and silently fell back to a fresh `model.train()` call. Because that call
never specified `data=`, it defaulted to Ultralytics' `coco8.yaml` (a 4-image smoke-test set) and
trained a throwaway 80-class COCO model for 50 "epochs" (~29 seconds total, since coco8 is tiny)
into `/content/runs/detect/train/` -- not the real dataset, not the real class set, not saved
anywhere the project uses.

**No real damage:** the actual `helmet_v1` files on Drive were never touched by this (confirmed --
the `best.pt` downloaded afterward is byte-identical, same sha256, to the original 30-epoch
model), and the wasted Colab time was under a minute. Caught because the downloaded file's
checksum and the synced `results.csv` row count (still 31 lines, not 51) didn't match what a real
extended run should have produced -- the same "verify before trusting a result" habit that's
caught every other real bug on this project, including the user's own first two Colab bugs.

**Fix:** `docs/HELMET_TRAINING.md` corrected -- there is no resume-a-completed-run path in
Ultralytics. The real way to add more epochs after a run has finished is to load its `best.pt` as
a pretrained starting point for a brand-new, fully-specified `model.train()` call (explicit
`data=`/`project=`/`name=`, new run folder `helmet_v1b` so the original isn't overwritten) --
not a true LR-schedule-continuous resume, but the standard fallback when resume isn't available.

**Process note:** I should have verified this Ultralytics behavior (or at least flagged it as
unverified) before handing off a training command I had not run myself -- same lesson as the wrong
clone-path instruction earlier this project (2026-10-03): a Colab cell I write but can't execute
locally gets the same "verify before trusting" treatment as any other claim, not a pass because
it's "just a config change."

---

## 2026-10-09 — helmet_v1b (+20 epochs, corrected training command): no real improvement

Ran the corrected "continue fine-tuning" command from `docs/HELMET_TRAINING.md` (load v1's
`best.pt` as a pretrained start, 20 more epochs, explicit `data=`/`project=`/`name=`, new run
folder `helmet_v1b` so v1 wasn't overwritten). This time it trained on the real dataset (confirmed:
`data: data/helmet_roboflow/data.yaml` in `docs/train_helmet_v1b/args.yaml`, 21 rows in
`results.csv` = 20 real epochs).

**Result: no meaningful change from v1, within noise on every metric that matters:**

| Metric | v1 (30 ep) | v1b (+20 ep) | Delta |
|---|---|---|---|
| head AP50 | 0.963 | 0.957 | -0.006 |
| helmet AP50 | 0.982 | 0.977 | -0.005 |
| person AP50 | 0.034 | 0.032 | -0.002 |
| head->helmet confusion | 17% | 16% | -1pt |
| mAP50 (all classes) | 0.659 | 0.655 | -0.004 |

**The "results.png shows mAP50 still rising at epoch 30" theory from the previous entry was wrong
in practice** -- stated here plainly as a negative result, not hidden. The model had effectively
already converged; the visually-still-climbing curve was diminishing-returns noise, not real
headroom. `results.csv`'s single-point precision metric did drop sharply (0.967 -> 0.623) between
v1 and v1b, but this tracks a confidence-threshold operating point, not the PR-curve-integrated
AP50 (flat) -- read as calibration drift from restarting the LR schedule (fine-tuning isn't a true
resume, see the entry above), not a real quality regression.

**Decision: keep `models/helmet_v1_best.pt` (the original 30-epoch run) as the model of record.**
v1b is archived for evidence (`docs/train_helmet_v1b/`) but not promoted -- no measured benefit,
and the precision/calibration shift is a plausible real risk for the live demo with no offsetting
gain to justify it. `models/helmet_v1b_best.pt` was intentionally not copied into the project.

**Process note:** this is the second time in two days a specific, reasoned prediction
("investigate why mAP50 is low" -> correct; "more epochs will fix it" -> tested and wrong) was
checked against a real result instead of being left as an assumption. Worth remembering for the
report: a plausible-sounding curve read is still a hypothesis until tested, not a conclusion.

---

## 2026-10-10 — ONNX export + INT8 quantization benchmarked on the helmet model

Ran `src/export_helmet_model.py` (new): exports `models/helmet_v1_best.pt` to ONNX, dynamic-
quantizes to INT8, benchmarks all three forms (PyTorch, ONNX fp32, ONNX INT8) on real CPU latency
(5 warmup + 30 timed runs, laptop i7-13620H, same methodology as
`docs/IMPLEMENTATION_REPORT.md`'s parking-project comparison). Writes
`docs/train_helmet/export_benchmark.md`.

**Results:**

| Format | Size (MB) | ms/frame | FPS |
|---|---|---|---|
| PyTorch (.pt) | 6.23 | 53.3 | 18.8 |
| ONNX (fp32) | 12.27 | 23.3 | 43.0 |
| ONNX (INT8, dynamic) | 3.36 | 34.0 | 29.4 |

**Same pattern as the parking project, on a second, independent model:** ONNX export is a clear
win (+56.3% throughput vs PyTorch; parking project: +76%). INT8 quantization shrinks the model
72.6% but is 46.1% *slower* than ONNX fp32 -- matching the parking project's own counter-intuitive
finding almost exactly. Two different models, two different tasks, same result: dynamic INT8
quantization doesn't help on a generic x86 CPU without matching hardware acceleration. This is now
good evidence it's a real property of this CPU class, not a one-off fluke.

**Decision: deploy ONNX fp32, not INT8.** Both comfortably clear the <300ms/frame target on this
laptop CPU (23-53ms) -- not yet the Pi 4B specifically, that re-measurement is still Step 5/6's
job, not assumed to transfer from this number either.

---

## 2026-10-11 — First real webcam test: detection works, zone drawing was wrong

User ran `demo_helmet.py` on their laptop webcam for the first time (ONNX model, real face,
real room). **Helmet/no-helmet detection worked correctly** -- "NO HELMET" fired on a real bare
head at confidence 0.77-0.83, high-risk zone alert triggered as designed, events logged to
`helmet_events.jsonl` exactly as expected. This is the first real confirmation that the
Hard-Hat-Workers-trained model generalizes past its own validation images to an actual live
webcam feed -- closes part of the domain-gap question flagged in Step 2, for the detection side
at least (zone calibration is separate, see below).

**Real bug found and fixed:** the "normal zone" boundary rendered as a stray horizontal line in
the wrong place (screenshot showed it cutting across the middle of the frame, not forming a
sensible rectangle). Cause: `src/helmet_zone_configs.py`'s polygons were hardcoded pixel
coordinates assuming a 1280x720 frame -- the user's actual webcam captures at a different
resolution, so the absolute pixel values landed in the wrong place relative to the real frame
size. The exact same category of mistake as nothing before on this project, but a new instance of
it: a config number that looked reasonable but was never checked against the actual runtime
condition it would run under.

**Fix:** rewrote the zone polygons as *fractions* of frame width/height (0.0-1.0) instead of
absolute pixels, with a `scale_polygon()` helper that converts to real pixel coordinates using the
webcam's actual reported resolution (`cv2.CAP_PROP_FRAME_WIDTH/HEIGHT`), read fresh at the start
of every run instead of assumed. Verified the conversion is backward-compatible (1280x720 input
reproduces the exact original pixel values) and scales correctly for 640x480 and 1920x1080 too.
This makes the zone config resolution-independent -- it'll draw proportionally correctly on
whatever camera actually gets used, not just the one resolution it happened to be eyeballed
against.

**Still open:** the zone's *position* (which fraction of the frame is "high-risk" vs "normal")
remains an unverified draft -- this fix only makes the already-chosen fractions render correctly
at any resolution, it doesn't make the fractions themselves correct for the real foam-mockup rig.
That calibration still needs a real reference frame from the actual demo setup, per the existing
note in `helmet_zone_configs.py`.

---

## 2026-10-11 — Real Pi 4B latency: 0.30 FPS, far below the <300ms/frame target

Reviewed Hung's `docs/RASPI_CHECK_REPORT.md` (completed 2026-10-09, separate from the helmet
pivot work -- this is shared infrastructure carried over from the parking project's Pi 4B).

**Real, measured result: 0.30 FPS running stock `yolov8n.pt` on the Pi 4B** -- 61x slower than the
laptop CPU baseline (18.4 FPS), i.e. ~3.3s/frame. This is **11x over the <300ms/frame target**
stated in `docs/PROJECT_REPORT.md` Step 1's resource constraints. This was the raw, un-optimized
PyTorch model, not yet the ONNX export from Step 4 -- but a 61x gap is far larger than the ~56%
speedup ONNX gave on a laptop CPU, so ONNX alone will not close it.

**Confound not yet ruled out:** Hung's own report flags the Pi's power supply as 5V/2A, below the
recommended 5V/3A for a Pi 4B, with a low-power warning from the OS itself -- CPU throttling from
an underpowered supply can look exactly like "the hardware is just too slow" without being the
true ceiling. Camera sanity check (RASPI_CHECK_REPORT.md Step 3) also hasn't been run yet.

**Action before concluding anything stronger:** (1) swap in a proper 5V/3A supply and re-measure
before trusting 0.30 FPS as the Pi's real ceiling: (2) re-measure with the actual ONNX export
(Step 4) instead of stock `yolov8n.pt`, since that's what would really be deployed. If the gap
is still this large after both, this is real, concrete evidence for reconsidering the Coral USB
accelerator the team previously and deliberately deferred buying (parking project's own hardware
notes: "don't buy this first... prove the Pi genuinely can't hit >=5 FPS before spending on this")
-- 0.30 FPS is well under even that 5 FPS bar, so that condition may now be met, pending the two
checks above ruling out the power-supply confound first.

---

## 2026-10-11 — Live webcam confirms the 17% head/helmet confusion; added smoothing fix

User moved their head during the webcam test and saw the model flicker between "helmet" and
"NO HELMET" on the exact same real head within the same second -- confidence dropped to 0.44 on
the misread frame (vs. 0.77-0.83 standing still). This is the live, reproduced version of the 17%
head->helmet confusion already found in the confusion matrix (2026-10-09) -- motion blur was the
missing piece explaining *when* it happens, not just *that* it happens.

**Real risk this creates:** `zone.py`'s dwell counter only accumulates on frames classified as
"head" -- a flickering classification could meaningfully delay, or in the worst case prevent,
reaching the dwell threshold for a real violation, which is exactly backwards for a safety system
(more motion = more likely to be a real moving/rushing worker, not less deserving of a correct
flag).

**Fix: per-track majority-vote class smoothing**, added to `src/demo_helmet.py`. Each track's last
`CLASS_SMOOTHING_WINDOW=5` frames vote on "no helmet or not"; the majority decides, not the single
current frame (ties lean toward "no helmet" -- the safer direction to err on). 5 frames is
deliberately short so the smoothing itself doesn't meaningfully delay the 1-3s dwell thresholds.

**Verified in isolation** (not just assumed) with three synthetic sequences before trusting it:
- A single blurry misread surrounded by consistent "no helmet" reads: smoothed out, stays flagged
  the whole time (no false recovery).
- A single blurry misread surrounded by consistent "helmet" reads: smoothed out, stays compliant
  (no false violation).
- A genuine state change (helmet put on mid-sequence): still correctly reflected within the
  5-frame window, just with the expected small lag.

Raw per-frame confidence is still shown in the label and logged -- smoothing changes which
*decision* the system acts on, not what gets recorded.

---

## 2026-10-11 — Pi 4B latency plan: re-test ONNX/INT8/resolution on-device before buying anything

Thought through how to actually close the 0.30 FPS gap found in Hung's Pi check, instead of
jumping straight to buying hardware. Reasoning:

- A 61x laptop-vs-Pi gap is too large to be fully explained by the known power-supply issue
  (5V/2A vs the recommended 5V/3A, RASPI_CHECK_REPORT.md) -- real-world Pi4-vs-modern-laptop gaps
  for this kind of workload are usually cited around 10-20x, not 61x. Fixing the PSU is still
  worth doing first (free, rules out a real confound), but shouldn't be expected to close the
  whole gap alone.
- The 0.30 FPS number was measured on stock, un-optimized `yolov8n.pt` -- not yet the ONNX export
  that Step 4 already found +56.3% faster on the laptop.
- INT8 was rejected on the laptop's x86 CPU (Step 4, 2026-10-10) -- but the old parking project's
  own report explicitly flagged ARM/Jetson INT8 behavior as unverified and likely different from
  x86. Never actually tested on ARM before now.
- Reduced input resolution (imgsz) is a cheap, no-retraining lever that was also never tried.

**Decision: test the cheap software levers on the real Pi before considering the Coral USB
accelerator.** Buying hardware this late in the course timeline, with the added complexity of a
TFLite/Edge-TPU conversion pipeline, is a real cost/risk that should only be paid if software
options are confirmed insufficient -- not assumed insufficient.

Wrote two scripts for Hoang/Hung to run:
- `src/export_helmet_sizes.py` (laptop side) -- exports fixed-size ONNX variants at imgsz 416 and
  320 alongside the existing 640 export.
- `src/benchmark_pi.py` (Pi side) -- benchmarks ONNX fp32 @ 640, INT8 @ 640, and fp32 @ 416/320,
  real latency on the Pi itself, not assumed from the laptop numbers.

**Real bug caught and fixed while building these, before handing them off:** the first version of
`export_helmet_sizes.py` exported each size straight from `helmet_v1_best.pt` in a loop --
Ultralytics' `.export()` always derives its output filename from the model's own stem
(`helmet_v1_best.onnx`) regardless of `imgsz`, so each loop iteration **silently overwrote the
real, actively-deployed 640 ONNX file** before my own rename-after step could move it out of the
way. Running the script once actually deleted `models/helmet_v1_best.onnx` -- caught immediately
by re-testing the live demo rather than assuming the export had gone cleanly, restored the file by
re-exporting from the untouched `.pt` (verified via sha256 and a real predict() call that it's
functionally identical), then fixed the root cause: each size now exports from a uniquely-named
temporary copy of the `.pt` so Ultralytics can never derive a colliding output filename again.
Verified the fix with a sha256 checksum of `helmet_v1_best.onnx` taken before and after a full
re-run of the fixed script -- byte-identical, confirmed untouched.

**Real, not just theoretical, accuracy note already visible on the laptop:** the 640 model found
16 boxes on the benchmark image, 416 found 5, 320 found **0**. Resolution reduction is a real
speed/accuracy trade, not a free win -- flagged directly in `benchmark_pi.py`'s own output so
whoever runs it on the Pi doesn't just chase the fastest number.

**Still needed:** the team actually running `benchmark_pi.py` on the real Pi (after fixing the
power supply) and reporting the real numbers back -- nothing in this entry claims a Pi result that
hasn't been measured yet.

## 2026-10-13 — End-to-end pipeline benchmark + a run-to-run variance finding (Hieu)

**Why:** the midterm exam's Step 4 critique kept flagging the same gap -- every prior benchmark
(`export_helmet_model.py`, `benchmark_pi.py`) measures raw ONNX `session.run()` only, not the
actual deployed pipeline (`demo_helmet.py`'s `model.track()` + per-track class-smoothing + two
`RestrictedZone.update()` calls). Wrote `src/benchmark_end_to_end.py` to close that gap by timing
the real call path, importing `zone.py`/`helmet_zone_configs.py` unchanged rather than
re-implementing their logic.

**Real finding caught while building this, before trusting any number:** re-running the EXISTING
standalone benchmark (`export_helmet_model.py`) on this same laptop, same image, same model, on
three separate occasions gave ONNX fp32 numbers of 23.3ms, 30.8ms, and 29.7ms/frame -- roughly
+-30% run-to-run variance from background system load, not from the model or code changing at all.
Comparing a fresh full-pipeline number against an old standalone constant from a different run
would have confounded that noise with the real pipeline overhead. Fixed by measuring both
standalone `session.run()` and the full pipeline in the *same* process invocation, back to back,
so they share the same system-load conditions and the delta between them is attributable to the
pipeline code, not noise.

**Result (laptop CPU, two independent same-run comparisons):**

| Run | Standalone ONNX | Full pipeline (track + zones) | Overhead |
|---|---|---|---|
| 1 | 25.4 ms/frame (39.4 FPS) | 52.9 ms/frame (18.9 FPS) | +27.5 ms (+109%) |
| 2 | 26.6 ms/frame (37.6 FPS) | 57.0 ms/frame (17.5 FPS) | +30.4 ms (+115%) |

The absolute millisecond figures still move around run to run (same cause as above), but the
**relative overhead is stable at roughly +110%** across both runs -- Ultralytics' `track()` call
plus this project's own per-frame Python logic currently costs about as much as the model
inference itself. Still comfortably passes the <300ms/frame target on this laptop CPU (53-57ms),
but this was previously an unmeasured, assumed-small gap, not a verified-small one.

**Still open:** this is a laptop CPU number, not the Raspberry Pi 4B -- the standalone-vs-full-
pipeline gap is now closed, the laptop-vs-Pi gap is not. Re-running this same script on the Pi
(once its power supply and camera check are done, per `docs/PI_BENCHMARK_GUIDE.md`) would be the
natural next step, since the 2x pipeline overhead found here might matter more, not less, on
slower ARM hardware.

## 2026-10-13 — Three more software-only fixes, no hardware required (Hieu)

Closing out the remaining gaps that don't actually need the Pi or the physical rig, rather than
leaving them all parked behind "waiting on hardware."

**1. INT8 accuracy impact -- partially measured, honestly scoped.** `docs/CHANGELOG.md` and this
file have said "INT8's accuracy impact specifically not measured" since 2026-10-10. A real
dataset-wide mAP comparison needs `data/helmet_roboflow/` locally, which needs `ROBOFLOW_API_KEY`
-- not available in this environment, a real and stated blocker, not worked around. What doesn't
need that: `src/compare_int8_accuracy.py` runs both the fp32 and INT8 ONNX models on the one real
benchmark image already in the repo (`val_batch0_pred.jpg`) and compares predictions directly.
Result: **identical box count (16) and class breakdown (head=3, helmet=13, person=0) between fp32
and INT8, average confidence within 0.001 (0.500 vs 0.499).** A single-image result, stated as
exactly that in the script's own output -- not a substitute for a real mAP number, but a real,
reassuring data point where previously there was none at all.

**2. Two real bugs found and fixed in `demo_helmet.py` while implementing the "uncertain
confidence band" idea from the midterm exam answer:**
- The confidence-band idea itself is now real code: a raw per-frame detection with confidence in
  [0.40, 0.60] is treated as "no helmet" (safety-first, same reasoning as the existing temporal
  tie-break) but tagged `confidence_band: "uncertain"` in the event log, and labelled
  `[UNCERTAIN - review]` on screen, so a human reviewing flagged events can tell a low-confidence
  call from a clean one instead of every alert carrying equal weight.
- **Found while testing that change, not something I was looking for:** `EVENTS_OUT.write_text()`
  was overwriting the whole event log on every run, silently discarding all prior history. There
  was no retention *policy* partly because there was nothing persisted for a policy to act on.
  Fixed by loading existing events on startup, purging anything past `EVENT_RETENTION_DAYS` (30,
  a stated starting default, not yet matched to a real policy), and appending instead of
  overwriting. Every event now also carries a real `logged_at` timestamp (previously only a
  relative mm:ss since script start, which can't support a retention rule at all) and a
  `reviewed: false` field as a hook for an eventual correction/appeal workflow.
- Verified with a manual test: seeded one 45-day-old and one 5-day-old synthetic event, ran the
  script -- it printed `Retention: dropped 1 event(s) older than 30 days` and kept the 5-day-old
  one, confirming both the persistence fix and the retention policy work. Also caught and fixed a
  third, smaller bug this same test surfaced: the end-of-run "Violations flagged" count was
  counting carried-over history as if it were new activity from that run -- now reported
  separately (`Violations flagged this run: N (+M carried over, T total in log)`).

**Still genuinely blocked, not just deferred:** domain-gap verification against the real demo rig,
zone-polygon calibration against a real camera frame, and the Pi-specific latency/accuracy numbers
all require hardware (camera, rig, Pi) that isn't available in this environment -- these are not
"didn't get to it," they're "can't do it without the physical setup," and are left exactly as open
as they were.

## 2026-10-16 -- First real `benchmark_pi.py` run on the Pi 4B (Hung), partial

Hung ran `src/benchmark_pi.py` on the Pi. Real result, pasted verbatim:

| Format    | imgsz | ms/frame | FPS | <300ms? |
|---|---|---|---|---|
| ONNX fp32  | 640 | [missing: `helmet_v1_best.onnx` -- not copied to the Pi, skipped] | | |
| ONNX INT8  | 640 | [missing: `helmet_v1_best_int8.onnx` -- not copied to the Pi, skipped] | | |
| ONNX fp32  | 416 | 256.9 | 3.9 | YES |
| ONNX fp32  | 320 | 127.6 | 7.8 | YES |

`vcgencmd get_throttled` → `0x0` -- no undervoltage/throttling flag at the moment of the run (a
snapshot, not a sustained-load guarantee, but a real improvement over the under-spec-PSU state
`docs/RASPI_CHECK_REPORT.md` found on 2026-10-11).

**The two rows that ran are not a usable answer, and I checked rather than assumed that from the
docstring's own warning.** Re-ran all three real exported sizes against the laptop's benchmark
image, matching each ONNX file to its own correct `imgsz` (the first attempt used the wrong default
and errored -- `Got invalid dimensions ... Got: 640 Expected: 416`, fixed by passing `imgsz=` into
`model.predict()` explicitly):

| imgsz | boxes | head | helmet | person |
|---|---|---|---|---|
| 640 | 16 | 3 | 13 | 0 |
| 416 | 5  | **0** | 5 | 0 |
| 320 | 0  | 0 | 0 | 0 |

At 416 -- the faster of the two rows that actually ran on the Pi -- every head detection on this
image disappears; the 5 surviving boxes are all `helmet`. That is not "somewhat less accurate," it
is "blind to the exact class this whole project exists to catch" on this image. At 320, nothing is
detected at all, confirming the docstring's own warning rather than just repeating it. Both Pi rows
pass the <300ms latency target while being unusable for the safety purpose -- latency and
usefulness are not the same test, and this is a real example of latency passing while usefulness
fails.

**Next step, not yet done:** copy `helmet_v1_best.onnx` and `helmet_v1_best_int8.onnx` from the
laptop's `models/` to the Pi's `models/` and re-run `benchmark_pi.py` -- those are the two formats
most likely to actually detect anything useful, and they're exactly the two rows still missing.
Until that run exists, "can the Pi meet the latency target without losing detection" is still an
open question, not a settled "yes" from the 320 row alone.

## 2026-10-17 -- Real dataset-wide mAP: fp32 vs INT8 (Hieu), closes a gap flagged since 2026-10-13

`src/compare_int8_accuracy.py` (2026-10-13) could only compare fp32 vs INT8 on one real image,
explicitly because `ROBOFLOW_API_KEY` wasn't available in this environment to download the full
val set. Hieu ran it himself on his own machine, where that's not a blocker: found the key had
already been used for the old signs-project Colab notebook, set it locally, ran
`src/download_helmet_data.py` (1413 val images, class order confirmed `head/helmet/person`,
matching `CLASS_NAMES`), then `yolo val` on both ONNX models against the real val set. Full
results committed at `models/mAP/fp32/` and `models/mAP/int8/` (plots, confusion matrices, PR
curves -- not just the numbers below).

| | Box(P) | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **fp32 -- all** | 0.961 | 0.623 | 0.660 | 0.463 |
| **fp32 -- head** | 0.920 | 0.926 | 0.960 | 0.674 |
| **fp32 -- helmet** | 0.964 | 0.942 | 0.980 | 0.693 |
| **fp32 -- person** | 1.000 | 0.000 | 0.041 | 0.023 |
| **INT8 -- all** | 0.961 | 0.623 | 0.659 | 0.462 |
| **INT8 -- head** | 0.917 | 0.930 | 0.957 | 0.672 |
| **INT8 -- helmet** | 0.966 | 0.938 | 0.980 | 0.692 |
| **INT8 -- person** | 1.000 | 0.000 | 0.040 | 0.023 |

Inference speed in this same run: fp32 30.1ms/image, INT8 39.9ms/image (CPU) -- INT8 slower
again, consistent with the original Step 4 rejection (2026-10-10) and the laptop's own prior
`export_benchmark.md` finding, not a new result.

**What this actually answers, from the 3 open questions in the guide (`docs/HIEU_ROBOFLOW_MAP_GUIDE.md`):**

1. **fp32 vs INT8 overall:** mAP50 0.660 vs 0.659, mAP50-95 0.463 vs 0.462 -- a real, dataset-wide
   confirmation of the single-image finding from 2026-10-13 (identical box count/class, confidence
   delta 0.001): INT8 quantization costs essentially nothing in accuracy here. Combined with INT8
   being slower on this CPU, the Step 4 decision to deploy fp32 and reject INT8 holds for both
   reasons now, not latency alone.
2. **`head` class (the 17% head-\>helmet confusion):** mAP50 0.960 (fp32) vs 0.957 (INT8) -- a
   0.3 point difference, noise-level, not something quantization is making meaningfully worse.
   The 17% confusion is a model/data problem, not a quantization side-effect.
3. **`person` class (AP50=0.034 at training time):** mAP50 0.041 (fp32) vs 0.040 (INT8) here --
   matches the original training-time number closely (0.034 vs 0.041, same order of magnitude,
   small difference expected since this is the ONNX-exported model validated fresh, not the exact
   same run), and fp32 vs INT8 themselves are within noise of each other. Confirms what was
   already suspected: the `person` gap is a data/model problem (31:1 class imbalance), not
   something quantization makes better or worse.

**Bonus cross-check, not one of the 3 questions but worth noting:** the fp32 ONNX mAP50 numbers
(head 0.960, helmet 0.980) match the original `.pt` model's training-time AP50 (head 0.963, helmet
0.982, 2026-10-09) almost exactly -- confirms the ONNX export itself didn't silently degrade
accuracy, a question that was never directly checked before now.

## 2026-10-17 (later) -- `audit_dataset.py` re-pointed at the real dataset, clean result

The script had been hardcoded to `data/ua_detrac_yolo/` (the old parking project's vehicle
dataset) since before the pivot -- flagged repeatedly as "needs updating if reused" but never
actually done until now that `data/helmet_roboflow/` exists locally. Re-pointed it: Roboflow's
export layout puts `images/`/`labels/` under each split (`train`/`valid`/`test`), not split under
a shared `images/`/`labels/` like the old UA-DETRAC conversion did, and class names changed to
`head`/`helmet`/`person` (order confirmed against `data/helmet_roboflow/data.yaml`, matching
`CLASS_NAMES` already used in `compare_int8_accuracy.py`/`benchmark_end_to_end.py`). Also had to
filter the label-folder listing to `.txt` only -- Ultralytics drops a `labels.cache` file in the
same folder (created by the `yolo val` runs today), and without filtering it showed up as a false
orphan label.

**Real result, ran against all 7,035 images:** 4916 train + 1413 valid + 706 test checked, **0
issues of every kind** -- no corrupt images, no orphan images/labels, no malformed label lines,
no out-of-range coordinates, no zero-size boxes, no bad class ids. This is a real, clean signal
about the source dataset itself (Roboflow's own curated export), not a guess -- the one prior
dataset this project used (UA-DETRAC + merged motorbike sources) needed an actual fix for a real
corrupt-file bug (2026-09-18); this one needed none.

## 2026-10-17 (later still) -- Decisions on the two remaining model gaps (head→helmet 17%, person AP50)

Both gaps were left open in `docs/team_guides/HIEU_DATA_MODEL.md` as "pick a direction, don't
leave it blank." Neither can be resolved by code changes alone -- both require retraining
(augmentation, per-class loss weighting, or oversampling all change what the model learns, not
just how it's measured), which needs the team's Colab T4 time, not something decidable or
runnable from this environment (no local GPU). Decision made now so neither gap sits undecided
going into the report/demo, with the reasoning stated plainly so the team can override it:

**`head`→`helmet` 17% confusion -- documented as a known limitation for this cycle, not
retrained.** Reasoning: today's dataset-wide mAP run (see entry above) confirmed this gap is real
and stable (head mAP50 0.960, consistent across fp32/INT8) with no code-level fix available --
only a retrain can move it, and a retrain's actual effect on this specific confusion is unproven
(the one retrain already tried this project, `helmet_v1b` +20 epochs, showed no real improvement
on anything, 2026-10-09). Spending remaining time on an uncertain retrain this close to a live
demo is a real risk, not just caution for its own sake. **If the team decides there's time for
one more retrain attempt, per-class loss weighting (upweighting `head`'s loss contribution) is
the lower-effort, more targeted lever to try first** -- cheaper than designing new augmentation,
and directly targets the class under-performing, rather than the whole dataset.

**`person` AP50 (0.034 train-time, 0.041 on today's full-val re-check) -- documented as a known
limitation, not retrained.** Reasoning: same as above (today's run confirmed it's unaffected by
quantization, so it's a data problem specifically -- 31:1 class imbalance vs `helmet`), plus this
class is **not used by the zone/violation logic at all** (`src/demo_helmet.py` only acts on
`head`/`helmet`), so unlike the `head` gap this one has no safety consequence for the live demo,
only a reporting-honesty one. Lower priority than `head` for that reason. **If retrained anyway,
oversampling `person`-containing training images** (same technique already used successfully for
the old parking project's motorcycle class, 2026-09-18) is the direct lever.

**Both now belong in the report's limitations section, Step 7 (monitoring/iteration) as planned
future work, not as open TODOs for this cycle.**

## 2026-10-18 -- Step 8 Ưu tiên 2 closed: retention + correction/appeal + access governance

Report mục 11 (Ưu tiên hiệu chỉnh) named 3 open Step 8 gaps: retention policy, appeal/correction
mechanism, governance on log access. All 3 are code-or-decision work, not blocked on physical
rig/Pi access, so closed now rather than left for Hạnh to chase down separately:

- **Retention:** already implemented 2026-10-13 (`EVENT_RETENTION_DAYS = 30` in
  `src/demo_helmet.py`) -- the report text just hadn't been updated to reflect it.
- **Correction/appeal:** new `src/review_events.py` -- CLI to mark a specific flagged event
  (matched by `track_id` + `timestamp`, since no identity is attached to a track by design)
  reviewed, with a `correction` (`confirmed`/`false_positive`) and note. Original event kept,
  never edited/deleted -- audit trail stays complete. Self-tested against a throwaway events file
  (not the real log): wrote 1 synthetic violation event, ran the mark-reviewed command, confirmed
  `reviewed`/`correction`/`review_note`/`reviewed_at` were added and `--list-unreviewed` then
  showed nothing pending.
- **Access governance:** decided -- raw `helmet_events.jsonl` (per-event detail) stays
  team+safety-officer-only for this prototype; aggregate `compliance_snapshot` events are what's
  shareable more broadly, consistent with the "no raw video, only zone/timestamp/confidence"
  privacy posture already in Step 8.

Full writeup: `docs/EVENTS_GOVERNANCE.md`. Report action for Hạnh: update Step 8 score (was
3.0/4, named these exact 3 gaps) and mục 11 Ưu tiên 2 status once reviewed.

## 2026-10-21 -- Real Pi 4B results, full 4/4 benchmark_pi.py rows + first real end-to-end pipeline run (Hung)

Both missing model files confirmed copied to the Pi (`ls -lh ~/EdgeAI/models/`, `helmet_v1_best.onnx`
and `helmet_v1_best_int8.onnx` timestamped Oct 21 16:50). `benchmark_pi.py` now ran to completion,
closing the item flagged 2026-10-16 as only 2/4 rows:

```
Format        imgsz   ms/frame    FPS     <300ms?
ONNX fp32     640     472.1       2.1     no
ONNX INT8     640     553.3       1.8     no
ONNX fp32     416     234.4       4.3     YES
ONNX fp32     320     129.7       7.7     YES
```

**This is the decision-forcing result the team has been waiting on.** Neither 640 config meets the
<300ms target on the real Pi (fp32 472ms, INT8 553ms -- INT8 again slower than fp32 despite being
smaller, a third confirmation of the same counter-intuitive pattern already seen on laptop x86 CPU,
now also true on Pi's ARM CPU). The two configs that DO meet <300ms (416, 320) are the ones the
2026-10-16 laptop box-count check already showed losing detection -- 416 found 0 `head` boxes on the
benchmark image (the safety-critical class), 320 found 0 boxes of any class. So **no single
CPU-only configuration on this Pi both meets the latency target and keeps working detection** --
exactly the branch the guide (`docs/team_guides/HUNG_TESTING_EVAL.md` item 4) already named in
advance: next step is Coral USB Accelerator (if budget/time allow) or running the main demo on
laptop with the Pi documented as a known hardware limitation. This is a deploy decision for
Hoàng/Hạnh, not resolved by this benchmark alone.

**Thermal check across two separate sustained-load runs, no throttling either time:**
`vcgencmd get_throttled` = `0x0` after both the `demo_helmet.py` run below (116s continuous) and the
full `benchmark_pi.py` run (4 settings x 18 runs each). Temp readings: 41.3°C before benchmark_pi,
47.2°C after; 41.8°C before the demo_helmet run, 58.9°C after. All well under the ~80°C Pi 4B
throttle point -- closes the "more than one temp reading" gap flagged 2026-10-16 (that was a single
reading at 43.4°C after 5 min). `RASPI_CHECK_REPORT.md` (last edited 2026-10-09) still records a
5V/2A supply; Hieu later reported (2026-10-24, verbally, not in the repo) that Hung has since
switched to a 5V/3A adapter -- **unconfirmed whether that was before or after these runs.** No
undervoltage/throttle event was detected by `vcgencmd` (`0x0`) during either of these two runs
regardless. Action: Hung confirms the adapter rating and the date he swapped it, then
`RASPI_CHECK_REPORT.md` Bước 1 gets updated.

**First real run of the full pipeline on the Pi** (not just standalone ONNX benchmark):
`python3 demo_helmet.py --no-display --source test_loop.mp4` (test_loop.mp4, not a live camera --
see camera gap below) -- 150 frames, 116.0s wall-clock = 1.3 FPS (773ms/frame), 3 violations
flagged, event log written successfully including the 2026-10-18 `reviewed`/retention fields.
Loaded `helmet_v1_best.onnx` (the fp32 640 model, demo_helmet.py's default) -- compared against this
same run's own standalone benchmark_pi.py fp32@640 row (472.1ms/frame), the full pipeline
(track() + class-smoothing + 2x zone-check) is **~64% slower** than standalone ONNX inference alone
on the Pi (773.3ms vs 472.1ms). Same direction as the laptop finding (~110% slower there,
`docs/CHANGELOG.md` 2026-10-13) -- a smaller relative overhead on Pi than on laptop, but confirms
the gap is real on both platforms, not a laptop-only artifact. No crashes, no errors -- first
confirmed working run of the actual deployed pipeline (not a standalone benchmark) on real ARM
hardware.

**Still open, not closed by this run:** `RASPI_CHECK_REPORT.md` Bước 3 (camera sanity check -- a
live `cv2.VideoCapture(0)` frame) is still unchecked -- confirmed directly by Hùng's own report
(`docs/HUNG_RUN_AND_REPORT.md`): "Pi chưa gắn camera" (no camera attached to the Pi yet). Also
worth being precise about what `test_loop.mp4` actually is, per the same report: a 10-second loop
of **one static image**, not varied real footage. So this run is solid evidence the pipeline runs
without crashing on real ARM hardware and gives a real FPS/timing number, but the "3 violations
flagged" and detection behavior should not be read as validating accuracy on varied real content
-- same frame repeated, not a motion/domain-gap test.

## 2026-10-24 -- Correction: 416/320 DO keep `head` on the full val set (Hieu), reverses the 2026-10-16/21 conclusion

The 2026-10-16 claim "416 loses all `head`, 320 detects nothing" came from **one image**
(`val_batch0_pred.jpg`) and was then carried into the 2026-10-21 conclusion ("no CPU-only config
both passes <300ms and keeps detection"). Never measured dataset-wide until now. Ran `yolo val` on
the full 1413-image val set, using the actual fixed-size ONNX files for 416/320 (480 only as the
dynamic `.pt`, no 480 ONNX exists yet):

| imgsz | Pi ms/frame (2026-10-21) | head AP50 | head Recall | helmet AP50 | all mAP50 |
|---|---|---|---|---|---|
| 640 fp32 | 472.1 (fails <300) | 0.960 | 0.926 | 0.980 | 0.660 |
| 480 (.pt) | not exported / not measured | 0.950 | 0.910 | 0.975 | 0.653 |
| 416 ONNX | 234.4 (passes) | 0.936 | 0.888 | 0.967 | 0.644 |
| 320 ONNX | 129.7 (passes) | 0.896 | 0.845 | 0.936 | 0.619 |

Both 416 and 320 clear the project's 0.80 AP50 bar for head/helmet dataset-wide. The one-image check
was real (that image's 3 small heads are missed at 416) but it was a worst-case sample, not the
typical case. **Cost is still real:** head recall falls 0.926 -> 0.888 (416) / 0.845 (320), i.e. more
missed no-helmet cases, the costlier failure for this system. **416 is the sensible candidate**
(small recall loss, 66ms margin under target); 320 loses ~8 recall points.

**Not yet answered:** (1) these are dataset-val numbers, not the demo rig's domain -- must re-check
on real rig frames; (2) <300ms was met by standalone inference only. The full pipeline was ~64% slower
than standalone on the Pi (773 vs 472ms), so 416 end-to-end is roughly 385ms by that ratio --
an estimate, **unmeasured**. Next: Hung runs `demo_helmet.py --model models/helmet_v1_best_416.onnx`
(and 320) on the Pi. If end-to-end still misses 300ms, levers not yet tried: 480 ONNX export,
detect every 2nd frame, crop to the zone region.
