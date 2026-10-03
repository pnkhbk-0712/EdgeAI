# Project pivot: Illegal Parking Detection -> Construction Helmet (PPE) Compliance

**Date:** 2026-10-03
**Branch:** `helmet-safety-pivot` (the original illegal-parking project stays intact on `main` --
nothing from that work is deleted, this is a parallel branch, not a rewrite of history).

## Why we pivoted

Not a technical failure of the parking project -- v1/v2 models both trained successfully
(mAP50 0.743 / 0.746), 4 real pilot videos were collected and zone-verified, and a full labeling
pipeline was ready to go. The reasons for pivoting are logistics, not capability:

1. **The teacher wants a live, in-class demo, not a pre-recorded video.** The parking project's
   core violation event (a vehicle parked >60s, or stopped >5s) is awkward to stage live and
   convincingly in a classroom -- there's no real street, real traffic, or real restricted-parking
   sign to point a camera at.
2. **Team bandwidth this semester is tight** across multiple other courses. The team wants a
   project that is easy to demo well and score well on, rather than maximizing novelty, given the
   time available.

A cheaper fix (live webcam demo using printed vehicle photos moved in and out of a virtual zone)
was proposed first and would have worked technically, but the team decided a project that is
*inherently* easy to demonstrate with simple physical props is a better use of remaining time than
staging a workaround for a project whose natural subject (street traffic) can't be brought into a
classroom at all.

## New topic: Construction Helmet / PPE Compliance Detection with Risk-Zone Tiering

### The problem, with real numbers (not a generic "safety is important" claim)

- Vietnam recorded **8,286 workplace accidents in 2024** (+12.1% YoY), including **675 fatal
  accidents and 727 deaths** (+4.81% YoY), with **>43,000 billion VND** in material damage
  ([vneconomy.vn](https://vneconomy.vn/tai-nan-lao-dong-gay-thiet-hai-hon-43-000-ty-dong-trong-nam-2024.htm),
  [nhandan.vn](https://nhandan.vn/post-861015.html)).
- Construction is one of the most dangerous sectors, accounting for **62% of all workplace
  fatalities** ([vneconomy.vn](https://vneconomy.vn/tai-nan-chet-nguoi-trong-xay-dung-lien-tuc-tang.htm)).
- Within construction specifically, **69.1% of accidents are falls from height and falling
  objects** ([nhandan.vn](https://nhandan.vn/bao-dam-an-toan-tai-cac-cong-trinh-xay-dung-post708121.html))
  -- exactly the injury mechanism a hard hat is designed to mitigate.
- Root causes are organizational, not just "no equipment available": inadequate training (53.8%
  of cases) and insufficient safety-management procedures (35.2%), plus substandard PPE or PPE not
  verified in use. A human site supervisor cannot watch every worker, 100% of the time -- this is
  the gap a continuously-running edge AI monitor fills, the same "why edge AI over manual
  monitoring" argument the parking project made for its own domain.

### Legal grounding (verify exact clause numbers before quoting verbatim in the proposal)

- **Luật An toàn, vệ sinh lao động số 84/2015/QH13** -- employers must provide PPE for hazardous
  work; workers must use it once provided.
- **Thông tư (Bộ LĐ-TB&XH), danh mục PTBHLĐ theo nghề** -- hard hats are effectively mandatory on
  any active construction site; harnesses are mandatory specifically for work at height.
- **QCVN 18:2021/BXD** (Bộ Xây dựng) -- national technical regulation on construction-site safety.
- Masks/respirators are NOT a blanket requirement everywhere on site -- they're tied to
  dust/chemical-generating tasks specifically (cutting, grinding, welding), per occupational dust
  exposure limits (Bộ Y tế). This is why the scope below is hard-hat-first, not a generic "PPE"
  catch-all -- the legal mandate is strongest and least ambiguous for the hard hat.

### Scope (v1 / MVP)

**In scope:**
1. **Hard hat detection** -- YOLOv8n, classes: `helmet`, `no-helmet` (and optionally `person` if
   the chosen dataset ships it as a separate class for cleaner counting). Reuses the parking
   project's training/export/quantization pipeline almost unchanged -- swap dataset and class
   names only.
2. **Risk-zone tiering** -- reuses `src/zone.py`'s `RestrictedZone` class directly (it's already
   fully generic: polygon + dwell frames + optional schedule, nothing vehicle-specific in it).
   Two zone tiers per camera view:
   - **High-risk zone** (scaffolding / elevated work mockup in the physical demo) -- a `no-helmet`
     detection here is an immediate/urgent alert.
   - **Normal zone** (ground level) -- a `no-helmet` detection here is logged, not urgently
     alerted.
3. **Per-person tracking + dwell debounce** -- same mechanism as the parking project's per-vehicle
   dwell counter, just reused for people: require N consecutive frames without a helmet before
   flagging, so a single frame where the helmet is occluded (person looks down, etc.) doesn't
   trigger a false violation.
4. **Compliance-rate logging** -- reuse the `events.jsonl` logging pattern from the parking
   project; aggregate into a simple "% compliant over the last N minutes" figure for the Testing &
   Evaluation section and for a live second screen during the demo.
5. *(stretch, time-permitting)* a sound/visual alert cue on a high-risk-zone violation, for demo
   impact.

**Explicitly OUT of scope for v1 (documented here so it isn't silently forgotten, same practice as
every other limitation logged in RUNLOG.md on the old project):**
- **Safety harness / lanyard detection.** Investigated first: available public datasets are much
  smaller (~2,000-4,000 images vs. tens of thousands for hard hats), harnesses are visually much
  harder to detect (thin straps, heavy occlusion by clothing/posture), and published results are
  noticeably weaker (one YOLOv11 harness-detection paper reports mAP 73.7%, well below what hard-hat
  detectors typically reach). More fundamentally, "is a harness visible" does not establish the
  thing that actually matters for fall safety -- whether it's **clipped to an anchor point** --
  which is a relational/spatial-reasoning problem, not a plain object-detection one. Flagged as
  future work rather than attempted this cycle, given the team's time constraints.
- Masks/respirators and other PPE types -- same reasoning as above (task-conditional requirement,
  not a blanket one, and adds scope without a correspondingly strong legal/dataset case).

### What's reused vs. new

| Component | Status |
|---|---|
| `src/zone.py` (`RestrictedZone`) | Reused as-is for risk-zone tiering |
| Per-track dwell-frame debounce logic | Reused as-is (same pattern, new object class) |
| `events.jsonl` logging pattern | Reused as-is |
| YOLOv8n train/export/quantize pipeline (Colab notebook) | Reused, swap dataset + class names |
| Dataset | New -- hard hat detection datasets (Kaggle/Roboflow), see next RUNLOG entry |
| Physical demo rig | New -- foam-board scaffolding mockup + mannequin/figure props with and
  without helmets, built by the team for the live classroom demo |

### Status

Old project (`main` branch): untouched, fully documented, available if the team needs to fall
back to it. This branch (`helmet-safety-pivot`) starts fresh data/model work next.
