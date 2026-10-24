"""Risk-zone config for the live classroom demo (2026-10-03; made resolution-independent 2026-10-11
after a real webcam test showed the zone drawn in the wrong place -- the hardcoded 1280x720 pixel
polygons only looked right on a webcam that happened to capture at exactly that resolution; most
laptop webcams don't (640x480, 1920x1080, etc. are all common), so the "normal zone" line was
landing at the wrong height/width on a real run.

**DRAFT placeholders -- not yet calibrated against the real physical rig.** The team plans to
build a foam-board scaffolding mockup + figures with/without helmets for the live demo. Until
that rig exists and a reference frame from the actual demo webcam is available, these polygons
are just a reasonable starting guess, expressed as *fractions* of the frame (0.0-1.0 on each axis)
so they scale correctly to whatever resolution the actual webcam captures at, instead of being
pinned to one specific pixel size:
  - HIGH_RISK_ZONE: upper portion of frame, where the foam scaffolding mockup is meant to sit
    (farther from camera / "elevated").
  - NORMAL_ZONE: lower portion, ground-level area in front of it.

**Before the real demo**, follow the same verify-before-trusting workflow used throughout the
parking project: grab a real frame from the actual demo webcam with the actual rig in place, draw
these polygons on it (scaled via `scale_polygon` below), and visually confirm they land where the
rig/figures actually are -- then update the fractions here. Do not trust these numbers unverified.

Dwell thresholds are intentionally very short compared to the parking project (that one waited
60s for a "parked" car; this is a live demo, not hours of footage) -- the point is immediate
feedback when a presenter moves a no-helmet figure into a zone, not debouncing a slow real-world
process.
"""

# DRAFT -- recalibrate against a real demo-webcam frame before relying on this. Fractions of
# frame width/height (0.0-1.0), not pixels -- see scale_polygon().
HIGH_RISK_ZONE_POLYGON_FRAC = [(0.234, 0.0), (0.766, 0.0), (0.766, 0.472), (0.234, 0.472)]
# DRAFT -- recalibrate against a real demo-webcam frame before relying on this.
NORMAL_ZONE_POLYGON_FRAC = [(0.0, 0.556), (1.0, 0.556), (1.0, 1.0), (0.0, 1.0)]


def scale_polygon(polygon_frac, frame_width, frame_height):
    """Convert a fractional (0.0-1.0) polygon to real pixel coordinates for this frame size --
    call once per run with the actual captured frame's width/height, not a hardcoded assumption."""
    return [(round(x * frame_width), round(y * frame_height)) for x, y in polygon_frac]

# High-risk zone: near-instant alert (this is a live demo, not a real multi-minute dwell case).
HIGH_RISK_DWELL_SECONDS = 1.0
# Normal zone: a little debounce so a momentary head-turn/occlusion doesn't false-trigger.
NORMAL_DWELL_SECONDS = 3.0

# PPE items that must be worn in each zone (2026-10-24). A GROUP ASSUMPTION, not a rule from the
# law: QCVN 18:2021/BXD 2.19.1.1 only says PPE must fit the job and its hazards, so which items a
# zone requires is a site decision -- change it here. Items the loaded model cannot detect are
# skipped at run time (src/ppe_compliance.py), so the old helmet-only model still runs.
# Names: helmet | vest | gloves | goggles.
REQUIRED_ITEMS = {
    "high_risk": ["helmet", "vest", "gloves", "goggles"],
    "normal": ["helmet", "vest"],
}

# Items judged "missing" when a large-enough person simply has no evidence for them (no positive box in the
# body region). Chosen from a measured comparison on the Construction-PPE test set (ppe_A, conf 0.3, GT person
# boxes; docs/RUNLOG.md 2026-10-25): explicit no_* classes catch only 6-27% of missing helmet/goggles/gloves
# (AP50 of those classes is 0.05-0.11), absence catches 100% at precision 0.67-0.89; for the vest the explicit
# no_vest class is the better trade (R 0.85 / P 0.91 vs R 1.00 / P 0.73), so it is not listed. Re-measure with
# src/eval_ppe_system.py after retraining.
INFER_ABSENT = ("helmet", "goggles", "gloves")

# LEGACY (3-class helmet_v1 model, used by benchmark_end_to_end.py only): demo_helmet.py now reads
# class names from model.names instead of this table. Roboflow's export orders classes
# alphabetically (head, helmet, person); confirm against data/helmet_roboflow/data.yaml.
CLASS_NAMES = {0: "head", 1: "helmet", 2: "person"}
NO_HELMET_CLASS_ID = 0  # "head" detected without a helmet on it
