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

# Roboflow's YOLOv8 export orders classes alphabetically by default: head, helmet, person.
# CONFIRM this against the actual downloaded data/helmet_roboflow/data.yaml before trusting it --
# see the note printed by src/download_helmet_data.py.
CLASS_NAMES = {0: "head", 1: "helmet", 2: "person"}
NO_HELMET_CLASS_ID = 0  # "head" detected without a helmet on it
