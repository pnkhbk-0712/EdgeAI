"""Risk-zone config for the live classroom demo (2026-10-03).

**DRAFT placeholders -- not yet calibrated against the real physical rig.** The team plans to
build a foam-board scaffolding mockup + figures with/without helmets for the live demo. Until
that rig exists and a reference frame from the actual demo webcam is available, these polygons
are just a reasonable starting guess for a generic 1280x720 webcam frame:
  - HIGH_RISK_ZONE: upper portion of frame, where the foam scaffolding mockup is meant to sit
    (farther from camera / "elevated").
  - NORMAL_ZONE: lower portion, ground-level area in front of it.

**Before the real demo**, follow the same verify-before-trusting workflow used throughout the
parking project: grab a real frame from the actual demo webcam with the actual rig in place, draw
these polygons on it, and visually confirm they land where the rig/figures actually are -- then
update the coordinates here. Do not trust these numbers unverified.

Dwell thresholds are intentionally very short compared to the parking project (that one waited
60s for a "parked" car; this is a live demo, not hours of footage) -- the point is immediate
feedback when a presenter moves a no-helmet figure into a zone, not debouncing a slow real-world
process.
"""

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# DRAFT -- recalibrate against a real demo-webcam frame before relying on this.
HIGH_RISK_ZONE_POLYGON = [(300, 0), (980, 0), (980, 340), (300, 340)]
# DRAFT -- recalibrate against a real demo-webcam frame before relying on this.
NORMAL_ZONE_POLYGON = [(0, 400), (1280, 400), (1280, 720), (0, 720)]

# High-risk zone: near-instant alert (this is a live demo, not a real multi-minute dwell case).
HIGH_RISK_DWELL_SECONDS = 1.0
# Normal zone: a little debounce so a momentary head-turn/occlusion doesn't false-trigger.
NORMAL_DWELL_SECONDS = 3.0

# Roboflow's YOLOv8 export orders classes alphabetically by default: head, helmet, person.
# CONFIRM this against the actual downloaded data/helmet_roboflow/data.yaml before trusting it --
# see the note printed by src/download_helmet_data.py.
CLASS_NAMES = {0: "head", 1: "helmet", 2: "person"}
NO_HELMET_CLASS_ID = 0  # "head" detected without a helmet on it
