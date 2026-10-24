"""Person-centric PPE compliance (replaces the single-item head/helmet check in demo_helmet.py).

Per frame, each tracked person is assessed for every required item the model can detect:
  present  - a positive detection (helmet/vest/gloves/goggles) lies in that item's body region
  missing  - an explicit "no_*" detection lies there, or the item is in infer_absent and a large enough person has no evidence for it
  unknown  - nothing to go on (gloves/goggles without a "no_*" call, or a person too small to judge)
Only "missing" feeds the per-(track, item) smoothing and the dwell zones; "unknown" never raises an
alert. Class names come from the model (model.names), normalised and aliased below, so the old
3-class helmet model (head/helmet/person) still works for the helmet item.
"""
from collections import deque

# item -> (positive class, explicit-negative class or None, (min, max) vertical band as a fraction
# of the person box height, measured from its top; helmets sit slightly above a tight person box).
ITEMS = {
    "helmet": ("helmet", "no_helmet", (-0.25, 0.35)),
    "goggles": ("goggles", "no_goggles", (-0.25, 0.35)),
    "vest": ("vest", "no_vest", (0.10, 0.80)),
    "gloves": ("gloves", "no_gloves", (0.20, 1.10)),
}
ALIASES = {
    "head": "no_helmet", "hardhat": "helmet", "hard_hat": "helmet", "no_hardhat": "no_helmet",
    "safety_vest": "vest", "glasses": "goggles", "no_glasses": "no_goggles", "no_goggle": "no_goggles",
    "glove": "gloves", "no_glove": "no_gloves",
}
H_MARGIN = 0.10          # horizontal slack around the person box, fraction of its width
MIN_PERSON_H_FRAC = 0.12  # persons shorter than this fraction of the frame are not assessed


def norm_name(name):
    n = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(n, n)


def supported_items(model_names):
    """Items the model can actually detect (needs the positive class) -- never judge the rest."""
    have = {norm_name(n) for n in model_names}
    return [item for item, (pos, _, _) in ITEMS.items() if pos in have]


def _in_region(person, det_box, band):
    px1, py1, px2, py2 = person
    pw, ph = px2 - px1, py2 - py1
    cx, cy = (det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2
    ry = (cy - py1) / ph
    return px1 - H_MARGIN * pw <= cx <= px2 + H_MARGIN * pw and band[0] <= ry <= band[1]


def assess_person(person_box, dets, items, frame_h, infer_absent=()):
    """dets: [(normalised_name, xyxy, conf)] for all non-person detections in the frame.
    Returns {item: (status, conf)}; conf is the evidence confidence (None when inferred/unknown).
    infer_absent: items judged missing when a large enough person has no evidence for them at all --
    only sensible for items the model detects reliably (see docs/RUNLOG.md 2026-10-25 for the measured choice)."""
    if person_box[3] - person_box[1] < MIN_PERSON_H_FRAC * frame_h:
        return {item: ("unknown", None) for item in items}
    out = {}
    for item in items:
        pos, neg, band = ITEMS[item]
        evid = [(c, n) for n, box, c in dets if n in (pos, neg) and _in_region(person_box, box, band)]
        if evid:
            conf, name = max(evid)  # strongest evidence wins when positive and negative conflict
            out[item] = ("present" if name == pos else "missing", conf)
        else:
            out[item] = ("missing", None) if item in infer_absent else ("unknown", None)
    return out


class ItemSmoother:
    """Majority vote over each (track_id, item)'s last `window` assessed frames; ties lean missing
    (safer), same rule demo_helmet.py used for the helmet alone."""

    def __init__(self, window=5):
        self.window = window
        self.hist = {}

    def update(self, key, is_missing):
        h = self.hist.setdefault(key, deque(maxlen=self.window))
        h.append(bool(is_missing))
        return sum(h) >= len(h) / 2
