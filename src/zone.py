"""Restricted-zone overlap logic (Report Sec. V / VII.D).

A vehicle is flagged as an illegal-parking violation once its detection box's
center point stays inside the restricted polygon for at least DWELL_FRAMES
consecutive frames -- this is the "remains inside that polygon beyond a short
threshold" rule described in the report.

Answers locked in from docs/ZONE_LABEL_DEFINITIONS.md (Hieu, 2026-09-15):
  Q1 containment rule -> kept: bbox CENTER point (no change).
  Q2 dwell threshold   -> changed 2s -> 60s (see DWELL_SECONDS in demo.py).
  Q3 sign schedule     -> not yet known (pilot site not surveyed): active_hours=None,
                          active_days=None for now, i.e. restriction applies at all
                          times/days until real sign data comes back from filming.
  Q4 partial occlusion -> counts as a violation if a box is still detected (no
                          special-casing needed -- this is already the default
                          behavior, since the model's own box is trusted as-is).
  Q4 red-light stop beside the zone -> should NOT count, but this is NOT something
                          the dwell-time rule can distinguish from real parking on
                          its own: a long red-light cycle can easily exceed 60s.
                          The real fix is at zone-drawing time, not in code -- see
                          `active_hours`/`active_days` are schedule-only; there is no
                          "is this actually a traffic queue" check here.  Whoever
                          draws the polygon (Hoang, at deployment) must draw it tight
                          around the restricted parking strip itself, and keep it off
                          the traffic lane where vehicles legitimately queue for a
                          light. Flag this explicitly in Report Sec. VII/VIII as a
                          known limitation, not a solved problem.
"""
from datetime import datetime

import cv2
import numpy as np


class RestrictedZone:
    def __init__(self, polygon, dwell_frames, active_hours=None, active_days=None):
        """polygon: list of (x, y) points. dwell_frames: frames before a stationary
        vehicle counts as a violation (e.g. fps * seconds).

        active_hours: optional (start_hour, end_hour) e.g. (6, 21) -- the zone only
            enforces within this window; None means the restriction applies all day.
        active_days: optional "odd" | "even" -- the zone only enforces on odd/even
            calendar dates (VN "ngay le/ngay chan" signs); None means every day.
        """
        self.polygon = np.array(polygon, dtype=np.int32)
        self.dwell_frames = dwell_frames
        self.active_hours = active_hours
        self.active_days = active_days
        self.track_dwell = {}      # track_id -> consecutive frames spent inside the zone
        self.already_flagged = set()  # track_ids already reported as a violation

    def contains(self, cx, cy):
        return cv2.pointPolygonTest(self.polygon, (float(cx), float(cy)), False) >= 0

    def is_active(self, now=None):
        """Whether the restriction is currently in force, per the real sign's
        conditions (Design Note: signs are often time- or odd/even-day-conditional).
        Always True while active_hours/active_days are both None (unconditional sign,
        or conditions not yet known from the pilot site)."""
        now = now or datetime.now()
        if self.active_hours is not None:
            start_hour, end_hour = self.active_hours
            if not (start_hour <= now.hour < end_hour):
                return False
        if self.active_days == "odd" and now.day % 2 == 0:
            return False
        if self.active_days == "even" and now.day % 2 != 0:
            return False
        return True

    def update(self, track_id, cx, cy, now=None):
        """Call once per frame per tracked vehicle. Returns (inside, is_violation_now,
        newly_flagged); newly_flagged is True the single frame a violation newly
        crosses the dwell threshold (so the caller emits one event)."""
        if not self.is_active(now):
            # Outside the sign's own restricted hours/days -- never a violation,
            # and don't let dwell time accumulate while inactive.
            self.track_dwell[track_id] = 0
            return False, False, False

        inside = self.contains(cx, cy)
        if inside:
            self.track_dwell[track_id] = self.track_dwell.get(track_id, 0) + 1
        else:
            self.track_dwell[track_id] = 0

        is_violation_now = self.track_dwell[track_id] >= self.dwell_frames
        newly_flagged = is_violation_now and track_id not in self.already_flagged
        if newly_flagged:
            self.already_flagged.add(track_id)
        return inside, is_violation_now, newly_flagged

    def draw(self, frame, color=(0, 215, 255), thickness=2):
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=thickness)
        return frame
