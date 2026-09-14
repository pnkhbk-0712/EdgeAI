"""Restricted-zone overlap logic (Report Sec. V / VII.D).

A vehicle is flagged as an illegal-parking violation once its detection box's
center point stays inside the restricted polygon for at least DWELL_FRAMES
consecutive frames -- this is the "remains inside that polygon beyond a short
threshold" rule described in the report.
"""
import cv2
import numpy as np


class RestrictedZone:
    def __init__(self, polygon, dwell_frames):
        """polygon: list of (x, y) points. dwell_frames: frames before a stationary
        vehicle counts as a violation (e.g. fps * seconds)."""
        self.polygon = np.array(polygon, dtype=np.int32)
        self.dwell_frames = dwell_frames
        self.track_dwell = {}      # track_id -> consecutive frames spent inside the zone
        self.already_flagged = set()  # track_ids already reported as a violation

    def contains(self, cx, cy):
        return cv2.pointPolygonTest(self.polygon, (float(cx), float(cy)), False) >= 0

    def update(self, track_id, cx, cy):
        """Call once per frame per tracked vehicle. Returns True the single frame
        a violation newly crosses the dwell threshold (so the caller emits one event)."""
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
