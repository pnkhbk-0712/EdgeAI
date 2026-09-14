"""Deterministic unit test for the zone-overlap + dwell-time rule (src/zone.py).

The public sample clip used in src/demo.py (car-detection.mp4) turns out to
contain only vehicles briefly driving through frame -- none actually stops
long enough to violate, so that end-to-end run correctly reports zero
violations. This test proves the trigger logic itself is correct using
synthetic, controlled coordinates instead of relying on footage that happens
to contain a real violation.

Run: python test_zone.py   (from the tests/ folder, or `python -m tests.test_zone`)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from zone import RestrictedZone  # noqa: E402

ZONE = [(0, 0), (100, 0), (100, 100), (0, 100)]
INSIDE = (50, 50)
OUTSIDE = (500, 500)
DWELL_FRAMES = 5


def test_no_violation_before_dwell_threshold():
    zone = RestrictedZone(ZONE, dwell_frames=DWELL_FRAMES)
    for _ in range(DWELL_FRAMES - 1):
        inside, is_violation, newly_flagged = zone.update(track_id=1, cx=INSIDE[0], cy=INSIDE[1])
        assert inside is True
        assert is_violation is False
        assert newly_flagged is False
    print("PASS: no violation fires before the dwell threshold is reached")


def test_violation_fires_exactly_once_at_threshold():
    zone = RestrictedZone(ZONE, dwell_frames=DWELL_FRAMES)
    flags = []
    for _ in range(DWELL_FRAMES + 3):
        _, _, newly_flagged = zone.update(track_id=1, cx=INSIDE[0], cy=INSIDE[1])
        flags.append(newly_flagged)
    assert flags.count(True) == 1, f"expected exactly one new-violation flag, got {flags.count(True)}"
    assert flags[DWELL_FRAMES - 1] is True, "violation should fire on the Nth frame, not later"
    print("PASS: violation fires exactly once, on the correct frame")


def test_pass_through_never_violates():
    """A vehicle passing through the zone for fewer frames than the dwell
    threshold (Report Sec. VIII.A test case: 'vehicle passing through
    without stopping') must never be flagged."""
    zone = RestrictedZone(ZONE, dwell_frames=DWELL_FRAMES)
    for _ in range(DWELL_FRAMES - 2):
        inside, is_violation, newly_flagged = zone.update(track_id=2, cx=INSIDE[0], cy=INSIDE[1])
        assert newly_flagged is False
    for _ in range(3):
        zone.update(track_id=2, cx=OUTSIDE[0], cy=OUTSIDE[1])
    assert 2 not in zone.already_flagged
    print("PASS: a pass-through vehicle is never flagged (matches Sec. VIII.A)")


def test_leaving_the_zone_resets_the_dwell_counter():
    zone = RestrictedZone(ZONE, dwell_frames=DWELL_FRAMES)
    zone.update(track_id=3, cx=INSIDE[0], cy=INSIDE[1])
    zone.update(track_id=3, cx=INSIDE[0], cy=INSIDE[1])
    zone.update(track_id=3, cx=OUTSIDE[0], cy=OUTSIDE[1])  # leaves early
    for _ in range(DWELL_FRAMES - 1):
        _, _, newly_flagged = zone.update(track_id=3, cx=INSIDE[0], cy=INSIDE[1])
        assert newly_flagged is False
    print("PASS: leaving the zone resets the dwell counter")


def test_two_vehicles_tracked_independently():
    zone = RestrictedZone(ZONE, dwell_frames=DWELL_FRAMES)
    for _ in range(DWELL_FRAMES):
        zone.update(track_id=10, cx=INSIDE[0], cy=INSIDE[1])
    for _ in range(DWELL_FRAMES - 1):
        zone.update(track_id=11, cx=INSIDE[0], cy=INSIDE[1])
    assert 10 in zone.already_flagged
    assert 11 not in zone.already_flagged
    print("PASS: multiple vehicles are tracked independently (Sec. VIII.A multi-vehicle case)")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} zone-logic tests passed.")
