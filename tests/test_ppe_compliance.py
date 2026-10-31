"""Unit tests for person-centric PPE assessment (src/ppe_compliance.py) -- synthetic boxes only.

Run: python tests/test_ppe_compliance.py   (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ppe_compliance import ItemSmoother, assess_person, norm_name, split_dets, supported_items  # noqa: E402
from zone import RestrictedZone  # noqa: E402

FRAME_H = 720
PERSON = (100, 100, 200, 400)  # 100x300 px: head band y 100..205, torso 130..340
ALL = ["helmet", "vest", "gloves", "goggles"]


def det(name, cx, cy, conf=0.9, half=15):
    return (norm_name(name), (cx - half, cy - half, cx + half, cy + half), conf)


def test_aliases_and_supported_items():
    assert norm_name("Safety-Vest") == "vest" and norm_name("head") == "no_helmet"
    assert supported_items({0: "head", 1: "helmet", 2: "person"}.values()) == ["helmet"]
    assert set(supported_items(["helmet", "vest", "gloves", "goggles", "person"])) == set(ALL)


def test_item_goes_to_one_person_only():
    a, b = (100, 100, 300, 400), (250, 100, 450, 400)  # overlapping boxes, glove at x=140 is A's
    glove = det("gloves", 140, 300)
    da, db = split_dets([a, b], [glove])
    assert len(da) == 1 and db == []
    assert assess_person(b, db, ["gloves"], FRAME_H)["gloves"][0] == "unknown"


def test_present_and_missing_helmet():
    ok = assess_person(PERSON, [det("helmet", 150, 120)], ["helmet"], FRAME_H)
    assert ok["helmet"][0] == "present"
    bad = assess_person(PERSON, [det("no_helmet", 150, 120)], ["helmet"], FRAME_H)
    assert bad["helmet"][0] == "missing"


def test_item_outside_person_box_is_ignored():
    far = assess_person(PERSON, [det("helmet", 600, 120)], ["helmet"], FRAME_H)
    assert far["helmet"][0] == "unknown"


def test_vest_absence_is_missing_but_gloves_goggles_stay_unknown():
    r = assess_person(PERSON, [], ALL, FRAME_H, infer_absent=("vest",))
    assert r["vest"][0] == "missing"
    assert r["gloves"][0] == "unknown" and r["goggles"][0] == "unknown" and r["helmet"][0] == "unknown"


def test_explicit_no_vest_beats_inference():
    r = assess_person(PERSON, [det("no_vest", 150, 250, 0.7)], ["vest"], FRAME_H)
    assert r["vest"] == ("missing", 0.7)  # explicit evidence keeps its confidence


def test_infer_absent_is_configurable():
    r = assess_person(PERSON, [], ["helmet", "gloves"], FRAME_H, infer_absent=("helmet",))
    assert r["helmet"][0] == "missing" and r["gloves"][0] == "unknown"


def test_small_person_is_not_assessed():
    tiny = (100, 100, 120, 130)  # 30 px tall < 12% of 720
    r = assess_person(tiny, [], ALL, FRAME_H)
    assert all(v[0] == "unknown" for v in r.values())


def test_conflicting_evidence_takes_higher_confidence():
    r = assess_person(PERSON, [det("helmet", 150, 120, 0.55), det("no_helmet", 150, 125, 0.8)], ["helmet"], FRAME_H)
    assert r["helmet"][0] == "missing"


def test_smoother_majority_and_tie_leans_missing():
    s = ItemSmoother(window=5)
    for _ in range(3):
        assert s.update((1, "helmet"), False) is False
    assert s.update((1, "helmet"), True) is False  # 1 of 4 missing
    assert s.update((1, "helmet"), True) is False  # 2 of 5
    assert s.update((1, "helmet"), True) is True   # window slides: 3 of 5
    t = ItemSmoother(window=4)
    t.update("k", False); t.update("k", True)
    assert t.update("k", True) is True and t.update("k", False) is True  # 2 of 4 tie -> missing


def test_zone_dwell_is_tracked_per_track_and_item():
    zone = RestrictedZone([(0, 0), (300, 0), (300, 500), (0, 500)], dwell_frames=3)
    for _ in range(3):
        r_helmet = zone.update((7, "helmet"), 150, 250)
    r_vest = zone.update((7, "vest"), 150, 250)
    assert r_helmet[2] is True   # helmet crossed its dwell and is newly flagged
    assert r_vest[2] is False    # vest of the same person has only 1 frame of dwell


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
