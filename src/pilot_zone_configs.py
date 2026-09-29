"""Per-clip zone configs for the 4 real pilot videos (2026-09-29/30).

**Confirmed against full video playback (2026-09-30).** Each polygon was checked against 10
frames spread across the whole ~5-minute clip, not just the single still frame used to draft it
-- see docs/zone_check/*_fullcheck_grid.jpg and docs/RUNLOG.md 2026-09-30. All 4 clips have a
genuinely fixed camera and a zone that holds for the full duration.

Sign group determines the dwell threshold (see docs/ZONE_LABEL_DEFINITIONS.md Q2):
  - "stop"  (cam dung -- no-stopping signs, single or double diagonal slash): a much shorter
    dwell than the existing 60s "park" default, since a real stopping violation is typically
    over in seconds, not a minute. 5s is a starting proposal, not a measured value -- there is
    no real dataset yet to tune this against.
  - "park"  (cam do -- no-parking signs, no diagonal, just vertical bars for day parity): keeps
    the existing, already-justified 60s dwell used elsewhere in this project.

All 4 clips are 1920x1080 source @ 29.97fps (cv2 correctly auto-rotates to display as portrait
1080x1920 -- confirmed empirically, see docs/RUNLOG.md 2026-09-29 sanity check).
"""
from pathlib import Path

VIDEO_DIR = Path(r"D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\Video")
FPS = 29.97

PILOT_CLIPS = {
    "nostop_nopark": {
        "video_path": VIDEO_DIR / "Biển cấm dừng và đỗ xe.mov",
        "sign_group": "stop",  # double-diagonal "X" -- cam dung VA do (both prohibited)
        "dwell_seconds": 5.0,
        "active_hours": None,  # no time plate seen in sampled frames -> assume all-day
        "active_days": None,
        # DRAFT polygon: sidewalk row of ~5 real parked motorbikes near the "Ép Kính" shop,
        # visible bottom-right of the reference frame (nostop_nopark_001140_t0038.03s.jpg).
        # (First attempt at this polygon covered the wrong area -- the shop facade and the
        # camera rig's own mirror, not an actual parking spot -- caught by rendering the
        # polygon back onto the frame before committing; see docs/RUNLOG.md 2026-09-30.)
        "zone_polygon": [(740, 830), (1080, 830), (1080, 1030), (740, 1030)],
    },
    "nostop": {
        "video_path": VIDEO_DIR / "Biển cấm dừng xe.mov",
        "sign_group": "stop",  # single diagonal slash -- cam dung only
        "dwell_seconds": 5.0,
        "active_hours": None,
        "active_days": None,
        # DRAFT polygon: sidewalk strip under the shop awning, where 2 real motorbikes are
        # parked in the reference frame (nostop_001140_t0038.03s.jpg).
        "zone_polygon": [(600, 550), (1080, 550), (1080, 800), (600, 800)],
    },
    "nopark_even": {
        "video_path": VIDEO_DIR / "Biển cấm đỗ xe vào ngày chẵn.mov",
        "sign_group": "park",  # 2 vertical bars, no diagonal -- P.131c
        "dwell_seconds": 60.0,
        "active_hours": (6, 22),  # read directly off the sign's time plate in-frame
        "active_days": "even",
        # Sidewalk strip where real vehicles (car + motorbikes) actually sit, right under the
        # "PHO" shop awning. CORRECTED 2026-09-30: the original polygon (y=680-850) was too high
        # -- it covered the tree canopy above the sidewalk, not the vehicles themselves, which
        # sit at y~830-1030. Caught when the user reviewed the full-video verification grid and
        # noticed the red zone band sat above the actual cars/motorbikes in every frame; a
        # coordinate-grid re-check confirmed it. See docs/RUNLOG.md 2026-09-30.
        #
        # KNOWN SOURCE-FOOTAGE QUIRK (2026-09-30): the camera framing itself jumps to a wider
        # angle between t=148s and t=149s (confirmed real, not a seek artifact -- checked every
        # 1s across the transition, see docs/zone_check/nopark_even_framing_shift_t135-150.jpg).
        # Likely the phone was bumped/adjusted mid-recording. This single polygon was spot-
        # checked against both the pre-149s and post-149s framing (see
        # docs/zone_check/nopark_even_early_segment_zonecheck.jpg) and still lands on the real
        # parked motorbikes in both, so it's kept as one polygon rather than split in two -- but
        # it's an approximation pre-149s, not pixel-exact like it is post-149s. If precision here
        # ever matters, trim/label the clip starting at 149s instead of 0s.
        "zone_polygon": [(0, 830), (980, 830), (980, 1030), (0, 1030)],
    },
    "nopark_odd": {
        "video_path": VIDEO_DIR / "Biển cấm đỗ xe vào ngày lẻ.mov",
        "sign_group": "park",  # 1 vertical bar, no diagonal -- P.131b
        "dwell_seconds": 60.0,
        "active_hours": (6, 22),  # read directly off the sign's time plate in-frame
        "active_days": "odd",
        # DRAFT polygon: sidewalk strip right of the sign, where a real motorbike is parked in
        # the reference frame (nopark_odd_001140_t0038.03s.jpg).
        "zone_polygon": [(550, 850), (1080, 850), (1080, 1300), (550, 1300)],
    },
}


def dwell_frames_for(tag: str, fps: float = FPS) -> int:
    return max(1, round(PILOT_CLIPS[tag]["dwell_seconds"] * fps))


if __name__ == "__main__":
    for tag, cfg in PILOT_CLIPS.items():
        print(f"{tag}: group={cfg['sign_group']} dwell={cfg['dwell_seconds']}s "
              f"({dwell_frames_for(tag)} frames) hours={cfg['active_hours']} "
              f"days={cfg['active_days']} video_exists={cfg['video_path'].exists()}")
