"""Extract frames from real pilot-site videos for CVAT labeling (2026-09-29).

Real footage finally exists -- 4 fixed-camera clips, one per VN restricted-parking sign type
(see docs/ZONE_LABEL_DEFINITIONS.md Q3 and docs/RUNLOG.md 2026-09-29 for the real
active_hours/active_days read directly off the sign in each clip). This script turns each
~5-minute clip into a manageable, labelable set of still frames -- not every frame, since
consecutive video frames are near-duplicates and would waste annotation effort for no benefit
(Report/Proposal Sec. VI.C already commits to a fixed sampling rate for exactly this reason).

Sampling rate defaults to 1 frame every SAMPLE_INTERVAL_SEC seconds, chosen to land in the
500-1000-sample range the course's own Step 2 lecture note gives as a reasonable mini-dataset
size -- not every-Nth-frame by frame count, which would depend on each clip's exact fps.

Output is organized per clip (data/pilot_frames/<tag>/), because each clip needs its own zone
polygon and its own active_hours/active_days schedule -- keeping them separate makes it obvious
which CVAT task/config a given frame belongs to, rather than one mixed pool.
"""
import cv2
from pathlib import Path

VIDEO_DIR = Path(r"D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\Video")
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "pilot_frames"

# (source filename, short tag for output folder + filename prefix)
# Tag reflects the filename as given; docs/RUNLOG.md 2026-09-29 has the actual sign identified
# visually per clip (worth double-checking "nostop" against P.131a/P.130 conventions before
# finalizing zone_config, since the filename and the sign glyph did not obviously match on
# a first look for that one clip).
CLIPS = [
    ("Biển cấm dừng và đỗ xe.mov", "nostop_nopark"),
    ("Biển cấm dừng xe.mov", "nostop"),
    ("Biển cấm đỗ xe vào ngày chẵn.mov", "nopark_even"),
    ("Biển cấm đỗ xe vào ngày lẻ.mov", "nopark_odd"),
]

SAMPLE_INTERVAL_SEC = 2.0  # ~150 frames per 5-min clip, ~600 total across 4 clips


def extract(video_path: Path, tag: str):
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if any(out_dir.iterdir()):
        print(f"[{tag}] already has frames, skipping (delete {out_dir} to force a re-extract)")
        return len(list(out_dir.iterdir()))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"SKIP (could not open): {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(fps * SAMPLE_INTERVAL_SEC))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            t_sec = frame_idx / fps
            out_name = f"{tag}_{frame_idx:06d}_t{t_sec:07.2f}s.jpg"
            cv2.imwrite(str(out_dir / out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        frame_idx += 1
    cap.release()

    print(f"[{tag}] {video_path.name}: {total_frames} source frames @ {fps:.1f} fps "
          f"-> {saved} frames saved every {SAMPLE_INTERVAL_SEC}s to {out_dir}")
    return saved


def main():
    total = 0
    for filename, tag in CLIPS:
        video_path = VIDEO_DIR / filename
        if not video_path.exists():
            print(f"MISSING (skipped): {video_path}")
            continue
        total += extract(video_path, tag)

    print(f"\nTotal frames extracted: {total}")
    print(f"Output root: {OUT_ROOT}")
    print("Next: zip each per-clip folder and create one CVAT task per clip (each needs its "
          "own zone polygon + active_hours/active_days), or import folders directly if CVAT "
          "is self-hosted with local file access.")


if __name__ == "__main__":
    main()
