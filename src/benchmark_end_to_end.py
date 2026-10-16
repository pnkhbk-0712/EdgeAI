"""End-to-end pipeline latency benchmark (2026-10-13) -- closes the gap repeatedly flagged in
Step 4 of the midterm critique: every prior benchmark (export_helmet_model.py, benchmark_pi.py)
measures the raw ONNX session.run() call only. The actual deployed pipeline in demo_helmet.py is
heavier than that standalone number: it goes through Ultralytics' own track() call (letterboxing,
NMS, ByteTrack bookkeeping) plus this project's own per-frame Python work (per-track
class-smoothing vote, two RestrictedZone.update() calls) -- none of which a standalone ONNX
benchmark can see. This script times the REAL call path, importing zone.py and
helmet_zone_configs.py unchanged rather than re-implementing their logic, so the measured
overhead is the actual code's overhead.

Real finding while building this (2026-10-13): three independent runs of the EXISTING standalone
benchmark (export_helmet_model.py) on this same laptop, same image, same model, produced ONNX fp32
numbers of 23.3ms, 30.8ms, and 29.7ms/frame -- roughly +-30% run-to-run variance from background
system load, not from anything in the model or code. The absolute millisecond figure is therefore
noisy on a shared laptop CPU; comparing standalone-vs-full-pipeline against an old constant from a
DIFFERENT run would confound that run-to-run noise with the real pipeline overhead being measured
here. This script avoids that by measuring BOTH standalone session.run() and the full pipeline in
the SAME process invocation, back to back, so system load is shared between the two numbers and
the delta between them is attributable to the pipeline code, not to noise.

Caveat stated plainly, the same way every other laptop number in this project is labelled: this
still runs on the laptop CPU, not the Raspberry Pi 4B. This closes the "standalone model vs full
pipeline" gap, not the "laptop vs Pi" gap, which stays open and tracked separately
(docs/PI_BENCHMARK_GUIDE.md, src/benchmark_pi.py).

Usage:
    python src/benchmark_end_to_end.py
"""
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zone import RestrictedZone
from helmet_zone_configs import (
    NO_HELMET_CLASS_ID,
    HIGH_RISK_ZONE_POLYGON_FRAC, NORMAL_ZONE_POLYGON_FRAC, scale_polygon,
    HIGH_RISK_DWELL_SECONDS, NORMAL_DWELL_SECONDS,
)

# Matches demo_helmet.py's CLASS_SMOOTHING_WINDOW (defined there, not in helmet_zone_configs.py) --
# duplicated here rather than imported, since demo_helmet.py's own module-level code (argparse
# setup, etc.) runs on import and isn't meant to be imported from another script.
CLASS_SMOOTHING_WINDOW = 5

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "helmet_v1_best.onnx"
BENCH_IMAGE = ROOT / "docs" / "train_helmet" / "val_batch0_pred.jpg"
N_WARMUP = 5
N_RUNS = 30
ASSUMED_FPS = 15.0


def bench_standalone_onnx(img):
    """Same measurement export_helmet_model.py makes -- raw onnxruntime session.run() only,
    no Ultralytics wrapper, no tracker, no zone logic. Run here, in the same process and right
    before the full-pipeline measurement, so both numbers share the same background system load."""
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    resized = cv2.resize(img, (640, 640))
    blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    for _ in range(N_WARMUP):
        sess.run(None, {input_name: blob})
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        sess.run(None, {input_name: blob})
    elapsed = time.perf_counter() - t0
    return elapsed / N_RUNS * 1000  # ms/frame


def main():
    img = cv2.imread(str(BENCH_IMAGE))
    if img is None:
        raise SystemExit(f"Could not load benchmark image: {BENCH_IMAGE}")
    h, w = img.shape[:2]

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH} -- run src/export_helmet_model.py first")

    print("1/2 Standalone ONNX session.run() (same-run baseline, for a fair comparison)...")
    standalone_ms = bench_standalone_onnx(img)

    print("2/2 Full pipeline (Ultralytics track() + class-smoothing + 2x zone-check)...\n")
    model = YOLO(str(MODEL_PATH), task="detect")

    high_risk_zone = RestrictedZone(scale_polygon(HIGH_RISK_ZONE_POLYGON_FRAC, w, h),
                                     dwell_frames=max(1, int(HIGH_RISK_DWELL_SECONDS * ASSUMED_FPS)))
    normal_zone = RestrictedZone(scale_polygon(NORMAL_ZONE_POLYGON_FRAC, w, h),
                                  dwell_frames=max(1, int(NORMAL_DWELL_SECONDS * ASSUMED_FPS)))
    track_class_history = defaultdict(lambda: deque(maxlen=CLASS_SMOOTHING_WINDOW))

    def process_one_frame():
        results = model.track(source=img, persist=True, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return
        for box in r.boxes:
            track_id = int(box.id.item()) if box.id is not None else -1
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            is_no_helmet_raw = cls_id == NO_HELMET_CLASS_ID
            hist = track_class_history[track_id]
            hist.append(is_no_helmet_raw)
            is_no_helmet = sum(hist) >= len(hist) / 2
            if is_no_helmet:
                high_risk_zone.update(track_id, cx, cy)
                normal_zone.update(track_id, cx, cy)

    for _ in range(N_WARMUP):
        process_one_frame()

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        process_one_frame()
    elapsed = time.perf_counter() - t0
    ms_per_frame = elapsed / N_RUNS * 1000
    fps = 1000 / ms_per_frame if ms_per_frame > 0 else float("inf")

    overhead_ms = ms_per_frame - standalone_ms
    overhead_pct = overhead_ms / standalone_ms * 100 if standalone_ms else 0.0

    print(f"{'Stage':<48}{'ms/frame':<12}{'FPS':<8}")
    print(f"{'Standalone ONNX session.run() (this run)':<48}{standalone_ms:<12.1f}{1000/standalone_ms:<8.1f}")
    print(f"{'Full pipeline: track() + smoothing + zones':<48}{ms_per_frame:<12.1f}{fps:<8.1f}")
    print(f"\nOverhead from track()/tracker + this project's own Python logic, not visible in the "
          f"standalone benchmark: +{overhead_ms:.1f} ms/frame ({overhead_pct:+.0f}%)")
    print(f"\n{'PASSES' if ms_per_frame < 300 else 'FAILS'} the <300ms/frame target -- on this laptop CPU.")
    print("Caveat: still a laptop CPU number, NOT the Raspberry Pi 4B -- that measurement is still "
          "pending (docs/PI_BENCHMARK_GUIDE.md). This only closes the standalone-vs-full-pipeline gap, "
          "and the absolute ms figure itself varies run to run (see module docstring) -- trust the "
          "relative overhead, not the single-run millisecond value, until this is re-measured a few times.")


if __name__ == "__main__":
    main()
