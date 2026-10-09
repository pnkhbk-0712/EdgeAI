"""Re-measure real latency ON the Raspberry Pi 4B itself (2026-10-11) -- Hung's hardware check
found 0.30 FPS running stock yolov8n.pt, 61x slower than the laptop baseline and 11x over the
<300ms/frame target (docs/RUNLOG.md 2026-10-11). That number was the raw, un-optimized PyTorch
model on a Pi with a known-underpowered 5V/2A supply -- this script re-measures with the things
that could actually close the gap, so the team isn't guessing which lever (if any) works:

  1. ONNX fp32 @ 640 (already the laptop's winning format -- Pi number may differ, not assumed)
  2. ONNX INT8 @ 640 (REJECTED on the laptop's x86 CPU -- but ARM can behave oppositely for INT8;
     the old parking project's own report flagged this as unverified for ARM, never tested)
  3. ONNX fp32 @ 416 and @ 320 (reduced input resolution -- inference cost scales roughly with
     image area, so this is a cheap, no-retraining lever). Each size is its OWN exported file,
     not one file resized at inference time -- ONNX export bakes a fixed input shape into the
     graph, confirmed the hard way when the first version of this script tried resizing the
     640-fixed file and ONNX Runtime correctly rejected it. Run src/export_helmet_sizes.py on the
     laptop first to produce the 416/320 files.

**Known real tradeoff, not just theoretical:** on the laptop, the 640 model found 16 boxes on the
benchmark image, the 416 model found 5, and the 320 model found 0 -- lower resolution is faster
but a real accuracy cost, not free. Keep that in mind reading the FPS numbers below: the fastest
setting may not be the usable one.

Run this AFTER swapping in a proper 5V/3A supply (docs/RASPI_CHECK_REPORT.md flagged the current
5V/2A as under the Pi 4B's recommendation, with the OS's own low-power warning -- rule that out
as a confound before trusting any number here as the hardware's real ceiling).

Setup on the Pi (once):
    pip install ultralytics opencv-python onnxruntime
    # copy these 4 files from the laptop (models/) to the Pi, same relative path:
    #   helmet_v1_best.onnx       (fp32 @ 640)
    #   helmet_v1_best_int8.onnx  (INT8 @ 640)
    #   helmet_v1_best_416.onnx   (fp32 @ 416 -- run src/export_helmet_sizes.py on the laptop first)
    #   helmet_v1_best_320.onnx   (fp32 @ 320 -- same)
    # and one real image to benchmark against, e.g. docs/train_helmet/val_batch0_pred.jpg

Usage (on the Pi):
    python3 benchmark_pi.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
BENCH_IMAGE = ROOT / "docs" / "train_helmet" / "val_batch0_pred.jpg"

# (label, filename, imgsz this file was exported at)
VARIANTS = [
    ("ONNX fp32", "helmet_v1_best.onnx", 640),
    ("ONNX INT8", "helmet_v1_best_int8.onnx", 640),
    ("ONNX fp32", "helmet_v1_best_416.onnx", 416),
    ("ONNX fp32", "helmet_v1_best_320.onnx", 320),
]
N_WARMUP = 3
N_RUNS = 15  # fewer than the laptop script -- a slow Pi run shouldn't take forever to benchmark


def bench(onnx_path, img, imgsz):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    resized = cv2.resize(img, (imgsz, imgsz))  # resize to THIS file's own fixed input size
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

    print(f"Benchmarking on this Pi ({N_WARMUP} warmup + {N_RUNS} timed runs per setting)...\n")
    print(f"{'Format':<14}{'imgsz':<8}{'ms/frame':<12}{'FPS':<8}{'<300ms?':<8}")

    results = []
    for name, filename, imgsz in VARIANTS:
        path = MODELS / filename
        if not path.exists():
            print(f"{name:<14}{imgsz:<8}[missing: {filename} -- skipped]")
            continue
        ms = bench(path, img, imgsz)
        passes = "YES" if ms < 300 else "no"
        print(f"{name:<14}{imgsz:<8}{ms:<12.1f}{1000/ms:<8.1f}{passes:<8}")
        results.append((name, imgsz, ms))

    if not results:
        raise SystemExit("\nNo model files found -- see the setup instructions in this script's "
                          "docstring for which files to copy from the laptop.")

    best = min(results, key=lambda r: r[2])
    print(f"\nBest: {best[0]} @ imgsz={best[1]} -> {best[2]:.1f}ms/frame "
          f"({'PASSES' if best[2] < 300 else 'still FAILS'} the <300ms target)")
    print("\nReport these numbers back -- paste the whole table, not just the best line, so the "
          "ONNX-vs-INT8 and resolution trade-offs are visible, not just the winner. Remember the "
          "320 setting found 0 boxes on the laptop's benchmark image -- fast but check it still "
          "detects anything real before treating it as the answer.")


if __name__ == "__main__":
    main()
