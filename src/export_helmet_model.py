"""Export + quantize + benchmark the helmet model (Report Step 4, 2026-10-10).

Same methodology as the parking project's ONNX/INT8 comparison (docs/IMPLEMENTATION_REPORT.md):
export to ONNX, dynamic-quantize to INT8, then measure real per-frame CPU latency for all three
forms (PyTorch .pt, ONNX fp32, ONNX INT8) on a real image -- not just file size. The parking
project found INT8 was *slower* than fp32 on a generic x86 CPU despite being 73% smaller; this
script exists to re-measure that on the helmet model rather than assume the same result repeats.

Usage:
    python src/export_helmet_model.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PT = ROOT / "models" / "helmet_v1_best.pt"
ONNX_FP32 = ROOT / "models" / "helmet_v1_best.onnx"
ONNX_INT8 = ROOT / "models" / "helmet_v1_best_int8.onnx"
BENCH_IMAGE = ROOT / "docs" / "train_helmet" / "val_batch0_pred.jpg"
REPORT_OUT = ROOT / "docs" / "train_helmet" / "export_benchmark.md"

IMGSZ = 640
N_WARMUP = 5
N_RUNS = 30


def export_onnx():
    if ONNX_FP32.exists():
        print(f"[skip] {ONNX_FP32.name} already exists")
        return
    model = YOLO(str(MODEL_PT))
    exported = model.export(format="onnx", imgsz=IMGSZ)
    Path(exported).rename(ONNX_FP32)
    print(f"Exported: {ONNX_FP32}")


def quantize_int8():
    if ONNX_INT8.exists():
        print(f"[skip] {ONNX_INT8.name} already exists")
        return
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(ONNX_FP32), str(ONNX_INT8), weight_type=QuantType.QUInt8)
    print(f"Quantized: {ONNX_INT8}")


def load_bench_image():
    img = cv2.imread(str(BENCH_IMAGE))
    if img is None:
        raise SystemExit(f"Could not load benchmark image: {BENCH_IMAGE}")
    img = cv2.resize(img, (IMGSZ, IMGSZ))
    return img


def bench_pytorch(img):
    model = YOLO(str(MODEL_PT))
    for _ in range(N_WARMUP):
        model.predict(img, imgsz=IMGSZ, verbose=False)
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        model.predict(img, imgsz=IMGSZ, verbose=False)
    elapsed = time.perf_counter() - t0
    return elapsed / N_RUNS * 1000  # ms/frame


def bench_onnx(onnx_path, img):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    for _ in range(N_WARMUP):
        sess.run(None, {input_name: blob})
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        sess.run(None, {input_name: blob})
    elapsed = time.perf_counter() - t0
    return elapsed / N_RUNS * 1000  # ms/frame


def main():
    print("1/3 Exporting ONNX (fp32)...")
    export_onnx()
    print("2/3 Quantizing to INT8...")
    quantize_int8()

    print("3/3 Benchmarking on CPU (real image, "
          f"{N_WARMUP} warmup + {N_RUNS} timed runs each)...")
    img = load_bench_image()

    ms_pt = bench_pytorch(img)
    ms_onnx = bench_onnx(ONNX_FP32, img)
    ms_int8 = bench_onnx(ONNX_INT8, img)

    size_pt = MODEL_PT.stat().st_size / 1e6
    size_onnx = ONNX_FP32.stat().st_size / 1e6
    size_int8 = ONNX_INT8.stat().st_size / 1e6

    rows = [
        ("PyTorch (.pt)", size_pt, ms_pt),
        ("ONNX (fp32)", size_onnx, ms_onnx),
        ("ONNX (INT8, dynamic)", size_int8, ms_int8),
    ]

    print("\n--- Results ---")
    print(f"{'Format':<24}{'Size (MB)':<12}{'ms/frame':<12}{'FPS':<8}")
    for name, size, ms in rows:
        print(f"{name:<24}{size:<12.2f}{ms:<12.1f}{1000/ms:<8.1f}")

    onnx_speedup = (ms_pt - ms_onnx) / ms_pt * 100
    int8_vs_onnx = (ms_onnx - ms_int8) / ms_onnx * 100
    size_reduction = (size_onnx - size_int8) / size_onnx * 100

    verdict = (
        f"ONNX fp32 is {'faster' if onnx_speedup > 0 else 'slower'} than PyTorch "
        f"by {abs(onnx_speedup):.1f}%. "
        f"INT8 is {'faster' if int8_vs_onnx > 0 else 'slower'} than ONNX fp32 "
        f"by {abs(int8_vs_onnx):.1f}%, despite being {size_reduction:.1f}% smaller."
    )
    print("\n" + verdict)

    report = f"""# Helmet model export/quantization benchmark (2026-10-10)

CPU-only, same methodology as `docs/IMPLEMENTATION_REPORT.md`'s parking-project comparison --
real per-frame latency, not just file size, measured on `{BENCH_IMAGE.name}`
({N_WARMUP} warmup + {N_RUNS} timed runs per format).

| Format | Size (MB) | ms/frame | FPS |
|---|---|---|---|
| PyTorch (.pt) | {size_pt:.2f} | {ms_pt:.1f} | {1000/ms_pt:.1f} |
| ONNX (fp32) | {size_onnx:.2f} | {ms_onnx:.1f} | {1000/ms_onnx:.1f} |
| ONNX (INT8, dynamic) | {size_int8:.2f} | {ms_int8:.1f} | {1000/ms_int8:.1f} |

**{verdict}**

Acceptance check against Step 3/4's target (<300ms/frame on the Pi 4B -- this is a laptop CPU
number, not yet the Pi itself, see Step 5's "testing on target device" gap):
{'PASSES' if ms_onnx < 300 else 'FAILS'} the latency target on ONNX fp32 at {ms_onnx:.1f}ms/frame.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()
