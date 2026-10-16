"""INT8 vs fp32 accuracy comparison (2026-10-13) -- closes a gap named repeatedly throughout this
project's own documentation: "INT8's accuracy impact specifically was not measured, since it was
already rejected on latency grounds before an accuracy check would be decisive." INT8 was already
correctly rejected on latency (Step 4) -- this script exists only to stop saying "not measured"
when it's cheap to actually check, not to reopen the deployment decision.

**Honest scope, stated plainly:** a real mAP/precision/recall comparison needs the labeled
validation set (data/helmet_roboflow/), which requires ROBOFLOW_API_KEY -- not available in this
environment. This script is NOT that. It compares both models' actual predictions on the one real
benchmark image already in this repo (docs/train_helmet/val_batch0_pred.jpg) -- box count, class
assignment, and per-box confidence. That is a real, single-image signal, not a dataset-wide
accuracy number, and the output says so directly rather than implying more than it measured.

Usage:
    python src/compare_int8_accuracy.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
FP32_MODEL = ROOT / "models" / "helmet_v1_best.onnx"
INT8_MODEL = ROOT / "models" / "helmet_v1_best_int8.onnx"
BENCH_IMAGE = ROOT / "docs" / "train_helmet" / "val_batch0_pred.jpg"
CLASS_NAMES = {0: "head", 1: "helmet", 2: "person"}


def run_and_summarize(model_path, label):
    model = YOLO(str(model_path), task="detect")
    results = model.predict(str(BENCH_IMAGE), verbose=False)
    r = results[0]
    boxes = r.boxes
    n = 0 if boxes is None else len(boxes)
    by_class = {0: 0, 1: 0, 2: 0}
    confs = []
    if boxes is not None:
        for cls_id, conf in zip(boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
            by_class[int(cls_id)] = by_class.get(int(cls_id), 0) + 1
            confs.append(float(conf))
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    print(f"{label:<12} boxes={n:<4} head={by_class.get(0,0):<4} helmet={by_class.get(1,0):<4} "
          f"person={by_class.get(2,0):<4} avg_conf={avg_conf:.3f}")
    return n, by_class, avg_conf


def main():
    for p, name in [(FP32_MODEL, "fp32"), (INT8_MODEL, "INT8")]:
        if not p.exists():
            raise SystemExit(f"Missing model: {p} -- run src/export_helmet_model.py first")

    print(f"Comparing predictions on the single real benchmark image "
          f"({BENCH_IMAGE.name}) -- NOT a dataset-wide mAP comparison.\n")
    n_fp32, by_class_fp32, conf_fp32 = run_and_summarize(FP32_MODEL, "fp32")
    n_int8, by_class_int8, conf_int8 = run_and_summarize(INT8_MODEL, "INT8")

    print(f"\nBox count: fp32={n_fp32}, INT8={n_int8} "
          f"({'same' if n_fp32 == n_int8 else f'INT8 found {n_int8 - n_fp32:+d}'})")
    print(f"Average confidence: fp32={conf_fp32:.3f}, INT8={conf_int8:.3f} "
          f"(delta {conf_int8 - conf_fp32:+.3f})")

    print("\nThis is a single-image, qualitative signal -- not a substitute for a real mAP "
          "comparison on the labeled validation set (blocked here on ROBOFLOW_API_KEY not being "
          "available in this environment; data/helmet_roboflow/ was never downloaded locally). "
          "The Step 4 decision to deploy fp32 and reject INT8 was already made on latency alone "
          "and does not depend on this result -- this only removes the 'accuracy impact not "
          "measured at all' gap, it does not reopen the deployment choice.")


if __name__ == "__main__":
    main()
