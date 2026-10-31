"""INT8 post-training quantization of the PPE ONNX models: size, laptop CPU latency, test-split accuracy (2026-10-30).

Two variants per resolution: dynamic (same method as the helmet model, src/export_helmet_model.py) and static QDQ
calibrated on CALIB_N train images (what the Step 4 lecture calls PTQ). Accuracy = `yolo val` on the 111-image test split.
Latency is the laptop CPU, NOT the Pi: copy the *_int8_*.onnx files over and time them with src/benchmark_pi.py to decide.
Usage: python src/quantize_ppe.py [--sizes 640 416]
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static
from ultralytics import YOLO

MODELS = Path("models")
DATA = "data/ppe_merged/data_sampled.yaml"
CALIB_N = 100


class Calib(CalibrationDataReader):
    def __init__(self, size, name):
        files = sorted(Path("data/ppe_merged/train/images").glob("cppe_*"))[:CALIB_N]
        self.it = iter([{name: self.load(f, size)} for f in files])

    @staticmethod
    def load(f, s):
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        r = s / max(h, w)
        im = cv2.resize(im, (round(w * r), round(h * r)))
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[:im.shape[0], :im.shape[1]] = im
        return canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255

    def get_next(self):
        return next(self.it, None)


def latency_ms(path, size, runs=50):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    x = np.random.rand(1, 3, size, size).astype(np.float32)
    for _ in range(5):
        s.run(None, {inp.name: x})
    t = []
    for _ in range(runs):
        a = time.perf_counter()
        s.run(None, {inp.name: x})
        t.append((time.perf_counter() - a) * 1000)
    return float(np.median(t))


def accuracy(path, size):
    r = YOLO(str(path), task="detect").val(data=DATA, split="test", imgsz=size, device="cpu", batch=1, plots=False, verbose=False)
    return r.box.map50, r.box.map50_per_class if hasattr(r.box, "map50_per_class") else None, r


ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[640, 416])
args = ap.parse_args()
print("size variant            MB   lat_ms   mAP50   gloves  goggles helmet  vest   (names from model)")
for size in args.sizes:
    fp32 = MODELS / f"ppe_v1_best_{size}.onnx"
    dyn = MODELS / f"ppe_v1_best_int8dyn_{size}.onnx"
    sta = MODELS / f"ppe_v1_best_int8static_{size}.onnx"
    name = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    if not dyn.exists():
        quantize_dynamic(str(fp32), str(dyn), weight_type=QuantType.QUInt8)
    if not sta.exists():
        pre = MODELS / f"_pre_{size}.onnx"   # constant folding: calibration crashes on scalar constants otherwise
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
        quantize_static(str(pre), str(sta), Calib(size, name), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=False)
    for label, p in (("fp32", fp32), ("int8-dynamic", dyn), ("int8-static", sta)):
        m50, _, r = accuracy(p, size)
        pc = dict(zip(r.names.values(), r.box.ap50))  # ap_class_index order may skip classes
        idx = {r.names[i]: a for i, a in zip(r.box.ap_class_index, r.box.ap50)}
        print(f"{size:<4} {label:<14} {p.stat().st_size / 1e6:6.2f} {latency_ms(p, size):8.1f}  {m50:.3f}  "
              f"{idx.get('gloves', 0):.3f}   {idx.get('goggles', 0):.3f}   {idx.get('helmet', 0):.3f}  {idx.get('vest', 0):.3f}", flush=True)
