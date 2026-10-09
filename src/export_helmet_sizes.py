"""Export fixed-size ONNX variants at smaller imgsz, for the Pi 4B resolution-tradeoff test
(docs/RUNLOG.md 2026-10-11).

Why separate files instead of one resizable model: `model.export(format="onnx")` without
`dynamic=True` bakes a FIXED input shape into the graph (learned the hard way -- the first version
of benchmark_pi.py tried resizing a 640-fixed ONNX file to 416 and ONNX Runtime correctly rejected
it: "Got invalid dimensions ... Got: 416 Expected: 640"). A dynamic-shape export would avoid that,
but dynamic graphs can run slower than a static one of the same size in ONNX Runtime, which would
confound "is resolution actually cheaper" with "is dynamic-shape overhead the real cost" -- so
each size gets its own static-shape export instead, for a fair comparison.

**Real bug caught here (2026-10-11), fixed, worth remembering:** Ultralytics' `.export()` always
writes to `<model_stem>.onnx` regardless of `imgsz` -- it does NOT encode the size in the output
filename. The first version of this script exported straight from `helmet_v1_best.pt` for each
size in a loop and renamed the result afterward -- which meant every export in the loop first
overwrote `models/helmet_v1_best.onnx` (the real, actively-used 640 deployment file) before being
renamed away. Running it once silently deleted the working demo's model file; caught immediately
by re-testing the demo rather than assuming the export succeeded cleanly. Fixed below by exporting
from a uniquely-named COPY of the .pt for each size, so Ultralytics' derived output name never
collides with the real `helmet_v1_best.onnx`.

Usage:
    python src/export_helmet_sizes.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PT = ROOT / "models" / "helmet_v1_best.pt"
SIZES = [416, 320]  # 640 already exists as models/helmet_v1_best.onnx -- never touch that file


def main():
    for size in SIZES:
        out_path = ROOT / "models" / f"helmet_v1_best_{size}.onnx"
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue
        # Export from a uniquely-named copy so Ultralytics' derived output filename
        # (<copy_stem>.onnx) can never collide with models/helmet_v1_best.onnx.
        temp_pt = ROOT / "models" / f"_tmp_export_{size}.pt"
        shutil.copyfile(MODEL_PT, temp_pt)
        try:
            model = YOLO(str(temp_pt))
            exported = model.export(format="onnx", imgsz=size)
            Path(exported).rename(out_path)
            print(f"Exported: {out_path}")
        finally:
            temp_pt.unlink(missing_ok=True)

    print("\nCopy these (plus the existing helmet_v1_best.onnx and helmet_v1_best_int8.onnx) "
          "to the Pi for src/benchmark_pi.py.")


if __name__ == "__main__":
    main()
