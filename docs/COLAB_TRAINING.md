# Training on Colab — paste these cells in order (Hieu)

Reuses the exact scripts already tested locally (`src/convert_ua_detrac.py`,
`src/add_motorbike_data.py`, `src/oversample_motorcycle.py`, `src/audit_dataset.py`) rather
than rewriting the logic in the notebook — one tested version, no drift between local and
Colab.

## Cell 1 — clone the repo

```python
!git clone https://github.com/pnkhbk-0712/EdgeAI.git
%cd EdgeAI
!pip install -q ultralytics kagglehub
```

## Cell 2 — Kaggle auth (use Colab Secrets, not a pasted token)

Click the **key icon** on the left sidebar → **Add new secret** → name it `KAGGLE_API_TOKEN`,
paste your own token from kaggle.com/settings → API → Create New Token → toggle **Notebook
access** on. Then:

```python
import os
from google.colab import userdata
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
os.environ["KAGGLEHUB_CACHE"] = "/content/kagglehub_cache"  # Colab's own disk, disposable
```

(Never paste the raw token directly into a code cell — Secrets keeps it out of the notebook
file itself, so it's not exposed if the notebook is ever shared.)

## Cell 3 — download UA-DETRAC and set the path the scripts expect

```python
import kagglehub
ua_path = kagglehub.dataset_download("bratjay/ua-detrac-orig")
os.environ["UA_DETRAC_ROOT"] = ua_path
print(ua_path)
```

## Cell 4 — convert, merge motorcycle data, oversample, audit (in this order)

```python
!python src/convert_ua_detrac.py
!python src/add_motorbike_data.py
!python src/oversample_motorcycle.py
!python src/audit_dataset.py
```

The last line should print `corrupt_images: 0`, `orphan_images: 0`, `orphan_labels: 0`,
`malformed_lines: 0`, `out_of_range: 0`, `zero_size_box: 0`, `bad_class_id: 0`. If any of
those are non-zero, stop and read `docs/audit_report.txt` before training on it — training on
bad data just wastes GPU time.

## Cell 5 — train

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.train(
    data="data/ua_detrac_yolo/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,       # the T4 GPU
    project="runs",
    name="edgeai_v1",
)
```

30 epochs is a reasonable first real run, not a final number — raise it once this works.

## Cell 6 — check per-class results, not just the overall number

Ultralytics prints a per-class table automatically after training/validation. **Find the
`motorcycle` row specifically** — car:motorcycle was 154:1 before oversampling (now ~22:1 in
training), so overall mAP can look fine while motorcycle mAP is quietly poor. If motorcycle
mAP is much worse than car/bus/truck, that's expected at this data volume (699 boxes vs.
50k+ for car) — note it honestly in the report rather than only reporting the overall number.

## Cell 7 — download the trained weights to hand off to Hoang

```python
from google.colab import files
files.download("runs/edgeai_v1/weights/best.pt")
```

Send `best.pt` to Hoang for the optimization/export step (Report Sec. VII.C-D).
