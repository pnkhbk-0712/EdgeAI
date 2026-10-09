# Training the helmet-detection model (Colab)

Same workflow as the old project's `smartcity.ipynb` (GPU runtime + Drive persistence), just
pointed at the new dataset. See `docs/PIVOT_HELMET_DETECTION.md` for why this project exists and
`src/download_helmet_data.py` for dataset provenance (Hard Hat Workers Dataset, CC0, 7,035 images).

## Colab cells

```python
# 1. Mount Drive (same as before, for run persistence across disconnects)
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 2. Clone this branch (NOT main -- the helmet work lives on helmet-safety-pivot).
# NOTE: the GitHub repo root IS this prototype/ folder's contents directly -- there is no
# nested LA4/prototype path inside the clone, only locally on disk. Don't add it here.
%cd /content
!rm -rf /content/EdgeAI_helmet
!git clone -b helmet-safety-pivot https://github.com/pnkhbk-0712/EdgeAI.git /content/EdgeAI_helmet
%cd /content/EdgeAI_helmet
!pip install -q ultralytics roboflow
```

```python
# 3. Download the dataset (needs ROBOFLOW_API_KEY in Colab Secrets -- same key used before)
import os
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
!python src/download_helmet_data.py
```

```python
# 4. Confirm class order before training -- Roboflow usually exports alphabetically
# (head, helmet, person), but DO NOT assume -- read it back:
!cat data/helmet_roboflow/data.yaml
```

If the printed class order doesn't match `CLASS_NAMES` in `src/helmet_zone_configs.py`
(`{0: "head", 1: "helmet", 2: "person"}`), fix that dict before running the live demo --
everything else (training, export) uses the dataset's own data.yaml and doesn't care about this
mapping, but `demo_helmet.py` hardcodes it for display/zone-check purposes.

```python
# 5. Train -- same CLI pattern as edgeai_v1-3 (30 epochs, batch 16, imgsz 640)
!yolo train model=yolov8n.pt data=data/helmet_roboflow/data.yaml \
    epochs=30 imgsz=640 batch=16 device=0 \
    project=/content/drive/MyDrive/EdgeAI_runs name=helmet_v1
```

```python
# 6. Download the result (best.pt + the run folder with results.csv, confusion matrix, etc.)
from google.colab import files
files.download('/content/drive/MyDrive/EdgeAI_runs/helmet_v1/weights/best.pt')
```

After downloading, same integration steps used for edgeai_v2 on the old project:
1. Copy `best.pt` to `models/helmet_v1_best.pt` (gitignored, like all `.pt` files).
2. Copy the small run artifacts (`args.yaml`, `results.csv`, `confusion_matrix*.png`,
   `results.png`, `labels.jpg`) to `docs/train_helmet/` and commit those (they're small and are
   the evidence trail for the Model Selection & Training section of the report).
3. Sanity-check against a few real webcam frames before trusting it for the live demo, same
   "verify before trusting" habit used throughout -- don't assume validation-set mAP transfers
   directly to the classroom's actual lighting/camera/distance.

## Resuming training (2026-10-09)

`helmet_v1`'s first 30-epoch run finished with mAP50=0.659 overall (head 0.963 / helmet 0.982 are
fine; the weak aggregate traces to the `person` class, see `docs/RUNLOG.md` 2026-10-09 and
`docs/PROJECT_REPORT.md` Step 3 -- `person` isn't used by the zone-violation logic, so it isn't
the reason to resume). The reason to resume: **`results.png` shows mAP50/mAP50-95 still rising and
val loss still falling at epoch 30 -- the run was stopped before it converged, not after.** More
epochs on the exact same data is the correct, cheap lever here -- not more data, not a from-scratch
retrain.

```python
# 1. Mount Drive (fresh runtime each session -- remount if needed)
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 2. Install ultralytics (resume only needs this -- no repo clone, no re-download required,
# the checkpoint and dataset reference already live on Drive from the first run)
!pip install -q ultralytics
```

```python
# 3. Resume from the last checkpoint, extended to 50 total epochs (20 more than the first run).
# NOTE: resume=True alone would stop immediately, since the checkpoint already reached the
# original epochs=30 target -- passing a larger `epochs` here is what tells Ultralytics to keep
# going instead of treating the run as already complete.
from ultralytics import YOLO
model = YOLO("/content/drive/MyDrive/EdgeAI_runs/helmet_v1/weights/last.pt")
results = model.train(resume=True, epochs=50)
```

```python
# 4. Download the updated result
from google.colab import files
files.download('/content/drive/MyDrive/EdgeAI_runs/helmet_v1/weights/best.pt')
```

Same integration steps as above once downloaded -- overwrite `models/helmet_v1_best.pt` (keep the
old one momentarily to compare, per `docs/CHANGELOG.md`'s rollback rule, don't just clobber it),
refresh `docs/train_helmet/` with the new `results.csv`/`results.png`/confusion matrices, and
re-check the per-class AP50 and the head/helmet confusion rate specifically against the first
run's numbers (head 0.963, helmet 0.982, 17% head->helmet confusion) before updating the report.
