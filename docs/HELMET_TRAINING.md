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

## Training further (2026-10-09, corrected same day -- see the warning below)

`helmet_v1`'s first 30-epoch run finished with mAP50=0.659 overall (head 0.963 / helmet 0.982 are
fine; the weak aggregate traces to the `person` class, see `docs/RUNLOG.md` 2026-10-09 and
`docs/PROJECT_REPORT.md` Step 3 -- `person` isn't used by the zone-violation logic, so it isn't
the reason to train further). The reason to add epochs: **`results.png` shows mAP50/mAP50-95
still rising and val loss still falling at epoch 30 -- the run was stopped before it converged,
not after.** More epochs is the correct, cheap lever here -- not more data, not a from-scratch
retrain.

**`resume=True` does NOT work here -- tried it, it silently trained garbage instead.** Ultralytics
strips the optimizer/epoch state from `last.pt`/`best.pt` once a run finishes *normally* (ours
did, all 30/30 epochs) -- `resume` only works on a checkpoint from a run that was *interrupted*.
Pointing `resume=True` at a completed run's checkpoint prints a warning
(`not a resumable training checkpoint ... Starting new training instead`) and silently falls back
to a **brand-new** `model.train()` call -- which, because no `data=` was given, defaulted to
`coco8.yaml` (Ultralytics' 4-image smoke-test set) and trained a throwaway 80-class COCO model for
50 epochs into `/content/runs/detect/train/`. No damage done (the real `helmet_v1` files on Drive
were untouched, and the wasted run only took ~29s since coco8 is tiny) -- but it means nothing
useful came out of that run, and `resume=True` should not be used to continue a completed run.

**What actually works:** load the completed run's `best.pt` as a *pretrained starting point* for a
fresh `model.train()` call with every argument given explicitly (`data=`, `project=`, `name=`) --
not a true LR-schedule-preserving resume, but a standard, well-understood "continue fine-tuning"
pattern that doesn't depend on stripped checkpoint state.

```python
# 1. Mount Drive (fresh runtime each session -- remount if needed)
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 2. Need the repo again this time -- data= below is a path relative to it (resume=True didn't
# need this because it never actually used the real dataset; this corrected version does).
%cd /content
!rm -rf /content/EdgeAI_helmet
!git clone -b helmet-safety-pivot https://github.com/pnkhbk-0712/EdgeAI.git /content/EdgeAI_helmet
%cd /content/EdgeAI_helmet
!pip install -q ultralytics roboflow
```

```python
# 3. Re-download the dataset into this fresh runtime (it lived only in the previous runtime's
# /content, not on Drive -- the model checkpoint is what's on Drive, not the dataset files).
import os
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
!python src/download_helmet_data.py
```

```python
# 4. Continue training from the completed run's best.pt, 20 more epochs, into a NEW run folder
# (helmet_v1b) so the original helmet_v1 results aren't overwritten -- keeps a rollback point
# per docs/CHANGELOG.md.
from ultralytics import YOLO
model = YOLO("/content/drive/MyDrive/EdgeAI_runs/helmet_v1/weights/best.pt")
results = model.train(
    data="data/helmet_roboflow/data.yaml",
    epochs=20, imgsz=640, batch=16, device=0,
    project="/content/drive/MyDrive/EdgeAI_runs", name="helmet_v1b",
)
```

```python
# 5. Download the result
from google.colab import files
files.download('/content/drive/MyDrive/EdgeAI_runs/helmet_v1b/weights/best.pt')
```

Integration once downloaded: save as `models/helmet_v1b_best.pt` (don't overwrite
`helmet_v1_best.pt` -- keep both until v1b is confirmed at least as good, per
`docs/CHANGELOG.md`'s rollback rule), copy the new `results.csv`/`results.png`/confusion matrices
to `docs/train_helmet/`, and compare per-class AP50 and the head/helmet confusion rate against
v1's numbers (head 0.963, helmet 0.982, 17% head->helmet confusion) before deciding which model
the report and the live demo actually use.
