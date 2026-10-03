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
# 2. Clone this branch (NOT main -- the helmet work lives on helmet-safety-pivot)
%cd /content
!rm -rf /content/EdgeAI
!git clone -b helmet-safety-pivot https://github.com/pnkhbk-0712/EdgeAI.git
%cd EdgeAI/LA4/prototype
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
