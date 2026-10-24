# Huấn luyện model PPE 4 món (runbook)

Dữ liệu: `data/ppe_merged/` (sinh bằng `python src/build_ppe_dataset.py`; xem `docs/DATASET_AUDIT_PPE.md`).
Lớp: person, helmet, no_helmet, vest, no_vest, gloves, no_gloves, goggles, no_goggles.
Hai cấu hình so sánh (cùng val/test thật của Construction-PPE, 113/111 ảnh, đã loại ảnh gần giống train):

| Tên | `data=` | Train |
|---|---|---|
| A | `data_cppe.yaml` | chỉ Construction-PPE (1.132 ảnh worksite) |
| C | `data.yaml` | Construction-PPE + PPEv5 (6.772 ảnh) |

Tiêu chí chọn: AP50/recall **từng lớp** trên test; C chỉ thắng nếu vest/helmet không giảm so với A.

## Cách 1 — GPU máy này (RTX 4060 Laptop 8 GB)
venv hiện tại cài torch **CPU** (`torch 2.6.0+cpu`), nên cần venv CUDA mới trên D: (khoảng 3 GB tải, cần đồng ý trước).
```
python -m venv D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\_tools\venv_cuda
set PIP_CACHE_DIR=D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\_tools\pip_cache
D:\...\_tools\venv_cuda\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
D:\...\_tools\venv_cuda\Scripts\pip install ultralytics
set YOLO_CONFIG_DIR=D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\_tools\yolo_cfg
yolo detect train data=data/ppe_merged/data_cppe.yaml model=models/yolov8n.pt imgsz=640 epochs=80 batch=16 patience=30 device=0 project=D:/DOANMINHHIEU/STUDIES/Ky9/01_Ai/_tools/yolo_runs name=ppe_A
yolo detect train data=data/ppe_merged/data.yaml       model=models/yolov8n.pt imgsz=640 epochs=60 batch=16 patience=30 device=0 project=D:/DOANMINHHIEU/STUDIES/Ky9/01_Ai/_tools/yolo_runs name=ppe_C
```
Trên CPU thì ~40 s cho 113 ảnh @640 mỗi epoch (ước từ lần chạy thử 1 epoch @320), tức vài giờ cho A — không nên.

## Cách 2 — Colab (như model mũ trước, `docs/COLAB_TRAINING.md`)
Nén `data/ppe_merged/` (khoảng 395 MB), upload Drive, giải nén `/content/ppe_merged`, rồi:
```
sed -i "s#^path:.*#path: /content/ppe_merged#" /content/ppe_merged/data*.yaml
!yolo detect train data=/content/ppe_merged/data_cppe.yaml model=yolov8n.pt imgsz=640 epochs=80 batch=16 patience=30 name=ppe_A
!yolo detect train data=/content/ppe_merged/data.yaml model=yolov8n.pt imgsz=640 epochs=60 batch=16 patience=30 name=ppe_C
```
(`path:` phải tuyệt đối — Ultralytics resolve đường dẫn tương đối theo thư mục datasets của nó, không theo yaml.)

## Đo và lưu
```
yolo detect val model=<best.pt> data=<yaml> split=test imgsz=640     # lặp lại với imgsz=480 và 416
```
Ghi bảng AP50/recall từng lớp (A vs C, 640/480/416) vào `docs/RUNLOG.md`; chép `best.pt` về `models/ppe_v1_best.pt`, xuất ONNX fp32
(`src/export_helmet_sizes.py` chỉnh tên model), rồi nhờ Hùng chạy `benchmark_pi.py` và `demo_helmet.py --model` trên Pi.
