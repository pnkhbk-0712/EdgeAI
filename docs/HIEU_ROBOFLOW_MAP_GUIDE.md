# Hiếu — đo mAP thật fp32 vs INT8 trên cả val set (cần ROBOFLOW_API_KEY)

File này chỉ để **thực thi** — bối cảnh đầy đủ (vì sao cần, hiện đang thiếu gì) nằm ở
`docs/team_guides/HIEU_DATA_MODEL.md`. Đọc file này là đủ để làm.

**Chạy toàn bộ các bước dưới đây ở máy bạn tự kiểm soát được** (Colab hoặc máy cá nhân) —
không chạy được qua Claude, vì bước 1-2 cần key tài khoản Roboflow của bạn, không nên chia sẻ key
đó trong chat hay commit vào git.

## Bước 1 — Tìm lại hoặc lấy ROBOFLOW_API_KEY

`src/download_helmet_data.py` ghi rõ trong docstring: key này **đã từng dùng** cho dataset biển
báo của project cũ (trước khi pivot sang helmet) — nên thử tìm lại trước khi tạo key mới:

- Mở lại Colab notebook cũ của project biển báo → icon 🔑 "Secrets" bên trái → xem có
  `ROBOFLOW_API_KEY` đã lưu sẵn không.
- Không tìm lại được thì lấy key mới: đăng nhập [roboflow.com](https://roboflow.com) bằng tài
  khoản nhóm đã dùng trước đó → **Settings → Roboflow API** → copy key.

## Bước 2 — Set key (an toàn, không hardcode vào file)

**Trên Colab** (khuyến nghị — dùng Secrets thay vì gõ thẳng vào cell):

```python
from google.colab import userdata
import os
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
```

**Trên máy Windows cá nhân** (PowerShell, chỉ tồn tại trong phiên terminal đang mở):

```powershell
$env:ROBOFLOW_API_KEY = "key-của-bạn"
```

Không paste key vào file `.py`/`.md`/commit message nào — chỉ set làm biến môi trường.

## Bước 3 — Tải dataset thật

```bash
cd "D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\LA4\prototype"
python src/download_helmet_data.py
```

Script tự xoá `data/helmet_roboflow/` cũ nếu có trước khi tải lại (tránh bug SDK coi thư mục có
sẵn là "đã tải rồi" rồi bỏ qua, đã note trong chính script). Chạy xong sẽ in ra đường dẫn
`data/helmet_roboflow/data.yaml` — **xác nhận file này thực sự tồn tại** trước khi qua bước sau.

## Bước 4 — Kiểm tra thứ tự class trong `data.yaml`

```bash
cat data/helmet_roboflow/data.yaml
```

Roboflow export theo alphabet mặc định (head, helmet, person) — đối chiếu với `CLASS_NAMES` đang
dùng trong `src/compare_int8_accuracy.py` (`{0: "head", 1: "helmet", 2: "person"}`) để chắc không
bị lệch thứ tự trước khi đọc kết quả mAP theo class ở bước sau.

## Bước 5 — Chạy mAP thật cho cả fp32 và INT8

Không cần viết script mới — Ultralytics đã có sẵn lệnh validate chuẩn, chạy được thẳng trên file
`.onnx`:

```bash
python -m ultralytics val model=models/helmet_v1_best.onnx data=data/helmet_roboflow/data.yaml imgsz=640
python -m ultralytics val model=models/helmet_v1_best_int8.onnx data=data/helmet_roboflow/data.yaml imgsz=640
```

Mỗi lệnh in ra bảng mAP50/mAP50-95/precision/recall **theo từng class** (head/helmet/person) và
tổng, cộng với confusion matrix lưu trong `runs/detect/val*/`. Chạy cả 2 lệnh, không chỉ 1 — mục
đích là so sánh fp32 vs INT8, không phải chỉ xem 1 model.

## Bước 6 — Đọc kết quả, so sánh

So 2 bảng kết quả theo đúng 3 câu hỏi đang để ngỏ trong `docs/RUNLOG.md`/`CHANGELOG.md`:

1. mAP50 tổng + theo từng class của fp32 so với INT8 — lệch bao nhiêu % thật (không phải 1 ảnh
   như `compare_int8_accuracy.py` đã làm trước đó)?
2. Class `head` (gap 17% nhận nhầm thành `helmet`, đã biết từ confusion matrix cũ) — mAP riêng
   class này có khác biệt đáng kể giữa fp32/INT8 không?
3. Class `person` (AP50=0.034, biết từ lúc train v1) — số liệu này có đổi khi đo lại trên INT8
   không, hay vẫn thấp y hệt (xác nhận gap này là do data/model, không phải do quantization)?

## Gửi lại kết quả — dùng mẫu này

```
=== Kết quả mAP thật fp32 vs INT8 (val set đầy đủ) ===

fp32  — mAP50 tổng: ___   mAP50-95 tổng: ___
        head: ___   helmet: ___   person: ___

INT8  — mAP50 tổng: ___   mAP50-95 tổng: ___
        head: ___   helmet: ___   person: ___

Nhận xét: (INT8 có làm tệ hơn đáng kể không, đặc biệt ở class head/person?)
```

Sau khi có số thật, cập nhật `docs/RUNLOG.md` (thêm mục mới, không sửa đè mục cũ) và
`docs/CHANGELOG.md` — ghi rõ đây là **mAP thật trên cả dataset**, thay thế phần "single-image
signal" mà `compare_int8_accuracy.py` đã nói rõ là không đủ để kết luận.

## Tài liệu liên quan

- `docs/team_guides/HIEU_DATA_MODEL.md` — bối cảnh đầy đủ, các gap khác đang mở của Data & Model.
- `src/download_helmet_data.py`, `src/compare_int8_accuracy.py` — code liên quan, đã có sẵn.
- `docs/RUNLOG.md`, `docs/CHANGELOG.md` — nơi ghi kết quả thật sau khi đo xong.
