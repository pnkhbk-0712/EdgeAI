# Hiếu — đo mAP thật fp32 vs INT8 trên cả val set (cần ROBOFLOW_API_KEY)

File này chỉ để **thực thi** — bối cảnh đầy đủ (vì sao cần, hiện đang thiếu gì) nằm ở
`docs/team_guides/HIEU_DATA_MODEL.md`. Đọc file này là đủ để làm.

## Chạy ở đâu? — Trả lời thẳng: **máy cá nhân của bạn**, không phải Colab

Lý do cụ thể, không phải chọn ngẫu nhiên:

- 2 file model cần so sánh (`models/helmet_v1_best.onnx`, `models/helmet_v1_best_int8.onnx`)
  **đã có sẵn** trên máy bạn (nằm trong `models/` của chính repo này) — mọi lệnh Python trong
  dự án từ trước đến giờ đều chạy ở máy này.
- 2 file đó **không có trên Colab** (bị gitignore, không nằm trong git, Colab clone/pull repo sẽ
  không có chúng) — muốn chạy trên Colab thì phải upload tay 2 file này lên trước, thêm một bước
  không cần thiết.
- Máy bạn đã cài sẵn `ultralytics`/`onnxruntime` (dùng suốt từ đầu dự án), không cần cài gì thêm.
- Chạy mAP (so câu trả lời model với nhãn thật) không cần GPU — máy CPU thường chạy vẫn ra kết
  quả đúng, chỉ chậm hơn chút so với GPU Colab, không đáng kể cho việc chạy 1 lần.

**Colab chỉ cần đụng tới ở Bước 1 dưới đây** — để XEM lại key cũ đã lưu, không cần chạy gì ở đó.
Nếu không tìm lại được key cũ thì cũng không cần Colab luôn, lấy key mới thẳng từ roboflow.com.

## Bước 1 — Tìm lại hoặc lấy ROBOFLOW_API_KEY

`src/download_helmet_data.py` ghi rõ trong docstring: key này **đã từng dùng** cho dataset biển
báo của project cũ (trước khi pivot sang helmet) — nên thử tìm lại trước khi tạo key mới:

- Mở lại Colab notebook cũ của project biển báo (chỉ để xem, không chạy cell nào) → icon 🔑
  "Secrets" ở thanh bên trái → xem có `ROBOFLOW_API_KEY` đã lưu sẵn không → bấm vào, copy giá trị.
- Không tìm lại được thì lấy key mới: đăng nhập [roboflow.com](https://roboflow.com) bằng tài
  khoản nhóm đã dùng trước đó → **Settings → Roboflow API** → copy key.

Sau bước này bạn chỉ cần **chuỗi key** (dạng chữ+số) — đóng Colab lại, mọi thứ còn lại làm trên
máy cá nhân.

## Bước 2 — Set key trên máy cá nhân (PowerShell)

Mở PowerShell **ngay trong terminal bạn sẽ chạy các lệnh ở Bước 3-5** (biến này chỉ tồn tại
trong phiên terminal đang mở, đóng terminal là mất, lần sau mở lại phải set lại):

```powershell
$env:ROBOFLOW_API_KEY = "key-của-bạn"
```

Không paste key vào bất kỳ file `.py`/`.md` nào, không đưa vào commit message — chỉ set làm biến
môi trường tạm thời như trên.

## Bước 3 — Tải dataset thật (chạy trên máy cá nhân, terminal vừa set key ở Bước 2)

```powershell
cd "D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\LA4\prototype"
python src/download_helmet_data.py
```

Script tự xoá `data/helmet_roboflow/` cũ nếu có trước khi tải lại (tránh bug SDK coi thư mục có
sẵn là "đã tải rồi" rồi bỏ qua, đã note trong chính script). Chạy xong sẽ in ra đường dẫn
`data/helmet_roboflow/data.yaml` — **xác nhận file này thực sự tồn tại** trước khi qua bước sau
(mở thử bằng File Explorer hoặc `Test-Path data/helmet_roboflow/data.yaml` cũng được, phải ra
`True`).

## Bước 4 — Kiểm tra thứ tự class trong `data.yaml`

```powershell
Get-Content data/helmet_roboflow/data.yaml
```

Roboflow export theo alphabet mặc định (head, helmet, person) — đối chiếu với `CLASS_NAMES` đang
dùng trong `src/compare_int8_accuracy.py` (`{0: "head", 1: "helmet", 2: "person"}`) để chắc không
bị lệch thứ tự trước khi đọc kết quả mAP theo class ở bước sau.

## Bước 5 — Chạy mAP thật cho cả fp32 và INT8 (vẫn trên máy cá nhân)

Không cần viết script mới — Ultralytics đã có sẵn lệnh validate chuẩn, chạy được thẳng trên file
`.onnx`:

```powershell
python -m ultralytics val model=models/helmet_v1_best.onnx data=data/helmet_roboflow/data.yaml imgsz=640 project=models/mAP name=fp32
python -m ultralytics val model=models/helmet_v1_best_int8.onnx data=data/helmet_roboflow/data.yaml imgsz=640 project=models/mAP name=int8
```

`project=`/`name=` ghi kết quả vào `models/mAP/fp32/` và `models/mAP/int8/` (thay vì mặc định
`runs/detect/val*`) — thư mục này **không bị gitignore**, giống cách `docs/train_helmet/` đang
lưu kết quả train, nên commit được để cả nhóm/thầy xem lại số liệu gốc, không chỉ số đã chép tay.

Mỗi lệnh in ra bảng mAP50/mAP50-95/precision/recall **theo từng class** (head/helmet/person) và
tổng, cộng với confusion matrix lưu trong thư mục kết quả tương ứng. Chạy cả 2 lệnh, không chỉ 1
— mục đích là so sánh fp32 vs INT8, không phải chỉ xem 1 model. Mỗi lệnh có thể chạy vài phút
trên CPU, tuỳ tốc độ máy — đây là bình thường, không phải lỗi.

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
signal" mà `compare_int8_accuracy.py` đã nói rõ là không đủ để kết luận, và trỏ tới
`models/mAP/fp32/` / `models/mAP/int8/` làm nguồn số liệu gốc thay vì chép tay lại hết.

## Nếu thật sự muốn chạy trên Colab thay vì máy cá nhân

Không khuyến nghị (thêm việc không cần thiết), nhưng nếu máy cá nhân quá yếu hoặc có lý do khác,
cần làm thêm 2 bước Bước 3-5 ở trên không có:

1. Trên Colab: `!git clone -b helmet-safety-pivot https://github.com/pnkhbk-0712/EdgeAI.git`
   rồi `%cd EdgeAI/prototype`.
2. Upload tay 2 file `models/helmet_v1_best.onnx` và `models/helmet_v1_best_int8.onnx` từ máy cá
   nhân lên đúng thư mục `models/` trên Colab (vì chúng bị gitignore, `git clone` không có) —
   dùng panel Files bên trái Colab, kéo-thả vào đúng thư mục.
3. Từ đó làm lại Bước 1-2 (set key bằng `google.colab.userdata`, xem lại hướng dẫn cũ trong
   `docs/team_guides/HIEU_DATA_MODEL.md` nếu cần cú pháp) rồi Bước 3-5 y hệt, chỉ đổi `python ...`
   thành `!python ...` (Colab cell cần dấu `!` trước lệnh shell).

## Tài liệu liên quan

- `docs/team_guides/HIEU_DATA_MODEL.md` — bối cảnh đầy đủ, các gap khác đang mở của Data & Model.
- `src/download_helmet_data.py`, `src/compare_int8_accuracy.py` — code liên quan, đã có sẵn.
- `docs/RUNLOG.md`, `docs/CHANGELOG.md` — nơi ghi kết quả thật sau khi đo xong.
