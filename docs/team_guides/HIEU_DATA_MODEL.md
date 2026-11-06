# Hiếu — Data & Model — việc cần làm tiếp (cập nhật 2026-11-03)

## MỚI (2026-11-03) — Thầy bắt làm lại Step 3 & 4; chờ quyết định đo YOLO11n

Commit `84be42d` đã **push** lên `helmet-safety-pivot` (báo cáo Tuần 11/12 v3–v9, tài liệu giải thích, sửa kính).
- **Step 3 phải là "chọn model trước, train sau"**: đặt tiêu chí → so sánh ứng viên → nói vì sao chọn. **Step 4** phải so sánh các cách tối ưu kèm lý do chọn/không. Khung mới: `docs/CHON_MODEL_STEP3_STEP4.md` (6 ứng viên: YOLOv8n/s, YOLO11n/s, EfficientDet-Lite0, SSD-MobileNetV2; số COCO là của nhà sản xuất, chưa phải nhóm đo).
- **Rủi ro lớn nhất:** YOLO11n trên giấy nhẹ và chính xác hơn YOLOv8n (2,6 M vs 3,2 M tham số; mAP COCO 39,5 vs 37,3) mà nhóm chưa thử → thầy dễ hỏi "sao không YOLO11n". YOLOv8s ước lượng 335–680 ms trên Pi (chưa đo).
- **Cần quyết định:** (a) đo YOLO11n, (b) đo cả YOLO11n + YOLOv8s, (c) không đo, dùng bảng ước lượng ghi rõ "ước lượng". Nếu đo: tải `yolo11n.pt`/`yolov8s.pt` (cần đồng ý), train cấu hình C′ 60 epoch `plots=True` (~22 phút/model với v8n), rồi nhờ Hùng đo Pi.
- **Số đã có cho Step 4:** 416 vs 640 (găng 0.722→0.664, kính 0.772→0.641); INT8 dynamic nhỏ 3.7× nhưng chậm hơn (21→33 ms), INT8 static sập (mAP 0.000 @640, 0.168 @416); SH17 (config C'') **không cải thiện** găng/kính (kính 0.772→0.666) nên vẫn dùng C′.
- **Sửa code:** `ppe_compliance.py` nới dải kính lên 60% chiều cao khung người (cận mặt bị báo sai "NO GOGGLES"). **Chưa chạy lại trên webcam.**
- **Chưa làm:** sửa báo cáo v9 + `PRESENTATION_TUAN12.md` theo khung Step 3/4 mới (chờ quyết định ở trên).

## MỚI (2026-10-25) — Pivot sang hệ thống PPE 4 món; model `ppe_v1` đã train (commit `9d5f681`, chưa push)

Thầy (tuần 11): mũ chỉ là cấu phần nhỏ, màu mũ không có ý nghĩa → làm đủ mũ + áo phản quang + găng tay + kính. Phần màu/vai trò mũ chuyển phụ lục.
- **Dữ liệu** (`data/ppe_merged/`, 9 lớp): Construction-PPE (AGPL-3.0) + PPEv5 v16 (CC BY 4.0). Sổ nguồn + giấy phép: `docs/DATASET_AUDIT_PPE.md`. Val/test chỉ lấy từ Construction-PPE, đã bỏ ảnh gần giống train (dHash).
- **Model** `models/ppe_v1_best.pt` (cấu hình C': 1132 ảnh Construction-PPE + 2500 ảnh PPEv5 cân bằng lớp). mAP50 test (111 ảnh, nhỏ/nhiễu) **0.559**; A (chỉ Construction-PPE) 0.512. Theo lớp: helmet 0.91, vest 0.86, gloves 0.72, goggles 0.77, person 0.79; **lớp `no_*` yếu** (no_helmet 0.20, no_gloves 0.12, no_goggles 0.19, no_vest 0.46).
- **Vì vậy** "thiếu món" được suy từ *không thấy món đó* trên người đủ lớn (helmet/goggles/gloves; `INFER_ABSENT` trong `helmet_zone_configs.py`), riêng vest dùng lớp `no_vest`. Đo cấp hệ thống (`src/eval_ppe_system.py`): precision 0.72–0.91, recall 0.97–1.00; tập test có tỉ lệ thiếu cao hơn thực tế nên precision thật sẽ thấp hơn.
- Độ phân giải gần như không ảnh hưởng (mAP50 0.559 @640 → 0.539 @416). ONNX: `ppe_v1_best_640.onnx`, `ppe_v1_best_416.onnx`. Laptop CPU 25/10 ms — **chưa có số Pi**.
- **Chưa kiểm chứng:** `none` → `no_vest` là suy luận; găng/kính ở xa; cảnh đông người; chưa có khung hình rig thật.
- Việc kế tiếp của Hiếu: viết lại báo cáo Word Step 1/2 quanh PPE 4 món (đưa màu/vai trò xuống phụ lục; thêm biểu đồ AP theo lớp, sổ nguồn/giấy phép gồm AGPL-3.0 và ghi nguồn CC BY) + script 5 phút mới; sau khi Hoàng có khung hình rig → đo lại trên tập rig. Tuỳ chọn: thêm dữ liệu (SH17 là CC BY-NC-SA, RamcoMills PPEs, ảnh có giấy phép rõ).
- Runbook: `docs/PPE_TRAINING.md`. Log: `docs/RUNLOG.md` 2026-10-25.

> Các mục bên dưới là của model mũ cũ `helmet_v1` (3 lớp) — vẫn đúng cho model đó nhưng **không còn là model chính**.

Đây là bản tổng hợp từ `RUNLOG.md`/`CHANGELOG.md` dành riêng cho phần Data & Model, để không phải
lật lại cả file log mỗi lần hỏi "còn thiếu gì".

## Trạng thái thật hiện tại

- Model đang dùng: `models/helmet_v1_best.pt` (YOLOv8n, 30/30 epoch, Hard Hat Workers Dataset).
  AP50 thật theo class: `head`=0.963, `helmet`=0.982, `person`=0.034 (mất cân bằng 31:1 so với
  `helmet`, phát hiện sau khi train xong, không audit trước). `person` không được zone logic dùng
  tới, nên không chặn demo.
- Confusion matrix cho thấy **17% các trường hợp `head` thật bị nhận nhầm thành `helmet`** — tức
  model có thể báo "có mũ" trong khi thực ra không. Vấn đề an toàn thật, không chỉ là số liệu xấu.
- `helmet_v1b` (train thêm 20 epoch từ v1) — đã test, **không cải thiện thật**, không dùng.
- ✅ **mAP thật trên cả dataset (xong 2026-10-17):** chạy `yolo val` trên 1413 ảnh val set thật
  cho cả fp32 và INT8. mAP50 all: 0.660 (fp32) vs 0.659 (INT8) — **INT8 gần như không mất gì về
  accuracy**, xác nhận quyết định Step 4 (chọn fp32, loại INT8) đúng vì lý do tốc độ, không phải
  đánh đổi accuracy lấy tốc độ. Kết quả đầy đủ ở `models/mAP/fp32/` và `models/mAP/int8/`.
- ✅ **Audit dataset thật (xong 2026-10-17):** `src/audit_dataset.py` đã sửa để chạy trên
  `data/helmet_roboflow/` (trước đó hardcode dataset UA-DETRAC cũ). Chạy thật trên cả 7,035 ảnh
  (4916 train + 1413 valid + 706 test): **0 lỗi** (không ảnh hỏng, không orphan, không nhãn sai
  định dạng, không toạ độ lệch phạm vi, không box 0-kích-thước, không class id sai). Dataset nguồn
  sạch, không cần sửa gì thêm.
- ✅ **Đã quyết định (2026-10-17): gap `head→helmet` 17% và gap `person` — ghi nhận là known
  limitation cho chu kỳ này, không retrain.** Lý do: cả 2 đã xác nhận không phải do quantization
  (ổn định giữa fp32/INT8) — là vấn đề model/data thật, chỉ retrain mới sửa được, mà hiệu quả thật
  của 1 lần retrain nữa là chưa chắc (lần retrain duy nhất đã thử, `helmet_v1b`, không cải thiện
  gì). Dành thời gian còn lại cho rủi ro retrain không chắc ăn, gần ngày demo live, là rủi ro thật
  — không phải chỉ thận trọng cho có. Chi tiết đầy đủ + hướng làm nếu nhóm quyết định retrain thêm
  (per-class loss weighting cho `head`, oversample cho `person`) ở `docs/RUNLOG.md` 2026-10-17.

## Việc cần làm tiếp — không còn gì bắt buộc cho phần Data & Model

Tất cả các mục ưu tiên trước đó (mAP thật, audit dataset, quyết định 2 gap còn lại) đã xong. Việc
còn lại chỉ là **tuỳ chọn, nếu nhóm quyết định dành thêm thời gian**:

- Nếu muốn thử cải thiện `head` recall: per-class loss weighting khi retrain (xem lý do chọn
  hướng này thay vì augmentation mới ở `docs/RUNLOG.md` 2026-10-17).
- Nếu muốn thử cải thiện `person`: oversample ảnh có `person` trong training set (kỹ thuật đã
  dùng thành công cho class motorcycle ở project cũ, 2026-09-18).
- Cả 2 việc trên đưa vào report ở mục Step 7 (Monitoring & Iteration) như kế hoạch tương lai, đã
  ghi rõ trong report hay chưa cần Hạnh xác nhận lại khi rà soát report tổng.

## Tài liệu liên quan

- `docs/RUNLOG.md` — toàn bộ log chi tiết, theo ngày, có số liệu thật cho mọi claim ở trên.
- `docs/CHANGELOG.md` — bảng tóm tắt model/script nào thay đổi gì.
- `models/mAP/fp32/`, `models/mAP/int8/` — kết quả mAP thật đầy đủ (plots, confusion matrix).
- `src/audit_dataset.py`, `src/compare_int8_accuracy.py`, `src/benchmark_end_to_end.py` — script
  liên quan, đã chạy thật, không chỉ viết xong để đó.
