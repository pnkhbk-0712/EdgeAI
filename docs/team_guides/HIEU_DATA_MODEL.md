# Hiếu — Data & Model — việc cần làm tiếp (cập nhật 2026-10-17)

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
