# Hiếu — Data & Model — việc cần làm tiếp (cập nhật 2026-10-17)

Đây là bản tổng hợp từ `RUNLOG.md`/`CHANGELOG.md` dành riêng cho phần Data & Model, để không phải
lật lại cả file log mỗi lần hỏi "còn thiếu gì".

## Trạng thái thật hiện tại

- Model đang dùng: `models/helmet_v1_best.pt` (YOLOv8n, 30/30 epoch, Hard Hat Workers Dataset).
  AP50 thật theo class: `head`=0.963, `helmet`=0.982, **`person`=0.034** (mất cân bằng 31:1 so với
  `helmet`, phát hiện sau khi train xong, không audit trước). `person` hiện không được zone logic
  dùng tới, nên không chặn demo, nhưng là một gap thật, chưa quyết định sửa hay chỉ ghi nhận.
- **Lỗi an toàn nghiêm trọng nhất, đã đo thật:** confusion matrix cho thấy **17% các trường hợp
  `head` thật bị nhận nhầm thành `helmet`** — tức model có thể báo "có mũ" trong khi thực ra không.
  Đây là gap ưu tiên cao nhất, chưa fix.
- `helmet_v1b` (train thêm 20 epoch từ v1) — đã test, **không cải thiện thật**, không dùng.
- ✅ **mAP thật trên cả dataset (xong 2026-10-17):** chạy `yolo val` trên 1413 ảnh val set thật
  cho cả fp32 và INT8. mAP50 all: 0.660 (fp32) vs 0.659 (INT8) — **INT8 gần như không mất gì về
  accuracy**, xác nhận quyết định Step 4 (chọn fp32, loại INT8) đúng vì lý do tốc độ, không phải
  đánh đổi accuracy lấy tốc độ. Kết quả đầy đủ (confusion matrix, PR curve...) ở `models/mAP/fp32/`
  và `models/mAP/int8/`, số liệu tóm tắt ở `docs/RUNLOG.md` 2026-10-17. Không còn là việc cần làm.

## Việc cần làm — ưu tiên theo thứ tự

1. **Quyết định gap `head→helmet` 17%:** đây là vấn đề an toàn thật, không chỉ là số liệu xấu.
   mAP thật vừa đo xác nhận gap này **không phải do quantization** (head mAP50 0.960 fp32 vs 0.957
   INT8, lệch không đáng kể) — là vấn đề model/data thật, cần xử lý riêng. Hướng xử lý khả thi:
   thêm augmentation riêng cho `head`, hoặc per-class loss weighting, hoặc (nếu hết thời gian) ghi
   rõ vào report là limitation đã biết, có kế hoạch Step 7 (monitoring) theo dõi tiếp — nhưng cần
   chọn một hướng, không để trống.
2. **Quyết định gap `person` AP50=0.034→0.041 (đo lại, cùng bậc số liệu):** mAP thật cũng xác nhận
   gap này không đổi giữa fp32/INT8 (0.041 vs 0.040) — đúng là vấn đề mất cân bằng dữ liệu (31:1),
   không phải lỗi quantization hay export. Hướng xử lý: oversample lại class này, hoặc ghi nhận là
   known limitation vì không ảnh hưởng logic zone hiện tại.
3. Nếu có thời gian: cập nhật lại `src/audit_dataset.py` để chạy trên `data/helmet_roboflow/` thật
   (hiện tại file này hardcode dataset UA-DETRAC cũ).

## Không phụ thuộc ai khác

Phần này tự làm được hết. Hoàng/Hùng cần model file (`models/helmet_v1_best*.onnx`) đã export
sẵn — không cần chờ gì thêm từ Hiếu để bắt đầu phần của họ.

## Tài liệu liên quan

- `docs/RUNLOG.md` — toàn bộ log chi tiết, theo ngày, có số liệu thật cho mọi claim ở trên.
- `docs/CHANGELOG.md` — bảng tóm tắt model/script nào thay đổi gì.
- `models/mAP/fp32/`, `models/mAP/int8/` — kết quả mAP thật đầy đủ (plots, confusion matrix).
- `src/compare_int8_accuracy.py`, `src/benchmark_end_to_end.py` — 2 script software-only trước đó.
