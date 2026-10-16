# Hiếu — Data & Model — việc cần làm tiếp (cập nhật 2026-10-16)

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
- `src/compare_int8_accuracy.py` (mới, 2026-10-13) — so fp32 vs INT8 trên **1 ảnh thật** duy nhất có
  sẵn local: số box và phân lớp giống nhau hoàn toàn, confidence lệch 0.001. Đây **không phải** so
  sánh mAP trên cả dataset — chỉ là tín hiệu một-ảnh, nói rõ trong chính script.
- **Blocker thật, chưa làm được:** `ROBOFLOW_API_KEY` không có trong environment hiện tại →
  `src/download_helmet_data.py` không chạy được → không có `data/helmet_roboflow/` ở local →
  không thể audit nhãn (`audit_dataset.py`, đang hardcode dataset xe máy cũ, cũng cần sửa nếu dùng
  lại) hoặc đo mAP INT8-vs-fp32 thật trên cả dataset.

## Việc cần làm — ưu tiên theo thứ tự

1. **Nếu muốn đo mAP thật trên cả dataset** (không chỉ 1 ảnh): tự set `ROBOFLOW_API_KEY` ở máy
   chạy thật (Colab hoặc máy có quyền), tải `data/helmet_roboflow/` về, rồi chạy lại so sánh
   fp32/INT8 trên val set thật. Không ai khác làm được bước này — cần key của Roboflow account.
2. **Quyết định gap `head→helmet` 17%:** đây là vấn đề an toàn thật, không chỉ là số liệu xấu.
   Hướng xử lý khả thi: thêm augmentation riêng cho `head`, hoặc per-class loss weighting, hoặc
   (nếu hết thời gian) ghi rõ vào report là limitation đã biết, có kế hoạch Step 7 (monitoring)
   theo dõi tiếp — nhưng cần chọn một hướng, không để trống.
3. **Quyết định gap `person` AP50=0.034:** tương tự — oversample lại class này, hoặc ghi nhận là
   known limitation vì không ảnh hưởng logic zone hiện tại.
4. Nếu có thời gian sau bước 1: cập nhật lại `src/audit_dataset.py` để chạy trên
   `data/helmet_roboflow/` thật (hiện tại file này hardcode dataset UA-DETRAC cũ).

## Không phụ thuộc ai khác

Phần này tự làm được hết, trừ bước 1 (cần `ROBOFLOW_API_KEY` của chính Hiếu/account Roboflow của
nhóm). Hoàng/Hùng cần model file (`models/helmet_v1_best*.onnx`) đã export sẵn — không cần chờ gì
thêm từ Hiếu để bắt đầu phần của họ.

## Tài liệu liên quan

- `docs/RUNLOG.md` — toàn bộ log chi tiết, theo ngày, có số liệu thật cho mọi claim ở trên.
- `docs/CHANGELOG.md` — bảng tóm tắt model/script nào thay đổi gì.
- `src/compare_int8_accuracy.py`, `src/benchmark_end_to_end.py` — 2 script mới nhất, software-only.
