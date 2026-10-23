# Hoàng — Deployment — việc cần làm tiếp (cập nhật 2026-10-21)

## Mới xong (2026-10-21) — Hùng đã có đủ 4/4 dòng benchmark Pi thật, cần bạn chốt deploy format

Kết quả thật (không còn phải chờ): **không cấu hình CPU-only nào trên Pi vừa đạt <300ms vừa còn
detect đủ 3 class.** 640 fp32/INT8 (duy nhất còn detect đủ) đo được 472.1ms/553.3ms — cả 2 đều
không đạt. 416/320 đạt tốc độ nhưng đã mất detection (416 mất hết `head`, 320 mất hết mọi thứ — đo
từ trước, 2026-10-16). Chi tiết: `docs/RUNLOG.md` 2026-10-21, `docs/team_guides/HUNG_TESTING_EVAL.md`.

**Việc cần quyết định (mục 4 dưới đây giờ mới có đủ dữ liệu để chốt):** theo đúng nhánh đã nêu sẵn
ở guide cũ — hoặc (a) demo chính chạy trên laptop, Pi ghi vào report là hạn chế phần cứng đã biết,
hoặc (b) xem Coral USB Accelerator nếu nhóm còn thời gian/ngân sách. Không có lựa chọn (c) "dùng Pi
CPU-only" nữa — số liệu thật đã loại phương án đó.

## Trạng thái thật hiện tại

- **Zone polygon (`src/helmet_zone_configs.py`):** đã sửa để tính theo tỉ lệ phân giải
  (`scale_polygon()`, 2026-10-11, sau khi test webcam thật thấy render sai ở độ phân giải khác
  1280×720). Nhưng **hình dạng polygon vẫn là DRAFT** — chưa ai đo/canh theo khung hình thật từ
  rig vật lý, chỉ là ước lượng.
- **Rig vật lý (foam-board scaffolding mockup + mannequin/figure props):** theo
  `docs/PIVOT_HELMET_DETECTION.md` đây là hạng mục mới cần làm cho bản pivot này — **chưa xác
  nhận trong repo là đã dựng xong**.
- **Webcam cho demo:** chưa chốt. Phần so sánh kỹ thuật/giá đã có trong bài thi giữa kỳ (Logitech
  C270 — fixed-focus, rẻ — vs Hikvision DS-U02 — 1080p) nhưng đó là phân tích, không phải quyết
  định cuối của nhóm.
- **`src/demo_helmet.py`:** code hoàn chỉnh, đã test bằng 1 video smoke-test (xe, tái dùng từ
  project cũ, chỉ để test pipeline không lỗi) — **chưa từng chạy với camera/rig thật của chính
  project này.** Đây là gap "domain-gap verification" đã được flag từ trước, vẫn còn mở.
- Model deploy format (`.pt` / ONNX fp32 / INT8) trên Pi: **chưa chốt**, phụ thuộc kết quả
  benchmark đầy đủ của Hùng (xem `docs/team_guides/HUNG_TESTING_EVAL.md`) — 2 dòng Pi đã đo được
  (416/320) đều mất detection, không dùng được; còn thiếu 640 fp32/INT8.

## Việc cần làm — theo thứ tự

1. **Xác nhận/dựng rig vật lý** (scaffolding mockup khu high-risk, khu ground-level normal,
   mannequin/figure có và không có mũ) — nếu đã có, chỉ cần xác nhận lại với Hạnh; nếu chưa, đây
   là việc cần làm sớm nhất vì mọi thứ sau (canh zone, test camera) đều chờ cái này.
2. **Chốt webcam demo** cùng Hạnh — C270 hay Hikvision DS-U02, hoặc cái gì khác đang có sẵn.
3. **Khi có rig + camera:** chụp 1 khung hình thật, dùng nó để canh lại polygon trong
   `HIGH_RISK_ZONE_POLYGON_FRAC` / `NORMAL_ZONE_POLYGON_FRAC` (`src/helmet_zone_configs.py`) —
   hiện tại tỉ lệ đúng về mặt kỹ thuật (resolution-independent) nhưng hình dạng vùng là đoán, chưa
   đo thật.
4. **Chốt model deploy** sau khi Hùng có đủ 4 dòng benchmark Pi (640 fp32/INT8 + 416/320): nếu
   640 đạt <300ms trên Pi → dùng 640 (duy nhất giữ đủ detection). Nếu không đạt → quyết định chạy
   demo chính trên laptop, ghi Pi là hạn chế hardware đã biết trong report, hoặc xem Coral USB
   Accelerator nếu nhóm còn thời gian/ngân sách.
5. **Chạy `src/demo_helmet.py` thật** với rig + camera + model đã chốt — đây là lần đầu test với
   dữ liệu thật của chính project (domain-gap verification), không phải video tái dùng từ project
   cũ nữa.

## Phụ thuộc

- Rig vật lý + webcam: cần chốt cùng Hạnh (bước 1-2).
- Model deploy format: chờ Hùng (bước 4) — xem `docs/team_guides/HUNG_TESTING_EVAL.md`.
- Model file: đã có sẵn ở `models/` (Hiếu đã export), không cần chờ thêm.

## Tài liệu liên quan

- `docs/PIVOT_HELMET_DETECTION.md` — lý do pivot, scope rig/demo ban đầu.
- `docs/RUNLOG.md` (2026-10-11) — lý do polygon phải tính theo tỉ lệ, không hardcode pixel.
- `src/helmet_zone_configs.py`, `src/demo_helmet.py` — code cần chỉnh khi có khung hình thật.
