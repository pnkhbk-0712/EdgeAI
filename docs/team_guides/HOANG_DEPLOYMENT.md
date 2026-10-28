# Hoàng — Deployment — việc cần làm tiếp (cập nhật 2026-10-28)

## MỚI (2026-10-25) — Hệ thống đổi sang PPE 4 món, rig cần thêm đồ

**Làm theo từng bước: `docs/RIG_BUILD_GUIDE.md`** (chuẩn bị → canh vùng → chụp ≈100 khung → chạy demo → gửi lại). Công cụ: `python src/capture_rig_frames.py --tag <kịch bản>`.

Model mới `ppe_v1` (mũ, áo phản quang, găng tay, kính) thay `helmet_v1`; `demo_helmet.py` giờ đánh giá theo từng người và báo `ppe_violation` kèm `missing_item`. Model mũ cũ **không dùng được** cho demo mới (không có lớp `person` đủ tốt).
1. **Rig:** hình nộm/người đóng cần có/không có **áo phản quang, găng tay, kính** (không chỉ mũ). Găng/kính là vật nhỏ, model kém hơn mũ/áo khi ở xa → đặt người đủ lớn trong khung (cao ≥ 12% chiều cao khung, dưới mức đó hệ thống không đánh giá).
2. **Canh vùng:** polygon trong `helmet_zone_configs.py` vẫn là DRAFT và **có khe hở y 0.472–0.556** (người có tâm ở đó không thuộc vùng nào → không có sự kiện). Canh lại trên khung hình rig thật, bỏ khe này. `REQUIRED_ITEMS` (mỗi vùng bắt buộc món nào) là giả định nhóm, chốt với Hạnh.
3. **Chụp 50–150 khung hình thật từ rig** (đủ tổ hợp thiếu/đủ từng món, nhiều khoảng cách) cho Hiếu làm tập kiểm tra giữ riêng — không đưa vào train.
4. **Chạy demo thật:** `python src/demo_helmet.py --model models/ppe_v1_best_640.onnx` (hoặc `_416.onnx`) với camera + rig; ghi lại sự kiện sai/thiếu.
- Pi: chờ Hùng đo `benchmark_pi.py --ppe` (chưa có số) rồi mới chốt 640 hay 416. Laptop CPU 25/10 ms không phải số của Pi.

> Các mục bên dưới viết trước pivot (model mũ `helmet_v1`); số Pi 640/416 ở đó là của model cũ.

## Mới xong (2026-10-21) — Hùng đã có đủ 4/4 dòng benchmark Pi thật, cần bạn chốt deploy format

Kết quả thật: 640 fp32/INT8 đo được 472.1ms/553.3ms trên Pi — không đạt <300ms. **ĐÍNH CHÍNH
2026-10-24:** bản trước của mục này nói 416/320 "mất detection" — đó là kết luận từ 1 ảnh và đã
sai. Đo lại trên cả val set: 416 giữ `head` AP50 0.936 (recall 0.888), 320 giữ 0.896 (0.845), vs 640
là 0.960 (0.926). **Vậy chạy trên Pi CPU-only ở 416 là khả thi** (234ms standalone), không cần
laptop hay Coral. Chi tiết + bảng đầy đủ: `docs/RUNLOG.md` 2026-10-24.

**Việc cần chốt:** (1) chờ Hùng đo pipeline đầy đủ ở 416/320 trên Pi (ước tính ~385ms ở 416,
chưa đo); (2) khi có rig + camera, kiểm tra 416 trên khung hình thật của rig — số trên là val set
của dataset, không phải rig của mình, và recall `head` giảm ~4 điểm so với 640 (nhiều ca "không
mũ" bị bỏ sót hơn).

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
