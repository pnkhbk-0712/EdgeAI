# Hạnh — Lead & Integration — việc cần làm tiếp (cập nhật 2026-10-16)

## Trạng thái thật hiện tại

- Report tự đánh giá (`docs/Project_Report_EdgeAI_Group9.docx`, mục 9-11, theo đúng format chấm
  của thầy) đang ghi **Readiness Score 87.5/100 (28/32)**, có mục "Ưu tiên hiệu chỉnh" xếp hạng
  theo độ ưu tiên — cần người theo dõi việc nào trong đó đã xong, việc nào chưa, và ai làm.
- **Rig vật lý** (foam-board scaffolding mockup + mannequin) — theo
  `docs/PIVOT_HELMET_DETECTION.md` là hạng mục mới cần dựng cho bản pivot, **chưa xác nhận trong
  repo là đã có** — cần chốt với Hoàng.
- **Webcam demo** — chưa chốt giữa Logitech C270 (rẻ, fixed-focus) và Hikvision DS-U02 (1080p),
  phần so sánh kỹ thuật đã có trong bài thi giữa kỳ nhưng chưa thành quyết định chính thức.
- **Pi 4B** — lần benchmark đầu (2026-10-16) chỉ chạy được 2/4 cấu hình, và cả 2 cấu hình chạy
  được đều mất khả năng detect `head` (xem chi tiết ở `docs/team_guides/HUNG_TESTING_EVAL.md`).
  Quyết định "demo chạy trên Pi hay laptop" **vẫn chưa chốt được**, chờ Hùng chạy lại đủ 4 dòng.
- Model: gap thật đã biết — 17% `head` bị nhận nhầm `helmet`, class `person` AP50 chỉ 0.034 (chi
  tiết: `docs/team_guides/HIEU_DATA_MODEL.md`) — cần quyết định ghi vào report là limitation hay
  dành thời gian fix trước khi nộp/demo.

## Việc cần làm — theo thứ tự

1. **Rà lại mục "Ưu tiên hiệu chỉnh"** trong `Project_Report_EdgeAI_Group9.docx` — với mỗi việc,
   xác nhận ai làm, deadline, đã xong hay chưa. Đối chiếu với 3 file guide còn lại
   (`HIEU_DATA_MODEL.md`, `HUNG_TESTING_EVAL.md`, `HOANG_DEPLOYMENT.md`) để không bị trùng/sót.
2. **Chốt cùng Hoàng:** rig vật lý đã dựng xong chưa, webcam nào sẽ dùng cho demo chính thức —
   đây là 2 quyết định đang chặn phần của Hoàng (canh zone, test camera thật).
3. **Theo dõi kết quả benchmark Pi đầy đủ của Hùng** (4/4 dòng, đặc biệt dòng 640 fp32/INT8) —
   đây là input quyết định "demo chạy trên Pi hay laptop", ảnh hưởng trực tiếp tới phần trình bày
   trong report và cách demo live trong lớp.
4. **Khi rig + camera + model deploy đã chốt:** chạy lại bộ test plan đã đề xuất trong bài thi
   giữa kỳ (Camera compatibility / Image quality / Motion test / Detection performance /
   End-to-end performance / Zone verification) như một bước tích hợp thật trước demo, không chỉ
   trên giấy.
5. **Khi báo cáo tiến độ lại với thầy/nhóm:** dùng `docs/RUNLOG.md` và `docs/CHANGELOG.md` làm
   nguồn số liệu chính — một số số liệu cũ trong `docs/PROJECT_REPORT.md` (project parking cũ) đã
   lỗi thời, và ngay cả số ONNX latency ghi trong báo cáo pivot ban đầu sau này đo lại thấy dao
   động ±30% giữa các lần chạy (đã sửa trong `docs/train_helmet/export_benchmark.md`) — tránh
   trích lại số cũ chưa đối chiếu lại.

## Phụ thuộc

Việc của Hạnh phần lớn là tổng hợp + quyết định, không bị chặn kỹ thuật — nhưng bước 2-4 cần input
thật từ Hoàng và Hùng trước khi chốt được.

## Tài liệu liên quan

- `docs/team_guides/HIEU_DATA_MODEL.md`, `HUNG_TESTING_EVAL.md`, `HOANG_DEPLOYMENT.md` — 3 file
  còn lại trong bộ 4 guide này.
- `docs/RUNLOG.md`, `docs/CHANGELOG.md` — log/nguồn số liệu chính, theo ngày.
- `docs/PIVOT_HELMET_DETECTION.md` — lý do pivot, scope ban đầu (rig, webcam, dataset).
