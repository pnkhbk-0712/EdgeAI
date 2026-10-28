# Hạnh — Lead & Integration — việc cần làm tiếp (cập nhật 2026-10-28)

## MỚI (2026-10-25) — Dự án pivot sang PPE 4 món (mũ, áo phản quang, găng tay, kính)

Theo góp ý thầy tuần 11. Code + model + tài liệu đã commit `9d5f681` trên nhánh `helmet-safety-pivot` (**chưa push, chưa PR vào main**). Chi tiết số liệu: `docs/team_guides/HIEU_DATA_MODEL.md`, `docs/RUNLOG.md` 2026-10-25.
- Đã có: `src/ppe_compliance.py` (đánh giá theo từng người, theo vùng cơ thể), `demo_helmet.py` viết lại theo người (sự kiện `ppe_violation` + `missing_item`), `review_events.py --item`, model `ppe_v1` (mAP50 0.559 trên 111 ảnh test; `no_*` yếu nên "thiếu" suy từ không thấy món), test 18/18 pass.
- **Cần bạn quyết định:** (1) `REQUIRED_ITEMS` trong `helmet_zone_configs.py` — hiện high_risk = 4 món, normal = mũ + áo, đó là *giả định nhóm* (QCVN 18:2021/BXD 2.19.1.1 chỉ nói PTBVCN theo công việc/nguy cơ); (2) đồng ý push nhánh / mở PR; (3) giấy phép: Construction-PPE là AGPL-3.0, PPEv5 là CC BY 4.0 → báo cáo phải ghi nguồn.
- **Báo cáo Word `Project_Report_EdgeAI_Group9.docx` chưa cập nhật theo PPE** (Hiếu không sửa file của bạn). Cần sửa: phạm vi/Step 1 (4 món), Step 2 (dữ liệu, sổ nguồn), Step 4/6 (model PPE, số Pi chưa đo), Step 8 (log theo món). Bản nháp Step 1/2 của Hiếu: `docs/BaoCao_Tuan11_Step1_Step2_v5.docx` (chưa theo PPE 4 món, sẽ viết lại).
- Chưa có: số Pi cho `ppe_v1` (chờ Hùng), khung hình rig thật (chờ Hoàng), tập test thực tế. Không trích số model mũ cũ (`helmet_v1`) làm kết quả hệ thống.

> Các mục bên dưới là trạng thái trước pivot (model mũ `helmet_v1`); đọc cùng phần MỚI ở trên.

## Mới xong (2026-10-21) — Hùng đã có đủ 4/4 dòng benchmark Pi, quyết định deploy giờ không còn bị chặn

Kết quả thật: 640 đo 472ms (fp32)/553ms (INT8) trên Pi, không đạt <300ms. **ĐÍNH CHÍNH
2026-10-24:** bản trước nói 416/320 "mất detection" (suy từ 1 ảnh) — sai. Đo lại trên cả val set,
416 giữ `head` AP50 0.936 (recall 0.888 vs 0.926 ở 640), 320 giữ 0.896; 416 chạy 234ms trên Pi.
**Không cần laptop hay Coral — hướng đi hiện tại là Pi CPU-only ở 416**, còn chờ Hùng đo pipeline
đầy đủ (ước tính ~385ms, chưa đo) và Hoàng kiểm tra trên rig thật. Xem `docs/RUNLOG.md`
2026-10-24. **Việc của bạn:** khi Hùng gửi số pipeline ở 416, cập nhật report Step 4/6 theo số
đó (và sửa các chỗ report đang ghi 416/320 mất detection). Bước 3 camera sanity check trên Pi
vẫn còn thiếu (lần chạy vừa rồi dùng video file, không phải camera thật) — không chặn quyết định
deploy, nhưng vẫn cần làm trước khi report coi phần cứng Pi là đã kiểm tra xong.

## Mới xong (2026-10-18) — Step 8 Ưu tiên 2, phần không cần rig/Pi

Report mục 11 Ưu tiên 2 (Step 8, Ethical) nêu 3 gap: retention, appeal/correction mechanism,
governance truy cập log. Cả 3 đã xong — **retention** đã có từ 2026-10-13 (report chưa cập nhật
theo); **appeal/correction** giờ có script thật (`src/review_events.py`); **access governance**
đã quyết định (chỉ team + safety officer xem log thô, báo cáo rộng hơn dùng
`compliance_snapshot`). Chi tiết: `docs/EVENTS_GOVERNANCE.md`, `docs/RUNLOG.md` 2026-10-18.
**Việc còn lại của bạn:** cập nhật điểm Step 8 (hiện 3.0/4, báo cáo nêu đúng 3 gap này) và mục 11
Ưu tiên 2 trong `Project_Report_EdgeAI_Group9.docx` sau khi rà lại `EVENTS_GOVERNANCE.md`.

Ưu tiên 3 (Step 4) phần "đo accuracy impact của INT8" cũng đã xong — mAP thật fp32 vs INT8 trên
cả val set, xem `docs/team_guides/HIEU_DATA_MODEL.md`. Phần còn mở của Ưu tiên 3 (end-to-end
latency trên Pi thật) vẫn chờ Hùng, không phải việc code/doc.

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
