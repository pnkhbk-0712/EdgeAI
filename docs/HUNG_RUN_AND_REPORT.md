# Hùng — chạy ngay + gửi kết quả (Pi 4B, 640 fp32/INT8)

File này chỉ để **thực thi lần chạy benchmark còn thiếu** và báo kết quả đúng format — bối cảnh
đầy đủ (vì sao cần, gap đã phát hiện) nằm ở `docs/team_guides/HUNG_TESTING_EVAL.md`. Đọc file này
là đủ để làm, không cần đọc lại file kia trước.

## Bước 1 — Copy 2 file còn thiếu từ laptop sang Pi

Hai file này **chưa có** trên Pi (lần chạy trước bị skip vì thiếu):

- `models/helmet_v1_best.onnx` (ONNX fp32 @ 640)
- `models/helmet_v1_best_int8.onnx` (ONNX INT8 @ 640)

**Cách 1 — scp, chạy trên laptop** (thay `<pi-user>` và `<pi-ip>`, lấy IP bằng `hostname -I` chạy
trên Pi nếu không nhớ):

```bash
cd "D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\LA4\prototype"
scp models/helmet_v1_best.onnx models/helmet_v1_best_int8.onnx <pi-user>@<pi-ip>:~/EdgeAI/prototype/models/
```

**Cách 2 — USB:** chép 2 file trên vào USB, cắm Pi, copy đúng vào `~/EdgeAI/prototype/models/`
(giữ đúng tên file, đừng đổi tên).

Chi tiết đầy đủ (nếu cần): `docs/PI_BENCHMARK_GUIDE.md` Bước 3.

## Bước 2 — Chạy lại benchmark, lần này phải ra đủ 4 dòng

```bash
cd ~/EdgeAI/prototype/src
python3 benchmark_pi.py
```

Nếu vẫn thấy `[missing: ... -- skipped]` ở dòng 640 → file chưa copy đúng chỗ, kiểm tra lại
đường dẫn `~/EdgeAI/prototype/models/` trên Pi trước khi chạy lại.

## Bước 3 — Đo nhiệt độ khi chạy liên tục (không chỉ lúc khởi động)

```bash
vcgencmd measure_temp
```

Chạy lệnh này **lại một lần nữa sau khi `benchmark_pi.py` đã chạy xong vài phút** (hoặc chạy nó
thêm 1 lần trong lúc benchmark đang chạy ở terminal khác), không chỉ đo 1 lần lúc mới mở máy —
mục đích là xem nhiệt độ có tăng đáng kể khi CPU chạy liên tục không.

## Bước 4 — (không gấp, làm sau khi 1-3 xong) Đo thử pipeline thật, không chỉ ONNX thuần

Trên laptop, pipeline thật (track() + zone check) chậm hơn ONNX thuần ~110%
(`src/benchmark_end_to_end.py`). Chưa biết overhead này lớn tới đâu trên Pi. Nếu còn thời gian:

```bash
cd ~/EdgeAI/prototype/src
python3 demo_helmet.py --no-display
```

(Cần có `ultralytics`/`torch` cài trên Pi cho bước này — nặng hơn benchmark ONNX thuần, không bắt
buộc nếu Pi không cài kịp hoặc hết thời gian.)

## Gửi lại kết quả — dùng đúng mẫu này, điền vào rồi gửi

```
=== Kết quả benchmark Pi 4B (640 fp32/INT8 rerun) ===

Bảng đầy đủ 4 dòng (copy nguyên output benchmark_pi.py):
Format        imgsz   ms/frame    FPS     <300ms?
ONNX fp32     640     472.1       2.1     no
ONNX INT8     640     553.3       1.8     no
ONNX fp32     416     234.4       4.3     YES
ONNX fp32     320     129.7       7.7     YES

Nhiệt độ lúc khởi động: 41.3°C
Nhiệt độ sau khi chạy benchmark: 47.2°C
vcgencmd get_throttled: 0x0 (không bị giảm xung)

(Nếu đã làm Bước 4) demo_helmet.py --no-display chạy được: Có / Không
  Nếu chạy được, FPS/ms-per-frame in ra cuối: ___
```

## Quyết định tiếp theo tùy kết quả (không cần tự quyết, chỉ cần báo đúng số)

- **Nếu 640 fp32 hoặc INT8 đạt <300ms:** đây là câu trả lời thật cho "Pi dùng được không, dùng
  cấu hình nào" — báo ngay cho nhóm, đặc biệt Hoàng (ảnh hưởng quyết định deploy), để chốt.
- **Nếu cả 2 đều không đạt <300ms:** 416/320 cũng không phải phương án thay thế được (đã mất
  detection `head`, xem `HUNG_TESTING_EVAL.md`) — bước tiếp theo của nhóm sẽ là xem Coral USB
  Accelerator hoặc chạy demo chính trên laptop, ghi Pi là hạn chế phần cứng đã biết trong report.

## Tài liệu liên quan

- `docs/team_guides/HUNG_TESTING_EVAL.md` — bối cảnh đầy đủ, vì sao 416/320 không dùng được.
- `docs/PI_BENCHMARK_GUIDE.md` — hướng dẫn cài lib/copy file chi tiết hơn nếu cần.
- `docs/RUNLOG.md` (mục 2026-10-16) — log đầy đủ lần chạy đầu tiên, để so sánh khi có số mới.
