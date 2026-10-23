# Hùng — Testing & Evaluation — việc cần làm tiếp (cập nhật 2026-10-21)

## Mới xong (2026-10-21) — benchmark_pi.py đủ 4/4 dòng + pipeline thật trên Pi

```
Format        imgsz   ms/frame    FPS     <300ms?
ONNX fp32     640     472.1       2.1     no
ONNX INT8     640     553.3       1.8     no
ONNX fp32     416     234.4       4.3     YES
ONNX fp32     320     129.7       7.7     YES
```

> **ĐÍNH CHÍNH 2026-10-24:** đoạn "416/320 mất `head`" ở dưới (và kết luận "không cấu hình nào
> dùng được" trước đó) chỉ dựa trên **1 ảnh**. Đo lại trên cả 1413 ảnh val: 416 giữ `head` AP50
> 0.936 (recall 0.888), 320 giữ 0.896 (recall 0.845) — đều qua ngưỡng 0.80. Xem `docs/RUNLOG.md`
> 2026-10-24. **416 (234ms trên Pi) là ứng viên chính.**

**Kết quả:** 640 không đạt <300ms (472/553ms); 416/320 đạt (234/130ms) và — theo đo lại ở trên —
vẫn giữ detection. Chưa biết: pipeline đầy đủ (không chỉ ONNX đơn lẻ) ở 416 có còn <300ms không
(ước tính ~385ms theo tỉ lệ +64%, chưa đo).

**Việc tiếp theo cho bạn (Hùng):** chạy trên Pi, lần lượt 2 lệnh, gửi lại output:
```
python3 demo_helmet.py --no-display --source ~/EdgeAI/data/samples/test_loop.mp4 --model ~/EdgeAI/models/helmet_v1_best_416.onnx
python3 demo_helmet.py --no-display --source ~/EdgeAI/data/samples/test_loop.mp4 --model ~/EdgeAI/models/helmet_v1_best_320.onnx
```

Cũng đã chạy thật `demo_helmet.py --source test_loop.mp4` trên Pi — pipeline đầy đủ chạy được,
không lỗi, 1.3 FPS (chậm hơn ~64% so với benchmark ONNX đơn lẻ cùng độ phân giải — cùng hướng với
laptop, đóng việc #6 cũ). Nhiệt độ đo 2 lần sau tải liên tục (58.9°C, 47.2°C), `throttled=0x0` cả
2 lần — không throttle, đóng việc #5 cũ. Chi tiết đầy đủ: `docs/RUNLOG.md` 2026-10-21.

**Còn 1 việc chưa đóng:** `RASPI_CHECK_REPORT.md` Bước 3 (camera sanity check) — lần chạy trên
dùng `test_loop.mp4` (video có sẵn), không phải camera thật (`source 0`). Cần chạy lại lệnh ở
Bước 3 (`cv2.VideoCapture(0)`) để xác nhận camera thật + OS + Python stack hoạt động, trước khi
coi phần cứng Pi là đã kiểm tra xong hoàn toàn.

Thay cho `docs/PI_NEXT_STEP.md` (đã merge vào đây) — file đó đã xoá, mọi thứ liên quan Pi benchmark
giờ nằm ở đây. `docs/PI_BENCHMARK_GUIDE.md` vẫn giữ, dùng khi cần hướng dẫn chi tiết từng bước
(scp file, cài thư viện...).

> **Muốn làm luôn, không cần đọc hết file này trước?** Qua thẳng `docs/HUNG_RUN_AND_REPORT.md` —
> bản rút gọn chỉ có lệnh cần chạy + mẫu báo cáo kết quả.

## Trạng thái thật hiện tại

### 1. `RASPI_CHECK_REPORT.md` — chưa xong hoàn toàn

Bước 1, 2, 4, 5 đã điền. **Bước 3 (camera sanity check) còn để trống** — chưa xác nhận camera +
OS + Python stack chạy được 1 khung hình thật trước khi đưa YOLO vào. Cần làm bước này trước khi
số liệu FPS/latency đo sau đó được tin là không bị nhiễu bởi lỗi camera.

### 2. Lần chạy `benchmark_pi.py` đầu tiên (2026-10-16) — chỉ chạy được 2/4 dòng

```
ONNX fp32     640     [missing: helmet_v1_best.onnx -- skipped]
ONNX INT8     640     [missing: helmet_v1_best_int8.onnx -- skipped]
ONNX fp32     416     256.9ms   3.9 FPS   YES (<300ms)
ONNX fp32     320     127.6ms   7.8 FPS   YES (<300ms)
```

`vcgencmd get_throttled` = `0x0` lúc đo — tốt, nhưng chỉ là 1 lần đo, chưa chắc giữ được khi chạy
liên tục lâu (sustained load).

**Đã kiểm tra lại trên laptop — 2 dòng chạy được ở trên KHÔNG dùng được cho demo thật.** So khớp
từng model với đúng kích thước ảnh của chính nó (không dùng mặc định) trên cùng ảnh test
(`val_batch0_pred.jpg`):

| imgsz | boxes | head | helmet | person |
|---|---|---|---|---|
| 640 | 16 | 3 | 13 | 0 |
| 416 | 5  | **0** | 5 | 0 |
| 320 | 0  | 0 | 0 | 0 |

416 — nhanh hơn trong 2 dòng đã chạy — **mất toàn bộ detection lớp `head`** trên ảnh này (lớp quan
trọng nhất, vì đó chính là "không đội mũ"). 320 không detect được gì cả. Tốc độ đạt mục tiêu
(<300ms) không có nghĩa là dùng được — 2 dòng này không phải câu trả lời.

## Việc cần làm — theo thứ tự

1. **Hoàn thành Bước 3 trong `RASPI_CHECK_REPORT.md`** (camera sanity check) nếu chưa làm.
2. **Copy 2 file còn thiếu từ laptop sang Pi** (từ `models/` laptop sang `models/` Pi, giữ đúng
   tên, cách chép xem `docs/PI_BENCHMARK_GUIDE.md` Bước 3):
   - `models/helmet_v1_best.onnx` (ONNX fp32 @ 640)
   - `models/helmet_v1_best_int8.onnx` (ONNX INT8 @ 640)
3. **Chạy lại `python3 benchmark_pi.py`** trên Pi — lần này sẽ ra đủ 4 dòng. Đây là 2 dòng duy
   nhất có khả năng vừa đạt tốc độ vừa còn detect đủ cả 3 class (vì trên laptop chỉ có imgsz=640
   mới thấy đủ). **Gửi lại nguyên cả bảng 4 dòng**, không chỉ dòng nhanh nhất.
4. **Nếu 640 fp32 hoặc INT8 đạt <300ms trên Pi:** đó là câu trả lời thật cho "Pi dùng được không,
   dùng cấu hình nào" — báo lại nhóm (đặc biệt Hoàng, vì ảnh hưởng quyết định deploy) để chốt.
   Nếu cả 2 đều không đạt: 416/320 cũng không phải phương án thay thế (đã mất detection) — bước
   tiếp theo là xem Coral USB Accelerator hoặc demo chính chạy laptop, Pi ghi là limitation.
5. **Sau khi có kết quả bước 4:** đo thêm nhiệt độ Pi khi chạy liên tục (`vcgencmd measure_temp`
   sau vài phút), không chỉ 1 lần đo lúc khởi động.
6. *(Sau khi bước 2-4 xong, không gấp)* — trên laptop, full pipeline (track() + zone check) chạy
   chậm hơn ~110% so với benchmark ONNX đơn lẻ (`src/benchmark_end_to_end.py`). Chưa biết overhead
   này lớn tới đâu trên CPU ARM chậm hơn của Pi — nếu có thời gian, đo thử pipeline thật
   (`src/demo_helmet.py --no-display`) trên Pi, không chỉ benchmark ONNX thuần.

## Phụ thuộc

Không chờ ai — 2 file `.onnx` 640 đã có sẵn trên laptop (Hiếu đã export), chỉ cần copy sang Pi.

## Tài liệu liên quan

- `docs/PI_BENCHMARK_GUIDE.md` — hướng dẫn chi tiết từng bước (copy file, cài lib).
- `docs/RASPI_CHECK_REPORT.md` — checklist phần cứng, điền tiếp Bước 3.
- `docs/RUNLOG.md` (mục 2026-10-16) — log đầy đủ cho kết quả benchmark lần này.
