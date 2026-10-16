# Hùng — Testing & Evaluation — việc cần làm tiếp (cập nhật 2026-10-16)

Thay cho `docs/PI_NEXT_STEP.md` (đã merge vào đây) — file đó đã xoá, mọi thứ liên quan Pi benchmark
giờ nằm ở đây. `docs/PI_BENCHMARK_GUIDE.md` vẫn giữ, dùng khi cần hướng dẫn chi tiết từng bước
(scp file, cài thư viện...).

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
