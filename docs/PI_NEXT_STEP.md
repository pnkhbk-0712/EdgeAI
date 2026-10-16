# Bước tiếp theo cho Hùng — sau lần benchmark Pi đầu tiên (2026-10-16)

## Kết quả lần chạy vừa rồi

```
ONNX fp32     640     [missing: helmet_v1_best.onnx -- skipped]
ONNX INT8     640     [missing: helmet_v1_best_int8.onnx -- skipped]
ONNX fp32     416     256.9ms   3.9 FPS   YES (<300ms)
ONNX fp32     320     127.6ms   7.8 FPS   YES (<300ms)
```

`vcgencmd get_throttled` = `0x0` — tốt, lúc đo không bị throttle nguồn điện.

## Vấn đề: 2 dòng chạy được KHÔNG dùng được cho demo thật

Đã kiểm tra lại trên laptop, so khớp từng model với đúng kích thước ảnh của nó (không dùng mặc
định) trên cùng ảnh test (`val_batch0_pred.jpg`):

| imgsz | boxes | head | helmet | person |
|---|---|---|---|---|
| 640 | 16 | 3 | 13 | 0 |
| 416 | 5  | **0** | 5 | 0 |
| 320 | 0  | 0 | 0 | 0 |

**416** — nhanh hơn trong 2 dòng đã chạy — mất **toàn bộ** detection lớp `head` trên ảnh này, chỉ
còn thấy `helmet`. Mà `head` (đầu trần, không đội mũ) chính là thứ cả project này tồn tại để phát
hiện. **320** không detect được gì cả, đúng như cảnh báo cũ đã ghi.

Nói cách khác: cả 2 dòng đã đo đều đạt mục tiêu tốc độ (<300ms) nhưng không dùng được cho mục đích
an toàn thật. Tốc độ đạt ≠ hữu dụng.

## Việc cần làm — chỉ 2 bước

### 1. Copy 2 file còn thiếu từ laptop sang Pi

Hai file này **chưa có** trên Pi lần trước, cần chép thêm (từ `models/` trên laptop sang `models/`
trên Pi, giữ đúng tên):

- `models/helmet_v1_best.onnx` (ONNX fp32 @ 640)
- `models/helmet_v1_best_int8.onnx` (ONNX INT8 @ 640)

Cách chép — xem lại `docs/PI_BENCHMARK_GUIDE.md` Bước 3 (scp hoặc USB), cùng cách đã dùng cho 2
file 416/320 lần trước.

### 2. Chạy lại benchmark

```bash
cd ~/EdgeAI/prototype/src
python3 benchmark_pi.py
```

Lần này sẽ ra đủ cả 4 dòng. Đây là 2 dòng còn thiếu — và cũng là 2 dòng duy nhất có khả năng vừa
đạt tốc độ vừa còn detect được `head` (vì trên laptop, 640 là độ phân giải duy nhất còn thấy đủ cả
3 class).

**Gửi lại nguyên bảng 4 dòng** (không chỉ dòng nhanh nhất) — nếu 640 fp32 hoặc INT8 đạt <300ms trên
Pi, đó là câu trả lời thật cho câu hỏi "Pi có dùng được không, dùng cấu hình nào". Nếu cả 2 đều
không đạt, thì 416/320 cũng không phải phương án thay thế được (đã mất detection), và bước tiếp theo
sẽ là xem xét Coral USB Accelerator hoặc demo chính chạy trên laptop.
