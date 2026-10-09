# Hướng dẫn benchmark model mũ bảo hộ trên Raspberry Pi — Hùng

## Bối cảnh (vì sao cần làm cái này)

Lần check Pi trước (`docs/RASPI_CHECK_REPORT.md`) đo được **0.30 FPS** chạy `yolov8n.pt` gốc —
chậm hơn laptop **61 lần**, vượt mục tiêu `<300ms/frame` tới **11 lần**. Trước khi kết luận Pi
"không đủ sức" và đi mua Coral USB Accelerator (tốn tiền thật + tốn thời gian tích hợp), cần test
3 hướng rẻ hơn trước, vì:

- Con số 0.30 FPS đo trên model **gốc chưa tối ưu**, chưa phải bản ONNX đã export (Step 4 đã đo
  trên laptop: ONNX nhanh hơn PyTorch **+56.3%**)
- INT8 bị loại trên laptop (CPU x86) vì **chậm hơn** — nhưng kiến trúc ARM của Pi có thể phản ứng
  **ngược lại**, chưa ai test thật trên Pi cả
- Giảm độ phân giải ảnh đầu vào (640→416→320) là cách tăng tốc rẻ, không cần train lại

Kết quả benchmark này quyết định: Pi có dùng được không, dùng cấu hình nào, hay phải mua thêm
phần cứng.

## Việc cần làm — theo đúng thứ tự

### Bước 0 — Đổi nguồn cấp điện (nếu chưa làm)

`RASPI_CHECK_REPORT.md` ghi nguồn đang dùng là **5V/2A**, trong khi Pi 4B cần **5V/3A**, hệ thống
còn tự cảnh báo nguồn yếu. Nguồn yếu có thể làm CPU bị throttle (giảm xung nhịp), khiến FPS đo
được thấp hơn thực tế Pi có thể đạt. Đổi đúng nguồn **trước khi** đo, nếu không số liệu đo được
sẽ không đáng tin.

### Bước 1 — Clone repo trên Pi (nếu Pi chưa có repo)

```bash
cd ~
git clone -b helmet-safety-pivot https://github.com/pnkhbk-0712/EdgeAI.git
cd EdgeAI/prototype
```

Nếu Pi đã từng clone repo rồi, chỉ cần cập nhật đúng nhánh:

```bash
cd ~/EdgeAI/prototype   # hoặc đúng đường dẫn repo trên Pi
git fetch origin
git checkout helmet-safety-pivot
git pull
```

### Bước 2 — Cài thư viện

Script benchmark chỉ cần 3 thư viện này (không cần `ultralytics`/`torch` nặng nề):

```bash
pip install opencv-python-headless numpy onnxruntime
```

Dùng `opencv-python-headless` (không phải `opencv-python`) vì không cần phần hiển thị cửa sổ,
nhẹ và cài nhanh hơn nhiều trên ARM.

### Bước 3 — Copy file model từ laptop sang Pi

File `.onnx` **không nằm trong git** (bị gitignore vì là file binary nặng), nên `git pull` không
tự có được — phải chép tay 5 file sau từ laptop (đã có sẵn, do Claude export):

- `models/helmet_v1_best.onnx` (ONNX fp32 @ 640)
- `models/helmet_v1_best_int8.onnx` (ONNX INT8 @ 640)
- `models/helmet_v1_best_416.onnx` (ONNX fp32 @ 416)
- `models/helmet_v1_best_320.onnx` (ONNX fp32 @ 320)
- `docs/train_helmet/val_batch0_pred.jpg` (ảnh thật dùng để benchmark)

**Cách 1 — qua mạng (scp), chạy trên laptop:**

```bash
ssh <pi-user>@<pi-ip> "mkdir -p ~/EdgeAI/prototype/docs/train_helmet"
scp models/helmet_v1_best.onnx models/helmet_v1_best_int8.onnx models/helmet_v1_best_416.onnx models/helmet_v1_best_320.onnx <pi-user>@<pi-ip>:~/EdgeAI/prototype/models/
scp docs/train_helmet/val_batch0_pred.jpg <pi-user>@<pi-ip>:~/EdgeAI/prototype/docs/train_helmet/
```

Không nhớ IP Pi thì chạy `hostname -I` ngay trên Pi để lấy.

**Cách 2 — USB:** chép 5 file trên vào USB, cắm vào Pi, copy đúng vào 2 thư mục `models/` và
`docs/train_helmet/` tính từ gốc repo trên Pi (giữ đúng tên file, đừng đổi tên).

### Bước 4 — Chạy benchmark trên Pi

```bash
cd ~/EdgeAI/prototype/src
python3 benchmark_pi.py
```

Script tự in ra bảng kết quả 4 dòng: `ONNX fp32 @640`, `ONNX INT8 @640`, `ONNX fp32 @416`,
`ONNX fp32 @320` — mỗi dòng có ms/frame, FPS, và có đạt mục tiêu `<300ms/frame` hay không.

**Lưu ý quan trọng khi đọc kết quả:** độ phân giải thấp **có cái giá thật về độ chính xác**, không
phải chỉ nhanh hơn miễn phí — trên laptop đã đo: 640 tìm được 16 box, 416 tìm được 5, **320 tìm
được 0 box** trên cùng 1 ảnh test. Đừng chỉ chọn cấu hình nhanh nhất mà không để ý nó còn detect
được gì không.

## Báo cáo lại

Gửi lại **nguyên bảng 4 dòng** (không chỉ dòng nhanh nhất) vào nhóm hoặc trực tiếp cho mình, kèm:

- Đã đổi nguồn 5V/3A trước khi đo chưa (có/không)
- Nhiệt độ Pi lúc đo (nếu tiện, `vcgencmd measure_temp`)

Dựa vào số liệu thật này sẽ quyết định: dùng cấu hình nào cho demo trên Pi, hay cần mua Coral
Accelerator, hay demo chính chạy trên laptop và ghi Pi là hạn chế đã biết trong report.
