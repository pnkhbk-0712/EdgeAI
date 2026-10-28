# Hướng dẫn dựng rig PPE và chụp khung hình thật (cho Hoàng)

Mục tiêu: có một khung cảnh thật để (1) canh lại vùng cảnh báo, (2) chụp ảnh cho tập test giữ riêng, (3) chạy `demo_helmet.py` lần đầu với dữ liệu của chính dự án.
Model `ppe_v1` nhận 4 món: **mũ, áo phản quang, găng tay, kính**. Rig phải cho phép bày từng món "có" hoặc "không có".

Mọi lệnh chạy trong **PowerShell**, tại thư mục gốc của repo (thư mục có `src`, `models`, `docs`). Dấu `>` là dòng lệnh, đừng gõ dấu đó.

## 0. Tổng quan: cần gì, làm gì, ra gì

| Bước | Việc | Cần có | Kết quả |
|---|---|---|---|
| 1 | Cài môi trường | Python 3.10–3.12, Git | chạy được script |
| 2 | Lấy code + 2 file model | repo nhánh `helmet-safety-pivot`, file ONNX từ Hiếu | `models/ppe_v1_best_*.onnx` |
| 3 | Kiểm tra máy | camera cắm sẵn | in được "OK" |
| 4 | Chuẩn bị rig + đồ | mũ, áo, găng, kính, người đóng | khung cảnh sẵn sàng |
| 5 | Canh vùng | `capture_rig_frames.py` | sửa `helmet_zone_configs.py`, ảnh chụp màn hình |
| 6 | Chụp ≈100 khung | `capture_rig_frames.py` | `data/rig_frames/*.jpg` |
| 7 | Chạy demo thật | `demo_helmet.py` + model | `helmet_events.jsonl` + ghi chú |
| 8 | Gửi lại | | zip ảnh + file đã sửa + ghi chú |

## 1. Cài môi trường (làm một lần)

Cần: **Python 3.10–3.12** (kiểm tra `> python --version`) và **Git**. Lần cài đầu tải khá nhiều gói (torch + ultralytics), cần mạng ổn.

```
> git clone https://github.com/pnkhbk-0712/EdgeAI.git
> cd EdgeAI
> git checkout helmet-safety-pivot
```
(Nếu đã có repo: `> git checkout helmet-safety-pivot` rồi `> git pull`.)
Nếu thư mục `prototype` nằm trong repo, `cd` vào thư mục chứa `src` và `requirements.txt`.

```
> python -m venv .venv
> .venv\Scripts\Activate.ps1
> pip install -r requirements.txt
```
Nếu PowerShell chặn script: `> Set-ExecutionPolicy -Scope Process Bypass` rồi kích hoạt lại venv.
Mỗi lần mở cửa sổ PowerShell mới phải chạy lại dòng `Activate.ps1`.

> Ghi chú: `run_demo.bat` trong repo trỏ tới đường dẫn Python **riêng của Hiếu**, đừng dùng trên máy bạn. Chạy `python src\...` như bên dưới.

## 2. Lấy 2 file model (không có trên GitHub)

File model quá nặng để commit, **xin Hiếu gửi** (Drive/Zalo), mỗi file ≈ 12,7 MB:

| File | Dung lượng | Đặt vào |
|---|---|---|
| `ppe_v1_best_640.onnx` | ≈ 12,75 MB | `models\ppe_v1_best_640.onnx` |
| `ppe_v1_best_416.onnx` | ≈ 12,66 MB | `models\ppe_v1_best_416.onnx` |

**Chỉ dùng model `ppe_v1`.** File mặc định của `demo_helmet.py` là model mũ cũ, không có lớp `person`, chạy sẽ báo lỗi. Luôn thêm `--model models\ppe_v1_best_640.onnx`.

## 3. Kiểm tra máy trước khi dựng rig (2 phút)

```
> python -c "import cv2, ultralytics, onnxruntime; print('OK')"
> python tests	est_ppe_compliance.py
> python tests	est_zone.py
> python -c "import cv2; c=cv2.VideoCapture(0); print('camera 0 mở được:', c.isOpened()); c.release()"
```
- Lệnh 1 in `OK`: thư viện đã đủ. Lỗi `No module named ...` → quên kích hoạt venv hoặc `pip install` chưa xong.
- Lệnh 2 và 3 (test): không được có dòng lỗi/`AssertionError`; chạy xong không in lỗi là đạt.
- Lệnh 4: `True` là camera số 0 dùng được. `False` → thử `VideoCapture(1)`; camera ngoài thường là số 1. Lúc chạy script thêm `--source 1`.

## 4. Chuẩn bị rig và đồ

| Cần | Ghi chú |
|---|---|
| Camera demo | Cái sẽ dùng thật (C270 hay DS-U02, chốt với Hạnh). Cố định bằng chân đế/băng dính, **không xê dịch sau khi canh vùng**. |
| 2–3 người đóng (hoặc hình nộm) | Người thật cho ảnh tốt hơn hình nộm. |
| Mũ bảo hộ, áo phản quang, găng tay, kính bảo hộ | Mỗi món ít nhất 1 cái. Áo nên màu vàng/cam phản quang thật. |
| Khung "giàn giáo" bằng xốp | Chỉ cần hai khu: **khu cao/nguy hiểm** (phía xa/trên khung hình) và **khu dưới đất** (phía gần). Không cần đẹp. |
| Ánh sáng | Đủ sáng, không ngược sáng (không đứng trước cửa sổ). Có thời gian thì chụp thêm 1 lượt ánh sáng khác. |

Xin phép người đóng trước khi chụp; nói rõ ảnh chỉ dùng cho bài tập này.

## 5. Canh vùng cảnh báo (làm trước khi chụp)

1. Đặt camera đúng chỗ sẽ dùng khi demo.
2. Chạy: `> python src\capture_rig_frames.py --tag zonecheck` (thêm `--source 1` nếu cần).
3. Cửa sổ hiện hình camera + 2 khung: **đỏ = high_risk**, **vàng = normal**. Xem khung có nằm đúng khu giàn giáo / khu dưới đất không. Bấm **Q** để thoát.
4. Nếu lệch: mở `src\helmet_zone_configs.py`, sửa hai dòng `HIGH_RISK_ZONE_POLYGON_FRAC` và `NORMAL_ZONE_POLYGON_FRAC`. Mỗi điểm là `(x, y)` theo **tỉ lệ 0.0–1.0** của khung (0,0 = góc trên trái, 1,1 = góc dưới phải), không phải pixel. Lưu file, chạy lại lệnh bước 2 để xem.
5. **Bỏ khe hở:** bản nháp để trống dải y 0.472–0.556 (người có tâm ở đó không thuộc vùng nào → không có cảnh báo). Cho đáy khu đỏ chạm đỉnh khu vàng (cùng một giá trị y).
6. Người đóng đứng đủ gần: **chiều cao người ≥ 12% chiều cao khung**, nhỏ hơn thì hệ thống không đánh giá món nào.
7. Chụp ảnh màn hình có 2 khung đè lên rig, gửi Hạnh/Hiếu làm bằng chứng đã canh. File `zonecheck_*.jpg` nếu lỡ bấm SPACE chỉ là ảnh thử, có thể xoá.

Mỗi vùng đòi món nào (`REQUIRED_ITEMS`: khu đỏ = mũ, áo, găng, kính; khu vàng = mũ, áo) là quyết định của Hạnh — **đừng tự sửa**, hỏi Hạnh.

## 6. Chụp khung hình test

Chạy lệnh cho từng kịch bản, bấm **SPACE** để lưu 1 khung, **Q** để thoát. Ví dụ:
```
> python src\capture_rig_frames.py --tag no_helmet
```
Ảnh lưu ở `data\rig_frames\<tag>_NNN.jpg` (đánh số tiếp, không ghi đè; thư mục này bị `.gitignore` chặn nên không bị commit).
Giữa các lần bấm, đổi dáng/hướng/vị trí đứng một chút, **đừng bấm liên tục 20 lần cùng một tư thế**.

| Tag | Người đóng mặc | Chụp ≥ |
|---|---|---|
| `full_near`, `full_mid`, `full_far` | đủ cả 4 món, đứng gần / vừa / xa | 8 mỗi tag |
| `no_helmet` | thiếu mũ, còn lại đủ | 10 |
| `no_vest` | thiếu áo, còn lại đủ | 10 |
| `no_gloves` | thiếu găng, còn lại đủ | 10 |
| `no_goggles` | thiếu kính, còn lại đủ | 10 |
| `none` | không mặc món nào | 8 |
| `two_people` | 2 người, mỗi người thiếu món khác nhau | 10 |
| `zone_high`, `zone_normal` | thiếu món, đứng hẳn trong khu đỏ / khu vàng | 8 mỗi tag |

Tổng ≈ 100 khung. Tối thiểu 50, tốt là 150.
Mẹo: tay phải thấy rõ khi đeo găng, mặt quay về camera khi đeo kính. Nếu ngoài khung hình thì model không nhận ra, đó không phải lỗi rig.
Tag chỉ gồm chữ, số, dấu `_`.

## 7. Chạy demo thật

```
> python src\demo_helmet.py --source 0 --model models\ppe_v1_best_640.onnx
```
Đổi `models\ppe_v1_best_416.onnx` để so sánh. Thêm `--source 1` nếu camera ngoài. Bấm **Q** trên cửa sổ video để dừng.

**Đọc màn hình:**
- Khung xanh lá mỏng quanh món = có món; khung đỏ mỏng = model báo thẳng `no_*`.
- Khung dày quanh người: xanh = `person #N OK`; cam/đỏ = `#N NO <MÓN>`. Nhãn thêm `[HIGH-RISK]` khi đang trong khu đỏ và thiếu quá ~1 giây; `[logged]` ở khu vàng (~3 giây). `[UNCERTAIN - review]` = model không chắc (độ tin cậy 0,40–0,60).
- Góc trên trái: tỉ lệ tuân thủ 60 giây gần nhất.

**Kết quả ghi ra:** `data\samples\helmet_events.jsonl` (mỗi sự kiện `ppe_violation` có `missing_item`, `zone`, `track_id`). Xem các ca chưa duyệt: `> python src\review_events.py --list-unreviewed`.

**Ghi lại cho từng kịch bản** (bảng đơn giản trong Word/Excel là đủ): tag, món thực sự thiếu, nhãn hệ thống hiện, trễ bao lâu mới hiện, đúng / báo nhầm / bỏ sót.

Chạy xong kiểm tra: ít nhất 3 kịch bản thiếu từng món trong khu đỏ cho ra sự kiện đúng `missing_item`; người đủ 4 món không bị báo.

## 8. Gửi lại cho Hiếu

1. Thư mục `data\rig_frames\` (nén zip, gửi tay — ảnh có người nên **không đưa lên GitHub**).
2. Ảnh chụp màn hình bước 5.7.
3. File `src\helmet_zone_configs.py` đã sửa (nếu sửa).
4. File ghi chú bước 7 + `data\samples\helmet_events.jsonl`.
5. Thông tin: camera nào, độ phân giải (dòng in ra "Camera reports …" khi chạy demo), ánh sáng nào.

## Lỗi thường gặp

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| `Model not found` | File ONNX chưa đặt đúng `models\` hoặc gõ sai tên. |
| `Model classes [...] have no 'person' class` | Đang dùng model mũ cũ. Thêm `--model models\ppe_v1_best_640.onnx`. |
| `Cannot open source 0` | Camera bị app khác dùng (Zoom, Teams…) hoặc sai số. Đóng app khác, thử `--source 1`. |
| `No module named cv2/ultralytics` | Chưa kích hoạt venv (`.venv\Scripts\Activate.ps1`) hoặc `pip install -r requirements.txt` lỗi giữa chừng — chạy lại. |
| Khung zone đè sai chỗ dù đã sửa | Chưa lưu file, hoặc đang sửa nhầm file. Phải sửa `src\helmet_zone_configs.py`. |
| Người đứng giữa hai khu mà không có cảnh báo | Còn khe hở giữa hai polygon (bước 5.5). |
| Hình giật, FPS thấp | Bình thường trên laptop CPU yếu; thử model `_416`. Không phải lỗi rig. |
| Không báo thiếu găng/kính dù đã bỏ | Món nhỏ, đứng xa/quay lưng thì model không thấy → thử đứng gần hơn. Ghi vào ghi chú, đừng cố sửa. |

## Chưa kiểm chứng

- Hướng dẫn này **chưa chạy thử trên máy của Hoàng**. `capture_rig_frames.py` mới chỉ được kiểm tra biên dịch, chưa mở cửa sổ camera thật. Nếu gặp lỗi không có trong bảng trên, chụp màn hình lỗi gửi Hiếu.
- Số ảnh tối thiểu chọn theo ước lượng (đủ để nhìn xu hướng, chưa đủ để kết luận thống kê chắc chắn).
- Phiên bản Python tối đa 3.12 và việc `pip install` thành công trên máy bạn chưa được thử (Hiếu dùng Python 3.12 / môi trường riêng).
