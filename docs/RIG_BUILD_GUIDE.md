# Hướng dẫn dựng rig PPE và chụp khung hình thật (cho Hoàng)

Mục tiêu: có một khung cảnh thật để (1) canh lại vùng cảnh báo, (2) chụp ảnh cho tập test giữ riêng, (3) chạy `demo_helmet.py` lần đầu với dữ liệu của chính dự án.
Model `ppe_v1` nhận 4 món: **mũ, áo phản quang, găng tay, kính**. Rig phải cho phép bày từng món "có" hoặc "không có".

## 1. Chuẩn bị (≈ 30 phút)

| Cần | Ghi chú |
|---|---|
| Camera demo | Cái sẽ dùng thật (C270 hay DS-U02 — chốt với Hạnh). Cố định bằng chân đế/băng dính, **không xê dịch sau khi canh vùng**. |
| 2–3 người đóng (hoặc hình nộm) | Người thật cho ảnh tốt hơn hình nộm: model học từ ảnh người thật. |
| Mũ bảo hộ, áo phản quang, găng tay, kính bảo hộ | Mỗi món có ít nhất 1 cái. Áo nên màu vàng/cam phản quang thật. |
| Khung "giàn giáo" bằng xốp | Chỉ cần tạo hai khu: khu **cao/nguy hiểm** (phía xa/trên khung hình) và khu **dưới đất** (phía gần). Không cần đẹp. |
| Ánh sáng | Đủ sáng, không ngược sáng (không đứng trước cửa sổ). Chụp thêm 1 lượt ánh sáng khác nếu có thời gian. |

## 2. Canh vùng cảnh báo (làm trước khi chụp)

1. Đặt camera đúng chỗ sẽ dùng khi demo.
2. Chạy: `python src/capture_rig_frames.py --tag zonecheck` (thêm `--source 1` nếu camera ngoài không phải số 0).
3. Cửa sổ hiện hình camera + 2 khung: **đỏ = high_risk**, **vàng = normal**. Xem khung có nằm đúng khu giàn giáo / khu dưới đất không.
4. Nếu lệch: sửa `HIGH_RISK_ZONE_POLYGON_FRAC` và `NORMAL_ZONE_POLYGON_FRAC` trong `src/helmet_zone_configs.py` (toạ độ là **tỉ lệ 0.0–1.0** của khung, không phải pixel), chạy lại bước 2.
5. **Bỏ khe hở:** bản nháp hiện để trống dải y 0.472–0.556 (người có tâm ở đó không thuộc vùng nào → không có cảnh báo). Cho đáy khu đỏ chạm đỉnh khu vàng.
6. Người đóng đứng đủ gần: **chiều cao người ≥ 12% chiều cao khung**, nhỏ hơn thì hệ thống không đánh giá.
7. Chụp 1 ảnh màn hình có 2 khung đè lên rig, gửi Hạnh/Hiếu làm bằng chứng đã canh.

## 3. Chụp khung hình test

Chạy lệnh cho từng kịch bản, bấm **SPACE** để lưu 1 khung, **Q** để thoát. Ảnh lưu ở `data/rig_frames/<tag>_NNN.jpg` (không bị commit).
Giữa các lần bấm, đổi dáng/hướng/vị trí đứng một chút, **đừng bấm liên tục 20 lần cùng một tư thế** (ảnh giống nhau không thêm giá trị).

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

Tổng ≈ 100 khung. Mục tiêu tối thiểu 50, tốt là 150.
Mẹo: tay phải thấy rõ khi đeo găng, mặt quay về camera khi đeo kính — nếu ngoài khung thì không nhận ra, không phải lỗi rig.

## 4. Chạy demo thật

```
python src/demo_helmet.py --source 0 --model models/ppe_v1_best_640.onnx
```
(đổi `_416.onnx` để so sánh). Cho từng người đóng đi vào khu đỏ / vàng khi thiếu món và ghi lại:
- Có hiện nhãn `NO <MÓN>` đúng không? Sau bao lâu (khu đỏ ~1 s, khu vàng ~3 s)?
- Báo nhầm (đang đủ mà báo thiếu) hoặc bỏ sót — ghi tag kịch bản + mô tả.
Sự kiện được ghi ở `data/samples/helmet_events.jsonl`.

## 5. Gửi lại

- Thư mục `data/rig_frames/` (nén zip, gửi tay cho Hiếu — ảnh có người nên **không đưa lên GitHub**).
- Ảnh chụp bước 2.7 và file `src/helmet_zone_configs.py` đã sửa (nếu sửa).
- Ghi chú bước 4 (đúng/sai/bỏ sót, camera nào, ánh sáng nào).
- Xin phép người đóng trước khi chụp; nói rõ ảnh chỉ dùng cho bài tập này.

## Chưa kiểm chứng

Số lượng ảnh tối thiểu chọn theo ước lượng (đủ để nhìn xu hướng, **chưa đủ để kết luận thống kê chắc chắn**). Hướng dẫn chưa được chạy thử trên camera thật: script `capture_rig_frames.py` mới chỉ kiểm tra biên dịch, chưa mở cửa sổ camera.
