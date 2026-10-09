# Raspberry Pi Check Report — điền vào đây (Hùng, trước khi chạy test trên thiết bị thật)

Mục tiêu: xác nhận Pi đã sẵn sàng để chạy bộ test case trong Sec. VIII của report — trước khi
đo bất kỳ số liệu latency/FPS/recall nào trên "thiết bị thật", phải chắc bản thân cái Pi đã
chạy đúng, không phải debug phần cứng lẫn với debug kết quả test.

Đây là báo cáo tình trạng phần cứng, không phải báo cáo kết quả test — kết quả test thật
(precision/recall/false-alarm theo Table III) vẫn ghi vào chỗ khác khi có model + dữ liệu thật.

## Bước 1 — Thông tin phần cứng

- [x] Model board: Raspberry Pi 4 Model B (4GB RAM) 
- [x] RAM thực tế (chạy `free -h`, ghi lại dòng `Mem:`): total 3.7Gi / used 429Mi / free 2.7Gi /                / shared 40Mi / buff-cache 693Mi / available 3.3Gi
- [x] Dung lượng thẻ nhớ / còn trống (`df -h`): Tổng 28G (/dev/mmcblk0p2), đã dùng 6.7G, còn trống 20G
- [x] Nguồn đang dùng (V/A ghi trên adapter): 5V/3A — nếu không phải đúng
      loại khuyến nghị (5V/3A cho Pi 4B, 27W USB-C PD cho Pi 5), ghi rõ, vì nguồn yếu gây
      brownout trông giống lỗi model chứ không phải lỗi nguồn

## Bước 2 — Cài hệ điều hành + kết nối

- [x] Đã flash Raspberry Pi OS (64-bit): OS: Debian GNU/Linux 13 (trixie) / Ngày flash: 23/09/2025
- [x] SSH vào được từ laptop qua WiFi: Có
- [x] `python3 --version` trên Pi: Python 3.13.5
- [x] Cài được `opencv-python` + `ultralytics` chưa: Có

## Bước 3 — Camera sanity check (chưa đụng tới model)

Chạy lệnh test 1 khung hình, xác nhận camera + OS + Python stack hoạt động trước khi đưa
YOLO vào:

```bash
python3 -c "import cv2; cv2.imwrite('t.jpg', cv2.VideoCapture(0).read()[1])"
```

- [ ] Lệnh chạy không lỗi: ⬜ Có ⬜ Không
- [ ] File `t.jpg` mở ra thấy đúng hình thật (không đen, không lỗi màu): ⬜ Có ⬜ Không
- [ ] Loại camera đang dùng: ⬜ USB webcam ⬜ Pi Camera Module — nếu Pi Camera, xác nhận đã
      bật `camera` trong `raspi-config` hoặc `libcamera` (Pi 5 dùng stack camera khác Pi 4B)

## Bước 4 — Chạy thử pipeline có sẵn (model tạm, chưa cần model đã train)

Mục đích: xác nhận Pi đủ khả năng chạy được `ultralytics` + `cv2` cùng lúc trước khi đưa model
thật lên, tách rủi ro "Pi quá yếu" ra khỏi rủi ro "model có vấn đề".

- [ ] Copy `src/demo.py` (hoặc bản rút gọn) lên Pi, chạy với model mặc định `yolov8n.pt`:
      ⬜ Chạy được ⬜ Lỗi (ghi traceback nếu có): ________________________
- [ ] FPS đo được (in ra ở cuối `demo.py`, dòng "Wall-clock time"): ________________________
      — so sánh với baseline laptop (18.4 FPS CPU, xem `docs/IMPLEMENTATION_REPORT.md`) chỉ để
      biết Pi chậm hơn bao nhiêu, không kỳ vọng bằng laptop
- [ ] Nhiệt độ Pi sau ~5 phút chạy liên tục (`vcgencmd measure_temp` nếu có): ________________________
      — nếu gần 80°C, ghi lại, đây là dấu hiệu cần tản nhiệt trước khi đo latency thật

## Bước 5 — Kết luận sẵn sàng hay chưa

- [ ] Pi đã sẵn sàng để nhận model thật + chạy bộ test case Sec. VIII: ⬜ Sẵn sàng ⬜ Chưa
- Nếu chưa, việc cần làm tiếp theo là gì: ________________________
- Rủi ro/lo ngại đã thấy (VD: nóng máy, RAM thấp, camera không ổn định): ________________________

---

Sau khi điền xong, báo lại nhóm (đặc biệt Hoàng — nếu có vấn đề phần cứng thuộc phần triển
khai của Hoàng) và cập nhật task tương ứng trong tracker của nhóm. Đây là bước chuẩn bị, không
phải kết quả cuối — kết quả test thật vẫn cần chờ model đã train xong (`m4b` trong tracker của
Hiếu) trước khi đo được recall/false-alarm theo Table III.
