# Zone & Label Definitions — điền vào đây (Hieu, Prep phase)

Mục tiêu: quyết định trước 5 câu hỏi dưới đây, để lúc label trong CVAT hoặc chỉnh `zone.py`
không phải đoán giữa chừng. Không cần code — chỉ cần điền chữ vào chỗ `___`.

## 1. Xe tính là "trong zone" khi nào?

Code hiện tại (`src/zone.py`, hàm `contains()`) dùng: **tâm hình chữ nhật bounding box** của xe
nằm trong polygon vùng cấm (`cv2.pointPolygonTest` trên điểm center).

- [ ] Giữ nguyên cách này (đơn giản, đã chạy được) — **khuyến nghị cho MVP**
- [ ] Đổi sang: điểm **đáy-giữa** bbox (bottom-center) thay vì tâm — chính xác hơn cho camera
      góc nghiêng/trên cao, vì đó là điểm xe thực sự "chạm đất". Nếu chọn cái này, báo lại để
      sửa `zone.py`.
- [ ] Đổi sang: % diện tích overlap giữa bbox và polygon ≥ ___ % (VD 50%) — cần cho xe chỉ
      chèn 1 phần vào vùng cấm mà vẫn tính là vi phạm.

**Điền câu trả lời của bạn ở đây:** ___________________________

## 2. Ngưỡng thời gian đứng yên (dwell time) bao lâu thì tính là "đỗ"?

Code hiện tại: **2 giây** (`DWELL_SECONDS = 2.0` trong `demo.py`).

- [ ] Giữ 2 giây (đủ để loại xe chạy qua, nhưng khá ngắn cho "đỗ" thật)
- [ ] Đổi thành ___ giây (ghi số cụ thể)

Lưu ý: luật VN phân biệt "dừng xe" (dừng ngắn, tài xế ở lại trong xe) và "đỗ xe" (để xe lại,
tài xế có thể rời đi) — hệ thống hiện tại **không phân biệt 2 loại này**, chỉ dùng 1 ngưỡng
thời gian chung. Nếu muốn tách riêng, ghi rõ ở đây: ___________________________

## 3. Biển báo tại site có điều kiện giờ/ngày không? (xem Design Note trong artifact)

- Giờ cấm: ___________ (VD: 6h–21h, hoặc "cả ngày" nếu không ghi giờ)
- Ngày: ☐ Mọi ngày &nbsp;☐ Ngày lẻ &nbsp;☐ Ngày chẵn &nbsp;☐ Chưa biết (chưa đi khảo sát)

→ Giá trị này sẽ điền vào `active_hours` / `active_days` của `zone_config` sau.

## 4. Trường hợp mập mờ (edge case) — xử lý sao?

| Tình huống | Tính là vi phạm? |
|---|---|
| Xe che khuất 1 phần bởi xe khác, nhưng box vẫn detect được | ☐ Có &nbsp;☐ Không &nbsp;☐ Tuỳ % che khuất: ___ |
| Xe dừng đèn đỏ ngay cạnh vùng cấm (không cố ý đỗ) | ☐ Có &nbsp;☐ Không |
| Xe máy dừng rất ngắn để trả khách rồi đi (dưới ngưỡng dwell) | Không (mặc định theo ngưỡng câu 2) |
| Nhiều xe cùng lúc trong vùng cấm | Mỗi xe tính vi phạm riêng theo track_id (đã code sẵn) |

## 5. Danh sách class cần label (khớp với report + dataset đã có)

- [ ] car
- [ ] motorcycle
- [ ] bus
- [ ] truck
- [ ] (nếu làm luôn phần biển báo) sign_normal / sign_odd / sign_even

---

**Sau khi điền xong**, gửi lại câu trả lời câu 1–4 để mình chỉnh `zone.py` cho khớp (nếu có gì
đổi so với mặc định hiện tại) — còn nếu giữ nguyên mặc định, coi như task này xong mà không cần
sửa code gì cả.

---

## ĐÃ CHỐT (Hieu, 2026-09-15) — đã sửa code khớp với các câu trả lời này

1. Giữ nguyên tâm bbox — **không đổi code**.
2. Đổi ngưỡng dwell **2s → 60 giây**. Đã sửa `DWELL_SECONDS` trong `demo.py` và
   `demo_synthetic.py`, chạy lại thật và xác nhận trigger đúng ở giây 60.3.
3. Chưa khảo sát site nên chưa biết giờ/ngày cụ thể → tạm để `active_hours=None`,
   `active_days=None` (cấm mọi lúc, mọi ngày) trong `demo.py`. **Cần cập nhật 2 giá trị này
   sau khi có thông tin biển báo thật từ site.**
4. Che khuất một phần vẫn tính vi phạm nếu model vẫn detect được — không cần sửa gì (đã là
   hành vi mặc định). Riêng trường hợp "dừng đèn đỏ cạnh vùng cấm" — **code không tự phân biệt
   được** với đỗ xe thật nếu đèn đỏ lâu hơn 60 giây. Cách xử lý thực tế: **vẽ vùng cấm (polygon)
   sát vào đúng phần vỉa hè/lề đường cấm đỗ, không lấn vào làn xe chạy** — đây là việc của người
   vẽ zone lúc lắp đặt thật (Hoàng), không phải lỗi code cần fix. Đã ghi rõ trong
   `src/zone.py` và `docs/RUNLOG.md`.
5. Class cần label: car, motorcycle, bus, truck, sign_normal, sign_odd, sign_even — khớp với
   việc Hùng đang làm (gắn nhãn biển báo).
