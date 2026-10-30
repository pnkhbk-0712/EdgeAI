# Dataset audit & source ledger — hệ thống PPE 4 món (bắt đầu 2026-10-24)

Mỗi nguồn dữ liệu ghi: tên, link, giấy phép, số ảnh, ngày tải, vị trí, kết luận. Chỉ số **đo trực tiếp từ file nhãn** (`src/class_stats.py`, `src/audit_dataset.py`, `src/check_near_duplicates.py`).

## Nguồn 1 — Roboflow PPEv5 (PD2), version 16 "Flip"
- Link: https://universe.roboflow.com/pd2/ppev5/dataset/16 · giấy phép **CC BY 4.0** (ghi nguồn) · tải 2026-10-24 (zip 330.788.169 byte) · `data/ppe_sources/ppev5/v16/` (zip gốc cạnh đó).
- 8.945 ảnh (train 6.738 / valid 2.207 / không có test), 640×640 kéo giãn, lật ngang ngẫu nhiên (1 ảnh ra cho mỗi ảnh gốc). Audit: 0 lỗi định dạng, 0 file nhãn rỗng.
- Lớp (id theo `data.yaml`): 0 boots, 1 glasses, 2 gloves, 3 helmet, 4 no boots, 5 no glasses, 6 no gloves, 7 no helmet, 8 no vest, 9 person, 10 vest. Đã xem tay 10 crop mỗi lớp: tên lớp **đúng nghĩa**.

| Lớp | train box (ảnh) | valid box (ảnh) |
|---|---|---|
| boots | 1647 (1070) | 339 (242) |
| glasses | 1344 (1264) | 5 (5) |
| gloves | 138 (93) | 3222 (1615) |
| helmet | 140 (122) | 2 (2) |
| no boots | 3323 (2166) | 0 |
| no glasses | 1192 (694) | 2 (2) |
| no gloves | 2296 (2084) | 359 (357) |
| no helmet | 2248 (1646) | 5 (5) |
| no vest | 589 (567) | 7 (7) |
| person | 2142 (1881) | 20 (16) |
| vest | 74 (65) | 1 (1) |

**Phát hiện (quan trọng):**
1. Bộ này **không phải ảnh công trường**: là ảnh cận cảnh/ảnh sản phẩm từng bộ phận. Các lớp `no_*` là *bộ phận cơ thể trần*: `no boots` = bàn chân trần, `no glasses` = vùng mắt trần, `no gloves` = bàn tay trần, `no helmet` = đầu/mặt không mũ, `no vest` = thân trên không áo. `gloves` là ảnh găng kiểu catalog nền trắng.
2. **Valid không cùng phân bố với train** (tên gọi "75/25" nhưng thực chất là hai nguồn khác nhau): valid có 3.222 box gloves nhưng chỉ 2 helmet, 1 vest; train có 138 gloves, 140 helmet, 74 vest. Không dùng valid gốc để đánh giá; phải chia lại theo nhóm ảnh.
3. **Mũ và áo phản quang dương tính rất ít** (helmet 142, vest 75 box cả hai tập) → bộ này **không** bù được cho mũ/áo. Nó chỉ có ích làm dữ liệu bổ sung cho găng, kính và các lớp "thiếu" dạng bộ phận trần, với khác biệt miền lớn so với công trường.
4. Ảnh gần giống valid↔train (dHash 64-bit): ≤4 bit 5,4% (120/2207), ≤8 bit 24,2% (535) — ngưỡng 8 có thể bao gồm ảnh sản phẩm nền trắng giống nhau, chưa xem tay.

Kết luận tạm: giữ làm nguồn **bổ sung** cho gloves/goggles/no_gloves/no_goggle (và `no vest` là thân không áo); không dùng nó để chứng minh mũ/áo. Ánh xạ dự kiến → `glasses→goggles, no glasses→no_goggles, gloves, no gloves→no_gloves, helmet, no helmet→no_helmet, vest, no vest→(suy luận, cân nhắc), person`; bỏ boots/no boots. Cần Construction-PPE (ảnh công trường thật) làm nền và làm tập kiểm tra.

## Nguồn 2 — Ultralytics Construction-PPE
- Link: https://docs.ultralytics.com/datasets/detect/construction-ppe/ · tải: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip · giấy phép **AGPL-3.0** (ghi nguồn) · 178 MB · tải 2026-10-25 · `data/ppe_sources/construction_ppe/` (zip gốc cạnh `raw/`).
- 1.416 ảnh công trường thật/dàn dựng: train 1.132 / val 143 / test 141. Lớp: helmet, gloves, vest, boots, goggles, no_helmet, no_goggle, no_gloves, no_boots, Person, none. Audit: 0 lỗi định dạng.
- 30/143 ảnh val và 30/141 ảnh test có ảnh gần giống trong train (khung hình video, dHash ≤ 8/64 bit) → loại khỏi val/test; còn 113 / 111 ảnh.
- `none` được ánh xạ thành `no_vest` — **suy luận từ crop mẫu, tài liệu của Ultralytics không nêu, chưa kiểm chứng**.
- Dùng làm nền train (1.132 ảnh) và là nguồn duy nhất của val/test.

### Nguồn 3 — SH17 (thêm 2026-10-30)
- **Link:** https://github.com/ahmadmughees/SH17dataset ; tải qua Kaggle `mugheesahmad/sh17-dataset-for-ppe-detection` (archive.zip 14.1 GB, tải thủ công).
- **Giấy phép:** CC BY-NC-SA 4.0 (README: chỉ cho giáo dục/nghiên cứu/huấn luyện ML); ảnh lấy từ Pexels (giấy phép Pexels). Phải ghi nguồn; model dùng cho học tập, không thương mại.
- **Nội dung:** 8.099 ảnh, 17 lớp. Ánh xạ id→tên kiểm bằng cách đối chiếu số box YOLO với tên trong voc_labels (300 file đầu khớp từng lớp).
- **Dùng:** chỉ ảnh có `Glasses` hoặc `Gloves`, bỏ 117 ảnh có mũ/áo thấy rõ mà không gán nhãn (`src/scan_sh17_missing_labels.py`, duyệt tay 24 mẫu: ~1/3 ca mũ, ~4/5 ca áo là nhãn thiếu thật) → 2.538 ảnh (`src/prep_sh17.py`, cạnh dài 960 px). Ánh xạ: Person→person, Glasses→goggles, Gloves→gloves, Helmet→helmet, Safety-vest→vest; các lớp khác bỏ.
- **Chỉ vào train** (0 ảnh bị loại vì gần giống val/test). Không có nhãn `no_*`.
- **Còn lại:** nhãn thiếu ở ngưỡng conf<0.6 và ở lớp găng/kính chưa dò được; chỉ duyệt tay 24 ảnh.
