# Dataset audit — phản hồi thầy tuần 11 (2026-10-24)

Thầy góp ý: dữ liệu chưa đủ để xác nhận train/valid/test, chưa biết có những loại mũ nào, chưa phân biệt người trong vùng. Kết quả kiểm tra đo thật bên dưới.

## 1. Ảnh gần giống giữa các tập (`src/check_near_duplicates.py`, dHash)

MD5 chỉ bắt file trùng từng byte. dHash bắt cùng một ảnh gốc nhưng khác augmentation (đã xem tay 2 cặp d=0: đúng là cùng ảnh, khác lật/crop/màu).

| So sánh | Ảnh có bản sao gần (≤4 bit) | ≤8 bit |
|---|---|---|
| valid vs train | 51/1413 (3.6%) | 61 (4.3%) |
| test vs train | 19/706 (2.7%) | 25 (3.5%) |
| test vs valid | 9/706 (1.3%) | 10 (1.4%) |

Ảnh hưởng lên điểm: đo lại `helmet_v1_best.pt` @640 trên valid đã loại 61 ảnh (1352 ảnh còn lại): `head` AP50 0.962 (trước 0.960), `helmet` 0.982 (0.980), mAP50 0.660 (0.660). **Rò rỉ có thật nhưng nhỏ, không làm điểm đẹp lên.** Hạn chế còn lại: cả 3 tập vẫn cùng một nguồn ảnh, nên điểm này chưa nói gì về rig/camera của nhóm.

## 2. Loại mũ — mẫu ngẫu nhiên (seed 0, `src/sample_helmet_types.py`)

120 box `helmet` từ train (box ≥24px), xem tay từng crop:
- Gần như toàn bộ là mũ bảo hộ công trường (nón cứng). Nghi ngờ nhãn sai: ~2/120 (#46 tóc bạc; #21 mũ xám vành rộng chưa chắc là nón cứng).
- Màu theo tự động (HSV): đỏ 35, vàng 23, cam 23, trắng 17, xanh dương 12, xám 5, đen 4, xanh lá 1. **Tự động sai khoảng 10%** khi xem tay (cam↔vàng, trắng↔vàng, xanh↔đỏ) — dùng làm số ước lượng, không phải số chính xác.
- Kiểu: đa số nón cứng vành ngắn, có vài mũ vành rộng và mũ có lỗ thông gió.

60 box `head` (đầu trần): đa số là đầu trần, nhiều người mặc sơ mi/vest (kiểu quản lý, khách). Nhưng có **~4/60 (≈6%) không phải đầu trần thật**: mũ lưỡi trai/mũ đỏ (#1), mũ y tế xanh (#35), mũ quân đội (#54), ảnh không phải đầu người (#13, con cú). Nghĩa là lớp "có mũ nhưng không phải mũ bảo hộ" **không có nhãn riêng**, đang lẫn trong `head`.

## 3. Chưa làm được từ dữ liệu có sẵn
- Phân vai trò (thi công / giám sát / khách): ảnh không chứa thông tin này. Chỉ suy ra được từ quy ước (màu mũ, áo phản quang, trang phục) và quy ước đó phải do nhóm/công trường đặt.
- Gán lại nhãn toàn bộ 6.677 box `head` + 19.747 box `helmet` bằng tay: không khả thi trong kỳ này.

## 4. Dataset chia lại theo vai trò (`src/build_role_dataset.py` → `data/helmet_roles/`)

Quy ước phổ thông (màu mũ là **đại diện** cho vai trò, không xác minh được vai trò thật từ ảnh):

| Lớp mới | Màu mũ | Ý nghĩa |
|---|---|---|
| `head` | — | đầu trần (giữ nguyên nhãn gốc, còn ~6% nhiễu: mũ lưỡi trai/y tế/quân đội) |
| `helmet_supervisor` | trắng (+ xám = trắng bị bóng) | kỹ sư / giám sát |
| `helmet_worker` | vàng, cam | công nhân |
| `helmet_other` | xanh dương, xanh lá, đỏ, đen | khách / chuyên trách (an toàn, PCCC, điện) |

Bỏ lớp `person` (AP50 ~0.04, không dùng cho quyết định). Không tách được "có mũ không phải mũ bảo hộ" vì không có nhãn.
valid/test chỉ giữ ảnh không có bản sao gần trong train (valid 1352, test 681 ảnh); train giữ nguyên 4916.

| Tập | head | supervisor | worker | other |
|---|---|---|---|---|
| train (4916 ảnh) | 4612 | 3136 | 4783 | 6000 |
| valid (1352) | 1313 | 848 | 1201 | 1721 |
| test (681) | 715 | 355 | 646 | 858 |

Độ chính xác màu: xem tay 60 box ngẫu nhiên của valid, sai 4/60 (~7%: mũ xanh→trắng, mũ cam→"other" ×2, mũ nâu→trắng) và 1 box nguồn không phải mũ. Ranh giới cam/đỏ là chỗ dễ sai nhất. **Chưa huấn luyện lại** trên bộ nhãn này nên chưa có số mAP.

## 5. Tách `head` thành "không đội mũ" và "đội mũ không phải mũ bảo hộ" (`data/helmet_roles_by_class/`)

Cách làm (`src/flag_headwear.py`, `src/split_head_folder.py`): script lọc 339/4414 crop `head` có vùng đỉnh đầu không giống tóc/da (ngưỡng 0,5), xem tay 7 tờ ảnh, chọn bằng mắt. Phần còn lại giữ là `no_hat`.

| Folder | train | valid | test | Ghi chú |
|---|---|---|---|---|
| `no_hat` | 3021 | 862 | 495 | không đội gì |
| `hat_not_safety` | 24 | 4 | 3 | mũ lưỡi trai, mũ rằn ri quân đội, mũ len, mũ y tế, mũ rộng vành, mũ trùm |
| `head_mislabeled_helmet` | 3 | 1 | 1 | nhãn gốc ghi `head` nhưng thực chất là mũ bảo hộ |

Giới hạn: bộ lọc tự động bỏ sót mũ tối màu (giống tóc đen). Lấy mẫu ngẫu nhiên 120 crop trong `no_hat`, khoảng 3–5 ảnh còn nghi là có mũ/không phải đầu, tức `no_hat` sạch ước ~96% và `hat_not_safety` thực tế có thể gấp 3–4 lần số đã tìm được. `hat_not_safety` quá ít (31 ảnh) để huấn luyện một lớp riêng; chỉ đủ làm ví dụ minh hoạ và tập kiểm tra nhỏ.

## 6. Dữ liệu bổ sung bên ngoài: Roboflow `cap/hat` (euc, CC BY 4.0, v3)

Tải về `data/helmet_suplement/cap-hat.v3i.yolov8/` (train 800 + valid 1016 = 1816 ảnh, 512x512 kéo giãn). Nguồn: https://universe.roboflow.com/euc-rswvc/cap-hat-rwszm (CC BY 4.0, phải ghi nguồn). Số box thật: motor helmet 2022, cap-hat 674, bareface 20, face mask-bandana 4, mask-full face mask 3, hoodie 1. Crop (≥24px) vào `data/helmet_roles_by_class/hat_not_safety_ext/` (`src/export_ext_hat_crops.py`).

Xem tay 60 crop ngẫu nhiên mỗi lớp:
- `cap-hat` (674 crop): đúng là mũ không phải mũ bảo hộ — mũ lưỡi trai, mũ phớt, mũ len, mũ rộng vành, mũ bere. Khoảng 4/60 là tranh vẽ/hoạt hình, một số ảnh mờ. Hầu hết là chân dung cận mặt (khác miền với công trường).
- `motor helmet` (2022 crop): **nhãn không đáng tin.** Không có mũ bảo hiểm xe máy; ảnh là mũ lưỡi trai/phớt/len bị augmentation mạnh (xoay, lật, nhiễu, đen trắng), nhiều ảnh trùng gốc. Đã đổi tên folder thành `motor_helmet_label_unreliable`, không dùng.
- `hoodie` chỉ 1 box; các lớp khẩu trang/bareface < 25 box: bỏ qua.

Kết luận: nguồn này cho thêm ~674 ảnh `hat_not_safety` thật (mũ lưỡi trai, phớt, len…), tập trung vào chân dung nên chỉ làm tập kiểm tra/bổ sung. **Chưa có ảnh mũ bảo hiểm xe máy** từ nguồn nào đã xem; cần nguồn khác hoặc tự chụp.

Cập nhật 2026-10-24: 674 crop `cap-hat` đã gộp vào `data/helmet_roles_by_class/hat_not_safety/{train,valid}` (tiền tố `ext_`) → hat_not_safety = train 303 / valid 399 / test 3. Crop `motor helmet` (nhãn sai) chuyển sang `_unused_motor_helmet_label_unreliable/`. Lưu ý valid > train vì bộ ngoài chia 800/1016; nếu huấn luyện, nên xáo lại.

## 7. Chia lại theo màu mũ (thay các folder vai trò) — 2026-10-24

Tra cứu: không tìm thấy văn bản pháp luật Việt Nam quy định màu mũ theo vai trò (QCVN 06:2012/BLĐTBXH, TCVN 6407:1998, QCVN 18:2021/BXD chỉ nói chất lượng mũ / PPE; chưa đọc toàn văn). Cách phân màu theo vai trò chỉ là thông lệ từng công trường (ví dụ diễn đàn vnreview). Vì vậy `data/helmet_roles_by_class/` giờ chia theo **màu** (`src/export_color_crops.py`); folder `helmet_supervisor/worker/other` đã xoá (tái tạo được bằng `src/export_role_crops.py` từ `data/helmet_roles/`).

| Folder | train | valid | test |
|---|---|---|---|
| helmet_white | 2144 | 564 | 258 |
| helmet_yellow | 2241 | 576 | 331 |
| helmet_orange | 1169 | 291 | 145 |
| helmet_red | 2717 | 837 | 397 |
| helmet_blue | 1448 | 350 | 190 |
| helmet_green | 40 | 19 | 4 |
| helmet_black | 228 | 70 | 35 |
| helmet_grey | 75 | 28 | 11 |

Cùng với `no_hat` và `hat_not_safety` (mục 5–6). `data/helmet_roles/` (YOLO 4 lớp theo ánh xạ ví dụ trắng→giám sát, vàng/cam→công nhân, còn lại→other) giữ nguyên, chưa dùng huấn luyện.

## 8. Sửa nhãn màu mũ (white/black lẫn màu khác) — 2026-10-24

Lỗi người dùng báo: folder `helmet_white` và `helmet_black` chứa mũ màu khác. Nguyên nhân: gán màu bằng đa số điểm ảnh, ảnh tối/mờ bị coi là black/grey, nền sáng thành white. Đã chuyển logic ra `src/helmet_colour.py` (dùng chung `export_color_crops.py` và `build_role_dataset.py`): hue chỉ gán khi đủ điểm ảnh bão hòa (cam cần bão hòa cao vì da người là cam nhạt); white chỉ khi sáng và gần như không có điểm ảnh bão hòa; còn lại vào `helmet_unclear` (gộp black + grey cũ, quá ít và nhiễu).

| Folder | train | valid | test |
|---|---|---|---|
| helmet_white | 1072 | 295 | 140 |
| helmet_yellow | 2479 | 624 | 360 |
| helmet_orange | 1144 | 293 | 147 |
| helmet_red | 2890 | 876 | 416 |
| helmet_blue | 1747 | 432 | 223 |
| helmet_green | 67 | 23 | 5 |
| helmet_unclear | 663 | 192 | 80 |

Xem tay sau sửa (mẫu nhỏ): white ~95% đúng (3/60 nghi), yellow ~90% (10), red/blue ~80% (10), orange lẫn yellow, green sai nhiều (~4/10, chỉ 95 ảnh). `helmet_unclear` chưa duyệt, ~40% (trong 60 ảnh xem) là mũ trắng có logo/nền màu. Đã dựng lại `data/helmet_roles/` bằng logic mới nhưng supervisor bị thiếu (mũ trắng trong unclear xếp vào other): chưa dùng để huấn luyện.
