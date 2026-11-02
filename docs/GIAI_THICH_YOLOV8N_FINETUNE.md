# YOLOv8n và fine-tune — giải thích cho phần Step 3–4

Mục đích: để trả lời được khi thầy hỏi "model này là gì, vì sao chọn nó, huấn luyện như thế nào". Số liệu của nhóm lấy từ báo cáo v8 và `docs/RUNLOG.md`.

## 1. YOLOv8n là gì

**Bài toán nó giải: object detection.** Nhìn một ảnh, trả về danh sách các hộp, mỗi hộp gồm: vị trí (x, y, rộng, cao), loại vật (ví dụ `helmet`, `vest`) và độ tin cậy (0–1). Khác phân loại ảnh (chỉ trả "ảnh này là gì"), detection nói *vật ở đâu*.

**YOLO = "You Only Look Once".** Model nhìn ảnh **một lần** qua mạng nơ-ron và ra luôn tất cả hộp. Cách cũ (hai giai đoạn) đề xuất vùng rồi mới phân loại từng vùng, chính xác nhưng chậm. YOLO chọn tốc độ, nên hợp với thiết bị yếu như Raspberry Pi.

**v8 = thế hệ thứ 8, do hãng Ultralytics phát hành.** Nhóm dùng thư viện `ultralytics` (có sẵn huấn luyện, xuất ONNX và theo dõi người `model.track`, nên không phải tự viết).

**n = "nano"**, bản nhỏ nhất trong họ v8 (n, s, m, l, x, càng về sau càng nặng và chính xác hơn).

| Thông số của model nhóm (`ppe_v1`) | Giá trị |
|---|---|
| Tham số | 3.139.947 (khoảng 3,1 triệu) |
| Tính toán một ảnh 640×640 | 8,8 GFLOPs |
| File ONNX | 12,66 MB (416) / 12,75 MB (640) |
| Số lớp đầu ra | 9 (person, helmet, no_helmet, vest, no_vest, gloves, no_gloves, goggles, no_goggles) |

**Ba khối của mạng (hiểu ở mức ý tưởng):**
1. **Backbone** — rút đặc trưng từ ảnh (cạnh, hình dạng, rồi bộ phận như đầu, tay).
2. **Neck** — trộn đặc trưng ở nhiều kích thước để vật nhỏ (găng, kính) và vật lớn (người) đều thấy được.
3. **Head** — đoán hộp và loại vật. YOLOv8 không dùng "anchor box" định sẵn (anchor-free) nên đơn giản hơn các đời cũ.

Sau mạng có bước **NMS** (loại bớt các hộp trùng nhau trên cùng một vật).

**Vì sao nhóm chọn bản n:** Pi 4B chỉ có CPU. Đo thực tế: model chạy khoảng 209 ms/khung ở 416 và 491 ms ở 640 (chỉ model). Bản lớn hơn nặng hơn nhiều lần nên gần như chắc chắn không đạt ngưỡng 300 ms. **Đây là lập luận, nhóm chưa huấn luyện YOLOv8s để đo thật** (báo cáo ghi rõ chuyện này).

**Vì sao găng và kính khó với YOLOv8n:** đây là vật nhỏ, và model nhỏ ít sức chứa hơn. Ở 416, ảnh bị thu nhỏ nên vật nhỏ mất chi tiết: AP50 găng 0,722 → 0,664 và kính 0,772 → 0,641.

## 2. Fine-tune là gì

**Ý tưởng:** không dạy model từ con số 0. Lấy một model **đã được huấn luyện sẵn** trên bài toán lớn, rồi **huấn luyện tiếp một lượt ngắn** trên dữ liệu của mình. Cách này gọi là **transfer learning** (học chuyển giao); fine-tune là bước huấn luyện tiếp đó.

**Ví dụ dễ hiểu:** một người đã biết đọc và biết nhận ra đồ vật nói chung, nay chỉ cần học thêm "đâu là mũ bảo hộ, đâu là áo phản quang" — nhanh hơn nhiều so với dạy một đứa trẻ từ đầu.

**Điểm xuất phát của nhóm: `yolov8n.pt`.** Đây là model Ultralytics đã huấn luyện sẵn trên bộ COCO (80 loại vật thông thường như người, xe, ghế...). Nó đã biết:
- nhận ra cạnh, góc, hình dạng, kết cấu;
- nhận ra người và nhiều vật quen thuộc.

Nó **chưa biết** mũ bảo hộ, áo phản quang, găng, kính trong ngữ cảnh PPE.

**Quá trình nhóm làm (khớp với `docs/PPE_TRAINING.md`):**
1. Nạp `yolov8n.pt`. Phần đầu ra (head) được thay để có **9 lớp** của nhóm thay vì 80 lớp COCO; phần backbone và neck giữ lại phần lớn kiến thức cũ.
2. Cho model xem ảnh train có nhãn (hộp + tên lớp). Với mỗi ảnh, so dự đoán với nhãn thật, tính **loss** (độ sai), rồi chỉnh trọng số để giảm sai.
3. Lặp lại qua toàn bộ ảnh nhiều lần. Mỗi lượt qua hết tập train gọi là **1 epoch**.
4. Sau mỗi epoch, đo trên tập **val** (ảnh không dùng để học) để xem model có tiến bộ thật không, rồi giữ bản tốt nhất (`best.pt`).
5. Cuối cùng đánh giá một lần trên tập **test** (111 ảnh) và báo kết quả.

**Cấu hình thực tế đã dùng:**

| Cấu hình | Dữ liệu train | Epoch | Batch | Cỡ ảnh | patience |
|---|---|---|---|---|---|
| A | Construction-PPE (1.132 ảnh) | 80 | 16 | 640 | 30 |
| C′ (được chọn) | A + 2.500 ảnh PPEv5 (3.632 ảnh) | 60 | 16 | 640 | 30 |
| C″ | C′ + 2.538 ảnh SH17 (6.170 ảnh) | 60 | 16 | 640 | 30 |

- **Batch 16:** mỗi lần chỉnh trọng số, model xem 16 ảnh cùng lúc.
- **patience 30:** nếu 30 epoch liền không cải thiện trên val thì dừng sớm.
- **Nơi chạy:** A và C′ chạy trên máy cá nhân; C″ chạy trên Google Colab (GPU T4).
- **Augmentation có sẵn của Ultralytics** (ghép ảnh mosaic, đổi màu HSV, lật ngang) tạo thêm biến thể từ dữ liệu có sẵn.

**Vì sao fine-tune thay vì train từ đầu:**
- Dữ liệu của nhóm nhỏ (vài nghìn ảnh). Train từ đầu cần hàng trăm nghìn ảnh mới ổn.
- Hội tụ nhanh hơn (60–80 epoch thay vì hàng trăm).
- Kết quả thường tốt hơn khi dữ liệu ít.

**Điều fine-tune không làm được:** nếu dữ liệu thiếu nhãn hoặc khác miền (ảnh cận cảnh, ảnh quảng cáo), model vẫn học sai theo. Bằng chứng trong dự án: thêm SH17 (C″) **không** cải thiện găng/kính.

## 3. Các thước đo bạn sẽ nói tới

- **IoU:** hộp đoán trùng bao nhiêu với hộp thật (0–1). Thường coi đúng khi IoU ≥ 0,5.
- **Precision:** trong các lần model báo "có món", bao nhiêu lần đúng.
- **Recall:** trong các món thật có trong ảnh, model tìm thấy bao nhiêu.
- **AP50:** điểm trung bình của một lớp, tính độ chính xác theo mức IoU 0,5. **mAP50** là trung bình của mọi lớp. Nhóm báo AP50 **từng lớp** vì mAP chung bị kéo bởi lớp yếu và không cho biết món nào kém.
- **Precision/recall "thiếu món":** nhóm đo riêng cho quyết định cuối ("người này thiếu món X"), không chỉ cho từng hộp. Hệ thống ưu tiên recall cao (thà báo nhầm hơn bỏ sót).

## 4. Nối với yêu cầu của thầy ở Step 3–4

| Thầy kiểm | Nhóm trả lời |
|---|---|
| Kiến trúc có hợp thiết bị yếu không | YOLOv8n, 3,1 triệu tham số, chạy được trên Pi bằng ONNX Runtime CPU |
| Có so với model khác không | Chưa có số đo; chỉ có lập luận vì sao không chạy YOLOv8s. So 3 cấu hình dữ liệu A/C′/C″ trên cùng test |
| Huấn luyện ở đâu, cách nào, có log không | Máy cá nhân (A, C′) và Colab (C″); fine-tune từ `yolov8n.pt`; có `results.csv` của A và C′, vẽ ở Hình 3.3. Log C″ nằm trên Colab, chưa lấy |
| Metric hợp bài toán không | AP50 từng món + precision/recall "thiếu món" |
| Ngưỡng chấp nhận đặt trước không | Ngưỡng găng/kính đặt **sau** khi có số đo; ghi rõ trong báo cáo |
| Có thử tối ưu không | Giảm độ phân giải (có tác dụng), INT8 (không nhanh hơn), pruning/distillation (không làm, kèm lý do) |
| Trước và sau tối ưu đo gì | Độ chính xác, kích thước, độ trễ. **RAM, năng lượng, INT8 trên Pi chưa đo** |
| Cần tối ưu không | Có: 640 không đạt 300 ms trên Pi; 416 đạt sát |

## 5. Câu hỏi hay gặp và trả lời ngắn

- **YOLOv8n khác YOLOv8s ở đâu?** Cùng kiến trúc, bản n nhỏ và nhanh hơn, bản s lớn hơn và chính xác hơn nhưng chậm hơn.
- **Pretrained weights lấy từ đâu?** `yolov8n.pt` của Ultralytics, huấn luyện trên COCO (80 lớp).
- **Fine-tune có phải train lại toàn bộ không?** Có huấn luyện lại, nhưng bắt đầu từ trọng số đã biết nên nhanh và cần ít dữ liệu hơn; lớp đầu ra được thay cho 9 lớp của nhóm.
- **Sao không train từ đầu?** Dữ liệu ít, tốn thời gian, kết quả thường kém hơn.
- **Overfitting là gì và nhóm kiểm thế nào?** Model thuộc lòng ảnh train mà không tổng quát. Nhóm dùng tập val riêng để chọn `best.pt` và tập test không dùng để học để báo cáo; nhóm còn loại ảnh gần trùng giữa train và val/test bằng dHash.
- **Sao mAP50 chỉ 0,56?** Là trung bình của 9 lớp, trong đó các lớp `no_*` khó (ví dụ `no_goggles` rất ít ảnh trong test: 33 box). Mũ 0,91 và áo 0,87 tốt; găng và kính yếu hơn.
- **Model có nhận kính cận thường là kính bảo hộ không?** Thường không; dữ liệu train là kính bảo hộ, và ảnh thử trong lớp cho thấy kính cận thường hay không được nhận.

## 6. Phần nhóm chưa chắc / chưa làm

- Chưa so sánh kiến trúc bằng số đo (YOLOv8s).
- Chưa kiểm chứng nguyên nhân C″ không tốt hơn C′.
- Mỗi cấu hình chỉ train một lần, tập test 111 ảnh nhỏ, nên chênh khoảng 0,02 là nhiễu.
- Mô tả cấu trúc mạng ở trên là mức khái quát; nếu thầy hỏi chi tiết từng lớp thì mở `ultralytics` hoặc tài liệu chính thức, không tự suy đoán.
