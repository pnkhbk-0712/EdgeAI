# Step 3 và Step 4 làm lại theo ý thầy: chọn model có so sánh, rồi mới huấn luyện

Ý thầy: Step 3 phải **chọn kiến trúc phù hợp trước** (đặt tiêu chí, so sánh các ứng viên, nói vì sao chọn), **sau đó mới huấn luyện**. Step 4 cũng phải **so sánh các cách tối ưu** và nói vì sao chọn cách này. Báo cáo cũ đã nhảy thẳng vào "dùng YOLOv8n" nên bị chê. File này là khung viết lại. **Phần "số của nhà sản xuất" chưa phải nhóm đo; phần "nhóm đo" ghi rõ cái nào đã có, cái nào chưa.**

## Step 3.1 — Chọn kiến trúc

### Bước 1: tiêu chí (đặt trước khi chọn)

| # | Tiêu chí | Vì sao quan trọng với bài toán này |
|---|---|---|
| 1 | Độ trễ trên Raspberry Pi 4B (CPU) < 300 ms/khung | Thiết bị đích chỉ có CPU; đây là ràng buộc cứng của Step 1 |
| 2 | Phát hiện được vật nhỏ (găng, kính) | Hai trong bốn món cần kiểm là vật nhỏ |
| 3 | Kích thước và bộ nhớ nhỏ | Pi 4B chỉ 4 GB RAM |
| 4 | Dễ xuất ONNX, chạy CPU, có sẵn theo dõi người (`model.track`) | Cần ghép món vào từng người qua nhiều khung hình |
| 5 | Công sức huấn luyện và có sẵn trọng số học trước | Dữ liệu ít, thời gian có hạn, cần fine-tune |
| 6 | Giấy phép | Dự án học tập; cần biết ràng buộc nếu dùng thật |

### Bước 2: danh sách ứng viên và so sánh

**Số của nhà sản xuất**, đo trên bộ COCO (80 lớp), phần cứng của họ, **không phải Raspberry Pi và không phải dữ liệu PPE**. Dùng để so sơ bộ, không thay cho số nhóm đo.

| Ứng viên | mAP COCO (50-95) | Tham số | FLOPs | Tốc độ CPU (nhà sản xuất) | Giấy phép | Ghi chú |
|---|---|---|---|---|---|---|
| **YOLOv8n** | 37,3 | 3,2 M | 8,7 B | 80,4 ms (CPU ONNX) | AGPL-3.0 | Đang dùng |
| YOLOv8s | 44,9 | 11,2 M | 28,6 B | 128,4 ms | AGPL-3.0 | Cùng họ, lớn hơn |
| **YOLO11n** | 39,5 | 2,6 M | 6,5 B | 56,1 ms | AGPL-3.0 | Đời mới hơn: chính xác hơn và nhẹ hơn YOLOv8n |
| YOLO11s | 47,0 | 9,4 M | 21,6 B | 90,0 ms | AGPL-3.0 | Cùng họ, lớn hơn |
| EfficientDet-Lite0 (TFLite) | 25,7 | — | — | 37 ms (Pixel 4, 4 luồng) | (chưa kiểm) | Họ TensorFlow Lite, hợp thiết bị di động |
| SSD-MobileNetV2 320×320 (TFLite) | 20,2 | — | — | 24 ms (Pixel 4, 4 luồng) | (chưa kiểm) | Nhanh nhưng chính xác thấp |

Nguồn: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), [TensorFlow blog 2021](https://blog.tensorflow.org/2021/06/easier-object-detection-on-mobile-with-tf-lite.html). Hai dòng TFLite đo trên điện thoại Pixel 4 nên không so thẳng được với số CPU của YOLO; mình không tìm thấy số Raspberry Pi 4 cho chúng.

### Bước 3: đánh giá từng ứng viên theo tiêu chí

| Ứng viên | 1. Độ trễ Pi | 2. Vật nhỏ | 3. Kích thước | 4. Hệ sinh thái | 5. Công sức train | Kết luận |
|---|---|---|---|---|---|---|
| YOLOv8n | **Đã đo: 209 ms model, 289 ms pipeline ở 416** | Yếu vừa (găng 0,72, kính 0,77 ở 640) | 12,7 MB ONNX | Tốt, tích hợp tracking | Có trọng số COCO, train 22 phút trên GPU | **Dùng được, đã có số đo** |
| YOLOv8s | **Ước lượng 335–680 ms** ở 416 (suy từ tỉ lệ tốc độ CPU 1,6× hoặc FLOPs 3,3×); chưa đo | Có thể tốt hơn (model lớn hơn) | lớn hơn khoảng 3 lần | Như v8n | Train lâu hơn khoảng 2–3 lần | **Loại: gần như chắc chắn vượt 300 ms** (ước lượng, cần đo xác nhận) |
| YOLO11n | **Ước lượng khoảng 150 ms** ở 416 (suy từ tỉ lệ tốc độ CPU 0,70×); chưa đo | Chưa biết trên dữ liệu PPE | nhỏ hơn v8n | Như v8n, cùng thư viện | Như v8n | **Ứng viên mạnh nhất, phải đo trên dữ liệu của nhóm** |
| EfficientDet-Lite0 / SSD-MobileNetV2 | Không có số Pi | SSD-MobileNetV2 kém vật nhỏ ở 320×320 | nhỏ | Phải dùng bộ công cụ TensorFlow riêng, không có tracking sẵn | Cao: pipeline huấn luyện mới | **Loại: tốn công, chính xác COCO thấp, không có số đo cho Pi** |

### Bước 4: quyết định

**Tạm thời giữ YOLOv8n làm model chính**, vì là model duy nhất nhóm đã huấn luyện, đo trên Pi và có kết quả đầy đủ. **Nhưng phải nói thật: YOLO11n về giấy tờ nhẹ hơn và chính xác hơn, nhóm chưa thử.** Thầy rất dễ hỏi "sao không YOLO11n?". Cách trả lời trung thực là hoặc (a) đã đo cả hai, hoặc (b) chưa đo, ghi rõ.

## Việc cần đo để Step 3.1 có số thật (đề xuất)

Huấn luyện **YOLO11n** (và YOLOv8s nếu còn thời gian) trên đúng dữ liệu C′, đúng cấu hình đã dùng cho YOLOv8n (60 epoch, batch 16, ảnh 640), rồi đo cùng 111 ảnh test và đo trên Pi. Lệnh mẫu (chạy ở thư mục `prototype`):

```bash
yolo detect train data=data/ppe_merged/data_sampled.yaml model=yolo11n.pt imgsz=640 epochs=60 batch=16 patience=30 device=0 project=D:/DOANMINHHIEU/STUDIES/Ky9/01_Ai/_tools/yolo_runs name=ppe_C_yolo11n plots=True
```

- **Cần bạn đồng ý tải:** `yolo11n.pt` (và `yolov8s.pt` nếu làm) từ kho `ultralytics/assets` trên GitHub, nhỏ vài chục MB. Mình chưa tải gì.
- **Thời gian:** YOLOv8n mất khoảng 22 phút cho 60 epoch trên máy bạn (GPU). YOLO11n tương tự; YOLOv8s có thể gấp 2–3 lần.
- **Nhiệt độ:** bạn đã lo cho máy. Mình sẽ chạy từng model một, theo dõi nhiệt GPU, dừng nếu quá nóng.
- **Đo trên Pi:** Hùng chạy `benchmark_pi.py` cho model mới (cần xuất ONNX 416 và 640).
- **Lần này bật `plots=True`** để có sẵn `results.png`, đường cong, ma trận nhầm lẫn cho cả hai model.

Kết quả cần điền vào bảng: AP50 từng món (mũ, áo, găng, kính), tham số, kích thước ONNX, độ trễ laptop và Pi, thời gian huấn luyện.

## Step 4 — so sánh các cách tối ưu và lý do chọn

Thầy muốn thấy nhóm **thử nhiều cách, so sánh, rồi chọn**, không chỉ nêu cách đã chọn:

| Cách tối ưu | Đã làm? | Kết quả đo | Quyết định và lý do |
|---|---|---|---|
| Giảm độ phân giải 640 → 416 | **Có, đo trên Pi** | Model 491 → 209 ms; pipeline ≈565 → ≈289 ms; găng 0,722 → 0,664, kính 0,772 → 0,641 | **Chọn**: cách duy nhất đưa pipeline về sát 300 ms; mất găng/kính chấp nhận được |
| INT8 dynamic | Có, trên laptop | Nhỏ hơn 3,7 lần (12,75 → 3,47 MB), mAP50 0,563 → 0,560, nhưng chậm hơn (21 → 33 ms) | **Không chọn**: không nhanh hơn, mục tiêu là tốc độ |
| INT8 static | Có, trên laptop | mAP50 sập về 0,000 (cấu hình mặc định) | **Không chọn**: hỏng; chưa thử cách hiệu chỉnh khác |
| Pruning | Không | — | Chưa thử; lập luận: điểm nghẽn là độ phân giải, không phải số tham số |
| Knowledge distillation | Không | — | Chưa thử; student đã là model nhỏ nhất; cần train teacher và tự viết loss |
| **Đổi sang model nhỏ hơn (YOLO11n)** | **Chưa** | Ước lượng nhanh hơn khoảng 30% theo tốc độ CPU nhà sản xuất | **Cần đo.** Đây có thể là cách tối ưu hiệu quả hơn pruning/distillation |
| Hạ xuống 320 | Chỉ đo trên model mũ cũ (129,7 ms trên Pi); từng thấy mất lớp đầu | — | Chưa đo trên model PPE; rủi ro mất găng/kính lớn |

Báo cáo cũ đã có phần này (mục 4.1–4.3), cần **thêm cột "vì sao chọn/không chọn"** (đã có ở bảng trên) và thêm dòng YOLO11n nếu đo được.

## Cách trình bày với thầy (30 giây)

"Step 3 em làm đúng thứ tự: đặt tiêu chí, lập danh sách ứng viên gồm YOLOv8n, YOLOv8s, YOLO11n và các model TensorFlow Lite, so sánh theo độ trễ trên Pi, vật nhỏ, kích thước, hệ sinh thái. YOLOv8s bị loại vì ước tính vượt 300 ms; EfficientDet và SSD bị loại vì chính xác thấp và tốn công huấn luyện lại. Em chọn YOLOv8n vì đã đo trên Pi, [và so với YOLO11n em đã đo như sau ...]."

Phần trong ngoặc vuông **chỉ nói nếu đã đo thật**.

## Chưa làm / rủi ro

- Các số "ước lượng" của YOLOv8s và YOLO11n trên Pi suy từ tỉ lệ số của nhà sản xuất, **chưa đo**; ghi "ước lượng" khi trình bày.
- Giấy phép của hai model TensorFlow Lite mình chưa kiểm.
- Lý do thật nhóm chọn YOLOv8n lúc đầu (cùng họ model mũ giai đoạn đầu, có sẵn công cụ) là lý do thực dụng, không phải kết quả so sánh. Nếu thầy hỏi, nói thật và trỏ vào bảng so sánh vừa làm.
- Chưa sửa báo cáo v9 và kịch bản thuyết trình theo khung mới; chờ bạn quyết định có đo YOLO11n hay không.
