# Thuyết trình + demo tuần 12 (10 phút) — Step 3, Step 4 và sản phẩm chạy thật

Thầy yêu cầu mỗi nhóm **chuẩn bị sẵn sản phẩm và demo, trong 10 phút**. Kịch bản chia: **khoảng 4 phút nói (Step 3–4), 4 phút demo, 1–2 phút kết luận và hỏi đáp.** Nếu 10 phút đã gồm cả hỏi đáp thì rút phần nói còn 3,5 phút (bỏ đoạn INT8, mục 3:00–3:20).

Người nói: Hiếu (Data & Model); người thao tác demo: bạn tự thao tác trên laptop (người mặc đồ bảo hộ: Hoàng hoặc Hùng, bạn chốt). Mở sẵn **`docs/BaoCao_Tuan12_Step1-4_v9.docx`** ở các hình ghi trong cột "Chiếu". Mọi số đã có trong báo cáo hoặc `docs/RUNLOG.md`. Khái niệm (YOLOv8n, fine-tune): `docs/GIAI_THICH_YOLOV8N_FINETUNE.md`; cách đọc các hình kết quả: `docs/GIAI_THICH_HINH_KET_QUA.md`.

**Quy ước:** những gì đã bị bỏ khỏi báo cáo v9 thì **không chủ động nói**: căn cứ pháp lý (QCVN), việc bỏ phân vùng rủi ro, lý do C″ không giúp, RAM/năng lượng chưa đo, danh sách việc còn thiếu và bảng phân công. Các phần đó chỉ có ở mục câu hỏi bên dưới để trả lời **nếu thầy hỏi**, và trả lời thật.

## Phân bổ 10 phút

| Giờ | Việc |
|---|---|
| 0:00–0:40 | Mở đầu, Step 1–2 nói qua |
| 0:40–2:20 | Step 3 (model, huấn luyện, A/C′/C″) |
| 2:20–3:40 | Step 4 (416, INT8, pruning/distillation) |
| 3:40–7:40 | **Demo trên laptop** (mục dưới) |
| 7:40–8:20 | Kết luận |
| 8:20–10:00 | Hỏi đáp / dự phòng |

## Kịch bản nói (phần slide)

| Giờ | Chiếu | Lời nói |
|---|---|---|
| 0:00–0:40 | Tóm tắt, bảng 1.4, bảng 2.2 | "Tuần trước thầy góp ý là chỉ phát hiện mũ thì quá nhỏ, hệ thống phải đủ bộ bảo hộ. Nhóm đã làm lại Step 1 và Step 2: hệ thống giờ kiểm tra **bốn món** — mũ, áo phản quang, găng, kính — cho từng người. Dữ liệu từ ba bộ công khai có kiểm giấy phép: Construction-PPE, PPEv5 và SH17, tập test 111 ảnh đã loại ảnh gần trùng. Hôm nay em tập trung vào Step 3, Step 4 và demo." |
| 0:40–1:15 | Step 3.1, Hình 3.3 | "Step 3: nhóm chọn **YOLOv8n**, bản nhỏ nhất của họ YOLOv8, 3,14 triệu tham số, vì thiết bị đích là Raspberry Pi 4B chỉ có CPU. Nhóm **fine-tune** từ trọng số `yolov8n.pt` đã học sẵn trên COCO, thay phần đầu ra thành 9 lớp của nhóm, huấn luyện 60 đến 80 epoch. Hình 3.3 là đường cong huấn luyện của A và C′." |
| 1:15–2:20 | Hình 3.2, Hình 3.4–3.5 | "Nhóm so **ba cấu hình dữ liệu** trên cùng 111 ảnh test, đo AP50 từng món. A chỉ Construction-PPE: mAP50 0,512. **C′** thêm 2.500 ảnh PPEv5: **0,564, được chọn**. C″ thêm SH17: 0,514, không giúp; kính còn giảm từ 0,77 xuống 0,65. Kết quả theo món của C′: mũ 0,913, áo 0,867, găng 0,722, kính 0,772. Ngưỡng nhóm đặt: mũ và áo ≥ 0,80, găng và kính ≥ 0,70 ở mức tham khảo. Nhóm chưa chạy YOLOv8s, chỉ có lập luận vì YOLOv8n đã chỉ đạt khoảng 3,5 FPS trên Pi." |
| 2:20–3:00 | Hình 3.1, Hình 4.2 | "Step 4: kỹ thuật hiệu quả nhất là **giảm độ phân giải 640 xuống 416**. Trên Pi thật, độ trễ model giảm từ 491 xuống 209 ms, cả pipeline từ khoảng 565 xuống 289 ms, sát ngưỡng 300 ms. Cái giá: găng giảm xuống 0,664, kính xuống 0,641. Nhóm **chọn 416** để demo." |
| 3:00–3:20 | Hình 4.1 | "Nhóm cũng thử **INT8**: model nhỏ hơn 3,7 lần nhưng **không nhanh hơn** trên laptop, và INT8 static với cấu hình mặc định bị sập. INT8 chưa đo trên Pi." |
| 3:20–3:40 | Mục 4.3 | "Pruning và distillation nhóm **không làm**, kèm lý do trong báo cáo: model đã là bản nhỏ nhất, điểm nghẽn là độ phân giải. Kết luận Step 4: **có cần tối ưu**, vì 640 không đạt 300 ms trên Pi." |
| 3:40–3:50 | (chuyển sang demo) | "Giờ em xin demo sản phẩm. Em demo trên laptop vì demo trên Raspberry Pi hiện nhận dạng chưa ổn ở 416; số đo trên Pi em đã báo ở Step 4." |

## Demo sản phẩm trên laptop (4 phút, 3:40–7:40)

**Quyết định (31/10/2026): demo chạy trên laptop**, vì demo trên Raspberry Pi hiện nhận dạng quá kém ở 416 (Hình 4.3). Nói thẳng với thầy ngay đầu phần demo, đừng để thầy tự phát hiện: Pi là thiết bị đích và các số đo trên Pi đã báo ở Step 4; demo hôm nay chạy cùng pipeline trên laptop, model `.pt` ở 640 (nhận dạng tốt hơn bản 416 trên Pi).

### Chuẩn bị (làm trước giờ, tập ít nhất hai lần)

1. **Lệnh demo** (webcam laptop, phím Q để thoát):
   ```bash
   cd D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\LA4\prototype
   python src/demo_helmet.py --model models/ppe_v1_best.pt --source 0
   ```
   Cửa sổ "PPE compliance demo" hiện khung người (viền xanh lá = đủ món, đỏ = thiếu món, kèm nhãn "NO HELMET", ...). **Ô xanh mảnh** quanh từng món là món model nhận ra; chỉ cho thầy xem để thấy model đang "nhìn" gì.
2. **Máy:** cắm sạc, tắt ứng dụng nặng khác, mở sẵn cửa sổ lệnh để chạy ngay; chạy demo vài phút nên máy không nóng nhiều. Máy đã chạy bằng CPU (torch bản CPU), không cần GPU.
3. **Chọn nền và ánh sáng:** tường trơn, đủ sáng, không có màn hình hay bảng chữ phía sau người (nền rối làm nhận dạng kém).
4. **Dụng cụ:** mũ bảo hộ, áo phản quang, găng, và **kính bảo hộ thật** nếu có (kính cận thường thường không được nhận ra); thử trước xem kính nào model thấy.
5. **Vị trí người:** đứng cách camera khoảng 1,5–2 m, **thấy cả người** (đầu đến gần chân), **một người trong khung hình** khi demo từng món, hai tay giơ ra trước thân khi cần cho thấy găng.
6. **Bản dự phòng:** quay sẵn 30–60 giây demo thành công (quay màn hình) để nếu webcam hoặc máy lỗi thì chuyển sang video và nói rõ đó là bản ghi. Có thể dùng video lớp học `AQPnsH...mp4` ở thư mục `01_Ai` làm nguồn: `--source "đường dẫn video"` (xem cảnh nhiều người, kết quả sẽ có nhiều báo cáo thiếu).

### Các cảnh demo (khoảng 40 giây mỗi cảnh)

| # | Cảnh | Cái thầy nên thấy | Câu nói |
|---|---|---|---|
| 1 | Đủ 4 món | Khung người xanh, nhãn "person #n OK" | "Người đủ bốn món, hệ thống không báo." |
| 2 | Bỏ mũ | Nhãn đỏ "NO HELMET" | "Bỏ mũ, hệ thống nêu đúng món thiếu." |
| 3 | Bỏ áo | "NO VEST" | "Bỏ áo phản quang." |
| 4 | Bỏ găng hoặc kính | "NO GLOVES" / "NO GOGGLES" | "Găng và kính là hai món khó nhất, nhóm báo kết quả ở mức tham khảo." |
| 5 | Hai người, một đủ một thiếu | Chỉ người thiếu bị gắn nhãn | "Hệ thống gán món cho từng người riêng." |
| 6 | Nhật ký sự kiện (30 giây) | Mở `data/samples/helmet_events.jsonl`, hoặc `python src/review_events.py --list-unreviewed` | "Mỗi vi phạm ghi một dòng, không lưu hình, xóa sau 30 ngày." |

Nếu cảnh nào lệch (ví dụ vẫn báo thiếu găng khi đã đeo): **nói thẳng ngay** — "đây đúng là điểm yếu nhóm đã báo ở Step 3: găng và kính nhận chưa ổn định" — rồi sang cảnh tiếp, đừng cố chỉnh tại chỗ.

### Mong đợi trung thực (từ các lần thử gần nhất)

- Mũ và áo thường ổn (nhận 0,6–0,8 trong video thử).
- **Găng và kính hay chập chờn hoặc không nhận ra, đặc biệt kính cận thường.** Cảnh bạn thử trước đó người đủ 4 món vẫn bị báo thiếu một số món; demo bằng laptop nhìn chung tốt hơn Pi nhưng **mình chưa thử lại sau các sửa mới trên webcam của bạn**, nên cần tập thật.
- Đã làm: mỗi món chỉ gán cho đúng một người; vùng độ tin cậy 0,40–0,60 không còn biến món có thành thiếu. Chưa làm: làm mượt lâu hơn cho găng và kính; coi "không thấy kính/găng" là "không biết".

## Kết luận (7:40–8:20)

"Tóm lại: YOLOv8n fine-tune từ COCO, cấu hình C′ tốt nhất; giảm độ phân giải xuống 416 là cách tối ưu hiệu quả nhất trên Pi; hệ thống phát hiện đủ bốn món theo từng người, trong đó mũ và áo ổn, găng và kính còn yếu. Em xin hết, mời thầy hỏi."

**Không nói quá:** đừng nói hệ thống "hoạt động tốt". Số đúng là: mũ 0,91, áo 0,87 tốt; găng 0,72 và kính 0,77 ở 640 chỉ ở mức tham khảo; ở 416 dưới 0,70.

## Chọn cái để show

1. **Báo cáo v9** cho phần nói: Hình 3.3, 3.2, 3.4–3.5, 3.1, 4.2, 4.1. Mở sẵn từng hình, đừng cuộn tìm.
2. **Demo thật** là phần chính của 4 phút giữa (mục trên); Hình 4.3 (demo cũ trên Pi có lỗi) chỉ chiếu nếu thầy hỏi vì sao không demo trên Pi.
3. Ảnh Hình 2.1–2.4 (nhóm trong lớp) chỉ là ảnh thử rig, **không** nói đã dùng để huấn luyện hay đo.

## Nếu thầy bảo "mở dataset"

| Thầy hỏi | Mở cái gì |
|---|---|
| Xem dữ liệu | Thư mục `D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\Dataset_Show_Thay\` → `00_TONG_HOP.md`, rồi vào từng nguồn: `mau_anh_co_nhan\` (ảnh có khung nhãn), `thong_ke_lop.txt`, `README.txt`, `thu_muc_goc\` (ảnh gốc) |
| Link, giấy phép, ngày tải | `README.txt` trong từng nguồn, hoặc `docs/DATASET_AUDIT_PPE.md` |
| Số box theo lớp | `thong_ke_lop.txt`, hoặc Hình 2.6 trong báo cáo |
| Có rò rỉ train/test không | Báo cáo mục 2.4: loại 30/143 val và 30/141 test ảnh gần trùng (dHash) |
| Số đo thô | `docs/RUNLOG.md` (mục 25/10, 30/10, 31/10) |
| Bằng chứng đã chạy huấn luyện/đánh giá | `D:\DOANMINHHIEU\STUDIES\Ky9\01_Ai\Bang_Chung_Chay\` → `README.md`; mỗi cấu hình (A, C_prime, C_second) có `best.pt`, `args.yaml`, `results.csv` (A và C′), `val_test_640/` (ma trận nhầm lẫn, PR, ảnh dự đoán) |
| Datasheet phần cứng Pi 4B | trang Raspberry Pi 4B trên raspberrypi.com, **bạn mở sẵn**; mình chưa kiểm trang đó nên không ghi số vào đây |

Ảnh mẫu của PPEv5 và SH17 nhiều ảnh cận cảnh/quảng cáo. Nói trước là nhóm đã biết (báo cáo mục 2.6) để thầy không tưởng nhóm không phát hiện.

## Câu hỏi thầy có thể hỏi (trả lời ngắn, thật)

1. **YOLOv8n là gì, fine-tune là gì?** Xem `GIAI_THICH_YOLOV8N_FINETUNE.md`. Một câu: YOLOv8n là bản nhỏ nhất của model phát hiện vật thể YOLOv8; fine-tune là lấy trọng số đã học sẵn trên COCO rồi huấn luyện tiếp trên dữ liệu PPE của nhóm.
2. **Sao chọn 416?** 640 chỉ 1,8 FPS (≈565 ms) trên Pi, không đạt 300 ms; 416 ≈289 ms. Mất găng và kính (0,66 và 0,64), nên hai món này chỉ tham khảo.
3. **Sao không YOLOv8s?** Chưa thử. Lập luận từ tốc độ YOLOv8n trên Pi; nếu cần số thật thì khoảng 1,5 giờ trên Colab.
4. **Sao INT8 không nhanh hơn?** Đo trên laptop (dynamic 33 ms so với fp32 21 ms), cùng chiều với model mũ trên Pi (553 so với 472 ms). **Chưa đo INT8 trên Pi.**
5. **Sao thêm SH17 mà không tốt hơn?** Chưa biết chắc. Giả thuyết: SH17 không có nhãn `no_*`, ảnh khác phong cách, còn nhãn thiếu. Mỗi cấu hình chỉ train một lần, test 111 ảnh nên chênh khoảng 0,02 là nhiễu.
6. **Độ chính xác trên ảnh thật?** Chưa có. Số hiện tại là trên tập test công khai. Ảnh rig đang chờ Hoàng.
7. **Sao đo AP từng món, không chỉ mAP?** mAP chung bị kéo bởi các lớp yếu và không cho biết món nào kém.
8. **Có đường cong huấn luyện không?** Có cho A và C′ (Hình 3.3, từ `results.csv`, mAP50 trên tập val). C″ **không có** (log mất khi runtime Colab hết hạn); còn `best.pt` và kết quả đánh giá, xem `Bang_Chung_Chay`.
9. **Năng lượng, RAM?** Chưa đo.
10. **Log huấn luyện C″ đâu?** Mất khi runtime Colab hết hạn trước khi lưu thư mục `runs`. Còn `best.pt`, ảnh đánh giá trên Colab và đánh giá lại trên máy cá nhân khớp nhau (mAP50 0,512). Train lại khoảng 1–1,5 giờ nếu thầy cần.
11. **Sao không demo trên Raspberry Pi?** Demo trên Pi ở 416 nhận dạng chưa ổn (Hình 4.3: người đủ bốn món vẫn bị báo thiếu một số món). Nhóm đã đo độ trễ trên Pi (Step 4) nhưng chưa có bản demo trên Pi đủ tin cậy; hôm nay demo cùng pipeline trên laptop. Thừa nhận đây là hạn chế.
12. **Giấy phép?** Construction-PPE AGPL-3.0, PPEv5 CC BY 4.0, SH17 CC BY-NC-SA (học tập). Model được chọn (C′) không dùng SH17.

## Việc cần xong trước giờ nói

- [ ] Chạy thử demo trên **laptop với webcam** ít nhất hai lần ở đúng nền, ánh sáng, vị trí sẽ dùng; thử kính nào model thấy.
- [ ] Quay video dự phòng 30–60 giây demo thành công.
- [ ] Hạnh đọc lại v9 lần cuối (không còn ô nền vàng `[CẦN ĐIỀN]`).
- [ ] Mở sẵn: báo cáo v9 ở Hình 3.1–4.2, `Dataset_Show_Thay`, `Bang_Chung_Chay`, `RUNLOG.md`, terminal ở thư mục `prototype`/`src`.
- [ ] Tập đúng 10 phút một lần (nói + demo); nếu quá giờ, bỏ đoạn INT8 và rút gọn đoạn pruning.
