# Phân tích kết quả validation

Kết quả hiện có ở `reports/generated/m0_m1_validation_20261011/`:

- `README.md`: bảng điểm, CI và giới hạn diễn giải.
- `metrics.png`, `answer_transitions.png`: hình đưa vào báo cáo.
- `cases.html`, `images/`: 24 ví dụ có ảnh, sáu câu mỗi nhóm chuyển trạng thái EM.
- `report.json`, `manifest.json`: số liệu chính xác, nguồn và checksum.

Đã kiểm đủ 1.407 câu trên 1.383 ảnh, cùng giao thức suy luận và tham chiếu;
đối chiếu hash toàn bộ ảnh dùng trong prediction với ảnh local.
Các điểm BERTScore lấy từ báo cáo đã lưu, có đối chiếu hash prediction và
giao thức, không nạp lại model ngôn ngữ để tính lần nữa.

M1 có 463 câu từ không khớp sang khớp tham chiếu, 42 câu chuyển ngược lại.
Đây là exact match, chưa phải nhãn đúng/sai do nhìn ảnh. Ví dụ được chọn bằng
SHA-256 trong từng nhóm; mỗi nhóm giữ ảnh khác nhau. Không dùng các ví dụ này
để suy ra tỷ lệ lỗi ảnh trên toàn tập. Bốn nhóm đều giữ lại để phân tích.

M1 đã được chọn bằng validation. Báo cáo này mô tả validation của run đó;
test cuối và nhãn người vẫn cần làm để trả lời câu hỏi nghiên cứu.

## Tái tạo

Cài thêm `pip install -e ".[report]"` trên môi trường phân tích local.
Lệnh dưới dùng bộ prediction đã có trong máy hiện tại. Trên checkout mới,
lấy đúng artifact từ kho dự án và kiểm hash ở `report.json`.
Luôn chọn output mới khi chạy lại, vì công cụ từ chối ghi đè report đã có.

```powershell
python scripts/report_validation.py `
  --m0 artifacts/validation_review_20261003/checkpoint_800/m0.jsonl `
  --m1 artifacts/colab_history/20261007/m1_full_validation/m1.jsonl `
  --annotations data/processed/vivqax/image_disjoint_v1/validation.jsonl `
  --annotations-sha256 7f0b2b7d374d386030ddea50bc57d3e8e6c76e257e9fc4d1203bfb331b5f5b6e `
  --environment configs/environments/local_full.yaml `
  --m0-semantic artifacts/semantic_evaluation_20261010/m0_validation_metrics_bertscore.json `
  --m1-semantic artifacts/semantic_evaluation_20261010/m1_validation_metrics_bertscore.json `
  --output-dir reports/generated/m0_m1_validation_reproduced
```

Nếu chưa có BERTScore, bỏ cả hai cờ semantic; report chỉ có ba chỉ số còn lại.
Các chỉ số giữ nguyên cách chuẩn hóa của repo. Không thay từ đồng nghĩa vào
prediction sau khi thấy kết quả. Mọi sửa metric phải thành phiên bản mới và
áp dụng cho mọi nhánh cùng lúc.

Ảnh trong report là thumbnail để đọc. Hash kiểm chứng luôn dùng ảnh gốc,
không dùng bản thumbnail nén lại. Thư mục report sinh tự động không đưa vào Git.
