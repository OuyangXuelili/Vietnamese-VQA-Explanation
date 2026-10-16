# Dữ liệu ViVQA-X dùng trong dự án

Ba file annotation ViVQA-X gốc được chuẩn hóa bằng
`scripts/prepare_vivqax.py`; ảnh tương ứng lấy theo manifest từ COCO, không
tải toàn bộ COCO. Dữ liệu nguồn và ảnh nằm ngoài Git. Số liệu dưới đây là kết
quả kiểm tra local ngày 23/09/2026, không chép từ README của bộ dữ liệu.

| Phiên bản | Train | Validation | Test | Ghi chú |
|---|---:|---:|---:|---|
| Split gốc đã chuẩn hóa | 29.459 | 1.459 | 1.968 | 32.886 câu, 28.180 ảnh; validation/test chung 48 ảnh |
| `image_disjoint_v1` | 29.459 | 1.407 | 1.968 | Bỏ 52 câu validation thuộc các ảnh chung với test; train và test giữ nguyên |

`data/processed/vivqax/image_disjoint_v1/provenance.json` ghi hash đầu vào,
chính sách chia tập và các ID ảnh bị loại. Manifest và checksum ảnh được kiểm
tra trước khi train; kết quả cụ thể cùng lệnh tái tạo nằm trong
[giao thức dữ liệu](data_protocol.md). Không dùng test để chọn prompt,
checkpoint hay cách chia tập.

Tài liệu này chưa xác nhận điều kiện cấp phép và tái phân phối của annotation
ViVQA-X hoặc ảnh COCO. Không công bố lại dữ liệu dẫn xuất chỉ vì bản sao đang
có ở local/Drive; cần kiểm tra điều khoản từ nguồn trước.

Theo [repo tác giả ViVQA-X](https://github.com/duongtruongbinh/ViVQA-X), bộ dữ
liệu được xây từ VQA-X tiếng Anh qua dịch, lựa chọn chất lượng và hậu xử lý.
Vì vậy không mô tả annotation tiếng Việt là hoàn toàn do người Việt gán nhãn
trực tiếp trên ảnh. Cần kiểm tra thủ công lỗi dịch, câu hỏi mơ hồ và giải
thích không được ảnh hỗ trợ. Audit cú pháp/checksum không thay thế bước này;
chưa có báo cáo kiểm tra ngữ nghĩa thủ công được nghiệm thu trong repo.

Tách ảnh trong train/validation/test chỉ ngăn rò rỉ giữa các tập của thí
nghiệm này. Chưa xác định được ảnh COCO/VQA-X nào từng xuất hiện trong dữ
liệu pretraining của backbone; không tuyên bố bộ test hoàn toàn mới với Qwen.
