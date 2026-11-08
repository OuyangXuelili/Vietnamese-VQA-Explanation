# Rà lỗi ngôn ngữ trong đầu ra đã lưu

Phân tích bổ sung trên tám lượt test, mỗi lượt 1.968 câu. Đếm phản hồi
có ít nhất một ký tự trong vùng Unicode U+3040–U+30FF hoặc U+3400–U+9FFF
ở đáp án hoặc lời giải thích.

| Nhánh | Phản hồi có ký tự Hán/Nhật | Tỷ lệ |
|---|---:|---:|
| M0 | 61 | 3,10% |
| M1 | 6 | 0,30% |
| M2 seed 42 | 21 | 1,07% |
| M2 seed 43 | 20 | 1,02% |
| M2 seed 44 | 20 | 1,02% |
| M3 seed 42 | 64 | 3,25% |
| M3 seed 43 | 66 | 3,35% |
| M3 seed 44 | 67 | 3,40% |

Các tham chiếu test có 0 ca theo phép tìm này. Tất cả các ca bị đánh dấu
vẫn có `format_valid=true`, tức chương trình tách được đáp án và giải
thích. Validation seed 42 có 45/4/11/35 ca ở M0/M1/M2/M3 trên 1.407 câu.

Phép tìm ký tự giúp chọn phản hồi cần đọc lại. Cần đối chiếu ảnh, câu hỏi
và văn bản để phân biệt lỗi lẫn ngôn ngữ với tên riêng hoặc chữ trên ảnh.
Phép đo chưa bao quát lỗi tiếng Việt viết bằng chữ Latin.

Cùng phép tìm trên validation ngày 7/11 cho kết quả 4 ca ở M1,
35 ở M3 và 32 ở M4, mỗi nhánh 1.407 câu. ID từng ca nằm trong
`archive_verification.json` của thư mục `validation_20261107`.
Cần đọc lại theo ảnh để phân loại những ca này.

Kiểm thêm bộ chosen 1.000 câu và bộ train 29.459 câu ngày 7/11 cho
0 hàng chứa ký tự trong cùng các vùng Unicode trên. Kết quả này giới
hạn ở phép rà ký tự. Hash và số hàng của hai tệp được lưu trong
`artifacts/output_review/m4_validation_30_20261107/private/training_language_check.json`.

Hồ sơ local trong `artifacts/review_fixes_20261105/` gồm
`prediction_audit.json`, `mixed_script_counts.csv` và `mixed_script_cases.json`.
