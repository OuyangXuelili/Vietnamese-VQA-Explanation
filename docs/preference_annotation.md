# Cặp giải thích ưu tiên

## Bộ đang dùng — 23/10/2026

Bộ chính đã khóa tại
`data/preferences/vivqax_m1_3683_v2/frozen_1000_reviewed_20261015/`:
1.000 cặp trên 1.000 ảnh train, còn 117 cặp dự phòng. Cả M2 và M3 dùng
cùng ảnh, câu hỏi, đáp án và chosen; M2 học chosen, M3 học chosen/rejected.

Có 992 rejected tổng hợp có kiểm soát và 8 output M1 nguyên văn; toàn bộ
chosen là văn bản tổng hợp. Metadata giữ nguồn tạo và các quyết định kiểm
ảnh. Đây chưa phải bộ nhãn được người đánh giá độc lập thẩm định.
Không đổi nguồn hoặc gán tên người để thay cho bước thẩm định này.

Bộ v1 gồm 186 cặp giữ làm lịch sử kiểm triển khai, không dùng cho các run
chính. [Hướng dẫn và diễn biến v1](archive/preference_annotation_v1_20261014.md)
được lưu nguyên nội dung. Không cần duyệt hoặc sinh lại các bộ đã khóa.

## Tiêu chí giữ cặp

| Nội dung | Điều kiện |
|---|---|
| Đáp án | Đúng theo ảnh; không chỉ dựa vào exact match với tham chiếu |
| Chosen | Giải thích có bằng chứng liên quan, đủ hỗ trợ đáp án |
| Rejected | Có lỗi cụ thể về đối tượng, thuộc tính, quan hệ/hành động hoặc hỗ trợ đáp án |
| Đối lập | Cùng ảnh, câu hỏi và đáp án; chỉ khác giải thích |
| Bằng chứng | Ghi chi tiết nhìn thấy và đúng cụm có vấn đề trong rejected |
| Nguồn | Văn bản M1 giữ nguyên là model; văn bản tổng hợp có nguồn và hash recipe |

Không dùng cặp chỉ khác cách diễn đạt/độ dài, cả hai giải thích đều đúng,
hoặc cần đoán ý định/trạng thái không thấy. Câu chưa chắc giữ uncertain.
Không sửa annotation gốc giữa run và không loại câu khó khỏi test.

Kiểm văn bản chỉ giúp phát hiện đường tắt. Việc phân biệt chosen/rejected
dễ bằng chữ không chứng minh mô hình dùng ảnh; đối chứng đổi ảnh và chấm
mù cuối là các phép đánh giá riêng.

## File cần giữ

| Tệp | Vai trò |
|---|---|
| `accepted.jsonl` | Cặp dùng cho M3 |
| `chosen_only.jsonl` | Cùng các ID/chosen dùng cho M2 |
| `manifest.json` | Đầu vào, checksum và giao thức |
| `quality_review.jsonl` | Bằng chứng kiểm ảnh từng cặp |
| `decision_record.json`, `readiness.json` | Quyết định khóa và kiểm đầu vào |
| `text_only.json` | Kiểm đường tắt chỉ đọc chữ |

Giữ bộ đã khóa nguyên byte. Sửa dữ liệu cần phiên bản mới, không cập nhật
nhãn hoặc hash của bộ đã được dùng để train.

## Nếu xây dựng một bộ mới

1. Chọn trước phạm vi train-only và khử trùng theo ảnh.
2. Xem ảnh/câu hỏi, kiểm đáp án và hai giải thích theo tiêu chí trên.
3. Lưu quyết định, bằng chứng, nguồn và thời gian; không ghi nguồn tổng hợp
   thành nhãn người. Giao diện có thể ẩn thông tin kỹ thuật, dữ liệu vẫn giữ.
4. Xuất JSONL, kiểm annotation/ảnh/hash và đối chiếu ID/chosen của hai nhánh.
5. Khóa bộ và khai study/launch record trước train.

Các lệnh `expand_preferences.py`, `finalize_preference_review.py` và
`validate_preferences.py` nằm trong `scripts/`. Chạy `--help` để chọn đúng
đầu vào; output phải là thư mục mới. Audit schema không thay kiểm đúng/sai
theo ảnh và không tự cho phép mở GPU train.

Trang duyệt cặp là công cụ dữ liệu, không phải demo hỏi đáp ảnh.
Bước hiện tại là hoàn tất đối chứng LR và đánh giá cuối, không mở lại pilot.
Xem [huấn luyện/đánh giá](preference_training.md) và
[chấm mù test](human_evaluation.md).
