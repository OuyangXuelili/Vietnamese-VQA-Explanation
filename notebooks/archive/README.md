# Notebook lịch sử

Giữ nguyên nội dung/output các kiểm thử trước M1. Không cần chạy lại chúng.

| Notebook | Vai trò |
| --- | --- |
| 00 | Chuẩn bị Colab và dữ liệu smoke |
| 01 | Suy luận Qwen gốc trên 20 câu |
| 02 | SFT 8 bước và nạp lại adapter |
| 04 | So sánh M0/M1 trên 20 câu |
| 05 | Kiểm tra dữ liệu đầy đủ |
| 06 | Đo token trên 512 mẫu |
| 07 | Một lượt forward/backward, không cập nhật optimizer |
| 08 | Kiểm tra optimizer và dừng 10 → resume 20 |
| 03_m1_and_preference_before_reorganization_20261013 | Bản local nguyên vẹn trước khi rút gọn notebook 03; chứa cả phần thử chưa chạy |
| 03_m1_completed_with_outputs_20261013 | Bản tải từ Colab, 20 cell và 24 output; giữ nguyên byte |
| 04_preferences_completed_with_outputs_20261014 | Bản tải từ cùng notebook 04 sau train M2/M3, 7 cell; hai nhánh đủ 24 update / một epoch |

Không dùng kết quả smoke thay kết quả full M1. Notebook 05–07 có thể chọn
release mới nhất khi chạy lại; 08 ghim release rehearsal cũ. Không Run all
để xem output lịch sử. Đường dẫn cloud trong notebook được giữ nguyên.

Huấn luyện tiếp: [04_train_preferences.ipynb](../04_train_preferences.ipynb).
Kết quả hiện có: [mô tả dự án](../../docs/project_description.md).

Bản `with_outputs` chỉ lưu bằng chứng local/Drive, không đưa vào Git hoặc
source ZIP. Source/config của từng run M1 vẫn dùng release đã ghim trong
run đó; không dùng source mới để chạy lại các cell cũ.
