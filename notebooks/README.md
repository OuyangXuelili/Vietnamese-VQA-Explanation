# Notebook

| Tệp | Vai trò |
|---|---|
| [04_train_preferences.ipynb](04_train_preferences.ipynb) | Huấn luyện M2/M3: 16 ô, gồm 10 ô code |
| [05_evaluate_preferences.ipynb](05_evaluate_preferences.ipynb) | Xuất validation hoặc tính BERTScore: 12 ô, gồm 7 ô code |

Hai template dùng cho các lượt sau. Chọn launch record và seed ở đầu,
không tạo notebook mới theo từng seed hoặc mỗi lần mất phiên.
Mặc định chỉ inspect; gate, train và export phải được bật riêng.
Template 05 để trống record/checksum để người chạy chọn rõ lượt cần đánh giá.
Tác vụ semantic cũng yêu cầu chọn thư mục output riêng ở phần cấu hình.
Record mới phải ghim source đã hỗ trợ tác vụ; không sửa record/source lịch sử.

Code thuật toán ở `src/`, lệnh chạy ở `scripts/`, tham số ở `configs/`.
Notebook giữ thứ tự thực hiện và hiển thị output, không chứa trainer khác.

Trong 05, chọn `TASK` trước setup. `semantic` đọc prediction đã đủ và dùng
source phân tích/venv riêng; không khôi phục ảnh hay nạp lại Qwen.
`validation` giữ source suy luận trong record lịch sử. Bảng ba seed dùng
`scripts/report_validation_seeds.py`, không tạo thêm notebook cho mỗi seed.
Với record có `baseline_source` và `baseline_sha256`, CLI kiểm đủ prediction,
manifest và identity M0/M1 trước khi sao chép sang output của lượt mới.
Chỉ M2/M3 được sinh mới; baseline lệch hoặc thiếu thì dừng trước export.
Biên nhận nằm ở `baseline_reuse.json`; inspect chỉ kiểm, chưa sao chép.

Source ZIP chỉ gồm hai template trong bảng trên và từ chối đóng gói nếu
template còn output hoặc execution count. Notebook archive giữ nguyên;
test đọc archive thuộc nhóm `historical`, được chạy trong bộ kiểm local đầy đủ.

## Lịch sử và output

[M1 có output](https://colab.research.google.com/drive/13Abjl7kJyDKzbxyYsCetJ540OzdK2P_E)
và các [notebook lịch sử](archive/README.md) giữ nguyên.
Các bản thực thi 04 nằm trong `artifacts/preference_v2/` và Drive; không
thay chúng bằng template mới. Lượt 43 đã lưu bản hoàn tất 57 ô / 71 khối
output tại `artifacts/preference_v2/seed43_validation_20261017/04_validation_completed_20261017.ipynb`.

Bản 04 local trước tổ chức lại và bảng hash 75 notebook được giữ ở
`artifacts/notebook_cleanup_20261017/`. Bản 04 hoàn tất seed 44 được lưu tại
`artifacts/preference_v2/seed44_validation_20261018/04_validation_completed_20261019.ipynb`:
59 ô, 75 khối output; 58 ô trước khi thêm bước chạy cuối giữ nguyên.
Không dùng template mới để thay các bản thực thi này.

Notebook 05 hoàn tất BERTScore seed 44 ngày 20/10, 12 ô / 7 ô code / 5 khối
output, không có error. Bản có output ở
`artifacts/validation_analysis_20261020/05_semantic_completed_20261020.ipynb`
và notebook Drive; template local vẫn mặc định không thực thi export.

[Train/resume/validation](../docs/preference_training.md) ·
[Kết quả hiện có](../docs/project_description.md)
