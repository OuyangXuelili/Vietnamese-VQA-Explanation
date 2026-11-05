# Notebook

Bảng dưới mô tả template trong source, không phải số ô của các bản đã chạy
trên Drive. Bản 04 cùng tên có output vẫn là hồ sơ thực thi; template cho
run mới lấy từ source release, không chép đè lên tệp đó.

| Tệp | Vai trò |
|---|---|
| [04_train_preferences.ipynb](04_train_preferences.ipynb) | Huấn luyện M2/M3: 16 ô, gồm 10 ô code |
| [05_evaluate_preferences.ipynb](05_evaluate_preferences.ipynb) | Xuất validation hoặc tính BERTScore: 12 ô, gồm 7 ô code |
| [06_demo_live.ipynb](06_demo_live.ipynb) | Kiểm kỹ thuật demo M1 trên GPU: 11 ô, gồm 7 ô code |
| [07_demo_web.ipynb](07_demo_web.ipynb) | Mở web hỏi ảnh mới trên T4: 7 ô, gồm 5 ô code |
| [08_evaluate_test.ipynb](08_evaluate_test.ipynb) | Xuất/resume tám identity test: 11 ô, gồm 7 ô code |
| [09_train_answer_preservation.ipynb](09_train_answer_preservation.ipynb) | Thí nghiệm M4: 13 ô, gồm 8 ô code |

04/05 dùng lại cho các lượt sau. Chọn launch record và seed ở đầu,
không tạo notebook mới theo từng seed hoặc mỗi lần mất phiên.
Mặc định chỉ inspect; gate, train và export phải được bật riêng.
Template 05 để trống record/checksum để người chạy chọn rõ lượt cần đánh giá.
Tác vụ semantic cũng yêu cầu chọn thư mục output riêng ở phần cấu hình.
Record mới phải ghim source đã hỗ trợ tác vụ; không sửa record/source lịch sử.

Trong Drive, tệp `05_evaluate_lr5e6.ipynb` có output ngày 22/10 là hồ sơ
đã thực thi, không phải template dùng lại: hai ô recovery cuối chỉ dành cho
lần khôi phục đó. Không Run all hoặc xóa output để làm sạch. Lần đánh giá
tiếp dùng `05_evaluate_preferences.ipynb` trong source đã kiểm, chọn rõ
`TASK`, record và thư mục output; không cần chạy lại notebook 04.

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

06 tách phần kiểm demo khỏi train và validation. Chọn source/checksum ở đầu;
mặc định chưa chạy GPU. Không chạy 06 song song với export trong 05.
Ba ảnh validation được cố định ngoài 12 ví dụ replay, không chọn theo chất lượng.
Luồng này không thay việc kiểm giao diện và thử ảnh ngoài dataset.

Source ZIP chỉ gồm sáu template trong bảng trên và từ chối đóng gói nếu
template còn output hoặc execution count. Notebook archive giữ nguyên;
test đọc archive thuộc nhóm `historical`, được chạy trong bộ kiểm local đầy đủ.

09 dùng launch riêng `answer_preservation_seed42.json` và source mới đã
kiểm. Nhập checksum từ receipt; mặc định chỉ inspect, chưa chạy gate/train.
M4 được lưu ở run riêng. Các bản 04/05/08 đã có output là lịch sử của
nghiên cứu chính, không chép template mới lên chúng.

08 dùng một launch record đã ghim source, gói đầu vào test và kế hoạch. Mỗi
identity có thư mục output riêng trên Drive. Chạy lại cùng record để nối
phần còn thiếu; không sửa 04/05 hoặc tạo notebook cho từng seed. Mặc định
chỉ kiểm đầu vào. Hướng dẫn chấm phải khóa trước suy luận; nếu điều chỉnh
thứ tự luyện chấm người thì lưu amendment riêng, giữ nhãn người chưa hoàn tất.

Khi mount Drive khác, đặt `RESULTS_DIR` trỏ đúng kho kết quả gốc và đọc ô
tiến độ trước khi export. Không lấy thư mục cùng tên làm bằng chứng đã nối đúng.
`RUN_IDENTITIES` chọn phần còn thiếu; `RUN_DIR_NAMES` giữ đường dẫn của lượt
phục hồi và được dùng cả khi kiểm tiến độ. Mặc định vẫn chạy đủ tám identity.

Nếu đường dẫn checkpoint đổi, `scripts/relocate_test_run.py` kiểm giao thức,
hash và toàn bộ prefix rồi chép sang một thư mục mới. Ghim hash công cụ ở
`RELOCATION_SHA256`, chọn identity trong `RELOCATE_IDENTITIES`. Lượt gốc giữ
nguyên; biên nhận `storage_relocation_verified.json` cho phép kiểm lại prefix
trước mỗi lần nối tiếp. Không dùng cách này để đổi trọng số hoặc cấu hình.

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
