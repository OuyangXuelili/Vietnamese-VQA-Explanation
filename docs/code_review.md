# Review code

Kiểm dữ liệu/split, completion masking, reference DPO, resume và cách chấm
kết quả. Đọc diff, chạy test liên quan và đối chiếu output trước khi đồng bộ.

## Luồng review

1. Đọc diff theo từng mục tiêu. Kiểm thay đổi có cần cho stage hiện tại,
   có trùng chức năng đã có và có làm sai hướng dẫn hoặc lineage của run cũ.
2. Chạy Ruff và bộ test local bằng `scripts/check_project.py`.
   Nếu đổi dữ liệu hoặc pipeline đầu vào, chạy thêm `--full-data` ở máy đủ ảnh.
3. Dùng test hành vi: dữ liệu sai bị từ chối, prompt/ảnh bị mask đúng,
   gradient cập nhật policy, reference giữ nguyên, resume khớp run liên tục,
   và nháp trình duyệt giữ đúng câu/nguồn khi nhập hoặc khôi phục.
4. Đọc cảnh báo scanner theo ngữ cảnh rồi ghi cách xử lý. Kiểm dependency và
   token trước khi xuất source. CI xác nhận kiểm tra trên môi trường sạch.
5. Review kết quả mô hình trên ảnh và mẫu đã định; ghi rõ bước nào chỉ kiểm
   code, bước nào đã chạy Qwen trên GPU và bước nào đã có nhãn người.

## Công cụ và vai trò

| Công cụ | Phần cần kiểm |
|---|---|
| Git diff và review nhánh/PR | Scope, lịch sử thay đổi, API, cấu trúc và giải thích quyết định |
| Ruff | Lỗi Python, import, các mẫu code dễ sai; nhóm S hỗ trợ rà bảo mật |
| pytest và CI | Hành vi, regression và khả năng chạy trên môi trường sạch |

Ruff và pytest/CI đã tích hợp trong repo. CI CPU dùng Python 3.10/3.11,
dependency verification/demo và Node, với `--fail-on-skip`.

Lượt 12/10 đã chạy Ruff nhóm S trên `src/` và `scripts/`; báo cáo tại
`artifacts/code_cleanup_20261012/ruff_security.json`. Cảnh báo RNG áp dụng
cho seed lấy mẫu/bootstrap, subprocess dùng danh sách argv không qua shell,
và URL COCO được tạo từ base HTTPS cố định. Kết quả scanner cần review các
luồng tương ứng; tránh sửa RNG hoặc giao thức thí nghiệm chỉ để hết cảnh báo.

Đối chiếu nhận xét với code và input thực tế trước khi sửa. Test chức năng
cần kiểm hành vi; tìm chuỗi trong source chỉ dùng để kiểm cấu trúc.

## Tên file

- Python và YAML dùng `snake_case`; tên mô tả nhiệm vụ hoặc thuật toán.
  Ví dụ: `audit_dataset.py`, `answer_metrics.py`, `sft_chosen.yaml`.
- Script là lệnh: `train_`, `evaluate_`, `prepare_`, `export_`, `package_`.
  `launch_preference_training.py` đọc record; `train_preference.py` chạy trainer.
- Notebook dùng `<stage>_<task>.ipynb`. M1 có output giữ trong `archive/`;
  `04_train_preferences.ipynb` dành cho M2/M3.
- Config đang dùng có tên ổn định. Thiết kế cũ đưa vào `configs/archive/`.
  Ngày/phiên bản dùng cho snapshot, run và protocol cần truy vết.
- Tên test theo phần được kiểm: `test_m1_notebook_history.py` kiểm lịch sử
  M1; `test_preference_notebook.py` kiểm notebook huấn luyện tiếp.

Đổi tên phải cập nhật CLI, import, config, tài liệu và test cùng lượt.
Run, release và notebook đã chạy giữ đường dẫn/nội dung gốc.
Git giữ source text dùng LF; notebook giữ nguyên byte để đối chiếu hash.

## Tài liệu công cụ

- [Ruff linter](https://docs.astral.sh/ruff/linter/)
- [pytest](https://docs.pytest.org/en/stable/)
