# Chạy dự án trên local và cloud

Local chuẩn bị dữ liệu, duyệt cặp và chấm output. Colab hoặc Kaggle cung cấp
GPU cho train và suy luận. Hai môi trường dùng cùng package trong `src/` và
CLI trong `scripts/`; notebook điều phối lệnh và hiển thị kết quả.

## Đường dẫn và tệp

`configs/environments/*.yaml` khai báo annotation, ảnh, thư mục run và
checksum ảnh. Đường dẫn tương đối được tính từ gốc repo. Các profile
`local.yaml` và `local_full.yaml` lần lượt dùng bộ smoke và bộ đầy đủ.
Mẫu profile cloud có sẵn để điền vị trí dữ liệu trên máy đang chạy.

| Nội dung | Nơi lưu |
|---|---|
| Logic, cấu hình, test, hướng dẫn | Git và source release |
| Ảnh/annotation đã chuẩn bị | `data/` local và dataset bundle trên Drive |
| Checkpoint, prediction, log | `runs/` hoặc kho bền vững trong profile |
| Source có checksum | `artifacts/releases/` và Drive `source/releases/` |
| Hồ sơ tải về | `artifacts/colab_history/<ngày>/` |

[Mục lục artifacts](../artifacts/README.md) chỉ rõ bản review hiện hành.
`notebooks/04_train_preferences.ipynb` là notebook đang dùng; các notebook
trong `archive/` giữ lịch sử M1. Notebook 04 chỉ gọi luồng M2/M3; không
chứa code train M1, đánh giá cũ hoặc sinh lại ứng viên.

## Local

Python 3.10+ và `.[dev]` đủ cho audit và các test nhẹ. Kiểm trainer nhỏ dùng
Torch/Torchvision CPU rồi `.[dev,verification]`; kiểm API/demo thêm `[demo]`.

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
.\.venv\Scripts\python.exe scripts/run_pipeline.py preflight --environment configs/environments/local_full.yaml --all-train-records
```

Máy hiện tại đã có 28.180 ảnh ViVQA-X. Quy trình lấy dữ liệu trên máy mới ở
[data_protocol.md](data_protocol.md). Môi trường CPU phục vụ các kiểm tra
nhỏ; runtime GPU cần nghiệm thu riêng với Qwen/QLoRA.

## Source và môi trường GPU

1. Đóng source bằng `scripts/package_source.py`, lưu ZIP và SHA-256.
2. Upload vào Drive `source/releases/`; tải lại ZIP để đối chiếu hash.
3. Giải nén vào thư mục mới trên runtime, dùng profile cloud trỏ dữ liệu thật.
4. Cài đúng môi trường của stage và kiểm phiên bản/hash trước khi chạy.

M1 dùng source/môi trường đã ghim trong run M1. M2/M3 dùng
`requirements-cloud-preference.txt` trong runtime GPU mới. Phiên Colab
`/content` là ổ tạm; lưu checkpoint và log vào kho bền vững đã kiểm.
Luồng M2/M3 ở [preference_training.md](preference_training.md).

Notebook 04 đọc `source/preference_training_launch_v1.json` trên kho dự án.
Record ghim release/hash, đường dẫn bộ cặp cuối và một nhóm run ổn định.
`scripts/bootstrap_cloud_source.py` khôi phục release trong thư mục mới;
notebook kiểm hash công cụ trước khi gọi. Setup không tải model.
Mặc định `MODE='inspect'`; `data_locked=false` chặn train ngay từ đầu.

Kaggle có profile mẫu `kaggle.example.yaml`. Trước một run mới, kiểm GPU,
bundle, dung lượng output và khả năng tải checkpoint ra kho bền vững.
Các run M1 lịch sử tiếp tục theo lineage đã ghi trong nhật ký.

## Khi thay đổi source

Chạy `scripts/check_project.py` trước release. Nếu đổi dữ liệu hoặc pipeline
đầu vào, kiểm thêm `--full-data` trên máy đủ ảnh. Mỗi run/release có output
mới và hash của source, config, dữ liệu đã dùng.

M2/M3 dùng hash dependency của trainer để kiểm gate/resume; CSS demo và
asset trang duyệt được ghi trong fingerprint release riêng. Đổi loss,
collator, validator hoặc môi trường train sẽ làm gate tương ứng thay đổi.
Các gate đã tạo bằng schema source cũ được giữ cùng run cũ.

Lịch sử review và thiết lập trước lần rút gọn được lưu ở
[archive/local_cloud_before_cleanup_20261012.md](archive/local_cloud_before_cleanup_20261012.md).
