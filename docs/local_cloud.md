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
`notebooks/04_train_preferences.ipynb` điều phối train M2/M3;
`05_evaluate_preferences.ipynb` chỉ đánh giá validation. Hai template dùng
launch record theo seed, không train lại M1. Notebook có output trong
`archive/`, `artifacts/colab_history/`, `artifacts/preference_v2/` và Drive
giữ nguyên. Không cập nhật template đè lên notebook đang chạy.

## Local

Python 3.10+; cài Torch/Torchvision CPU, `.[dev,verification,demo,report]` và Node 22
theo [hướng dẫn local](../README.md#local). `.[dev]` chỉ đủ cho audit và
một phần test, không đủ để báo cổng kiểm đầy đủ đã đạt.

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
.\.venv\Scripts\python.exe scripts/run_pipeline.py preflight --environment configs/environments/local_full.yaml --all-train-records
```

Lệnh trên kiểm cả artifact lịch sử trên máy này. Checkout mới không có
run ngoài Git dùng `scripts/check_project.py --source-only`; chỉ loại nhóm
`historical`, không cho skip kiểm source vì thiếu thư viện. Cả hai chế độ
đều dùng `--fail-on-skip`.

Máy hiện tại đã có 28.180 ảnh ViVQA-X. Quy trình lấy dữ liệu trên máy mới ở
[data_protocol.md](data_protocol.md). Môi trường CPU phục vụ các kiểm tra
nhỏ; runtime GPU cần nghiệm thu riêng với Qwen/QLoRA.

## Source và môi trường GPU

Kho Drive giữ dữ liệu và lịch sử chạy, không phải một checkout Git được
đồng bộ tự động. Mã dùng để chạy lấy từ `source/releases/<release>.zip`,
đối chiếu SHA-256 với record của stage. Các script rời, `project_source.zip`
và ghi chú cũ trong `source/` là hồ sơ lịch sử; không lấy tên hoặc ngày sửa
của chúng để suy ra source mới nhất.

Notebook trong Drive có thể chứa output và cấu hình của một phiên đã chạy;
template trong source để chuẩn bị phiên mới. Không chép template đè notebook
có output, không đổi release của run cũ khi đồng bộ source mới. Báo cáo
phân tích tạo trên local nằm ngoài ZIP source và phải lưu riêng nếu cần
đọc lại trên Drive. Xem `artifacts/README.md` để tìm bản hiện hành.

1. Đóng source bằng `scripts/package_source.py`, lưu ZIP và SHA-256.
2. Upload vào Drive `source/releases/`; tải lại ZIP để đối chiếu hash.
3. Giải nén vào thư mục mới trên runtime, dùng profile cloud trỏ dữ liệu thật.
4. Cài đúng môi trường của stage và kiểm phiên bản/hash trước khi chạy.

M1 dùng source/môi trường đã ghim trong run M1. M2/M3 dùng
`requirements-cloud-preference.txt` trong runtime GPU mới. Phiên Colab
`/content` là ổ tạm; lưu checkpoint và log vào kho bền vững đã kiểm.
Luồng M2/M3 ở [preference_training.md](preference_training.md).

Notebook 04 chọn launch record/hash và seed ở đầu; không đọc launch v1
của lượt 186 cặp. Mỗi record ghi release/hash, đường dẫn bộ cặp và nhóm run.
`scripts/bootstrap_cloud_source.py` khôi phục release trong thư mục mới;
notebook kiểm hash công cụ trước khi gọi. Setup không tải model.
Mặc định tắt `RUN_GATE`, `RUN_TRAINING` và `RUN_EXPORT`; inspect phải thành
công trước khi chạy. `data_locked=false` chặn bộ train chưa khóa.

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
