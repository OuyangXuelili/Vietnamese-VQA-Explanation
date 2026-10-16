# Harness kiểm tra và chạy thí nghiệm

Harness dùng chung cho local và Colab. `scripts/check_project.py` kiểm lint/test;
`scripts/run_pipeline.py` lưu cấu hình, log và kết quả cho mỗi run.
M1 đã hoàn tất một epoch và đánh giá đủ validation. Các job tiếp theo xem
TIEN_DO; không chạy lại pilot mỗi khi nối runtime.

## Lệnh thường dùng

Trong thư mục repo, sau `pip install -e ".[dev]"`:

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
.\.venv\Scripts\python.exe scripts/check_project.py --with-data
.\.venv\Scripts\python.exe scripts/check_project.py --full-data
```

Lệnh đầu chạy Ruff và pytest, không cần ảnh hay GPU. `--with-data` thêm audit/preflight bộ 500 ảnh; `--full-data` thêm audit nghiêm ngặt và preflight toàn bộ 29.459 câu train của `image_disjoint_v1`. Hai tùy chọn dữ liệu chỉ chạy được khi các file tương ứng đã có trên máy. Dùng `--dry-run` để xem các lệnh mà không tạo run.

Chạy riêng từng stage và đổi môi trường khi ở Colab:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py audit --environment configs/environments/local_full.yaml
.\.venv\Scripts\python.exe scripts/run_pipeline.py preflight --environment configs/environments/local_full.yaml --all-train-records
.\.venv\Scripts\python.exe scripts/run_pipeline.py token-profile --environment configs/environments/local_full.yaml --dry-run
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-smoke --dry-run
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-full-preflight --environment configs/environments/local_full.yaml
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-optimizer-pilot --environment configs/environments/local_full.yaml --dry-run
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-full --environment configs/environments/local_full.yaml --dry-run
```

File `configs/environments/colab_full.example.yaml` là mẫu để đặt đường dẫn dữ liệu đã giải nén và thư mục run trên Drive. Không thay đường dẫn trong code. `token-profile` dùng `AutoProcessor` và collator SFT trên 512 câu train được chọn bằng seed cố định; cần Torch/Transformers nhưng **không tải trọng số model hay train GPU**. Trên Colab chạy qua `notebooks/archive/06_sft_token_profile.ipynb`. `sft-smoke` thật cần GPU CUDA, chỉ huấn luyện 8 bước theo `sft_smoke.yaml`; không dùng nó để huấn luyện M1 đầy đủ.

`sft-batch-pilot` dùng cùng thiết lập QLoRA/LoRA và collator với SFT smoke, nhưng chọn đúng một ID train theo `sft_batch_pilot.yaml`; đo forward/backward, token, loss, gradient, thời gian và VRAM. Không gọi optimizer, không lưu adapter/checkpoint. Chạy bằng `notebooks/archive/07_sft_batch_pilot.ipynb` trên T4, sau khi source/bundle đã xác minh. Kết quả `batch_pilot.json` chỉ chứng minh batch này khả thi, không là bằng chứng hiệu năng cho mọi mẫu hoặc kết quả M1.

`sft-full-preflight` xác nhận đúng 29.459/1.407/1.968 câu, không trùng ảnh, và tính lại SHA-256 của từng ảnh train cùng 256 validation cố định theo manifest; không tải mô hình. `sft-optimizer-pilot` dùng cùng trainer nhưng chỉ 20 optimizer step, đánh giá/lưu tại bước 10 và 20 để đo throughput và kiểm tra checkpoint. `sft-full` chạy một epoch trên toàn train và chọn checkpoint bằng validation loss; xem [giao thức M1](sft_full.md). Pilot không phải kết quả cuối.

## Bằng chứng của mỗi run

Mỗi stage tạo thư mục mới `runs/<stage>_<UTC>/` (hoặc `run_root` trong profile), không ghi đè thư mục cũ:

- `execution.json`: stage, lệnh, đường dẫn, thời điểm bắt đầu/kết thúc, trạng thái/return code, Git commit + dirty flag, hash SHA-256 annotation, config và các file source/config liên quan.
- `console.log`: stdout/stderr của tiến trình con.
- `audit.json`, `token_profile.json` hoặc `training/`: đầu ra riêng của stage.

Chỉ coi run là đạt khi `execution.json` có `status: passed`, `returncode: 0` và đầu ra stage hợp lệ. Một phiên train dừng chủ động có `status: paused` và `training/session_result.json`; đó không phải M1 đã hoàn tất. Hash source giúp nhận ra code khác nhau dù cùng Git commit nhưng working tree chưa commit. Thư mục Drive của run không phải bản sao của repo; source release là ZIP riêng có manifest/checksum.
Với profile full, `image_checksums` trỏ tới CSV hash từng ảnh; M1 preflight/pilot/full kiểm tra lại byte của các ảnh được dùng trước mỗi phiên train/resume. Audit thông thường chỉ kiểm tra ảnh mở được. Kiểm tra hash ảnh tốn thêm thời gian và I/O, nhưng chặn dữ liệu đổi âm thầm khi tiếp tục từ checkpoint.

## Giới hạn hiện tại

Token profile mặc định đo trên mẫu, không đại diện chắc chắn cho mọi câu train. P0 của M1 đã có report toàn 29.459 train và 256 validation; dùng lại khi hash khớp. Pilot/rehearsal kiểm tính khả thi và trạng thái resume, không chứng minh chất lượng. M1 chỉ hoàn tất sau kiểm checkpoint cuối một epoch; chất lượng còn cần prediction validation đầy đủ và chấm ảnh theo kế hoạch.
