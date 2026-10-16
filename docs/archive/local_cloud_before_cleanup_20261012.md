# Local và cloud dùng chung một codebase

Thứ tự thực thi hiện hành nằm trong [kế hoạch](research_plan.md).
Không chạy lại các notebook lịch sử bên dưới để chuẩn bị M1.

Phần thiết lập M2/M3 mới ở [hướng dẫn preference training](preference_training.md).
Môi trường local chỉ bổ sung Accelerate/PEFT để kiểm trainer nhỏ trên CPU;
không cài CUDA, bitsandbytes hoặc tải backbone về máy này.

> Phần "Kết quả review 22/09/2026" bên dưới là mốc lịch sử, không phải trạng thái hiện tại của trainer hoặc dữ liệu. Xem [TIEN_DO.md](../TIEN_DO.md) để biết kết quả kiểm thử và các việc còn mở mới nhất.

## Kết quả review 22/09/2026

1. **Đã sửa:** cấu hình dữ liệu mặc định chứa Drive và `/content`. Local nay dùng
   `data/raw`, `data/processed`, `runs`. `configs/environments/local.yaml` là profile
   chạy được; đường dẫn tương đối được neo theo repository, không theo terminal.
2. **Đã sửa:** audit ảnh trước đây chỉ kiểm tra file tồn tại. Nay kiểm tra PIL.verify,
   chặn path thoát image root, và `--strict` trả exit code lỗi khi dữ liệu không hợp lệ,
   ảnh thiếu/hỏng, ID trùng hoặc ảnh trùng giữa các split.
3. **Đã bổ sung:** `scripts/run_pipeline.py` chạy audit/preflight/SFT smoke bằng cùng
   CLI trên local/cloud; mỗi run có execution.json và console.log được stream ra màn hình.
4. **Chưa thay trainer:** configs/training/sft.yaml và dpo.yaml là bản thiết kế,
   không phải cấu hình được trainer smoke hỗ trợ. Đã ghi rõ và bỏ đường dẫn Drive.
   Chỉ sft_smoke.yaml được kiểm thử GPU. Không tự tăng limit rồi gọi là train đầy đủ.
5. **Legacy:** notebook 02 nhúng source overlay là cách bootstrap tạm trước đây.
   Giữ nguyên để tái lập run cũ; không tiếp tục phát triển bằng sửa source nhúng.
   Luồng mới dùng snapshot source có manifest SHA256 rồi gọi CLI, xem notebook 03.
6. **Chưa giải quyết trong lượt này:** đánh giá chất lượng giải thích; so M0/M1 cùng
   cấu hình; full trainer/resume; DPO. Local unit tests không thay thế kiểm tra GPU.
7. **Điểm dữ liệu phải chốt trước full experiment:** stats.json có 48 ảnh chung
   giữa validation và test ở bộ đầy đủ. Train không chung ảnh với hai tập này.
   Smoke đã loại các ảnh chung nên audit smoke sạch không chứng minh full split sạch.
   Không tự sửa split chính thức: cần ghi rõ protocol original và image-disjoint,
   giữ manifest phiên bản riêng trước khi so sánh. Không xem điểm test để chọn cách chia.

## Bố trí

```text
src/vqa_explanation/        logic dữ liệu, huấn luyện, suy luận, đánh giá dùng chung
scripts/                   CLI và runner (không cần Google Drive trên local)
configs/environments/      nơi lưu dữ liệu/kết quả của từng máy
configs/model,training/    tham số thuật toán; độc lập nơi lưu trữ
data/raw/                  dữ liệu nguồn và ảnh local, không commit
data/processed/            annotation/manifest đã xử lý, không commit
runs/<stage_timestamp>/    log, metadata, metrics/checkpoint, không commit
artifacts/releases/        source snapshot có checksum để chuyển lên cloud
notebooks/                 điều phối và xem kết quả, không là bản source thứ hai
```

Downloads chỉ là nơi nhận file, không phải nơi notebook đọc dữ liệu hoặc checkpoint.
Sau khi đối chiếu checksum, lưu output vào `artifacts/colab_history/<ngày>/`;
source ZIP vào `artifacts/releases/`. Notebook đang dùng là
`notebooks/03_portable_runner.ipynb`, các notebook cũ ở `notebooks/archive/`.
Các thư mục review có ngày giữ ảnh và ghi chú của từng mốc; không xóa chỉ vì ảnh
giống nhau. Khi dọn file local, ghi nguồn, đích và checksum để có thể khôi phục.

## Chạy trên Windows local

Tại repository, cài môi trường nhỏ: `python -m pip install -e ".[dev]"`.
Không cần torch/GPU để audit/preflight. Local hiện đã có đủ 28.180 ảnh ViVQA-X
tham chiếu (không phải toàn bộ COCO); bộ smoke 500 ảnh/588 câu vẫn giữ riêng.
Không tải lại dữ liệu hoặc model về máy yếu chỉ để chạy kiểm tra.

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py audit
.\.venv\Scripts\python.exe scripts/run_pipeline.py preflight
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-smoke --dry-run
```

Kiểm tra mọi đường dẫn ảnh train của bộ đầy đủ (không train):

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py preflight --environment configs/environments/local_full.yaml --all-train-records
```

Nếu cần tái tạo bộ smoke, đặt annotation gốc ở data/raw/vivqax/annotations rồi
chạy `scripts/prepare_vivqax.py`, `scripts/download_coco_smoke.py` từ repository.
Downloader smoke chỉ tải ảnh thuộc manifest smoke và bỏ qua JPEG hợp lệ đã có.
Với dữ liệu đầy đủ, xem `docs/data_protocol.md` và dùng
`configs/environments/local_full.yaml`; không gọi trainer smoke là full train.
Nếu dữ liệu nằm ổ khác, tạo profile YAML ngoài Git và truyền `--environment <path>`.
Không thay đường dẫn trong source. Local có GPU phù hợp có thể dùng cùng runner;
trainer hiện cần CUDA/4-bit, không hứa chạy CPU hay đã kiểm thử GPU Windows.

## Chạy cloud

Drive là storage bền vững, `/content` là ổ làm việc tạm của Colab. Copy dataset
đúng phiên bản vào ổ tạm để train; lưu run/checkpoint trên Drive. Dữ liệu ảnh
đầy đủ đã được tải và kiểm tra ở local ngày 23/09; bundle để chuyển lên Colab
ở `artifacts/datasets/vivqax_image_disjoint_v1/`. Dùng source snapshot mới,
không ghi đè project_source.zip cũ.

1. Chạy pytest và Ruff trên local; cập nhật TIEN_DO.md.
2. `python scripts/package_source.py` tạo archive + .sha256, không gồm dataset/model/.env.
3. Upload hai file cùng phiên bản vào Drive/source/releases; kiểm tra hash sau upload.
4. Giải nén source ở thư mục runtime mới, cài package; chọn profile cloud và chạy CLI.
5. Lưu config thực tế, metadata và console.log. Mọi sửa lỗi cloud phải đưa về local,
   test lại và tạo snapshot mới. Không tuyên bố đồng bộ nếu chưa kiểm tra bản cloud.

## Chuyển sang Kaggle (đang chuẩn bị, chưa chạy GPU)

Không cần thêm notebook: dùng cùng CLI trong terminal/cell Kaggle. Cần tài khoản
được bật GPU, quota còn đủ và Internet để lấy model. Không mặc định tài khoản mới
đã có GPU; xác nhận trong Settings trước khi tải bundle lên. Source, dữ liệu và
checkpoint của dự án phải để private, không tạo public dataset để lấy URL tải.

1. Dùng bundle hiện có ở `artifacts/datasets/vivqax_image_disjoint_v1/`, gồm
   hai ZIP ảnh, ZIP annotation, `bundle.json` và `bundle.sha256`; không tải lại COCO.
   Đưa source release và bundle vào private inputs sau khi được phép upload.
2. Kiểm source SHA-256 trước giải nén. Cài package từ source đã kiểm. Dùng
   `scripts/verify_coco_bundle.py --help` để kiểm/extract bundle trên ổ làm việc;
   không tự đổi split hay bỏ checksum để tiết kiệm thời gian.
3. Đặt biến `VQA_ANNOTATIONS`, `VQA_IMAGE_ROOT`, `VQA_IMAGE_CHECKSUMS` theo vị trí
   giải nén thực tế, rồi dùng `configs/environments/kaggle.example.yaml`. Đường
   dẫn Kaggle không đưa vào `src/`. `run_root` là `/kaggle/working/vqa_runs`.
4. Ghi lại GPU, package versions, dtype, revision và hash dữ liệu. Đo tốc độ trên
   một đoạn train ngắn với loss hữu hạn, token/labels đúng trước lần chạy dài.
   Không coi `T4 x2` là một GPU 32 GB; trainer hiện chỉ dùng một GPU.
5. Trước khi kết thúc phiên, lưu version có output/export checkpoint rồi kiểm
   file đã tồn tại ở storage bền vững. `/kaggle/working` không phải bản sao lưu
   bảo đảm sống qua mọi lần ngắt. Không tắt phiên khi chưa xác nhận bản lưu.

Checkpoint-400 hiện thuộc source và môi trường Colab cũ. Không lấy source tối ưu
mới rồi ép resume: cổng source/environment/dtype sẽ từ chối đúng thiết kế. Nếu
tiếp tục checkpoint cũ, cần source cũ và migration đã kiểm; nếu đổi BF16 emulation
sang FP16, đó là run mới với provenance riêng, không tuyên bố resume tương đương.
Chưa chọn chạy lại M1 từ đầu chỉ vì đã có cấu hình Kaggle.

## Kiểm tra tối ưu preprocessing

`scripts/benchmark_sft_collator.py` so tensor của bản xử lý ảnh hai lần với bản
tái dùng image grid trên cùng câu train và processor Qwen ghim revision. Cài
Transformers 4.57.6, Torch/Torchvision CPU nếu chạy kiểm tra này trên local;
không tải trọng số backbone và không train trên CPU. Ví dụ:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_sft_collator.py --samples 32 --repeats 3 --output artifacts/benchmarks/collator_report.json
```

Report kiểm input IDs, attention mask, labels, pixel values và image grid. Tốc độ
đo là preprocessing trên CPU, không được dùng làm hệ số tăng tốc GPU hay ETA M1.
Không đổi eval schedule, dữ liệu hoặc batch trong cùng lần sửa preprocessing.

## Nguồn tham khảo thiết kế (không phải đồ án sinh viên)

- DrivenData CCDS: data/raw, data/processed, package dùng chung, source control:
  https://cookiecutter-data-science.drivendata.org/using-the-template/
- Lightning-Hydra Template: cấu hình paths/environment tách cấu hình thí nghiệm:
  https://github.com/ashleve/lightning-hydra-template
- Salesforce LAVIS: project vision-language có module dữ liệu/model/task và configs:
  https://github.com/salesforce/LAVIS

Chỉ áp dụng nguyên tắc phù hợp. Không chuyển trainer sang Lightning/Hydra hoặc
thêm registry phức tạp khi CLI + YAML đã đủ cho quy mô hiện tại.
