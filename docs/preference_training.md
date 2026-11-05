# Huấn luyện và đánh giá M2/M3

Bộ chính đã khóa 1.000 cặp. M2 học chosen bằng SFT; M3 học chosen/rejected
bằng DPO. Cả hai bắt đầu từ M1-3683, cùng seed, learning rate 1e-5,
batch hiệu dụng 8 và ba epoch / 375 update. DPO dùng beta 0,1 và reference
M1 frozen. Thiết kế đầy đủ ở [kế hoạch](research_plan.md).

Seed 42 và 43 đã train xong, có validation/BERTScore/CI kiểm độc lập.
Seed 44 đã hoàn tất hai nhánh và xuất đủ validation ngày 19/10. Ngày 20/10
đã tính BERTScore seed 44 và tổng hợp đủ ba seed. Không chạy lại một lượt
đã hoàn tất chỉ để xem output.
Ngày 22/10, checkpoint cuối đối chứng LR 5e-6 đã kiểm độc lập; hai nhánh
đều `passed`, 375 update. M3 dùng attempt `m3_train_resume175`. Export
validation đã hoàn tất 1.407 câu mỗi nhánh. Ngày 23/10 đã kiểm prediction,
manifest, hồ sơ train và ảnh thật; EM/F1/ROUGE-L cùng CI đã lập. BERTScore LR
hoàn tất trên T4 lúc 18:01 ngày 23/10: M2/M3 77,16 / 81,65. File thật và
CI đã kiểm độc lập; không train hoặc sinh lại prediction. Giữ LR 1e-5 cho
nghiên cứu chính ba seed; LR 5e-6 là đối chứng riêng.

## Notebook và code

| Tệp | Công việc |
|---|---|
| `notebooks/04_train_preferences.ipynb` | Setup → inspect → M2 gate/train → M3 gate/train → đọc report |
| `notebooks/05_evaluate_preferences.ipynb` | Setup đánh giá độc lập → inspect → xuất/nối tiếp validation → kiểm file kết quả |
| `scripts/launch_preference_training.py` | Chuyển launch record thành tham số stage |
| `src/vqa_explanation/training/preference.py` | Trainer SFT chosen và DPO |
| `src/vqa_explanation/training/preference_data.py` | Kiểm dữ liệu, study, completion và collator |
| `scripts/run_study_validation.py` | Xuất và kiểm prediction M0–M3 |

Notebook mới là template cho lượt sau, không thay bản Colab đã có output.
Mỗi seed dùng launch record/config riêng, không cần tạo notebook riêng.
M1 vẫn ở [notebook có output](https://colab.research.google.com/drive/13Abjl7kJyDKzbxyYsCetJ540OzdK2P_E)
và các snapshot trong `notebooks/archive/`, `artifacts/colab_history/`,
`artifacts/preference_v2/`. Không xóa hoặc đổi tên các bản này.

## Một lượt train

1. Chọn `LAUNCH_RECORD`, SHA-256 và `EXPECTED_SEED` đã khai trước train.
   Giữ `RUN_GATE=False`, `RUN_TRAINING=False` trong lúc setup/inspect.
2. Mount kho dự án một lần. Đặt `PROJECT_DIR` khi không tìm được duy nhất
   một kho chứa selection M1; không phụ thuộc email hoặc ID Drive cố định.
3. Khôi phục đúng source release và venv riêng. Bundle preference/ảnh được
   kiểm checksum; ảnh đọc từ ổ cục bộ của Colab, checkpoint/log lưu trên Drive.
4. Inspect cả M2/M3. Thiếu ảnh, review, selection hoặc config khác study
   sẽ dừng trước khi nạp model. Inspect không chứng minh GPU đã chạy đúng.
5. Bật `RUN_GATE` rồi chạy riêng ô gate của nhánh cần kiểm. Gate năm update
   dùng output riêng, kiểm policy/reference/trọng số lưu. Không lặp gate
   nếu source, dữ liệu, config và runtime không đổi và gate hợp lệ đã có.
6. Bật `RUN_TRAINING` rồi chạy ô train tương ứng. Train chính nạp M1 sạch,
   không học tiếp adapter gate. Hoàn tất M2 rồi làm M3.
7. Đọc `training_result.json`, Trainer state, checkpoint và log. Chỉ nghiệm
   thu khi đủ 375 update / epoch 3 và các kiểm trọng số/reference đã qua.

Template không tự bật train/export khi bấm Run all. Khi đã bật các cờ,
chỉ chạy ô cần thiết; không Run all qua những gate/run đã tồn tại.

Môi trường dùng `requirements-cloud-preference.txt` và
`scripts/setup_preference_environment.py`; không cài chồng thư viện train
vào kernel Colab hoặc thay môi trường M1. Process con dùng đúng interpreter,
bỏ Python path/home kế thừa và tắt user-site. Log được chuyển tiếp ra notebook.

### Nối tiếp sau mất phiên

Giữ nguyên record, config, dữ liệu và tổng ngân sách. Chạy lại setup/inspect,
điền checkpoint đã kiểm vào `RESUME` và đặt `ATTEMPT` mới, ví dụ
`m3: resume150`. Output mới là `m3_train_resume150`; run/gate cũ không đổi.

Launcher kiểm hash adapter, optimizer, scheduler, RNG, scaler FP16,
runtime/dtype/GPU và ngân sách trước khi resume. Không sửa `num_train_epochs`
để bù thời gian mất kết nối hoặc dùng checkpoint thiếu file.

## Validation

Notebook 05 không cần chạy notebook 04 trước. Nó mount cùng kho, đọc launch
đánh giá và kiểm hash launch train, seed, run group, source đánh giá.
Mặc định `RUN_EXPORT=False`; chỉ bật sau khi inspect thành công.
Không dùng test để chọn checkpoint hoặc điều chỉnh thiết kế.

Seed 43 chọn M2 `m2_train`, M3 `m3_train_resume150`. Hai attempt được khai
rõ trong record; không tìm thư mục mới nhất hoặc chép đè run M3 ban đầu.
CLI tương đương:

```text
python scripts/run_study_validation.py --environment <GPU_ENV> --study-config configs/experiments/preference_study_seed43.yaml --run-group <RUN_SEED43> --m1-adapter <M1_3683> --m3-stage m3_train_resume150 --output-dir <VALIDATION_SEED43> --inspect
```

Bỏ `--inspect` để xuất/nối tiếp cùng thư mục. CLI đối chiếu ID, manifest,
adapter, source và môi trường rồi bỏ qua các câu đã hoàn tất. Không ghép
prediction từ hai môi trường hoặc chép baseline chỉ theo tên file.
Một record validation mới phải ghim đúng source và các đầu vào trước khi chạy.

### Dùng lại baseline M0/M1

Notebook 05 chuyển `baseline_source` và `baseline_sha256` trong record cho CLI.
Hash phải ghim đúng bốn file: `m0.jsonl`, `m0.jsonl.manifest.json`,
`m1.jsonl`, `m1.jsonl.manifest.json`. Có thể truyền trực tiếp bằng
`--baseline-source <BASELINE_DIR> --baseline-sha256 '<JSON_HASH_MAP>'`.
Hai cờ phải đi cùng nhau; không kết hợp với `--relocate-from` vì phép đó
áp dụng cho toàn bộ study, gồm cả M2/M3.

Inspect kiểm đủ 1.407 câu, ID/reference, hash ảnh/annotation, M1, model revision,
source suy luận, Python/thư viện, prompt và generation đã ghi trong manifest.
Nguồn phải là hai baseline đầy đủ có cùng giao thức. Đổi địa chỉ adapter M1
được chấp nhận khi hash trọng số/config giữ nguyên; nội dung prediction giữ
nguyên, chỉ metadata đường dẫn được chuyển và ghi nhận rõ.

Khi chạy thật, chỉ sao chép M0/M1 đã kiểm; không nạp hai model baseline.
M2/M3 vẫn xuất/nối tiếp từ hai attempt đang chọn của lượt mới. Biên nhận
`baseline_reuse.json` giữ hash nguồn/đích và số câu. File đích đã khớp được
giữ nguyên khi nối phiên; file khác hoặc nguồn đổi sau inspect làm tác vụ dừng.
Không tự sinh lại baseline khi kiểm thất bại hoặc dùng M2/M3 cũ cho LR mới.

Source cũ chưa có hai cờ này. Lượt đánh giá mới phải tạo record mới ghim
release đã kiểm; không chỉnh record hoặc source release lịch sử để thêm tính năng.
Template 05 để trống `VALIDATION_RECORD`/`VALIDATION_SHA256` để chọn lượt mới,
giữ `RUN_EXPORT=False` tới khi inspect đạt.

Nếu file chính mất dòng nhưng còn recovery snapshot, dùng
`scripts/recover_validation.py` với cùng study, run, M1 và `--m3-stage`,
thêm `--source-dir`, `--output-dir` mới và `--inspect`.
Công cụ kiểm hash/prefix trước khi sao chép, giữ nguyên bản lỗi. Bỏ `--inspect`
chỉ sau khi kế hoạch phục hồi đúng; record nối tiếp phải trỏ đích mới và ghim
biên nhận recovery. Không sửa record hoặc source cũ.
Source mới chặn validation trước khi nạp model nếu file chính bị mất/rút
ngắn so với snapshot đóng. Kiểm này không được vá vào release đã ghim.
Nếu chỉ đổi địa chỉ kho, `--relocate-from` của CLI validation cũng yêu cầu
output mới và đối chiếu identity; không bỏ kiểm để tiếp tục.

Notebook LR hoàn tất ngày 22/10 có hai ô bảo trì dùng một lần; giữ nguyên
để đối chiếu, không dùng Run all như template. Tiếp tục bằng template 05
sạch, chọn `TASK` và record ở đầu; không nối thêm ô sửa phiên ở cuối.

Khi ngắt lệnh hoặc có lỗi callback, helper dừng cây tiến trình do lệnh đó
tạo ra. Đây không phải thao tác lưu checkpoint mới; sau ngắt vẫn phải kiểm
file bền vững và chỉ tiếp tục từ checkpoint/snapshot đã xác minh.

## Bảng số liệu và các phép đo riêng

`scripts/report_study_validation.py` đọc output đã đủ; thêm `--semantic-dir`
khi có report BERTScore. Công cụ kiểm canonical, ID/reference, manifest và
evidence train trước khi lập bảng. Đếm dòng trong notebook chỉ để theo dõi,
không thay các kiểm này. BERTScore dùng môi trường semantic riêng.

Notebook 05 có hai tác vụ: `validation` sinh/nối prediction; `semantic`
chấm prediction đã lưu. Với semantic, chọn `ANALYSIS_RELEASE` và checksum
của source phân tích đã kiểm. Không sửa record source suy luận lịch sử.
Các ô vẫn chạy theo thứ tự setup → inspect → chạy → kết quả; không cần
khôi phục ảnh, adapter Qwen hoặc chạy notebook 04 trước.

```text
python scripts/setup_cloud_environment.py --profile semantic --environment-dir <SEMANTIC_ENV>
<SEMANTIC_PYTHON> scripts/run_study_semantic.py --results <VALIDATION_DIR> --annotations <VALIDATION_JSONL> --output-dir <SEMANTIC_DIR> --inspect
```

Bỏ `--inspect` để chấm. Mặc định dùng CUDA, batch 16; đây là đánh giá văn bản,
không phải train trên CPU. CLI kiểm đủ 1.407 câu mỗi nhánh, tham chiếu,
manifest và giao thức trước khi nạp BERT. Kết quả lưu riêng từng nhánh;
khi nối phiên, chỉ tính nhánh thiếu, không ghi đè report đã có. Thiếu hoặc
đổi đầu vào/giao thức thì dừng, không tự bỏ qua lỗi. `completed.json` chỉ
được tạo sau khi cả bốn report đã qua kiểm.

Sau khi lập đủ report từng seed kèm BERTScore:

```text
python scripts/report_validation_seeds.py --seed42 <REPORT42_JSON> --seed43 <REPORT43_JSON> --seed44 <REPORT44_JSON> --output-dir <NEW_SUMMARY_DIR>
```

Công cụ đọc lại manifest, prediction, semantic report và hồ sơ train; tính
lại điểm trước khi tổng hợp. M0/M1 là baseline cố định. M2/M3 và chênh lệch
M3−M2 báo mean ± sample SD qua ba seed. CI bootstrap theo ảnh của từng seed
giữ riêng; không lấy trung bình CI hoặc coi ba baseline lặp là ba run mới.

Đối chứng đổi ảnh dùng `scripts/check_image_dependence.py`. Giữ bundle
200 ảnh validation đã cố định, không prepare lại sau khi xem kết quả.
Dùng đúng adapter cuối của các nhánh và thư mục output riêng. Đây là phép
đo độ nhạy ảnh, không phải điểm VQA trên bộ ảnh hoán vị.

ROUGE-L/BERTScore đo tương đồng văn bản, chưa chứng minh giải thích được ảnh
hỗ trợ. Cần đủ ba seed, test sau khóa thiết kế và chấm mù thật theo
[hướng dẫn đánh giá](human_evaluation.md).

## Đối chứng learning rate

`preference_study_lr5e6.yaml` khai báo một đối chứng seed 42, cùng bộ 1.000
cặp, M1 và ba epoch. Hai config `sft_chosen_lr5e6.yaml`, `dpo_lr5e6.yaml`
chỉ đổi LR từ 1e-5 xuống 5e-6. Đây là bảng độ nhạy riêng, không thay sáu
run chính hoặc tăng số seed của nghiên cứu chính.

Launch ghim hồ sơ seed 42 gốc. Kiểm trước train từ chối nếu đổi M1, dữ liệu,
ngân sách, beta, giới hạn token hoặc dùng lại run group cũ. Trainer và
collator giữ nguyên; gate phải chạy riêng vì cấu hình LR đã đổi.
Giữ notebook 04 có output lịch sử; bản thực thi đối chứng dùng cùng template
16 ô, không chèn controller hoặc phần đánh giá vào notebook lịch sử.

## Các lượt lịch sử

Lượt 186 cặp ngày 14/10: mỗi nhánh một epoch / 24 update, LR 5e-6.
Giữ config `sft_chosen.yaml`, `dpo.yaml`, launch v1, checkpoint và output
nguyên vẹn; không dùng cho run mới hoặc đổi nhãn thành kết quả chính.

Điểm vào chính là notebook 04 hoặc `scripts/launch_preference_training.py`
với launch record đã ghim. `train_preference.py` và wrapper `train_dpo.py`
là CLI mức thấp: mặc định study/config thuộc lượt 186 cặp. Nếu gọi trực tiếp,
phải chọn rõ `--study-config` và `--config` của lượt cần chạy; không dùng
mặc định để khởi chạy bộ 1.000 cặp. Không đổi config lịch sử để sửa mặc định.

Xử lý cặp đã chuyển sang CLI `expand_preferences.py`,
`finalize_preference_review.py` và [quy trình dữ liệu](preference_annotation.md).
Không nối các ô mở rộng dữ liệu vào notebook train đã hoàn tất.
