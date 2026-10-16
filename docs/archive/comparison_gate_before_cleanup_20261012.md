# Kiểm tra trước khi so sánh điểm

## Trạng thái hiện tại (07/10/2026)

M1 đã hoàn tất một epoch; chọn checkpoint-3683 theo loss trên 256 validation
cố định. Mốc 1.200 và 3.683 đều đã xuất đủ 1.407 câu, khớp protocol M0/800.
Không chạy lại train, inference hoặc test cho bước nghiệm thu này.

Ba báo cáo paired image-bootstrap (2.000 lượt, seed 42) nằm ở
`artifacts/m1_validation_analysis_20261007`. Khi so hai checkpoint M1, dùng
`compare_runs.py --left-experiment M1_SFT --right-experiment M1_SFT
--checkpoint-comparison`, cùng các cờ annotation/hash và bootstrap.
Chế độ này chỉ cho validation đầy đủ, adapter khác nhau; không đổi nhãn M1
thành M2 để qua kiểm. Khoảng tin cậy không đo biến thiên giữa các seed train.

Bộ rà 100 ảnh của checkpoint-3683 ở `artifacts/m1_validation_review_20261007`.
Ảnh/câu được chọn bằng hash của ID và seed 42, một câu mỗi ảnh; không dựa vào
prediction hoặc điểm. Mở `index.html`, nhìn ảnh rồi ghi đáp án, mức hỗ trợ,
bằng chứng và người duyệt. Xuất JSONL để lưu, nhập lại khi tiếp tục. Hiện
100 dòng đều pending. Đây là bộ chẩn đoán song song, không còn là điều kiện
số lượng để chốt M1. Biên bản `validation_loss_full_validation_v2` vẫn kiểm
checkpoint, model/revision, loss-selection và đủ 1.407 prediction trước khi
sinh bộ preference mới. Nhãn được đính kèm phải là nhãn đã duyệt thật, có hash.
`scripts/prepare_validation_review.py` kiểm full coverage và hash trước khi
tạo bộ này; từ chối ghi đè. Không thay bộ calibration train bằng bộ validation.

## Lịch sử ngân sách và cổng trước (04/10/2026)

Người làm đồ án đã chọn M1 đủ một epoch: giao thức `vqa_nle_m1_epoch1_v1`,
đích 3.683 bước. Lượt 1.100 → 1.200 hiện giữ nguyên; chỉ nối khi checkpoint
1.200 đã lưu và qua kiểm. Sau một epoch, bổ sung loss của checkpoint-3683
trên đúng 256 câu cố định: mốc cuối không có eval theo lịch 100 bước.
`final_validation` của trainer là kết quả sau load-best, không mặc định
thuộc 3.683. Chọn loss thấp nhất, xuất đủ 1.407 validation cho 1.200 và
ứng viên đó nếu khác; so với
800/M0 và chấm ảnh. Các mốc 1.000 phía dưới là lịch sử ngân sách trước.
Không chạy prediction cả 36 checkpoint, không dùng test chọn và không tự
đổi dtype/source của lineage để tăng tốc.

Lượt 900 → 1.000 đã hoàn tất: mã thoát 0, trạng thái `paused`, checkpoint
đã qua kiểm adapter/optimizer/scheduler/RNG. Chưa đánh giá đủ 1.407 câu
cho 1.000. Các hướng dẫn
600/800 phía dưới ghi lại giao thức và kết quả lịch sử, không phải lệnh cần
chạy lại. Giữ nguyên prediction M0, M1-600, M1-800 và 100 ứng viên train.

### Lịch sử lượt giới hạn tới 1.200

Ngân sách trước cho phép 1.000 → 1.200. Lượt này mất runtime sau
log 1.180; payload checkpoint-1100 đã kiểm trên CPU, đủ để qua bước kiểm
resume. Tab hiện tại có T4 và đã mount Drive; trạng thái thực thi ở TIEN_DO.
Quyết định một epoch phía trên thay thế đích dừng và các mốc đánh giá của
ngân sách cũ, nhưng giữ nguyên source, dữ liệu và kiểm tra resume.
Quy tắc M1 lịch sử ở [sft_full.md](sft_full.md); công việc hiện tại ở
[kế hoạch](research_plan.md).
Không chọn checkpoint mới nhất một cách tự động hoặc dùng test để chọn.

Loss 600/700/800/900/1.000: 1,121677 / 1,115386 / 1,106721 / 1,098756 /
1,095884. Bước 1.000 có loss thấp nhất trong các mốc này, nhưng chưa phải
baseline đã chốt. Khi đánh giá, giữ release `source_20261002_062529_455947`
và giao thức của 600/800; tạo job mới, dùng lại M0 nếu manifest khớp.
Đường dẫn checkpoint thật và hash ở TIEN_DO; không dùng lại ô 8–9 đang cấu
hình cho 800 như thể đó là kết quả của 1.000.

## Quyết định sau checkpoint-800 (02/10/2026)

Checkpoint-800 đã lưu đủ và lượt chạy kết thúc `paused`, mã thoát 0.
Không chạy lại ô train 600 → 800. Chưa chốt 800 làm M1 cuối cùng.

### Dữ liệu đã học

- Tập train có 29.459 mẫu; validation đầy đủ có 1.407 mẫu, test có 1.968 mẫu.
- Trainer của release 28/09 dùng batch 1, tích lũy gradient 8, một GPU.
  800 optimizer steps tương đương khoảng 6.400 lượt mẫu, khoảng 21,7% một epoch.
  Đây là lượng xử lý danh nghĩa, không phải kiểm kê số ảnh/câu hỏi duy nhất;
  các bước tính lại sau mất phiên không cộng thành dữ liệu mới.
- Log xác nhận epoch 0,22. Không được mô tả là đã train hết 29.459 mẫu.
- Loss trên 256 validation cố định: 600 = 1,121677; 700 = 1,115386;
  800 = 1,106721. Chưa có bằng chứng hội tụ hoặc chất lượng giải thích đạt yêu cầu.
  Không thêm dataset hoặc đổi split chỉ vì GPU chậm.

### Lượt đánh giá tiếp theo

1. Dùng lại release đánh giá `source_20261002_062529_455947` và môi trường
   đã ghi trong manifest của checkpoint-600. Không dùng source train thay thế.
2. Adapter: `training_runs/sft-full_20261002_113657_805762/training/checkpoint-800`.
3. Tạo job mới `evaluation_local/baseline_validation_800_v1` trong Drive đang mount
   (thư mục `evaluation_runs` hiện là shortcut chỉ đọc tới kết quả cũ).
   Giữ nguyên báo cáo và dự đoán của 600.
4. Tái sử dụng nguyên M0 và manifest từ
   `evaluation_runs/baseline_validation_600_mount_v1` qua hàm
   `prepare_validation_job` trong ô đánh giá notebook 03.
   Exporter phải kiểm tra hash và giao thức trước khi bỏ qua M0.
5. Chỉ xuất M1-800 trên đủ 1.407 validation: seed 42, ảnh 200.704 pixels,
   tối đa 128 token, cùng prompt, model revision, quantization và dtype như 600.
   Không chạy suy luận toàn bộ bằng CPU để thay thế GPU.
6. So sánh EM, token F1, ROUGE-L của 600 và 800 trên cùng ID. Chấm tay cùng
   tập 100 ảnh validation cố định để chẩn đoán; không chọn riêng câu trả lời đẹp.
   ROUGE-L tăng không đủ chứng minh lời giải thích bám ảnh.
   Bộ 100 ứng viên train để hiệu chỉnh preference là bộ khác, không thay thế
   đánh giá trên validation. Đánh giá người cuối cùng dùng 200 ảnh test theo kế hoạch.

### Chọn bước tiếp theo từ kết quả thật

- Nếu 800 cải thiện đáp án và không giảm chất lượng giải thích: giữ làm ứng viên,
  cân nhắc một lượt giới hạn tới 1000 nếu ngân sách GPU cho phép, rồi so lại.
- Nếu điểm gần như không đổi hoặc giảm: phân tích các câu đổi đáp án và chấm tay
  trước khi train thêm. Chênh lệch vài câu không đủ để tuyên bố cải tiến chắc chắn.
- Nếu lỗi chủ yếu do thiếu kiến thức/nhận diện và loss còn giảm: có cơ sở train tiếp
  trên tập hiện tại. Nếu lỗi do giải thích bịa hoặc nhãn sai: ưu tiên kiểm nhãn và
  thiết kế bước học giải thích/preference; tăng số bước không bảo đảm sửa được.
- Chưa tự chạy DPO hoặc đánh giá test. Chỉ dùng test sau khi đóng băng lựa chọn.

Lượt 600 → 800 mất 8.521,65 giây (khoảng 2 giờ 22 phút), gồm validation/lưu
checkpoint. Nếu tốc độ tương tự, thêm 200 bước cần khoảng 2 giờ 22 phút;
đi từ 800 tới 3683 còn khoảng 34 giờ. Đây là ngoại suy, không phải cam kết.
Không dùng `train_steps_per_second` in ở cuối lượt resume để tính tốc độ thực:
các bộ đếm tích lũy có thể bao gồm bước trước resume.

Trạng thái 03/10: đã hoàn tất 1.407/1.407 validation cho 800, gate không có lỗi.
EM 600 → 800: 66,10% → 67,95%; Token-F1: 71,61% → 73,19%;
ROUGE-L: 40,37% → 41,59%. Có 61 câu EM sai → đúng và 35 câu đúng → sai.
M0 tái sử dụng nguyên byte. Tại thời điểm đó giữ 800 làm ứng viên; chuẩn bị
ứng viên preference train-only và phần chấm ảnh, không suy ra grounding từ ROUGE.

## Chạy đánh giá đầy đủ cho checkpoint M1

Sau khi kiểm tra checkpoint đã lưu, dùng cùng một lệnh để xuất M0, xuất M1
và so sánh đủ 1.407 câu validation. Lệnh này không huấn luyện và không đọc test:

```bash
python -m scripts.run_baseline_validation --environment configs/environments/colab_full.example.yaml --checkpoint /content/drive/MyDrive/Vietnamese_VQA_NLE/runs/sft-full_20261001_172354_777822/training/checkpoint-600 --output-dir /content/drive/MyDrive/Vietnamese_VQA_NLE/runs/baseline_validation_600_v1
```

Local dùng config đường dẫn local và checkpoint đã tải về; không sao chép đường
dẫn Drive vào `src/`. Thêm `--dry-run` để kiểm tra ảnh, hash validation và checkpoint,
in lệnh mà không nạp GPU. Khi ngắt phiên, chạy lại **cùng lệnh** với `--resume`.
M0/M1 chạy tuần tự bằng hai tiến trình riêng để giải phóng bộ nhớ GPU giữa hai nhánh.
Runner bắt đầu mới nếu cả file dự đoán và manifest chưa tồn tại. Nếu một trong
hai đã có, exporter phải kiểm tra resume; thiếu manifest không được bỏ qua.

Mỗi export có manifest ghim nguồn, dữ liệu, ảnh, adapter và môi trường. Chạy tiếp
chỉ bổ sung ID còn thiếu sau khi kiểm tra các dòng đã lưu. Nếu nguồn, trọng số,
giao thức hoặc môi trường đổi, chọn output mới hoặc khôi phục đúng môi trường;
không trộn dự đoán. Dòng JSON cuối bị ghi dở sẽ làm lệnh dừng để kiểm tra, không
tự xóa dữ liệu. Dùng một phiên ghi duy nhất, không chạy đồng thời trên hai VM.
Khóa file chỉ bảo vệ các tiến trình khi hệ thống file hỗ trợ khóa, không phải
bảo đảm phối hợp giữa nhiều máy qua Drive.

### Dùng lại M0 cho checkpoint mới

Runner hiện hỗ trợ `--m0-predictions /path/to/previous/m0.jsonl`. Chỉ dùng khi
export và manifest đã có đủ 1.407 câu, cùng ảnh, nhãn, prompt, revision,
tham số sinh, source suy luận và phiên bản thư viện. Nếu thiếu câu hoặc không
khớp, runner dừng trước khi nạp model; không ghi tiếp vào cache.

Khi cache hợp lệ, chỉ còn hai công đoạn: xuất M1 và so sánh. Không tạo bản sao
M0 trong output mới. `m0_cache.json` ghi đường dẫn, SHA-256 prediction/manifest
và mã validator; resume phải giữ nguyên các hash này. Cache không được nằm
bên trong output mới. File gốc và kết quả 600/800 giữ nguyên.

M0 hiện có được tạo bằng release đánh giá `source_20261002_062529_455947`,
không phải source train 28/09 hoặc source local hiện tại. Vì vậy không chạy
cache này trong môi trường M1 đang train. Khi chuyển sang đánh giá, khôi phục
release/thư viện của ô 8, rồi chạy **wrapper mới ở ngoài cây source đã ghim**.
Đặt `PYTHONPATH` trỏ tới `src` và repo của release đánh giá trước khi chạy
wrapper; các exporter và `project_root()` phải lấy từ release đó. Kiểm bằng
`--dry-run` trước; không sửa manifest/hash cũ để ép cache khớp. Wrapper mới
chưa được chạy GPU trên checkpoint 1.000/1.200.

`commands.json`, `console.log`, hai export và manifest nằm trong cùng output.
Chỉ có `full_validation.json` sau khi hai nhánh xuất xong và qua cổng coverage.
Báo cáo đã có không bị ghi đè. Checkpoint-600 là ứng viên cần đánh giá, không
phải bằng chứng chất lượng đã đủ hay quyết định dừng train cuối cùng.

## Cổng so sánh

`scripts/compare_runs.py LEFT.jsonl RIGHT.jsonl --output runs/comparison/report.json`

Lệnh chỉ tính EM/F1 và chênh lệch đáp án khi:

- Hai file có đúng cùng tập question_id, không trùng ID; ghép theo ID, không theo dòng.
- `image_id`, đường dẫn và SHA-256 ảnh do exporter ghi, câu hỏi, đáp án tham chiếu và split
  khớp nhau. Thiếu hoặc sai định dạng SHA-256 thì từ chối so sánh đáp án.
- Cùng model/revision, seed, min/max pixels, max_new_tokens, prompt_sha256,
  generation_config, load_in_4bit và compute_dtype; không trộn nhiều protocol
  trong một file.
- Không thiếu provenance. File không tương thích trả exit code 2 và report không có scores.
- Trong mỗi file chỉ có một split, không gộp validation với test. Các trường
  `experiment_id` và `adapter`, nếu được ghi, phải nhất quán trong file;
  hai file vẫn được phép đại diện cho hai adapter/nhánh khác nhau.

Khi nghiệm thu trên **toàn bộ validation** của `image_disjoint_v1`, thêm
`--expected-validation-annotations` trỏ tới bản `validation.jsonl` đã chốt:

```bash
python scripts/compare_runs.py LEFT.jsonl RIGHT.jsonl --output runs/comparison/full_validation.json --expected-validation-annotations data/processed/vivqax/image_disjoint_v1/validation.jsonl --expected-validation-sha256 7f0b2b7d374d386030ddea50bc57d3e8e6c76e257e9fc4d1203bfb331b5f5b6e
```

Cổng này đòi manifest có đúng 1.407 question ID duy nhất, mọi dòng mang
`split=validation`, và cả hai file dự đoán có **đúng cùng tập ID** đó, không
thiếu/thừa hoặc gắn split khác. Mỗi prediction còn phải khớp manifest gốc ở
`image_id`, `image_path`, câu hỏi, đáp án, lời giải thích tham chiếu và split.
File thứ nhất phải ghi `experiment_id=M0_PRETRAINED`, `adapter=null`; file thứ hai
phải ghi `experiment_id=M1_SFT` và một đường dẫn adapter không rỗng, nhất quán
trong cả file. Thiếu metadata hoặc hai file cùng là M0/M1 đều bị từ chối.
Báo cáo lưu SHA-256 của manifest; khi dùng manifest chính thức, **bắt buộc**
truyền `--expected-validation-sha256` để kiểm tra byte trước khi so điểm. Hash
trong ví dụ là của bản `image_disjoint_v1/validation.jsonl` local ngày
24/09/2026; bản trên Colab phải đối chiếu cùng hash. Nếu thiếu/thừa ID,
report vẫn được ghi nhưng `comparable=false`, `scores=null`, exit code 2.
Với smoke 20 câu hoặc phân tích tập con có ghi rõ phạm vi, không dùng cặp cờ
manifest/hash này. Cổng từ chối metadata rỗng hoặc sai kiểu, gồm seed và hash
prompt, kể cả khi cả hai bản export cùng thiếu provenance.
Cổng không có manifest vẫn dùng được cho smoke cũ nhưng chỉ kiểm tra hai export
khớp nhau, không chứng nhận tham chiếu chuẩn hay danh tính M0/M1. Hai cờ report
`canonical_references_checked` và `run_identity_metadata_checked` phản ánh
đúng mức kiểm tra này. `experiment_id`/`adapter` là metadata do exporter ghi,
không phải bằng chứng mật mã rằng trọng số adapter đã được nạp. Manifest hiện
không có SHA-256 ảnh; hash ảnh chỉ được so giữa hai export, không đọc lại file ảnh.

**Giới hạn hiện tại:** output smoke cũ chưa ghi đủ provenance nên bị từ chối;
không được tự điền các trường thiếu bằng suy đoán. Inference exporter đã bổ sung
metadata lấy từ runner đã nạp: generation_config, hash system/user/chat template,
min/max pixels, token limit, quantization và dtype; mỗi ảnh có SHA256 để truy vết.
Đã test luồng export hai file rồi qua gate bằng runner giả lập, không dùng GPU.
Ngày 23/09/2026, pipeline so sánh smoke M0/M1 đã chạy xong trên T4 với cùng
20 câu validation. Gate trả `Comparable: true`, `issues: []`; EM cả hai bằng
0,35 và token F1 cả hai bằng 0,403333. Run trên Drive:
`Vietnamese_VQA_NLE/runs/paired_smoke_20260923_003416_446950/`.
Đây vẫn chỉ là smoke test 20 câu, chưa phải đánh giá chất lượng cuối cùng.
Không sửa notebook legacy hoặc kết quả cũ chỉ để cho gate qua.

Nếu cả hai file có lời giải thích hợp lệ và
cùng tham chiếu cho toàn bộ câu hỏi, report tính thêm ROUGE-L F1 theo token
ngăn bằng khoảng trắng: chuẩn hóa Unicode NFC, viết thường, bỏ dấu câu, lấy
điểm cao nhất trên các lời giải thích tham chiếu
cho từng câu rồi lấy trung bình. Phần này nằm riêng trong
`explanation_rouge_l_f1`; thiếu dữ liệu ở bất kỳ câu nào thì không tính điểm
lời giải thích trên tập con, nhưng vẫn giữ kết quả so sánh đáp án. ROUGE-L chỉ
đo trùng khớp từ/ngữ thứ tự, **không chứng minh lời giải thích bám ảnh**;
cần chấm tay theo rubric visual grounding trong kế hoạch. Cổng cũng không
kiểm tra rò rỉ training hay kiểm định thống kê.
Không diễn giải chênh lệch điểm trên 20 câu thành kết luận khoa học.

## So sánh M0–M3 trên validation hoặc test

CLI hiện hỗ trợ chọn hai nhánh bất kỳ bằng `--left-experiment` và
`--right-experiment`: M0_PRETRAINED, M1_SFT, M2_SFT_CHOSEN, M3_DPO.
Ví dụ sau chỉ chạy khi đã đóng băng model/giao thức và có prediction thật:

```powershell
python scripts/compare_runs.py runs/m2/test.jsonl runs/m3/test.jsonl --output runs/comparison/m2_m3_test.json --expected-annotations data/processed/vivqax/image_disjoint_v1/test.jsonl --expected-sha256 <HASH_TEST_DA_CHOT> --split test --left-experiment M2_SFT_CHOSEN --right-experiment M3_DPO
```

Thay `<HASH_TEST_DA_CHOT>` bằng hash manifest đã chốt, không hash một tập con
để bỏ qua coverage. Cổng đòi đủ 1.968 câu test hoặc 1.407 câu validation,
đúng split, ID, tham chiếu và danh tính nhánh đã chỉ định. M0 không có adapter;
M1/M2/M3 phải có adapter path nhất quán. Không dùng đồng thời nhóm cờ generic
này với `--expected-validation-*`; lệnh validation cũ vẫn giữ mặc định M0/M1.
Đây là kiểm metadata, không chứng minh checkpoint thực sự được nạp. Chưa có
kết quả M2/M3 thật; test tự động sử dụng dữ liệu giả lập tạm thời.

Thêm `--bootstrap-resamples 2000 --bootstrap-seed 42` để tính CI95 chênh
EM/F1 sau khi gate đủ ID/tham chiếu đã qua. Công cụ yêu cầu canonical có hash;
không tự chấm tập giao hoặc các run không tương thích. Lấy mẫu theo image ID,
giữ tất cả câu của một ảnh cùng nhau, điểm vẫn được tính theo câu hỏi. Cluster
sắp theo image ID để seed có thể tái lập. Đây là bất định lấy mẫu của hai run,
không đo độ ổn định giữa seed huấn luyện hoặc sự hỗ trợ của ảnh.

Bảng cuối còn dùng BERTScore làm chỉ số ngữ nghĩa phụ. Trước hết chạy
`scripts/evaluate.py ... --bertscore --output ...` riêng cho từng prediction
đã đóng băng. Sau đó truyền đúng hai báo cáo bằng `--left-semantic-report` và
`--right-semantic-report` vào lệnh `compare_runs.py`. Gate kiểm SHA-256 của
prediction, split, đủ ID/image ID, aggregate từ điểm từng câu và protocol
model/commit/layer/version giống nhau trước khi tính delta hoặc bootstrap.
Không sửa JSON để ghép báo cáo với prediction khác. Ví dụ validation M0–M1:

```powershell
python scripts/compare_runs.py M0.jsonl M1.jsonl --output M0_M1_semantic.json --expected-validation-annotations data/processed/vivqax/image_disjoint_v1/validation.jsonl --expected-validation-sha256 <HASH_VALIDATION> --left-semantic-report M0_metrics_bertscore.json --right-semantic-report M1_metrics_bertscore.json --bootstrap-resamples 2000 --bootstrap-seed 42
```

BERTScore không nhìn ảnh; CI của nó cũng không biến chỉ số văn bản thành bằng
chứng visual grounding. Kết luận chính vẫn cần rubric chấm ảnh đã định trước.

## Notebook smoke lịch sử

`notebooks/archive/04_paired_smoke.ipynb` gọi cùng CLI hai lần bằng subprocess riêng,
dùng chung tham số và adapter M1 đã lưu. Nó kiểm tra thêm SHA-256 ảnh và dtype,
đủ 20 ID duy nhất, lưu log/command/dependency freeze và completion.json trên Drive.
Source snapshot được cố định bằng SHA-256, không dùng source overlay của notebook cũ.
Chạy lại setup tạo run mới; cell inference từ chối ghi đè run dở.
Đã kiểm tra cú pháp và luồng lệnh bằng test local; không thay thế kiểm thử GPU.
Ngày 23/09: giải phóng phiên notebook 02 sau khi xác minh xong, chạy đủ cả 3 ô
của notebook 04; bản có output đã đồng bộ về local.
