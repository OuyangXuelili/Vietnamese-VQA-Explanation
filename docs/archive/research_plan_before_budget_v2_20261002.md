# Kế hoạch thực thi đồ án VQA tiếng Việt có giải thích

Cập nhật 30/09/2026. **Đây là bản kế hoạch chính để thực hiện từ lúc này.**
Đọc từ P0 đến P5; không chạy lại notebook 00–08 theo thứ tự. Tiến độ thực tế
và đường dẫn run nằm trong [TIEN_DO.md](../TIEN_DO.md). Các kế hoạch trước được
thu gọn ở cuối tài liệu, không dùng làm danh sách việc hiện hành.

## 1. Mục tiêu, phạm vi và tiêu chí hoàn thành

Tên đề tài giữ nguyên: **Nghiên cứu cải thiện chất lượng lời giải thích trong
hỏi đáp hình ảnh tiếng Việt**.

Câu hỏi chính: từ cùng checkpoint SFT và cùng nguồn câu trả lời tốt, DPO có
giảm lời giải thích không được ảnh hỗ trợ hơn SFT tiếp không, và độ đúng đáp
án thay đổi thế nào? Đánh giá chất lượng văn bản dựa trên ảnh, không tuyên bố
văn bản phản ánh chính xác suy luận nội bộ của mô hình.

Chốt một backbone Qwen2.5-VL-3B-Instruct, revision trong
`configs/model/qwen25_vl_3b.yaml`, một bộ ViVQA-X `image_disjoint_v1`, bốn nhánh:

| Nhánh | Điểm khởi đầu | Dữ liệu/cách học | Vai trò |
|---|---|---|---|
| M0 | Qwen gốc | Không train | Mốc trước thích nghi |
| M1 | Qwen gốc | SFT 29.459 câu, một epoch | Baseline VQA + giải thích |
| M2 | Adapter M1 được chọn | SFT chosen-only, hai epoch | Tách lợi ích của dữ liệu tốt |
| M3 | Đúng adapter M1 đó | DPO chosen/rejected, hai epoch; reference M1 cố định | Đo lợi ích học theo cặp |

Hai epoch M2/M3 là **ngân sách thiết kế ban đầu**, không phải thông số tối ưu
đã tìm được. Cùng câu hỏi/chosen, số lượt tiếp xúc và batch; DPO đọc thêm
rejected/reference nên không gọi là cùng chi phí tính toán.

Đủ phần nghiên cứu khi có: manifest dữ liệu đóng băng, checkpoint M1–M3,
prediction M0–M3 trên cùng held-out, bộ cặp đã duyệt, chấm giải thích bằng ảnh,
bảng kết quả có độ bất định, phân tích lỗi và kết luận trả lời câu hỏi trên.
DPO không bắt buộc thắng; kết quả âm có đối chứng vẫn phải được báo cáo.
Demo và báo cáo hoàn chỉnh là đầu ra cuối, không thay thế thực nghiệm.

Không thêm OCR, RAG, GRPO, KD, verifier, backbone thứ hai hoặc huấn luyện từ
đầu vào phạm vi bắt buộc. Không hứa tính mới toàn cầu từ việc ghép SFT/DPO;
đóng góp của người thực hiện là bộ cặp, thiết kế đối chứng và phân tích.

## 2. Điểm xuất phát và nguyên nhân chậm

- Đã có đủ 28.180 ảnh; train/validation/test = 29.459/1.407/1.968 câu.
  Audit, checksum, masking, SFT smoke, optimizer pilot và dừng chủ động/resume
  đã có bằng chứng. Ngày 27/09 local qua 190 test và full-data preflight.
- Ngày 30/09 xác nhận P0 đạt và M1 đã train 300/3.683 bước, run dừng chủ động
  có checkpoint trên Drive. Việc trước mắt là resume bằng source/môi trường
  đã ghim, không quét lại P0. Phiên 300 bước đầu mất ~3 giờ 17 phút; phần
  train còn lại ước tính ~37 giờ GPU nếu tốc độ tương đương, chưa tính gián đoạn.
- Chưa hoàn tất full M1, cặp ưu tiên nghiệm thu, trainer M2/M3, human evaluation,
  bootstrap và demo. Các file placeholder không được tính là hoàn thành.
- Cập nhật cuối 30/09: phiên resume ngắt ở log 481; checkpoint mới nhất thấy
  trên Drive là 400. Colab từ chối cấp GPU do quota. Ưu tiên tối ưu preprocessing
  có kiểm tra tensor tương đương, rồi xác nhận GPU/quota Kaggle miễn phí. Không
  tự thuê GPU, tạo thêm notebook hoặc ép checkpoint cũ chạy với source/dtype mới.
  Mốc 300 và ETA phía trên là lịch sử; xem TIEN_DO để biết trạng thái hiện hành.
- Chậm chủ yếu vì đi nhiều vòng kiểm tra hạ tầng trước khi chốt thực nghiệm;
  tài liệu lẫn lịch sử với việc tiếp theo; chưa định lượng công tạo nhãn/GPU.
- Từ nay **mỗi lần làm tiếp lấy công việc chưa hoàn thành đầu tiên trong P0–P5**.
  Không quay về khảo sát đổi đề tài hay chạy lại smoke nếu không có lỗi mới.

## 3. Lộ trình thực hiện

### P0 — Đóng băng đầu vào và môi trường chạy M1

Ngân sách công: tối đa một ngày làm việc, không tính thời gian chờ GPU.

1. Dùng ảnh/annotation đã kiểm tra, không tải lại COCO hoặc tạo split khác.
2. Rà token **một lần trên toàn train** bằng config mới
   `configs/training/sft_readiness_full.yaml`; không chạy stage `token-profile`
   mặc định vì stage đó vẫn lấy 512 mẫu lịch sử. Kiểm thêm 256 validation
   dùng tính loss bằng cùng processor/collator qua `--validation-count 256`;
   danh sách validation shuffle cùng seed 42 như M1 và được ghi trong report.
   Chỉ qua khi không thiếu image/assistant token và không vượt 1.536 token.
   Không cắt token ảnh hoặc âm thầm bỏ mẫu dài; nếu có lỗi, sửa chính sách rồi
   đóng băng lại trước khi train. Không quét test để chọn ngưỡng.
3. Lưu phiên bản thực tế Torch/Transformers/PEFT/Accelerate/bitsandbytes và
   processor; pin bộ cài tái lập từ runtime đã xác nhận. Không nâng package
   trong khi resume. M1 giữ Transformers 4.57.6 đã được kiểm thử.
4. Đóng băng một source release M1 có checksum. Khi đang chạy M1, tiếp tục
   phát triển M2/DPO/eval ở local; không thay source/config của runtime M1.
   Mọi lần resume M1 dùng lại **đúng release M1**, không lấy ZIP mới nhất.
5. Giữ full train một epoch. `sft.yaml` nay eval/save mỗi **100 bước** thay vì
   500 để giảm mất công khi Colab ngắt; eval vẫn dùng 256 câu cố định, giữ ba
   checkpoint. Đây là đổi config cho run mới, không sửa checkpoint cũ.
6. Chốt giờ GPU khả dụng. Không tự thuê GPU/tăng chi phí; free Colab không
   bảo đảm lịch. Nếu không đủ tiến độ, báo đúng chỗ nghẽn thay vì mở thêm pilot.

Đầu ra: báo cáo token có coverage, environment lock của GPU, release/hash,
cấu hình M1 và lịch phiên train. Không yêu cầu một pilot 20 bước mới chỉ vì
đổi tần suất save. Nếu đổi phiên bản train/backbone/collator, chỉ kiểm lại
đường bị thay đổi trước khi cho chạy dài.

### P1 — Chạy M1 chính thức và lấy baseline M0/M1

GPU: train M1, sau đó suy luận M0/M1 validation; **không chạy hai model cùng
GPU một lúc**. Trong thời gian GPU chạy, local thực hiện P2 phần công cụ,
P3 phần trainer/evaluation và viết chương dữ liệu/phương pháp.

1. Chạy `sft-full`, không `sft-smoke` hay `sft-optimizer-pilot`. Khởi đầu có
   thể đặt dừng bước 300 để làm phiên đầu của **chính run M1**; sau đó resume
   600, 900... tới cuối. Đây là các đoạn của cùng lịch train, không train lại
   từ đầu hoặc reset optimizer. Tổng bước cuối lấy từ Trainer, không tự làm
   tròn số mẫu/batch bằng tay.
2. Sau checkpoint đầu tiên, kiểm loss hữu hạn, bước tăng, checkpoint đủ và
   log thời gian/VRAM. Không dừng để tạo lại smoke nếu các dấu hiệu bình thường.
   Điều chỉnh số bước mỗi phiên dựa trên thời gian thực tế, luôn chừa thời
   gian save. Colab có thể ngắt trước lịch dự kiến; không dùng keep-alive né quota.
3. Full M1 phải có `training_result.json` passed, đủ target step, best checkpoint
   theo validation loss, adapter nạp được. Đánh giá validation chính là lần
   kiểm reload adapter; không thêm một notebook chỉ để reload một câu.
4. M0/M1 cùng đủ 1.407 validation, seed 42, min/max pixels 200704, 4-bit,
   max_new_tokens 128, cùng prompt/generation. Không dùng điểm smoke cũ để so.
5. Đọc cố định 100 câu validation có cả đúng/sai; viết phân loại lỗi, không
   chỉ chọn ví dụ đẹp. Từ đây chốt rubric và công cụ gán nhãn.

Đầu ra: adapter M1 đã chọn, M0/M1 validation JSONL + metric, bảng lỗi 100 câu.
Điểm M1 thấp hơn M0 không được xem là lỗi phần mềm nếu pipeline đúng. Chỉ sửa
và chạy lại khi có bằng chứng lỗi; mọi lần thử phải giữ lại kết quả.

### P2 — Tạo bộ cặp ưu tiên, ưu tiên chất lượng hơn số lượng

1. Chọn tối đa 2.000 câu train, ưu tiên một câu/ảnh, seed 42; tuyệt đối không
   dùng validation/test làm cặp train. 100 câu đầu dùng hiệu chỉnh hướng dẫn,
   sau đó duyệt lại theo rubric chốt và có thể giữ trong bộ train cuối.
   Đã có `scripts/select_preference_questions.py` để đóng băng pool, hash và
   100 ID hiệu chỉnh; xem lệnh trong `docs/preference_annotation.md`.
   Pool không chứa nhãn accepted hoặc cặp train; cần M1 và bước duyệt bên dưới.
2. Mục tiêu chính **1.000 cặp đã duyệt**. Dùng giải thích tham chiếu được kiểm
   bằng ảnh làm ứng viên chosen; sinh tối đa hai ứng viên/câu từ M1 để tìm
   lỗi tự nhiên. Có thể tạo lỗi có kiểm soát, nhưng phải ghi nguồn synthetic,
   kiểm ảnh và thống kê riêng. Không sinh vô hạn tới khi đủ cặp.
3. Chosen và rejected giữ đúng cùng ảnh/câu hỏi/đáp án đúng, cùng format;
   khác ở giải thích. Nếu hai giải thích đều đúng hoặc ảnh không rõ, loại cặp
   và lưu lý do, không để trợ lý tự đóng nhãn `accepted` như thể người đã duyệt.
4. Ghi question ID, image hash, answer, hai giải thích, cụm từ sai/bằng chứng,
   loại lỗi, nguồn model/checkpoint/prompt/seed, reviewer và trạng thái. Validator
   phải nối lại canonical train, chặn duplicate nội dung/ảnh sai/split sai,
   chỉ xuất accepted cho train. Đã có freeze/export trong `validate_preferences.py`
   với `--annotations --image-root --output-dir`; dùng `--input` riêng chỉ kiểm schema.
   Code này không thay người duyệt hay chứng minh cặp có chất lượng.
5. Người thực hiện duyệt từng cặp. Người thứ hai kiểm 20% ngẫu nhiên nếu có;
   sửa mọi bất đồng trước freeze. Nếu chỉ một người, ghi đúng giới hạn.
6. Nếu đạt ít hơn 1.000 cặp sau 2.000 câu, dùng số cặp sạch thực có và cập nhật
   cỡ mẫu trước M2/M3, không hạ tiêu chuẩn nhãn. Không tự gọi cỡ mẫu nào là
   bảo đảm đủ power. Nếu gần như không có rejected rõ lỗi, phải xem lại giả
   thuyết trên validation, không bịa lỗi để ép ra kết luận.

Đầu ra: candidates, review log, accepted pairs version/hash, chosen-only export,
thống kê loại lỗi/độ dài/nguồn/tỷ lệ loại; xem `preference_annotation.md`.

### P3 — Triển khai và chạy đối chứng M2/M3

**Phần code còn phải hoàn thành**, không chạy placeholder rồi báo thành công:

| Hạng mục | Cách triển khai | Điều kiện nghiệm thu |
|---|---|---|
| M2 | Tái dùng collator/loader SFT, tải adapter M1 trainable; đọc chosen export | Không khởi tạo lại LoRA ngẫu nhiên; chỉ accepted train; output có lineage M1 |
| M3 | Dùng DPOTrainer hỗ trợ VLM của TRL, không tự viết framework RL | Pin phiên bản tương thích, ảnh ở cả hai nhánh; completion-only log-prob |
| Reference | M1 đóng băng; ưu tiên tính trước reference log-prob để giảm VRAM | Cache gắn hash ảnh/cặp/model/processor/template; đối chiếu online một batch |
| Evaluation | Mở rộng comparator hiện có sang M0–M3 và test, không bỏ kiểm ID/protocol | Phát hiện file thiếu/trùng/sai nguồn; metric không phụ thuộc thứ tự dòng |
| Inference gián đoạn | Xuất shard/manifest hoặc resume với protocol hash | Mỗi ID đúng một lần; không trộn output khác model; không ghi đè run cũ |
| Human evaluation | Xuất bộ chấm ẩn model, nhập nhãn, thống kê/paired bootstrap | Seed cố định, lỗi nhãn bị chặn, không tự sinh nhãn người |

Ngày 29/09: comparator đã hỗ trợ cặp M0–M3 và manifest validation/test qua
`--expected-annotations --expected-sha256 --split --left-experiment
--right-experiment`. Cổng giữ nguyên kiểm đủ ID, tham chiếu, protocol và
metadata adapter. Đã kiểm bằng fixture; chưa có prediction M2/M3 thật.
Bootstrap/human evaluation và kiểm hash trọng số adapter vẫn chưa hoàn thành.

TRL hiện có ví dụ chính Qwen2.5-VL-3B; dùng làm nền triển khai, không cài bản
mới nhất tùy ý vào môi trường M1 đang chạy. Nếu cần phiên bản khác, tạo
environment M2/M3 riêng và kiểm nạp M1; cùng environment đó cho cả hai nhánh.
Reference không được vô tình trở thành Qwen gốc khi tắt adapter M1.

Chốt cấu hình ban đầu cho một seed 42, effective batch 8, hai epoch mỗi nhánh:
M2 LR 2e-5; M3 LR 5e-6, beta 0,1, sigmoid DPO, reference cố định. Đây là
điểm bắt đầu đăng ký trước, **chưa được GPU xác nhận**, không mặc định tối ưu.
Hai learning rate khác nhau nghĩa đây là so sánh hai công thức huấn luyện
đã định, không phải thí nghiệm chỉ thay đúng hàm loss. Nêu giới hạn này khi
kết luận; không quy toàn bộ chênh lệch cho DPO hoặc tuyên bố tốt hơn mọi SFT.
Giữ cùng prompt/ảnh và không cắt image token; validate độ dài cặp trước train.
Tắt dropout cho cả M2/M3, preprocessing ảnh xác định; reference chạy eval.
Config thiết kế ở `sft_chosen.yaml` và `dpo.yaml`, chưa phải trainer chạy được;
runner tương lai phải ánh xạ các trường riêng như `checkpoint_selection`,
`preflight_max_length`, không truyền nguyên YAML vào DPOConfig.

Chọn **checkpoint cuối sau hai epoch** cho cả M2/M3; không lấy minimum DPO
loss so với minimum SFT loss để chọn hai model theo hai tiêu chí khác nhau.
Lưu checkpoint cuối epoch 1 để chẩn đoán/resume, không chọn lại sau khi xem
test. M1 vẫn dùng best validation loss theo giao thức riêng đã định.

Test trước M2/M3: unit tests loss/mask/split/reference/cache và một lượt GPU
ngắn tối đa 10 update cho trainer mới; xác nhận policy đổi, reference không
đổi, loss hữu hạn, save/reload đúng. Đó là **một kiểm tra tích hợp cần thiết**,
không phải chuỗi smoke mới. Sau khi qua, chạy hai nhánh chính từ M1 sạch,
không dùng trọng số đã bị cập nhật trong test làm điểm khởi đầu.

Không mở grid search mặc định. Chạy xong validation đầy đủ, báo cả M2/M3 dù
kết quả xấu. Nếu có quyết định thêm cấu hình để xử lý vấn đề chất lượng, phải
ghi sửa protocol và ngân sách trước khi chạy, chỉ dựa validation; không
thay mục tiêu test để tìm một bảng đẹp. Ba seed là mở rộng nếu còn GPU, không
để nó chặn vòng thực nghiệm một seed và việc viết báo cáo.

### P4 — Đóng băng, đánh giá test và phân tích

1. Freeze adapter hashes, inference config, evaluator, rubric và manifest
   đánh giá. Sau đó chạy M0–M3 đủ 1.968 test/model, cùng câu hỏi/tham chiếu.
2. Tự động: normalized EM, whitespace Token-F1, ROUGE-L, lỗi format, độ dài,
   latency. Không gọi EM một đáp án là official VQA consensus accuracy.
3. Chấm 200 ảnh test khác nhau, một câu/ảnh, cùng mẫu mọi model. Hai người
   nếu có; ẩn model, xáo thứ tự. Đáp án đúng/mơ hồ, giải thích hỗ trợ đáp án
   0/1/2, chi tiết không được ảnh hỗ trợ có/không/không rõ. Thử rubric trước
   bằng 30 ảnh validation, không thử rồi sửa rubric trên test.
4. Báo toàn mẫu, nhóm đáp án đúng với mẫu số riêng, tỷ lệ đáp án đúng kèm
   giải thích đạt 2. Chênh lệch M3–M2 là so sánh chính; paired bootstrap theo
   ảnh 2.000 lần, CI 95%. Agreement dùng nhãn độc lập trước phân xử; một
   người thì không báo kappa giả. CI tập test không thay nhiều seed train.
5. Báo lỗi object/attribute/relation-action/giải thích lặp đáp án; xem độ dài
   để phát hiện ưu tiên văn phong. Bảng kết quả không chỉ có các ví dụ đẹp.

Đầu ra: bảng điểm bốn model, CI, nhãn người/anonymization map, phân tích lỗi
và 10–20 ví dụ thành công/thất bại được chọn có quy tắc. Không đổi cấu hình
sau test rồi gọi kết quả mới là test độc lập. Có lỗi evaluator thật thì sửa,
version lại và tính lại cho **tất cả** model, ghi rõ nguyên nhân.

### P5 — Demo, báo cáo, bàn giao và ôn bảo vệ

- Một demo Gradio đủ dùng: tải ảnh, câu hỏi tiếng Việt, đáp án + giải thích,
  chọn M1/M3 hoặc xem so sánh cố định; model load một lần, có xử lý lỗi ảnh.
  Demo inference, không train; ví dụ cached phải ghi cached. Không xây login,
  database/web backend lớn vì không phục vụ câu hỏi nghiên cứu.
- Báo cáo: (1) bài toán/đóng góp; (2) công trình liên quan; (3) dữ liệu và
  phương pháp; (4) thiết lập/kết quả/đối chứng; (5) phân tích/giới hạn/kết luận.
  Bảng và hình lấy từ artifact, không điền số thủ công hoặc làm tròn có lợi.
- Bàn giao source release, environment lock, config/data hashes, adapters,
  predictions/metrics, nhãn review, hướng dẫn tái lập. Xin phép/phù hợp license
  trước khi công khai dữ liệu; không upload credentials.
- Ôn giải thích được: vì sao tách ảnh, mask assistant, SFT vs DPO, reference
  M1, M2 là đối chứng gì, vì sao ROUGE chưa đo bám ảnh, và giới hạn của một seed.
  Không lấy việc dùng trợ lý lập trình làm bằng chứng tự hiểu kết quả.

## 4. Lệnh đã có và thứ tự dùng

Các lệnh cloud dưới chạy từ repository đã giải nén đúng release, Drive đã
mount và bundle ảnh đã xác minh ở `/content/vqa_work/vivqax_image_disjoint_v1`.
Đây không phải lệnh bootstrap Drive. Dùng đường dẫn trong environment YAML,
không sửa `src/` thành đường dẫn Drive.

Điểm vào Colab hiện hành là `notebooks/03_portable_runner.ipynb`: ô 1–4
chuẩn bị/quét token, ô 5 định nghĩa runner, ô 6 chạy M1 thật tới bước 300
nếu P0 đạt. Không Run all lại khi đã có M1; dùng checkpoint thật để resume.
Launcher cập nhật riêng nhưng trainer vẫn ghim source release M1 đã xác minh.

```bash
# P0: quét toàn train một lần trên CPU/cloud, không tải trọng số model.
python scripts/profile_sft_tokens.py --annotations /content/vqa_work/vivqax_image_disjoint_v1/metadata/all.jsonl --image-root /content/vqa_work/vivqax_image_disjoint_v1 --config configs/training/sft_readiness_full.yaml --validation-count 256 --output /content/drive/MyDrive/Vietnamese_VQA_NLE/runs/token_full_M1_v1.json

# P1: phiên đầu của M1 chính thức. Runner tự làm preflight/checksum.
python scripts/run_pipeline.py sft-full --environment configs/environments/colab_full.example.yaml --stop-after-step 300

# P1: phiên tiếp theo; thay đường dẫn bằng checkpoint thật từ session_result.json.
python scripts/run_pipeline.py sft-full --environment configs/environments/colab_full.example.yaml --resume-from-checkpoint <CHECKPOINT_M1_THAT> --stop-after-step 600

# Phiên cuối: bỏ --stop-after-step, giữ nguyên config và source M1.
python scripts/run_pipeline.py sft-full --environment configs/environments/colab_full.example.yaml --resume-from-checkpoint <CHECKPOINT_M1_THAT>
```

Các dấu `<...>` là chỗ phải thay bằng artifact thật, không chạy nguyên xi.
Nếu file token output đã tồn tại, kiểm hash/coverage để dùng lại; nếu cần
quét phiên bản mới thì đặt tên mới. `status=passed` của profiler chỉ nghĩa đo
xong: vẫn phải kiểm `sample_size=29459`, `validation_sample_size=256`,
`summary.samples=29715` và `summary.over_max_length=0`. Đối chiếu danh sách
256 ID với data plan của M1. Nó cần Torch/Transformers/processor nhưng
không cần CUDA hoặc tải trọng số 3B.

Lệnh suy luận validation chạy được hiện nay (M0, sau đó M1):

```bash
python scripts/run_m0.py --annotations /content/vqa_work/vivqax_image_disjoint_v1/metadata/all.jsonl --image-root /content/vqa_work/vivqax_image_disjoint_v1 --output <OUTPUT_M0_MOI.jsonl> --split validation --limit 0 --seed 42 --load-in-4bit --min-pixels 200704 --max-pixels 200704 --max-new-tokens 128
python scripts/run_m0.py --annotations /content/vqa_work/vivqax_image_disjoint_v1/metadata/all.jsonl --image-root /content/vqa_work/vivqax_image_disjoint_v1 --output <OUTPUT_M1_MOI.jsonl> --split validation --limit 0 --seed 42 --load-in-4bit --min-pixels 200704 --max-pixels 200704 --max-new-tokens 128 --adapter <ADAPTER_TU_TRAINING_RESULT> --experiment-id M1_SFT
python scripts/evaluate.py <PREDICTION.jsonl> --split validation --output <METRICS_MOI.json>
python scripts/compare_runs.py <M0.jsonl> <M1.jsonl> --expected-validation-annotations /content/vqa_work/vivqax_image_disjoint_v1/metadata/validation.jsonl --expected-validation-sha256 <HASH_VALIDATION_DA_GHIM> --output <COMPARISON_MOI.json>
```

`--limit 0` mới là toàn tập; mặc định script chỉ 20 câu. Export hiện chặn ghi
đè và chưa resume: nếu bị ngắt, giữ file partial, chưa tính điểm chính thức.
Hoàn thành cơ chế resume/shard P3 trước loạt test dài; nếu chạy lại dùng output
mới. M2/DPO/human-eval chưa có CLI hoàn chỉnh: phải triển khai ở P2/P3, không
đưa các lệnh tưởng tượng vào notebook. Không đưa đường dẫn adapter/checkpoint
vào báo cáo như thể chúng đã tồn tại trước khi run kết thúc.

## 5. Quy tắc test và nhịp làm việc từ nay

| Tình huống | Kiểm tra cần làm | Không làm |
|---|---|---|
| Chỉ sửa kế hoạch/comment | Kiểm nội dung, link/CLI và lint theo gate phát hành | Không chạy GPU, không audit lại 28.180 ảnh |
| Sửa dữ liệu, parser, metric | Unit/regression đúng phần; full gate trước release; full-data nếu thay input pipeline | Không restart train chỉ để xem một điểm smoke |
| Runtime mới/resume M1 | Hash source/dataset, dependency lock, checkpoint đầy đủ; runner kiểm tự động | Không chạy lại notebook 01–08 |
| Trainer/collator/LoRA mới | Unit trước, một integration GPU ngắn có lý do | Không thêm nhiều pilot nối tiếp khi cùng một đường đã qua |
| Train chính đang chạy | Đọc log/checkpoint mốc save; làm việc local song song | Không sửa runtime, không reload model để kiểm liên tục |
| Kết quả thấp nhưng pipeline hợp lệ | Phân tích validation và giữ kết quả | Không gọi đó là bug và train lại vô hạn |

Mỗi mốc P có một release khi cần đưa code lên cloud, không đóng ZIP sau mỗi
comment. Trước phát hành chạy `scripts/check_project.py` bằng `.venv` local;
full-data chỉ khi sửa dữ liệu/pipeline đầu vào. Không lặp audit thủ công rồi
preflight thủ công ngay trước lệnh train vốn đã kiểm lại checksum.
Harness phục vụ fail-fast, provenance và tái lập; không phải một nhánh nghiên
cứu cần tiếp tục mở rộng. Dùng CLI/cloud launcher có sẵn; không tạo notebook
09/10 chỉ để gọi một script. Giữ notebook cũ làm bằng chứng, không xóa.

## 6. Thời lượng, nguồn lực và đường găng

Mục tiêu điều phối: **10–15 ngày làm việc** nếu GPU và người duyệt sẵn sàng,
không phải cam kết ngày xong trên Colab miễn phí. P0 ngày 1; P1 ngày 2–4;
P2 và code P3 làm song song ngày 2–7; train P3 ngày 7–9; P4 ngày 9–12;
P5 và hoàn thiện báo cáo ngày 12–15. Viết báo cáo từ ngày 2, không chờ train xong.
Nếu ngày thực tế trượt vì quota, ghi nguyên nhân và cập nhật lịch, không tăng
thêm hạng mục. Không tự thuê GPU ngoài ngân sách đã được người dùng duyệt.

| Công việc | Cách dự toán/điều kiện |
|---|---|
| M1 | Ngoại suy pilot T4 cũ 36,8–44,3 giờ/epoch; save/eval 100 bước có thêm overhead, đo lại từ checkpoint đầu của run thật |
| Suy luận validation + test | 4 × (1.407 + 1.968) = 13.500 lượt; giờ = số lượt × giây/lượt thực đo / 3.600, chưa tính ứng viên preference |
| Sinh cặp | Tối đa 2 ứng viên × 2.000 câu; dừng khi đạt mục tiêu sạch; ghi số lượt và GPU giờ thực tế |
| Duyệt 1.000 cặp | Nếu 30–90 giây/cặp thì khoảng 8–25 giờ công, chưa tính sửa nhãn; đo bằng 100 câu đầu |
| Chấm 200 ảnh × 4 model × 2 người | 1.600 lượt chấm; nếu 30–60 giây/lượt thì khoảng 13–27 giờ công, có thể chia hai người |
| M2/M3 | Đo giây/update ở integration đầu; dự toán theo số cặp/epoch, cộng reference precompute và save; không lấy tốc độ SFT làm tốc độ DPO |

Các con số duyệt nhãn là kịch bản dự toán, không phải benchmark đã đo. Đây là
phần trợ lý không được giả làm người chấm. Nếu chỉ có một người, giảm số người
và công bố giới hạn; không âm thầm thay human evaluation bằng một LLM judge.

Ưu tiên khi thiếu thời gian: bỏ seed phụ/backbone phụ/UI trang trí; giữ M2,
test độc lập và đánh giá bằng ảnh. Không tự cắt train thành smoke rồi gọi
full M1. Nếu buộc đổi cỡ train hoặc bỏ M3, phải version lại protocol và phạm
vi đề tài trước thí nghiệm, báo rõ với người dùng/giảng viên.

## 7. Cách học từ công trình đi trước

- [NLX-GPT](https://github.com/fawazsammani/nlxgpt): dùng pipeline train–infer–eval
  và lưu prediction; phân biệt điểm giải thích trên toàn mẫu với nhóm đáp án
  đúng. Không sao chép kiến trúc cũ chỉ để thêm số model.
- [e-ViL](https://arxiv.org/abs/2105.03761): kết hợp đánh giá tự động và người
  chấm; kế hoạch 200 ảnh ở đây là bản thích nghi nhỏ hơn, không nhận là tái lập
  nguyên protocol của bài báo.
- [VINLE-GRPO](https://github.com/T-Sunm/VINLE-GRPO): tham khảo cách tách
  config/train/inference/evaluation trong cùng bài toán tiếng Việt; không kéo
  GRPO/reward vào phạm vi. Repo tự ghi công trình ICISN 2026, chưa chạy tái lập.
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer): dùng hỗ trợ VLM,
  reference log-prob cache và cảnh báo cắt token ảnh; phiên bản cần kiểm tương
  thích thực tế trước khi pin, tài liệu hiện hành không chứng minh runtime
  Transformers 4.57.6 đã chạy được DPO.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): GPU/hạn mức
  thay đổi, phiên free tối đa 12 giờ tùy điều kiện; không lập lịch như máy
  riêng chạy liên tục. Không coi quyền truy cập công cụ là đảm bảo có GPU.

Đây là tài liệu phương pháp/repo nghiên cứu, không phải rubric chấm đồ án
PTIT. Phần tham khảo rộng hơn ở `related_repositories.md`. Hướng có đầu tư
nằm ở câu hỏi rõ, dữ liệu kiểm duyệt, đối chứng và kết quả tái lập, không ở
số thư viện hay notebook.

<details>
<summary>Kế hoạch và ghi chép trước 28/09 — chỉ để tra cứu lịch sử</summary>

## Kế hoạch nghiên cứu trước 28/09

## Kết luận rà soát ngày 27/09/2026

Giữ hướng **cải thiện chất lượng lời giải thích trong hỏi đáp hình ảnh tiếng
Việt**. Phạm vi đủ cho một đồ án thực nghiệm ngành Khoa học dữ liệu nếu hoàn
thành các đối chứng và đánh giá dưới đây. Đây là nhận định về thiết kế, không
phải xác nhận đáp ứng rubric riêng của hội đồng hay cam kết điểm.

Hiện phần hạ tầng đi trước phần nghiên cứu: đã có dữ liệu đầy đủ, trainer SFT,
kiểm tra đầu vào và phục hồi checkpoint; chưa có M1 đầy đủ, dữ liệu ưu tiên
đã duyệt, trainer M2/M3, đánh giá giải thích bằng người hay demo. Không dùng
số notebook, số test hoặc một run smoke để chứng minh chất lượng đồ án.

### Câu hỏi và đóng góp cần hoàn thành

**Với cùng mô hình đã SFT và cùng nguồn câu trả lời tốt, học từ cặp lời giải
thích tốt–sai có giảm chi tiết không được ảnh hỗ trợ hơn SFT tiếp hay không,
và có làm giảm độ chính xác đáp án không?**

- Phần kế thừa: Qwen2.5-VL, ViVQA-X, SFT, LoRA và DPO. Không gọi các thành
  phần này là thuật toán mới của đồ án; không cần tự viết Transformer từ đầu.
- Phần tự thực hiện: kiểm tra/chuẩn hóa dữ liệu, bộ cặp ưu tiên tiếng Việt có
  lý do lựa chọn và nguồn gốc; thí nghiệm M0–M3 kiểm soát các yếu tố; đánh giá
  lỗi thị giác và phân tích trường hợp phương pháp không hiệu quả.
- Khác biệt dự kiến nằm ở thiết kế dữ liệu và bằng chứng thực nghiệm, chưa có
  cơ sở tuyên bố mới trên toàn thế giới. Cần trích dẫn cả ViVQA-X và hướng rất
  gần VINLE-GRPO; thay GRPO bằng DPO hoặc đổi backbone tự nó chưa đủ.
- Chỉ đo mức phù hợp của lời giải thích với ảnh/câu hỏi/đáp án. Không gọi câu
  sinh ra là bằng chứng về quá trình suy luận nội bộ hoặc tính trung thực nhân
  quả của mô hình. Không tuyên bố giúp người dùng tin đúng hơn nếu chưa nghiên
  cứu người dùng.

### Phạm vi cố định

| Hạng mục | Quyết định |
|---|---|
| Bài toán | Một ảnh tự nhiên + câu hỏi tiếng Việt → đáp án ngắn + lời giải thích |
| Dữ liệu | ViVQA-X, split dẫn xuất `image_disjoint_v1`; không thêm OCR/RAG/video |
| Backbone | Qwen2.5-VL-3B-Instruct đã ghim revision; LoRA là cách train |
| Đối chứng | M0 gốc, M1 SFT, M2 SFT chosen-only, M3 DPO; M2/M3 cùng M1 |
| Thước đo chính | Chất lượng giải thích do người chấm trên ảnh; EM đáp án đi kèm |
| Thước đo phụ | Token-F1, ROUGE-L, lỗi định dạng, độ dài và tốc độ |
| Không mở rộng lúc này | GRPO, KD, nhiều backbone, bộ verifier, thêm notebook pilot |

### Thứ tự thực hiện tiếp

1. **Chốt điều kiện chạy M1.** Full train là 29.459 câu. Pilot T4 cho ước
   lượng thô 36,8–44,3 giờ/epoch, chưa gồm toàn bộ suy luận, tạo cặp và M2/M3.
   Chọn runtime có khả năng resume, lịch checkpoint và ngân sách giờ trước
   khi chạy. Lưu bộ phiên bản thư viện của runtime GPU đã kiểm thử để cài lại:
   `pyproject.toml` mới ghim Transformers, các thư viện train khác còn là cận
   dưới; resume yêu cầu phiên bản khớp, nên cài lại tùy ý có thể bị chặn.
   Nếu chỉ đủ train một tập con thì tạo manifest cố định, cập nhật
   cấu hình/protocol và ghi đúng tên; không tự giảm mẫu trong run full M1.
   Quét token toàn train bằng processor (không cần model/GPU) hoặc hoàn thành
   cách kiểm tra trước train; 512 mẫu đã đo không chứng minh toàn tập hợp lệ.
2. **Hoàn thành M1 và đánh giá validation.** Chọn checkpoint bằng loss trên
   256 validation cố định như config hiện tại. M0 và M1 cùng suy luận đủ
   1.407 câu validation, cùng prompt/độ phân giải/generation. Đọc 100 trường
   hợp validation để xác định lỗi; không sửa test từ các kết quả này.
3. **Thử quy trình gán nhãn cặp trên 100 câu train**, duyệt bằng ảnh, đo tỷ lệ
   cặp dùng được và thời gian. Sau đó chốt số cặp theo năng lực rà soát; mục
   tiêu ban đầu 1.000–2.000 cặp đã duyệt, không phải quota phải đạt bằng nhãn
   tự động kém. Quy tắc ở `preference_annotation.md`.
4. **Triển khai M2/M3 và test.** M2 dùng chosen; M3 dùng chosen/rejected từ
   đúng cùng manifest. M3 reference cố định tại M1. Kiểm tra loss/log-prob
   chỉ trên assistant, ảnh giống nhau hai nhánh, reference không cập nhật,
   dấu chosen/rejected đúng, adapter lưu/nạp được. Pilot kỹ thuật ngắn rồi
   chạy thí nghiệm chính, không lại mở một chuỗi notebook mới.
5. **Đánh giá và phân tích.** Dùng giao thức `experiment_protocol.md`, đóng
   băng config/rubric trước test. Không bỏ một mô hình vì điểm kém. Nếu DPO
   không hơn M2, báo cáo kết quả âm và tìm nguyên nhân thay vì đổi metric.
6. **Demo và báo cáo.** Demo dùng checkpoint thật, hiển thị cả đáp án/giải
   thích, giới hạn và ví dụ thất bại. Viết rõ nguồn kế thừa, việc cá nhân thực
   hiện, cách tái lập; người bảo vệ cần giải thích được masking, split, LoRA,
   SFT/DPO và ý nghĩa từng đối chứng, dù dùng trợ lý lập trình.

### Điều kiện gọi là hoàn thành phần nghiên cứu

Có checkpoint và prediction đầy đủ của các nhánh đã đăng ký, bộ cặp đã kiểm
duyệt, bảng điểm kèm cỡ mẫu/độ bất định, đánh giá giải thích bằng ảnh và phân
tích lỗi. DPO không cần thắng mọi metric, nhưng không được thiếu đối chứng
M2 hoặc chỉ trình diễn vài ảnh đẹp. Nếu phải bỏ M3 vì tài nguyên, phải sửa
câu hỏi nghiên cứu/phạm vi với giảng viên; không gọi demo SFT là đã hoàn thành
kế hoạch so sánh preference optimization.

Nguồn đối chiếu và giới hạn so sánh được ghi tại
[related_repositories.md](related_repositories.md). Các phần ghi ngày
19–24/09 bên dưới được giữ làm lịch sử; giao thức hiện hành là mục này cùng
`experiment_protocol.md`.

---

Cập nhật kế hoạch dữ liệu: 24/09/2026. **Trạng thái hiện hành** nằm ở
[TIEN_DO.md](../TIEN_DO.md); các bảng trạng thái ngày 19/09 phía dưới là
nhật ký lịch sử, không phải trạng thái chạy hiện tại. M0/M1 smoke và paired
smoke đã hoàn tất; dữ liệu ảnh đầy đủ đã xác minh local và trên Colab,
audit/preflight full-data đã qua; optimizer pilot M1 20 bước và nạp lại adapter
đã qua trên T4, nhưng full SFT/DPO chưa chạy. Thời gian pilot khiến khả năng
chạy đủ một epoch trên Colab free cần được tính lại trước khi bắt đầu. Xem
[giao thức dữ liệu](data_protocol.md) trước khi mở rộng thực nghiệm.
Tài liệu này giữ thiết kế nghiên cứu và các mốc cũ. Kết quả mới, lệnh kiểm tra
và run tương ứng được ghi ở `TIEN_DO.md`; có code chưa đồng nghĩa đã chạy GPU.

## 1. Mục tiêu đồ án

**Nghiên cứu cải thiện chất lượng lời giải thích trong hỏi đáp hình ảnh tiếng Việt.**

Đầu vào: một ảnh và một câu hỏi tiếng Việt.
Đầu ra: đáp án ngắn cùng lời giải thích dựa trên bằng chứng trong ảnh.

Câu hỏi nghiên cứu: DPO có cải thiện độ phù hợp với ảnh của lời giải thích so với
SFT tiếp trên cùng nguồn dữ liệu tốt hay không? Độ chính xác đáp án thay đổi thế nào?
Đây là giả thuyết cần kiểm chứng, chưa phải kết quả đã đạt được.

Backbone hiện chọn Qwen2.5-VL-3B-Instruct để có mốc làm việc cố định. LoRA/4-bit
là cách triển khai huấn luyện trên GPU cloud; bản thân chúng không phải đóng góp
mới của đồ án. Không đổi model chỉ để làm đề tài có vẻ khác.

## 2. Mốc khởi đầu ngày 19/09

Ngày 19/09 đã chạy thử M0 và đang chuẩn bị SFT. Bảng dưới đây là ảnh chụp
tiến độ lúc đó, **không phải trạng thái hiện tại**; xem `TIEN_DO.md`.

| Hạng mục | Trạng thái có bằng chứng | Giới hạn cần nhớ |
|---|---|---|
| Dữ liệu ViVQA-X | Đã có pipeline chuẩn hóa và bộ smoke 500 ảnh | Chưa phải toàn bộ ảnh để train đầy đủ |
| M0: mô hình gốc | Theo kết quả lần trước: 20 validation, EM 0,40; Token-F1 0,4333; đúng định dạng 20/20 | Chỉ kiểm tra nhỏ; không phải kết quả cuối hay điểm lời giải thích |
| Code SFT | Đã thay script placeholder bằng trainer thật | Viết xong code không đồng nghĩa đã train thành công |
| Kiểm tra local | 26 test pass, Ruff pass; preflight 32 câu hỏi train có đủ ảnh | Chưa kiểm tra forward/backward thực tế trên GPU |
| Colab | Đã nối lại runtime GPU trong lần thao tác gần nhất | Runtime có thể ngắt; phải kiểm tra lại trước chạy |
| Notebook SFT | File local đã tạo; đã điền đường dẫn trong hộp tải file Chrome | Chưa xác nhận upload thành công, chưa có kết quả SFT trên Drive |
| M2, M3/DPO | Mới là kế hoạch; DPO chưa triển khai trainer | Chưa có preference data hoàn chỉnh hoặc kết quả |
| Demo, báo cáo cuối | Chưa nghiệm thu | Không coi các thư mục khung là sản phẩm hoàn chỉnh |

## 3. Thiết kế SFT smoke ban đầu (19/09)

Sau khi M0 chạy được, bước kế tiếp là kiểm tra máy cloud có huấn luyện được
đáp án kèm giải thích hay không. Vì vậy đã viết SFT smoke test: một lần train
rất ngắn để bắt lỗi trước khi dùng nhiều dữ liệu và giờ GPU.

- Chọn cố định 32 câu hỏi từ **train**, kiểm tra ảnh và sự tách biệt với held-out.
- Dự kiến chạy 8 lần cập nhật trọng số, batch 1, tích lũy gradient 2 lần.
  Đây không phải một epoch của cả 32 mẫu và không đủ để đánh giá chất lượng.
- Chỉ tính loss trên phần đáp án/giải thích; không bắt mô hình học lại câu hỏi
  hoặc token ảnh. Mỗi câu dùng lời giải thích đầu tiên trong smoke test.
- Dừng nếu loss không hữu hạn, dữ liệu không hợp lệ hoặc trọng số không đổi.
- Dự kiến lưu checkpoint, nạp lại adapter bằng tiến trình mới và chạy validation.

Tại mốc 19/09, các kiểm tra đã có trong code nhưng phần GPU **chưa chạy
thành công**; M0 không bị ghi đè. Kết quả chạy GPU về sau được ghi trong
`TIEN_DO.md` và `docs/sft_smoke.md`.

## 4. File nào dùng để làm gì?

Repo chính trên máy: `C:/VsCode/Image to text/Vietnamese-VQA-Explanation`.

| File/thư mục trong repo | Vai trò |
|---|---|
| `docs/research_plan.md` | Thiết kế nghiên cứu và mốc khởi đầu; xem `TIEN_DO.md` để biết tiến độ mới nhất |
| `docs/sft_smoke.md` | Hướng dẫn kỹ thuật cho lần chạy thử SFT hiện tại |
| `src/vqa_explanation/training/sft.py` | Trainer: tải model, gắn LoRA, train, lưu và kiểm tra kết quả |
| `src/vqa_explanation/training/sft_data.py` | Chọn dữ liệu train, tạo đầu vào ảnh–văn bản và mask loss |
| `scripts/train_sft.py` | Lệnh khởi động trainer |
| `configs/training/sft_smoke.yaml` | Thông số lần chạy 8 bước |
| `notebooks/archive/02_sft_smoke.ipynb` | Điều phối chạy trên Colab: Drive → chuẩn bị → train → nạp lại → đánh giá |
| `scripts/run_m0.py` | Suy luận; vừa bổ sung khả năng nạp adapter và chặn ghi đè output |
| `src/vqa_explanation/inference/qwen25_vl.py` | Tải model/adapter; chọn FP16 cho GPU không hỗ trợ BF16 native |
| `tests/unit/test_sft_data.py` | Kiểm tra mask loss, tách train/held-out và chọn mẫu |
| `tests/unit/test_sft_notebook.py` | Kiểm tra notebook hợp lệ và code đính kèm khớp code local |

Notebook 02 từng đính kèm source overlay vì archive trên Drive khi đó đã cũ.
Overlay này được giữ để tái lập run lịch sử; các lượt chạy mới dùng source
release có checksum, tránh lệch phiên bản giữa notebook và repo.

## 5. Nơi lưu tại mốc 19/09

| Nơi lưu | Nội dung | Trạng thái |
|---|---|---|
| Repo local nêu trên | Code, cấu hình, test, notebook, kế hoạch | Các chỉnh sửa SFT đã lưu local; chưa commit/push |
| Drive `Vietnamese_VQA_NLE/datasets/` | Annotation, danh sách mẫu, archive 500 ảnh | Đã chuẩn bị từ giai đoạn trước |
| Drive `predictions/M0_smoke.jsonl` và `metrics/M0_smoke_metrics.json` | Đầu ra M0 và điểm đánh giá | Đã lưu theo lần chạy trước |
| Drive `runs/M1_smoke_<thời gian>/` | Code dùng cho run, checkpoint/adapter, log và kết quả SFT | **Đường dẫn dự kiến**, chưa xác nhận run SFT nào được tạo |
| Colab `/content/M1_smoke_<thời gian>/` | Code và ảnh giải nén để chạy nhanh | **Dữ liệu tạm dự kiến**, mất khi runtime bị thu hồi |

Checkpoint là trạng thái huấn luyện để phục hồi; adapter là phần trọng số LoRA
đã học; log ghi diễn biến loss; predictions là câu trả lời sinh ra; metrics là
điểm tính từ predictions. Các file có vai trò riêng để kiểm chứng và tái lập.
Tên run có thời gian giúp lần thử sau không ghi đè lần trước.

## 6. Lộ trình nghiên cứu và điều kiện hoàn thành

### Bước A — SFT smoke (mốc lịch sử, đã qua)

1. Xác nhận notebook SFT đã mở trên Colab, đúng tài khoản Drive và có GPU.
2. Mount Drive, kiểm tra archive và giải nén 500 ảnh vào ổ tạm.
3. Chạy SFT 8 steps, quan sát lỗi bộ nhớ, loss và cập nhật trọng số.
4. Xác nhận checkpoint/adapter thực sự tồn tại trên Drive.
5. Nạp lại adapter, sinh dự đoán cho 20 validation và lưu báo cáo.

**Đạt khi:** train kết thúc, loss hữu hạn, trọng số thay đổi, adapter nạp lại được
và có prediction/metrics. Nếu lỗi thì sửa và ghi lại nguyên nhân ở nhật ký.
Mục này ghi quy trình smoke ban đầu; trạng thái hiện tại và run đã nghiệm thu
được ghi trong `TIEN_DO.md`. Không dùng các bước smoke như kế hoạch M1 hiện hành.

### Bước B — Chuẩn bị thực nghiệm đầy đủ

- Đã chốt `image_disjoint_v1`, chuẩn bị đủ 28.180 ảnh và kiểm tra bundle trên
  Colab. Audit/preflight cloud qua; run và số liệu nằm trong `TIEN_DO.md`.
- Đã cố định source snapshot và model revision. Chưa coi đây là hoàn tất
  thiết kế huấn luyện: `sft_smoke.yaml` chỉ chọn 32 câu/8 bước; `sft.yaml`
  hiện được runner M1 riêng sử dụng cho full preflight và huấn luyện chính thức.
  Không lấy smoke trainer chạy trên 29.459 câu rồi gọi là M1 đầy đủ.
- Runner M1 riêng có preflight CPU, optimizer pilot 20 bước và đường chạy
  đầy đủ: validation loss định kỳ, lưu/chọn checkpoint và resume sang run mới.
  Pilot trên T4 đã đạt cả optimizer/checkpoint/nạp lại adapter; số liệu và run
  ở `TIEN_DO.md`. Nó **chưa phải M1 đầy đủ**. Ước lượng tuyến tính thô từ pilot
  cho một epoch khoảng 44,3 giờ, có thể cao vì pilot eval/save dày hơn cấu
  hình full. Trước full M1 cần tách thời gian train/eval, chốt tài nguyên và
  lịch checkpoint/resume; tiếp tục rà token toàn tập hoặc có chính sách xử lý
  mẫu quá ngưỡng rõ ràng. Mỗi thử nghiệm ghi config/source/data hash.
- Stage `token-profile` và notebook 06 đã chạy trên Colab CPU:
  `token-profile_20260923_072219_478275` passed, 512/29.459 câu train được
  lấy mẫu cố định; p99 tổng 398 token, lớn nhất 426, không mẫu nào vượt 1.536.
  Mẫu này chưa chứng minh giới hạn token toàn tập. Collator batch 1 không pad
  lên `max_length`, nên giảm ngưỡng mà không đổi dữ liệu không tự giảm VRAM.
  Stage này chưa thay thế forward/backward một batch hoặc full SFT.
- Stage `sft-batch-pilot` và notebook 07 đã chạy qua trên T4 với một câu train
  dài 426 token bằng cùng setup QLoRA của SFT smoke. Run
  `sft-batch-pilot_20260923_115904_954833` có `execution.json` và
  `batch_pilot.json` passed; peak reserved 4,48 GiB, loss 2,264 và gradient
  LoRA khác 0. Đây chỉ là forward/backward, không optimizer/checkpoint;
  thời gian 6,44 giây không được ngoại suy thẳng thành tổng giờ full SFT.
- Chạy M0/M1 với cùng tập đánh giá và thiết lập suy luận để so sánh công bằng.
  Không so trực tiếp M1 smoke với M0 cũ vì giới hạn độ phân giải đang khác nhau.

### Bước C — Train và đánh giá M1

SFT trên dữ liệu train đã kiểm tra; chọn checkpoint bằng validation. Đo riêng
độ chính xác đáp án và chất lượng lời giải thích. Loss giảm hoặc lời giải thích
trôi chảy chưa chứng minh lời giải thích đúng với ảnh.

### Bước D — Dữ liệu ưu tiên và thí nghiệm đối chứng

Xây dựng cặp lời giải thích tốt hơn/kém hơn cho cùng ảnh–câu hỏi; lưu nguồn gốc,
tiêu chí lựa chọn và trạng thái rà soát. Chỉ dùng train để xây dữ liệu huấn luyện.
Chốt rubric kiểm tra bằng chứng trong ảnh và hạn chế câu giải thích bịa đặt.

| Mã | Cách làm | Mục đích |
|---|---|---|
| M0 | Mô hình pretrained | Mốc trước thích nghi |
| M1 | SFT đáp án và giải thích | Baseline đã học dữ liệu tiếng Việt |
| M2 | M1 + SFT tiếp trên câu được chọn (chosen) | Kiểm tra lợi ích riêng của dữ liệu tốt |
| M3 | M1 + DPO trên cặp chosen/rejected | Kiểm tra lợi ích học theo cặp ưu tiên |

M2/M3 bắt đầu từ cùng checkpoint M1, cùng nguồn chosen; ghi rõ ngân sách train.
Không khẳng định DPO tốt hơn trước khi có kết quả.

### Bước E — Đánh giá cuối, demo và báo cáo

Đóng băng model/cấu hình/rubric rồi mới đánh giá test. Báo cáo điểm đáp án,
đánh giá lời giải thích trên tập mẫu cố định có kiểm tra thủ công, phân tích lỗi
và ví dụ cả thành công lẫn thất bại. Hoàn thiện demo ảnh–câu hỏi–đáp án–giải thích
và hướng dẫn tái lập; ghi rõ nguồn dữ liệu, thư viện, mô hình và phần tự triển khai.

## 7. Nhật ký ngắn

- 22/09, hoàn tất phần bị ngắt: exporter suy luận ghi protocol metadata từ model/
  processor đã nạp, hash ảnh, kiểm tra ID và đường dẫn, không ghi đè output.
  Kiểm thử mock export M0/M1 → comparison; toàn bộ 46 tests và Ruff pass.
  Chưa chạy inference GPU mới hoặc train mới; kết quả smoke cũ giữ nguyên.
  Notebook 02 đóng băng lịch sử; dùng snapshot source mới cho lần chạy tiếp theo.

- 22/09: upload và chạy `02_sft_smoke.ipynb` trên Tesla T4, run
  `M1_smoke_20260922_065719`. Train 8 steps passed, loss 1,906347,
  VRAM đỉnh khoảng 4,33 GiB; nạp lại adapter và 20 validation passed.
  Sau ngắt runtime đã mount lại Drive, xác nhận 20 ID duy nhất và file adapter.
  EM 0,35, Token-F1 0,403333, format 20/20. Không so trực tiếp M0 do khác
  độ phân giải; đây là nghiệm thu kỹ thuật, chưa phải M1 chính thức.
  Kết quả trong `Vietnamese_VQA_NLE/runs/M1_smoke_20260922_065719/`.
  Local chạy lại 26 tests và Ruff pass. Không sửa hay chạy dự án VQA cũ.

- 19/09, hoàn tất cell bị nghi chưa chạy: cập nhật ô 2 notebook M0 trực tiếp trên
  Colab và bản local, thêm log 4 bước, bỏ xóa source, kiểm tra từng ảnh trong
  manifest. Chạy thực tế thành công: Drive mounted, SHA-256 đúng, 500 ảnh hợp lệ,
  588 câu hỏi; file prediction trên Drive có 20 mẫu M0. Không chạy lại suy luận
  và chưa train SFT. Notebook đã được yêu cầu lưu cùng output mới.

- 19/09, kiểm tra lại theo ảnh người dùng: đọc trực tiếp output từng ô M0 trên
  Chrome và tính lại metric từ bản prediction tải về. Xác nhận đủ 20 ID validation,
  không lệch đáp án tham chiếu, EM 0,40 và F1 0,433333. Colab hiện ngắt kết nối;
  output cũ còn lưu không đồng nghĩa môi trường vẫn sẵn sàng. Phát hiện archive
  code cũ chưa chặn ghi đè prediction; cần đồng bộ source trước chạy lại.
  Chi tiết bằng chứng và các điểm code cần sửa đã ghi tại `TIEN_DO.md`.

- Trước 19/09: chuẩn hóa dữ liệu, chuẩn bị smoke 500 ảnh và hoàn thành M0 20 validation.
- 19/09: khôi phục ngữ cảnh sau gián đoạn; xác định SFT/DPO cũ chỉ là khung.
- 19/09: triển khai code SFT smoke, hỗ trợ nạp adapter để kiểm tra sau train;
  26 test local và Ruff pass; preflight 32 mẫu có đủ ảnh.
- 19/09: kết nối lại Colab GPU; thao tác tải notebook gặp lỗi điều khiển hộp chọn file.
  Chưa có checkpoint SFT hoặc kết quả GPU mới được xác nhận.
- 19/09: gom giải thích, bản đồ file, trạng thái và kế hoạch vào tài liệu này theo
  yêu cầu người dùng. Bản kế hoạch cập nhật hiện lưu local, chưa đồng bộ Drive.

Mỗi lần làm tiếp cần ghi: việc đã thực hiện → bằng chứng/kết quả → file lưu →
việc tiếp theo. Chỉ đánh dấu hoàn thành khi đã kiểm tra đầu ra thực tế.

</details>
