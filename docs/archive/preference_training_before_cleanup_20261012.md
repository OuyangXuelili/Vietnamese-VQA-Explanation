# Thiết lập giai đoạn M2/M3

M1 đã hoàn tất bằng release 28/09; giữ nguyên source và run lịch sử. Notebook
03 tiếp tục điều phối giai đoạn mới bằng release riêng. Hướng nghiên cứu không đổi: so SFT
chosen-only với DPO từ cùng M1, cùng dữ liệu đã duyệt.

## Local dùng để làm gì

Repo ở `C:\VsCode\Image to text\Vietnamese-VQA-Explanation`.
Local có annotation và 28.180 ảnh; không phải tải toàn bộ COCO lần nữa.
Profile `configs/environments/local_full.yaml` trỏ dữ liệu và output local.

Môi trường `.venv` đã có Torch CPU, Transformers 4.57.6, Accelerate 1.14.0
và PEFT 0.20.0. Không có CUDA/bitsandbytes hoặc trọng số Qwen. Accelerate/PEFT
dùng cho kiểm trainer nhỏ trên CPU, không biến máy này thành máy train Qwen.

Trên máy mới, cài `.[dev]` cho audit/metrics. Nếu cần các test trainer, cài
Torch/Torchvision bản CPU trước rồi `pip install -e ".[dev,verification]"`.
Không gọi `pip install -r requirements-cloud-preference.txt` trên local yếu.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/check_project.py --full-data
```

Một lệnh kiểm trước mốc lớn; không chạy audit/smoke lặp lại khi đầu vào không đổi.

## Duyệt dữ liệu

Trang pilot nên dùng:
`artifacts/preference_editor_m1_3683_2000_20261008/pilot_review_compact.html`.
Trang gồm nhận xét và đề xuất AI; kiểm ảnh trước khi xác nhận.
`pilot_review_current.html` giữ lượt cũ ẩn hỗ trợ. `index.html` vẫn là trang đủ
2.000 câu dùng sau khi chốt
rubric và kế hoạch mở rộng.
Nháp AI để trống người xác nhận. Đã rà 100 ảnh, kiểm lại 17 nháp; còn
11 đề xuất sửa tự nhiên và 4 lỗi có kiểm soát. Chi tiết trong
`artifacts/preference_ai_triage_m1_3683_pilot_20261011_qa/README.md`.
Người xác nhận thực tế có thể dùng mã ẩn danh ổn định thay tên thật;
nhận xét AI không dùng mã của người.
Đủ 2.000 pending từ M1-3683; chưa có accepted. Nguồn `review_2000.zip` ở
`artifacts/colab_history/20261008/preference_candidates_completed`, SHA-256:
`365f080096f964a4a1d670faa7192a62697f45fb92f0480cdd555c7319b8af45`.
`calibration_ai_draft.md` ghi kiểm ảnh 30 câu đầu, không phải file nhãn hoặc
input trainer. Các trang checkpoint-800 cũ giữ nguyên làm lịch sử.

1. Mở trang, chọn câu, nhìn ảnh và kiểm cả đáp án M1 lẫn đáp án tham chiếu.
2. Ghi chosen/rejected, cụm sai, bằng chứng, loại lỗi, người duyệt và quyết định.
   Không tạo cặp nếu ảnh không đủ xác định đáp án hoặc hai giải thích đều đúng.
3. Trang tự lưu bản nháp theo đúng nguồn trong trình duyệt hiện tại. Vẫn bấm
   **Xuất JSONL** để có artifact bền vững, chuyển máy hoặc chạy validator.
   Có thể nhập lại file đã lưu để tiếp tục. Giao diện không tự xác nhận chất lượng nhãn.
   Trang mới khôi phục nháp v2/v3 cùng nguồn và hàng đợi trong cùng trình duyệt/origin.
   Nếu chuyển từ `file://` sang localhost, đổi hàng đợi hoặc có hai nháp khác nhau,
   xuất JSONL từ trang cũ rồi nhập vào trang hiện hành. Không xóa nháp cũ.
4. Chạy `export_preference_review.py`, sau đó `validate_preferences.py`
   để tạo bundle mới. Lệnh đầy đủ ở [hướng dẫn gán nhãn](preference_annotation.md).

100 ứng viên từ checkpoint-800 chỉ để hiệu chỉnh rubric. Bộ chính sinh từ
M1 được chọn, giữ pool train-only và version mới; không trộn nguồn mà không ghi.
Mục tiêu 1.000 accepted. Nếu số sạch khác, sửa `expected_pairs` ở **cả hai**
config và ghi quyết định trước train. Không tự lấy subset cho riêng DPO.

`accepted` nghĩa là có một cặp dùng được cho huấn luyện, không phải chỉ là
M1 trả lời đúng. Nếu M1 đã trả lời và giải thích đúng, có thể chọn `rejected`
với lý do không có cặp đối lập rõ. Câu không thể xác định từ ảnh chọn `uncertain`.
Câu sai đáp án không dùng cho cặp chỉ so sánh giải thích. Người duyệt vẫn ghi
mã xác nhận, thời gian và lý do cho các quyết định loại; giữ toàn bộ 2.000 dòng
khi xuất. Có thể dùng mã ẩn danh ổn định, không bắt buộc tên thật.

Sau khi xuất nhãn, thứ tự công việc là:

1. Audit file vừa xuất và báo cáo pilot 100 câu; xem tỷ lệ cặp dùng được.
2. Chốt số cặp và cách mở rộng trên pool train đã có, ghi trước khi train.
3. Export rồi validate/freeze: `accepted.jsonl`, `chosen_only.jsonl`, `manifest.json`.
4. Trên Colab, kiểm mỗi trainer tối đa 10 update với bundle thật; lưu kết quả gate.
5. M2 và M3 cùng bắt đầu lại từ M1-3683, dùng cùng ID/chosen và học một epoch.

Bước 1–3 chạy local; bước 4–5 dùng GPU. Nháp AI và report `in_progress` chưa
phải bundle train. Các lệnh export/freeze ở [hướng dẫn gán nhãn](preference_annotation.md);
lệnh trainer và cách truyền gate ở phần dưới.

## Chốt M1 trước khi train

Sau khi hoàn tất 3.683 bước, dùng `evaluate_m1_epoch.py` đo mốc cuối và chọn
theo loss validation. Checkpoint-3683 và đủ 1.407 validation đã được kiểm.
Ghi selection record bằng `scripts/freeze_m1_selection.py`, không dùng test.
Rà 100 ảnh validation cố định song song; 20–30 ảnh đầu để thử rubric. Đây là
chẩn đoán, không phải yêu cầu đủ 100 nhãn mới được chốt M1 hoặc train.

`m1_selection.json` là biên bản chốt đầu vào, khác file loss-selection của
`evaluate_m1_epoch.py`. Schema:

```json
{
  "schema_version": 2,
  "protocol": "validation_loss_full_validation_v2",
  "status": "selected",
  "selection_split": "validation",
  "annotations_sha256": "SHA256_CANONICAL_ALL",
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
  "adapter": {"files": {}, "sha256": "HASH_MANIFEST_ADAPTER"},
  "diagnostic_review_required": false,
  "evidence": {
    "epoch_result": {"path": "../run/training/training_result.json", "sha256": "..."},
    "trainer_state": {"path": "../run/training/trainer_state.json", "sha256": "..."},
    "loss_selection": {"path": "../epoch_eval/selection.json", "sha256": "..."},
    "validation_predictions": {"path": "../evaluation/m1.jsonl", "sha256": "..."},
    "validation_manifest": {"path": "../evaluation/m1.jsonl.manifest.json", "sha256": "..."},
    "validation_annotations": {"path": "../data/validation.jsonl", "sha256": "..."}
  }
}
```

Đây là mẫu schema, không phải artifact đã đạt. Đường dẫn evidence tương đối
so với biên bản; điền hash byte thật. Lệnh `train_preference.py --inspect-adapter
<M1_ADAPTER>` xuất nguyên object `adapter`. Human review mỗi dòng có
`question_id`, `review_status=reviewed`, `reviewer`, `answer_assessment`
(correct/incorrect/uncertain), `explanation_support` (0/1/2). Evidence `human_review`
là tùy chọn; nếu cung cấp phải gồm nhãn đã duyệt, không rỗng, một câu/ảnh,
không trùng ID và thuộc validation. Không ép số lượng 100.
Kiểm metadata không chứng minh người duyệt thật sự nhìn ảnh.

Biên bản được ghi mới sau kiểm evidence; chạy lại cùng byte thì chỉ đọc,
không thay quyết định cũ. Lượt sinh ứng viên dùng cả `--m1-selection` và
`--m1-selection-sha256` để không nhầm checkpoint-800 với M1 đã chốt.
Các bản selection tải vào `artifacts/colab_history` giữ đường dẫn evidence
tương đối của layout Drive gốc. Đã kiểm ngày 12/10: sáu đường dẫn trong bản
`m1_selection_frozen_v2/selection.json` không resolve ở vị trí local này.
Đây là bản lưu hồ sơ, không phải bundle local đầy đủ để truyền vào trainer.
Trên Colab dùng file selection trong layout gốc cùng các evidence thật;
nếu chuyển kho, khôi phục layout hoặc tạo biên bản mới sau kiểm đủ hash.
Không sửa record cũ hoặc bỏ kiểm evidence để qua preflight. Demo replay
đọc ba file evidence qua YAML riêng và kiểm hash, không cần layout train.
Pool 2.000 câu train giữ nguyên; sinh vào output mới của checkpoint-3683.
Không train M2/M3 khi chưa có accepted/chosen-only và kiểm chất lượng cặp.
Chạy `scripts/audit_preference_review.py` trên từng bản JSONL xuất từ editor
trước khi convert. Báo cáo audit có thể theo dõi bản dở, nhưng không thay
`export_preference_review.py` hoặc validator đóng băng.
Sau pilot, chạy `scripts/summarize_preference_pilot.py` để chỉ tính acceptance
rate và dải quy mô pool khi đủ 100 quyết định hợp lệ. Report đang dở không được
dùng để thay đổi rubric hoặc mở cổng train.
Projection chỉ dùng accepted có rejected nguyên văn M1, không cộng negative
manual/synthetic. Có thể export subset đã kiểm trong pool còn pending;
vẫn cần chốt N, validate/freeze và GPU gate trước train chính.

Trên Colab, chạy ô 1 → 9 → 10 của notebook 03. Ô 10 kiểm hash M1 và hai
file pool đã chốt trước khi cài thư viện hoặc giải nén ảnh. Thiếu package thì
cài đúng phiên bản suy luận đã ghim, dừng để restart Python rồi chạy lại
1 → 9 → 10. Bước này không cần ô 6 hoặc cache M0; các ô M1 lịch sử giữ nguyên.

## Trainer và môi trường mới

Logic ở `training/preference.py`, input/collator ở `training/preference_data.py`.
Một CLI `scripts/train_preference.py` đọc hai config; không tạo notebook mới.

Có thêm config phụ `configs/training/sft_chosen_matched_lr.yaml`: cùng M2,
chỉ đổi LR thành 5e-6 như DPO. Chưa chạy GPU. Khi chốt N khác 1.000, cập nhật
`expected_pairs` cho mọi config sẽ dùng, gồm config phụ nếu chọn chạy.
Biến thể này cần output riêng và gate cùng config; không dùng gate của M2
2e-5. Vẫn mang `experiment_id=M2_SFT_CHOSEN` vì cùng loại trainer; phân biệt
bằng đường dẫn run và hash config. Không thay adapter/prediction M2 chính.
Quyết định chạy theo ngân sách trước test, không chọn biến thể theo điểm test.

- M2 nạp **trọng số adapter M1**, không khởi tạo LoRA ngẫu nhiên mới.
- M3 nạp hai bản adapter M1 trên cùng backbone 4-bit: policy trainable,
  reference frozen. Reference không phải model gốc tắt adapter.
- Cả hai tắt dropout, cùng chosen và thứ tự dữ liệu, seed 42, một epoch,
  batch 1 × tích lũy 8. Với 1.000 cặp: 125 optimizer update mỗi nhánh.
- DPO sigmoid dùng tổng log-prob của completion, mask prompt/ảnh; dùng ảnh ở
  chosen, rejected và reference. Không có reward model, rollout RL hoặc cache ref.
- Giới hạn token là cổng từ chối, không cắt chuỗi. Quét collator thật một lần
  trước train. DPO có `max_length=null`, guard cả hai phía ở 1536 token.
- Dtype ở config train có hiệu lực: auto = FP16 trên T4, BF16 chỉ khi native.
  Dtype legacy trong model YAML không điều khiển hai stage mới.

Trong **runtime GPU mới**, cài `requirements-cloud-preference.txt`, chạy
`pip check`, ghi phiên bản và GPU. File này ghim các thư viện chính; không phải
lock toàn bộ phụ thuộc bắc cầu hoặc bằng chứng đã chạy GPU. CUDA wheel/runtime
còn phải xác nhận trong cổng GPU. Torch/Torchvision ghim bản `+cu128`;
đã kiểm wheel Python 3.13/Linux trong [index PyTorch chính thức](https://download.pytorch.org/whl/cu128/torch/).
Không thay môi trường M1.

Ví dụ sau khi có bundle và selection thật, tại repo:

```powershell
python scripts/train_preference.py --config configs/training/dpo.yaml --environment configs/environments/local_full.yaml --preferences <FROZEN_BUNDLE> --m1-adapter <M1_ADAPTER> --m1-selection <SELECTION_JSON> --preflight-only
```

Lệnh này chỉ kiểm metadata và ảnh; không tải model, không train. Trên cloud
dùng profile environment cloud riêng. Nếu thiếu accepted/selection, dừng;
không thay bằng sample JSONL để qua cổng.

Exporter lưu `original_candidate_origin` cho mọi cặp, kể cả bản người sửa hoặc
synthetic. Trước train, model ID và fingerprint adapter của nguồn này phải khớp
M1 được chọn. Nguồn `model` ở chosen/rejected cũng phải khớp. Bundle cũ thiếu
trường này cần export lại từ đúng candidate manifest; không tự điền hash M1.

Mỗi trainer mới chạy một GPU gate tối đa 10 update (`--pilot-steps 5`, output
mới), kiểm loss, policy update, reference giữ nguyên và byte weights đã lưu.
Run chính bỏ pilot-steps, thêm `--gpu-gate <GATE_RUN/training_result.json>`.
Khởi lại từ M1 sạch, không dùng adapter gate. Hash source/data/config/M1,
runtime và dtype phải khớp. DPO dự toán vượt 8 giờ thì xem lại ngân sách chung.

Run mới ghi data_plan, execution, Trainer log/state, checkpoint mỗi 25 update
và ở đích cuối, adapter policy và training_result. Giữ các checkpoint đã lưu.
Mỗi checkpoint có `resume_manifest.json` ghi hash adapter, optimizer, scheduler,
RNG, scaler FP16 và các file khác. Marker chỉ ghi sau khi Trainer lưu xong.

Khi Colab mất VM, dùng `--resume-from-checkpoint <CHECKPOINT>` và một
`--output-dir` mới. Nguồn, dữ liệu, M1, config, phiên bản thư viện, dtype,
loại GPU và ngân sách phải khớp. Checkpoint mất file hoặc khác hash bị từ chối
trước tải model. Trainer khôi phục trạng thái và bỏ qua dữ liệu đã học; reference
được nạp lại từ M1 cố định, policy nạp từ checkpoint. Resume không tăng epoch.

`--stop-after-step <N>` dừng chủ động và lưu ở N, giữ lịch LR/đích gốc;
trạng thái là `paused`, chưa phải kết quả chính. Chỉ nối checkpoint nhỏ hơn
đích gốc. Gate GPU bị dừng chủ động chưa được coi là gate passed.
Các ô train M1 cũ không dùng CLI này. Run cuối ghi số update của phiên hiện
tại riêng với global step; đọc loss theo `log_history` khi đã resume vì
`Trainer.train_loss` không phải trung bình đầy đủ của các phiên trước.
CPU tests và toy training không xác nhận Qwen 3B/QLoRA chạy vừa T4.

## Cơ sở thiết lập

[NLX-GPT](https://github.com/fawazsammani/nlxgpt) tách dữ liệu, train và đánh giá
VQA-NLE. Repo này dùng cùng nguyên tắc, không lấy architecture hay kết quả của họ.
[TRL 0.29.0 DPO source](https://github.com/huggingface/trl/blob/v0.29.0/trl/trainer/dpo_trainer.py)
là đối chiếu cho completion masking và loss; không dùng nguyên trainer vì có
cast QLoRA trainable sang BF16. Bản 0.28 được kiểm chưa chuyển `image_grid_thw`
trong đường forward DPO. Vì vậy dùng Transformers Trainer và loss chuẩn DPO
riêng, giữ lại collator Qwen đã kiểm. Đây là lựa chọn triển khai, không phải thuật toán mới.

Chưa train M2/M3, chưa tạo nhãn người. Trạng thái đồng bộ đọc từ TIEN_DO.
Công cụ chấm test theo [hướng dẫn](human_evaluation.md); demo còn cần hoàn thiện.
