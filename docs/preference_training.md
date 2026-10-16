# Huấn luyện M2/M3

Lượt 186 cặp dưới đây đã chạy, được giữ nguyên để kiểm và tham khảo nội bộ.
Nghiên cứu chính mới dùng [kế hoạch](research_plan.md),
`configs/experiments/preference_study_v2.yaml` và hai config `*_full.yaml`.
Bộ duyệt lại `frozen_1000_reviewed_20261015` đã khóa 1.000 cặp; snapshot
trước đó chỉ giữ làm lịch sử. Mỗi nhánh vẫn phải qua kiểm GPU trước train.
Negative gồm 992 cặp có kiểm soát và 8 output M1 nguyên văn; baseline chỉ
đọc chữ đạt 62,55%, nên không bỏ bước kiểm đổi ảnh và chấm theo ảnh.
Trainer đọc số epoch từ config, số update là `ceil(N/8) × epochs`.
Không dùng launcher v1 cho run mới. Launcher mới kiểm config và review
trước khi nạp model. Validation chính cần `--study-config` trỏ vào study v2;
bỏ tùy chọn này chỉ để đọc lượt 186 cặp lịch sử.

Lượt chính dùng launch `source/preference_training_launch_v2_20261015.json`.
Setup trong mục 1.000 cặp của notebook 04 khôi phục ZIP bằng
`scripts/restore_preference_bundle.py`, kiểm hash và không ghi đè bộ cũ.
Chạy setup → preflight → M2 gate/train → M3 gate/train, không chạy toàn bộ
notebook hoặc nối tiếp từ adapter 186 cặp. Gate GPU chưa chạy thì chưa train.

## Kiểm bộ cặp và độ nhạy ảnh

Kiểm độ khó mà không sửa bundle cũ:

```powershell
python scripts/audit_preference_quality.py --pairs <FROZEN>/accepted.jsonl --output-dir <AUDIT_NEW>
```

Lệnh tạo baseline chỉ đọc văn bản và review chất lượng pending, không tạo
nhãn đã xác nhận. Khi có review thật, thêm `--quality-review <REVIEW_JSONL>`.
Study chính khóa hash review; thiếu trường/khác nội dung cặp bị từ chối.
Review v2 yêu cầu nhận xét riêng từng cặp về độ liên quan của chi tiết sai
và khả năng chọn chỉ bằng chữ. Không kế thừa các cờ xác nhận chung cho cả
lô. Chạy `review_preference_images.py sheet` với ứng viên và manifest M1
để chuẩn bị đầu vào có nguồn; sheet cũ không dùng để nhập nhãn mới.
Các kiểm tra này chỉ xác minh hồ sơ, không tự chấm được nội dung ảnh.

Pool tiếp theo dùng `select_preference_questions.py --exclude-selection
<POOL_CU>`; lặp tùy chọn để loại toàn bộ ảnh các pool trước. Bản đã chọn:
`data/preferences/question_pool_v2/batch01`, 2.000 câu/ảnh train mới.
Pool câu hỏi chưa phải cặp preference. Luồng hiện hành ưu tiên ứng viên M1
cũ; chỉ sinh thêm cho những ảnh chưa có ứng viên:

```text
python scripts/expand_preferences.py run --annotations <ALL_JSONL> --image-root <IMAGE_ROOT> --old-pairs <V1_FROZEN>/accepted.jsonl --candidates <SAVED_M1_CANDIDATES> --selection <NEW_POOL> --m1-adapter <M1_3683> --m1-selection <M1_SELECTION> --output-dir <NEW_RUN> --target 1000
```

Lệnh cần GPU; local chỉ kiểm source/dữ liệu. `--inspect` kiểm đầu vào mà
không nạp model. Chạy lại cùng lệnh để nối tiếp, không đổi thư mục output,
source hoặc đầu vào giữa chừng. `decisions.jsonl` giữ cả kết quả loại;
`results.html` chỉ hiển thị ảnh và cặp được giữ, không ghi tên người duyệt.
Bundle đủ số vẫn chưa tự khóa config train; cần kiểm audit chất lượng.
Luồng mới thử sửa một cụm sai trong output M1 trước và giữ rejected nguyên
văn nếu qua kiểm. Khi không có lỗi M1 dùng được, mới tạo đối lập có kiểm
soát. Output sai đáp án/định dạng không được đổi nhãn thành negative M1.
Các bản viết lại giữ nguồn synthetic. Source mới không resume vào journal
expansion cũ; journal cũ vẫn giữ để đối chiếu.

Phép đổi ảnh dùng CLI riêng, không sửa annotation canonical:

```powershell
python scripts/check_image_dependence.py prepare --environment configs/environments/local_full.yaml --bundle artifacts/image_dependence/validation_200_seed42
python scripts/check_image_dependence.py run --environment <GPU_ENV> --bundle <BUNDLE> --output-dir <RUN_NEW> --experiment-id M1_SFT --adapter <M1_ADAPTER>
```

Bundle đã cố định 200 ảnh validation; copy nguyên bundle sang cloud, không
prepare lại sau khi xem output. `run` xuất đúng ảnh và ảnh hoán vị, kiểm
identity rồi ghi score. Chạy cùng lệnh để resume; `score` chỉ đọc prediction
đã đủ. Chỉ số so với đáp án ảnh gốc là diagnostic, không phải kết quả VQA
chuẩn trên ảnh hoán vị. Seed 42 đã hoàn tất phép đo này trên T4;
chưa thấy lợi thế M3 so với M2.

M1-3683 đã được chọn và có đủ prediction validation. Bộ v1 đã khóa
186 cặp / 186 ảnh, gồm 78 rejected M1 nguyên văn và 108 negative có kiểm soát.
Đầu vào ở `data/preferences/vivqax_m1_3683_v1/frozen`. Dữ liệu duyệt bằng AI giữ actor rõ trong metadata, không ghi
thành nhãn người. Cách duyệt/export ở
[preference_annotation.md](preference_annotation.md).

Run 14/10 đã hoàn tất: M2/M3 mỗi nhánh 24 update, epoch 1,0 trên T4.
Hai gate được giữ riêng, run chính nạp M1 sạch. Adapter đã kiểm byte sau lưu;
reference M3 không đổi. Báo cáo train nằm trong
`reports/generated/preference_training_20261014/` của kho kết quả, ngoài Git.
M2 đã có đủ validation; M3 chưa được xác nhận hoàn tất. Chưa có test.

## Validation sau train

Lượt 1.000 cặp dùng cùng CLI, thêm study đã khóa:

```text
python scripts/run_study_validation.py --environment configs/environments/colab_full.example.yaml --study-config configs/experiments/preference_study_v2.yaml --run-group <RUN_1000_SEED42> --m1-adapter <M1_3683> --output-dir <VALIDATION_NEW> --inspect
```

Chỉ bỏ `--inspect` sau khi cả M2/M3 đủ ba epoch. CLI đối chiếu config,
review hash, canonical, source trainer, kế hoạch và Trainer state; chọn
checkpoint-375 cho cả hai nhánh. Nếu một nhánh chưa xong thì dừng, không
dùng checkpoint của lượt 186 cặp. Bốn nhánh được suy luận trong cùng môi
trường; test vẫn đóng. Bảng đa seed chỉ có sau khi đủ các seed đã chốt.

Nếu mất phiên ở lượt chính: mount Drive rồi chạy ba cell trong mục
`Validation — 1,000-pair study` theo thứ tự. Setup tự đọc launch đã ghim,
khôi phục riêng môi trường đánh giá và ảnh nếu VM mới. Không cần chạy setup
train, gate hoặc cell train. CLI kiểm manifest/ID trước khi nối tiếp, bỏ qua
nhánh đã đủ 1.407 câu và không ghi lại các câu đã lưu. Nếu môi trường hoặc
adapter khác bản gốc, dừng kiểm tra thay vì ghép hai lượt suy luận.

Nếu file chính bị mất dòng do Drive chưa đồng bộ nhưng còn recovery snapshot,
dùng `scripts/recover_validation.py` với cùng environment/study/run/M1, thêm
`--source-dir <VALIDATION_OLD> --output-dir <VALIDATION_RECOVERED> --inspect`.
Script kiểm receipt, hash, manifest và các prefix; chỉ chọn bản dài nhất
không mâu thuẫn. Bỏ `--inspect` để sao chép sang thư mục mới, giữ nguyên
bản gốc. Sau đó chạy CLI validation với output-dir mới. Không nối thẳng
vào snapshot, sửa ID, cắt file lỗi hoặc bỏ cổng kiểm môi trường.

Phần dưới mô tả cell/launcher của lượt 186 cặp lịch sử:

Ba cell cuối notebook gọi `scripts/run_study_validation.py`: setup source,
kiểm đầu vào, xuất dự đoán. CLI kiểm plan và hash checkpoint M1/M2/M3;
xuất đủ 1.407 câu cho M2 → M3 → M0 → M1 cùng môi trường và generation config.
M0/M1 được suy luận lại, không train lại. Bảng baseline cũ vẫn giữ riêng.
Các file JSONL có manifest và khóa ghi; mất phiên thì gọi lại cùng CLI để
nối tiếp. Không bấm cell train hoặc chạy tất cả để tiếp tục validation.

Nếu mất VM, chạy cell 1 → 2 → 3 để mount kho, khôi phục môi trường và ảnh;
chờ từng cell hoàn tất. Sau đó chạy cell 8 → 9 → 10 trong mục validation.
Không chạy lại cell 4–6. Output cũ không chứng minh kernel mới đã có biến;
kiểm execution count và log mới trước khi chuyển cell. CLI resume kiểm
JSONL/manifest đã lưu và bỏ qua ID hoàn tất, không train lại các adapter.

Nếu kho được mount qua shortcut làm đường dẫn adapter khác, dùng
`--relocate-from <RUN_CU>` với một `--output-dir` mới. CLI chỉ cho phép
đổi địa chỉ adapter sau khi hash trọng số, dữ liệu, source suy luận và
runtime khớp; kiểm từng prediction rồi tạo bản sao có biên bản chuyển kho.
Run gốc không đổi. Record launch có thể khai `relocate_from`; các lần
resume sau kiểm biên bản đã tạo rồi tiếp tục cùng thư mục mới.

Trong lượt xuất mới, CLI lưu thêm bản sao đóng tệp ở `recovery/` sau câu
25 và mỗi 100 câu. Mỗi bản có JSONL, manifest và checksum; không phải
checkpoint train. Chỉ tính số câu thực đọc được trong file, không suy số
đã lưu từ dòng tiến độ trên trình duyệt. Khi Drive mất file đang mở,
giữ nguyên bản lỗi và kiểm snapshot trước khi khôi phục.

Kết quả nằm ở `evaluation_runs/m0_m3_validation_20261014`. Chỉ sau khi đủ
bốn file mới so M2/M3, M1/M2, M1/M3 và M0/M1, kèm CI paired image-bootstrap.
EM/F1/ROUGE-L chưa thay phần chấm bám ảnh bởi người; BERTScore làm sau khi
đối chiếu đủ prediction. Không mở test để chọn cấu hình.

## Chuẩn bị đầu vào

- `accepted.jsonl`, `chosen_only.jsonl`, `manifest.json` từ cùng bộ đã freeze.
- Adapter M1-3683 và selection record đã ghim model/revision và hash.
- Sáu evidence của selection: kết quả epoch, Trainer state, loss selection,
  prediction validation, manifest prediction và annotation validation.
- Profile environment trỏ tới ảnh/annotation đầy đủ và nơi lưu run.

Đường dẫn evidence được tính từ selection record. Bản tải ở
`artifacts/colab_history/20261007/m1_selection_frozen_v2` giữ layout Drive gốc;
cần khôi phục đủ layout/evidence trước khi dùng làm đầu vào trainer trên máy khác.
Demo replay có YAML trỏ trực tiếp ba evidence cần dùng.

Chốt `expected_pairs=N` cho các config sẽ chạy. Hai nhánh dùng cùng ID,
ảnh, đáp án và chosen. Sau preflight CPU, mỗi trainer chạy gate GPU tối đa
10 update, rồi run chính nạp lại M1 sạch.

`scripts/finalize_preference_review.py` bàn giao bản duyệt bằng một lệnh:
audit/báo cáo trước, export/freeze chỉ khi có `--freeze` và N chính xác.
Đọc `handoff.md`, kiểm manifest và dùng thư mục `frozen` làm preferences.
Lệnh không cập nhật config, chọn N hoặc chạy GPU thay người thực hiện.

## Trainer và môi trường mới

Logic ở `training/preference.py`, input/collator ở `training/preference_data.py`.
CLI `scripts/train_preference.py` đọc config và chạy trainer.

Phần train của `04_train_preferences.ipynb` có 7 cell: kho lưu, source,
preflight, M2, M3, kết quả và mô tả ngắn. Mục validation thêm 4 cell,
tổng 11 cell; output train giữ nguyên. `scripts/launch_preference_training.py` chuyển
record khởi chạy thành tham số CLI trên; không chứa loss hoặc trainer mới.
Hai cell M2/M3 gọi cùng helper, tránh chép lại phần dựng lệnh.

Record `source/preference_training_launch_v1.json` giữ đường dẫn môi trường,
preferences, M1/selection, hai config và `run_group`. Bộ cuối đã đặt
`data_locked=true`, `expected_pairs=186`; hai config khớp N. Record dành
cho Drive trỏ `preference_local/vivqax_m1_3683_v1/frozen`. Local vẫn dùng đường dẫn
repo và profile local riêng, không mount Drive.
Launcher đối chiếu checksum của decision record rồi đối chiếu manifest
frozen với quyết định đó trước khi tải model. Thay bộ cặp dù cùng N cũng
bị từ chối. Record cũ không có hai trường pin giữ các kiểm metadata/gate cũ;
record chính hiện hành có đầy đủ hai trường.
Runtime mới khôi phục bundle ảnh vào ổ `/content` bằng verifier hiện có;
record ghim đường dẫn/hash bundle. Ảnh của bộ cặp tiếp tục được đối chiếu
trong preflight. Không đọc hàng nghìn JPEG trực tiếp từ Drive.

Trong notebook: `MODE='inspect'` chỉ kiểm metadata; `'gate'` chạy 5 update;
`'train'` cần kết quả gate cùng nhánh. Run group không đổi khi chạy lại
cell cấu hình, nên bước train đọc đúng gate đã lưu. Chọn nhóm run mới khi
đổi source/dữ liệu, không ghi đè output cũ. Resume truyền checkpoint qua
`RESUME`, đặt `ATTEMPT` của nhánh đó thành tên mới (ví dụ `resume25`).
Output mới nằm cùng nhóm run; gate gốc vẫn được dùng. Không lặp setup M1.

M2 chính dùng `configs/training/sft_chosen.yaml`; M3 dùng
`configs/training/dpo.yaml`, cùng LR 5e-6. Bản M2 2e-5 giữ làm thiết kế cũ,
chưa train. Protocol sửa ngày 12/10 trước run để tránh nhiễu do khác LR.
CLI mặc định đọc `study_budget_v2.yaml`, kiểm hai nhánh và lưu hash trong
run. Bản cũ bị từ chối nếu khai là nhánh chính. Nếu thay thiết kế, cần file
study mới qua `--study-config`, không đổi protocol sau khi thấy test.

Kiểm cấu hình, không cần GPU/dữ liệu:

```powershell
python scripts/train_preference.py --config configs/training/sft_chosen.yaml --check-comparison
```

Hai config đã chốt `expected_pairs=186`; CLI từ chối nếu khác N của study.
Lệnh chạy M2 dùng config matched-LR, M3 dùng dpo; cả hai cần gate riêng.

- M2 nạp **trọng số adapter M1**, không khởi tạo LoRA ngẫu nhiên mới.
- M3 nạp hai bản adapter M1 trên cùng backbone 4-bit: policy trainable,
  reference frozen. Reference không phải model gốc tắt adapter.
- Cả hai tắt dropout, cùng chosen và thứ tự dữ liệu, seed 42, một epoch,
  batch 1 × tích lũy 8. Với 186 cặp: 24 optimizer update mỗi nhánh.
- DPO sigmoid dùng tổng log-prob của completion, mask prompt/ảnh; dùng ảnh ở
  chosen, rejected và reference. Không có reward model, rollout RL hoặc cache ref.
- Giới hạn token là cổng từ chối, không cắt chuỗi. Quét collator thật một lần
  trước train. DPO có `max_length=null`, guard cả hai phía ở 1536 token.
- Dtype ở config train có hiệu lực: auto = FP16 trên T4, BF16 chỉ khi native.
  Dtype legacy trong model YAML không điều khiển hai stage mới.

Trong **runtime GPU mới**, cell 2 tạo môi trường riêng bằng
`scripts/setup_preference_environment.py`. Môi trường ở `/content/vqa_envs/<release>`;
không lấy thư viện hệ thống của Colab. CLI kiểm ảnh, M2/M3 đều dùng Python
của môi trường này. Kernel chỉ mount Drive và điều phối. Log subprocess được
chuyển tiếp ra notebook, kể cả lỗi cài đặt. Process con bỏ Python path/home
kế thừa và không nạp user-site để giữ môi trường riêng. Ghi phiên bản và GPU trong run.
Cài `requirements-cloud-preference.txt`, chạy `pip check` trong môi trường đó.
File này ghim các thư viện chính; không phải
lock toàn bộ phụ thuộc bắc cầu hoặc bằng chứng đã chạy GPU. CUDA wheel/runtime
còn phải xác nhận trong cổng GPU. Torch/Torchvision ghim bản `+cu128`;
đã kiểm wheel Python 3.13/Linux trong [index PyTorch chính thức](https://download.pytorch.org/whl/cu128/torch/).
Không thay môi trường M1.

Pip điều khiển venv không có pip qua `--python`, theo
[tài liệu pip](https://pip.pypa.io/en/stable/topics/python-option/).
Không bỏ qua dependency check vì các gói cài sẵn của Colab bị lệch phiên bản.

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

M2/M3 đã train xong; chưa tạo nhãn người. Trạng thái đồng bộ đọc từ TIEN_DO.
Công cụ chấm test theo [hướng dẫn](human_evaluation.md); demo còn cần hoàn thiện.
