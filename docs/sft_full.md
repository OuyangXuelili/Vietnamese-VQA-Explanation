# M1 trên bộ ViVQA-X đầy đủ

M1 đã train đủ 3.683 bước, một epoch trên 29.459 mẫu, ngày 07/10/2026.
Run `sft-full_20261007_023531_870900` có trạng thái `passed`, mã thoát 0;
checkpoint cuối và adapter được Trainer chọn đã kiểm lại trên CPU.
Trainer chọn checkpoint-3500, loss 1,0595598 trên 256 validation cố định.

Đã đo riêng checkpoint-3683 ngày 07/10: loss `1.05936598777771`, thấp hơn
rất nhẹ so với checkpoint-3500 (`1.0595598220825195`). Theo quy tắc đã
đặt trước, checkpoint-3683 được chọn. Đã đánh giá đủ 1.407 validation và
đóng băng selection trong
`artifacts/colab_history/20261007/m1_selection_frozen_v2/selection.json`.
EM 70,93%, Token-F1 75,11%, ROUGE-L 44,29%, BERTScore 82,47%.
Đây là điểm validation tự động, chưa phải chấm mức hỗ trợ theo ảnh.
Các phần pilot/resume bên dưới giữ để tái lập lịch sử, không phải danh sách
cần chạy lại. Vị trí lưu bằng chứng được ghi cùng từng mốc.

Hướng dẫn hiện hành: [kế hoạch](research_plan.md). Không chạy lại
pilot/rehearsal đã qua chỉ để bắt đầu full M1. Ngày 28/09 đổi config run mới
sang eval/save mỗi 100 bước, giữ 256 validation và một epoch full train;
checkpoint/source lịch sử giữ nguyên, không resume chúng bằng config mới.

## Trạng thái và phạm vi

Đã có mã runner. Preflight CPU kiểm tra đúng split `image_disjoint_v1`, 29.459
câu train, 1.407 validation, 1.968 test; kiểm tra đường dẫn ảnh train và tập
validation cố định 256 câu. Nó **không** tokenize toàn tập hoặc train mô hình.
Mốc lịch sử: M1 chạy đến bước **300/3.683** ngày 29/09/2026, dừng chủ động
và lưu checkpoint-300 trong run `sft-full_20260929_033221_507806`. Đã đối chiếu
Drive ngày 30/09: `execution.json=paused`, exit code 0. Phiên tiếp đã dùng
notebook M1 lịch sử (nay lưu ở `notebooks/archive/` và Drive), đúng source
M1 ghim 28/09; không chạy lại pilot hoặc reset optimizer.
P0 đủ 29.715 mẫu đã đạt; dùng lại report khi hash/môi trường vẫn khớp.

Optimizer pilot 20 bước đã qua trên T4 ngày 23/09/2026 (bằng chứng lịch sử).
Run nằm tại `Vietnamese_VQA_NLE/runs/sft-optimizer-pilot_20260923_140601_713060/`.
Notebook xác nhận `execution.json` passed, 20 optimizer step, checkpoint có
optimizer/scheduler, adapter tồn tại; ô cuối nạp lại adapter và sinh được một
câu validation (ID `393271001`). Đây không phải đánh giá chất lượng.

Pilot đo `train_seconds=865,54`, VRAM đỉnh `4,37 GiB`, LoRA-B thay đổi
`0,000213`, best validation loss `1,768` trên **32 câu validation** của pilot
(khác 256 câu của cấu hình full). `trainer_state.json` trên Drive ghi hai lượt
eval tốn 72,47 và 73,71 giây; `train_runtime=864,89` giây bao gồm hai lượt
eval và checkpoint. Trừ riêng thời gian eval rồi nhân theo 29.459/160 mẫu
train cho khoảng **36,8 giờ**, vẫn chứa thời gian lưu checkpoint. Nhân trực
tiếp toàn bộ pilot cho khoảng **44,3 giờ**. Cả hai chỉ là ước lượng thô; full
M1 khi đó dự kiến eval/save mỗi 500 bước với 256 câu; nay đổi thành 100 bước
cho run mới (28/09), tăng overhead nhưng giảm mất công khi ngắt. Không lấy tốc độ
pilot làm kết quả cuối. Cần chốt khả năng duy trì runtime/checkpoint trước
khi khởi chạy full M1 trên Colab free.
Theo [FAQ chính thức của Colab](https://research.google.com/colaboratory/faq.html),
runtime miễn phí chạy tối đa 12 giờ tùy điều kiện, GPU và hạn mức không được
bảo đảm. Vì thế không thể dựa vào một phiên T4 để chạy hết một epoch ước tính
trên 36 giờ; nếu giữ toàn bộ train cần kế hoạch resume qua nhiều phiên hoặc
nguồn GPU khác. Không dùng cách giữ phiên/né hạn mức trái chính sách.
Không dùng 20 câu smoke hoặc pilot để kết luận về chất lượng đồ án.

`sft_optimizer_pilot.yaml` chỉ dùng để đo tốc độ/cập nhật trọng số/lưu checkpoint.
`sft.yaml` dùng toàn bộ train trong một epoch, batch 1, tích lũy gradient 8,
validation và save mỗi 100 optimizer step. Checkpoint tốt nhất được chọn theo
validation loss trên 256 câu cố định. Cần đánh giá riêng đáp án và lời giải
thích bằng suy luận trên validation đầy đủ sau train; loss không chứng minh lời
giải thích bám ảnh.

## Đánh giá checkpoint cuối sau một epoch

`scripts/evaluate_m1_epoch.py` bổ sung loss của checkpoint-3683 vì bước
cuối không chia hết cho 100. Lệnh chỉ nhận run `passed`, mã thoát 0 và
checkpoint đã kiểm đầy đủ. Nó dùng source train đã ghim, đúng phiên bản
thư viện, dtype, collator và 256 câu validation trong `data_plan.json`.
Không train thêm, không đọc ảnh test và không sửa run cũ.

Script nằm trong source mới; `--source` phải trỏ tới trainer release 28/09,
không phải source mới. Chạy bằng tiến trình riêng trong môi trường train
đang ghim; không cài lại thư viện theo source đánh giá suy luận ở bước này.

```bash
python /path/to/current/scripts/evaluate_m1_epoch.py \
  --source /path/to/pinned/train-release \
  --environment /path/to/pinned/train-release/configs/environments/colab_full.example.yaml \
  --run /path/to/completed/m1-run \
  --output-dir /path/to/new/epoch-selection
```

`--check-only` kiểm đầu vào mà không nạp model hoặc ghi output. Kết quả
`selection.json` so loss đo riêng ở 3.683 với `final_validation` đã đo trên
model được Trainer chọn. Nếu bằng nhau giữ lựa chọn của Trainer. Artifact
ghi đường dẫn/hash adapter, ID validation và nguồn của từng loss. Không
chép lại `training_result.json` hoặc gọi loss bước cuối là loss của model tốt nhất.

Sau khi selection đạt, xuất prediction đủ 1.407 validation bằng giao thức
suy luận đã ghim và kiểm lời giải thích bằng ảnh. Lệnh chọn theo loss chưa
thay thế bước đó. Dùng API nạp/đọc trọng số adapter công khai của
[PEFT](https://huggingface.co/docs/peft/package_reference/functional);
không dùng phương thức nội bộ của Trainer để nạp checkpoint.

## Thứ tự chạy

**Quy trình lịch sử 300 → 600, không chạy lại:** mở notebook M1 lịch sử,
mount Drive và đọc run/checkpoint;
khôi phục source ghim + bundle ảnh vào runtime; kiểm report P0 và environment
lock; chạy ô resume 300 → 600. Ô chạy chặn thiếu trạng thái checkpoint hoặc
đã có run tiếp từ cùng checkpoint. Lần trở lại sau phải đọc run mới trước khi
chọn mốc tiếp theo. Đừng dùng “Chạy tất cả” để khởi động lại một cách máy móc.
Các lệnh pilot bên dưới là hướng dẫn tái lập lịch sử, không phải việc cần làm
lại để tiếp tục M1 hiện tại.

Trên máy có đủ ảnh, chạy CPU:

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py --full-data
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-full-preflight --environment configs/environments/local_full.yaml
```

Trên Colab, `notebooks/archive/08_m1_optimizer_pilot.ipynb` là điểm vào cho pilot:
nó xác minh source release và bundle ảnh bằng SHA-256, chạy preflight, rồi
chạy đúng 20 optimizer step. Có thể chạy bằng lệnh tương đương dưới đây sau
khi chuẩn bị source/dữ liệu theo `docs/data_protocol.md`, dùng
`configs/environments/colab_full.example.yaml`:

```bash
python scripts/run_pipeline.py sft-optimizer-pilot --environment configs/environments/colab_full.example.yaml
```

Chỉ khi `execution.json` báo `passed`, `training/training_result.json` có
`experiment_id=M1_SFT_PILOT`, `max_lora_weight_change > 0`, loss hữu hạn,
checkpoint đầy đủ, đo được `train_seconds`/`peak_vram_gib`, và kiểm tra nạp
lại adapter bằng một lượt suy luận riêng, mới cân nhắc:

```bash
python scripts/run_pipeline.py sft-full --environment configs/environments/colab_full.example.yaml
```

Mỗi lệnh tạo run mới; không ghi đè run cũ. Nếu full M1 bị ngắt, lấy checkpoint
hoàn chỉnh `runs/sft-full_.../training/checkpoint-N` và chạy lại từ đó:

```bash
python scripts/run_pipeline.py sft-full --environment configs/environments/colab_full.example.yaml --resume-from-checkpoint /content/drive/MyDrive/Vietnamese_VQA_NLE/runs/sft-full_.../training/checkpoint-N
```

Runner từ chối resume nếu run cũ đã hoàn tất, checkpoint thiếu trạng thái
optimizer/scheduler/RNG hoặc adapter, `trainer_state.json` sai bước checkpoint,
hoặc source/config/annotation/environment/checksum manifest ảnh đổi. Trước
mỗi phiên, trainer cũng kiểm tra lại SHA-256 của từng ảnh train và validation
được dùng, so data plan cũ/mới và phiên bản Torch/Transformers/PEFT/Accelerate/
bitsandbytes. Hiện pin Transformers 4.57.6 để hành vi accumulation/resume không
âm thầm thay đổi giữa các phiên.
Nó tạo run tiếp theo riêng, giữ nguyên run cũ. Không dùng test để chọn
checkpoint, prompt hay thông số train.

Nếu Colab chết đột ngột và `execution.json` của run cũ vẫn là `running`, trước
tiên xác nhận runtime cũ đã ngừng và chọn một checkpoint hoàn chỉnh. Chỉ khi
runner ghi đủ định danh tiến trình, heartbeat đã cũ ít nhất 10 phút và không
phát hiện tiến trình còn sống, mới thêm `--recover-interrupted-run` vào lệnh
resume trên. Run cũ không bị sửa; run tiếp tục được tạo riêng. Metadata cũ
không có các dấu hiệu này sẽ bị từ chối an toàn. Đây mới là cơ chế đã kiểm thử
local bằng giả lập, **chưa được nghiệm thu bằng hard disconnect GPU thật**.
Checkpoint từ source release cũ không dùng được với mã runner mới vì cổng
provenance yêu cầu source SHA-256 khớp; không bỏ cổng này để ép resume.

**Cách tái lập rehearsal dừng chủ động đã chạy ngày 24/09:** tạo pilot mới
dừng ở bước 10. Kiểm tra `execution.json` là `paused` và
`training/session_result.json` trỏ tới
`checkpoint-10` đủ adapter/optimizer/scheduler/RNG. Sau đó resume pilot này
đến bước 20 bằng source, dữ liệu và phiên bản thư viện **không đổi**:

```bash
python scripts/run_pipeline.py sft-optimizer-pilot --environment configs/environments/colab_full.example.yaml --stop-after-step 10
python scripts/run_pipeline.py sft-optimizer-pilot --environment configs/environments/colab_full.example.yaml --resume-from-checkpoint /content/drive/MyDrive/Vietnamese_VQA_NLE/runs/sft-optimizer-pilot_.../training/checkpoint-10
```

Có thể để script tạo hai run mới và kiểm tra bằng chứng tự động sau khi đã
xác minh source release, bundle ảnh và profile Colab:

```bash
python scripts/rehearse_m1_resume.py --environment configs/environments/colab_full.example.yaml
```

Script chỉ chạy pilot 10→20, không chạy full M1. Nó ghi
`resume_rehearsal.json` vào run thứ hai sau khi kiểm tra trạng thái, checkpoint
có optimizer/scheduler/RNG, loss hữu hạn, 10 bước optimizer sau resume,
adapter và ít nhất một tensor LoRA-B thực sự thay đổi giữa checkpoint 10/20.
Report ghi độ thay đổi lớn nhất hữu hạn của các tensor này. Việc nạp lại
adapter để sinh một câu validation vẫn phải thử riêng; script chưa kiểm tra
chất lượng dự đoán. Notebook 08 đã ghim release lịch sử và hai ô pilot/reload
độc lập cuối chỉ định nghĩa hàm; “Chạy tất cả” không tự chạy pilot đó nhưng
**vẫn chạy rehearsal GPU và tạo hai run mới**. Không chạy lại chỉ để xem log cũ.
Giữ nguyên source release, dữ liệu và profile giữa hai lượt; nếu một lượt lỗi,
hai run vẫn được giữ để xem log, không ghi đè run lịch sử.

Rehearsal chỉ đạt khi `execution.json=passed`, `training_result.json` ghi
`global_step=20`, `resumed_from_global_step=10`, `optimizer_steps_this_run=10`,
loss hữu hạn và tensor LoRA-B trong checkpoint thay đổi. Sau đó còn phải thử
nạp lại adapter và suy luận một câu validation riêng. Đây vẫn chỉ là pilot, không
phải M1 đầy đủ. Report và run của lượt 24/09 nằm ở `TIEN_DO.md`.
Hard disconnect và nạp lại adapter từ run resume để suy luận vẫn chưa được thử.
Full M1 có thể dừng chủ động tại bội số `save_steps` (hiện 100), rồi dùng
checkpoint của run `paused` để tiếp tục. Nên cân nhắc thời lượng phiên và chu kỳ lưu checkpoint
trước khi cấp GPU: chọn số bước mỗi phiên theo tốc độ checkpoint đầu của run thật.

## Điều kiện nghiệm thu M1

- Full run `passed`, mã thoát 0, `global_step=3683`, `epoch=1.0`; checkpoint
  cuối đủ adapter/optimizer/scheduler/RNG, adapter được chọn khớp trọng số
  best checkpoint. Bước này đã qua kiểm tra ngày 07/10.
- Có `selection.json` sau khi đo riêng loss checkpoint-3683 trên đúng 256
  validation. Không lấy loss sau load-best làm loss của checkpoint cuối.
- M0 và M1 suy luận trên cùng validation đầy đủ, cùng ảnh/prompt/pixel/generation,
  không chỉ so training loss.
- Báo cáo độ đúng đáp án, chất lượng lời giải thích và chấm tay tính bám ảnh;
  chỉ đánh giá test một lần sau khi đóng băng quyết định.
