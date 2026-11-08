# Cấu hình

Các YAML mô tả model, dữ liệu, thuật toán và môi trường. Study ghép các
nhánh so sánh; launch record ghim source, đường dẫn và checksum của lượt chạy.
Không chọn config chỉ theo tên `m2`/`m3` hoặc sửa YAML đã dùng cho run cũ.

## Study và cấu hình huấn luyện

| Phạm vi | Study | Cấu hình nhánh |
|---|---|---|
| Chính: 1.000 cặp, seed 42 | [preference_study_v2.yaml](experiments/preference_study_v2.yaml) | [sft_chosen_full.yaml](training/sft_chosen_full.yaml), [dpo_full.yaml](training/dpo_full.yaml) |
| Lặp seed 43 | [preference_study_seed43.yaml](experiments/preference_study_seed43.yaml) | [sft_chosen_seed43.yaml](training/sft_chosen_seed43.yaml), [dpo_seed43.yaml](training/dpo_seed43.yaml) |
| Lặp seed 44 | [preference_study_seed44.yaml](experiments/preference_study_seed44.yaml) | [sft_chosen_seed44.yaml](training/sft_chosen_seed44.yaml), [dpo_seed44.yaml](training/dpo_seed44.yaml) |
| Đối chứng LR, seed 42 | [preference_study_lr5e6.yaml](experiments/preference_study_lr5e6.yaml) | [sft_chosen_lr5e6.yaml](training/sft_chosen_lr5e6.yaml), [dpo_lr5e6.yaml](training/dpo_lr5e6.yaml) |
| M4: DPO thêm NLL đáp án | [answer_preservation.yaml](experiments/answer_preservation.yaml) | [dpo_answer.yaml](training/dpo_answer.yaml); M1/M3 làm mốc so sánh |
| Pilot lịch sử: 186 cặp | [study_budget_v2.yaml](experiments/study_budget_v2.yaml) | [sft_chosen.yaml](training/sft_chosen.yaml), [dpo.yaml](training/dpo.yaml) |

Nghiên cứu chính M2/M3 dùng ba seed 42/43/44 từ cùng M1; M4 mới có seed 42
và đánh giá validation. Trường `planned_seeds` hoặc `status` trong YAML đã
khóa phản ánh kế hoạch khi tạo, không tự chứng minh run đã hoàn tất.
[Kế hoạch nghiên cứu](../docs/research_plan.md) và hồ sơ kết quả ghi trạng thái thật.

`experiments/m0_pretrained.yaml`, `m1_sft.yaml`, `m2_sft_chosen.yaml`,
`m3_dpo.yaml` mô tả vai trò các nhánh. Chúng không thay study/launch record;
riêng mô tả M2 vẫn trỏ tới cấu hình pilot cũ. Không dùng nó để chọn bộ chính.

## M1 và các phép thử kỹ thuật

- [sft.yaml](training/sft.yaml): M1 đầy đủ trên split đã khóa. Entry point là
  `scripts/train_sft_full.py`; M1 hiện đã hoàn tất.
- [sft_smoke.yaml](training/sft_smoke.yaml): smoke tám update, mặc định của
  `scripts/train_sft.py`.
- [sft_batch_pilot.yaml](training/sft_batch_pilot.yaml),
  [sft_optimizer_pilot.yaml](training/sft_optimizer_pilot.yaml): kiểm batch,
  optimizer và resume. Không thay kết quả M1 cuối.
- [sft_readiness.yaml](training/sft_readiness.yaml),
  [sft_readiness_full.yaml](training/sft_readiness_full.yaml): kiểm điều kiện
  đầu vào. [sft_chosen_lr2e5.yaml](archive/sft_chosen_lr2e5.yaml) là cấu hình
  đối chứng cũ lưu để truy vết, không phải mặc định hiện hành.

## Các nhóm còn lại

| Thư mục | Vai trò |
|---|---|
| `model/` | [Qwen2.5-VL-3B](model/qwen25_vl_3b.yaml), revision, QLoRA và LoRA |
| `data/` | [Dữ liệu chính](data/vivqax.yaml) và [smoke](data/vivqax_smoke.yaml) |
| `environments/` | Đường dẫn annotation, ảnh, run theo máy; tách khỏi thuật toán |
| `inference/` | Precision và cấu hình suy luận; không đổi profile của benchmark đã khóa |
| `evaluation/` | Metric, rubric theo ảnh, chẩn đoán validation và kiểm số học |
| `demo/` | Evidence M1 đã ghim và cấu hình web replay/live |

Local đầy đủ dùng [local_full.yaml](environments/local_full.yaml);
[local.yaml](environments/local.yaml) dành cho smoke. Tệp `*.example.yaml`
là mẫu đường dẫn cho Colab/Kaggle; phải tạo cấu hình thực tế theo kho của
phiên đang dùng, không chép đường dẫn Drive vào `src/`.

Profile [legacy_fp16.yaml](inference/legacy_fp16.yaml) giữ giao thức kết quả
đã xuất. [matched_fp16.yaml](inference/matched_fp16.yaml) và
[diagnostic_fp32.yaml](inference/diagnostic_fp32.yaml) dành cho phép kiểm
tương ứng; không trộn điểm từ các profile thành một bảng so sánh.

[Bảng lệnh](../scripts/README.md) · [Train/resume](../docs/preference_training.md) ·
[Giao thức thực nghiệm](../docs/experiment_protocol.md) · [Demo](../docs/demo.md)
