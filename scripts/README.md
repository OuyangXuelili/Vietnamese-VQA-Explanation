# Các lệnh chạy

Chạy từ thư mục repo bằng Python của môi trường đã cài. Local dùng
`.venv\Scripts\python.exe`; trên Colab dùng môi trường của notebook.
`--help` mô tả tham số của từng lệnh. Lệnh train/export chỉ dùng để tái lập
hoặc nối phần còn thiếu của một record đã kiểm, không chạy lại theo mục lục.

## Lệnh chính

| Công việc | Lệnh | Đầu vào → đầu ra | Nơi chạy |
|---|---|---|---|
| Kiểm source | [check_project.py](check_project.py) | Source, test, dependency → Ruff và pytest; `--source-only` cho checkout thiếu lịch sử | Local CPU, cần Node |
| Quản lý đợt sửa | [task_harness.py](task_harness.py) | Mục tiêu, tiêu chí, bằng chứng → receipt trong `artifacts/tasks/` | Local CPU |
| Tái lập M1 đầy đủ | [train_sft_full.py](train_sft_full.py) | Environment, model và `training/sft.yaml` → run/checkpoint M1 | GPU CUDA; M1 đã hoàn tất |
| Train M2/M3/M4 | [launch_preference_training.py](launch_preference_training.py) | Launch record, project dir, arm → inspect/gate/run theo record | Inspect ở CPU; gate/train cần GPU |
| Xuất validation M0–M3 | [run_study_validation.py](run_study_validation.py) | Study, run group, adapter M1, environment → prediction và manifest | Inspect ở CPU; export cần GPU |
| Xuất test đã khóa | [run_study_test.py](run_study_test.py) | Plan + SHA-256, identity, adapter → prediction/resume riêng từng identity | Inspect ở CPU; export cần GPU |
| Tính BERTScore validation | [run_study_semantic.py](run_study_semantic.py) | Prediction đã đủ và annotation → semantic score | Mặc định CUDA; không train/nạp Qwen |
| Bảng validation | [report_study_validation.py](report_study_validation.py) | Prediction, semantic, study và ảnh → bảng, CI, biên nhận kiểm | Local CPU |
| Tổng hợp ba seed | [report_validation_seeds.py](report_validation_seeds.py) | Ba report đã kiểm → trung bình và độ lệch chuẩn | Local CPU |
| Bảng test | [report_test.py](report_test.py) | Plan, tám run, annotation, semantic đã lưu → bảng và CI | Local CPU |
| Demo M1 | [run_demo.py](run_demo.py) | Config và evidence → web replay; live nhận thêm adapter và ảnh mới | Replay ở CPU; live cần GPU |
| Đóng gói source | [package_source.py](package_source.py) | Source theo allowlist → ZIP, manifest, `.sha256` trong `artifacts/releases/` | Local CPU; chưa upload |

Launcher M2/M3/M4 yêu cầu `--launch-record`, `--project-dir`, `--arm`;
`--mode` mặc định là `inspect`. Chỉ chọn `gate` hoặc `train` khi đã kiểm
record, source, config, dữ liệu và checkpoint. Không tự tạo record mới để
tiếp tục run cũ. Xem [hướng dẫn train/resume](../docs/preference_training.md)
và [M4](../docs/answer_preservation.md).

Chọn [study/config](../configs/README.md) trước khi chạy validation. Tham số
`--study-config` bị bỏ trống trong `run_study_validation.py` dành cho lượt
186 cặp lịch sử. `run_study_semantic.py` đọc validation đã xuất, không phải
lệnh tổng hợp semantic cho tám identity test. Quy trình test và M4 được điều
phối trong [notebook 08 và 10](../notebooks/README.md).

## Chạy ở local

Sau khi [cài dependency](../README.md#local):

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py --source-only
.\.venv\Scripts\python.exe scripts/package_source.py --help
.\.venv\Scripts\python.exe scripts/launch_preference_training.py --help
```

Bỏ `--source-only` khi máy có đủ hồ sơ lịch sử để kiểm cả nhóm historical.
`python scripts/package_source.py` không có tham số sẽ tạo một release mới;
`--help` chỉ xem hướng dẫn. Lệnh này không chứa dữ liệu ảnh, checkpoint hoặc
notebook có output và không tự đồng bộ Drive.

## Dữ liệu và cặp ưu tiên

- [prepare_vivqax.py](prepare_vivqax.py),
  [make_image_disjoint_split.py](make_image_disjoint_split.py),
  [audit_dataset.py](audit_dataset.py): chuẩn hóa và kiểm dữ liệu/split.
- [download_coco_full.py](download_coco_full.py),
  [verify_coco_full.py](verify_coco_full.py),
  [package_coco_full.py](package_coco_full.py),
  [verify_coco_bundle.py](verify_coco_bundle.py): tải, kiểm và đóng gói ảnh.
- [select_preference_questions.py](select_preference_questions.py),
  [generate_preference_candidates.py](generate_preference_candidates.py),
  [prepare_preference_review.py](prepare_preference_review.py),
  [review_preference_images.py](review_preference_images.py): chuẩn bị ứng viên
  và trang xem ảnh; sinh ứng viên cần GPU, xem/chỉnh hồ sơ chạy local.
- [audit_preference_quality.py](audit_preference_quality.py),
  [audit_preference_review.py](audit_preference_review.py),
  [validate_preferences.py](validate_preferences.py),
  [finalize_preference_review.py](finalize_preference_review.py): kiểm chất lượng,
  hồ sơ và bundle. Kiểm metadata không thay việc đối chiếu từng cặp với ảnh.

Bộ 1.000 cặp đã khóa. Không chạy lại các lệnh chuẩn bị để thay bộ dữ liệu
nghiên cứu sau khi đã xem test. Xem [giao thức dữ liệu](../docs/data_protocol.md)
và [gán nhãn cặp](../docs/preference_annotation.md).

## Đánh giá và demo

- [analyze_validation.py](analyze_validation.py),
  [prepare_validation_review.py](prepare_validation_review.py): đối chiếu và
  xem các prediction đã lưu, không train.
- [prepare_rubric_calibration.py](prepare_rubric_calibration.py),
  [check_calibration.py](check_calibration.py),
  [blind_review.py](blind_review.py): chuẩn bị, kiểm và tổng hợp nhãn theo
  [giao thức chấm](../docs/human_evaluation.md); không tự tạo nhãn người.
- [check_image_dependence.py](check_image_dependence.py): đối chứng đổi ảnh;
  xuất dự đoán cần GPU, không thay kết quả benchmark gốc.
- [check_demo_live.py](check_demo_live.py),
  [diagnose_inference.py](diagnose_inference.py): kiểm demo/suy luận theo
  profile. Inspect không nạp model; phép đo thật cần GPU. Xem [demo](../docs/demo.md).

## Lệnh tái lập và thử nghiệm cũ

| Lệnh | Phạm vi cần lưu ý |
|---|---|
| [train_sft.py](train_sft.py) | Mặc định smoke tám update theo `sft_smoke.yaml`; không phải M1 đầy đủ |
| [train_dpo.py](train_dpo.py) | Wrapper DPO mặc định `dpo.yaml` và study 186 cặp; không phải nghiên cứu chính 1.000 cặp |
| [train_preference.py](train_preference.py) | CLI trainer cấp thấp; mặc định study pilot cũ, cần chỉ rõ cấu hình; ưu tiên launcher từ record |
| [run_pipeline.py](run_pipeline.py) | Runner audit/preflight/pilot/M1; chọn stage rõ ràng, dùng `--dry-run` để xem lệnh |
| [profile_sft_tokens.py](profile_sft_tokens.py), [pilot_sft_batch.py](pilot_sft_batch.py), [benchmark_sft_collator.py](benchmark_sft_collator.py) | Đo token, batch hoặc collator; không phải kết quả huấn luyện cuối |
| [resume_pinned_m1.py](resume_pinned_m1.py), [rehearse_m1_resume.py](rehearse_m1_resume.py), [evaluate_m1_epoch.py](evaluate_m1_epoch.py), [freeze_m1_selection.py](freeze_m1_selection.py) | Khôi phục/kiểm/chốt M1; giữ record và source gốc khi tái lập |

Các công cụ cũ còn lại phục vụ những hồ sơ chuẩn bị/phục hồi đã lưu. Không
đổi tên hoặc xóa chỉ vì không nằm trong luồng chạy hiện tại. Xem
[M1](../docs/sft_full.md), [notebook lịch sử](../notebooks/archive/README.md),
[local/cloud](../docs/local_cloud.md) và [harness](../docs/harness.md) khi cần tái lập.
