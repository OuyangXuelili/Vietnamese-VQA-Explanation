# Vietnamese VQA Explanation

Mã nguồn đề tài **Nghiên cứu cải thiện chất lượng lời giải thích trong hỏi
đáp hình ảnh tiếng Việt**. Đầu vào là ảnh và câu hỏi tiếng Việt; đầu ra gồm
đáp án và lời giải thích. Backbone là Qwen2.5-VL-3B-Instruct, dữ liệu ViVQA-X.

[Mô tả và kết quả](docs/project_description.md) ·
[Thiết kế nghiên cứu](docs/research_plan.md) ·
[Lệnh chạy](scripts/README.md) · [Cấu hình](configs/README.md)

## Kết quả hiện có

M1 đã học một epoch / 3.683 update trên 29.459 câu train. M2/M3 học thêm
từ cùng M1 trên 1.000 cặp, ba epoch / 375 update với ba seed 42, 43, 44.
Validation và test tự động đã hoàn tất. Chấm độc lập theo ảnh còn thiếu.

| Nhánh | Thiết lập | EM | Token F1 | ROUGE-L | BERTScore |
|---|---|---:|---:|---:|---:|
| M0 | Model gốc | 41,36 | 53,16 | 27,74 | 75,92 |
| M1 | SFT trên ViVQA-X | 70,17 | 75,21 | 44,52 | 82,48 |
| M2 | Từ M1, SFT trên chosen | 70,33 | 75,36 | 30,42 | 76,58 |
| M3 | Từ M1, DPO trên chosen/rejected | 67,72 | 73,11 | 35,47 | 79,17 |

Test gồm 1.968 câu / 1.921 ảnh; chỉ số nhân 100. M2/M3 là trung bình ba
seed từ cùng M1. M3 hơn M2 về tương đồng văn bản nhưng giảm 2,61 điểm EM
và vẫn dưới M1 về tương đồng giải thích. Chưa kết luận cải thiện hỗ trợ từ ảnh.
Xem [giao thức](docs/experiment_protocol.md),
[chấm theo ảnh](docs/human_evaluation.md) và [lỗi ngôn ngữ](docs/language_errors.md).

Bộ preference gồm 992 rejected tổng hợp và tám output M1 nguyên văn;
phép kiểm text-only đạt 62,55%. Cách tạo và các giới hạn ở
[giao thức cặp ưu tiên](docs/preference_annotation.md).

M4 bổ sung NLL trên token đáp án vào DPO, mới đánh giá validation seed 42.
So với M3 trên 1.407 câu, BERTScore tăng 0,53 điểm; EM chênh −0,07 điểm
phần trăm, CI 95% [−1,85; +1,63]. [Thiết kế và bảng M4](docs/answer_preservation.md).

## Tổ chức mã nguồn

```text
src/vqa_explanation/    dữ liệu, training, inference, evaluation, demo, utils
scripts/               lệnh chạy và mục lục
configs/               model, dữ liệu, thuật toán, môi trường, study
notebooks/             template 04–10; archive giữ notebook lịch sử
app/                   giao diện demo và trình xem validation
tests/                 unit, integration, frontend
docs/, reports/        hướng dẫn, bản thảo, bảng, hình và slide
data/, runs/, artifacts/ dữ liệu, checkpoint và bằng chứng ngoài Git
```

Logic dùng lại nằm trong `src/`; notebook điều phối và hiển thị kết quả.
[Mục lục notebook](notebooks/README.md) phân biệt train, đánh giá và demo.
Các bản đã chạy giữ nguyên output, gồm
[M1 có output](https://colab.research.google.com/drive/13Abjl7kJyDKzbxyYsCetJ540OzdK2P_E).
Template mới không thay thế hồ sơ thực thi. Xem thêm
[artifacts](artifacts/README.md) và [quy trình review](docs/code_review.md).

## Local

Dùng Python 3.10+ và môi trường riêng. Bộ kiểm đầy đủ cần Torch/Torchvision
CPU, các nhóm verification/demo/report và Node 22:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,verification,demo,report]"
node --version
.\.venv\Scripts\python.exe scripts/check_project.py --source-only
```

Checkout Git/source ZIP dùng `--source-only` vì không có hồ sơ chạy ngoài
Git. Máy có đủ lịch sử bỏ cờ này để kiểm cả nhóm historical. Cả hai chế độ
đều từ chối skip; gate không huấn luyện Qwen trên CPU.

Máy hiện tại đã có annotation và 28.180 ảnh tham chiếu. Dùng
`configs/environments/local_full.yaml` cho dữ liệu đầy đủ, `local.yaml` cho
smoke 500 ảnh. Checkout mới xem [dữ liệu](docs/data_protocol.md) và
[local/cloud](docs/local_cloud.md); không tải lại dữ liệu đã kiểm.

## Chạy và tái lập

| Nhu cầu | Hướng dẫn |
|---|---|
| Chọn lệnh/config đúng | [scripts](scripts/README.md), [configs](configs/README.md) |
| Tái lập M1 | [SFT đầy đủ](docs/sft_full.md) |
| Train/resume M2/M3 | [Preference training](docs/preference_training.md) |
| Xem prediction và phân tích lỗi | [Validation](docs/validation_analysis.md) |
| Hỏi ảnh mới trên GPU hoặc replay local | [Demo](docs/demo.md) |
| Kiểm và bàn giao một đợt sửa | [Harness](docs/harness.md) |

Demo M1 đã kiểm trên T4, gồm phục hồi mạng/backend ngày 7/11/2026. Replay
đọc kết quả đã lưu; live nhận ảnh mới và cần GPU. Web phụ thuộc phiên Colab.
Guard NaN/Inf ngăn trả văn bản từ phép tính lỗi; nguyên nhân số học còn cần
chẩn đoán. Ba ca nghiệm thu chức năng không thay benchmark chất lượng.

Git giữ source/config/test. Dataset, checkpoint và prediction được quản lý
riêng trên Drive; `/content` của Colab là ổ tạm. Source release có manifest
và checksum, không chứa ảnh, trọng số hoặc credentials. Notebook có output
là bằng chứng lịch sử riêng, không bị ghi đè khi cập nhật template.

Phần còn lại: đánh giá độc lập trên bộ ảnh đã khóa, hoàn thiện báo cáo/slide
hiện có và nghiệm thu demo. [Kế hoạch](docs/research_plan.md) ghi phạm vi và
điều kiện mở thí nghiệm tiếp theo.
