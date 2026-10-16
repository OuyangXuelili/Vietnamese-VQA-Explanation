# Vietnamese VQA Explanation

Mã nguồn cho đề tài **Nghiên cứu cải thiện chất lượng lời giải thích trong
hỏi đáp hình ảnh tiếng Việt**. Đầu vào là ảnh và câu hỏi tiếng Việt; đầu ra
gồm đáp án và lời giải thích.

Đọc [mô tả dự án](docs/project_description.md) để xem bài toán, dữ liệu và
kết quả; [kế hoạch](docs/research_plan.md) ghi thiết kế và các bước còn lại.

## Trạng thái — 16/10/2026

- M1 đã học một epoch, chọn checkpoint-3683 bằng loss trên 256 validation.
  Đánh giá đủ 1.407 validation: EM 70,93%, Token-F1 75,11%, ROUGE-L 44,29%,
  BERTScore 82,47%. [Kết quả M0/M1](docs/project_description.md#kết-quả-m0m1).
- Lượt 186 cặp giữ làm lịch sử kiểm triển khai, không dùng làm kết quả chính.
- Đã duyệt lại và khóa bộ chính **1.000 cặp / 1.000 ảnh train**, còn 117
  cặp dự phòng. Kiểm chéo 100 cặp và rà thêm 37 cặp, rút 20 cặp chưa đạt.
  Bộ mới có 992 negative có kiểm soát và 8 output M1 nguyên văn. Baseline
  chỉ đọc chữ đạt 62,55%, nên vẫn cần kiểm đổi ảnh và chấm theo ảnh sau train.
- M2/M3 bản chính, seed 42 đã hoàn tất 375 update / ba epoch. Validation
  đủ 1.407 câu mỗi nhánh: M1/M2/M3 đạt EM 70,93 / 69,72 / 69,51 và
  BERTScore 82,47 / 76,56 / 79,74. M3 hơn M2 về điểm giải thích tự động,
  nhưng chưa hơn M1. [Bảng bốn nhánh](docs/project_description.md#validation-bản-chính--seed-42).
- Đối chứng đổi ảnh seed 42 đã xong; chưa thấy lợi thế M3 so với M2.
  Seed 43 đang train; seed 44, test và chấm mù chưa hoàn tất.
  Demo live chưa nghiệm thu GPU.

## Các nhánh

| Mã | Cấu hình |
|---|---|
| M0 | Qwen2.5-VL-3B-Instruct gốc |
| M1 | SFT trên ViVQA-X |
| M2 | Từ M1, SFT tiếp trên chosen |
| M3 | Từ cùng M1, DPO trên chosen/rejected |

M2/M3 dùng cùng câu/ảnh/chosen, seed, epoch và LR; không coi là cùng chi phí
tính toán. Lượt cũ dùng LR 5e-6; kế hoạch chính mới dùng 1e-5, ba epoch.
Test giữ kín tới khi khóa lựa chọn; ROUGE/BERTScore không thay đánh giá theo ảnh.

## Tổ chức

```text
src/vqa_explanation/     dữ liệu, train, suy luận, đánh giá và demo
scripts/                CLI
configs/                thuật toán và đường dẫn từng môi trường
tests/                  kiểm thử
notebooks/              04_train_preferences.ipynb; lịch sử M1 ở archive
app/static/             giao diện demo
docs/, reports/         hướng dẫn, bảng và hình
data/, runs/, artifacts/ dữ liệu, checkpoint và bằng chứng ngoài Git
```

[Mục lục artifacts](artifacts/README.md) chỉ rõ bản duyệt, báo cáo pilot và
source release đang dùng. HTML/CSS/JavaScript của editor nằm riêng trong
`src/vqa_explanation/preference/templates/` và được nhúng khi xuất trang duyệt.
[Quy trình review](docs/code_review.md) mô tả các kiểm tra cần có trước khi đồng bộ.
Notebook `04_train_preferences.ipynb` có các mục riêng cho lượt 186 cặp,
validation, xử lý cặp và lượt chính 1.000 cặp; không chạy tất cả để resume.
Bản M1 đã chạy giữ trong `notebooks/archive/`, gồm cả output tải từ Colab.
[Mở M1 có output](https://colab.research.google.com/drive/13Abjl7kJyDKzbxyYsCetJ540OzdK2P_E).

## Local

Tại repo, dùng Python 3.10+ và môi trường riêng:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/check_project.py
```

Máy hiện tại có annotation và 28.180 ảnh ViVQA-X tham chiếu, không tải lại.
Profile `configs/environments/local_full.yaml` dùng dữ liệu đầy đủ;
`local.yaml` dùng smoke 500 ảnh. Local không mount Drive hoặc train Qwen CPU.
Checkout mới: xem [dữ liệu](docs/data_protocol.md) và
[local/cloud](docs/local_cloud.md). Kiểm trainer nhỏ cần thêm
`.[verification]`, không cài môi trường CUDA lên máy này.

## Việc tiếp theo

1. Đọc kết quả seed 42 và ví dụ tăng/giảm điểm; chưa mở test hoặc đổi metric.
2. Hoàn thành seed 43/44 và đối chứng đã chốt trong [kế hoạch](docs/research_plan.md).
3. Đổi ảnh, test sau khóa thiết kế và chấm theo ảnh: [giao thức](docs/experiment_protocol.md),
   [chấm mù](docs/human_evaluation.md).
4. Hoàn thiện [demo](docs/demo.md), phân tích và báo cáo.

[Harness](docs/harness.md) là kiểm code/provenance, không phải kết quả mô hình.
Notebook 00–08 và các run cũ giữ để tái lập, không chạy lại lần lượt.

## Lưu trữ

Git giữ source/config/test; Drive giữ dataset bundle, checkpoint, prediction
và log. Colab `/content` là ổ tạm. Source release có checksum, không chứa
ảnh, trọng số hoặc credentials.

CI chạy kiểm CPU trên Python 3.10 và 3.11, có dependency cho trainer nhỏ,
API và Node để kiểm lưu nháp. `pytest -m "not historical" --fail-on-skip`
không cho bỏ qua kiểm tra source vì thiếu dependency. Nhóm `historical` cần
artifact run ngoài Git và vẫn được chạy khi kiểm toàn bộ trên máy này.
