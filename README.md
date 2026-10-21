# Vietnamese VQA Explanation

Mã nguồn cho đề tài **Nghiên cứu cải thiện chất lượng lời giải thích trong
hỏi đáp hình ảnh tiếng Việt**. Đầu vào là ảnh và câu hỏi tiếng Việt; đầu ra
gồm đáp án và lời giải thích.

Đọc [mô tả dự án](docs/project_description.md) để xem bài toán, dữ liệu và
kết quả; [kế hoạch](docs/research_plan.md) ghi thiết kế và các bước còn lại.

## Trạng thái — 21/10/2026

- M1 đã học một epoch, chọn checkpoint-3683 bằng loss trên 256 validation.
  Đánh giá đủ 1.407 câu validation: EM 70,93; Token-F1 75,11; ROUGE-L 44,29;
  BERTScore 82,47. Các chỉ số nhân 100.
  [Kết quả M0/M1](docs/project_description.md#kết-quả-m0m1).
- Lượt 186 cặp giữ làm lịch sử kiểm triển khai, không dùng làm kết quả chính.
- Đã duyệt lại và khóa bộ chính **1.000 cặp / 1.000 ảnh train**, còn 117
  cặp dự phòng. Kiểm chéo 100 cặp và rà thêm 37 cặp, rút 20 cặp chưa đạt.
  Bộ mới có 992 negative có kiểm soát và 8 output M1 nguyên văn. Baseline
  xếp hạng chỉ đọc chữ đạt 62,55%; chất lượng cặp vẫn cần thẩm định theo ảnh.
- M2/M3 bản chính, seed 42 đã hoàn tất 375 update / ba epoch. Validation
  đủ 1.407 câu mỗi nhánh: M1/M2/M3 đạt EM 70,93 / 69,72 / 69,51 và
  BERTScore 82,47 / 76,56 / 79,74. M3 hơn M2 về độ tương đồng văn bản,
  nhưng chưa hơn M1. [Bảng bốn nhánh](docs/project_description.md#validation-bản-chính--seed-42).
- Đối chứng đổi ảnh seed 42 đã xong; chưa thấy lợi thế M3 so với M2.
  Seed 43 đã kiểm độc lập đủ validation, BERTScore và CI; M2/M3 đạt
  BERTScore 76,55 / 79,37, vẫn thấp hơn M1. Seed 44 đã hoàn tất đánh giá.
  Qua ba seed, BERTScore M2/M3 là 76,55 ± 0,01 / 79,26 ± 0,54
  (trung bình ± độ lệch chuẩn mẫu); M3 chưa hơn M1 hoặc chứng minh bám ảnh hơn.
  Test và chấm mù chưa thực hiện.
  Demo live chưa nghiệm thu GPU.
- Đối chứng LR 5e-6 seed 42: report M2/M3 đều `passed`, 375 update / ba epoch.
  M3 hoàn tất attempt `m3_train_resume175` ngày 21/10. Còn kiểm độc lập đầy đủ
  checkpoint cuối M3 và đánh giá validation của đối chứng; chưa có điểm chất lượng mới.

## Các nhánh

| Mã | Cấu hình |
|---|---|
| M0 | Qwen2.5-VL-3B-Instruct gốc |
| M1 | SFT trên ViVQA-X |
| M2 | Từ M1, SFT tiếp trên giải thích tốt hơn (chosen) |
| M3 | Từ cùng M1, DPO trên cặp giải thích tốt hơn/kém hơn (chosen/rejected) |

M2/M3 dùng cùng câu/ảnh/chosen, seed, epoch và LR; không coi là cùng chi phí
tính toán. Lượt 186 cặp dùng LR 5e-6; bộ chính 1.000 cặp dùng 1e-5, ba epoch.
Test giữ kín tới khi khóa lựa chọn; ROUGE/BERTScore không thay đánh giá theo ảnh.

Seed điều khiển tính ngẫu nhiên, trong thiết lập này chủ yếu là thứ tự học
dữ liệu. Mỗi seed là một lần huấn luyện độc lập từ cùng trọng số M1, không
học nối từ seed trước; cả hai nhánh đều tắt dropout. Kế hoạch dùng ba seed
42, 43, 44 để đo mức dao động giữa các lần chạy; validation đã đủ ba seed,
chưa có đánh giá test cuối.

## Tổ chức

```text
src/vqa_explanation/     dữ liệu, train, suy luận, đánh giá và demo
scripts/                CLI
configs/                thuật toán và đường dẫn từng môi trường
tests/                  kiểm thử
notebooks/              04 train, 05 validation; lịch sử M1 ở archive
app/static/             giao diện demo
docs/, reports/         hướng dẫn, bảng và hình
data/, runs/, artifacts/ dữ liệu, checkpoint và bằng chứng ngoài Git
```

[Mục lục artifacts](artifacts/README.md) chỉ rõ bản duyệt, báo cáo pilot và
source release đang dùng. HTML/CSS/JavaScript của editor nằm riêng trong
`src/vqa_explanation/preference/templates/` và được nhúng khi xuất trang duyệt.
[Quy trình review](docs/code_review.md) mô tả các kiểm tra cần có trước khi đồng bộ.
Notebook `04_train_preferences.ipynb` chỉ điều phối huấn luyện M2/M3;
`05_evaluate_preferences.ipynb` đánh giá độc lập sau train. Chọn record/seed
ở đầu, không tạo notebook theo từng lượt. Bản Colab đã có output giữ nguyên;
hai template không thay lịch sử thực thi. Xem [hướng dẫn notebook](notebooks/README.md).
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

1. Giữ kết quả cả ba seed 42/43/44 đã kiểm, gồm validation/BERTScore/CI.
2. Nghiệm thu checkpoint cuối và chạy validation đối chứng LR 5e-6 trong
   [kế hoạch](docs/research_plan.md). Notebook 05/CLI hỗ trợ dùng lại M0/M1 đã
   ghim khi đủ hash và cùng identity suy luận; M2/M3 lấy từ lượt LR mới.
3. Khóa cấu hình rồi đánh giá test và chấm theo ảnh: [giao thức](docs/experiment_protocol.md),
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
