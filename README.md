# Vietnamese VQA Explanation

Mã nguồn cho đề tài **Nghiên cứu cải thiện chất lượng lời giải thích trong
hỏi đáp hình ảnh tiếng Việt**. Đầu vào là ảnh và câu hỏi tiếng Việt; đầu ra
gồm đáp án và lời giải thích.

Đọc [mô tả dự án](docs/project_description.md) để xem bài toán, dữ liệu và
kết quả; [kế hoạch](docs/research_plan.md) ghi thiết kế và các bước còn lại.

## Trạng thái — 05/11/2026

M1 đã học đủ 29.459 câu train, một epoch / 3.683 update. Nghiên cứu chính
M2/M3 dùng 1.000 cặp trên 1.000 ảnh train, ba epoch / 375 update và ba seed
42, 43, 44 từ cùng M1. Validation và test tự động đã hoàn tất.

Test gồm 1.968 câu trên 1.921 ảnh, tám lượt xuất / 15.744 prediction.
Các chỉ số nhân 100; M2/M3 là trung bình ba seed.

| Nhánh | EM | Token F1 | ROUGE-L | BERTScore |
|---|---:|---:|---:|---:|
| M0 | 41,36 | 53,16 | 27,74 | 75,92 |
| M1 | 70,17 | 75,21 | 44,52 | 82,48 |
| M2 | 70,33 | 75,36 | 30,42 | 76,58 |
| M3 | 67,72 | 73,11 | 35,47 | 79,17 |

M3 tăng tương đồng giải thích so với M2 nhưng giảm 2,61 điểm EM.
Chấm độc lập theo ảnh còn thiếu; chưa kết luận DPO cải thiện hỗ trợ từ ảnh.
[Báo cáo và giới hạn](docs/project_description.md) ·
[Giao thức](docs/experiment_protocol.md) · [Chấm theo ảnh](docs/human_evaluation.md).

Bộ preference có 992 rejected tổng hợp và tám output M1 nguyên văn;
kiểm text-only đạt 62,55%. Trang rà soát và kiểm metadata không thay
thẩm định chất lượng trực tiếp theo ảnh.

Demo M1 đã chạy ảnh mới trên T4. Guard chặn NaN/Inf và phục hồi được yêu
cầu tiếp; nguyên nhân lỗi số học còn cần chẩn đoán.

M4 là thí nghiệm mới: DPO cộng loss giữ đáp án đúng, cùng bộ cặp/ngân sách
với M3. Code và cấu hình đã thêm; chưa có bảng đánh giá chất lượng M4.
[Thiết kế M4](docs/answer_preservation.md).

## Các nhánh

| Mã | Cấu hình |
|---|---|
| M0 | Qwen2.5-VL-3B-Instruct gốc |
| M1 | SFT trên ViVQA-X |
| M2 | Từ M1, SFT tiếp trên giải thích tốt hơn (chosen) |
| M3 | Từ cùng M1, DPO trên cặp giải thích tốt hơn/kém hơn (chosen/rejected) |
| M4 | Thí nghiệm bổ sung: DPO có loss giữ đáp án, chưa có kết quả chất lượng |

M2/M3 dùng cùng câu/ảnh/chosen, seed, epoch và LR; không coi là cùng chi phí
tính toán. Lượt 186 cặp dùng LR 5e-6; bộ chính 1.000 cặp dùng 1e-5, ba epoch.
Test giữ kín tới khi khóa lựa chọn; ROUGE/BERTScore không thay đánh giá theo ảnh.

Seed điều khiển tính ngẫu nhiên, trong thiết lập này chủ yếu là thứ tự học
dữ liệu. Mỗi seed là một lần huấn luyện độc lập từ cùng trọng số M1, không
học nối từ seed trước; cả hai nhánh đều tắt dropout. Kế hoạch dùng ba seed
42, 43, 44 để đo mức dao động giữa các lần chạy; validation và test tự động
đã đủ. Ba seed này chỉ áp dụng cho phần học thêm từ cùng M1.

## Tổ chức

```text
src/vqa_explanation/     dữ liệu, train, suy luận, đánh giá và demo
scripts/                CLI
configs/                thuật toán và đường dẫn từng môi trường
tests/                  kiểm thử
notebooks/              04–08 nghiên cứu chính/demo; 09 thí nghiệm giữ đáp án
app/static/             giao diện demo
app/validation_explorer/ giao diện đối chiếu kết quả đã lưu
docs/, reports/         hướng dẫn, bảng và hình
data/, runs/, artifacts/ dữ liệu, checkpoint và bằng chứng ngoài Git
```

[Mục lục artifacts](artifacts/README.md) chỉ rõ bản duyệt, báo cáo pilot và
source release đang dùng. HTML/CSS/JavaScript của editor nằm riêng trong
`src/vqa_explanation/preference/templates/` và được nhúng khi xuất trang duyệt.
[Quy trình review](docs/code_review.md) mô tả các kiểm tra cần có trước khi đồng bộ.
Notebook `04_train_preferences.ipynb` chỉ điều phối huấn luyện M2/M3;
`05_evaluate_preferences.ipynb` đánh giá độc lập sau train;
`06_demo_live.ipynb` kiểm kỹ thuật demo trên GPU. Chọn record/seed
ở đầu, không tạo notebook theo từng lượt. Bản Colab đã có output giữ nguyên;
các template không thay lịch sử thực thi. Xem [hướng dẫn notebook](notebooks/README.md).
Bản M1 đã chạy giữ trong `notebooks/archive/`, gồm cả output tải từ Colab.
[Mở M1 có output](https://colab.research.google.com/drive/13Abjl7kJyDKzbxyYsCetJ540OzdK2P_E).

## Làm việc bằng Codex

Đọc `AGENTS.md`, trạng thái hiện hành trong `TIEN_DO.md` và
[harness](docs/harness.md) trước một đợt sửa. `scripts/task_harness.py status`
cho biết các việc đang dở; `start`, `check`, `review`, `finish` lưu phạm vi,
kết quả kiểm và bàn giao. Dùng Python của checkout như các lệnh bên dưới.
Quy trình tận dụng gate hiện có; không cần tài khoản Plus thứ hai hoặc chạy
GPU để kiểm code. Hồ sơ task không thay nhãn người hay kết quả mô hình.

## Local

Tại repo, dùng Python 3.10+ và môi trường riêng. Kiểm đầy đủ source cần
Torch/Torchvision CPU, các nhóm verification/demo/report và Node 22:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,verification,demo,report]"
node --version
.\.venv\Scripts\python.exe scripts/check_project.py --source-only
```

Máy hiện tại có annotation và 28.180 ảnh ViVQA-X tham chiếu, không tải lại.
Profile `configs/environments/local_full.yaml` dùng dữ liệu đầy đủ;
`local.yaml` dùng smoke 500 ảnh. Local không mount Drive hoặc train Qwen CPU.
Checkout mới: xem [dữ liệu](docs/data_protocol.md) và
[local/cloud](docs/local_cloud.md). Local chỉ kiểm trainer nhỏ, không train
Qwen trên CPU. Cài `.[dev]` riêng chỉ đủ cho một phần công cụ, không đủ cổng
kiểm đầy đủ. Gate dừng nếu thiếu dependency làm test bị skip.

Checkout Git/source ZIP dùng `--source-only` để không yêu cầu hồ sơ chạy
ngoài Git. Trên máy có đủ artifact lịch sử, bỏ cờ này để kiểm cả lịch sử.
Cả hai chế độ đều từ chối skip trong phạm vi đã chọn.

## Việc tiếp theo

1. Hoàn tất luyện tiêu chí trên 30 ảnh validation và chấm độc lập 200 ảnh
   test đã cố định; đưa kết quả và phân tích lỗi vào báo cáo.
2. Chẩn đoán lỗi số học bằng profile rõ ràng; nghiệm thu cấu hình hữu hạn
   bằng lượt riêng. Giữ benchmark đã khóa để tái lập.
3. Kiểm GPU nhánh M4, sau đó đánh giá trên validation theo
   [thiết kế bổ sung](docs/answer_preservation.md). Chọn tham số trên validation.
4. Hoàn thiện báo cáo theo mẫu trường, slide và bản demo được nghiệm thu.

Để tái tạo phân tích validation ở local, dùng hồ sơ prediction/semantic đã
lưu, không train hoặc gọi GPU:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_validation.py --environment configs/environments/local_full.yaml --output-dir reports/generated/validation_diagnostics_local
```

Thư mục đích phải chưa tồn tại. Mở `index.html` trong thư mục kết quả để
lọc seed, nhóm câu hỏi và các câu M3 khớp thêm/mất khớp đáp án.

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
