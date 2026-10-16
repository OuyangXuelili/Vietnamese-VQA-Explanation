# Kiểm tra trước khi so sánh điểm

M1-3683 đã được chọn theo loss trên 256 câu validation cố định. M0 và M1
đã xuất đủ 1.407 câu validation. Các bước tiếp theo nằm trong
[kế hoạch](research_plan.md); kết quả hiện có ở [mô tả dự án](project_description.md).

## Đầu vào của phép so sánh

`scripts/compare_runs.py` ghép hai file theo `question_id` và kiểm:

- Đủ toàn bộ ID thuộc split đánh giá, mỗi ID xuất hiện một lần.
- Cùng ảnh/hash, câu hỏi, đáp án và giải thích tham chiếu.
- Cùng model revision, prompt, generation config, dtype và preprocessing.
- Đúng nhánh M0–M3 và adapter: M0 dùng backbone gốc; các nhánh đã học có adapter.

Truyền annotation và SHA-256 của split chuẩn để kiểm full coverage. File thiếu
ID hoặc khác giao thức trả kết quả `comparable=false` và mã thoát 2. Dữ liệu
đang sinh dở cần hoàn tất bằng exporter trước khi chấm.

## Validation và test

Validation có 1.407 câu; test có 1.968 câu. Khóa adapter, prompt và cấu hình
sinh trước khi xuất test. M2/M3 được so với nhau bằng cùng bundle preference,
M1 cha và giao thức đã định trong [experiment_protocol.md](experiment_protocol.md).

Khi so hai checkpoint M1 trên validation, truyền:

```text
--left-experiment M1_SFT --right-experiment M1_SFT --checkpoint-comparison
```

Dùng adapter khác nhau và các cờ annotation/hash đầy đủ. Các tùy chọn CLI
hiện hành xem bằng `python scripts/compare_runs.py --help`.

## Chỉ số và khoảng tin cậy

EM và Token-F1 đo đáp án sau chuẩn hóa. ROUGE-L đo tương đồng giải thích;
BERTScore có report riêng với model/revision đã ghim. Dùng paired bootstrap
theo ảnh, 2.000 lượt và seed 42, để mô tả bất định mẫu của chênh lệch hai run.
Báo đơn vị chỉ số và CI 95% trong bảng.

Mức hỗ trợ giải thích theo ảnh được chấm riêng theo
[human_evaluation.md](human_evaluation.md). Các chỉ số văn bản bổ sung cho
phần chấm này. Một seed train và bootstrap theo ảnh chỉ đo được các nguồn
bất định tương ứng của thiết kế hiện tại.

## Kết quả và tái lập

Bảng M0/M1 hiện hành ở [mô tả dự án](project_description.md#kết-quả-m0m1).
Báo cáo đầy đủ nằm trong `reports/generated/m0_m1_validation_20261011/`
của kho kết quả, ngoài Git.
Các so sánh checkpoint M1 ở `artifacts/m1_validation_analysis_20261007`.

Lịch sử ngân sách, checkpoint-600/800/1200 và lệnh của các phiên cũ được giữ
trong [bản trước khi rút gọn](archive/comparison_gate_before_cleanup_20261012.md).
Dùng source và môi trường ghim trong manifest khi tái lập một run lịch sử.
