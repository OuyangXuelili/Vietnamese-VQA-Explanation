# Giữ độ đúng đáp án khi học preference

## Thiết kế

Nghiên cứu bổ sung dùng cùng M1-3683 và bộ 1.000 cặp đã khóa của nghiên
cứu chính. M2 học chosen, M3 dùng DPO chuẩn, M4 dùng:

`loss = DPO + 0.1 × NLL(answer)`

NLL lấy trung bình trên các token giao với vùng chữ của đáp án trong
chosen. Prompt, token ảnh, nhãn định dạng và explanation không thuộc vùng
giám sát bổ sung. Offset phải khớp toàn bộ chuỗi token sau mở rộng ảnh;
không tìm đáp án bằng so khớp một chuỗi token riêng.

M4 dùng LR 1e-5, batch 1 × accumulation 8, ba epoch / 375 update, beta 0,1,
dropout tắt và reference M1 cố định. Hiện khai báo seed 42; chỉ mở các seed
còn lại sau khi cấu hình qua GPU và có kết quả validation. Đây là một
giả thuyết can thiệp vào suy giảm đáp án, chưa có bằng chứng cải thiện.

Nguồn dữ liệu không đổi trong đối chứng đầu tiên. Bộ cặp hiện có nhiều
rejected tổng hợp và chưa được người thẩm định độc lập; phải giữ giới hạn
này khi diễn giải. Cải thiện dữ liệu cần là một đối chứng riêng cùng N.

Việc phối hợp preference loss với NLL có cơ sở từ
[Iterative RPO](https://arxiv.org/html/2404.19733v3). Phần chỉ neo token
đáp án là điều chỉnh cho VQA, không phải triển khai nguyên vẹn thuật toán
của bài báo. Bài gốc nêu giới hạn khi dữ liệu không chủ yếu do mô hình
trước đó sinh ra.

## Code và cấu hình

| Thành phần | Đường dẫn |
|---|---|
| Study | `configs/experiments/answer_preservation.yaml` |
| M4 | `configs/training/dpo_answer.yaml` |
| Mask đáp án | `training/sft_data.py` |
| DPO và answer NLL | `training/preference.py` |
| CLI train | `scripts/train_preference.py` |
| Launch Colab | `scripts/launch_preference_training.py --arm m4` |

Kiểm đối chứng không cần model hoặc GPU:

```bash
python scripts/train_preference.py --config configs/training/dpo_answer.yaml --study-config configs/experiments/answer_preservation.yaml --check-comparison
```

Notebook 09 stage gói ảnh dành cho preference: 1.000 ảnh train và các
control/regression cần cho kiểm số học. Annotation vẫn đầy đủ và giữ hash
gốc; từng ảnh đối chiếu với bảng checksum canonical trước khi đóng gói.
Gói này giảm I/O setup, không lấy bớt cặp train. Environment riêng tại
`configs/environments/colab_preferences.example.yaml` không dùng để audit
toàn bộ kho ảnh. `scripts/stage_preference_data.py package --include-validation`
chuẩn bị thêm ảnh validation khi cần xuất chất lượng đầy đủ.

Launch record mới phải trỏ tới study/config mới và source release đã
kiểm, dùng `scope: answer_preservation_experiment`, khai báo `m4_config`.
Notebook 09 dành riêng cho M4, gồm 13 ô / tám ô code và gọi CLI hiện có.
Checksum launch được lấy từ receipt; template để trống và mặc định chưa
chạy GPU. Giữ các notebook đã có output của nghiên cứu chính.
Không sửa launch record, source hoặc output của nghiên cứu cũ.

Trước main train, thực hiện một gate năm update với đúng dữ liệu, M1,
source, precision và GPU. Gate phải xác nhận policy cập nhật, reference
không đổi, adapter lưu khớp và loss hữu hạn. Main train yêu cầu receipt
gate khớp toàn bộ identity; resume không được đổi weight hoặc source.

Trainer ghi riêng `loss_dpo`, `loss_answer_nll`, `answer_nll_weight` trong
log history. Thành phần NLL dùng lại forward chosen, không thêm một lượt
chạy vision. Kiểm CPU dùng model nhỏ để xác nhận gradient và resume;
chưa thay được kiểm Qwen thật trên GPU.

## Đánh giá

Xuất M1/M3/M4 bằng cùng profile `configs/inference/legacy_fp16.yaml`,
cùng ảnh, prompt, ngân sách token và source. Dùng
`scripts/run_m0.py --experiment-id M4_DPO_ANSWER --loading-profile ...`
với adapter M4 và các tham số export hiện có. Export mới lưu hash profile,
cấu hình nạp mô hình thực tế và fingerprint bộ xử lý ảnh. Resume hoặc
so sánh từ chối trộn cấu hình khác nhau.

`scripts/compare_runs.py` nhận M4 với canonical validation đầy đủ, hỗ trợ
paired CI và semantic reports. Bộ benchmark/chấm mù bốn nhánh cũ giữ
danh sách M0–M3. M4 hiện chỉ được xuất và so sánh trên validation.

Ưu tiên độ đúng đáp án và mức giải thích được ảnh hỗ trợ, cùng tỷ lệ
chi tiết không có trong ảnh. Log loss, ROUGE-L hoặc BERTScore tăng chưa
đủ kết luận cải thiện grounding. Cần nhãn theo ảnh và báo bất định.

Test cũ đã được xem khi đặt giả thuyết này. Muốn kết luận xác nhận cho
phiên bản nâng cấp, cần khóa thêm holdout trước khi xem output mới.
Không dùng test cũ để chọn lambda, seed hoặc checkpoint.

## Chẩn đoán NaN và profile suy luận

| Profile | Precision | Double quantization | Vai trò |
|---|---|---|---|
| `legacy_fp16` | FP16 | Tắt | Đường inference của benchmark cũ |
| `matched_fp16` | FP16 | Bật | Kiểm một thay đổi quantization |
| `diagnostic_fp32` | FP32 | Tắt | Kiểm một thay đổi precision |

Train giữ double quantization bật. Các profile inference được khai báo
tường minh; chưa chọn profile mới làm chuẩn khi chưa đo. Demo đọc
precision từ manifest đã chọn, giữ FP16 ngay cả trên GPU hỗ trợ BF16.

Kiểm đầu vào của phép chẩn đoán:

```bash
python scripts/diagnose_inference.py --environment configs/environments/local_full.yaml
```

Chạy trên GPU với environment đúng đường dẫn dữ liệu:

```bash
python scripts/diagnose_inference.py --environment <environment.yaml> --execute --output-dir <new_output_dir>
```

Bộ ca cố định gồm ảnh gây NaN đã biết và các control trước/sau lỗi.
Mặc định dùng M0 vì lỗi đã tái hiện ở backbone; có thể thêm `--adapter`
để xác nhận trên M1. Các probe tám token chỉ chẩn đoán phép tính, không
tạo prediction tính điểm và không tự đánh dấu đã sửa nguyên nhân.
Profile có probe hữu hạn vẫn phải qua kiểm generation đầy đủ, VRAM,
latency và hồi quy M1 trước khi triển khai.
