# SFT smoke: trạng thái và cách chạy

Ngày cập nhật: 2026-09-23.

## Đã kiểm tra local

- M0 trước đó đã hoàn thành 20 validation; không chạy lại/ghi đè ở bước này.
- Trainer mới: Qwen2.5-VL-3B-Instruct, revision cố định, QLoRA NF4.
- 32 câu hỏi train chọn bằng seed 20260918; mỗi câu dùng lời giải thích đầu tiên.
- 8 optimizer steps, batch 1, gradient accumulation 2; ảnh 256 visual tokens.
- LoRA chỉ gắn vào attention của language model; giữ nguyên vision encoder.
- FP16 trên T4, BF16 khi GPU hỗ trợ native.
- Prompt và token ảnh được mask khỏi loss. Template không khớp hoặc chuỗi quá
  dài thì dừng có lỗi, không cắt token ảnh ngầm.
- Kiểm tra trùng ảnh train/held-out, ID câu hỏi trùng và ảnh thiếu.
- Checkpoint mỗi 4 steps; adapter, log, phiên bản thư viện và hash source/data
  được ghi trong mỗi run riêng, không ghi đè run cũ.
- Kiểm tra loss hữu hạn và trọng số LoRA thay đổi sau huấn luyện.
- Preflight 32 câu hỏi và ảnh pass. Kiểm thử local hiện tại: 48 tests, Ruff pass.

## Kết quả GPU đã xác nhận

Run `M1_smoke_20260922_065719` trên Tesla T4 đã train đủ 8 bước:
loss trung bình 1,906347, trọng số LoRA thay đổi, VRAM đỉnh khoảng 4,33 GiB.
Nạp lại adapter và suy luận đã qua. Sau khi runtime mất, đọc lại Drive xác nhận
20 ID duy nhất, đủ adapter config/weights, EM 0,35, Token-F1 0,403333,
format 20/20, latency trung bình 5,8547 giây/mẫu.
Đây là kiểm tra pipeline, không phải thực nghiệm M1 cuối và chưa đo chất lượng
lời giải thích. Không so điểm trực tiếp với M0 cũ vì giới hạn độ phân giải khác.

## Chạy trên Colab

Mở `notebooks/archive/02_sft_smoke.ipynb` trên tài khoản Drive chứa
`MyDrive/Vietnamese_VQA_NLE` và chọn GPU T4. Chạy lần lượt các ô:

1. Mount Drive và xác nhận GPU.
2. Kiểm tra SHA-256 archive 500 ảnh, giải nén vào thư mục tạm mới, cài môi trường.
3. Chạy preflight và SFT 8 steps; đọc `smoke_result.json`.
4. Nạp lại adapter bằng tiến trình mới, chạy 20 validation, xuất metrics.

Notebook chứa source overlay để dùng với archive `source/project_source.zip`
đã có trên Drive. Code chuẩn vẫn nằm trong `src/` và `scripts/`; test
`test_sft_notebook.py` giữ fingerprint của overlay lịch sử và kiểm tra mã nhúng.
Overlay là mã của **run 22/09**, không còn giống source mới nhất trong repo;
không thay nó hoặc dùng notebook 02 để kết luận về code hiện hành. Mỗi run đã
lưu lại overlay để tái lập. Bản notebook 02 trên Colab đã có ô xác minh chạy
thành công; ngày 23/09 tải bản đó về local, giữ cả output thật và ghi chú hoàn tất.
Muốn đọc kết quả cũ thì mở notebook, không bấm “Chạy tất cả” vì thao tác đó tạo
run mới. Các thí nghiệm tiếp theo dùng source snapshot và CLI riêng.

Các đường dẫn bền vững:

```text
Vietnamese_VQA_NLE/runs/M1_smoke_<UTC timestamp>/
  source_overlay.json
  training/run_metadata.json
  training/checkpoint-4/
  training/checkpoint-8/
  training/adapter/
  training/smoke_result.json
  M1_smoke_validation.jsonl
  validation_metrics.json
```

Ảnh giải nén trong `/content/M1_smoke_<timestamp>/coco` là dữ liệu tạm,
sẽ mất khi runtime reset. Dữ liệu gốc và kết quả lưu trên Drive.

Tham khảo triển khai:
- https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl
- https://huggingface.co/docs/peft/developer_guides/quantization
