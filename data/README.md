# Data policy

Không đưa ảnh COCO, annotation đầy đủ, đầu ra do mô hình sinh hoặc cặp ưu tiên
chưa kiểm tra vào Git.

Repo chỉ giữ:

- schema và script xử lý;
- manifest không chứa dữ liệu nhạy cảm nếu được phép công bố;
- một vài bản ghi giả lập trong `data/sample/` để kiểm thử.

Dữ liệu local nằm trong `data/raw/` và `data/processed/` (ngoài Git).
Drive là bản lưu bền vững khi chạy cloud, không phải dependency của local.
Mỗi phiên bản dữ liệu cần có checksum, ngày tạo, script tạo và nguồn gốc.

## Bố trí hiện tại

```text
data/raw/vivqax/annotations/   ba file JSON gốc từ ViVQA-X
data/raw/coco/images/         28.180 ảnh COCO được ViVQA-X tham chiếu
data/processed/vivqax/        JSONL chuẩn hóa và stats.json
data/processed/vivqax/manifests/
                               danh sách 28.180 ảnh COCO cần lấy
data/processed/vivqax/image_disjoint_v1/
                               split dẫn xuất không trùng ảnh, có provenance
data/preferences/            pool câu hỏi và dữ liệu ưu tiên, ngoài Git
data/cache/coco_smoke_500.zip  gói ảnh smoke lịch sử
```

ZIP chuyển dữ liệu lên cloud nằm ở `artifacts/datasets/vivqax_image_disjoint_v1/`.
Ảnh đã giải nén và ZIP cùng tồn tại có chủ đích: một bản để dùng local, một bản
để khôi phục runtime cloud. Giữ annotation gốc và bản dẫn xuất riêng để truy vết.

Tái tạo toàn bộ dữ liệu dẫn xuất bằng:

```bash
python scripts/prepare_vivqax.py
```

Tải đúng các ảnh được ViVQA-X sử dụng (có thể chạy lại sau ngắt):

```bash
python scripts/download_coco_full.py
python scripts/verify_coco_full.py
python scripts/run_pipeline.py audit --environment configs/environments/local_full.yaml
```

Xem [giao thức dữ liệu](../docs/data_protocol.md) trước khi chọn split cho
thực nghiệm. Tạo bản `image_disjoint_v1` bằng
`python scripts/make_image_disjoint_split.py` nếu chưa có; script từ chối ghi đè
một phiên bản đã tạo. Không ghi đè ba JSONL chuẩn hóa gốc.

Google Drive dùng cấu trúc tương ứng dưới `Vietnamese_VQA_NLE/datasets/`:
`annotations`, `images`, `manifests`, `processed`, `preference` và `audit`.
