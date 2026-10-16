# Giao thức dữ liệu ViVQA-X

## Hai phiên bản, không ghi đè nhau

`data/processed/vivqax/{train,validation,test,all}.jsonl` là bản chuẩn hóa
theo split gốc. Bản gốc có 32.886 câu và 28.180 ảnh; train không trùng ảnh
với held-out, nhưng validation và test dùng chung 48 ảnh. Giữ bản này để
truy xuất nguồn gốc, không dùng trực tiếp để tuyên bố điểm đánh giá cuối.

`data/processed/vivqax/image_disjoint_v1/` là bản dẫn xuất cho thực nghiệm
chính. Chính sách được quyết định trước khi xem điểm mô hình: giữ nguyên train
và test, chỉ bỏ khỏi validation mọi câu hỏi thuộc 48 ảnh chung với test.
Kết quả: train 29.459, validation 1.407, test 1.968 câu; tổng 32.834. Có
52 câu validation bị loại. `provenance.json` lưu hash đầu vào, ID ảnh bị
loại, chính sách và audit. `images.csv` vẫn liệt kê đủ 28.180 ảnh, vì các
ảnh bị bỏ khỏi validation vẫn thuộc test.

Không dùng test để chọn prompt, checkpoint, tham số, rubric hay phiên bản
chia tập. Nếu báo cáo kết quả trên split gốc để so với công trình trước,
phải ghi rõ 48 ảnh validation/test chung và không trộn điểm của hai giao thức.

## Ảnh và kiểm tra

`scripts/download_coco_full.py` chỉ tải ảnh trong manifest ViVQA-X, không tải
toàn bộ COCO. File JPEG hợp lệ đã có được bỏ qua; chạy lại sau gián đoạn sẽ
tiếp tục, báo cáo ở `data/processed/vivqax/full_download.json`. Ảnh nằm ở
`data/raw/coco/images/`, bị loại khỏi Git. Sau khi tải xong phải chạy audit
`--verify-images --strict` trên `image_disjoint_v1/all.jsonl`. Chỉ khi số ảnh
thiếu/hỏng đều bằng 0 và không còn overlap mới mở rộng train/đánh giá.
Profile `configs/environments/local_full.yaml` trỏ vào bản dẫn xuất; nó không
tự biến trainer 8 bước thành quy trình full SFT. `local.yaml` vẫn dùng bộ smoke.

Ngày 23/09 đã xác minh local: `full_download.json` hoàn tất 28.180/28.180;
`full_image_verify.json` báo thiếu/hỏng/thừa đều 0, tổng 4.486.600.417 byte.
`full_image_checksums.csv` ghi SHA-256 từng ảnh và có SHA-256 toàn file
`26aae5559f66a951c0c89bb8991e6e096d44aa88f895720d88076f0f917380b6`.
Audit strict có kiểm tra ảnh ở `runs/audit_20260923_014156_678017/audit.json`
không lỗi/không overlap; preflight profile full qua. Ngày 23/09 đã upload
bundle lên Drive và xác minh lại trên Colab; xem mục dưới.

Không đồng bộ hàng chục nghìn ảnh lên Drive bằng tay. Đối với Colab, đóng
gói/đưa dữ liệu theo manifest có kiểm tra hash và đủ dung lượng cloud trước
khi chạy; `/content` là vùng tạm, Drive giữ checkpoint, log và đầu ra lâu dài.

## Bundle để chuyển lên Colab

`scripts/package_coco_full.py` tạo ba ZIP không nén lại JPEG trong
`artifacts/datasets/vivqax_image_disjoint_v1/`: hai archive ảnh theo nguồn
COCO 2014 và một archive annotation/provenance/checksum. `bundle.json` chứa
SHA-256 và kích thước từng ZIP; `bundle.sha256` khóa manifest. Script đọc
ngược mọi entry để so hash trước khi báo hoàn tất. Bundle local đã tạo và
`scripts/verify_coco_bundle.py` báo `verified`, 28.180 ảnh. Bản cùng manifest
đã upload lên `Vietnamese_VQA_NLE/datasets/releases/vivqax_image_disjoint_v1/`
và được xác minh hash lại trong Colab.

Trên Colab, sau khi upload nguyên thư mục bundle lên Drive và mount đúng tài
khoản, dùng source snapshot cùng phiên bản để chạy:

`notebooks/archive/05_full_data_preflight.ipynb` điều phối chính các bước dưới đây,
kiểm tra thêm SHA-256 `bundle.json` cố định từ local, và lưu log audit/preflight
lên Drive. Notebook đã chạy xong bốn ô trên Colab ngày 23/09; không chạy train.

```bash
python scripts/verify_coco_bundle.py \
  /content/drive/MyDrive/Vietnamese_VQA_NLE/datasets/releases/vivqax_image_disjoint_v1 \
  --extract-to /content/vqa_work/vivqax_image_disjoint_v1
python scripts/run_pipeline.py audit \
  --environment configs/environments/colab_full.example.yaml
python scripts/run_pipeline.py preflight \
  --environment configs/environments/colab_full.example.yaml \
  --all-train-records
```

Verifier đối chiếu hash các ZIP trước khi giải nén và từ chối ghi đè thư mục
đã tồn tại. M1 đã hoàn tất. Các lệnh trên tái lập bước kiểm dữ liệu trước train;
kết quả hiện có và các bước còn lại ở [mô tả dự án](project_description.md).
`preflight --all-train-records` kiểm tra đường dẫn ảnh của cả 29.459 câu train;
chưa kiểm tra VRAM, token length hay forward/backward trên GPU.

Run cloud: `runs/audit_20260923_042209_005946/` và
`runs/preflight_20260923_042241_971811/` trên Drive. Hai `execution.json`
đều `passed`, `returncode=0`. `audit.json` ghi 32.834 bản ghi, ba split không
trùng ảnh, `errors=[]`; `console.log` preflight ghi 29.459 câu train có ảnh.
Đây là mốc kiểm tra dữ liệu, chưa phải thí nghiệm mô hình.
