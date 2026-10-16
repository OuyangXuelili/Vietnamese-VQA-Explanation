# Chấm lời giải thích theo ảnh

Mẫu cuối đã đóng băng trước khi sinh prediction test: 200 ảnh khác nhau,
mỗi ảnh một câu, seed 42. File local:
`artifacts/human_evaluation/test_sample_200_seed42_v1.json`.
SHA-256: `8651cdfd737a07b5cfd401d595a45cda16adb9ba241864a9546b3f860a3330df`.
Chưa có prediction test hoặc nhãn chấm thật. Chỉ chọn ID từ annotation, không
chọn theo chất lượng đầu ra. Giữ nguyên mẫu sau khi đóng băng.

## Chuẩn bị sau khi có kết quả bốn nhánh

CLI ở `scripts/blind_review.py`; logic ở `evaluation/blind_review.py`.
M2/M3 phải xong, cấu hình và rubric phải chốt trước khi xuất test.

```powershell
python scripts/blind_review.py prepare --annotations data/processed/vivqax/image_disjoint_v1/all.jsonl --sample artifacts/human_evaluation/test_sample_200_seed42_v1.json --environment configs/environments/local_full.yaml --prediction M0_PRETRAINED=<M0_TEST_JSONL> --prediction M1_SFT=<M1_TEST_JSONL> --prediction M2_SFT_CHOSEN=<M2_TEST_JSONL> --prediction M3_DPO=<M3_TEST_JSONL> --output-dir artifacts/human_evaluation/blind_test_v1
```

Lệnh kiểm đủ canonical test và cùng giao thức suy luận, sau đó kiểm hash ảnh
thực tế trước khi tạo thư mục. Không lấy giao tập khi thiếu prediction.
Hai trang rater1/rater2 xáo vị trí model riêng cho mỗi câu. Trang và file nhãn
không chứa tên experiment, model, checkpoint hoặc đáp án tham chiếu. Output
được giữ nguyên; nếu model tự nhắc tên trong văn bản thì vẫn có thể lộ danh tính.

Chỉ đưa thư mục `images` và thư mục của người chấm cho người đó.
Giữ `private/model_key.jsonl` và `private/manifest.json` ở phía nghiên cứu;
không gửi khóa model cùng trang chấm. Hai người chấm độc lập, không xem nhãn
của nhau trước khi xuất file đầu tiên. Mỗi người có 800 đầu ra (200 × 4).

## Ghi nhãn và lưu

Mở `rater1/index.html` hoặc `rater2/index.html`. Ghi người chấm, trạng thái,
đáp án đúng/sai/mơ hồ, hỗ trợ 0/1/2, chi tiết không được ảnh hỗ trợ có/không/
không đủ bằng chứng, và ghi chú ảnh. Xuất JSONL thường xuyên; nhập lại để
tiếp tục. Trang mới tự lưu khi nhập, khôi phục nháp theo đúng nguồn và rater,
và cảnh báo rời trang khi chưa xuất. Nháp chưa đủ trường vẫn được giữ nhưng
không thể xuất thành nhãn hoàn chỉnh. Khi trình duyệt không cho lưu, trang
báo lỗi để xuất JSONL ngay. Nháp chỉ ở cùng trình duyệt/origin; không thay
file lưu bền vững. Reviewer là một người cố định mỗi file.

Tính năng này áp dụng cho trang tạo bằng source mới. Không ghi đè HTML hay
manifest của bộ chấm lịch sử; nếu cần trang mới thì tạo bundle riêng.

```powershell
python scripts/blind_review.py score --bundle artifacts/human_evaluation/blind_test_v1 --rating rater1=<RATER1_JSONL> --rating rater2=<RATER2_JSONL> --output artifacts/human_evaluation/results_v1.json
```

Scorer kiểm hash bundle, đủ mọi ID, trường nguồn bất biến và nhãn đã hoàn tất.
Báo riêng từng người: mẫu số, phân bố đúng/sai/mơ hồ, hỗ trợ trên toàn mẫu và
nhóm đáp án đúng, tỷ lệ đáp án đúng kèm hỗ trợ 2, phân bố chi tiết không được
ảnh hỗ trợ. Paired bootstrap theo ảnh (2.000 lần, seed 42) báo M3–M2 cho:

- Điểm hỗ trợ 0–2: chỉ số chính, chênh lệch dương là tăng hỗ trợ.
- Tỷ lệ đúng đáp án và hỗ trợ 2: chỉ số bổ sung, mẫu số là toàn bộ mẫu.
- Tỷ lệ nhãn `yes` về chi tiết không được ảnh hỗ trợ, và tỷ lệ `uncertain`
  tương ứng: báo cả hai, không bỏ câu mơ hồ khỏi mẫu số.

Các tỷ lệ trong JSON nằm trong [0, 1]; nhân 100 khi trình bày chênh lệch theo
điểm phần trăm. `yes` giảm nhưng `uncertain` tăng không đủ để kết luận model
bớt tạo chi tiết không được ảnh hỗ trợ. Khoảng tin cậy là từng chỉ số riêng,
không phải kiểm định nhiều giả thuyết đã hiệu chỉnh. Không chọn chỉ số phụ
có kết quả đẹp nhất làm kết luận chính sau khi chấm.

Kappa thường dùng cho đáp án/chi tiết không được ảnh hỗ trợ; kappa trọng số
bình phương dùng cho hỗ trợ 0/1/2. Kappa null khi nhãn suy biến làm mẫu số bằng
không. Một người chấm thì không có agreement. Metadata không chứng minh hai
người độc lập; việc thực hiện phải đúng như protocol đã ghi.
Giữ nhãn độc lập, nếu phân xử bất đồng thì lưu vào artifact mới. Bootstrap
phản ánh bất định mẫu đánh giá, không thay thế nhiều seed train.
