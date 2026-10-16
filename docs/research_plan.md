# Kế hoạch thực hiện

Chốt phạm vi ngày 14/10/2026. Đề tài: nghiên cứu cải thiện lời giải thích
trong hỏi đáp hình ảnh tiếng Việt.

## Mục tiêu và phạm vi

Từ cùng checkpoint M1, so sánh SFT bổ sung trên chosen với DPO trên
chosen/rejected: chất lượng giải thích theo ảnh có cải thiện không, đáp án
có bị ảnh hưởng không, và kết quả có ổn định qua các seed không?

M0–M3 là bốn cấu hình của Qwen2.5-VL-3B, không phải bốn kiến trúc.
Giữ M1-3683 và split ViVQA-X hiện có: 29.459 train, 1.407 validation,
1.968 test. Không train lại M1, không dùng test để chọn thiết kế.

Bộ nghiên cứu chính dùng 1.000 cặp, có thể tận dụng cặp v1 sau khi kiểm lại.
Không hạ tiêu chuẩn chất lượng để đạt số lượng.

## 1. Kết thúc lượt nhỏ, không biến thành kết quả chính

Lượt 186 cặp đã train M2/M3, mỗi nhánh một epoch / 24 update. Giữ nguyên
record, tên run, checkpoint và notebook có output; không đổi nhãn lịch sử.
Lượt này dùng để kiểm triển khai và tham khảo nội bộ. Không dùng làm bảng
kết quả chính hoặc bằng chứng DPO tốt hơn trong báo cáo cuối.

Nếu validation đang chạy, để hoàn tất và lưu prediction. Nếu bị ngắt,
chỉ nối tiếp lượt đó, không train lại hoặc kéo dài epoch của adapter cũ.
Không ưu tiên hoàn thành các phép đo phụ của lượt nhỏ trước bộ dữ liệu mới.
M0/M1 validation được dùng lại khi identity suy luận khớp; không trộn
bảng từ môi trường khác nhau. Test vẫn đóng.

## 2. Bộ preference dùng cho nghiên cứu chính

Mục tiêu 1.000 cặp, một câu/ảnh train. Đây là ngân sách của dự án,
không phải ngưỡng khoa học bảo đảm chất lượng.
Không lặp ảnh, giữ cặp yếu hoặc chuyển validation/test vào train để đủ số.

Làm từng pool 2.000 ảnh train mới; tối đa ba pool trước khi rà lại năng suất.
Loại toàn bộ ảnh của các pool đã duyệt khỏi pool tiếp theo bằng manifest,
không chỉ loại những cặp đã giữ. Không sinh lại pool 2.000 cũ. Có thể dùng
lại một cặp cũ nếu qua tiêu chuẩn mới, nhưng cần review mới gắn đúng hash.
Ưu tiên dùng lại ứng viên M1 cũ đã lưu để sửa cặp theo tiêu chuẩn mới,
không cần chạy M1 lại cho những ảnh đó.

Quy tắc giữ cặp:

- Đáp án hai phía giống nhau và đúng theo ảnh.
- Chosen nêu bằng chứng nhìn thấy được, liên quan trực tiếp đến câu hỏi.
- Rejected có một sai khác xác định được theo ảnh: thuộc tính, vị trí,
  số lượng, quan hệ/hành động hoặc đối tượng có thể xuất hiện trong ngữ cảnh.
- Hai câu đều tự nhiên và hợp lý khi chưa thấy ảnh. Không dùng đối lập
  như “đáp án bóng chày nhưng cầm vợt tennis” để học kiến thức ngôn ngữ.
- Ghi bằng chứng ảnh và cụm sai. Không giữ cặp chỉ khác văn phong,
  hai phía đều đúng, hoặc ảnh/đáp án tham chiếu không chắc chắn.

Ưu tiên lỗi M1 tự sinh. Negative có kiểm soát được phép, nhưng chỉ sửa
chi tiết thị giác; lưu nguồn thật trong metadata. Báo phân bố model/synthetic,
loại lỗi và độ dài; không áp quota loại lỗi làm giảm chất lượng.
Khi sửa lỗi M1, giữ nguyên rejected và chỉ thay cụm sai để tạo chosen;
không viết lại cả hai phía rồi coi là lỗi nguyên văn của M1. Output sai
đáp án hoặc định dạng không dùng làm negative M1 cho thiết kế cùng đáp án.

`scripts/audit_preference_quality.py` đo một baseline xếp hạng chỉ đọc
văn bản, cross-validation tách theo ảnh. Đây là phép sàng lọc shortcut,
không phải điểm Qwen hoặc chứng minh mô hình bỏ qua ảnh. Điểm thấp cũng
không thay được review ảnh. Review chất lượng dùng `quality_review.jsonl`,
gắn hash nội dung mỗi cặp; không tự chuyển bản pending thành confirmed.
Review v2 ghi lý do riêng cho độ liên quan và phép kiểm chỉ chữ; không dùng
một bộ cờ xác nhận chung cho cả lô. Khi audit ảnh phát hiện nhãn sai,
giữ snapshot cũ, đóng khóa train và xuất bản sửa sau khi duyệt lại.
Phân biệt cặp sửa bằng chứng trả lời với cặp chỉ sửa thuộc tính phụ của
đối tượng; nhóm sau đo tính đúng của chi tiết, chưa đủ để kết luận về lập luận.

CLI `expand_preferences.py` có nhánh sinh/sàng lọc bằng teacher 7B, kiểm ảnh
hai thứ tự A/B và kiểm chỉ chữ. Lô đầu vẫn lọt cặp đồng nghĩa, nên không
dùng kết quả teacher để tự khóa dữ liệu. Đợt 15/10 dùng lại ứng viên M1 đã
lưu, xem trực tiếp từng ảnh và ghi lý do riêng; sau đó kiểm chéo 100 cặp
và rà nhóm lỗi tương tự. Trang kết quả chỉ ghi nội dung/kết quả; nguồn thật
giữ riêng trong metadata. Đây không phải đánh giá của nhiều người độc lập.

Sau tối đa ba pool: nếu đủ 1.000 cặp và audit qua thì khóa N=1.000.
Nếu chưa đủ, giữ lại cặp đạt yêu cầu và rà nguồn negative/năng suất duyệt;
không tự giảm mục tiêu hoặc khóa một bộ thiếu cặp.
Không tiếp tục tăng N sau khi đã xem kết quả test.

Nghiệm thu: bundle frozen qua kiểm canonical/split/hash, tất cả cặp có
review chất lượng hợp lệ, audit độ khó/độ dài/nguồn đã đọc, M2 chosen-only
khớp đúng chosen của M3. Đáp án đúng được kiểm theo canonical trong exporter;
kiểm metadata không tự chứng minh mọi quyết định thị giác là đúng.

## 3. Huấn luyện chính

Cấu hình dự kiến ở `configs/experiments/preference_study_v2.yaml` và hai
config `sft_chosen_full.yaml`, `dpo_full.yaml`. Ngày 15/10 đã khóa N=1.000
và hash review tại `frozen_1000_reviewed_20261015`. Cổng dữ liệu vẫn kiểm
đúng bundle/review; mỗi nhánh phải qua kiểm GPU trước khi train chính.
Không thay config hoặc source của run 186 cặp đang dùng trên Colab.

| Thiết lập | SFT chosen / DPO |
|---|---|
| Khởi đầu | Cùng M1-3683; reference DPO là M1 frozen |
| Dữ liệu | Cùng ID, ảnh, đáp án, chosen và thứ tự shuffle theo seed |
| Epoch | 3 cho cả hai nhánh |
| Learning rate | 1e-5 cho cả hai nhánh |
| Seed | 42, 43, 44; tổng sáu run chính |
| Batch hiệu dụng | 1 × tích lũy 8, một GPU |
| DPO beta | 0,1 |
| Ảnh / guard | 200.704 pixels, 1.536 token; không cắt ảnh/completion |
| Checkpoint báo cáo | Cuối epoch 3; không chọn riêng checkpoint đẹp cho một nhánh |

Đây là thiết kế đối chứng có ngân sách, chưa phải bộ siêu tham số tối ưu.
Với N=1.000: 375 update/nhánh. Báo cả số mẫu duy nhất và số lượt đọc mẫu:
ba epoch không tạo thêm dữ liệu.
Cùng update/epoch không có nghĩa cùng FLOPs; ghi giờ GPU/VRAM riêng.

Mỗi seed có study record và config riêng, được tạo trước train; không sửa
config đang chạy. Không sinh notebook mới khi đổi seed hoặc mất phiên.
CLI xử lý trainer, notebook chỉ setup → kiểm đầu vào → gọi run → hiển thị.
Chỉ chạy gate ngắn khi dữ liệu/source/config/runtime đổi; không lặp smoke
trong phiên không đổi. Resume đầy đủ optimizer/scheduler/RNG theo ngân sách
gốc, giữ run/checkpoint cũ.

Số đo DPO lượt nhỏ khoảng 21 giây/update chỉ để dự toán ban đầu: 375 update
khoảng 2,2 giờ/seed, chưa tính setup và thay đổi độ dài. Gate bộ mới đo lại;
nếu một run dự toán quá 8 giờ thì xử lý tốc độ/môi trường trước chạy đầy đủ.
Không tăng epoch chỉ vì train loss giảm.

## 4. Đánh giá đúng câu hỏi nghiên cứu

1. Xuất đủ 1.407 validation cho sáu run chính và M0/M1 trong môi trường
   ghim. Báo chênh lệch M3–M2 cho từng seed; không chọn seed đẹp nhất.
2. Trên 200 ảnh validation cố định, đổi ảnh theo hoán vị không điểm cố định
   cho M1/M2/M3 seed 42; giữ câu hỏi và generation. Báo thay đổi đáp án,
   EM so với đáp án của ảnh gốc và output. Đây là kiểm độ nhạy, không phải
   benchmark ảnh sai hay chứng minh lời giải thích phản ánh suy luận nội bộ.
3. Một đối chứng độ nhạy LR, seed 42, LR 5e-6 cho cả hai nhánh, cùng dữ liệu
   và ba epoch; giữ beta 0,1. Tổng thêm hai run, bảng riêng, không dùng để
   thay thiết kế chính sau khi thấy test. Không mở grid search tiếp.
4. Khóa mọi adapter, prompt, generation và rubric; xuất đủ 1.968 test cho
   M0/M1 và sáu run chính. M0/M1 chỉ xuất một lần nếu identity khớp.
5. Chấm mù 200 ảnh test đã cố định, dùng M0/M1/M2/M3 seed 42, không chọn
   seed bằng điểm test. Hai người chấm độc lập; luyện rubric trên 30 ảnh
   validation trước. Nếu chỉ có một người thì báo đúng một người, không
   tạo agreement hoặc gán quyết định tự động thành nhãn người.

Chỉ số chính: chênh lệch hỗ trợ theo ảnh 0–2 giữa M3 và M2 trên toàn mẫu.
Bổ sung: đáp án đúng kèm hỗ trợ 2, chi tiết không được ảnh hỗ trợ và uncertain.
EM/F1 đo đáp án; ROUGE-L/BERTScore đo tương đồng văn bản, không thay phần chấm
theo ảnh. Giữ toàn mẫu trong mẫu số, kể cả output sai định dạng/đáp án sai.

Tự động báo từng seed và mean ± SD qua ba seed. Paired image-bootstrap
2.000 lần, CI 95%, đo bất định tập ảnh; không thay SD qua seed. Chấm mù chỉ
seed 42 phải được ghi rõ, không suy CI người chấm thành kết quả của ba seed.
Phân tích lỗi theo loại, nguồn negative, độ dài; báo cả trường hợp DPO không
cải thiện. Không lọc ví dụ test xấu ra khỏi bảng.

## 5. Demo và báo cáo

Giữ khung Word và code demo hiện có. Hoàn thiện luồng nhận ảnh/câu hỏi mới,
trả đáp án và lời giải thích, lựa chọn adapter đã nghiệm thu. Local chạy
giao diện/API; model chạy GPU, không train Qwen trên CPU.

Nghiệm thu live: ít nhất ba ảnh/câu chưa dùng trong replay, có identity
adapter, thời gian suy luận, xử lý lỗi, reload và endpoint xác thực. Replay
không được trình bày thành live. Lưu một bộ ảnh/bảng kết quả cuối để báo cáo.

Báo cáo chính: bài toán → dữ liệu → phương pháp → thiết kế đối chứng →
kết quả đa seed → chấm mù/phân tích lỗi → demo → giới hạn. Không đưa toàn bộ
lịch sử setup, smoke, hash hoặc lượt 186 cặp vào phần kết quả. Hồ sơ tái lập
vẫn giữ để kiểm khi cần; không xóa kết quả bất lợi hoặc đổi số liệu lịch sử.

Đóng góp là dữ liệu preference theo bằng chứng ảnh và nghiên cứu thực nghiệm
SFT–DPO cho tiếng Việt. Không tuyên bố tạo thuật toán DPO mới hoặc đã kiểm
khả năng khái quát trên nhiều backbone.

## Tiến độ và quy tắc dừng

| Thời gian dự kiến | Sản phẩm phải có |
|---|---|
| Tuần 1–2 | Bộ cặp chính frozen, audit chất lượng, cấu hình sáu run ghim |
| Tuần 3 | Sáu run và validation đủ; sửa sự cố có bằng chứng, không đổi mục tiêu |
| Tuần 4 | Đổi ảnh, LR sensitivity, test đủ sau khóa thiết kế |
| Tuần 5 | Chấm mù thật, CI/SD, phân tích lỗi |
| Tuần 6 | Live demo và cập nhật báo cáo cũ |
| Tuần 7–8 | Slide, kiểm tái lập, sửa báo cáo và dự phòng |

Đây là ngân sách kế hoạch, không bảo đảm GPU miễn phí liên tục. Tạm bỏ
backbone thứ hai, dữ liệu ngoài, augmentation và ablation cỡ mẫu khỏi phạm
vi bắt buộc. Chỉ mở thêm một nghiên cứu phụ khi các sản phẩm chính đã xong.

Chỉ lưu snapshot notebook ở cuối train/evaluation hoặc trước thao tác có
nguy cơ làm mất output. CLI tự lưu tiến độ; không tải notebook sau từng cell.
Một source release sau một mốc đã qua kiểm thử, không release sau mỗi sửa
comment. Mỗi mốc ghi một mục ngắn trong TIEN_DO, cập nhật trạng thái cùng
tài liệu; không thêm note hay khung báo cáo mới. Đồng bộ Drive phải kiểm file
thật; chưa upload thì ghi local-only.

## Cơ sở thiết kế

[DPO](https://arxiv.org/abs/2305.18290) là cơ sở loss/reference;
[mDPO](https://arxiv.org/abs/2406.11839) chỉ ra rủi ro ưu tiên ngôn ngữ khi
học preference đa phương thức; dự án này không triển khai nguyên mDPO.
[e-ViL](https://arxiv.org/abs/2105.03761) là cơ sở kết hợp đo tự động và
đánh giá lời giải thích bởi người. Các giá trị N/epoch/LR ở trên là lựa chọn
thực nghiệm của dự án, không phải yêu cầu từ những bài báo đó.
