# Kế hoạch thực hiện

Chốt phạm vi ngày 14/10/2026. Đề tài: nghiên cứu cải thiện lời giải thích
trong hỏi đáp hình ảnh tiếng Việt.

## Trạng thái ngày 1/11/2026

- Sáu lượt train M2/M3, validation, đối chứng đổi ảnh và LR đã hoàn tất.
- Test đủ tám identity và 15.744 prediction; metric tự động, SD ba seed
  và CI từng seed đã kiểm lại. Không còn lượt xuất test chính bị thiếu.
- Demo M1 đã chạy ảnh thật; guard NaN/Inf và nạp lại đã qua T4 ngày 31/10.
- [Bản thảo báo cáo](../reports/thesis/report.md) tiếp tục cấu trúc cũ,
  có bảng test, hai biểu đồ và thảo luận. Bản Word theo mẫu trường còn
  cần hoàn thiện định dạng; chưa thay khung báo cáo bằng một mẫu mới.
- Phần đánh giá theo ảnh bởi người vẫn thiếu: luyện tiêu chí, chấm gói
  cố định, tính hỗ trợ và đồng thuận từ nhãn thật. Sau đó bổ sung báo cáo
  và slide. Hiện chưa cần mở thêm lượt train hoặc backbone.

## Mục tiêu và phạm vi

Từ cùng checkpoint M1, so sánh SFT bổ sung trên chosen với DPO trên
chosen/rejected: chất lượng giải thích theo ảnh có cải thiện không, đáp án
có bị ảnh hưởng không, và kết quả có ổn định qua các seed không?

M0–M3 là bốn cấu hình của Qwen2.5-VL-3B, không phải bốn kiến trúc.
Giữ M1-3683 và split ViVQA-X hiện có: 29.459 train, 1.407 validation,
1.968 test. Không train lại M1, không dùng test để chọn thiết kế.

Bộ nghiên cứu chính dùng 1.000 cặp, có thể tận dụng cặp v1 sau khi kiểm lại.
Không hạ tiêu chuẩn chất lượng để đạt số lượng.

## 1. Lượt kiểm tra triển khai 186 cặp

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

Cấu hình ở `configs/experiments/preference_study_v2.yaml` và hai
config `sft_chosen_full.yaml`, `dpo_full.yaml`. Ngày 15/10 đã khóa N=1.000
và hash review tại `frozen_1000_reviewed_20261015`. Cổng dữ liệu vẫn kiểm
đúng bundle/review; mỗi nhánh phải qua kiểm GPU trước khi train chính.
Không thay config hoặc source của run 186 cặp đang dùng trên Colab.

| Thiết lập | SFT chosen / DPO |
|---|---|
| Khởi đầu | Cùng M1-3683; reference DPO là M1 frozen |
| Dữ liệu | Cùng ID, ảnh, đáp án, chosen và seed xáo trộn dữ liệu |
| Epoch | 3 cho cả hai nhánh |
| Learning rate | 1e-5 cho cả hai nhánh |
| Seed | 42, 43, 44; tổng sáu run chính |
| Batch hiệu dụng | 1 × tích lũy 8, một GPU |
| DPO beta | 0,1 |
| Ảnh / guard | 200.704 pixels, 1.536 token; không cắt ảnh/completion |
| Checkpoint báo cáo | Cuối epoch 3 cho cả hai nhánh |

Seed là giá trị điều khiển tính ngẫu nhiên của lần huấn luyện. M2/M3 nạp
lại adapter M1 đã học và tắt dropout; thay seed chủ yếu thay đổi thứ tự
xáo trộn dữ liệu, không khởi tạo lại adapter. Ba seed 42, 43, 44 là ba lần
lặp độc lập từ cùng M1. Các giá trị này không biểu thị chất lượng mô hình.
Tổng hợp các lần chạy để kiểm tra mức dao động của kết quả M3–M2.
Cùng seed không bảo đảm hai thuật toán có toàn bộ phép tính ngẫu nhiên giống nhau.

Trạng thái 20/10: đã hoàn tất train, validation, BERTScore và CI của cả
ba seed. BERTScore M2/M3: 76,55 ± 0,01 / 79,26 ± 0,54; chênh lệch
M3−M2 +2,71 ± 0,54 điểm (trung bình ± độ lệch chuẩn mẫu). M3 vẫn thấp
hơn M1; chưa thấy cải thiện đáp án hoặc bám ảnh. Bảng đã kiểm ở
`reports/generated/preference_validation_three_seeds_20261020/`.
Đối chứng LR đã hoàn tất như dưới đây; không mở grid search mới.
Ngày 22/10: checkpoint cuối LR 5e-6 đã kiểm độc lập; M2/M3 đều `passed`,
375 update / ba epoch. M3 dùng attempt `m3_train_resume175`. Export validation
đã hoàn tất đủ 1.407 câu mỗi nhánh, bốn comparison comparable. Ngày 23/10
đã kiểm độc lập file thật và ảnh, lập bảng/CI. M3 LR thấp đạt EM 71,07,
ROUGE-L 41,78; tốt hơn M3 LR chính ở seed 42 nhưng vẫn dưới M1 về ROUGE-L.
BERTScore LR hoàn tất trên T4 ngày 23/10: M2/M3 77,16 / 81,65;
M3−M2 +4,49 điểm, CI [+4,23; +4,74]. M3 LR thấp vẫn dưới M1 (82,47).
File thật, giao thức và CI đã kiểm độc lập. Giữ đây là đối chứng một seed,
không thay thiết kế chính hoặc kết luận về bám ảnh từ tương đồng văn bản.
Phần đo LR đã đủ; tiếp theo khóa cấu hình, prompt, generation và rubric
trước khi xuất test. Không mở thêm lượt tìm siêu tham số.

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
   ghim. Báo chênh lệch M3–M2 cho từng seed; không chọn riêng seed có điểm cao nhất.
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

Ngày 23/10 đã tạo bộ luyện rubric riêng gồm 30 ảnh validation / 120 output
mỗi người, nhãn còn trống. Hồ sơ chuẩn bị test đã đối chiếu đủ tám identity
và giữ nguyên mẫu 200 ảnh; chưa xuất prediction test. Công cụ và cách lưu
nhãn ở `docs/human_evaluation.md`. Khóa hướng dẫn trước khi mở test,
không đổi kế hoạch chính sang run LR thấp vì điểm validation thuận lợi hơn.

Điều chỉnh thứ tự ngày 30/10, trước prediction test: khóa hướng dẫn dựa
trên rà validation, giữ phần luyện chấm người chưa hoàn tất và thực hiện
trước khi chấm 200 ảnh test. Bản hướng dẫn có lock và kế hoạch amendment
riêng; không ghi nhận xét tham khảo thành nhãn người. Kế hoạch gốc được giữ
nguyên byte. Các mô hình, trọng số, seed, prompt, dữ liệu, mẫu và chỉ số
không thay đổi. Chỉ CLI có kiểm amendment mới chấp nhận lock này; không
thêm cờ bỏ qua điều kiện hoặc sửa rubric theo điểm test.

Luồng xuất từng identity ở `scripts/run_study_test.py`, dùng lại exporter
hiện có. Kiểm đầu vào bằng `--inspect`; chạy và resume chỉ sau khi rubric
đã khóa. Luồng phục hồi giữ prefix và thư mục lỗi, không ghi đè lịch sử.
Tiến độ và bảng kết quả tám identity dùng `scripts/report_test.py`; chỉ
tính điểm khi đủ toàn bộ test, giữ M0/M1 dùng chung và BERTScore thiếu rõ.

Chỉ số chính: chênh lệch hỗ trợ theo ảnh 0–2 giữa M3 và M2 trên toàn mẫu.
Bổ sung: đáp án đúng kèm hỗ trợ 2, chi tiết không được ảnh hỗ trợ và uncertain.
EM/F1 đo đáp án; ROUGE-L/BERTScore đo tương đồng văn bản, không thay phần chấm
theo ảnh. Giữ toàn mẫu trong mẫu số, kể cả output sai định dạng/đáp án sai.

Tự động báo từng seed và mean ± SD qua ba seed. Paired image-bootstrap
2.000 lần, CI 95%, đo bất định tập ảnh; không thay SD qua seed. Chấm mù chỉ
seed 42 phải được ghi rõ, không suy CI người chấm thành kết quả của ba seed.
Phân tích lỗi theo loại, nguồn negative, độ dài; báo cả trường hợp DPO không
cải thiện. Không lọc ví dụ test xấu ra khỏi bảng.

Ngày 27/10 đã hoàn tất công cụ phân tích toàn bộ validation ba seed:
1.407 câu / 1.383 ảnh, giữ nguyên output. M3 khớp thêm/mất khớp đáp án
so với M2 lần lượt là 72/75, 71/73 và 75/82 theo ba seed. Nhóm câu hỏi
được gán bằng quy tắc và chỉ dùng thăm dò; các nhóm dưới 30 câu có cờ
nhóm nhỏ. Trang xem ảnh và bốn đầu ra không tạo nhãn chất lượng, không
đọc prediction test. Kết quả ở
`reports/generated/validation_diagnostics_20261027/`.

Thứ tự hoàn tất phần thực nghiệm:

1. Khóa hướng dẫn từ validation theo amendment 30/10; nhãn người vẫn chưa
   hoàn tất. Người chấm luyện tiêu chí độc lập trước khi chấm test, không
   lấy quyết định của công cụ làm nhãn người hoặc thay hướng dẫn theo test.
2. Kiểm lại tám identity trong hồ sơ test bằng cấu hình đã khóa; xuất đủ
   15.744 prediction test, giữ M0/M1 dùng chung khi identity khớp. Nếu
   ngắt phiên, nối đúng lượt còn thiếu thay vì xuất lại toàn bộ.
3. Tính EM/F1/ROUGE-L/BERTScore, paired image-bootstrap và tổng hợp ba
   seed. Chấm mù 200 ảnh test bằng gói đã cố định; báo số người chấm thật,
   độ đồng thuận và CI theo đúng giao thức.
4. Đưa cả kết quả thuận lợi và bất lợi vào bảng cuối; lấy ví dụ từ trang
   phân tích với ID/seed rõ ràng. Không dùng ví dụ chọn tay thay kết quả toàn tập.

Chưa mở thêm backbone, train epoch hoặc tìm siêu tham số. Phần còn thiếu
là đánh giá theo ảnh và nghiệm thu sản phẩm, không phải thêm số lượt train.

## 5. Demo và báo cáo

Giữ khung Word và code demo hiện có. Hoàn thiện luồng nhận ảnh/câu hỏi mới,
trả đáp án và lời giải thích, lựa chọn adapter đã nghiệm thu. Local chạy
giao diện/API; model chạy GPU, không train Qwen trên CPU.

Nghiệm thu live: ít nhất ba ảnh/câu chưa dùng trong replay, có identity
adapter, thời gian suy luận, xử lý lỗi, reload và endpoint xác thực. Replay
không được trình bày thành live. Lưu một bộ ảnh/bảng kết quả cuối để báo cáo.

Ngày 27/10 giao diện mới đã chạy M1 thật trên T4 với ba ảnh mẫu ngoài kho
dữ liệu; copy/zoom/lịch sử/reload, Ctrl+Enter và bố cục 390px đã kiểm.
Metadata adapter khớp M1-3683, notebook/output và ảnh màn hình đã lưu tại
`artifacts/demo_acceptance_20261027/`. Mốc này nghiệm thu chức năng;
lời giải thích còn lặp/chung chung, không thay chấm theo ảnh hoặc test cuối.

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

## Nghiên cứu bổ sung — 5/11/2026

Triển khai M4: DPO cộng NLL riêng phần đáp án, giữ cùng dữ liệu, M1,
LR và ngân sách update của M3. Cấu hình hiện dành cho seed 42; mới có
code/kiểm CPU, chưa có kết quả GPU hoặc chất lượng. Study riêng ở
`configs/experiments/answer_preservation.yaml`; thiết kế và điều kiện
chuyển sang train tại [answer_preservation.md](answer_preservation.md).

Đối chứng đầu tiên giữ bộ 1.000 cặp để cô lập thay đổi loss. Việc nâng
chất lượng cặp sẽ là lượt riêng. M4 phát triển trên validation; test cũ
đã được xem, cần holdout mới được khóa trước để xác nhận bản nâng cấp.
Nghiên cứu chính và toàn bộ kết quả M0–M3 được giữ để tái lập.

## Cơ sở thiết kế

[DPO](https://arxiv.org/abs/2305.18290) là cơ sở loss/reference;
[mDPO](https://arxiv.org/abs/2406.11839) chỉ ra rủi ro ưu tiên ngôn ngữ khi
học preference đa phương thức; dự án này không triển khai nguyên mDPO.
[e-ViL](https://arxiv.org/abs/2105.03761) là cơ sở kết hợp đo tự động và
đánh giá lời giải thích bởi người. Các giá trị N/epoch/LR ở trên là lựa chọn
thực nghiệm của dự án, không phải yêu cầu từ những bài báo đó.
