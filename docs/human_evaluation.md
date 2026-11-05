# Chấm lời giải thích theo ảnh

Mẫu cuối đã đóng băng trước khi sinh prediction test: 200 ảnh khác nhau,
mỗi ảnh một câu, seed 42. File local:
`artifacts/human_evaluation/test_sample_200_seed42_v1.json`.
SHA-256: `8651cdfd737a07b5cfd401d595a45cda16adb9ba241864a9546b3f860a3330df`.
Tám lượt prediction test và bảng chỉ số tự động đã hoàn tất ngày 31/10/2026;
đã tải về local và đối chiếu lại. Nhãn chấm người vẫn chưa hoàn tất. Mẫu chỉ
chọn ID từ annotation, không chọn theo chất lượng đầu ra; giữ nguyên sau khi khóa.

Seed 42 ở bước này dùng để chọn mẫu đánh giá, không phải một lần train.
Bốn bộ output để chấm dùng M0/M1 và M2/M3 có seed huấn luyện 42 đã định
trước. Seed của phép bootstrap cũng là 42 nhưng chỉ điều khiển lấy mẫu
tính khoảng tin cậy, không tạo thêm lần huấn luyện.

## Gói chấm hiện hành ngày 1/11

Mở `artifacts/human_evaluation/review_delivery_20261101/index.html` để vào
bộ validation 30 ảnh và test 200 ảnh. Các trang mới giữ nguyên ảnh, ID,
thứ tự ẩn mô hình và output của bộ gốc. Bộ gốc vẫn được giữ nguyên.
`rater1.zip` và `rater2.zip` chỉ chứa trang/ảnh của từng vị trí, không có
khóa mô hình hoặc nhãn của vị trí còn lại. Giải nén trước khi mở trang.

Sau mỗi phiên dùng **Xuất nháp**; file này giữ cả dòng đang thiếu trường.
**Nhập nhãn** nhận lại nháp đã xuất, kiểm đúng nguồn trước khi nhập.
**Xuất nhãn hoàn chỉnh** yêu cầu mọi đầu ra đủ trường và một mã người
chấm nhất quán. Trang hiển thị số đầu ra/ảnh đủ trường và có nút đi đến
mục chưa xong. Tên tệp phân biệt validation/test và draft/ratings.

Các nhãn nguồn vẫn trống. File tiến độ đi kèm phản ánh thời điểm chuẩn bị,
không tự đọc nháp từ trình duyệt. Khi nhận JSONL, kiểm tiến độ test bằng:

```powershell
python scripts/blind_review.py inspect --bundle artifacts/human_evaluation/review_delivery_20261101/test --rating rater1=<file.jsonl> --output-dir <thu_muc_moi>
```

Lệnh chỉ kiểm độ đầy đủ và nguồn tệp, không tính điểm hay độ đồng thuận.
Chỉ dùng `score` sau khi nhãn đã hoàn tất. Với validation, tiếp tục dùng
`scripts/check_calibration.py --calibration artifacts/human_evaluation/review_delivery_20261101/calibration`.
Khi cần bản giao diện mới, `blind_review.py refresh-pages` tạo bundle mới
từ đúng bộ đã khóa, giữ ID, ảnh, khóa thứ tự và nhãn nguồn; không ghi đè
bundle cũ hoặc tự điền nhãn.

## Thống nhất cách chấm trước test

Bộ thực hành đã tạo ở
`artifacts/human_evaluation/rubric_calibration_30_seed42_v1/`: 30 ảnh validation,
mỗi ảnh một câu, bốn output nguyên văn của lượt chính seed 42. Không phải
duyệt lại bộ 1.000 cặp train. Mỗi người chấm 120 output; nhãn vẫn để trống.

Đọc `configs/evaluation/human_rubric_v1.yaml`, rồi mở
`bundle/rater1/index.html` hoặc `bundle/rater2/index.html` trong bộ thực hành.
Có thể dùng mã ẩn danh thay tên thật. Chấm độc lập, xuất JSONL trước khi
thảo luận điểm bất đồng. Bộ thực hành không được gộp vào kết quả test.

CLI `scripts/prepare_rubric_calibration.py prepare` kiểm hồ sơ train,
manifest và canonical validation trước khi tạo bộ mới. Lệnh `freeze` chỉ
khóa hướng dẫn sau khi file nhãn đã đầy đủ; không tự gán nhãn, ghi đè bản
cũ hoặc coi nhãn chưa chấm là xác nhận. Khi sửa hướng dẫn sau thảo luận,
giữ cả nhãn độc lập ban đầu và bản rubric cuối. Một người thì ghi đúng một
người; không báo agreement.

Kiểm tiến độ bằng `scripts/check_calibration.py`. Nếu chưa truyền `--rating`,
lệnh đọc các file nhãn trống trong bộ thực hành và báo đúng trạng thái hiện có:

```powershell
python scripts/check_calibration.py --calibration artifacts/human_evaluation/rubric_calibration_30_seed42_v1 --output-dir artifacts/human_evaluation/calibration_progress_local
```

Khi đã xuất nhãn, thêm `--rating rater1=<file.jsonl>` và
`--rating rater2=<file.jsonl>`, dùng thư mục đích mới. Công cụ kiểm các mục
còn thiếu, output bị sửa, ID thiếu/trùng và mã người chấm. Chỉ khi các file
đã hoàn tất mới tính tổng hợp và xuất `disagreements.csv` để thảo luận;
không mở nhãn của người khác trong lúc chấm độc lập. Một người thì không
có bảng bất đồng/độ đồng thuận. Báo cáo này dành riêng cho phía nghiên cứu,
không tự sửa nhãn hoặc khóa rubric.

`scripts/prepare_test_plan.py` đối chiếu ba seed chính và mẫu 200 ảnh trước
test. Kết quả hiện ở `artifacts/human_evaluation/final_test_preparation_v1/`:
tám identity, 1.968 câu mỗi run. Đây là bản chuẩn bị, không phải lệnh chạy
model hoặc kết quả test. Chưa có rubric đã khóa hay kiểm trọng số trên
runtime GPU thì trạng thái vẫn là `pending_rubric_and_runtime`.

## Xuất prediction test

Ngày 30/10, thứ tự được tách thành khóa hướng dẫn trước suy luận và hoàn
tất luyện chấm người trước khi chấm 200 ảnh test. Hướng dẫn cụ thể ở
`docs/evaluation_guidelines.md`, dựa trên bản rà 30 ảnh validation. Đây
không phải kết quả hiệu chỉnh giữa người chấm; nhãn người vẫn pending.

`scripts/freeze_evaluation_guidelines.py` chỉ tạo amendment khi chưa có
export test trong thư mục nghiên cứu. Bản kế hoạch gốc giữ nguyên; bản mới
phải giữ toàn bộ identity, trọng số, dữ liệu, mẫu, prompt và chỉ số, chỉ
đổi thứ tự đánh giá. Nhận xét validation giữ nguồn thật, không gán tên hoặc
đổi thành nhãn người. Báo cáo tự động ghi rõ calibration chưa hoàn tất.
Sau khi bắt đầu test không sửa hướng dẫn theo kết quả. Nhánh khóa rubric
từ nhãn người hoàn chỉnh vẫn dùng được với kế hoạch gốc.

Notebook `08_evaluate_test.ipynb` điều phối test bằng launch record ghim
source, kế hoạch và gói đầu vào. Không thêm ô test vào notebook 04/05.

`scripts/run_study_test.py` chạy từng identity trong kế hoạch đã khóa. M0/M1
chạy một lần; M2/M3 chạy riêng seed 42/43/44. Giữ nguyên cấu hình suy luận
của validation. Local dùng `--inspect`; suy luận thật chạy trên GPU Colab.

Ví dụ kiểm đầu vào local, không nạp model hoặc tạo prediction:

```powershell
python scripts/run_study_test.py --environment configs/environments/local_full.yaml --plan artifacts/human_evaluation/final_test_preparation_v1/plan.json --plan-sha256 debabdd51916c7c80e4380727d99a8ad6416ba46a0074b325daf7def2e610053 --identity m0 --output-dir artifacts/human_evaluation/test_runs/m0 --inspect
```

Trên Colab dùng environment của phiên hiện tại, có `image_checksums`, và
đường dẫn plan/output trong kho Drive. Thêm `--rubric` trỏ tới thư mục đã
khóa bằng `freeze`; M1/M2/M3 cần thêm `--adapter` đúng identity. Chạy
`--inspect` trước, rồi bỏ cờ này để xuất khi các điều kiện đã đủ. Thiếu
lock đúng với kế hoạch thì lệnh báo `blocked`. Lock hướng dẫn theo amendment
không xác nhận nhãn người hoặc cho phép báo độ đồng thuận.

Lệnh lưu từng dòng, kiểm trọng số và giao thức trước lần sinh đầu tiên.
Chạy lại cùng lệnh/thư mục sẽ xác minh phần đã lưu rồi chỉ sinh câu còn
thiếu. Run đã đủ 1.968 câu được kiểm lại mà không nạp model. Các prefix
25, 100, 200, ... và bản hoàn tất được giữ trong `recovery/`.

Nếu file chính bị cắt hoặc mất dòng, giữ nguyên thư mục lỗi. Dùng cùng
tham số, thêm `--recover-from <snapshot.jsonl>` và chọn **thư mục đích mới**;
kiểm trước bằng `--inspect`. Lượt này chỉ phục hồi prefix đã xác minh và
ghi nguồn phục hồi. Sau đó bỏ `--recover-from` để nối suy luận ở thư mục
mới. Snapshot và file lỗi vẫn được giữ nguyên. Không dùng lệnh khôi phục
validation cho output test.

## Kiểm và tổng hợp tám lượt test

`scripts/report_test.py --inspect` đọc trạng thái từng identity, không nạp
model hoặc tính điểm trên phần giao còn thiếu. Mặc định mỗi identity nằm
trong thư mục cùng tên dưới `--runs-root`; sau phục hồi có thể đổi đường
dẫn bằng `--run m3_seed43=<thư_mục_mới>`.

```powershell
python scripts/report_test.py --plan artifacts/human_evaluation/final_test_preparation_v1/plan.json --plan-sha256 debabdd51916c7c80e4380727d99a8ad6416ba46a0074b325daf7def2e610053 --annotations data/processed/vivqax/image_disjoint_v1/test.jsonl --runs-root artifacts/human_evaluation/test_runs --inspect
```

Khi cả tám lượt có đủ 1.968 câu và `completion.json` khớp, bỏ `--inspect`,
thêm `--output-dir` mới để xuất bảng EM, token-F1, ROUGE-L, chênh lệch và
CI theo ảnh. M0/M1 giữ một lần xuất; M2/M3 báo từng seed và mean/SD ba seed.
Không cộng ba seed thành 5.904 mẫu độc lập. File CSV dùng giá trị gốc;
nhân 100 khi trình bày chỉ số và chênh lệch theo điểm phần trăm.

BERTScore dùng CLI hiện có, cùng phiên bản môi trường và batch size cho
tám identity; ví dụ một lượt trên GPU:

```powershell
python scripts/evaluate.py artifacts/human_evaluation/test_runs/m0/m0.jsonl --split test --bertscore --bertscore-device cuda --bertscore-batch-size 16 --output artifacts/human_evaluation/test_semantic/m0.json
```

Sau khi đủ tám báo cáo, thêm một `--semantic identity=<file.json>` cho
mỗi identity vào lệnh tổng hợp. Công cụ kiểm hash prediction, đủ ID và
giao thức BERTScore trước khi ghép. Nếu chưa có BERTScore thì ghi rõ còn
thiếu; không điền số thay thế. Bảng tự động không thay kết quả chấm theo ảnh.

## Chuẩn bị sau khi có kết quả bốn nhánh

CLI ở `scripts/blind_review.py`; logic ở `evaluation/blind_review.py`.
M2/M3 phải xong, cấu hình và rubric phải chốt trước khi xuất test.

Bộ test đã tạo từ bốn output seed 42 tại
`artifacts/human_evaluation/blind_test_200_seed42_v1/`: 200 ảnh,
800 output cho mỗi vị trí người chấm. Đã kiểm hash ảnh, đủ ID, nội dung
khớp prediction và thứ tự ẩn mô hình riêng; nhãn đều để trống. Hoàn tất
luyện chấm validation trước khi bắt đầu chấm test. Chỉ gửi `images` và
thư mục của người đó; giữ `private` ở phía nghiên cứu.

Bảng test tự động và biên nhận kiểm nằm ở `artifacts/final_test_20261031/`.
Source suy luận của lượt này đã ghim trong release ngày 30/10. Bản source
mới có chặn lỗi NaN dành cho lượt mới; khi tái hiện nghiên cứu cũ phải dùng
đúng release đã ghim, không đổi checksum kế hoạch để ghép với code mới.

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
có kết quả thuận lợi nhất làm kết luận chính sau khi chấm.

Kappa thường dùng cho đáp án/chi tiết không được ảnh hỗ trợ; kappa trọng số
bình phương dùng cho hỗ trợ 0/1/2. Kappa null khi nhãn suy biến làm mẫu số bằng
không. Một người chấm thì không có agreement. Metadata không chứng minh hai
người độc lập; việc thực hiện phải đúng như protocol đã ghi.
Giữ nhãn độc lập, nếu phân xử bất đồng thì lưu vào artifact mới. Bootstrap
phản ánh bất định mẫu đánh giá, không thay thế nhiều seed train.
