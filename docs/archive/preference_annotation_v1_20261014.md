# Gán nhãn cặp ưu tiên

Cập nhật 14/10/2026. Bộ chính là 2.000 ứng viên từ M1-3683, train-only.
Pilot 100 ảnh đã chốt theo quyền người thực hiện giao: 12 giữ, 47 loại,
41 chưa chắc. Đã xử lý hết pool: xem 1.966 ảnh, tách 34 câu theo phạm vi.
Notes: 201 accepted / 1.106 rejected / 693 uncertain / 0 pending.
Rà toàn bộ cặp giữ và rút 15 khỏi bản train; khóa 186 cặp / 186 ảnh.
Bộ cuối ở `data/preferences/vivqax_m1_3683_v1/frozen`; `quality_review.json`
ghi riêng các cặp rút. Pilot, notes và các snapshot lịch sử không đổi.

Trang dùng hiện tại:
`artifacts/preference_editor_m1_3683_2000_20261008/review_results_final_20261014.html`.
Trang chỉ hiển thị kết quả và lý do. Loại người/AI duyệt vẫn giữ trong
JSONL và manifest để không lẫn nguồn nhãn. Trang nháp cũ được giữ nguyên.
Các trang checkpoint-800 và bản nháp trước giữ làm lịch sử, không chạy lại.
Bản review gốc: `artifacts/colab_history/20261008/preference_candidates_completed/review_2000.zip`,
SHA-256 `365f080096f964a4a1d670faa7192a62697f45fb92f0480cdd555c7319b8af45`.

## Rubric

| Phần | Cách kiểm |
|---|---|
| Đáp án | correct/incorrect/uncertain theo ảnh; không chỉ dựa exact match |
| Hỗ trợ giải thích M1 | 0: sai/không liên quan; 1: một phần; 2: đủ bằng chứng liên quan |
| Chosen | Giải thích đúng, có bằng chứng hỗ trợ đáp án |
| Rejected | Có vấn đề cụ thể: mâu thuẫn, chi tiết không thấy, hoặc thiếu hỗ trợ đáp án |
| Quyết định | accepted: có cặp hợp lệ; rejected: không có cặp/lý do loại; uncertain: chưa xác định |

Hai completion giữ cùng ảnh, câu hỏi và đáp án đúng; chỉ khác giải thích.
Ghi cụm cần kiểm và bằng chứng nhìn thấy. Cặp chỉ khác văn phong/độ dài hoặc
hai phía đều đúng không dùng. Câu về cảm xúc, sở thích hay thuộc tính không
thấy thì giữ uncertain; không tạo bằng chứng đoán. Không loại câu khó khỏi test.

`error_type` dùng object, attribute, relation_action hoặc natural_model_error.
Lỗi thuật ngữ/diễn đạt ghi riêng; thiếu liên quan không tự được gọi là bịa
đối tượng. Tham chiếu nghi sai: ghi lý do loại, không sửa canonical giữa run.
Từ đồng nghĩa có thể đúng theo ảnh nhưng không được đổi luật EM sau khi thấy điểm.

## Duyệt và lưu

1. Xem ảnh, câu hỏi, đầu ra M1 và tham chiếu. Điền đánh giá theo rubric.
2. Kiểm/sửa nháp; ghi chosen/rejected, loại lỗi, cụm cần kiểm và bằng chứng.
3. Ghi actor, thời gian và quyết định. Người kiểm dùng `reviewer_type=human`
   và mã ổn định; AI kiểm dùng `reviewer_type=ai`, không dùng mã người.
4. Xuất JSONL để lưu bền vững. Trang tự lưu nháp ở cùng trình duyệt/origin;
   khi đổi máy/cổng hoặc từ `file://` sang localhost, xuất rồi nhập lại.

Form có ba nhóm: đánh giá ảnh, cặp giải thích và quyết định. Nguồn/hash nằm
trong mục kỹ thuật thu gọn. Nút Điền đề xuất giữ pending và điền nguồn AI;
không tự xác nhận. Khi tự sửa, kiểm lại nguồn và recipe, không tự đặt hash.

Bộ pilot chỉ hiển thị 100 câu nhưng xuất đủ 2.000 dòng, giữ nhãn ngoài pilot.
Nháp lịch sử ở `artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc/ai_prefill`.
Bản chốt pilot ở `data/preferences/pilot_ai_20261012`; bản tiếp tục ở
`data/preferences/review_complete_20261014`. Lượt cuối thêm 916 ảnh và 48 cặp;
đã xem đủ 1.866 ảnh ngoài pilot. Bản train sau lọc ở
`data/preferences/vivqax_m1_3683_v1`; 78 model / 108 synthetic. Đây là kiểm
của cùng trợ lý AI, không phải nhiều người độc lập hoặc chấm mù.

Chính sách `ai_visual_review_v1` cho phép dùng quyết định AI đã xem ảnh,
có đầy đủ bằng chứng và provenance. Export giữ `ai_review_candidate`,
`reviewer_type=ai`, `human_verified=false`. Chosen AI và rejected AI tạo
mới là synthetic có recipe; rejected M1 nguyên văn vẫn là model. Việc
xóa dòng nguồn khỏi giao diện không được xóa nguồn khỏi dữ liệu kỹ thuật.
Phần chấm mù test vẫn cần người thật theo giao thức đánh giá cuối.

Lượt 151–250 có danh sách và quy tắc trong `selection.json` trước khi duyệt.
Tối đa 10 negative có kiểm soát, không bắt đủ số: đổi một chi tiết nhìn
rõ, giữ cùng đáp án và cấu trúc văn bản. Giữ 9 sau rà lại, loại cặp màu
ảnh sepia; lỗi diễn đạt hình tượng cũng không dùng làm lỗi thị giác.
Điểm `explanation_support` vẫn chấm M1 gốc, không chấm negative tạo mới.

Lượt 251–350 theo 100 ID pending khóa từ lượt trước: 17 giữ, 49 loại,
34 chưa chắc. Xem toàn lượt qua bảng ảnh; cặp giữ được mở lại ảnh gốc.
Có 7 negative M1 nguyên văn và 10 có kiểm soát; giữ nguyên pilot và các
nhãn ngoài lượt. Tổng 48 cặp gồm 25 model / 23 synthetic. `continuation.json`
khóa 100 ID tiếp theo. Mốc dự toán 200 không tự mở cổng train.

Lượt 351–450 giữ 15 cặp: 6 negative M1 và 9 có kiểm soát; 58 loại,
27 chưa chắc. Xem cả lượt qua bảng ảnh và mở lại ảnh gốc cho cặp giữ.
Rút các trường hợp độ mềm chuối, ăn cỏ và màu đĩa bay chưa tạo được
đối lập chắc chắn. Audit còn phát hiện cụm sân tennis không có trong M1
gốc; cặp này được loại trước export. Tổng 63 cặp: 31 model / 32 synthetic.

Lượt 451–550 giữ 16 cặp: 6 negative M1 và 10 có kiểm soát; 47 loại,
37 chưa chắc. Cặp giữ được rà lại ảnh gốc; không giữ lỗi đen/xám đậm
khi màu lông còn phụ thuộc ánh sáng. Tổng 79 cặp: 37 model / 42 synthetic.
Pilot và mọi nhãn ngoài lượt giữ nguyên.

Lượt 551–650 giữ 16 cặp: 6 negative M1 và 10 có kiểm soát; 56 loại,
28 chưa chắc. Tổng 95 cặp: 43 model / 52 synthetic. Ảnh gốc của cặp giữ
đã được mở lại; không đổi nhãn ngoài lượt hoặc pilot.

Lượt 651–750 giữ 15 cặp: 5 negative M1 và 10 có kiểm soát; 54 loại,
31 chưa chắc. Tổng 110 cặp: 48 model / 62 synthetic. Thay đối lập ván
có cánh quạt bằng ván trượt/xe đạp trước công bố; bản nháp local giữ lại.
Không giữ lỗi chỉ từ màu trời hoặc cách nói cầm vợt và bóng còn mơ hồ.

Ngày 14/10 xử lý ba nhóm liên tiếp, mỗi nhóm theo 100 ID đã khóa:

| Lượt | Giữ | M1 nguyên văn / có kiểm soát | Loại | Chưa chắc |
|---|---:|---:|---:|---:|
| 751–850 | 16 | 6 / 10 | 45 | 39 |
| 851–950 | 15 | 5 / 10 | 41 | 44 |
| 951–1.050 | 12 | 2 / 10 | 44 | 44 |

Tổng 153 cặp: 61 model / 92 synthetic. Mỗi cặp giữ đã được mở ảnh gốc.
Hai đối chứng cuối được đơn giản hóa, chỉ khác độ sạch sàn hoặc màu mây;
bản nháp nằm trong `artifacts/preference_review_1050_20261014/initial_publish`.
Không giữ cặp từ câu dịch mơ hồ, trạng thái ẩn hoặc đổi chỉ văn phong.
Một rejected M1 có thể thiếu liên quan dù chi tiết là đúng; không gọi
mọi cặp model là bịa đối tượng. Bộ train cuối đã khóa; chưa có nhãn người.

34 câu về ý định, sở thích hoặc tác động sức khỏe được tách chỉ từ câu hỏi.
Nhóm này để uncertain, `image_evidence=null`, `explanation_support=null`;
không tính là đã xem ảnh hay đánh giá M1. Các ID có trong `selection.json`.

## Truy vết văn bản

- Rejected giữ nguyên giải thích M1: nguồn model, không ghi thành manual.
- Rejected do người sửa: manual, giữ nguồn ứng viên M1 ban đầu.
- Negative có kiểm soát hoặc chosen AI được xác nhận nguyên văn: synthetic,
  lưu source và SHA-256 của recipe. Xác nhận không đổi tác giả thành người.
- Khi người tự viết chosen, dùng manual; không để mặc định manual cho văn
  bản AI được chép nguyên. Giữ cùng template, tránh chosen luôn dài hơn.

Exporter lưu `original_candidate_origin` kể cả khi sửa hai phía.
Các nguồn model phải khớp M1 được chọn; trainer kiểm lại fingerprint.
Mọi cặp accepted có hash canonical/ảnh, actor/timezone, bằng chứng,
đoạn rejected và provenance chosen/rejected. Đoạn lỗi phải có trong rejected.

## Audit → báo cáo pilot → export → freeze

Lệnh bàn giao thực hiện audit, lưu một snapshot JSONL và báo phần còn thiếu:

```powershell
python scripts/finalize_preference_review.py --review-dir artifacts/preference_editor_m1_3683_2000_20261008 --notes <JSONL_DA_XUAT> --pilot artifacts/preference_editor_m1_3683_2000_20261008/pilot_review_100.json --output-dir <THU_MUC_MOI>
```

Lệnh không tạo quyết định, ghi tên người hay tự train. `handoff.md` tóm tắt
số câu pending/accepted; `handoff.json` liệt kê ID các cặp nháp cần kiểm.
`pilot_report.json` giữ báo cáo đầy đủ. File nguồn và bản bàn giao trước
không bị ghi đè.

Sau khi hoàn tất pilot, mở rộng và chốt N, dùng cùng lệnh với `--freeze`,
`--expected-pairs N`, `--bundle`, `--expected-bundle-sha256`, `--annotations`
và `--image-root`. N phải bằng đúng số accepted trong bản xuất. Lệnh gọi lại
exporter/validator hiện có, tạo `frozen/accepted.jsonl`, `chosen_only.jsonl`
và `manifest.json`; M2/M3 dùng cùng chosen. Không bắt buộc duyệt hết các
câu ngoài phạm vi đã chốt. Thiếu xác nhận/sai N thì chỉ báo lý do, không
tạo bộ train; exit 2 khi yêu cầu freeze bị chặn, exit 1 khi nguồn/nhãn lỗi.

Đóng băng metadata chưa có nghĩa đủ quy mô nghiên cứu hoặc sẵn sàng GPU.
Kiểm hai config, M1 selection và GPU theo `preference_training.md` sau đó.
Các lệnh riêng dưới đây vẫn dùng được khi cần kiểm từng phần.

Ví dụ dưới là các bước công cụ riêng, không cần chạy lại bản đã khóa.
Nếu tạo phiên bản khác, dùng output mới; không ghi đè bundle hiện hành.

```powershell
$reviewDir = 'artifacts/preference_editor_m1_3683_2000_20261008'
$reviewNotes = 'data/preferences/reviewed_notes.jsonl'
$reviewPilot = "$reviewDir/pilot_review_100.json"

python scripts/audit_preference_review.py --review-dir $reviewDir --notes $reviewNotes --pilot $reviewPilot
python scripts/summarize_preference_pilot.py --review-dir $reviewDir --notes $reviewNotes --pilot $reviewPilot --triage artifacts/preference_ai_triage_m1_3683_pilot_20261012_qc/release/triage.jsonl --output-dir <THU_MUC_BAO_CAO_MOI> --target-pairs 186

python scripts/export_preference_review.py --bundle artifacts/colab_history/20261008/preference_candidates_completed/review_2000.zip --expected-sha256 365f080096f964a4a1d670faa7192a62697f45fb92f0480cdd555c7319b8af45 --notes $reviewNotes --annotations data/processed/vivqax/image_disjoint_v1/all.jsonl --image-root data/raw/coco --output-dir data/preferences/export_v1
python scripts/validate_preferences.py --input data/preferences/export_v1/reviewed_pairs.jsonl --annotations data/processed/vivqax/image_disjoint_v1/all.jsonl --image-root data/raw/coco --output-dir data/preferences/frozen_v1
```

Audit kiểm đầy đủ trường và nguồn, không chứng minh nhãn thị giác đúng.
Resolved cần actor/timezone; rejected/uncertain cần lý do; accepted cần hai
giải thích, đoạn lỗi, bằng chứng và nguồn. Pending có sửa nháp được đếm riêng.

Báo cáo pilot chỉ ước lượng sau khi đủ 100 quyết định hợp lệ. Báo tổng
accepted và nhóm rejected nguyên văn M1 riêng; không dùng negative tạo sẵn
để ngoại suy pool lỗi tự nhiên. Bảng chéo AI–người chỉ mô tả, không tính
agreement vì hai bộ nhãn khác semantics.

Sau pilot, chốt N/phạm vi trước train. 1.000 là dự toán, không bắt hạ chuẩn
để đủ quota. Có thể export accepted trong pool còn pending; giữ file notes
đầy đủ để truy vết. Không có accepted thì exporter/validator dừng.

Freeze xuất `accepted.jsonl`, `chosen_only.jsonl`, `manifest.json`: cùng
ID/ảnh/đáp án/chosen cho M2/M3. Kiểm train-only, hash, prompt/canonical,
trùng nguồn và nguồn văn bản. Schema-only `--input` không đủ để train.
Trạng thái `validated_metadata` không thay nghiệm thu chất lượng ảnh hoặc
GPU gate. Bước sau: [train M2/M3](preference_training.md).

## Tái tạo công cụ

Chỉ dùng khi cần bộ review mới; các artifact hiện hành đã có.

- `select_preference_questions.py`: shuffle câu seed 42, khử trùng theo
  ID/path ảnh; không phải lấy đều theo ảnh hoặc khử gần trùng nội dung.
- `generate_preference_candidates.py`: sinh từ M1 đã chọn và manifest;
  giữ raw output/model/revision/seed, không ép đáp án tham chiếu.
- `prepare_preference_review.py`: kiểm ZIP/hash và tạo editor.
- `attach_preference_drafts.py`, `build_preference_pilot.py`: gắn nháp và
  hàng đợi; không tạo nhãn người. Xuất nháp trước khi refresh trang.

Đánh giá cuối: [chấm mù](human_evaluation.md). DPO có thể ưu tiên văn bản
mà ít dùng ảnh; [mDPO](https://arxiv.org/abs/2406.11839) là cơ sở nêu rủi ro,
không phải một kiến trúc được thêm vào đồ án.
