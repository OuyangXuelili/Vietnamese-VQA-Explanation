# Hỏi đáp hình ảnh tiếng Việt có lời giải thích

Cập nhật 20/10/2026. Thiết kế và các bước còn lại ở [kế hoạch](research_plan.md).

## Bài toán

Đề tài: **Nghiên cứu cải thiện chất lượng lời giải thích trong hỏi đáp hình
ảnh tiếng Việt**.

Hệ thống nhận ảnh và câu hỏi, sinh đáp án cùng lời giải thích ngắn. Một đáp
án đúng vẫn có thể đi kèm chi tiết sai; một chi tiết có thật cũng có thể chưa
đủ hỗ trợ đáp án. Đồ án đánh giá riêng độ chính xác đáp án và mức hỗ trợ của
giải thích theo ảnh.

Câu hỏi chính: từ cùng M1 và cùng giải thích tốt hơn (chosen), DPO có cải thiện giải thích
so với SFT bổ sung không? Phạm vi là ảnh tự nhiên và tiếng Việt; không xây
backbone mới hoặc chuyển sang OCR/RAG. Kết quả đo chất lượng đầu ra, không
chứng minh lời giải thích phản ánh phép tính nội bộ của mô hình.

## Dữ liệu

[ViVQA-X](https://github.com/duongtruongbinh/ViVQA-X) cung cấp hỏi đáp và giải
thích tiếng Việt, xây từ VQA-X qua dịch/hậu xử lý; ảnh tương ứng lấy từ COCO.

| Tập `image_disjoint_v1` | Câu hỏi | Vai trò |
|---|---:|---|
| Train | 29.459 | Học M1 và tạo pool preference |
| Validation | 1.407 | Chọn M1, phân tích lỗi, hiệu chỉnh rubric |
| Test | 1.968 | Đánh giá cuối |
| Tổng | 32.834 | Không chung ảnh giữa các tập |

Nguồn tham chiếu 28.180 ảnh. Split gốc có 48 ảnh chung validation/test;
phiên bản thí nghiệm bỏ 52 câu validation trên các ảnh đó, giữ train/test.
ID và hash ở `data/processed/vivqax/image_disjoint_v1/provenance.json`.

Lỗi dịch/nhãn mơ hồ cần kiểm bằng ảnh. Split riêng không chứng minh dữ liệu
chưa xuất hiện trong pretraining Qwen. Quyền tái phân phối chưa xác nhận;
không công bố bundle chỉ vì đã có bản sao local/Drive. [Data card](data_card.md).

## Thiết kế thực nghiệm

Backbone: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.

| Nhánh | Cách học | Trạng thái |
|---|---|---|
| M0 | Mô hình gốc, không train | Đủ validation |
| M1 | SFT đáp án/giải thích trên toàn train | Một epoch; đã chọn checkpoint-3683 |
| M2 | Từ M1, SFT trên chosen | Cả ba seed train xong và đủ bảng validation |
| M3 | Từ cùng M1, DPO trên chosen/rejected | Cả ba seed train xong và đủ bảng validation |

SFT tính loss trên completion, mask phần prompt/ảnh. QLoRA giữ backbone
4-bit, học adapter rank 16, alpha 32 trên q/k/v/o projections. DPO dùng
reference M1 frozen, không dùng reward model hoặc rollout PPO.

Bộ chính đã khóa 1.000 cặp trên 1.000 ảnh train ngày 15/10. M2 học giải
thích chosen; M3 học thêm sự đối lập với giải thích kém hơn (rejected).
Cả hai dùng LR 1e-5, ba epoch / 375 update, batch hiệu dụng 8; M3 beta 0,1.
DPO giữ ảnh ở cả policy và reference. Đầu vào vượt giới hạn 1.536 token
bị từ chối, không cắt ảnh hoặc phần trả lời. T4 dùng FP16.

Kế hoạch lặp M2/M3 với các seed 42, 43, 44 để đo mức dao động kết quả.
Mỗi lượt bắt đầu độc lập từ cùng trọng số M1, không học tiếp lượt seed
trước. Dropout được tắt; seed chủ yếu điều khiển thứ tự xáo trộn dữ liệu.
Seed 42/43/44 đã train và đánh giá validation xong. Đây là hai phương pháp chạy
lặp, không phải sáu mô hình có kiến trúc khác nhau.

Lượt 186 cặp trước đó dùng một epoch / 24 update và LR 5e-6. Giữ các run
và output để tra cứu, không ghép vào bảng kết quả chính của bộ 1.000 cặp.
[Chi tiết huấn luyện](preference_training.md) và [kế hoạch](research_plan.md).

## Kết quả M0/M1

M1 học 29.459 mẫu qua một epoch, 3.683 optimizer update. Loss đo trên 256
validation cố định: 3683 = 1,059366; 3500 = 1,059560. Theo quy tắc đã định,
chọn 3683; chênh lệch nhỏ không tự chứng minh giải thích tốt hơn.

Bảng dùng đủ 1.407 câu trên 1.383 ảnh validation, chỉ số nhân 100.

| Chỉ số | M0 | M1-3683 | M1 − M0 |
|---|---:|---:|---:|
| Exact Match | 41,01 | 70,93 | +29,92 |
| Token-F1 | 52,41 | 75,11 | +22,70 |
| ROUGE-L F1 | 27,05 | 44,29 | +17,24 |
| BERTScore F1 | 75,82 | 82,47 | +6,65 |

Báo cáo `reports/generated/m0_m1_validation_20261011/` trong kho kết quả có
CI theo ảnh và 24 ví dụ chọn theo quy tắc. EM dùng một đáp án sau chuẩn hóa, không
phải VQA consensus accuracy. F1 tách khoảng trắng; ROUGE/BERTScore đo giống
văn bản. Chưa có test hoặc chấm người.

## Validation bản chính — seed 42

Đã kiểm đủ 1.407 câu mỗi nhánh trong cùng môi trường suy luận. BERTScore
chấm lại cả bốn nhánh với cùng evaluator; không ghép điểm từ run khác.

| Nhánh | EM | Token-F1 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| M0 | 41,01 | 52,41 | 27,05 | 75,82 |
| M1 | 70,93 | 75,11 | 44,29 | 82,47 |
| M2 | 69,72 | 74,41 | 30,23 | 76,56 |
| M3 | 69,51 | 73,92 | 36,45 | 79,74 |

M3 hơn M2 về ROUGE-L/BERTScore, nhưng cả hai thấp hơn M1. Chênh lệch EM
M3–M2 −0,21 điểm có CI 95% [−1,86; +1,49], chưa phân biệt được bằng phép
đo này. Không suy thành tương đương hoặc đã cải thiện bám ảnh.
Bảng đầy đủ, CI và 32 ví dụ ở
`reports/generated/preference_validation_seed42_20261016/` trong kho kết quả.
Đối chứng đổi ảnh seed 42 đã xong: EM M1 giảm 68 → 16,5; M2 giảm 69,5 →
17,5; M3 giảm 69 → 17,5 trên 200 ảnh validation. Chưa thấy lợi thế M3 so
với M2 về mức giảm EM. Phép đo này không thay chấm chất lượng giải thích
theo ảnh. Test và chấm mù chưa hoàn tất.

Selection: `artifacts/colab_history/20261007/m1_selection_frozen_v2/selection.json`.
Run cuối: `runs/m1/sft-full_20261007_023531_870900`.

## Validation bản chính — seed 43

Đã tải prediction/manifest và hồ sơ train từ Drive, kiểm checksum gói,
đối chiếu canonical và tính lại độc lập đủ 1.407 câu / 1.383 ảnh mỗi nhánh.
Không sinh lại dự đoán. BERTScore dùng cùng evaluator đã ghim cho cả bốn nhánh.

| Nhánh | EM | Token-F1 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| M0 | 41,01 | 52,41 | 27,05 | 75,82 |
| M1 | 70,93 | 75,11 | 44,29 | 82,47 |
| M2 | 69,72 | 74,26 | 30,33 | 76,55 |
| M3 | 69,58 | 73,95 | 35,43 | 79,37 |

M3 − M2: EM −0,14 điểm, CI 95% [−1,85; +1,64]; ROUGE-L +5,11 điểm,
CI [+4,33; +5,82]; BERTScore +2,82 điểm, CI [+2,59; +3,05].
Kết quả chưa cho thấy cải thiện đáp án; mức tăng
tương đồng văn bản không tự chứng minh giải thích bám ảnh hơn.
Bảng, CI và các ví dụ cố định ở
`reports/generated/preference_validation_seed43_semantic_20261017/` trong kho kết quả.

## Tổng hợp validation ba seed

Mỗi seed có đủ 1.407 câu / 1.383 ảnh; đã đối chiếu prediction, manifest,
evaluator và hồ sơ train. Không chọn riêng seed có điểm cao nhất.
Các điểm nhân 100; ± là độ lệch chuẩn mẫu giữa ba lần train, không phải CI.

| Nhánh | EM | Token-F1 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| M1 cố định | 70,93 | 75,11 | 44,29 | 82,47 |
| M2 | 69,56 ± 0,29 | 74,17 ± 0,29 | 30,26 ± 0,06 | 76,55 ± 0,01 |
| M3 | 69,27 ± 0,47 | 73,75 ± 0,33 | 35,16 ± 1,45 | 79,26 ± 0,54 |
| M3 − M2 | −0,28 ± 0,19 | −0,42 ± 0,10 | +4,90 ± 1,44 | +2,71 ± 0,54 |

M3 tăng tương đồng văn bản so với M2 ở cả ba seed, nhưng không vượt M1.
Seed 44 riêng: BERTScore M2/M3 76,54 / 78,67; chênh lệch +2,13,
CI bootstrap theo ảnh [+1,90; +2,36]. Không suy kết quả này thành cải thiện
đáp án hoặc giải thích bám ảnh. Bảng từng seed giữ CI riêng, không lấy
trung bình các khoảng tin cậy.

Kết quả ở `reports/generated/preference_validation_seed44_20261020/` và
`reports/generated/preference_validation_three_seeds_20261020/`.
Bước tiếp theo là đối chứng LR 5e-6 cho cả hai nhánh theo kế hoạch, sau đó
khóa thiết kế trước khi mở test và chấm mù.

## Dữ liệu cặp và đánh giá còn lại

Bộ chính có 992 rejected tổng hợp có kiểm soát và 8 rejected là output
M1 nguyên văn; toàn bộ chosen là văn bản tổng hợp. Các cặp đã được kiểm
ảnh có trợ giúp mô hình, chưa được người gán nhãn độc lập thẩm định.
Nguồn tạo và quyết định duyệt giữ trong metadata.

Lượt 186 cặp lịch sử được xây từ 2.000 ứng viên M1, gồm 78 rejected M1
nguyên văn và 108 negative có kiểm soát. Quy trình và kết quả lượt đó
giữ trong [hướng dẫn huấn luyện](preference_training.md); không thay thế
bộ chính hoặc kết quả chấm mù cuối.

Cặp hợp lệ giữ cùng đáp án đúng, chosen có bằng chứng liên quan và rejected
có vấn đề xác định được theo ảnh. Ghi riêng lỗi mâu thuẫn, chi tiết không
được hỗ trợ và thiếu liên quan. Nháp AI giữ nguồn AI, không gán tên người;
xác nhận nguyên văn cũng không đổi tác giả. [Rubric và xuất dữ liệu](preference_annotation.md).

Việc tiếp theo là đối chứng LR đã định, rồi khóa cấu hình trước khi xuất
test. Chấm mù mẫu đã chọn trước gồm
200 ảnh test, một câu/ảnh cho bốn nhánh. Chỉ số chính là hỗ trợ 0–2 M3–M2
với paired image-bootstrap, 2.000 lần/seed 42. Chỉ số phụ gồm đúng đáp án
kèm hỗ trợ 2, chi tiết không được ảnh hỗ trợ và uncertain.
Một người chấm thì không báo agreement hai người. [Giao thức](human_evaluation.md).

## Demo, đóng góp và giới hạn

FastAPI và HTML/CSS/JS nhận ảnh/câu hỏi, hiển thị đáp án/giải thích. Replay
đọc đúng output M1 của 12 ảnh validation; live còn cần nghiệm thu GPU.
Trang duyệt cặp không phải demo. Source dùng lại ở `src`; notebook 04 chỉ
điều phối M2/M3. Local chuẩn bị/đánh giá, cloud train. [Cách chạy](demo.md).

Đóng góp dự kiến: bộ cặp tiếng Việt có truy vết, đối chứng chosen-SFT/DPO và
phân tích lỗi theo ảnh. Qwen/LoRA/DPO là phương pháp kế thừa.
Giới hạn hiện có: cả ba seed cùng bắt đầu từ một M1, bộ cặp tổng hợp chưa có thẩm định
độc lập, annotation dịch và chưa biết mức trùng với dữ liệu pretraining.
Cùng LR/epoch không đồng nghĩa cùng chi phí tính toán. Báo cả kết quả DPO
thấp hơn SFT và chưa kết luận về chất lượng giải thích theo ảnh khi thiếu
đánh giá của người.

## Nguồn

[ViVQA-X](https://github.com/duongtruongbinh/ViVQA-X),
[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
[DPO](https://arxiv.org/abs/2305.18290),
[e-ViL](https://arxiv.org/abs/2105.03761),
[NLX-GPT](https://github.com/fawazsammani/nlxgpt),
[BERTScore](https://openreview.net/forum?id=SkeHuCVFDr),
[VINLE-GRPO](https://github.com/T-Sunm/VINLE-GRPO).
