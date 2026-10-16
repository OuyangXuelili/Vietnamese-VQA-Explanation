# Hỏi đáp hình ảnh tiếng Việt có lời giải thích

Cập nhật 16/10/2026. Thiết kế và các bước còn lại ở [kế hoạch](research_plan.md).

## Bài toán

Đề tài: **Nghiên cứu cải thiện chất lượng lời giải thích trong hỏi đáp hình
ảnh tiếng Việt**.

Hệ thống nhận ảnh và câu hỏi, sinh đáp án cùng lời giải thích ngắn. Một đáp
án đúng vẫn có thể đi kèm chi tiết sai; một chi tiết có thật cũng có thể chưa
đủ hỗ trợ đáp án. Đồ án đánh giá riêng độ chính xác đáp án và mức hỗ trợ của
giải thích theo ảnh.

Câu hỏi chính: từ cùng M1 và cùng chosen đã kiểm, DPO có cải thiện giải thích
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
| M2 | Từ M1, SFT chosen-only | Bộ 1.000 cặp: seed 42 xong 375 update, đủ validation |
| M3 | Từ cùng M1, DPO chosen/rejected | Bộ 1.000 cặp: seed 42 xong 375 update, đủ validation |

SFT tính loss trên completion, mask phần prompt/ảnh. QLoRA giữ backbone
4-bit, học adapter rank 16, alpha 32 trên q/k/v/o projections. DPO dùng
reference M1 frozen, không dùng reward model hoặc rollout PPO.

Lượt 186 cặp: M2/M3 cùng seed 42, N câu/ảnh/chosen, một epoch, batch hiệu dụng 8, dropout
tắt; M2/M3 cùng LR 5e-6, M3 beta 0,1. DPO giữ ảnh ở cả policy/reference và
kiểm hai completion trước train, không cắt input vượt 1.536 token.
FP16 trên T4; BF16 khi GPU hỗ trợ native ở run mới. Lineage M1 giữ nguyên.

N=186 đã khóa sau kiểm toàn pool và rút cặp yếu. Dự toán ban đầu là 1.000;
không bù số bằng cặp mơ hồ. Một epoch tương ứng 24 update mỗi nhánh. Cùng epoch không có nghĩa cùng token/FLOPs. M2 matched-LR
được chọn làm nhánh chính ngày 12/10 trước khi train; bản 2e-5 giữ làm lịch sử.

Lượt nhỏ giữ làm kiểm triển khai và tham khảo nội bộ, không phải kết quả
chính của báo cáo cuối. Bộ chính đã khóa 1.000 cặp / 1.000 ảnh train ngày
15/10, gồm 992 negative có kiểm soát và 8 output M1 nguyên văn. Đây không
phải đánh giá bởi nhiều người độc lập. SFT/DPO cùng ba epoch, LR 1e-5;
seed 42 đã xong, seed 43 đang chạy, seed 44 chưa chạy.
[Kế hoạch thực hiện](research_plan.md).

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
theo ảnh. Seed 43 đang train; seed 44, test và chấm mù chưa hoàn tất.

Selection: `artifacts/colab_history/20261007/m1_selection_frozen_v2/selection.json`.
Run cuối: `runs/m1/sft-full_20261007_023531_870900`.

## Dữ liệu cặp và đánh giá còn lại

M1-3683 sinh 2.000 ứng viên train-only, một câu/ảnh của pool seed 42. EM
70,15% trên pool là thống kê ứng viên, không phải tỷ lệ cặp hợp lệ.
Pilot 100 ảnh giữ nguyên: 12 giữ, 47 loại, 41 chưa chắc. Đã xử lý hết pool:
xem 1.966 ảnh, tách 34 câu theo phạm vi; không còn pending. Rà lại 201 cặp
ban đầu và rút 15 cặp yếu khỏi train. Bộ cuối có 186 cặp trên 186 ảnh,
gồm 78 rejected M1 nguyên văn và 108 negative có kiểm soát. Tập nhỏ,
được chọn lọc và thiên về lỗi object/attribute; phải báo giới hạn này.
Các nhãn này do cùng một trợ lý AI kiểm thị giác, không phải nhãn người
hoặc chấm mù. Nguồn actor và văn bản vẫn giữ trong metadata.

Cặp hợp lệ giữ cùng đáp án đúng, chosen có bằng chứng liên quan và rejected
có vấn đề xác định được theo ảnh. Ghi riêng lỗi mâu thuẫn, chi tiết không
được hỗ trợ và thiếu liên quan. Nháp AI giữ nguồn AI, không gán tên người;
xác nhận nguyên văn cũng không đổi tác giả. [Rubric và xuất dữ liệu](preference_annotation.md).

Với bộ mới, sau khi freeze cùng bộ cặp: kiểm mỗi trainer trên GPU, train M2/M3, khóa
cấu hình và xuất đầy đủ validation/test. Chấm mù mẫu đã chọn trước gồm
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
Giới hạn hiện có: lượt nhỏ một seed train, annotation dịch, cỡ mẫu chấm hữu hạn, pretraining
overlap chưa biết, chi phí tính khác nhau dù cùng LR/epoch. DPO không vượt SFT vẫn là kết
quả cần báo cáo. Thời gian còn lại chốt từ pilot và số đo GPU, không từ số
file hoặc loss giảm.

## Nguồn

[ViVQA-X](https://github.com/duongtruongbinh/ViVQA-X),
[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
[DPO](https://arxiv.org/abs/2305.18290),
[e-ViL](https://arxiv.org/abs/2105.03761),
[NLX-GPT](https://github.com/fawazsammani/nlxgpt),
[BERTScore](https://openreview.net/forum?id=SkeHuCVFDr),
[VINLE-GRPO](https://github.com/T-Sunm/VINLE-GRPO).
