# Kho mã tham khảo

- NLX-GPT: cấu trúc theo task và evaluation cho natural-language explanations.
- e-ViL: benchmark và đánh giá VQA-NLE.
- ViVQA-X: pipeline dữ liệu tiếng Việt, split và baseline.
- VINLE-GRPO: triển khai tối ưu hóa VQA-NLE tiếng Việt gần với đề tài.
- Hugging Face TRL: trainer, config, examples và test cho post-training.

Danh sách này ghi các nguồn tham khảo thiết kế, chưa thay thế phần tổng quan
nghiên cứu có trích dẫn trong báo cáo. Repo chỉ mượn nguyên tắc tổ chức phù hợp;
không nhận phương pháp đã công bố là đóng góp mới.

## Đối chiếu hướng nghiên cứu — 27/09/2026

Các nguồn dưới là bài báo/repo của tác giả, không phải danh sách đồ án đại học
đã biết điểm. Dùng để kiểm tra phương pháp, không suy ra yêu cầu chấm của PTIT.

| Nguồn | Điều liên quan | Cách dùng trong đồ án này |
|---|---|---|
| [ViVQA-X, ICISN 2025](https://github.com/duongtruongbinh/ViVQA-X) | Dữ liệu từ dịch VQA-X, lựa chọn chất lượng và hậu xử lý; có baseline | Kế thừa dữ liệu, kiểm tra lỗi dịch và split; không nhận việc tạo ViVQA-X là đóng góp cá nhân |
| [e-ViL, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Kayser_E-ViL_A_Dataset_and_Benchmark_for_Natural_Language_Explanations_in_ICCV_2021_paper.html) | Đánh giá riêng đáp án và giải thích, có cả metric tự động và người chấm | Cần human evaluation, không kết luận từ ROUGE-L; thiết kế mẫu nhỏ hơn và ghi rõ khác protocol gốc |
| [NLX-GPT, CVPR 2022](https://github.com/fawazsammani/nlxgpt) | Sinh đáp án cùng giải thích; repo công bố output filtered/unfiltered | Hướng đầu ra kép có tiền lệ; báo cả mẫu đúng đáp án và toàn mẫu, không chỉ chọn ví dụ thành công |
| [DPO, 2023](https://arxiv.org/abs/2305.18290) | Học theo cặp preference với reference policy | Thuật toán kế thừa; đóng góp phải nằm ở câu hỏi/dữ liệu/đối chứng, không ở tên DPO |
| [mDPO, 2024](https://arxiv.org/abs/2406.11839) | Chỉ ra rủi ro ưu tiên bỏ qua điều kiện ảnh | Không giả định DPO tự giảm hallucination; kiểm tra cả ảnh và độ lệch văn phong/độ dài |
| [VINLE-GRPO, repo tác giả ghi ICISN 2026](https://github.com/T-Sunm/VINLE-GRPO) | VQA-NLE tiếng Việt, SFT và GRPO, phần thưởng đáp án/giải thích | Công trình rất gần, phải thảo luận; đối chứng chosen-only và bộ cặp duyệt theo lỗi là hướng kiểm chứng của repo này, không tuyên bố đầu tiên |

README VINLE-GRPO là mô tả từ tác giả; chưa kiểm chứng lại kết quả công bố
bằng cách chạy repo đó. Không so trực tiếp điểm của họ với điểm repo này khi
backbone, split validation và evaluator chưa đồng nhất. Cũng không cần kéo
toàn bộ framework GRPO/reward của họ vào đồ án.

Bài đang mở trước đây,
[Generate then Select (ACL Findings 2023)](https://aclanthology.org/2023.findings-acl.147/),
là pipeline sinh rồi chọn đáp án dùng tri thức ngoài trên OK-VQA. Đây không
phải nguồn chính cho đánh giá giải thích tiếng Việt; không dùng nó để biện
minh việc thêm retrieval/selector vào phạm vi đã chốt.

Về quy mô đào tạo, [mô tả khóa luận HTTT của UET](https://uet.vnu.edu.vn/chuong-trinh-dao-tao-nganh-thong-thong-tin-chat-luong-cao-theo-thong-tu-23-2/)
nhấn mạnh nghiên cứu, đề xuất và thi hành giải pháp ở quy mô phù hợp. Nguồn
này không phải rubric ngành Khoa học dữ liệu của PTIT; việc duyệt đề tài và
chuẩn chấm cụ thể vẫn phải theo giảng viên/hội đồng của người thực hiện.

Kết luận của lượt rà soát: đủ chất liệu cho đồ án thực nghiệm, chưa đủ bằng
chứng hoàn thành. Tập trung M1, cặp ưu tiên có kiểm duyệt, M2/M3 và đánh giá
độc lập; không mở rộng số mô hình để thay cho đối chứng còn thiếu.

## Đồ án môn học quốc tế — 12/10/2026

Đây là báo cáo môn học công khai, không phải khóa luận tốt nghiệp hoặc bài
đã được mình tái lập. Dùng để học thiết kế, không sao chép số liệu hay suy ra
điểm của dự án hiện tại.

- [VQA Preference Alignment with ORPO and DPO, Stanford CS231n 2024](https://cs231n.stanford.edu/2024/papers/visual-question-and-answering-preference-alignment-with-orpo-and.pdf):
  một backbone BLIP-VQA-base, so SFT, SFT+DPO và ORPO; đo chất lượng cùng
  hiệu suất train. Đây là tiền lệ sát cho đối chứng phương pháp trên một model.
  Khác repo này: dữ liệu/đích trả lời tiếng Anh, không tách cặp giải thích
  tiếng Việt giữ cùng đáp án. Không thêm ORPO chỉ để giống báo cáo đó.
- [Med-Idefics, Stanford CS231n 2024](https://cs231n.stanford.edu/2024/papers/med-idefics-a-two-stage-fine-tuning-approach-for-enhanced-medica.pdf):
  một backbone IDEFICS2-8B, so base, fine-tune một giai đoạn và hai giai đoạn;
  có phép kiểm ảnh không liên quan và dữ liệu ngoài. Chỉ học cách đặt
  đối chứng; phần 5.1 viết dùng cả train/test khi fine-tune nên không làm
  theo chi tiết chia tập đó. Một báo cáo sinh viên không phải chuẩn tuyệt đối.
- [Hướng dẫn custom project của Stanford CS224n](https://cs224n.stanford.edu/project/custom-final-project-tips.pdf):
  yêu cầu câu hỏi cụ thể, thước đo rõ và giá trị bổ sung; nghiên cứu phương
  pháp có sẵn, ablation và kết quả âm vẫn có giá trị. Không bắt buộc tạo
  thuật toán mới hoặc vượt SOTA.

Quyết định cho repo này: giữ Qwen và bốn cấu hình chính; hoàn tất bộ cặp và
đánh giá theo ảnh. Đối chứng LR có ích trực tiếp hơn train thêm một backbone.
Vintern zero-shot chỉ là baseline ngoài tùy ngân sách, chưa có code/kết quả.
Muốn kết luận phương pháp dùng được trên nhiều kiến trúc phải làm lại cả
SFT-chosen/DPO trên backbone thứ hai, không chỉ thêm một dòng zero-shot.
