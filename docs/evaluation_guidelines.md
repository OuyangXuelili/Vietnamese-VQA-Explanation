# Hướng dẫn chấm theo ảnh

Áp dụng cho mẫu test đã chọn trước. Chấm đáp án và lời giải thích riêng.
Giữ nguyên ảnh, câu hỏi và output, kể cả câu sai tiền đề hoặc sai định dạng.

## Đáp án

- Đúng: nội dung trả lời phù hợp với câu hỏi và bằng chứng trong ảnh.
- Sai: nội dung sai hoặc không trả lời câu hỏi.
- Chưa chắc: ảnh hoặc tiền đề câu hỏi không đủ để kết luận.

Không dùng đáp án tham chiếu để đoán chi tiết không nhìn thấy. Từ ngoại ngữ
còn hiểu rõ được chấm theo nghĩa; từ pha trộn không xác định được thì ghi
chưa chắc. Lỗi dùng tiếng Việt ghi riêng trong ghi chú.

## Mức hỗ trợ của lời giải thích

- 0: rỗng, mâu thuẫn, không liên quan, chỉ lặp đáp án hoặc chỉ nêu công dụng chung.
- 1: có bằng chứng liên quan nhưng chưa đủ, hoặc có chi tiết còn mơ hồ.
- 2: bằng chứng nhìn thấy rõ và đủ hỗ trợ đáp án, không có chi tiết sai đáng kể.

Đáp án đúng không tự làm giải thích đúng. Ví dụ: trả lời hươu cao cổ nhưng
giải thích là ngựa vằn thì đáp án đúng, mức hỗ trợ 0.
Nếu giải thích gọi sai đối tượng đang được hỏi, mức hỗ trợ 0. Nếu có bằng
chứng đúng nhưng thêm thuộc tính phụ sai, mức hỗ trợ không quá 1.

## Chi tiết không được ảnh hỗ trợ

Chọn có khi xác định được chi tiết sai hoặc đối tượng bị thêm vào mà ảnh
cho phép đối chiếu rõ. Chọn chưa chắc khi bị che, quá nhỏ, ngoài khung hoặc
thuộc tính không thể quan sát trực tiếp. Chọn không khi không tìm thấy
chi tiết sai; nhãn này không đồng nghĩa mọi câu suy đoán đã được xác minh.

Các trường hợp đã gặp trong 30 ảnh validation:

- Cảm giác được yêu thương, nhu cầu cắt cỏ hay tư cách thành viên dàn nhạc
  không được khẳng định chỉ từ mèo nằm nghỉ, cỏ cao hoặc người cầm violin.
  Có thể ghi bằng chứng bối cảnh, nhưng đáp án về điều không xác minh được
  vẫn chưa chắc và mức hỗ trợ không quá 1.
- Logo, nút tay cầm, chỉ may hoặc gấu váy bị che không được suy ra từ tên
  đồ vật. Hình dáng có thể đủ nhận loại đồ vật nhưng chưa xác nhận chữ nhỏ.
- Nguyên liệu bị khuất, nhiệt độ, trạng thái đông lạnh hoặc đồ uống trong
  vật chứa không trong suốt dùng chưa chắc nếu không có bằng chứng rõ.
- Câu hỏi gán hành động uống trong khi người đang ăn: ghi sai tiền đề;
  không coi câu trả lời về một chai trên bàn là đúng hành động được hỏi.
- Ảnh tĩnh không tự chứng minh xe đang chạy hoặc một trạng thái trước đó.
  Chỉ chấm điều quan sát được, không bổ sung diễn biến vào output.

Mỗi quyết định có một ghi chú ngắn chỉ ra bằng chứng hoặc giới hạn của ảnh.
Giữ mọi mẫu trong mẫu số, bao gồm sai, rỗng và chưa chắc.

## Trình tự đánh giá

Hướng dẫn được khóa trước khi sinh test. Bản rà validation là tham khảo,
không phải nhãn của người chấm hoặc bằng chứng về độ đồng thuận. Việc luyện
chấm độc lập còn phải hoàn tất trước khi chấm 200 ảnh test; không sửa hướng
dẫn theo output hoặc điểm test. Nếu tiêu chí vẫn có điểm chưa rõ, ghi giới
hạn và lưu quyết định riêng, không sửa rubric đã dùng để chọn kết quả.

So sánh chính là M3–M2 theo mức hỗ trợ trên toàn mẫu. Báo riêng đáp án đúng,
đáp án đúng kèm hỗ trợ 2, chi tiết sai và chưa chắc. Giữ nguyên seed 42 cho
phần chấm theo ảnh; không chọn seed thuận lợi nhất sau khi xem test.
