# Notebook đang dùng

[04_train_preferences.ipynb](04_train_preferences.ipynb) điều phối M2/M3 trên Colab.
[M1: code và output đã chạy](https://colab.research.google.com/drive/13Abjl7kJyDKzbxyYsCetJ540OzdK2P_E)
giữ trong `archive/03_m1_completed_with_outputs_20261013.ipynb`.

Notebook chia mục cho lượt 186 cặp lịch sử, bộ chính 1.000 cặp, validation,
đối chứng đổi ảnh và lượt lặp seed 43. Mỗi mục gọi CLI ở `scripts/`;
logic nằm trong `src/`. Không sửa output hoặc chạy lại các cell lịch sử.

Mặc định `MODE='inspect'`. Bộ chính đã khóa 1.000 cặp; M2/M3 seed 42 đã
hoàn tất, seed 43 đang chạy. Chọn đúng mục và study config theo seed.
Khi chạy nhóm mới: kiểm metadata → `MODE='gate'` → `MODE='train'`.
Hai nhánh dùng cùng run group, bộ chosen và checkpoint M1; không dùng
adapter gate làm điểm khởi đầu run chính.

[Lịch sử M1 và các bước kiểm cũ](archive/README.md) giữ source/output
nguyên vẹn. Không chạy lại notebook lịch sử để xem kết quả.
Notebook dùng số thứ tự cho giai đoạn, tên sau số mô tả công việc.
Không đổi tên bản đã chạy hoặc viết thêm công việc mới vào lịch sử M1.

[Kết quả hiện có](../docs/project_description.md) · [Huấn luyện M2/M3](../docs/preference_training.md)
