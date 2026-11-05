# Reproduction guide

## Tái lập kết quả đã khóa

Bảng test và bản demo có guard dùng hai source release khác nhau:

| Phạm vi | Source release | SHA-256 |
|---|---|---|
| Benchmark test M0–M3 | `source_20261030_091738_603939` | `0d742f8c4ac771063f45bf11ce5bbc3de82316ceb5cca90c506548d57d4978c1` |
| Kiểm guard và demo ngày 31/10 | `source_20261031_130609_290375` | `aeb4954dc7e8dbed74bf3333db8daec6d09ca221c31290c1df95e2de5f2bb057` |

Khi tính lại bảng test, khôi phục đúng ZIP bằng
`scripts/bootstrap_cloud_source.py --archive ... --sha256 ... --work-root ...`.
Đặt thư mục `src` của bản khôi phục đầu tiên trong đường dẫn import của
tiến trình phân tích; kiểm `module.__file__` để tránh editable install nạp
nhầm source hiện hành. Giữ bước kiểm hash trong plan. Bộ kiểm từ source
mới có thể từ chối plan cũ khi source inference đã thay đổi; không bỏ kiểm
tra đó để chạy tiếp.

Dùng `scripts/report_test.py` trong source đã ghim với plan, annotation,
tám thư mục run và tám tệp BERTScore gốc. Ghi kết quả vào thư mục mới.
Bản tổng hợp ngày 1/11 đã kiểm đủ 15.744 prediction, tính lại trung bình
và SD, đối chiếu CI và nguồn file; hồ sơ tại
`artifacts/report_update_20261101/`. Bản thảo hiện có ở
`reports/thesis/report.md` và `reports/thesis/report.html`.

Phân tích điểm từ prediction và BERTScore đã lưu chạy trên CPU, không nạp
Qwen hoặc huấn luyện. Benchmark đã khóa giữ 4-bit, FP16 và giao thức sinh
trong plan. Giao thức khác phải là lượt riêng, giữ kết quả cũ.

Các bước smoke bên dưới là lịch sử khởi động dự án, không phải phần cần
chạy lại để tính bảng cuối.

## Chuẩn bị annotation và manifest

```bash
python scripts/prepare_vivqax.py
python scripts/download_coco_smoke.py
```

Tập smoke được cố định bằng seed `20260918`, gồm 400 ảnh train, 50 validation
và 50 test. Các ảnh trùng giữa validation và test không được chọn. Kết quả chuẩn:

- 500 ảnh JPEG hợp lệ;
- 588 cặp hỏi–đáp;
- 830 lời giải thích;
- không có ảnh trùng giữa ba split;
- SHA-256 của `coco_smoke_500.zip`:
  `4e0fce747dee9a1beaf36e2617deae6582115020d7c219f315525e329a4024f9`.

## Thứ tự thực nghiệm ban đầu (mốc lịch sử)

Kết quả hiện có ở [mô tả dự án](project_description.md), các bước còn lại ở
[kế hoạch thực hiện](research_plan.md). Danh sách dưới ghi thứ tự khởi động dự án.

1. Tạo môi trường và cài package.
2. Chạy notebook Colab bootstrap và kiểm tra đủ 500 ảnh.
3. Chạy inference M0 trên tập smoke.
4. Chạy SFT vài step trên tập smoke và xác nhận loss/checkpoint.
5. Chỉ sau khi smoke test đạt mới lấy đủ 28.180 ảnh.
6. Huấn luyện M1, tạo preference data, huấn luyện M2/M3.
7. Đóng băng checkpoint rồi đánh giá test.

## Cổng kiểm tra M0

Chạy trước 20 mẫu validation bằng `scripts/run_m0.py`. Mỗi bản ghi đầu ra phải
giữ model revision, seed, ID mẫu, đáp án chuẩn, lời giải thích chuẩn, đầu ra thô,
kết quả đã tách và thời gian suy luận. Không dùng test split ở giai đoạn chọn
prompt. Nếu 4-bit không khả dụng, bỏ `--load-in-4bit` và dùng runtime có đủ VRAM.
