# Demo hỏi đáp hình ảnh

Demo nhận ảnh và câu hỏi, hiển thị đáp án cùng lời giải thích của M1.
Chế độ replay đọc prediction đã lưu; chế độ live cần runtime GPU.

## Local: xem kết quả đã lưu

```powershell
.venv\Scripts\python.exe -m pip install -e ".[demo]"
.venv\Scripts\python.exe scripts/run_demo.py
```

Mở `http://127.0.0.1:7860`. Máy này dùng cấu hình `configs/demo/m1.yaml`
và ảnh trong `configs/environments/local_full.yaml`, không mount Drive.
Checkout mới cần tải đúng các tệp evidence/ảnh trước, không có trong source ZIP.

- 12 câu lấy theo ID tăng dần, một câu mỗi ảnh từ validation; không lọc theo
  đúng/sai hoặc chất lượng để chọn ví dụ đẹp. Không dùng test.
- Kiểm đủ 1.407 prediction với manifest/annotation, rồi kiểm hash từng ảnh ví dụ.
  Prediction/manifest/annotation phải khớp hash trong selection M1 đã đóng băng.
- Đáp án, giải thích, đầu ra gốc và thời gian đều của lần suy luận đã lưu.
  Giao diện ghi rõ **xem lại**, không nhận ảnh mới ở chế độ này.
- Không gọi replay là kiểm GPU hoặc đánh giá mô hình mới. Không tính điểm từ
  các ví dụ demo. Người xem vẫn phải đọc ảnh để nhận ra câu trả lời sai.

## GPU: hỏi ảnh mới

Trên runtime CUDA đã có môi trường suy luận và adapter, cài thêm `[demo]`.
Tạo một YAML riêng theo `configs/demo/m1.yaml`, trỏ tới **evidence đã ghim**
trên kho dữ liệu của runtime. Không sửa `src` hoặc chép đường dẫn Drive vào mã.
Có thể dùng `${VQA_PROJECT_ROOT}` trong YAML; biến phải được đặt thật.

```bash
python scripts/run_demo.py --mode live --config /path/to/demo.yaml \
  --adapter /path/to/checkpoint-3683
```

CLI kiểm hash adapter với selection và manifest validation, tải đúng revision
Qwen, rồi đối chiếu prompt/generation config với lần đánh giá M1. Chỉ tải model
khi CUDA có sẵn; không tự chuyển sang CPU hoặc offload CPU/disk khi thiếu VRAM.
Ảnh tải lên được chuẩn hóa RGB vào thư mục tạm và xóa sau suy luận; không lưu
vào Git, dataset hoặc log. Luồng demo không tự tạo nhãn hoặc cặp preference.

Mặc định chỉ nghe localhost. Để truy cập cloud từ máy khác cần đường chuyển
tiếp **HTTPS** có xác thực; không công khai cổng bằng HTTP. CLI chặn host ngoài
loopback nếu thiếu `VQA_DEMO_USER` và `VQA_DEMO_PASSWORD`. Không ghi mật khẩu
vào YAML/notebook/source ZIP. Không bật chia sẻ công khai trong lượt triển khai.
Chỉ một worker để không nạp hai model và không train song song với demo.

## Kiểm nghiệm thu

- Local: kiểm giao diện thật, chuyển ví dụ đúng ảnh/câu hỏi/đầu ra, không gán
  kết quả ví dụ cũ cho ảnh mới; không cho gọi `/api/predict` trong replay.
- Backend: câu hỏi rỗng/quá dài, JPEG/PNG hỏng/quá lớn, đường dẫn ví dụ lạ,
  output sai format, lỗi backend, đồng thời hai yêu cầu và timeout.
- Mỗi GPU xử lý một yêu cầu; yêu cầu thứ hai báo bận. HTTP chờ tối đa 90 giây.
  Timeout **không dừng cưỡng bức** CUDA: worker giữ khóa đến khi sinh xong,
  tránh tạo lượt chạy chồng nhau. Không tự sửa output sai format.
- Cần chạy thật trên GPU: ít nhất ba ảnh/câu mới, kiểm identity/prompt, nạp lại
  model, lỗi phiên runtime và thời gian/VRAM. Lưu ảnh màn hình và metadata;
  chưa có bằng chứng này thì demo live **chưa nghiệm thu**.
- Chỉ thêm M2/M3 sau khi có adapter/run chính hoàn tất. Không dùng nhãn lựa
  chọn mô hình cho nhánh chưa train. Frontend hiện chỉ giới thiệu M1.

API dùng FastAPI; giao diện HTML/CSS/JS trong `app/static`, service ở
`src/vqa_explanation/demo`. Phần upload/ảnh và timeout có giới hạn riêng;
không thay mã export/evaluation hoặc dữ liệu đã train. Tham khảo tài liệu
[kiểm thử](https://fastapi.tiangolo.com/tutorial/testing/) và
[HTTP Basic Auth](https://fastapi.tiangolo.com/advanced/security/http-basic-auth/).
