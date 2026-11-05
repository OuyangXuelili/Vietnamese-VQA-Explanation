# Demo hỏi đáp hình ảnh

Demo nhận ảnh và câu hỏi, hiển thị đáp án cùng lời giải thích của M1.
Chế độ replay đọc prediction đã lưu; chế độ live cần runtime GPU.

Nếu tensor vision hoặc logits thô chứa NaN/Inf, suy luận dừng và API live trả
HTTP 503 với thông báo lỗi số học tiếng Việt, không trả đáp án. Hãy thử ảnh
khác; nếu lỗi tiếp diễn, kiểm tra phiên GPU và báo người vận hành. Ảnh tạm
được xóa và khóa yêu cầu được giải phóng. Đây chỉ là chặn lỗi lan sang văn bản;
nguyên nhân precision/quantization vẫn cần điều tra GPU riêng. Các prediction
và kết quả benchmark cũ giữ nguyên, đóng băng; không sinh lại hoặc sửa điểm.

## Local: xem kết quả đã lưu

```powershell
.venv\Scripts\python.exe -m pip install -e ".[demo]"
.venv\Scripts\python.exe scripts/run_demo.py
```

Mở `http://127.0.0.1:7860`. Máy này dùng cấu hình `configs/demo/m1.yaml`
và ảnh trong `configs/environments/local_full.yaml`, không mount Drive.
Checkout mới cần tải đúng các tệp evidence/ảnh trước, không có trong source ZIP.

- 12 câu lấy theo ID tăng dần, một câu mỗi ảnh từ validation; không lọc theo
  đúng/sai hoặc chất lượng đầu ra. Không dùng test.
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
Qwen, rồi đối chiếu prompt, generation và kiểu tính toán với lần đánh giá M1. Chỉ tải model
khi CUDA có sẵn; không tự chuyển sang CPU hoặc offload CPU/disk khi thiếu VRAM.
Ảnh tải lên được chuẩn hóa RGB vào thư mục tạm và xóa sau suy luận; không lưu
vào Git, dataset hoặc log. Luồng demo không tự tạo nhãn hoặc cặp preference.

Mặc định chỉ nghe localhost. Để truy cập cloud từ máy khác cần đường chuyển
tiếp **HTTPS** có xác thực; không công khai cổng bằng HTTP. CLI chặn host ngoài
loopback nếu thiếu `VQA_DEMO_USER` và `VQA_DEMO_PASSWORD`. Không ghi mật khẩu
vào YAML/notebook/source ZIP. Không bật chia sẻ công khai trong lượt triển khai.
Chỉ một worker để không nạp hai model và không train song song với demo.

## Kiểm kỹ thuật trên Colab

Dùng `notebooks/06_demo_live.ipynb`, không thêm ô vào 04/05. Notebook khôi
phục source đã kiểm và tạo venv `demo` riêng. Đường dẫn mẫu nằm trong
`configs/demo/m1_colab.example.yaml` và `configs/environments/demo_colab.example.yaml`.
Các tệp evidence/checkpoint vẫn phải có thật, đúng hash; mẫu không tự tạo dữ liệu thiếu.

`scripts/check_demo_live.py --inspect` kiểm selection, adapter, prediction và ảnh,
chọn ba ảnh validation ngoài gallery theo ID tăng dần. Không nạp GPU, không ghi file.
Lượt chạy thật yêu cầu hash của bản inspect và thư mục output chưa tồn tại.
API nhận ba ảnh, kiểm câu hỏi rỗng/ảnh hỏng/yêu cầu đồng thời, giải phóng model,
rồi nạp lại và chạy lại câu đầu. Không mở cổng mạng hoặc tạo liên kết công khai.

Kết quả nằm trong `demo_runs/<run>/report.json` và `results.jsonl`; mỗi câu được
ghi ngay sau khi xử lý. Lỗi có report riêng, không ghi đè lượt cũ. Báo cáo lưu
identity, ảnh/câu, output gốc, thời gian và VRAM. `technical_passed` không có
nghĩa là đáp án đúng; `model_quality_reviewed` và `browser_ui_reviewed` vẫn là false.
Test fixture ở local không phải bằng chứng suy luận GPU.

Lượt `m1_live_20261024_01` đã qua trên T4: ba ảnh validation ngoài gallery,
một lần nạp lại cho output trùng, VRAM cực đại 5,359 GiB. Bản notebook có
output, report và nhật ký đối chiếu ở
`artifacts/demo_live_20261024/m1_live_20261024_01/`.
Đây chưa phải kiểm giao diện live với ảnh tải lên ngoài dữ liệu.

## Mở web trên GPU Colab

`notebooks/07_demo_web.ipynb` chỉ khởi chạy web, tách khỏi notebook 06 đã
lưu kết quả kiểm. Điền release/checksum đã kiểm, chọn T4 và chạy từ trên
xuống. Sau khi đối chiếu đường dẫn M1, bật `RUN_WEB`. Không cần giải nén
bộ ảnh hoặc chạy lại validation.

Web nghe trên loopback của runtime. Notebook lấy đúng HTTPS origin từ
`google.colab.kernel.proxyPort` rồi mở giao diện bằng
`serve_kernel_port_as_iframe`. Không dùng ngrok, cổng công khai hoặc wildcard
origin. Chỉ origin của phiên và hai trang cha Colab xác định được phép;
các bảo vệ mặc định của web local không thay đổi.

Trên giao diện: chọn JPEG/PNG tối đa 6 MB, nhập câu hỏi rồi bấm Hỏi về ảnh.
Đổi ảnh/câu hỏi sẽ xóa kết quả cũ. Ảnh chỉ dùng tạm trong lượt suy luận,
không ghi vào dataset hoặc nhật ký. Mất phiên hoặc phản hồi không phải
JSON được báo rõ; timeout trình duyệt không cưỡng bức dừng model.

`demo_runs/<WEB_RUN>/` giữ `server.log` và `receipt.json` về nguồn, model,
hash adapter và trạng thái khởi động. Không lưu URL proxy hoặc ảnh tải lên.
Cổng đang dùng, danh tính khác M1 hoặc quá hạn khởi động đều bị chặn;
lỗi chỉ dừng process do notebook vừa tạo. Ô cuối dừng web khi dùng xong.

Đây là web tương tác trong phiên Colab, không phải website hosted hoạt động
24/7. Google đã đánh dấu cách mở proxy trong cửa sổ riêng là không ổn định;
không thay cài đặt cookie/bảo mật để ép nó hoạt động. Muốn URL độc lập ổn
định cần GPU backend và phương án triển khai riêng.
Tham khảo [mã nguồn Colab](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py).

Ngày 24/10 đã tải ba ảnh validation ngoài gallery qua web và nhận đầu ra
M1 thật trên T4 (2,50–5,76 giây/lượt). Trong phiên này bộ chọn tệp trong
iframe không nhận điều khiển; mở đúng URL proxy của cùng runtime ở tab
riêng thì hoạt động. Không sửa cài đặt bảo mật để mở proxy. Bằng chứng và
giới hạn ở `artifacts/demo_web_20261024/`; chưa kiểm ảnh ngoài dữ liệu hoặc
coi những ví dụ này là đánh giá chất lượng.

Ngày 27/10 đã nghiệm thu giao diện mới trên T4, run `m1_web_20261027_01`.
Revision và hash adapter từ backend khớp M1-3683. Ba ảnh mẫu scikit-image
không trùng SHA-256 với kho 28.180 ảnh: mèo → “mèo” (6,21 s), vật kim loại
cạnh cốc → “thìa” (2,69 s), vật trắng ở giữa ảnh → “tên lửa” (2,88 s).
Đổi ảnh xóa kết quả cũ; sao chép, phóng ảnh, hướng dẫn, lịch sử và bố cục
390px đã kiểm trên web thật. Sau reload, ảnh/lịch sử tạm được xóa, backend
kết nối lại; gửi bằng Ctrl+Enter trả cùng output tên lửa trong 2,90 s.

Đây là kiểm chức năng trên ba ví dụ được chọn trước suy luận, không phải
benchmark độ chính xác hoặc bằng chứng ảnh chưa có trong pretraining.
Giải thích ca thìa còn lặp đáp án, ca tên lửa mô tả hình dạng thô; giữ nguyên
output. Chấm theo ảnh và bảng test cuối vẫn còn lại. Notebook có output,
ảnh desktop/mobile và metadata ở `artifacts/demo_acceptance_20261027/`;
nguồn ảnh: [scikit-image](https://scikit-image.org/docs/stable/api/skimage.data).

## Kiểm nghiệm thu cuối

### Thao tác giao diện

Ảnh và khu vực hỏi–đáp nằm cạnh nhau trên desktop, xếp dọc trên điện thoại.
Có thể chọn, kéo-thả hoặc dán JPEG/PNG; ảnh chỉ gửi khi đặt câu hỏi. Nút
phóng to mở ảnh gốc để đối chiếu. Gợi ý câu hỏi chỉ điền ô nhập, Ctrl+Enter
mới gửi. Trạng thái chờ không báo phần trăm hoặc thời gian hoàn thành giả.

Nút sao chép lấy đúng đáp án và giải thích hiện tại. Sáu câu hỏi gần nhất
của cùng ảnh chỉ giữ trong RAM, xóa khi đổi ảnh hoặc tải lại trang. Chọn lại
câu cũ hiển thị nhãn xem lại/lần chạy gốc, không gọi GPU. Thông tin revision
và output nguyên văn nằm trong mục thu gọn. Không dùng giao diện để sửa
output, tạo nhãn hoặc suy ra độ tin cậy của mô hình.

### Kiểm tra

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
