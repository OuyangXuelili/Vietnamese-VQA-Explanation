# Harness cho các phiên làm việc

Harness của dự án có hai phần: quản lý một đợt sửa bằng
`scripts/task_harness.py`, và kiểm code/chạy stage bằng các công cụ hiện có.
Nó dùng Python của đúng checkout, không gọi mô hình AI, đăng nhập tài khoản,
khởi chạy GPU, commit/push hoặc đồng bộ Drive.

## Bắt đầu và tiếp tục một việc

Khi bắt đầu phiên, đọc `AGENTS.md`, trạng thái mới nhất trong `TIEN_DO.md`,
phần kế hoạch liên quan và Git diff. Dùng lệnh sau để tìm các việc đang dở:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py status
```

Đọc biên nhận và tệp handoff của đúng mục tiêu trước khi nối tiếp. Không tự
chọn một việc khác, mở thí nghiệm mới hoặc chạy lại các notebook theo thứ tự.
Nếu thiếu nhãn người, GPU hoặc quyền truy cập, ghi rõ phần bị chặn và tiếp
tục những việc độc lập đã được giao. Một câu hỏi chỉ cần giải thích không
cần tạo hồ sơ hoặc chạy test.

Trong `status <task>`, `latest_check` cho biết lượt kiểm có dùng để nghiệm
thu được hay chưa. `failed_or_stale` cũng bao gồm lượt đang chạy; xem
`checks/<check-id>/receipt.json` để phân biệt `running`, `failed`,
`interrupted` và `stale`. Không chạy chồng một lượt khác vào cùng hồ sơ.

Mỗi đợt sửa có một hồ sơ; không tạo một hồ sơ cho từng file hoặc chỉnh sửa nhỏ.
Ví dụ tạo việc kiểm chức năng xuất và nhập nháp:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py start --kind frontend --title "Sửa xuất và nhập nháp" --accept "Xuất rồi nhập lại giữ nguyên các nhãn đã điền" --accept "Bản nháp sai nguồn bị từ chối"
```

Lệnh trả đường dẫn `artifacts/tasks/<id>`. Dùng đường dẫn đó cho các bước sau.
Hồ sơ ghi mục tiêu, tiêu chí, trạng thái Git và hash source/test/tài liệu đang
có, kể cả thay đổi chưa commit. Đây không thay thế snapshot đầy đủ trước sửa.

## Chọn kiểm tra theo thay đổi

| Việc làm | Kiểm tra cần có | Phần phải xem thêm |
|---|---|---|
| Code Python, script, cấu hình | Ruff và test hành vi liên quan; gate đầy đủ trước đồng bộ | Diff và tính tương thích |
| Dữ liệu hoặc pipeline đầu vào | Profile `data`, gọi gate `--full-data` | Split, ảnh và checksum của phiên bản dùng thật |
| Giao diện/API | Test frontend/API liên quan và kiểm trên trình duyệt | Lưu/khôi phục, lỗi, bố cục nếu bị ảnh hưởng |
| Báo cáo/slide | Đối chiếu số liệu nguồn, kiểm bản đã render | Trang, bảng, hình, phát biểu và trạng thái nhãn người |
| Tài liệu thuần túy | Đọc bản cuối và kiểm lệnh/liên kết đã sửa | Không chạy full test chỉ vì đổi câu chữ |
| Thí nghiệm GPU | Kế hoạch/run được người dùng giao và công cụ của stage | Nguồn, checkpoint, output thật và giới hạn chất lượng |

Ba profile thực thi chỉ chạy kiểm CPU; không tải Qwen hoặc train GPU:

```powershell
# Kiểm tập trung: bắt buộc chỉ rõ test liên quan
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py check artifacts/tasks/<id> --profile focused --test tests/unit/test_blind_review.py --test tests/unit/test_review_progress.py

# Gate hiện có: Ruff và toàn bộ pytest, gồm hồ sơ lịch sử trên máy này
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py check artifacts/tasks/<id> --profile full

# Chỉ dành cho checkout/source ZIP không có artifact lịch sử
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py check artifacts/tasks/<id> --profile full --source-only

# Khi sửa dữ liệu/pipeline đầu vào trên máy đủ ảnh
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py check artifacts/tasks/<id> --profile data
```

`<id>` là chỗ thay bằng ID thật, không chép nguyên dấu ngoặc vào PowerShell.
Mỗi lần kiểm tạo thư mục mới, lưu command, exit code, log, hash đầu vào và
phiên bản thư viện. Probe từ chối môi trường đang import repo khác. Lượt lỗi
hoặc bị ngắt giữ nguyên; lần thử tiếp theo có biên nhận riêng. Source/test/
config hoặc thư viện thay đổi trong/sau kiểm làm kết quả hết hiệu lực.
Cập nhật nhật ký hoặc báo cáo không tự làm test code hết hiệu lực; bằng
chứng tài liệu được đối chiếu hash riêng.

Dấu kiểm của task theo dõi mã, test, cấu hình và môi trường; không băm lại
toàn bộ ảnh hoặc checkpoint ở mỗi lượt kiểm code. Khi dữ liệu đổi, dùng
profile `data` và đính kèm biên nhận audit/preflight tương ứng. Nếu dữ liệu
đổi sau kiểm, chạy lại phần đó; hash source không thay kiểm dữ liệu.

Lệnh kiểm từ chối `PYTEST_ADDOPTS`, `PYTEST_PLUGINS`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD` đang có giá trị và `PYTHONOPTIMIZE` khác
0. Bỏ các biến này trước khi kiểm; chọn phạm vi qua profile và `--test`
để biên nhận ghi đúng những gì thực sự chạy.

## Nghiệm thu và bàn giao

Sau khi thực sự kiểm một tiêu chí, ghi nhận xét và tệp bằng chứng. Ví dụ dùng
biên nhận test hoặc ảnh màn hình đã lưu trong repo:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py review artifacts/tasks/<id> --criterion 1 --evidence artifacts/tasks/<id>/checks/<check-id>/receipt.json --note "Đã kiểm kết quả test xuất và nhập nháp"
```

Có thể lặp `--evidence` để đính kèm nhiều tệp. Mỗi tiêu chí dùng chỉ số bắt
đầu từ 1; chọn `--result failed` hoặc `blocked` khi chưa đạt. `review` ghi
lời xác nhận có hash bằng chứng, không tự chứng minh câu chữ đúng, tác giả
là người hay có một lần đánh giá độc lập. Không dùng nó để thay nhãn chấm
theo ảnh, kết quả GPU hoặc nghiệm thu trực quan chưa thực hiện.

Khi đủ tiêu chí:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py finish artifacts/tasks/<id> --status completed --summary "Đã sửa và kiểm hai hành vi xuất, nhập nháp"
```

Lệnh từ chối hoàn tất khi thiếu tiêu chí, bằng chứng đã đổi, lượt kiểm cuối
lỗi/bị ngắt/hết hiệu lực, hoặc dùng kiểm code thay kiểm dữ liệu. Việc code/
frontend cần kiểm đã thực thi; sửa code trong một việc mang nhãn docs/report
vẫn phải kiểm code. Việc hoàn tất được đóng; phát sinh mục tiêu mới thì tạo
hồ sơ mới, giữ biên nhận cũ.

Nếu chưa xong, lưu rõ kết quả hiện có và bước có thể tiếp tục:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/task_harness.py finish artifacts/tasks/<id> --status needs_input --summary "Gói chấm đã kiểm, còn thiếu nhãn thật" --next-step "Nhận nhãn của hai người chấm rồi kiểm đủ dòng trước khi tổng hợp"
```

`blocked`, `needs_input`, `in_progress` đều yêu cầu bước tiếp theo. Bản thân
chương trình không tự chạy lại, tự đổi tài khoản hoặc thông báo cho người
khác. Nếu tiến trình bị tắt đột ngột, biên nhận đang `running` chưa phải đạt;
đọc log và chạy một lượt kiểm mới khi tiếp tục.

## Mốc hoàn thiện và nguồn trạng thái

| Mốc trong kế hoạch | Điều kiện nghiệm thu | Bằng chứng cần giữ |
|---|---|---|
| Đánh giá bởi người | Kiểm đủ nhãn thật, đúng mẫu đã khóa; tổng hợp đúng protocol | Kết quả inspect/calibration, nhãn và báo cáo tổng hợp |
| Phân tích và báo cáo | Bảng/hình khớp kết quả đã khóa; nêu rõ kết quả chưa có | Nguồn số liệu, đối chiếu và các trang render đã xem |
| Demo | Hành vi bị sửa qua kiểm API/UI; thay đổi suy luận có kiểm GPU phù hợp | Test, ảnh màn hình và receipt GPU nếu thực sự chạy |
| Bàn giao | Hướng dẫn chạy khớp source, gate phù hợp qua, gói source kiểm được | Log, checksum và biên nhận đồng bộ nếu được giao |

`docs/research_plan.md` giữ thiết kế và kế hoạch; `TIEN_DO.md` giữ các mốc đã
quan sát. `artifacts/tasks/` chỉ giữ bằng chứng và bàn giao của từng việc,
không tạo bảng tiến độ nghiên cứu thứ hai. Sau mốc thật, cập nhật ngắn
TIEN_DO và chỉ rõ local-only/đã đồng bộ theo bằng chứng. Giữ trạng thái bất
lợi, giới hạn và các việc cần người thực hiện.

## Tự động đến đâu

`AGENTS.md` hướng dẫn Codex gọi harness trong đúng repo. CLI thực thi và
kiểm biên nhận khi được gọi; đây không phải hook toàn cục cưỡng chế mọi
tin nhắn hoặc dịch vụ chạy nền. Không chạy toàn bộ test cho câu hỏi thuần
giải thích. CI hiện có vẫn là một lần kiểm riêng, không suy ra CI đã qua
từ kết quả local. Không yêu cầu tài khoản Plus thứ hai để dùng harness này.

---


## Runner kiểm tra và chạy thí nghiệm

Harness dùng chung cho local và Colab. `scripts/check_project.py` kiểm lint/test;
`scripts/run_pipeline.py` lưu cấu hình, log và kết quả cho mỗi run.
M1 đã hoàn tất một epoch và đánh giá đủ validation. Các job tiếp theo xem
TIEN_DO; không chạy lại pilot mỗi khi nối runtime.

### Lệnh thường dùng

Trong repo, sau khi cài môi trường CPU đầy đủ theo [README](../README.md#local):

```powershell
.\.venv\Scripts\python.exe scripts/check_project.py
.\.venv\Scripts\python.exe scripts/check_project.py --source-only
.\.venv\Scripts\python.exe scripts/check_project.py --with-data
.\.venv\Scripts\python.exe scripts/check_project.py --full-data
```

Gate chạy Ruff và pytest CPU với `--fail-on-skip`. Mặc định kiểm cả lịch sử;
`--source-only` loại nhóm test cần run ngoài Git khi dùng checkout mới.
Cả hai vẫn cần dependency verification/demo và Node; thiếu thư viện không
được tính là đạt. Không tải Qwen hoặc train GPU trong gate.

`--with-data` thêm audit/preflight bộ 500 ảnh; `--full-data` kiểm toàn bộ
29.459 câu train của `image_disjoint_v1`. Các tùy chọn này cần dữ liệu local.
`--dry-run` chỉ hiển thị lệnh, không tạo run.

Chạy riêng từng stage và đổi môi trường khi ở Colab:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py audit --environment configs/environments/local_full.yaml
.\.venv\Scripts\python.exe scripts/run_pipeline.py preflight --environment configs/environments/local_full.yaml --all-train-records
.\.venv\Scripts\python.exe scripts/run_pipeline.py token-profile --environment configs/environments/local_full.yaml --dry-run
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-smoke --dry-run
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-full-preflight --environment configs/environments/local_full.yaml
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-optimizer-pilot --environment configs/environments/local_full.yaml --dry-run
.\.venv\Scripts\python.exe scripts/run_pipeline.py sft-full --environment configs/environments/local_full.yaml --dry-run
```

File `configs/environments/colab_full.example.yaml` là mẫu để đặt đường dẫn dữ liệu đã giải nén và thư mục run trên Drive. Không thay đường dẫn trong code. `token-profile` dùng `AutoProcessor` và collator SFT trên 512 câu train được chọn bằng seed cố định; cần Torch/Transformers nhưng **không tải trọng số model hay train GPU**. Trên Colab chạy qua `notebooks/archive/06_sft_token_profile.ipynb`. `sft-smoke` thật cần GPU CUDA, chỉ huấn luyện 8 bước theo `sft_smoke.yaml`; không dùng nó để huấn luyện M1 đầy đủ.

`sft-batch-pilot` dùng cùng thiết lập QLoRA/LoRA và collator với SFT smoke, nhưng chọn đúng một ID train theo `sft_batch_pilot.yaml`; đo forward/backward, token, loss, gradient, thời gian và VRAM. Không gọi optimizer, không lưu adapter/checkpoint. Chạy bằng `notebooks/archive/07_sft_batch_pilot.ipynb` trên T4, sau khi source/bundle đã xác minh. Kết quả `batch_pilot.json` chỉ chứng minh batch này khả thi, không là bằng chứng hiệu năng cho mọi mẫu hoặc kết quả M1.

`sft-full-preflight` xác nhận đúng 29.459/1.407/1.968 câu, không trùng ảnh, và tính lại SHA-256 của từng ảnh train cùng 256 validation cố định theo manifest; không tải mô hình. `sft-optimizer-pilot` dùng cùng trainer nhưng chỉ 20 optimizer step, đánh giá/lưu tại bước 10 và 20 để đo throughput và kiểm tra checkpoint. `sft-full` chạy một epoch trên toàn train và chọn checkpoint bằng validation loss; xem [giao thức M1](sft_full.md). Pilot không phải kết quả cuối.

### Bằng chứng của mỗi run

Mỗi stage tạo thư mục mới `runs/<stage>_<UTC>/` (hoặc `run_root` trong profile), không ghi đè thư mục cũ:

- `execution.json`: stage, lệnh, đường dẫn, thời điểm bắt đầu/kết thúc, trạng thái/return code, Git commit + dirty flag, hash SHA-256 annotation, config và các file source/config liên quan.
- `console.log`: stdout/stderr của tiến trình con.
- `audit.json`, `token_profile.json` hoặc `training/`: đầu ra riêng của stage.

Chỉ coi run là đạt khi `execution.json` có `status: passed`, `returncode: 0` và đầu ra stage hợp lệ. Một phiên train dừng chủ động có `status: paused` và `training/session_result.json`; đó không phải M1 đã hoàn tất. Hash source giúp nhận ra code khác nhau dù cùng Git commit nhưng working tree chưa commit. Thư mục Drive của run không phải bản sao của repo; source release là ZIP riêng có manifest/checksum.
Với profile full, `image_checksums` trỏ tới CSV hash từng ảnh; M1 preflight/pilot/full kiểm tra lại byte của các ảnh được dùng trước mỗi phiên train/resume. Audit thông thường chỉ kiểm tra ảnh mở được. Kiểm tra hash ảnh tốn thêm thời gian và I/O, nhưng chặn dữ liệu đổi âm thầm khi tiếp tục từ checkpoint.

### Giới hạn của runner

Token profile mặc định đo trên mẫu, không đại diện chắc chắn cho mọi câu train. P0 của M1 đã có report toàn 29.459 train và 256 validation; dùng lại khi hash khớp. Pilot/rehearsal kiểm tính khả thi và trạng thái resume, không chứng minh chất lượng. M1 chỉ hoàn tất sau kiểm checkpoint cuối một epoch; chất lượng còn cần prediction validation đầy đủ và chấm ảnh theo kế hoạch.
