# Tệp đầu ra và bản dùng hiện tại

Thư mục này lưu bằng chứng local. Git giữ mục lục này; source release, ảnh,
checkpoint, log và kết quả chạy được lưu riêng. Kết quả hiện có ở
[mô tả dự án](../docs/project_description.md).

Bộ nghiên cứu chính là
`../data/preferences/vivqax_m1_3683_v2/frozen_1000_reviewed_20261015/`.
Hồ sơ seed 42/43 nằm trong `preference_v2/`; đối chứng đổi ảnh nằm trong
`image_dependence/`. Các mục 186 cặp dưới đây là lịch sử, không phải bộ train chính.

| Mục đích | Vị trí hiện hành |
|---|---|
| Kết quả duyệt cặp M1 | `preference_editor_m1_3683_2000_20261008/review_results_final_20261014.html` |
| Pilot 100 đã chốt | `../data/preferences/pilot_ai_20261012/decision_record.json` |
| Notes toàn pool — 2.000 quyết định | `../data/preferences/review_complete_20261014/reviewed_notes.jsonl` |
| Bộ train đã khóa — 186 cặp | `../data/preferences/vivqax_m1_3683_v1/README.md` |
| Quyết định và danh sách cặp rút | `../data/preferences/vivqax_m1_3683_v1/decision_record.json`, `quality_review.json` |
| Kiểm lượt cuối và toàn bộ cặp giữ | `preference_review_complete_20261014/` |
| Nhận xét AI đã kiểm lại | `preference_ai_triage_m1_3683_pilot_20261012_qc/` |
| Báo cáo quy mô pilot | `preference_ai_triage_m1_3683_pilot_20261012_qc/pilot_report/` |
| Source đóng gói | `releases/source_<UTC>/` |
| Hồ sơ chạy trên Colab | `colab_history/<ngày>/` |
| Word hiện hành | `deliverables/current/` |
| Rà và sửa source 12/10 | `code_cleanup_20261012/` |
| Tổ chức notebook và kiểm toàn repo 13/10 | `code_organization_20261013/review.json` |
| Output M1 giữ nguyên | `../notebooks/archive/03_m1_completed_with_outputs_20261013.ipynb` |

`project_source.zip` ở gốc là snapshot ngày 18/09/2026. Để tiếp tục dự án,
dùng release đã xác minh trong `releases/` và đọc nhật ký để chọn đúng bản.
Môi trường M1 dùng release ghim của run M1; source mới dành cho các bước tiếp theo.

Các thư mục có hậu tố ngày, `v2`, `v3`, `corrected`, `qa` hoặc `rechecked`
lưu những lượt đã thực hiện. Giữ đường dẫn và byte của chúng để các hash và
liên kết provenance còn đối chiếu được. Khi mở lại một bước, tìm bản hiện hành
trong bảng trên trước; các bản khác phục vụ truy vết.

## Quy ước output

Tên thư mục xác định thí nghiệm/phiên chạy; tên file xác định nội dung.
Ví dụ `m0_m1_validation_20261011/report.json` là báo cáo so sánh validation,
`training/checkpoint-3683/trainer_state.json` là trạng thái Trainer tại bước 3683.

| File | Nội dung |
|---|---|
| `console.log` | Log của phiên chạy |
| `execution.json` | Tham số và trạng thái phiên |
| `training_result.json` | Kết quả huấn luyện |
| `trainer_state.json` | Global step, epoch và lịch sử loss |
| `m0.jsonl`, `m1.jsonl` | Prediction từng mô hình trong thư mục đã ghi split |
| `*.manifest.json` | Nguồn, cấu hình và mã kiểm tra của output |
| `reviewed_notes.jsonl` | Quyết định duyệt từng mẫu |
| `accepted.jsonl`, `chosen_only.jsonl` | Hai đầu vào M3/M2 từ cùng bộ cặp |

Run mới dùng `<experiment>/<run_id>/`; kết quả đánh giá dùng
`<comparison>_<split>_<date>/`. Mã thí nghiệm và checkpoint phải có trong
metadata. Số ở `checkpoint-<step>` là optimizer update, không phải số mẫu.
Ảnh giữ image ID của COCO; trang duyệt ghi phạm vi/số mẫu và ngày xuất.

Các tên lịch sử như `corrected`, `qa`, `rechecked` mô tả lượt sửa cũ nhưng
dài và khó chọn bản. Không dùng chúng cho output mới; dùng run ID và metadata,
rồi cập nhật mục lục hiện hành. Không đổi đường dẫn/hash của kết quả đã chạy.
