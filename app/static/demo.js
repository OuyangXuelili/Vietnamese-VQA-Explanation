'use strict';
const el = id => document.getElementById(id);
let mode = null, imageData = null, previewUrl = null, busy = false, exampleVersion = 0, uploadVersion = 0;
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Backend không phản hồi.');
  return data;
}
function error(message) { el('error').textContent = message; el('error').hidden = !message; }
function clearResult() {
  el('answer').textContent = '—'; el('explanation').textContent = 'Chưa có kết quả cho ảnh/câu hỏi này.';
  el('latency').textContent = 'Chưa có kết quả'; el('raw').textContent = ''; error('');
}
function show(result) {
  el('answer').textContent = result.answer || 'Không đọc được đáp án';
  el('explanation').textContent = result.explanation || 'Không có giải thích hợp lệ.';
  el('raw').textContent = result.raw_output;
  el('latency').textContent = `${result.latency_seconds.toFixed(2)} s${result.mode === 'replay' ? ' · lần chạy gốc' : ''}`;
  el('warning').textContent = result.warning;
  if (!result.format_valid) error('Đầu ra không đúng định dạng hai phần. Xem đầu ra gốc; không tự sửa kết quả.');
}
function setImage(url) { el('preview').src = url; el('preview').hidden = false; el('placeholder').hidden = true; }
async function loadExample(qid) {
  const version = ++exampleVersion; clearResult();
  try {
    const result = await api(`api/examples/${encodeURIComponent(qid)}`);
    if (version !== exampleVersion) return;
    setImage(`api/examples/${encodeURIComponent(qid)}/image`);
    el('question').value = result.question; el('image-meta').textContent = `Validation · ${qid}`;
    show(result);
  } catch (e) { if (version === exampleVersion) error(e.message); }
}
el('examples').addEventListener('change', event => loadExample(event.target.value));
el('question').addEventListener('input', clearResult);
el('upload').addEventListener('change', async event => {
  const version = ++uploadVersion;
  imageData = null; clearResult(); el('ask').disabled = true;
  const file = event.target.files[0]; if (!file) return;
  if (!['image/jpeg', 'image/png'].includes(file.type) || file.size > 6 * 1024 * 1024) {
    error('Chọn JPEG/PNG không quá 6 MB.'); return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file); setImage(previewUrl);
  el('image-meta').textContent = `${(file.size / 1024).toFixed(0)} KB`;
  const reader = new FileReader();
  reader.onload = () => { if (version !== uploadVersion) return;
    imageData = reader.result.split(',')[1]; el('ask').disabled = busy; };
  reader.onerror = () => error('Không đọc được ảnh.'); reader.readAsDataURL(file);
});
el('ask').addEventListener('click', async () => {
  if (busy || mode !== 'live') return;
  if (!imageData || !el('question').value.trim()) { error('Hãy chọn ảnh và nhập câu hỏi.'); return; }
  busy = true; el('ask').disabled = true; el('upload').disabled = true;
  el('question').disabled = true; clearResult(); el('latency').textContent = 'GPU đang xử lý…';
  try { show(await api('api/predict', { method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image_base64:imageData, question:el('question').value}) })); }
  catch (e) { error(e.message); el('latency').textContent = 'Chưa có kết quả'; }
  finally { busy = false; el('ask').disabled = false; el('upload').disabled = false;
    el('question').disabled = false; }
});
async function init() {
  try {
    const info = await api('api/info'); mode = info.mode;
    el('mode').textContent = mode === 'replay' ? 'Xem lại · không cần GPU' : 'Live · GPU';
    el('notice').textContent = info.message;
    el('identity').textContent = `${info.experiment_id} · ${info.model_id} · revision ${info.model_revision}`;
    if (mode === 'replay') {
      el('replay-controls').hidden = false;
      for (const item of info.examples) {
        const option = document.createElement('option'); option.value = item.question_id;
        option.textContent = item.question; el('examples').appendChild(option);
      }
      el('ask').textContent = 'Bản xem lại · không suy luận ảnh mới';
      await loadExample(info.examples[0].question_id);
    } else { el('upload-label').hidden = false; el('question').disabled = false; }
  } catch (e) { el('mode').textContent = 'Backend chưa sẵn sàng'; error(e.message); }
}
init();
