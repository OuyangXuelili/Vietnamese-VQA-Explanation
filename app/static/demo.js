'use strict';

const el = id => document.getElementById(id);
const suggestionIds = ['suggest-object', 'suggest-place', 'suggest-action'];
let mode = null, imageData = null, previewUrl = null, busy = false;
let exampleVersion = 0, uploadVersion = 0, currentResult = null, history = [];

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 100000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    let data;
    try { data = await response.json(); }
    catch { throw new Error('Phiên GPU chưa phản hồi đúng. Kiểm kết nối Colab rồi tải lại web.'); }
    if (!response.ok) throw new Error(data.detail || `Backend báo lỗi HTTP ${response.status}.`);
    return data;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Quá thời gian chờ. GPU có thể vẫn đang xử lý; không gửi liên tục.');
    if (e instanceof TypeError) throw new Error('Mất kết nối backend. Kiểm phiên GPU rồi thử lại.');
    throw e;
  } finally { clearTimeout(timer); }
}

function error(message) {
  el('error').textContent = message;
  el('error').hidden = !message;
}

function updateAskState() {
  const ready = mode === 'live' && Boolean(imageData);
  el('ask').disabled = busy || !ready || !el('question').value.trim();
  el('question').disabled = busy || mode !== 'live';
  el('upload').disabled = busy || mode !== 'live';
  el('choose-image').disabled = busy || mode !== 'live';
  el('remove-image').disabled = busy || !ready;
  el('zoom-image').disabled = !el('preview').src;
  el('character-count').textContent = `${el('question').value.length} / 500`;
  el('ask-label').textContent = busy ? 'Đang xử lý…' : 'Gửi câu hỏi';
  el('question-hint').textContent = mode === 'replay' ? 'Ví dụ đã lưu · không gửi câu hỏi mới'
    : ready ? 'Ctrl + Enter để gửi' : 'Chọn ảnh để bắt đầu';
  suggestionIds.forEach(id => { el(id).disabled = busy || mode !== 'live'; });
  el('copy-result').disabled = busy || !currentResult;
  el('result-panel').setAttribute('aria-busy', String(busy));
}

function clearResult() {
  currentResult = null;
  el('answer').textContent = '—';
  el('explanation').textContent = '';
  el('latency').textContent = 'Chưa có kết quả';
  el('raw').textContent = '';
  el('result-announcement').textContent = '';
  el('copy-label').textContent = 'Sao chép';
  el('result-content').hidden = true;
  el('result-empty').hidden = busy;
  el('loading').hidden = !busy;
  el('copy-result').disabled = true;
  error('');
}

function show(result, fromHistory = false) {
  currentResult = result;
  el('result-content').hidden = false;
  el('result-empty').hidden = true;
  el('loading').hidden = true;
  el('answer').textContent = result.answer || 'Không đọc được đáp án';
  el('result-announcement').textContent = `Đáp án: ${result.answer || 'Không đọc được đáp án'}`;
  el('explanation').textContent = result.explanation || 'Không có giải thích hợp lệ.';
  el('raw').textContent = result.raw_output || '';
  const seconds = Number.isFinite(result.latency_seconds) ? result.latency_seconds.toFixed(2) : '—';
  el('latency').textContent = `${seconds} s${result.mode === 'replay' || fromHistory ? ' · lần chạy gốc' : ''}`;
  el('result-source').textContent = result.mode === 'replay' ? 'M1 · Kết quả validation đã lưu'
    : fromHistory ? 'M1 · Xem lại trong phiên này' : 'M1 · Suy luận trực tiếp';
  el('warning').textContent = result.warning || 'Mô hình có thể sai. Hãy đối chiếu lời giải thích với ảnh.';
  if (!result.format_valid) error('Đầu ra không đúng định dạng hai phần. Xem đầu ra gốc; không tự sửa kết quả.');
}

function renderHistory() {
  el('history-list').replaceChildren();
  el('history-section').hidden = !history.length;
  for (const item of history) {
    const button = document.createElement('button');
    button.className = 'history-item';
    button.disabled = busy;
    const question = document.createElement('span');
    question.textContent = item.question;
    const answer = document.createElement('strong');
    answer.textContent = item.result.answer || 'Đầu ra chưa hợp lệ';
    button.appendChild(question);
    button.appendChild(answer);
    button.addEventListener('click', () => {
      if (busy) return;
      el('question').value = item.question;
      error('');
      show(item.result, true);
      updateAskState();
    });
    el('history-list').appendChild(button);
  }
}

function setImage(url) {
  el('preview').src = url;
  el('preview').hidden = false;
  el('placeholder').hidden = true;
  el('expand-hint').hidden = false;
  el('drop-zone').classList.add('has-image');
  el('zoom-image').disabled = false;
}

function clearImage() {
  if (el('image-dialog').open) el('image-dialog').close();
  el('large-preview').removeAttribute('src');
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  el('preview').removeAttribute('src');
  el('preview').hidden = true;
  el('placeholder').hidden = false;
  el('expand-hint').hidden = true;
  el('drop-zone').classList.remove('has-image');
  el('image-meta').textContent = 'Chưa chọn ảnh';
  el('file-details').hidden = true;
  el('file-name').textContent = '';
  el('file-size').textContent = '';
  el('choose-label').textContent = 'Chọn ảnh';
}

async function loadExample(qid) {
  const version = ++exampleVersion;
  clearResult();
  try {
    const result = await api(`api/examples/${encodeURIComponent(qid)}`);
    if (version !== exampleVersion) return;
    setImage(`api/examples/${encodeURIComponent(qid)}/image`);
    el('question').value = result.question;
    el('image-meta').textContent = `Validation · ${qid}`;
    show(result);
    updateAskState();
  } catch (e) { if (version === exampleVersion) error(e.message); }
}

function acceptFiles(files) {
  if (mode !== 'live' || busy) return;
  const version = ++uploadVersion;
  imageData = null;
  history = [];
  renderHistory();
  clearResult();
  clearImage();
  updateAskState();
  if (!files.length) return;
  if (files.length !== 1) { error('Mỗi lượt chọn một ảnh.'); return; }
  const file = files[0];
  if (!['image/jpeg', 'image/png'].includes(file.type) || file.size > 6 * 1024 * 1024 || !file.size) {
    error('Chọn JPEG/PNG không quá 6 MB, không dùng tệp rỗng.'); return;
  }
  previewUrl = URL.createObjectURL(file);
  setImage(previewUrl);
  el('choose-label').textContent = 'Đổi ảnh';
  el('image-meta').textContent = 'Đang đọc ảnh…';
  el('file-name').textContent = file.name;
  el('file-name').title = file.name;
  el('file-size').textContent = `${(file.size / 1024).toFixed(0)} KB`;
  el('file-details').hidden = false;
  const reader = new FileReader();
  reader.onload = () => {
    if (version !== uploadVersion) return;
    imageData = typeof reader.result === 'string' ? reader.result.split(',')[1] : null;
    el('image-meta').textContent = imageData ? 'Đã chọn ảnh' : 'Chưa đọc được ảnh';
    updateAskState();
  };
  reader.onerror = () => {
    if (version !== uploadVersion) return;
    imageData = null;
    clearImage();
    error('Không đọc được ảnh. Hãy chọn lại một tệp khác.');
    updateAskState();
  };
  reader.readAsDataURL(file);
}

async function askQuestion() {
  if (busy || mode !== 'live') return;
  const question = el('question').value.trim();
  if (!imageData || !question) { error('Hãy chọn ảnh và nhập câu hỏi.'); return; }
  busy = true;
  clearResult();
  updateAskState();
  renderHistory();
  const started = Date.now();
  el('latency').textContent = 'Đang xử lý · 0 s';
  const elapsed = setInterval(() => {
    el('latency').textContent = `Đang xử lý · ${Math.floor((Date.now() - started) / 1000)} s`;
  }, 1000);
  try {
    const result = await api('api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image_base64: imageData, question})});
    show(result);
    history.unshift({question, result});
    history = history.slice(0, 6);
  } catch (e) {
    error(e.message);
    el('latency').textContent = 'Chưa có kết quả';
    el('result-empty').hidden = false;
  } finally {
    clearInterval(elapsed);
    busy = false;
    el('loading').hidden = true;
    updateAskState();
    renderHistory();
  }
}

el('examples').addEventListener('change', event => loadExample(event.target.value));
el('question').addEventListener('input', () => { clearResult(); updateAskState(); });
el('question').addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.isComposing) {
    event.preventDefault();
    askQuestion();
  }
});
el('upload').addEventListener('change', event => acceptFiles(event.target.files));
el('choose-image').addEventListener('click', () => { if (!busy && mode === 'live') el('upload').click(); });
el('remove-image').addEventListener('click', () => {
  if (busy || mode !== 'live') return;
  el('upload').value = '';
  acceptFiles([]);
});
el('ask').addEventListener('click', askQuestion);
suggestionIds.forEach(id => el(id).addEventListener('click', () => {
  if (busy || mode !== 'live') return;
  el('question').value = el(id).textContent;
  clearResult();
  updateAskState();
  el('question').focus();
}));

let dragDepth = 0;
el('drop-zone').addEventListener('dragenter', event => {
  event.preventDefault();
  if (mode === 'live' && !busy) { dragDepth++; el('drop-overlay').hidden = false; }
});
el('drop-zone').addEventListener('dragover', event => event.preventDefault());
el('drop-zone').addEventListener('dragleave', event => {
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) el('drop-overlay').hidden = true;
});
el('drop-zone').addEventListener('drop', event => {
  event.preventDefault();
  dragDepth = 0;
  el('drop-overlay').hidden = true;
  acceptFiles(event.dataTransfer.files);
});
document.addEventListener('paste', event => {
  if (mode !== 'live' || busy) return;
  const files = [...(event.clipboardData?.items || [])].filter(item => item.kind === 'file')
    .map(item => item.getAsFile()).filter(Boolean);
  if (files.length) { event.preventDefault(); acceptFiles(files); }
});

el('zoom-image').addEventListener('click', () => {
  if (el('preview').hidden) return;
  el('large-preview').src = el('preview').src;
  el('image-dialog').showModal();
});
el('close-image').addEventListener('click', () => el('image-dialog').close());
el('help').addEventListener('click', () => el('help-dialog').showModal());
el('close-help').addEventListener('click', () => el('help-dialog').close());
el('copy-result').addEventListener('click', async () => {
  if (!currentResult || busy) return;
  try {
    await navigator.clipboard.writeText(`Câu hỏi: ${el('question').value}\nĐáp án: ${currentResult.answer || ''}\nGiải thích: ${currentResult.explanation || ''}`);
    el('copy-label').textContent = 'Đã sao chép';
  } catch { error('Trình duyệt chưa cho phép sao chép. Bạn có thể chọn và sao chép nội dung trực tiếp.'); }
});
el('reconnect').addEventListener('click', () => { if (!busy) init(); });

async function init() {
  try {
    const info = await api('api/info');
    if (!['live', 'replay'].includes(info.mode)) throw new Error('Backend trả chế độ không hợp lệ.');
    mode = info.mode;
    el('mode').textContent = mode === 'replay' ? 'Xem lại · không GPU' : 'GPU đã kết nối';
    el('connection').dataset.state = mode === 'live' ? 'ready' : 'replay';
    el('notice').textContent = info.message;
    el('connection-note').hidden = mode !== 'replay';
    el('reconnect').hidden = true;
    el('identity').textContent = `${info.experiment_id} · ${info.model_id}\nRevision: ${info.model_revision}`;
    if (mode === 'replay') {
      imageData = null;
      ++uploadVersion;
      clearImage();
      history = [];
      renderHistory();
      el('upload-label').hidden = true;
      el('suggestions').hidden = true;
      el('replay-controls').hidden = false;
      el('examples').replaceChildren();
      for (const item of info.examples) {
        const option = document.createElement('option');
        option.value = item.question_id;
        option.textContent = item.question;
        el('examples').appendChild(option);
      }
      if (info.examples.length) await loadExample(info.examples[0].question_id);
      else error('Chưa có ví dụ validation để xem lại.');
    } else {
      el('upload-label').hidden = false;
      el('suggestions').hidden = false;
      el('replay-controls').hidden = true;
    }
    updateAskState();
  } catch (e) {
    mode = null;
    el('mode').textContent = 'Chưa kết nối';
    el('connection').dataset.state = 'error';
    el('connection-note').hidden = false;
    el('notice').textContent = 'Chưa nhận được phản hồi từ backend.';
    el('reconnect').hidden = false;
    error(e.message);
    updateAskState();
  }
}

init();
