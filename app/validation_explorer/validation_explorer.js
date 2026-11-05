'use strict';

const data = VALIDATION_DATA;
const byId = id => document.getElementById(id);
const modelNames = {m0: 'M0 · Pretrained', m1: 'M1 · SFT', m2: 'M2 · SFT chosen', m3: 'M3 · DPO'};
const normal = value => String(value).toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd');
const params = new URLSearchParams(location.hash.slice(1));
let seed = ['42', '43', '44'].includes(params.get('seed')) ? params.get('seed') : '42';
let selectedId = params.get('q'), filtered = [], position = 0;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function outputs(item) { return {...item.baselines, ...Object.fromEntries(
  ['m2', 'm3'].map(arm => [arm, item.seeds[seed][arm]]))}; }

function matchingCases() {
  const query = normal(byId('search').value.trim());
  return data.cases.filter(item => {
    if (byId('type').value !== 'all' && item.question_type !== byId('type').value) return false;
    if (byId('transition').value !== 'all' && item.seeds[seed][byId('contrast').value] !== byId('transition').value) return false;
    if (!query) return true;
    return normal([item.question_id, item.question, item.reference_answer,
      ...Object.values(outputs(item)).flatMap(o => [o.answer, o.explanation])].join(' ')).includes(query);
  });
}

function showCase(item) {
  byId('qid').textContent = `Câu ${item.question_id} · Ảnh ${item.image_id}`;
  byId('image').src = item.thumbnail;
  byId('question').textContent = item.question;
  byId('case-type').textContent = `Nhóm heuristic: ${data.type_labels[item.question_type]}`;
  byId('reference-answer').textContent = item.reference_answer;
  byId('reference-explanation').textContent = item.reference_explanations.join('\n');
  byId('contrast-status').textContent = data.transition_labels[item.seeds[seed][byId('contrast').value]];
  byId('cards').replaceChildren();
  for (const [arm, output] of Object.entries(outputs(item))) {
    const card = node('article', undefined, 'output-card');
    card.appendChild(node('h3', modelNames[arm]));
    card.appendChild(node('span', output.scores.exact_match ? 'Khớp EM tham chiếu' : 'Không khớp EM',
      `badge${output.scores.exact_match ? ' match' : ''}`));
    if (!output.format_valid) card.appendChild(node('span', 'Sai định dạng', 'badge invalid'));
    card.appendChild(node('p', output.answer || '(rỗng)', 'answer'));
    card.appendChild(node('p', output.explanation || '(không có giải thích)', 'explanation'));
    const metrics = node('div', undefined, 'metrics');
    for (const [key, label] of [['token_f1', 'F1'], ['rouge_l', 'ROUGE-L'], ['bertscore_f1', 'BERTScore']]) {
      metrics.appendChild(node('span', `${label} ${(output.scores[key] * 100).toFixed(2)}`));
    }
    card.appendChild(metrics);
    card.appendChild(node('p', `Lần chạy gốc: ${output.latency_seconds.toFixed(2)} s`, 'latency'));
    const details = node('details');
    details.appendChild(node('summary', 'Đầu ra gốc · không chỉnh sửa'));
    details.appendChild(node('pre', output.raw_output));
    card.appendChild(details);
    byId('cards').appendChild(card);
  }
  try { history.replaceState(null, '', `#q=${encodeURIComponent(item.question_id)}&seed=${seed}`); }
  catch { /* Browsers may restrict history on local files. */ }
}

function render() {
  filtered = matchingCases();
  position = Math.max(0, filtered.findIndex(item => item.question_id === selectedId));
  selectedId = filtered[position]?.question_id;
  const start = Math.floor(position / 40) * 40;
  byId('case-list').replaceChildren();
  for (const item of filtered.slice(start, start + 40)) {
    const button = node('button', undefined, 'case-item');
    button.setAttribute('aria-current', String(item.question_id === selectedId));
    button.appendChild(node('span', item.question));
    button.appendChild(node('small', `ID ${item.question_id} · ${data.type_labels[item.question_type]}`));
    button.addEventListener('click', () => { selectedId = item.question_id; render(); });
    byId('case-list').appendChild(button);
  }
  byId('count').textContent = `${filtered.length.toLocaleString('vi-VN')} / ${data.n.toLocaleString('vi-VN')} câu · seed ${seed}`;
  byId('list-range').textContent = filtered.length ? `${start + 1}–${Math.min(start + 40, filtered.length)}` : '0';
  byId('position').textContent = filtered.length ? `${position + 1} / ${filtered.length}` : '0';
  byId('previous').disabled = !filtered.length || position === 0;
  byId('next').disabled = !filtered.length || position === filtered.length - 1;
  byId('detail').hidden = !filtered.length;
  byId('empty').hidden = Boolean(filtered.length);
  if (filtered.length) showCase(filtered[position]);
}

for (const [value, label] of Object.entries(data.type_labels)) {
  const option = node('option', label); option.value = value; byId('type').appendChild(option);
}
for (const [value, label] of Object.entries(data.transition_labels)) {
  const option = node('option', label); option.value = value; byId('transition').appendChild(option);
}
byId('seed').value = seed;
byId('seed').addEventListener('change', event => { seed = event.target.value; render(); });
byId('search').addEventListener('input', render);
for (const id of ['type', 'transition', 'contrast']) byId(id).addEventListener('change', render);
function move(offset) {
  if (!filtered.length) return;
  position = Math.max(0, Math.min(filtered.length - 1, position + offset));
  selectedId = filtered[position].question_id;
  render();
}
byId('previous').addEventListener('click', () => move(-1));
byId('next').addEventListener('click', () => move(1));
document.addEventListener('keydown', event => {
  if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(event.target.tagName)) return;
  if (event.key === 'ArrowRight') { event.preventDefault(); move(1); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); move(-1); }
});
byId('copy').addEventListener('click', async () => {
  if (!filtered.length) return;
  const item = filtered[position];
  const text = [`Validation · ${item.question_id} · seed ${seed}`, item.question,
    `Tham chiếu: ${item.reference_answer}`, ...Object.entries(outputs(item)).map(([arm, o]) =>
      `${modelNames[arm]}\nĐáp án: ${o.answer}\nGiải thích: ${o.explanation}`)].join('\n\n');
  try { await navigator.clipboard.writeText(text); byId('notification').textContent = 'Đã sao chép ca đối chiếu.'; }
  catch { byId('notification').textContent = 'Có thể chọn và sao chép văn bản trực tiếp nếu trình duyệt chặn clipboard.'; }
});
render();
