const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../app/static/demo.js'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../../app/static/index.html'), 'utf8');
const info = {mode: 'live', experiment_id: 'M1_SFT', model_id: 'fixture',
  model_revision: 'pinned', message: 'fixture', examples: []};
const result = {mode: 'live', answer: 'xanh', explanation: 'ảnh màu xanh',
  question: 'Ảnh màu gì?',
  raw_output: 'Đáp án: xanh\nGiải thích: ảnh màu xanh', format_valid: true,
  latency_seconds: 1.25, warning: 'fixture'};
const tick = () => new Promise(resolve => setImmediate(resolve));

test('all script controls exist once in HTML and native dialogs are labelled', () => {
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const [, id] of source.matchAll(/el\('([^']+)'\)/g)) {
    assert.ok(ids.includes(id), `Missing control ${id}`);
  }
  assert.match(html, /id="image-dialog" aria-label=/);
  assert.match(html, /id="help-dialog" aria-labelledby=/);
});

async function page(initial = info) {
  const elements = new Map(), readers = [], revoked = [], timers = new Map(), calls = [];
  const documentListeners = {}, copied = [];
  let nextTimer = 0;
  const createElement = () => ({value: '', textContent: '', hidden: true,
    disabled: true, listeners: {}, children: [], dataset: {}, attributes: {},
    classList: {add() {}, remove() {}},
    addEventListener(name, callback) { this.listeners[name] = callback; },
    appendChild(child) { this.children.push(child); },
    replaceChildren() { this.children = []; },
    removeAttribute(name) { delete this[name]; },
    setAttribute(name, value) { this.attributes[name] = value; },
    focus() { this.focused = true; },
    click() { if (!this.disabled) return this.listeners.click?.(); },
    showModal() { this.open = true; }, close() { this.open = false; },
  });
  const element = id => {
    if (!elements.has(id)) elements.set(id, createElement());
    return elements.get(id);
  };
  let fetchImpl = async url => ({ok: true, status: 200,
    json: async () => url === 'api/info' ? initial : result});
  const context = vm.createContext({
    document: {getElementById: element, createElement,
      addEventListener: (name, callback) => { documentListeners[name] = callback; }},
    navigator: {clipboard: {writeText: async text => copied.push(text)}},
    AbortController, TypeError,
    URL: {createObjectURL: file => `blob:${file.name}`, revokeObjectURL: url => revoked.push(url)},
    FileReader: class {
      readAsDataURL(file) { this.result = `data:image/png;base64,${file.name}`; readers.push(this); }
    },
    setTimeout: callback => { const id = ++nextTimer; timers.set(id, callback); return id; },
    clearTimeout: id => timers.delete(id),
    setInterval: () => ++nextTimer, clearInterval() {},
    fetch: async (...args) => { calls.push(args); return fetchImpl(...args); },
  });
  vm.runInContext(source, context);
  await tick();
  return {context, element, readers, revoked, calls, documentListeners, copied,
    setFetch: callback => { fetchImpl = callback; },
    fireTimers: () => [...timers.values()].forEach(callback => callback()),
    upload: file => element('upload').listeners.change({target: {files: file ? [file] : []}}),
    question: value => { element('question').value = value; element('question').listeners.input(); },
  };
}

test('live requires the current image and question, then shows the backend result', async () => {
  const h = await page();
  assert.equal(h.element('question').disabled, false);
  assert.equal(h.element('ask').disabled, true);
  await h.upload({name: 'photo', type: 'image/png', size: 100});
  h.readers[0].onload();
  assert.equal(h.element('ask').disabled, true);
  h.question('Ảnh màu gì?');
  assert.equal(h.element('ask').disabled, false);
  await h.element('ask').listeners.click();
  assert.equal(h.element('answer').textContent, 'xanh');
  assert.deepEqual(JSON.parse(h.calls.at(-1)[1].body),
    {image_base64: 'photo', question: 'Ảnh màu gì?'});
  h.question('Câu hỏi khác');
  assert.equal(h.element('answer').textContent, '—');
  assert.equal(h.element('raw').textContent, '');
});

test('invalid replacement removes the previous preview and cannot submit it', async () => {
  const h = await page();
  h.question('Ảnh màu gì?');
  await h.upload({name: 'valid', type: 'image/png', size: 100});
  h.readers[0].onload();
  await h.upload({name: 'invalid', type: 'text/plain', size: 100});
  assert.equal(h.element('preview').hidden, true);
  assert.equal(h.element('preview').src, undefined);
  assert.equal(h.element('ask').disabled, true);
  assert.deepEqual(h.revoked, ['blob:valid']);
  assert.match(h.element('error').textContent, /JPEG\/PNG/);
});

test('a late file read cannot replace a newer upload', async () => {
  const h = await page();
  h.question('Ảnh màu gì?');
  await h.upload({name: 'old', type: 'image/png', size: 100});
  await h.upload({name: 'new', type: 'image/png', size: 100});
  h.readers[0].onload();
  assert.equal(h.element('ask').disabled, true);
  h.readers[1].onload();
  await h.element('ask').listeners.click();
  assert.equal(JSON.parse(h.calls.at(-1)[1].body).image_base64, 'new');
});

test('replay never accepts an upload or claims live inference', async () => {
  const h = await page({...info, mode: 'replay', examples: [{question_id: 'q1', question: 'Ảnh?'}]});
  assert.match(h.element('mode').textContent, /Xem lại/);
  await h.upload({name: 'photo', type: 'image/png', size: 100});
  assert.equal(h.readers.length, 0);
  await h.element('ask').listeners.click();
  assert.equal(h.calls.some(([url]) => url === 'api/predict'), false);
});

test('an expired proxy response produces a readable message, not a JSON error', async () => {
  const h = await page();
  h.setFetch(async () => ({ok: false, status: 403,
    json: async () => { throw new SyntaxError('HTML response'); }}));
  await assert.rejects(h.context.api('api/predict'), /Phiên GPU chưa phản hồi đúng/);
});

test('network loss and a long request report different failures', async () => {
  const h = await page();
  h.setFetch(async () => { throw new TypeError('fetch failed'); });
  await assert.rejects(h.context.api('api/predict'), /Mất kết nối backend/);
  h.setFetch((url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => {
      const error = new Error('aborted'); error.name = 'AbortError'; reject(error);
    });
  }));
  const pending = h.context.api('api/predict');
  h.fireTimers();
  await assert.rejects(pending, /GPU có thể vẫn đang xử lý/);
});

test('suggestions only fill the question and never send automatically', async () => {
  const h = await page();
  h.element('suggest-object').textContent = 'Đối tượng chính là gì?';
  h.element('suggest-object').listeners.click();
  assert.equal(h.element('question').value, 'Đối tượng chính là gì?');
  assert.equal(h.element('question').focused, true);
  assert.equal(h.calls.some(([url]) => url === 'api/predict'), false);
  assert.equal(h.element('ask').disabled, true);
});

test('keyboard shortcut sends once, ignores Vietnamese IME composition', async () => {
  const h = await page();
  await h.upload({name: 'photo', type: 'image/png', size: 100});
  h.readers[0].onload();
  h.question('Ảnh màu gì?');
  const key = {ctrlKey: true, key: 'Enter', preventDefault() {}, isComposing: true};
  h.element('question').listeners.keydown(key);
  assert.equal(h.calls.some(([url]) => url === 'api/predict'), false);
  h.element('question').listeners.keydown({...key, isComposing: false});
  await tick();
  assert.equal(h.calls.filter(([url]) => url === 'api/predict').length, 1);
});

test('history is per-image, visibly cached, and never calls the model', async () => {
  const h = await page();
  await h.upload({name: 'photo', type: 'image/png', size: 100});
  h.readers[0].onload();
  h.question('Ảnh màu gì?');
  await h.element('ask').listeners.click();
  const count = h.calls.length;
  assert.equal(h.element('history-list').children.length, 1);
  h.question('Câu khác');
  h.element('history-list').children[0].listeners.click();
  assert.equal(h.element('question').value, 'Ảnh màu gì?');
  assert.equal(h.calls.length, count);
  assert.match(h.element('latency').textContent, /lần chạy gốc/);
  assert.match(h.element('result-source').textContent, /Xem lại/);
  await h.upload({name: 'second', type: 'image/png', size: 100});
  assert.equal(h.element('history-section').hidden, true);
  assert.equal(h.element('answer').textContent, '—');
});

test('copy uses exact current output, and removing an image invalidates everything', async () => {
  const h = await page();
  await h.upload({name: 'photo', type: 'image/png', size: 100});
  h.readers[0].onload(); h.question('Ảnh màu gì?');
  await h.element('ask').listeners.click();
  await h.element('copy-result').listeners.click();
  assert.equal(h.copied[0], 'Câu hỏi: Ảnh màu gì?\nĐáp án: xanh\nGiải thích: ảnh màu xanh');
  h.element('remove-image').listeners.click();
  assert.equal(h.element('ask').disabled, true);
  assert.equal(h.element('copy-result').disabled, true);
  assert.equal(h.element('preview').hidden, true);
  assert.equal(h.element('history-list').children.length, 0);
});

test('drop and paste share validation; normal text paste is not intercepted', async () => {
  const h = await page();
  let prevented = 0;
  h.documentListeners.paste({clipboardData: {items: [{kind: 'string'}]},
    preventDefault() { prevented++; }});
  assert.equal(prevented, 0);
  h.element('drop-zone').listeners.drop({dataTransfer: {files: [
    {name: 'first', type: 'image/png', size: 100},
    {name: 'second', type: 'image/png', size: 100}]}, preventDefault() {}});
  assert.match(h.element('error').textContent, /một ảnh/);
  assert.equal(h.readers.length, 0);
  h.documentListeners.paste({clipboardData: {items: [{kind: 'file',
    getAsFile: () => ({name: 'paste', type: 'image/jpeg', size: 100})}]},
    preventDefault() { prevented++; }});
  assert.equal(prevented, 1);
  assert.equal(h.readers.length, 1);
});

test('busy controls block overlapping requests and restore after backend error', async () => {
  const h = await page();
  await h.upload({name: 'photo', type: 'image/png', size: 100});
  h.readers[0].onload(); h.question('Ảnh màu gì?');
  let finish;
  h.setFetch(() => new Promise(resolve => { finish = resolve; }));
  const pending = h.element('ask').listeners.click();
  assert.equal(h.element('loading').hidden, false);
  assert.equal(h.element('question').disabled, true);
  assert.equal(h.element('remove-image').disabled, true);
  await h.element('ask').listeners.click();
  assert.equal(h.calls.filter(([url]) => url === 'api/predict').length, 1);
  finish({ok: false, status: 503, json: async () => ({detail: 'GPU không sẵn sàng'})});
  await pending;
  assert.equal(h.element('loading').hidden, true);
  assert.equal(h.element('ask').disabled, false);
  assert.equal(h.element('history-list').children.length, 0);
  assert.match(h.element('error').textContent, /GPU không sẵn sàng/);
});

test('failed reads and unknown backend modes do not enable inference', async () => {
  const h = await page();
  await h.upload({name: 'photo', type: 'image/png', size: 100});
  h.readers[0].onerror(); h.question('Ảnh màu gì?');
  assert.equal(h.element('ask').disabled, true);
  assert.equal(h.element('preview').hidden, true);
  const bad = await page({...info, mode: 'unknown'});
  assert.equal(bad.element('question').disabled, true);
  assert.equal(bad.element('connection').dataset.state, 'error');
});
