const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const asset = path.join(__dirname, '../../app/validation_explorer/validation_explorer');
const source = fs.readFileSync(asset + '.js', 'utf8');
const html = fs.readFileSync(asset + '.html', 'utf8');
const labels = {object: 'Đối tượng', other: 'Khác'};
const transitions = {gain: 'M3 khớp thêm', loss: 'M3 mất khớp',
  both_match: 'Cùng khớp', neither_match: 'Cùng không khớp'};

function output(answer) {
  return {answer, explanation: 'Một lời giải thích', raw_output: '<script>not markup</script>',
    format_valid: true, latency_seconds: 1.25,
    scores: {exact_match: 1, token_f1: .8, rouge_l: .6, bertscore_f1: .7}};
}

function page(count = 3, hash = '') {
  const elements = new Map(), copied = [], documentListeners = {}, urls = [];
  function make() {
    return {textContent: '', value: '', hidden: false, disabled: false, children: [],
      listeners: {}, attributes: {},
      appendChild(child) { this.children.push(child); },
      replaceChildren() { this.children = []; },
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; },
    };
  }
  const element = id => {
    if (!elements.has(id)) elements.set(id, make());
    return elements.get(id);
  };
  for (const id of ['type', 'transition']) element(id).value = 'all';
  element('contrast').value = 'vs_m2';
  const cases = Array.from({length: count}, (_, i) => ({question_id: String(i),
    image_id: String(i), question: i === 0 ? 'Con mèo ở đâu?' : `Câu hỏi ${i}`,
    question_type: i % 2 ? 'other' : 'object', reference_answer: 'mèo',
    reference_explanations: ['Có một con mèo.'], thumbnail: `images/${i}.jpg`,
    baselines: {m0: output('M0 cố định'), m1: output('M1 cố định')},
    seeds: Object.fromEntries(['42', '43', '44'].map(seed => [seed, {
      m2: output(`M2-${seed}`), m3: output(`M3-${seed}`),
      vs_m2: i % 2 ? 'loss' : 'gain', vs_m1: i % 2 ? 'gain' : 'loss'}]))}));
  const context = vm.createContext({
    VALIDATION_DATA: {n: count, n_images: count, cases, type_labels: labels,
      transition_labels: transitions},
    document: {getElementById: element, createElement: make,
      addEventListener: (name, cb) => { documentListeners[name] = cb; }},
    URLSearchParams, location: {hash},
    history: {replaceState: (a, b, url) => urls.push(url)},
    navigator: {clipboard: {writeText: async text => copied.push(text)}},
  });
  vm.runInContext(source, context);
  return {element, copied, documentListeners, urls,
    change: (id, value) => {
      element(id).value = value;
      element(id).listeners[id === 'search' ? 'input' : 'change']({target: {value}});
    }};
}

test('all controls exist and the app cannot issue network/model requests', () => {
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
  assert.equal(new Set(ids).size, ids.length);
  for (const [, id] of source.matchAll(/byId\('([^']+)'\)/g)) assert.ok(ids.includes(id), id);
  assert.equal(source.includes('fetch('), false);
  assert.equal(source.includes('innerHTML'), false);
});

test('renders the four real output slots as text, including raw markup literally', () => {
  const h = page();
  assert.equal(h.element('cards').children.length, 4);
  const first = h.element('cards').children[0];
  assert.equal(first.children[0].textContent, 'M0 · Pretrained');
  assert.equal(first.children.at(-1).children[1].textContent, '<script>not markup</script>');
  assert.match(h.element('count').textContent, /3 \/ 3/);
});

test('changing training seed changes M2/M3 but keeps M0/M1 fixed', () => {
  const h = page();
  h.change('seed', '44');
  const cards = h.element('cards').children;
  assert.equal(cards[0].children[2].textContent, 'M0 cố định');
  assert.equal(cards[1].children[2].textContent, 'M1 cố định');
  assert.equal(cards[3].children[2].textContent, 'M3-44');
  assert.match(h.element('count').textContent, /seed 44/);
});

test('search supports unaccented Vietnamese and a zero-result state hides old outputs', () => {
  const h = page();
  h.change('search', 'con meo o dau');
  assert.match(h.element('count').textContent, /1 \/ 3/);
  h.change('search', 'no matching case');
  assert.equal(h.element('detail').hidden, true);
  assert.equal(h.element('empty').hidden, false);
  assert.equal(h.element('next').disabled, true);
});

test('transition filters use the selected M2 or M1 contrast', () => {
  const h = page();
  h.change('transition', 'gain');
  assert.match(h.element('count').textContent, /2 \/ 3/);
  h.change('contrast', 'vs_m1');
  assert.match(h.element('count').textContent, /1 \/ 3/);
  assert.match(h.element('qid').textContent, /Câu 1/);
});

test('pagination does not truncate the dataset after the first 40 cases', () => {
  const h = page(85, '#q=39&seed=43');
  assert.equal(h.element('case-list').children.length, 40);
  h.element('next').listeners.click();
  assert.match(h.element('qid').textContent, /Câu 40/);
  assert.equal(h.element('list-range').textContent, '41–80');
  assert.match(h.element('count').textContent, /85 \/ 85/);
});

test('copy and keyboard navigation preserve the selected seed and do not change inputs', async () => {
  const h = page();
  h.documentListeners.keydown({key: 'ArrowRight', target: {tagName: 'INPUT'}});
  assert.match(h.element('qid').textContent, /Câu 0/);
  h.documentListeners.keydown({key: 'ArrowRight', target: {tagName: 'BODY'}, preventDefault() {}});
  assert.match(h.element('qid').textContent, /Câu 1/);
  await h.element('copy').listeners.click();
  assert.match(h.copied[0], /seed 42/);
  assert.match(h.copied[0], /M3-42/);
});
