(() => {
  const sourceNotes = JSON.parse(document.getElementById('review-data').textContent);
  const notes = sourceNotes.map(row=>({...row}));
  const drafts = JSON.parse(document.getElementById('draft-data').textContent);
  const queue = JSON.parse(document.getElementById('queue-data').textContent);
  const advisories = JSON.parse(document.getElementById('advisory-data').textContent);
  const assistanceMode='{{assistance_mode}}';
  const storageKey='{{storage_key}}';
  const legacyStorageKeys={{legacy_payload}};
  const immutable=['schema','source_question_id','image_path','bundle_sha256',
                   'reference_answer','candidate_answer','candidate_explanation'];
  const noteIndex = new Map(notes.map((row,i)=>[row.source_question_id,i]));
  const queueIndices = queue.question_ids.map(id=>noteIndex.get(id));
  const draftsById = new Map(drafts.map(row=>[row.source_question_id,row]));
  const advisoriesById = new Map(advisories.map(row=>[row.source_question_id,row]));
  const draftIndices = queueIndices.filter(i=>draftsById.has(notes[i].source_question_id));
  const flaggedIndices = queueIndices.filter(i=>{
    const advisory=advisoriesById.get(notes[i].source_question_id);
    return advisory && advisory.ai_verdict!=='pass_grounded';
  });
  let dirty=false;
  const picker = document.getElementById('case-picker');
  const form = document.getElementById('review-form');
  const message = document.getElementById('review-message');
  const panel = document.getElementById('draft-panel');
  const triagePanel = document.getElementById('ai-triage-panel');
  const apply = document.getElementById('apply-draft');
  const progress = document.getElementById('queue-progress');
  const nextPending = document.getElementById('next-pending');
  const nextDraft = document.getElementById('next-draft');
  const nextFlagged = document.getElementById('next-ai-flagged');
  const revealAssistance = document.getElementById('reveal-assistance');
  const revealedAssistance = new Set();
  let saveTimer=null;
  nextDraft.hidden=assistanceMode==='on_request' || !draftIndices.length;
  nextFlagged.hidden=assistanceMode==='on_request' || !flaggedIndices.length;
  const options = {
    review_status: ['pending','accepted','rejected','uncertain'],
    answer_assessment: ['','correct','incorrect','uncertain'],
    explanation_support: ['','0','1','2'], canonical_answer_verified: ['','true','false'],
    error_type: ['','object','attribute','relation_action','natural_model_error'],
    rejected_origin_kind: ['','model','manual','synthetic'],
    chosen_origin_kind: ['','manual','synthetic'],
    reviewer_type: ['','human','ai']
  };
  const labels = {
    review_status:'Quyết định', reviewer:'Mã xác nhận (có thể ẩn danh)',
    reviewer_type:'Nguồn quyết định',
    answer_assessment:'Đáp án M1',
    explanation_support:'Giải thích M1 hỗ trợ đáp án đến đâu?',
    canonical_answer_verified:'Đáp án tham chiếu đúng theo ảnh?',
    error_type:'Loại lỗi của rejected', unsupported_span:'Cụm có vấn đề trong rejected',
    image_evidence:'Bằng chứng nhìn thấy trong ảnh',
    proposed_chosen_explanation:'Chosen: giải thích đúng',
    proposed_rejected_explanation:'Rejected: giải thích kém hơn',
    exclusion_reason:'Lý do loại hoặc chưa chắc (nếu có)',
    rejected_origin_kind:'Loại nguồn rejected',
    rejected_origin_source:'Tệp/ID nguồn rejected',
    rejected_recipe_sha256:'SHA-256 recipe rejected',
    chosen_origin_kind:'Loại nguồn chosen',
    chosen_origin_source:'Tệp/ID nguồn chosen',
    chosen_recipe_sha256:'SHA-256 recipe chosen'
  };
  const optionLabels = {
    review_status:{pending:'Chưa chốt',accepted:'Giữ cặp',rejected:'Loại',uncertain:'Chưa chắc'},
    answer_assessment:{correct:'Đúng',incorrect:'Sai',uncertain:'Chưa chắc'},
    explanation_support:{'0':'0 — Không hỗ trợ / mâu thuẫn','1':'1 — Hỗ trợ một phần',
      '2':'2 — Đủ bằng chứng liên quan'},
    canonical_answer_verified:{true:'Đúng',false:'Không đúng'},
    error_type:{object:'Đối tượng',attribute:'Thuộc tính',relation_action:'Quan hệ / hành động',
      natural_model_error:'Lỗi tự nhiên khác của M1'},
    rejected_origin_kind:{model:'M1 nguyên văn',manual:'Người viết / sửa',synthetic:'AI / recipe'},
    chosen_origin_kind:{manual:'Người viết / sửa',synthetic:'AI / recipe'},
    reviewer_type:{human:'Người kiểm',ai:'AI kiểm'}
  };
  const reviewSections = [
    {title:'1. Đánh giá theo ảnh',hint:'Đọc ảnh, câu hỏi và đầu ra M1 trước khi ghi nhãn.',
      fields:['answer_assessment','canonical_answer_verified','explanation_support','image_evidence']},
    {title:'2. Cặp giải thích',
      hint:'Chỉ giữ cặp khi cùng đáp án đúng và xác định được vấn đề của rejected.',
      fields:['proposed_chosen_explanation','proposed_rejected_explanation','unsupported_span','error_type']},
    {title:'3. Quyết định',hint:'Chưa chắc thì để lại; không cần ghi tên thật. '+
      'Mã xác nhận chỉ điền khi bạn đã kiểm.',
      fields:['review_status','reviewer','exclusion_reason']}
  ];
  const technicalFields = ['rejected_origin_kind','rejected_origin_source','rejected_recipe_sha256',
    'chosen_origin_kind','chosen_origin_source','chosen_recipe_sha256','reviewer_type'];
  function createReviewSections() {
    const containers=new Map();
    reviewSections.forEach(section=>{
      const group=document.createElement('fieldset');
      const legend=document.createElement('legend'); legend.textContent=section.title;
      const hint=document.createElement('p'); hint.className='section-hint';
      hint.textContent=section.hint;
      const fields=document.createElement('div'); fields.className='review-fields';
      group.append(legend,hint,fields); form.append(group);
      section.fields.forEach(key=>containers.set(key,fields));
    });
    const details=document.createElement('details'); details.id='review-provenance';
    const summary=document.createElement('summary');
    summary.textContent='Thông tin kỹ thuật — nguồn và mã kiểm tra';
    const hint=document.createElement('p'); hint.className='section-hint';
    hint.textContent='Không phải tiêu chí chấm. Nút Điền đề xuất tự điền nguồn và hash. '+
      'Nếu tự tạo bản synthetic, cần ghi nguồn và SHA-256 recipe; không tự đặt mã.';
    const fields=document.createElement('div'); fields.className='review-fields';
    details.append(summary,hint,fields); form.append(details);
    technicalFields.forEach(key=>containers.set(key,fields));
    return containers;
  }
  function validateRows(rows) {
    if (!Array.isArray(rows) || rows.length!==notes.length) {
      throw new Error('Số câu trong bản nháp không khớp');
    }
    rows.forEach((row,i)=>{ immutable.forEach(key=>{
      if (row[key]!==sourceNotes[i][key]) {
        throw new Error('Nguồn câu hoặc bundle trong bản nháp không khớp');
      }
    }); });
  }
  function replaceEditable(rows) {
    rows.forEach((row,i)=>Object.keys(notes[i]).filter(key=>!immutable.includes(key))
      .forEach(key=>{notes[i][key]=row[key];}));
  }
  function saveBrowserDraft() {
    if (saveTimer!==null) {clearTimeout(saveTimer); saveTimer=null;}
    try {localStorage.setItem(storageKey,JSON.stringify(notes));}
    catch (error) {
      message.textContent='Không lưu được nháp trong trình duyệt; hãy Xuất JSONL ngay.';
    }
  }
  function queueBrowserSave() {
    if (saveTimer!==null) clearTimeout(saveTimer);
    saveTimer=setTimeout(saveBrowserDraft,300);
  }
  function restoreBrowserDraft() {
    try {
      const saved=localStorage.getItem(storageKey);
      if (saved) {
        const rows=JSON.parse(saved); validateRows(rows); replaceEditable(rows);
        dirty=true; return true;
      }
      const candidates=[];
      for (const key of legacyStorageKeys) {
        const old=localStorage.getItem(key);
        if (!old) continue;
        try {
          const rows=JSON.parse(old); validateRows(rows); candidates.push(rows);
        } catch (_) {
          message.textContent='Có nháp cũ không khớp nguồn. Nháp vẫn được giữ ở trang cũ.';
        }
      }
      if (!candidates.length) return false;
      const signature=rows=>JSON.stringify(rows.map(row=>Object.keys(notes[0])
        .filter(key=>!immutable.includes(key)).map(key=>row[key] ?? null)));
      if (candidates.some(rows=>signature(rows)!==signature(candidates[0]))) {
        message.textContent='Có các nháp cũ khác nhau. '+
          'Mở trang cũ, Xuất JSONL rồi nhập bản cần dùng.';
        return false;
      }
      replaceEditable(candidates[0]); saveBrowserDraft();
      dirty=true; return true;
    } catch (error) {
      message.textContent='Không đọc được nháp trình duyệt. '+
        'Nháp vẫn được giữ; có thể nhập JSONL đã lưu.';
      return false;
    }
  }
  function refreshPicker() {
    const selected=picker.value;
    picker.replaceChildren();
    queueIndices.forEach((i,position) => {
      const row=notes[i];
      const item = document.createElement('option'); item.value=i;
      item.textContent=`${position+1}. ${row.source_question_id} — ${row.review_status}`;
      picker.append(item);
    });
    picker.value=selected!=='' && queueIndices.includes(Number(selected))
      ? selected : String(queueIndices[0]);
    updateProgress();
  }
  function updateProgress() {
    const counts={pending:0,accepted:0,rejected:0,uncertain:0}; let modified=0;
    queueIndices.forEach(i=>{
      counts[notes[i].review_status]++;
      if (notes[i].review_status==='pending' && notes[i].reviewed_at!==null) modified++;
    });
    progress.textContent=`${queue.label}: ${queueIndices.length} câu · `+
      `accepted ${counts.accepted} · rejected ${counts.rejected} · `+
      `uncertain ${counts.uncertain} · pending ${counts.pending}`+
      (modified ? ` (${modified} đang ghi dở)` : '');
  }
  function draw() {
    const i=Number(picker.value); const row=notes[i]; form.replaceChildren();
    form.dataset.questionId=row.source_question_id;
    const draft=draftsById.get(row.source_question_id);
    const advisory=advisoriesById.get(row.source_question_id);
    const assistanceRevealed=assistanceMode==='visible' ||
      revealedAssistance.has(row.source_question_id);
    revealAssistance.hidden=assistanceRevealed || (!draft && !advisory);
    triagePanel.hidden=!advisory || !assistanceRevealed;
    if (advisory) {
      document.getElementById('ai-triage-verdict').textContent=
        `Verdict AI: ${advisory.ai_verdict}` +
        (advisory.explanation_issue_ai ? ` · Nhóm: ${advisory.explanation_issue_ai}` : '');
      document.getElementById('ai-triage-rationale').textContent=advisory.rationale || '';
      const evidence=document.getElementById('ai-triage-evidence');
      evidence.hidden=!advisory.image_evidence_ai;
      evidence.textContent=advisory.image_evidence_ai
        ? `Dấu hiệu AI ghi lại: ${advisory.image_evidence_ai}` : '';
    }
    panel.hidden=!draft || !assistanceRevealed;
    if (draft) {
      document.getElementById('draft-text').textContent=
        `Chosen: ${draft.proposed_chosen_explanation}

`+
        `Rejected: ${draft.proposed_rejected_explanation}

`+
        `Nhận xét AI (cần kiểm ảnh): ${draft.image_evidence_ai}`;
      apply.disabled=!isBlank(row);
    }
    const context=document.getElementById('review-context'); context.replaceChildren();
    document.querySelectorAll('article').forEach((item,j) => {
      item.hidden=true;
      if (j===i) Array.from(item.children).forEach(child=>context.append(child.cloneNode(true)));
    });
    const containers=createReviewSections();
    const fieldOrder=reviewSections.flatMap(section=>section.fields).concat(technicalFields);
    fieldOrder.forEach(key => {
      const label=labels[key];
      const wrapper=document.createElement('label'); wrapper.textContent=label;
      if (['image_evidence','exclusion_reason'].includes(key)) wrapper.className='wide-field';
      const singleLine=key==='reviewer' || key.endsWith('_source') || key.endsWith('_sha256');
      const input=document.createElement(
        options[key] ? 'select' : singleLine ? 'input' : 'textarea');
      if (singleLine) input.type='text';
      input.id='field-'+key;
      if (options[key]) options[key].forEach(value => {
        const item=document.createElement('option'); item.value=value;
        item.textContent=optionLabels[key]?.[value] || value || 'Chưa ghi'; input.append(item);
      });
      input.value=row[key] == null ? '' : String(row[key]);
      input.addEventListener('input', () => {
        picker.value=String(i);
        let value=input.value.trim() || null;
        if (key==='explanation_support' && value!==null) value=Number(value);
        if (key==='canonical_answer_verified' && value!==null) value=value==='true';
        row[key]=value; row.reviewed_at=new Date().toISOString(); dirty=true;
        apply.disabled=true;
        picker.selectedOptions[0].textContent=
          `${picker.selectedIndex+1}. ${row.source_question_id} — ${row.review_status}`;
        updateProgress();
        queueBrowserSave();
        message.textContent='Đã lưu nháp trong trình duyệt; hãy Xuất JSONL để lưu bền vững.';
      }); wrapper.append(input); containers.get(key).append(wrapper);
    });
  }
  function isBlank(row) {
    return row.review_status==='pending' && row.reviewed_at==null &&
      Object.keys(labels).filter(key=>key!=='review_status').every(key=>row[key]==null);
  }
  nextDraft.addEventListener('click',()=>{
    const i=Number(picker.value);
    picker.value=draftIndices.find(index=>index>i) ?? draftIndices[0]; draw();
  });
  nextFlagged.addEventListener('click',()=>{
    const i=Number(picker.value);
    picker.value=flaggedIndices.find(index=>index>i) ?? flaggedIndices[0]; draw();
  });
  nextPending.addEventListener('click',()=>{
    const current=queueIndices.indexOf(Number(picker.value));
    const ordered=queueIndices.slice(current+1).concat(queueIndices.slice(0,current+1));
    const target=ordered.find(index=>notes[index].review_status==='pending');
    if (target===undefined) {
      message.textContent='Queue không còn câu pending.'; return;
    }
    picker.value=target; draw();
  });
  revealAssistance.addEventListener('click',()=>{
    const row=notes[Number(picker.value)];
    revealedAssistance.add(row.source_question_id); draw();
    message.textContent='Đã mở hỗ trợ AI cho riêng câu này. '+
      'Đây vẫn là gợi ý, không phải nhãn người.';
  });
  apply.addEventListener('click',()=>{
    const row=notes[Number(picker.value)]; const draft=draftsById.get(row.source_question_id);
    if (!draft || !isBlank(row)) return;
    ['proposed_chosen_explanation','proposed_rejected_explanation','unsupported_span','error_type']
      .forEach(key=>{row[key]=draft[key];});
    ['chosen','rejected'].forEach(side=>{
      const origin=draft[side+'_provenance'];
      if (side==='rejected' && draft.origin_kind==='model') return;
      row[side+'_origin_kind']=origin.kind;
      row[side+'_origin_source']=origin.source;
      row[side+'_recipe_sha256']=origin.recipe_sha256;
    });
    dirty=true; saveBrowserDraft(); draw();
    message.textContent='Đã điền bản nháp AI, vẫn pending. '+
      'Kiểm ảnh và tự ghi đánh giá; xuất JSONL để lưu.';
  });
  window.addEventListener('beforeunload',event=>{
    if (dirty) { saveBrowserDraft(); event.preventDefault(); event.returnValue=''; }
  });
  function syncSelection() {
    const i=Number(picker.value);
    if (!queueIndices.includes(i)) picker.value=String(queueIndices[0]);
    if (form.dataset.questionId!==notes[Number(picker.value)].source_question_id) draw();
  }
  picker.addEventListener('input',syncSelection);
  picker.addEventListener('change',syncSelection);
  window.addEventListener('pageshow',syncSelection);
  document.getElementById('export-notes').addEventListener('click', () => {
    const content=notes.map(row=>JSON.stringify(row)).join('\n')+'\n';
    const url=URL.createObjectURL(new Blob([content],{type:'application/x-ndjson'}));
    const link=document.createElement('a'); link.href=url; link.download='reviewed_notes.jsonl';
    link.click(); dirty=false; setTimeout(()=>URL.revokeObjectURL(url),1000);
    message.textContent='Đã yêu cầu tải file nhãn. Bản nháp trình duyệt vẫn được giữ.';
  });
  document.getElementById('import-notes').addEventListener('change',async event => {
    try {
      const text=await event.target.files[0].text();
      const rows=text.split(/\r?\n/).filter(line=>line.trim()).map(line=>JSON.parse(line));
      if (rows.length!==notes.length) throw new Error('Số câu không khớp');
      if (dirty && !window.confirm('Nhập file sẽ thay phần chưa xuất. Tiếp tục?')) return;
      validateRows(rows); replaceEditable(rows);
      dirty=false; saveBrowserDraft(); refreshPicker(); draw();
      message.textContent='Đã nhập file; chưa xác nhận nhãn hợp lệ.';
    } catch (error) { message.textContent=error.message; }
  });
  const restored=restoreBrowserDraft(); refreshPicker();
  if (document.readyState==='loading') document.addEventListener('DOMContentLoaded',draw);
  else draw();
  if (restored) message.textContent='Đã khôi phục bản nháp lưu trong trình duyệt. '+
    'Xuất JSONL trước khi chuyển máy hoặc đổi trình duyệt.';
})();
