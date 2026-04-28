// LeRobot Annotate — frontend.
//
// Aligned with the v3.1 language schema (lerobot#3467) and the steerable
// annotation pipeline conventions (lerobot#3471). The annotation model is a
// flat list of "atoms" per episode, each shaped like:
//
//   { role, content, style, timestamp, tool_calls }
//
// where:
//   - style ∈ {"subtask","plan","memory"}  → routed to language_persistent
//   - style ∈ {"interjection","vqa"}        → routed to language_events
//   - style === null + role==="assistant"   → speech tool-call atom (events)
//
// VQA atoms can carry bounding boxes / points / counts / attributes / spatial
// relations — the assistant's `content` is a JSON string in the schema used by
// the steerable pipeline writer (see `_normalize_atom` server-side).

// ---------------------------------------------------------------------------
// Push-to-Hub status helpers (kept identical in spirit to the original tool)
// ---------------------------------------------------------------------------

function showPushStatus(type, message, url = null) {
  const statusEl = document.getElementById('pushHubStatus');
  if (!statusEl) {
    alert(`${type}: ${message}`);
    return;
  }
  statusEl.className = `helper status-${type}`;
  if (type === 'loading') {
    statusEl.innerHTML = `<span class="spinner"></span> ${message}`;
  } else if (type === 'success') {
    statusEl.innerHTML = `
      <div class="status-box status-success">
        <span class="status-icon">✓</span>
        <div class="status-content">
          <strong>Success!</strong>
          <p>${message}</p>
          ${url ? `<a href="${url}" target="_blank" class="status-link">View on Hugging Face Hub →</a>` : ''}
        </div>
      </div>`;
  } else if (type === 'error') {
    statusEl.innerHTML = `
      <div class="status-box status-error">
        <span class="status-icon">✗</span>
        <div class="status-content">
          <strong>Error</strong>
          <p>${message}</p>
        </div>
      </div>`;
  }
}

async function handlePushToHub() {
  const token = (document.getElementById('hfToken').value || '').trim();
  if (!token) {
    showPushStatus('error', 'Please enter your Hugging Face token');
    return;
  }
  const inPlace = document.getElementById('pushInPlace').checked;
  const newRepoIdValue = (document.getElementById('newRepoId').value || '').trim();
  if (!inPlace && !newRepoIdValue) {
    showPushStatus('error', 'Please enter a new repo ID or check "Push to original repo"');
    return;
  }
  const btnEl = document.getElementById('pushHubBtn');
  showPushStatus('loading', 'Pushing to Hub… This may take a while for large datasets.');
  btnEl.disabled = true;
  btnEl.innerHTML = '<span class="spinner"></span> Pushing...';
  try {
    const res = await fetch('/api/push_to_hub', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        hf_token: token,
        push_in_place: inPlace,
        new_repo_id: inPlace ? null : newRepoIdValue,
        private: document.getElementById('privateRepo').checked,
        commit_message:
          (document.getElementById('commitMessage').value || '').trim() ||
          'Add language annotations from LeRobot Annotate',
      }),
    });
    const data = await res.json();
    if (res.ok) {
      showPushStatus('success', data.message, data.url);
    } else {
      showPushStatus('error', data.detail || 'Push failed.');
    }
  } catch (err) {
    showPushStatus('error', `Network error: ${err.message}`);
  } finally {
    btnEl.disabled = false;
    btnEl.textContent = 'Push to Hub';
  }
}

// ---------------------------------------------------------------------------
// State and DOM refs
// ---------------------------------------------------------------------------

const state = {
  dataset: null,
  episodes: [],
  currentEpisode: null,
  // Per-episode list of atoms.
  // Each atom: { role, content, style, timestamp, tool_calls }
  atomsByEpisode: {},
  // Per-episode sorted frame-timestamp array (cached lazily).
  frameTimestamps: {},
  // Pending VQA drawing: 'bbox' uses [x1,y1,x2,y2] in image-relative
  // (0..1) coordinates; 'keypoint' uses [x,y] in 0..1.
  pendingDraw: null,
};

const $ = (id) => document.getElementById(id);

const els = {
  status: $('status'),
  connectForm: $('connectForm'),
  sourceSelect: $('sourceSelect'),
  repoInput: $('repoInput'),
  localInput: $('localInput'),
  revisionInput: $('revisionInput'),
  videoKeySelect: $('videoKeySelect'),
  connectHelper: $('connectHelper'),
  workspace: $('workspace'),
  episodeList: $('episodeList'),
  episodeSearch: $('episodeSearch'),
  episodeTitle: $('episodeTitle'),
  episodeMeta: $('episodeMeta'),
  episodeVideo: $('episodeVideo'),
  vqaCanvas: $('vqaCanvas'),
  timeline: $('timeline'),
  saveEpisode: $('saveEpisode'),
  resetEpisode: $('resetEpisode'),
  currentTimeDisplay: $('currentTimeDisplay'),
  totalTimeDisplay: $('totalTimeDisplay'),
  // Subtask
  subtaskStart: $('subtaskStart'),
  subtaskLabel: $('subtaskLabel'),
  subtaskSetStart: $('subtaskSetStart'),
  addSubtask: $('addSubtask'),
  subtaskList: $('subtaskList'),
  // Plan
  planStart: $('planStart'),
  planContent: $('planContent'),
  planSetStart: $('planSetStart'),
  addPlan: $('addPlan'),
  planList: $('planList'),
  // Memory
  memoryStart: $('memoryStart'),
  memoryContent: $('memoryContent'),
  memorySetStart: $('memorySetStart'),
  addMemory: $('addMemory'),
  memoryList: $('memoryList'),
  // Interjections + speech
  interjectionStart: $('interjectionStart'),
  interjectionUser: $('interjectionUser'),
  interjectionSpeech: $('interjectionSpeech'),
  interjectionSetStart: $('interjectionSetStart'),
  addInterjection: $('addInterjection'),
  interjectionList: $('interjectionList'),
  // VQA
  vqaType: $('vqaType'),
  vqaStart: $('vqaStart'),
  vqaSetStart: $('vqaSetStart'),
  vqaPause: $('vqaPause'),
  vqaQuestion: $('vqaQuestion'),
  vqaLabel: $('vqaLabel'),
  vqaDrawHint: $('vqaDrawHint'),
  vqaDrawState: $('vqaDrawState'),
  vqaClearDraw: $('vqaClearDraw'),
  vqaCountLabel: $('vqaCountLabel'),
  vqaCount: $('vqaCount'),
  vqaCountNote: $('vqaCountNote'),
  vqaAttrLabel: $('vqaAttrLabel'),
  vqaAttribute: $('vqaAttribute'),
  vqaAttrValue: $('vqaAttrValue'),
  vqaSubject: $('vqaSubject'),
  vqaRelation: $('vqaRelation'),
  vqaObject: $('vqaObject'),
  addVqa: $('addVqa'),
  vqaList: $('vqaList'),
  // Export
  exportBtn: $('exportBtn'),
  outputDir: $('outputDir'),
  copyVideos: $('copyVideos'),
  exportStatus: $('exportStatus'),
  pushInPlace: $('pushInPlace'),
  newRepoRow: $('newRepoRow'),
  pushHubBtn: $('pushHubBtn'),
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PERSISTENT_STYLES = new Set(['subtask', 'plan', 'memory']);
const EVENT_STYLES = new Set(['interjection', 'vqa']);

function setStatus(text, ok = false) {
  els.status.textContent = text;
  els.status.style.color = ok ? '#22c55e' : '#f97316';
}

function setHelper(el, message, ok = false) {
  if (!el) return;
  el.textContent = message;
  el.style.color = ok ? '#22c55e' : '#94a3b8';
}

function formatDuration(seconds) {
  if (seconds == null) return '';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}m ${secs}s`;
}

function formatTimeWithMs(seconds) {
  if (seconds == null) return '00:00.000';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 1000);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
}

function currentTime() {
  return Number(els.episodeVideo.currentTime.toFixed(3));
}

function snapToFrame(epIdx, ts) {
  const arr = state.frameTimestamps[epIdx];
  if (!arr || !arr.length) return Number(ts);
  // Linear scan; episodes are typically a few thousand frames.
  let best = arr[0];
  let bestDist = Math.abs(ts - best);
  for (let i = 1; i < arr.length; i++) {
    const d = Math.abs(ts - arr[i]);
    if (d < bestDist) {
      bestDist = d;
      best = arr[i];
    }
  }
  return best;
}

function getAtoms(epIdx) {
  if (!state.atomsByEpisode[epIdx]) state.atomsByEpisode[epIdx] = [];
  return state.atomsByEpisode[epIdx];
}

function atomsByStyle(epIdx, style) {
  return getAtoms(epIdx).filter((a) => a.style === style);
}

function speechAtoms(epIdx) {
  return getAtoms(epIdx).filter(
    (a) => a.style == null && a.role === 'assistant' && a.tool_calls && a.tool_calls.length,
  );
}

function deleteAtom(epIdx, atom) {
  const arr = getAtoms(epIdx);
  const idx = arr.indexOf(atom);
  if (idx >= 0) arr.splice(idx, 1);
  renderAll();
}

// ---------------------------------------------------------------------------
// Episode list and selection
// ---------------------------------------------------------------------------

function renderEpisodes() {
  els.episodeList.innerHTML = '';
  const query = els.episodeSearch.value.trim();
  const filtered = state.episodes.filter((ep) => ep.episode_index.toString().includes(query));
  filtered.forEach((ep) => {
    const li = document.createElement('li');
    li.textContent = `Episode ${ep.episode_index}`;
    const span = document.createElement('span');
    span.textContent = formatDuration(ep.duration);
    li.appendChild(span);
    if (state.currentEpisode === ep.episode_index) li.classList.add('active');
    li.addEventListener('click', () => selectEpisode(ep.episode_index));
    els.episodeList.appendChild(li);
  });
}

async function selectEpisode(epIdx) {
  state.currentEpisode = epIdx;
  els.episodeTitle.textContent = `Episode ${epIdx}`;
  const ep = state.episodes.find((e) => e.episode_index === epIdx);
  els.episodeMeta.textContent = ep ? `${ep.length} frames • ${formatDuration(ep.duration)}` : '';

  // Load saved annotations.
  try {
    const res = await fetch(`/api/episodes/${epIdx}/annotations`);
    const data = await res.json();
    state.atomsByEpisode[epIdx] = (data.atoms || []).slice();
  } catch (e) {
    state.atomsByEpisode[epIdx] = [];
  }

  // Load frame timestamps in the background — needed for snapping events.
  if (!state.frameTimestamps[epIdx]) {
    fetch(`/api/episodes/${epIdx}/frame_timestamps`)
      .then((r) => r.json())
      .then((d) => {
        state.frameTimestamps[epIdx] = (d.timestamps || []).slice().sort((a, b) => a - b);
      })
      .catch(() => {});
  }

  // Switch the video.
  const videoUrl = `/api/video/${epIdx}?video_key=${encodeURIComponent(state.dataset.selected_video_key)}`;
  els.episodeVideo.src = videoUrl;
  state.pendingDraw = null;

  resetEpisodeForms();
  renderAll();
}

async function saveEpisode() {
  if (state.currentEpisode == null) return;
  const atoms = getAtoms(state.currentEpisode);
  const res = await fetch(`/api/episodes/${state.currentEpisode}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ episode_index: state.currentEpisode, atoms }),
  });
  if (res.ok) {
    setHelper(els.connectHelper, `Saved ${atoms.length} atoms for episode ${state.currentEpisode}.`, true);
  } else {
    const data = await res.json().catch(() => ({}));
    setHelper(els.connectHelper, `Save failed: ${data.detail || res.status}`);
  }
}

function resetEpisodeForms() {
  els.subtaskStart.value = '';
  els.subtaskLabel.value = '';
  els.planStart.value = '';
  els.planContent.value = '';
  els.memoryStart.value = '';
  els.memoryContent.value = '';
  els.interjectionStart.value = '';
  els.interjectionUser.value = '';
  els.interjectionSpeech.value = '';
  els.vqaStart.value = '';
  els.vqaQuestion.value = '';
  els.vqaLabel.value = '';
  els.vqaCountLabel.value = '';
  els.vqaCount.value = '';
  els.vqaCountNote.value = '';
  els.vqaAttrLabel.value = '';
  els.vqaAttribute.value = '';
  els.vqaAttrValue.value = '';
  els.vqaSubject.value = '';
  els.vqaRelation.value = '';
  els.vqaObject.value = '';
  state.pendingDraw = null;
  setHelper(els.vqaDrawState, 'No annotation drawn yet.');
  redrawCanvas();
}

// ---------------------------------------------------------------------------
// Renderers per style
// ---------------------------------------------------------------------------

function renderAll() {
  renderEpisodes();
  renderTimeline();
  renderSubtasks();
  renderPlans();
  renderMemory();
  renderInterjections();
  renderVqa();
  redrawCanvas();
}

function renderTimeline() {
  els.timeline.innerHTML = '';
  if (state.currentEpisode == null) return;
  const duration = els.episodeVideo.duration || 0;
  if (!duration) return;

  const persistent = getAtoms(state.currentEpisode)
    .filter((a) => PERSISTENT_STYLES.has(a.style))
    .slice()
    .sort((a, b) => a.timestamp - b.timestamp);

  // Order subtask atoms first (they are the most common timeline anchor).
  const colorByStyle = { subtask: 'var(--accent-2)', plan: '#a78bfa', memory: '#34d399' };

  // Each persistent atom gets a marker; subtask atoms also get a span up to
  // the next subtask boundary so the timeline still reads like a segment list.
  const subtasks = persistent.filter((a) => a.style === 'subtask');
  subtasks.forEach((seg, i) => {
    const start = seg.timestamp;
    const end = i + 1 < subtasks.length ? subtasks[i + 1].timestamp : duration;
    const left = (start / duration) * 100;
    const width = Math.max(((end - start) / duration) * 100, 2);
    const span = document.createElement('span');
    span.style.position = 'absolute';
    span.style.left = `${left}%`;
    span.style.width = `${width}%`;
    span.style.background = colorByStyle.subtask;
    span.title = `subtask @ ${start.toFixed(3)}s: ${seg.content || ''}`;
    span.textContent = `${i}`;
    span.style.display = 'flex';
    span.style.alignItems = 'center';
    span.style.justifyContent = 'center';
    span.style.fontSize = '10px';
    span.style.fontWeight = '600';
    span.style.color = '#0b0e14';
    els.timeline.appendChild(span);
  });

  persistent
    .filter((a) => a.style !== 'subtask')
    .forEach((atom) => {
      const left = (atom.timestamp / duration) * 100;
      const tick = document.createElement('span');
      tick.className = 'timeline-tick';
      tick.style.left = `${left}%`;
      tick.style.background = colorByStyle[atom.style] || '#fff';
      tick.title = `${atom.style} @ ${atom.timestamp.toFixed(3)}s: ${atom.content || ''}`;
      els.timeline.appendChild(tick);
    });
}

function renderListGeneric(listEl, atoms, renderRow) {
  listEl.innerHTML = '';
  atoms.slice().sort((a, b) => a.timestamp - b.timestamp).forEach((atom) => {
    const row = document.createElement('div');
    row.className = 'segment-item';
    renderRow(row, atom);
    listEl.appendChild(row);
  });
}

function tsInput(atom, onChange) {
  const inp = document.createElement('input');
  inp.type = 'number';
  inp.step = '0.001';
  inp.value = atom.timestamp;
  inp.addEventListener('change', () => {
    atom.timestamp = Number(inp.value);
    onChange();
  });
  return inp;
}

function deleteBtn(atom) {
  const btn = document.createElement('button');
  btn.className = 'ghost';
  btn.textContent = 'Delete';
  btn.addEventListener('click', () => deleteAtom(state.currentEpisode, atom));
  return btn;
}

function styleBadge(style) {
  const badge = document.createElement('span');
  badge.className = `style-badge style-${style || 'speech'}`;
  badge.textContent = style || 'speech';
  return badge;
}

function renderSubtasks() {
  if (state.currentEpisode == null) return;
  const atoms = atomsByStyle(state.currentEpisode, 'subtask');
  renderListGeneric(els.subtaskList, atoms, (row, atom) => {
    row.appendChild(styleBadge('subtask'));
    row.appendChild(tsInput(atom, () => renderTimeline()));
    const text = document.createElement('input');
    text.type = 'text';
    text.value = atom.content || '';
    text.addEventListener('change', () => { atom.content = text.value; });
    text.classList.add('grow');
    row.appendChild(text);
    row.appendChild(deleteBtn(atom));
  });
}

function renderPlans() {
  if (state.currentEpisode == null) return;
  const atoms = atomsByStyle(state.currentEpisode, 'plan');
  renderListGeneric(els.planList, atoms, (row, atom) => {
    row.appendChild(styleBadge('plan'));
    row.appendChild(tsInput(atom, () => renderTimeline()));
    const ta = document.createElement('textarea');
    ta.rows = 2;
    ta.value = atom.content || '';
    ta.addEventListener('change', () => { atom.content = ta.value; });
    ta.classList.add('grow');
    row.appendChild(ta);
    row.appendChild(deleteBtn(atom));
  });
}

function renderMemory() {
  if (state.currentEpisode == null) return;
  const atoms = atomsByStyle(state.currentEpisode, 'memory');
  renderListGeneric(els.memoryList, atoms, (row, atom) => {
    row.appendChild(styleBadge('memory'));
    row.appendChild(tsInput(atom, () => renderTimeline()));
    const ta = document.createElement('textarea');
    ta.rows = 2;
    ta.value = atom.content || '';
    ta.addEventListener('change', () => { atom.content = ta.value; });
    ta.classList.add('grow');
    row.appendChild(ta);
    row.appendChild(deleteBtn(atom));
  });
}

function renderInterjections() {
  if (state.currentEpisode == null) return;
  // Pair user 'interjection' rows and assistant speech rows by timestamp.
  const byTs = new Map();
  for (const a of getAtoms(state.currentEpisode)) {
    if (a.style === 'interjection' && a.role === 'user') {
      const k = a.timestamp;
      if (!byTs.has(k)) byTs.set(k, { user: null, speech: null });
      byTs.get(k).user = a;
    } else if (a.style == null && a.role === 'assistant' && a.tool_calls) {
      const k = a.timestamp;
      if (!byTs.has(k)) byTs.set(k, { user: null, speech: null });
      byTs.get(k).speech = a;
    }
  }
  els.interjectionList.innerHTML = '';
  Array.from(byTs.entries())
    .sort((a, b) => a[0] - b[0])
    .forEach(([ts, pair]) => {
      const row = document.createElement('div');
      row.className = 'segment-item';
      const badgeWrap = document.createElement('div');
      badgeWrap.className = 'badge-stack';
      if (pair.user) badgeWrap.appendChild(styleBadge('interjection'));
      if (pair.speech) badgeWrap.appendChild(styleBadge(null));
      row.appendChild(badgeWrap);

      const tsEl = document.createElement('input');
      tsEl.type = 'number';
      tsEl.step = '0.001';
      tsEl.value = ts;
      tsEl.addEventListener('change', () => {
        const newTs = snapToFrame(state.currentEpisode, Number(tsEl.value));
        if (pair.user) pair.user.timestamp = newTs;
        if (pair.speech) pair.speech.timestamp = newTs;
        renderInterjections();
        renderTimeline();
      });
      row.appendChild(tsEl);

      const userInput = document.createElement('input');
      userInput.type = 'text';
      userInput.placeholder = 'user interjection (optional)';
      userInput.value = pair.user?.content || '';
      userInput.classList.add('grow');
      userInput.addEventListener('change', () => {
        const v = userInput.value.trim();
        if (v) {
          if (pair.user) pair.user.content = v;
          else getAtoms(state.currentEpisode).push({
            role: 'user', content: v, style: 'interjection', timestamp: ts, tool_calls: null,
          });
        } else if (pair.user) {
          deleteAtom(state.currentEpisode, pair.user);
        }
      });
      row.appendChild(userInput);

      const speechInput = document.createElement('input');
      speechInput.type = 'text';
      speechInput.placeholder = 'robot speech (say tool call)';
      speechInput.value = pair.speech?.tool_calls?.[0]?.function?.arguments?.text || '';
      speechInput.classList.add('grow');
      speechInput.addEventListener('change', () => {
        const v = speechInput.value.trim();
        if (v) {
          if (pair.speech) {
            pair.speech.tool_calls[0].function.arguments.text = v;
          } else {
            getAtoms(state.currentEpisode).push(buildSpeechAtom(ts, v));
          }
        } else if (pair.speech) {
          deleteAtom(state.currentEpisode, pair.speech);
        }
      });
      row.appendChild(speechInput);

      const del = document.createElement('button');
      del.className = 'ghost';
      del.textContent = 'Delete';
      del.addEventListener('click', () => {
        if (pair.user) deleteAtom(state.currentEpisode, pair.user);
        if (pair.speech) deleteAtom(state.currentEpisode, pair.speech);
      });
      row.appendChild(del);
      els.interjectionList.appendChild(row);
    });
}

function renderVqa() {
  if (state.currentEpisode == null) return;
  const atoms = getAtoms(state.currentEpisode).filter(
    (a) => a.style === 'vqa' && a.role === 'assistant',
  );
  // Pair with the user question at the same timestamp if present.
  const byTs = new Map();
  for (const a of getAtoms(state.currentEpisode)) {
    if (a.style !== 'vqa') continue;
    if (!byTs.has(a.timestamp)) byTs.set(a.timestamp, { user: null, assistant: null });
    if (a.role === 'user') byTs.get(a.timestamp).user = a;
    else if (a.role === 'assistant') byTs.get(a.timestamp).assistant = a;
  }

  els.vqaList.innerHTML = '';
  Array.from(byTs.entries())
    .sort((a, b) => a[0] - b[0])
    .forEach(([ts, pair]) => {
      const row = document.createElement('div');
      row.className = 'segment-item vqa-row';
      row.appendChild(styleBadge('vqa'));

      const tsEl = document.createElement('input');
      tsEl.type = 'number';
      tsEl.step = '0.001';
      tsEl.value = ts;
      tsEl.addEventListener('change', () => {
        const newTs = snapToFrame(state.currentEpisode, Number(tsEl.value));
        if (pair.user) pair.user.timestamp = newTs;
        if (pair.assistant) pair.assistant.timestamp = newTs;
        renderVqa();
        renderTimeline();
      });
      row.appendChild(tsEl);

      const summary = document.createElement('div');
      summary.className = 'vqa-summary grow';
      const q = pair.user?.content || '(no question)';
      let parsed = null;
      try { parsed = JSON.parse(pair.assistant?.content || 'null'); } catch (e) { parsed = null; }
      const kind = vqaShapeKind(parsed);
      const subtype = document.createElement('span');
      subtype.className = `vqa-kind vqa-kind-${kind || 'unknown'}`;
      subtype.textContent = kind || '?';
      summary.appendChild(subtype);
      const txt = document.createElement('span');
      txt.textContent = `Q: ${q}`;
      summary.appendChild(txt);
      const ans = document.createElement('code');
      ans.className = 'vqa-answer';
      ans.textContent = pair.assistant?.content || '';
      summary.appendChild(ans);

      row.appendChild(summary);

      const seek = document.createElement('button');
      seek.className = 'ghost';
      seek.textContent = 'Jump';
      seek.addEventListener('click', () => {
        els.episodeVideo.currentTime = ts;
      });
      row.appendChild(seek);

      const del = document.createElement('button');
      del.className = 'ghost';
      del.textContent = 'Delete';
      del.addEventListener('click', () => {
        if (pair.user) deleteAtom(state.currentEpisode, pair.user);
        if (pair.assistant) deleteAtom(state.currentEpisode, pair.assistant);
      });
      row.appendChild(del);
      els.vqaList.appendChild(row);
    });
}

function vqaShapeKind(payload) {
  if (!payload || typeof payload !== 'object') return null;
  if (Array.isArray(payload.detections)) return 'bbox';
  if (payload.point_format && payload.point) return 'keypoint';
  if (payload.count != null && payload.label) return 'count';
  if (payload.attribute && payload.value) return 'attribute';
  if (payload.subject && payload.relation && payload.object) return 'spatial';
  return null;
}

// ---------------------------------------------------------------------------
// Speech atom builder
// ---------------------------------------------------------------------------

function buildSpeechAtom(timestamp, text) {
  return {
    role: 'assistant',
    content: null,
    style: null,
    timestamp,
    tool_calls: [
      { type: 'function', function: { name: 'say', arguments: { text } } },
    ],
  };
}

// ---------------------------------------------------------------------------
// VQA canvas — draw on the video, render saved bboxes/points during playback.
// ---------------------------------------------------------------------------

function resizeCanvas() {
  const v = els.episodeVideo;
  const c = els.vqaCanvas;
  // Match the canvas's bitmap to the rendered video size (CSS pixels).
  const rect = v.getBoundingClientRect();
  c.width = Math.max(1, Math.round(rect.width));
  c.height = Math.max(1, Math.round(rect.height));
  redrawCanvas();
}

function redrawCanvas() {
  const c = els.vqaCanvas;
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (state.currentEpisode == null) return;

  // 1) Active overlays from saved VQA atoms whose timestamp is near current frame.
  const vidT = els.episodeVideo.currentTime || 0;
  const window = 1 / Math.max(1, (state.dataset?.fps || 30));
  const matches = getAtoms(state.currentEpisode).filter(
    (a) => a.style === 'vqa' && a.role === 'assistant' && Math.abs(a.timestamp - vidT) <= window * 0.75,
  );
  for (const atom of matches) {
    let payload = null;
    try { payload = JSON.parse(atom.content || 'null'); } catch (e) { continue; }
    drawAnswerOverlay(ctx, payload);
  }

  // 2) Pending (in-progress) drawing.
  if (state.pendingDraw) drawPending(ctx, state.pendingDraw);
}

function drawAnswerOverlay(ctx, payload) {
  if (!payload) return;
  ctx.save();
  if (Array.isArray(payload.detections)) {
    for (const det of payload.detections) {
      if (!det || !Array.isArray(det.bbox)) continue;
      drawBboxOnCanvas(ctx, det.bbox, det.bbox_format || 'xyxy', det.label || '');
    }
  } else if (payload.point_format && Array.isArray(payload.point)) {
    drawPointOnCanvas(ctx, payload.point, payload.point_format, payload.label || '');
  }
  ctx.restore();
}

function drawBboxOnCanvas(ctx, bbox, fmt, label) {
  // bbox is in 0..1 image-relative coordinates. Convert to canvas pixels.
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  let x1, y1, x2, y2;
  if (fmt === 'xywh') {
    x1 = bbox[0]; y1 = bbox[1]; x2 = bbox[0] + bbox[2]; y2 = bbox[1] + bbox[3];
  } else {
    x1 = bbox[0]; y1 = bbox[1]; x2 = bbox[2]; y2 = bbox[3];
  }
  // Heuristic: if any value > 1.5, assume pixel coordinates relative to the
  // *original* video pixel space. Scale by the video's intrinsic resolution.
  const v = els.episodeVideo;
  let sx = w, sy = h;
  if (Math.max(Math.abs(x1), Math.abs(x2)) > 1.5 || Math.max(Math.abs(y1), Math.abs(y2)) > 1.5) {
    sx = w / Math.max(1, v.videoWidth || w);
    sy = h / Math.max(1, v.videoHeight || h);
    x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy;
  } else {
    x1 *= w; y1 *= h; x2 *= w; y2 *= h;
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#22d3ee';
  ctx.fillStyle = 'rgba(34,211,238,0.15)';
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  if (label) {
    ctx.font = '12px ui-sans-serif, system-ui';
    const padX = 4, padY = 2;
    const m = ctx.measureText(label);
    ctx.fillStyle = '#0b0e14';
    ctx.fillRect(x1, y1 - 16, m.width + padX * 2, 16);
    ctx.fillStyle = '#22d3ee';
    ctx.fillText(label, x1 + padX, y1 - padY - 2);
  }
}

function drawPointOnCanvas(ctx, point, fmt, label) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  let x = point[0], y = point[1];
  const v = els.episodeVideo;
  if (Math.abs(x) > 1.5 || Math.abs(y) > 1.5) {
    x *= w / Math.max(1, v.videoWidth || w);
    y *= h / Math.max(1, v.videoHeight || h);
  } else {
    x *= w; y *= h;
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#facc15';
  ctx.fillStyle = '#facc15';
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.stroke();
  if (label) {
    ctx.font = '12px ui-sans-serif, system-ui';
    ctx.fillStyle = '#0b0e14';
    const m = ctx.measureText(label);
    ctx.fillRect(x + 10, y - 18, m.width + 8, 16);
    ctx.fillStyle = '#facc15';
    ctx.fillText(label, x + 14, y - 6);
  }
}

function drawPending(ctx, draw) {
  if (draw.kind === 'bbox' && draw.bbox) {
    drawBboxOnCanvas(ctx, draw.bbox, 'xyxy', draw.label || '');
  } else if (draw.kind === 'keypoint' && draw.point) {
    drawPointOnCanvas(ctx, draw.point, 'xy', draw.label || '');
  }
}

function canvasPosToNorm(ev) {
  const rect = els.vqaCanvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) / rect.width;
  const y = (ev.clientY - rect.top) / rect.height;
  return [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))];
}

let dragStart = null;

function onCanvasDown(ev) {
  const tab = activeTabName();
  if (tab !== 'vqa') return;
  const kind = els.vqaType.value;
  if (kind !== 'bbox' && kind !== 'keypoint') return;
  ev.preventDefault();
  const [x, y] = canvasPosToNorm(ev);
  dragStart = [x, y];
  if (kind === 'keypoint') {
    state.pendingDraw = { kind: 'keypoint', point: [x, y], label: els.vqaLabel.value || '' };
    setHelper(els.vqaDrawState, `keypoint @ (${x.toFixed(3)}, ${y.toFixed(3)})`, true);
    redrawCanvas();
    dragStart = null;
  } else {
    state.pendingDraw = { kind: 'bbox', bbox: [x, y, x, y], label: els.vqaLabel.value || '' };
    redrawCanvas();
  }
}

function onCanvasMove(ev) {
  if (!dragStart || !state.pendingDraw || state.pendingDraw.kind !== 'bbox') return;
  const [x, y] = canvasPosToNorm(ev);
  state.pendingDraw.bbox = [
    Math.min(dragStart[0], x), Math.min(dragStart[1], y),
    Math.max(dragStart[0], x), Math.max(dragStart[1], y),
  ];
  redrawCanvas();
}

function onCanvasUp() {
  if (state.pendingDraw && state.pendingDraw.kind === 'bbox' && state.pendingDraw.bbox) {
    const [x1, y1, x2, y2] = state.pendingDraw.bbox;
    setHelper(els.vqaDrawState, `bbox ${x1.toFixed(3)},${y1.toFixed(3)} → ${x2.toFixed(3)},${y2.toFixed(3)}`, true);
  }
  dragStart = null;
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function activeTabName() {
  const active = document.querySelector('.tab.active');
  return active ? active.dataset.tab : null;
}

function setupTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');
      const panel = document.getElementById(`${tab.dataset.tab}Panel`);
      if (panel) panel.classList.add('active');
      syncCanvasPointerEvents();
    });
  });
}

function setupVqaTypeSwitch() {
  const update = () => {
    const v = els.vqaType.value;
    document.querySelectorAll('.vqa-control').forEach((el) => {
      const kinds = (el.dataset.vqaType || '').split(/\s+/);
      el.hidden = !kinds.includes(v);
    });
    syncCanvasPointerEvents();
  };
  els.vqaType.addEventListener('change', () => {
    update();
    state.pendingDraw = null;
    redrawCanvas();
  });
  update();
}

function syncCanvasPointerEvents() {
  if (!els.vqaCanvas || !els.vqaType) return;
  const canDraw = activeTabName() === 'vqa' && ['bbox', 'keypoint'].includes(els.vqaType.value);
  els.vqaCanvas.style.pointerEvents = canDraw ? 'auto' : 'none';
}

// ---------------------------------------------------------------------------
// Add-atom handlers
// ---------------------------------------------------------------------------

function setupAddHandlers() {
  els.subtaskSetStart.addEventListener('click', () => { els.subtaskStart.value = currentTime(); });
  els.addSubtask.addEventListener('click', () => {
    if (state.currentEpisode == null) return;
    const ts = Number(els.subtaskStart.value);
    const content = els.subtaskLabel.value.trim();
    if (!content || Number.isNaN(ts)) return;
    getAtoms(state.currentEpisode).push({
      role: 'assistant', content, style: 'subtask',
      timestamp: snapToFrame(state.currentEpisode, ts), tool_calls: null,
    });
    els.subtaskLabel.value = '';
    renderAll();
  });

  els.planSetStart.addEventListener('click', () => { els.planStart.value = currentTime(); });
  els.addPlan.addEventListener('click', () => {
    if (state.currentEpisode == null) return;
    const ts = Number(els.planStart.value);
    const content = els.planContent.value.trim();
    if (!content || Number.isNaN(ts)) return;
    getAtoms(state.currentEpisode).push({
      role: 'assistant', content, style: 'plan',
      timestamp: snapToFrame(state.currentEpisode, ts), tool_calls: null,
    });
    els.planContent.value = '';
    renderAll();
  });

  els.memorySetStart.addEventListener('click', () => { els.memoryStart.value = currentTime(); });
  els.addMemory.addEventListener('click', () => {
    if (state.currentEpisode == null) return;
    const ts = Number(els.memoryStart.value);
    const content = els.memoryContent.value.trim();
    if (!content || Number.isNaN(ts)) return;
    getAtoms(state.currentEpisode).push({
      role: 'assistant', content, style: 'memory',
      timestamp: snapToFrame(state.currentEpisode, ts), tool_calls: null,
    });
    els.memoryContent.value = '';
    renderAll();
  });

  els.interjectionSetStart.addEventListener('click', () => { els.interjectionStart.value = currentTime(); });
  els.addInterjection.addEventListener('click', () => {
    if (state.currentEpisode == null) return;
    const tsRaw = Number(els.interjectionStart.value);
    if (Number.isNaN(tsRaw)) return;
    const ts = snapToFrame(state.currentEpisode, tsRaw);
    const userText = els.interjectionUser.value.trim();
    const speechText = els.interjectionSpeech.value.trim();
    if (!userText && !speechText) return;
    if (userText) {
      getAtoms(state.currentEpisode).push({
        role: 'user', content: userText, style: 'interjection', timestamp: ts, tool_calls: null,
      });
    }
    if (speechText) {
      getAtoms(state.currentEpisode).push(buildSpeechAtom(ts, speechText));
    }
    els.interjectionUser.value = '';
    els.interjectionSpeech.value = '';
    renderAll();
  });

  els.vqaSetStart.addEventListener('click', () => { els.vqaStart.value = currentTime(); });
  els.vqaPause.addEventListener('click', () => {
    els.episodeVideo.pause();
    els.vqaStart.value = currentTime();
  });
  els.vqaClearDraw.addEventListener('click', () => {
    state.pendingDraw = null;
    setHelper(els.vqaDrawState, 'No annotation drawn yet.');
    redrawCanvas();
  });
  els.addVqa.addEventListener('click', () => {
    if (state.currentEpisode == null) return;
    const tsRaw = Number(els.vqaStart.value);
    const question = els.vqaQuestion.value.trim();
    if (!question || Number.isNaN(tsRaw)) return;
    const ts = snapToFrame(state.currentEpisode, tsRaw);

    let answer = null;
    const t = els.vqaType.value;
    if (t === 'bbox') {
      if (!state.pendingDraw || state.pendingDraw.kind !== 'bbox') {
        setHelper(els.vqaDrawState, 'Draw a bounding box on the video first.'); return;
      }
      answer = {
        detections: [{
          label: els.vqaLabel.value.trim() || 'object',
          bbox_format: 'xyxy',
          bbox: state.pendingDraw.bbox.map((v) => Number(v.toFixed(4))),
        }],
      };
    } else if (t === 'keypoint') {
      if (!state.pendingDraw || state.pendingDraw.kind !== 'keypoint') {
        setHelper(els.vqaDrawState, 'Click a point on the video first.'); return;
      }
      answer = {
        label: els.vqaLabel.value.trim() || 'point',
        point_format: 'xy',
        point: state.pendingDraw.point.map((v) => Number(v.toFixed(4))),
      };
    } else if (t === 'count') {
      const count = Number(els.vqaCount.value);
      if (Number.isNaN(count)) return;
      answer = {
        label: els.vqaCountLabel.value.trim() || 'object',
        count,
      };
      const note = els.vqaCountNote.value.trim();
      if (note) answer.note = note;
    } else if (t === 'attribute') {
      answer = {
        label: els.vqaAttrLabel.value.trim() || 'object',
        attribute: els.vqaAttribute.value.trim() || 'attribute',
        value: els.vqaAttrValue.value.trim() || '',
      };
    } else if (t === 'spatial') {
      answer = {
        subject: els.vqaSubject.value.trim() || 'a',
        relation: els.vqaRelation.value.trim() || 'near',
        object: els.vqaObject.value.trim() || 'b',
      };
    }
    if (!answer) return;

    getAtoms(state.currentEpisode).push(
      { role: 'user', content: question, style: 'vqa', timestamp: ts, tool_calls: null },
      { role: 'assistant', content: JSON.stringify(answer), style: 'vqa', timestamp: ts, tool_calls: null },
    );
    els.vqaQuestion.value = '';
    state.pendingDraw = null;
    setHelper(els.vqaDrawState, 'No annotation drawn yet.');
    renderAll();
  });
}

// ---------------------------------------------------------------------------
// Connect form / dataset loading
// ---------------------------------------------------------------------------

function populateVideoKeys(keys, selected) {
  els.videoKeySelect.innerHTML = '';
  if (!keys) return;
  keys.forEach((key) => {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = key;
    if (key === selected) option.selected = true;
    els.videoKeySelect.appendChild(option);
  });
}

function setupConnect() {
  els.connectForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    setHelper(els.connectHelper, 'Loading dataset...');
    try {
      const res = await fetch('/api/dataset/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: els.sourceSelect.value,
          repo_id: els.repoInput.value.trim() || null,
          revision: els.revisionInput.value.trim() || null,
          local_path: els.localInput.value.trim() || null,
          video_key: els.videoKeySelect.value || null,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to load dataset');
      state.dataset = data;
      state.episodes = data.episodes || [];
      setStatus(`Loaded ${data.repo_id || data.root}`, true);
      setHelper(els.connectHelper, `Loaded ${state.episodes.length} episodes.`, true);
      els.workspace.style.display = 'grid';
      populateVideoKeys(data.video_keys, data.selected_video_key);
      renderEpisodes();
    } catch (err) {
      setStatus('Disconnected');
      setHelper(els.connectHelper, err.message);
    }
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function init() {
  els.workspace.style.display = 'none';
  setupTabs();
  setupVqaTypeSwitch();
  setupAddHandlers();
  setupConnect();
  syncCanvasPointerEvents();

  els.saveEpisode.addEventListener('click', saveEpisode);
  els.resetEpisode.addEventListener('click', () => {
    if (state.currentEpisode == null) return;
    state.atomsByEpisode[state.currentEpisode] = [];
    renderAll();
  });
  els.episodeSearch.addEventListener('input', renderEpisodes);

  els.episodeVideo.addEventListener('loadedmetadata', () => {
    resizeCanvas();
    updateTimeDisplay();
    renderTimeline();
  });
  els.episodeVideo.addEventListener('timeupdate', () => {
    updateTimeDisplay();
    redrawCanvas();
  });
  window.addEventListener('resize', resizeCanvas);

  els.vqaCanvas.addEventListener('mousedown', onCanvasDown);
  els.vqaCanvas.addEventListener('mousemove', onCanvasMove);
  els.vqaCanvas.addEventListener('mouseup', onCanvasUp);
  els.vqaCanvas.addEventListener('mouseleave', onCanvasUp);

  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      state.pendingDraw = null;
      setHelper(els.vqaDrawState, 'No annotation drawn yet.');
      redrawCanvas();
    }
  });

  // Toggle new repo input visibility based on push in place checkbox
  if (els.pushInPlace && els.newRepoRow) {
    const sync = () => {
      els.newRepoRow.style.display = els.pushInPlace.checked ? 'none' : 'flex';
    };
    els.pushInPlace.addEventListener('change', sync);
    sync();
  }
  if (els.pushHubBtn) els.pushHubBtn.addEventListener('click', handlePushToHub);

  els.exportBtn.addEventListener('click', async () => {
    els.exportStatus.textContent = 'Exporting...';
    try {
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          output_dir: els.outputDir.value.trim() || null,
          copy_videos: els.copyVideos.checked,
        }),
      });
      const data = await res.json();
      if (res.ok) {
        els.exportStatus.textContent = `Exported to ${data.output_dir} (persistent: ${data.persistent_rows}, events: ${data.event_rows})`;
      } else {
        els.exportStatus.textContent = data.detail || 'Export failed';
      }
    } catch (err) {
      els.exportStatus.textContent = `Network error: ${err.message}`;
    }
  });
}

function updateTimeDisplay() {
  if (els.currentTimeDisplay) {
    els.currentTimeDisplay.textContent = formatTimeWithMs(els.episodeVideo.currentTime);
  }
  if (els.totalTimeDisplay && els.episodeVideo.duration) {
    els.totalTimeDisplay.textContent = formatTimeWithMs(els.episodeVideo.duration);
  }
}

init();
