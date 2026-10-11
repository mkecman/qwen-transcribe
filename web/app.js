// ---------- helpers ----------
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const ICONS = {
  mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
  search: '<circle cx="11" cy="11" r="7.5"/><path d="m21 21-4.3-4.3"/>',
  copy: '<rect width="13" height="13" x="9" y="9" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  more: '<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>',
  settings: '<path d="M21 4h-7M10 4H3M21 12h-9M8 12H3M21 20h-5M12 20H3M14 2v4M8 10v4M16 18v4"/>',
  trash: '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
  refresh: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
  file: '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M8 13v3M12 11v7M16 13v3"/>',
  wave: '<path d="M2 10v3M6 6v11M10 3v18M14 8v7M18 5v13M22 10v3"/>',
  plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
  pilcrow: '<path d="M13 4v16"/><path d="M17 4v16"/><path d="M19 4H9.5a4.5 4.5 0 0 0 0 9H13"/>',
  globe: '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  alert: '<circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
};
const icon = (name) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ''}</svg>`;
const paintIcons = (root = document) => $$('i[data-icon]', root).forEach((el) => { el.innerHTML = icon(el.dataset.icon); });

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

async function api(path, { method = 'GET', body } = {}) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers['Content-Type'] = 'application/json'; }
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return res.json();
}

const wordCount = (t) => (t && t.trim() ? t.trim().split(/\s+/).length : 0);
const plural = (n, w) => `${n.toLocaleString()} ${w}${n === 1 ? '' : 's'}`;
function fmtDuration(s) {
  s = Math.round(s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
}
const fmtClock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const fmtDate = (ts) => new Date(ts * 1000).toLocaleString([], { weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
function dayLabel(ts) {
  const d = new Date(ts * 1000), now = new Date();
  const start = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((start(now) - start(d)) / 86400000);
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return d.toLocaleDateString([], { weekday: 'long' });
  if (d.getFullYear() === now.getFullYear()) return d.toLocaleDateString([], { month: 'long' });
  return d.toLocaleDateString([], { month: 'long', year: 'numeric' });
}
const isBusy = (s) => s && ['queued', 'processing', 'recording'].includes(s.status);

// Mirrors textutil.auto_title on the server: first sentence, or the first ~60 chars.
function autoTitle(text, limit = 60) {
  text = (text || '').split(/\s+/).join(' ').trim();
  if (!text) return '';
  const m = text.match(/^(.+?[.!?])(\s|$)/);
  let title = m && m[1].length <= limit ? m[1] : text;
  if (title.length > limit) title = `${title.slice(0, limit).replace(/\s+\S*$/, '').replace(/[,;:-]+$/, '')}…`;
  return title;
}

let toastTimer;
function toast(msg, ms = 2600) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), ms);
}

function confirmDialog(title, text, ok = 'OK', danger = false) {
  const dlg = $('#confirm');
  $('#confirm-title').textContent = title;
  $('#confirm-text').textContent = text;
  const btn = $('#confirm-ok');
  btn.textContent = ok;
  btn.className = `btn ${danger ? 'danger' : 'primary'}`;
  dlg.returnValue = '';
  dlg.showModal();
  return new Promise((resolve) => dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true }));
}

// ---------- state ----------
const state = {
  status: null,
  settings: null,
  sessions: [],
  current: null,      // full session object, or null for a fresh "new recording"
  query: '',
  rec: null,          // active Recorder
  dirty: null,        // { id, text } waiting to be saved
  saveTimer: null,
  pollTimer: null,
};

// ---------- status & settings ----------
async function pollStatus() {
  try { state.status = await api('/api/status'); } catch { state.status = { status: 'offline' }; }
  renderStatus();
  setTimeout(pollStatus, state.status.status === 'ready' ? 15000 : 1200);
}

let languagesRendered = false;
function renderStatus() {
  const st = state.status || {};
  const el = $('#status');
  el.className = `status ${st.status === 'ready' ? 'ready' : st.status === 'loading' ? 'loading' : 'error'}`;
  const text = {
    ready: `Ready · ${(st.device || 'GPU').replace('NVIDIA GeForce ', '')}`,
    loading: 'Loading model…',
    error: 'Model failed to load',
    offline: 'Server not running',
  }[st.status] || '…';
  $('.status-text', el).textContent = text;
  el.title = st.status === 'ready'
    ? `${st.model} on ${st.device} · ${st.vram_gb} GB VRAM · ${st.fast_decoder ? 'CUDA-graph decoder' : 'standard decoder'}`
    : (st.error || '');
  if (st.status === 'ready' && !languagesRendered && st.languages?.length) {
    languagesRendered = true;
    renderLanguages();
  }
  renderDock();
}

function languageOptions(select, value) {
  const langs = state.status?.languages || ['English'];
  select.innerHTML = '';
  select.add(new Option('English', 'English'));
  select.add(new Option('Auto-detect', 'auto'));
  const group = document.createElement('optgroup');
  group.label = 'Other languages';
  langs.filter((l) => l !== 'English').sort().forEach((l) => group.appendChild(new Option(l, l)));
  select.appendChild(group);
  select.value = value || 'English';
}

function renderLanguages() {
  languageOptions($('#lang'), state.settings?.language);
  languageOptions($('#set-language'), state.settings?.language);
}

function applyTheme() {
  const t = state.settings?.theme || 'system';
  if (t === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
}

async function saveSetting(key, value) {
  state.settings = await api('/api/settings', { method: 'PUT', body: { [key]: value } });
  if (key === 'theme') applyTheme();
  if (key === 'language') { $('#lang').value = value; $('#set-language').value = value; }
}

function fillSettingsForm() {
  const s = state.settings;
  $$('[data-setting]', $('#settings')).forEach((el) => {
    const v = s[el.dataset.setting];
    if (el.type === 'checkbox') el.checked = !!v;
    else el.value = String(v);
    if (el.type === 'range') el.nextElementSibling.textContent = `${(v / 1000).toFixed(2)} s`;
  });
}

function bindSettingsForm() {
  $$('[data-setting]', $('#settings')).forEach((el) => {
    const key = el.dataset.setting;
    const read = () => {
      if (el.type === 'checkbox') return el.checked;
      if (el.type === 'range' || key === 'paragraph_after_s') return Number(el.value);
      return el.value;
    };
    if (el.type === 'range') el.addEventListener('input', () => { el.nextElementSibling.textContent = `${(el.value / 1000).toFixed(2)} s`; });
    el.addEventListener('change', () => saveSetting(key, read()).catch((e) => toast(e.message)));
  });
}

// ---------- history ----------
async function refreshSessions() {
  try {
    state.sessions = await api(`/api/sessions?q=${encodeURIComponent(state.query)}`);
  } catch { return; }
  renderHistory();
  ensurePolling();
}

function sessionLine(s) {
  if (s.status === 'recording') return ['● Recording', 'rec'];
  if (s.status === 'queued') return ['Waiting to transcribe…', 'busy'];
  if (s.status === 'processing') return [`Transcribing ${Math.round((s.progress || 0) * 100)}%`, 'busy'];
  if (s.status === 'error') return ['Failed', 'err'];
  const parts = [];
  if (s.duration) parts.push(fmtDuration(s.duration));
  parts.push(plural(s.words, 'word'));
  return [parts.join(' · '), ''];
}

function renderHistory() {
  const nav = $('#history');
  nav.innerHTML = '';
  if (!state.sessions.length) {
    const empty = document.createElement('div');
    empty.className = 'history-empty';
    empty.textContent = state.query ? 'No matching transcripts' : 'Your transcripts will appear here';
    nav.appendChild(empty);
    return;
  }
  let group = null;
  for (const s of state.sessions) {
    const label = dayLabel(s.created_at);
    if (label !== group) {
      group = label;
      const h = document.createElement('div');
      h.className = 'history-group';
      h.textContent = label;
      nav.appendChild(h);
    }
    const item = document.createElement('button');
    item.className = `item${state.current?.id === s.id ? ' active' : ''}`;
    item.innerHTML = `<span class="ico">${icon(s.kind === 'file' ? 'file' : 'mic')}</span><span class="t"></span><span class="time"></span><span class="s"></span>`;
    $('.t', item).textContent = s.title;
    $('.time', item).textContent = fmtClock(s.created_at);
    const [line, cls] = sessionLine(s);
    const sub = $('.s', item);
    sub.textContent = line;
    if (cls) sub.classList.add(cls);
    item.title = s.snippet || s.title;
    item.addEventListener('click', () => openSession(s.id));
    nav.appendChild(item);
  }
}

function ensurePolling() {
  const busy = state.sessions.some((s) => isBusy(s) && s.id !== state.rec?.sessionId) || (isBusy(state.current) && !state.rec);
  if (busy && !state.pollTimer) state.pollTimer = setInterval(pollBusy, 800);
  else if (!busy && state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

async function pollBusy() {
  await refreshSessions();
  const cur = state.current;
  if (!cur || state.rec) return;
  const listed = state.sessions.find((s) => s.id === cur.id);
  if (isBusy(cur) || (listed && listed.status !== cur.status)) {
    try {
      state.current = await api(`/api/sessions/${cur.id}`);
      renderMain();
    } catch { /* deleted */ }
  }
}

// ---------- main view ----------
async function openSession(id) {
  if (state.rec) { toast('Stop the recording first'); return; }
  await flushSave();
  try {
    state.current = await api(`/api/sessions/${id}`);
  } catch (e) { toast(e.message); return; }
  renderMain();
  renderHistory();
  ensurePolling();
  $('#doc').scrollTop = 0;
}

async function newSession() {
  if (state.rec) { toast('Stop the recording first'); return; }
  await flushSave();
  state.current = null;
  renderMain();
  renderHistory();
}

function renderMeta() {
  const s = state.current, rec = state.rec, meta = $('#meta');
  meta.innerHTML = '';
  const add = (text, cls) => {
    if (meta.childNodes.length) {
      const sep = document.createElement('span');
      sep.className = 'sep';
      sep.textContent = '·';
      meta.appendChild(sep);
    }
    const el = document.createElement('span');
    if (cls) el.className = cls;
    el.textContent = text;
    meta.appendChild(el);
  };
  if (rec) add(rec.phase === 'finishing' ? 'Finishing…' : 'Recording', 'chip rec');
  else if (s?.status === 'queued') add('Waiting to transcribe', 'chip');
  else if (s?.status === 'processing') add(`Transcribing ${Math.round((s.progress || 0) * 100)}%`, 'chip');
  if (!s && !rec) { add('Live dictation'); return; }
  if (s) {
    add(s.kind === 'file' ? (s.source_name || 'Uploaded file') : 'Live dictation');
    add(fmtDate(s.created_at));
    if (s.duration) add(fmtDuration(s.duration));
  }
  const text = rec ? rec.text : (s?.text || '');
  add(plural(wordCount(text), 'word'));
}

function renderMain() {
  const s = state.current, rec = state.rec;
  const title = $('#title');
  if (document.activeElement !== title) title.value = s?.custom_title ? s.title : '';
  title.placeholder = s ? s.title : 'New recording';
  title.disabled = !s;
  renderMeta();

  const busy = isBusy(s) && !rec;
  const progress = $('#progress');
  progress.hidden = !busy;
  progress.classList.toggle('indeterminate', s?.status === 'queued' || (s?.status === 'processing' && !s.progress));
  $('.progress-bar', progress).style.width = `${Math.round((s?.progress || 0) * 100)}%`;

  const banner = $('#banner');
  banner.hidden = s?.status !== 'error';
  if (s?.status === 'error') {
    banner.innerHTML = `${icon('alert')}<span></span>`;
    $('span', banner).textContent = s.error || 'Transcription failed';
    if (s.has_audio) {
      const retry = document.createElement('button');
      retry.className = 'btn';
      retry.textContent = 'Retry';
      retry.addEventListener('click', retranscribe);
      banner.appendChild(retry);
    }
  }

  const editor = $('#editor'), live = $('#live'), empty = $('#empty');
  const showEmpty = !rec && (!s || (!s.text && s.status === 'done'));
  empty.hidden = !showEmpty;
  live.hidden = !rec;
  editor.hidden = !!rec || showEmpty;
  if (!editor.hidden) {
    if (document.activeElement !== editor || busy) editor.value = s.text;
    editor.readOnly = busy;
    editor.placeholder = busy ? 'Transcribing…' : '';
    autosize();
  }
  $('#copy-btn').disabled = !(rec ? rec.text : s?.text);
  renderDock();
}

function autosize() {
  const ed = $('#editor');
  ed.style.height = 'auto';
  ed.style.height = `${ed.scrollHeight}px`;
}

function scheduleSave() {
  const s = state.current;
  if (!s) return;
  state.dirty = { id: s.id, text: $('#editor').value };
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(flushSave, 600);
}

async function flushSave() {
  clearTimeout(state.saveTimer);
  const d = state.dirty;
  if (!d) return;
  state.dirty = null;
  try {
    const s = await api(`/api/sessions/${d.id}`, { method: 'PATCH', body: { text: d.text } });
    if (state.current?.id === s.id) {
      state.current.title = s.title;
      if (!s.custom_title) $('#title').placeholder = s.title;
    }
    refreshSessions();
  } catch (e) { toast(`Couldn't save: ${e.message}`); }
}

async function saveTitle() {
  const s = state.current;
  if (!s) return;
  const value = $('#title').value.trim();
  if (value === (s.custom_title ? s.title : '')) return;
  try {
    state.current = { ...s, ...(await api(`/api/sessions/${s.id}`, { method: 'PATCH', body: { title: value } })) };
    renderMain();
    refreshSessions();
  } catch (e) { toast(e.message); }
}

async function copyText() {
  const text = state.rec ? state.rec.text : (state.current?.text || '');
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    toast('Clipboard not available');
    return;
  }
  const btn = $('#copy-btn');
  btn.classList.add('copied');
  btn.innerHTML = `${icon('check')}<span>Copied</span>`;
  setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = `${icon('copy')}<span>Copy</span>`; }, 1600);
}

function downloadText() {
  const s = state.current;
  if (!s?.text) return;
  const name = (s.title || 'transcript').replace(/[\\/:*?"<>|]+/g, '').slice(0, 80).trim() || 'transcript';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([s.text], { type: 'text/plain;charset=utf-8' }));
  a.download = `${name}.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function retranscribe() {
  const s = state.current;
  if (!s) return;
  if (s.text && !(await confirmDialog('Re-transcribe this audio?', 'The saved audio is transcribed again from scratch. The current text, including your edits, will be replaced.', 'Re-transcribe'))) return;
  try {
    state.current = await api(`/api/sessions/${s.id}/retranscribe`, { method: 'POST' });
    renderMain();
    refreshSessions();
  } catch (e) { toast(e.message); }
}

async function deleteAudio() {
  const s = state.current;
  if (!s) return;
  if (!(await confirmDialog('Delete the saved audio?', 'The transcript stays. You won’t be able to re-transcribe this session afterwards.', 'Delete audio', true))) return;
  try {
    state.current = await api(`/api/sessions/${s.id}/audio`, { method: 'DELETE' });
    toast('Audio deleted');
  } catch (e) { toast(e.message); }
}

async function deleteSession() {
  const s = state.current;
  if (!s) return;
  if (!(await confirmDialog('Delete this transcript?', `“${s.title}” and its saved audio will be permanently deleted.`, 'Delete', true))) return;
  try {
    await api(`/api/sessions/${s.id}`, { method: 'DELETE' });
    state.dirty = null;
    state.current = null;
    renderMain();
    refreshSessions();
    toast('Transcript deleted');
  } catch (e) { toast(e.message); }
}

function renderMenu() {
  const s = state.current;
  const set = (action, enabled) => { $(`[data-action="${action}"]`).disabled = !enabled; };
  set('download', !!s?.text);
  set('retranscribe', !!s?.has_audio && !isBusy(s));
  set('delete-audio', !!s?.has_audio && !isBusy(s));
  set('delete', !!s && s.status !== 'recording');
}

// ---------- uploads ----------
async function uploadFiles(files) {
  files = [...files];
  if (!files.length) return;
  if (state.rec) { toast('Stop the recording first'); return; }
  await flushSave();
  let first = null;
  for (const f of files) {
    toast(`Uploading ${f.name}…`, 60000);
    const fd = new FormData();
    fd.append('file', f);
    fd.append('language', $('#lang').value || 'English');
    try {
      const s = await api('/api/upload', { method: 'POST', body: fd });
      first = first || s;
    } catch (e) { toast(`${f.name}: ${e.message}`); }
  }
  if (first) {
    toast(files.length > 1 ? `${files.length} files queued` : 'Transcribing…');
    await refreshSessions();
    await openSession(first.id);
  }
}

// ---------- microphones ----------
const cleanLabel = (l) => l.replace(/\s*\([0-9a-f]{4}:[0-9a-f]{4}\)\s*$/i, '').replace(/^Default - /, 'Default · ');

async function refreshMics() {
  const sel = $('#mic');
  let devices = [];
  try { devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput'); } catch { /* no permission API */ }
  const saved = store.get('mic');
  sel.innerHTML = '';
  if (!devices.length || devices.every((d) => !d.label)) {
    sel.add(new Option('Default microphone', 'default'));
    sel.dataset.labels = 'missing';
  } else {
    devices.filter((d) => d.deviceId !== 'communications').forEach((d) => sel.add(new Option(cleanLabel(d.label), d.deviceId)));
    delete sel.dataset.labels;
  }
  if (saved && [...sel.options].some((o) => o.value === saved)) sel.value = saved;
  sel.title = sel.selectedOptions[0]?.textContent || '';
}

// ---------- recording ----------
class Recorder {
  constructor(sessionId, onEvent) {
    this.sessionId = sessionId;
    this.onEvent = onEvent;
    this.phase = 'starting';
    this.ready = false;
    this.queue = [];
    this.text = '';
    this.level = 0;
  }

  async start({ deviceId, language, settings }) {
    const audio = {
      channelCount: 1,
      echoCancellation: !!settings.echo_cancellation,
      noiseSuppression: !!settings.noise_suppression,
      autoGainControl: !!settings.auto_gain,
    };
    if (deviceId && deviceId !== 'default') audio.deviceId = { exact: deviceId };
    this.stream = await navigator.mediaDevices.getUserMedia({ audio });
    try {
      this.ctx = new AudioContext({ sampleRate: 16000 });
      await this.ctx.audioWorklet.addModule('/static/worklet.js');
      this.src = this.ctx.createMediaStreamSource(this.stream);
    } catch {
      // Some browsers can't mix sample rates; capture natively and resample in the worklet.
      if (this.ctx) this.ctx.close();
      this.ctx = new AudioContext();
      await this.ctx.audioWorklet.addModule('/static/worklet.js');
      this.src = this.ctx.createMediaStreamSource(this.stream);
    }
    await this.ctx.resume();
    this.node = new AudioWorkletNode(this.ctx, 'pcm-capture');
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.src.connect(this.node).connect(mute).connect(this.ctx.destination);
    this.node.port.onmessage = (e) => this.onPcm(e.data);

    this.ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`);
    this.ws.binaryType = 'arraybuffer';
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = () => reject(new Error('Could not connect to the transcription server'));
    });
    this.ws.onmessage = (e) => this.onMessage(JSON.parse(e.data));
    this.ws.onclose = () => { if (this.phase !== 'finishing' && this.phase !== 'done') this.onEvent('lost'); };
    this.ws.send(JSON.stringify({ type: 'start', session_id: this.sessionId, language }));
    this.startedAt = performance.now();
  }

  onPcm({ pcm, level }) {
    this.level = level;
    if (this.phase === 'finishing' || this.phase === 'done') return;
    if (this.ready && this.ws.readyState === WebSocket.OPEN) this.ws.send(pcm);
    else this.queue.push(pcm);
  }

  onMessage(m) {
    if (m.type === 'started') {
      this.sessionId = m.session.id;
      this.text = m.session.text || '';
      this.ready = true;
      this.phase = 'recording';
      this.queue.forEach((b) => this.ws.send(b));
      this.queue = [];
    } else if (m.type === 'text') {
      this.text = m.stable + m.pending;
      this.committed = m.stable;
    } else if (m.type === 'stopped') {
      this.resolveStop?.(m.session);
    }
    this.onEvent(m.type, m);
  }

  paragraph() {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: 'paragraph' }));
  }

  releaseAudio() {
    try { this.node?.port && (this.node.port.onmessage = null); } catch { /* ignore */ }
    try { this.src?.disconnect(); } catch { /* ignore */ }
    this.stream?.getTracks().forEach((t) => t.stop());
    if (this.ctx && this.ctx.state !== 'closed') this.ctx.close();
  }

  async stop() {
    this.phase = 'finishing';
    this.releaseAudio();
    let session = null;
    if (this.ws?.readyState === WebSocket.OPEN) {
      const stopped = new Promise((resolve) => { this.resolveStop = resolve; });
      this.ws.send(JSON.stringify({ type: 'stop' }));
      session = await Promise.race([stopped, new Promise((r) => setTimeout(() => r(null), 60000))]);
      this.ws.close();
    }
    this.phase = 'done';
    return session;
  }
}

let timerInterval = null;
let meterFrame = null;

function renderDock() {
  const rec = state.rec, s = state.current;
  const ready = state.status?.status === 'ready';
  const btn = $('#rec-btn');
  const finishing = rec?.phase === 'finishing';
  btn.classList.toggle('recording', !!rec && !finishing);
  btn.classList.toggle('finishing', finishing);
  btn.disabled = finishing || (!rec && (!ready || isBusy(s)));
  $('.rec-label', btn).textContent = rec ? (finishing ? 'Finishing…' : 'Stop') : (s?.text ? 'Continue' : 'Record');
  btn.title = rec ? 'Stop (Ctrl+Space)' : !ready ? 'Waiting for the model to load' : 'Start dictating (Ctrl+Space)';
  $('#big-mic').disabled = !ready;
  $('#para-btn').hidden = !rec || finishing;
  $('#mic').disabled = !!rec;
  $('#lang').disabled = !!rec;
  $('.dock').classList.toggle('recording-on', !!rec);
  if (!rec) {
    $('#timer').textContent = '0:00';
    $('#rec-state').textContent = ready ? '' : (state.status?.status === 'loading' ? 'Loading model…' : '');
  }
}

function tickMeter() {
  const rec = state.rec;
  const bars = $$('#meter i');
  const lv = rec ? rec.level : 0;
  const shape = [0.55, 0.85, 1, 0.8, 0.5];
  bars.forEach((b, i) => {
    const jitter = rec ? 0.75 + Math.random() * 0.25 : 1;
    b.style.height = `${4 + Math.round(lv * shape[i] * jitter * 16)}px`;
  });
  meterFrame = rec ? requestAnimationFrame(tickMeter) : null;
}

function renderLive() {
  const rec = state.rec;
  if (!rec) return;
  const doc = $('#doc');
  const nearBottom = doc.scrollHeight - doc.scrollTop - doc.clientHeight < 120;
  const stable = rec.committed ?? rec.text;
  $('#stable').textContent = stable;
  $('#pending').textContent = rec.text.slice(stable.length);
  $('#live').classList.toggle('waiting', !rec.text);
  $('#live').classList.toggle('finishing', rec.phase === 'finishing');
  if (nearBottom) doc.scrollTop = doc.scrollHeight;
  if (!state.current?.custom_title) $('#title').placeholder = autoTitle(rec.text) || 'New recording';
  renderMeta();
  $('#copy-btn').disabled = !rec.text;
}

async function startRecording() {
  if (state.rec) return;
  if (state.status?.status !== 'ready') { toast('The model is still loading'); return; }
  if (isBusy(state.current)) { toast('This session is still being transcribed'); return; }
  await flushSave();
  const rec = new Recorder(state.current?.id || null, onRecorderEvent);
  state.rec = rec;
  rec.text = state.current?.text || '';
  rec.committed = rec.text;
  renderMain();
  renderLive();
  $('#rec-state').textContent = 'Starting…';
  try {
    await rec.start({ deviceId: $('#mic').value, language: $('#lang').value, settings: state.settings });
  } catch (e) {
    rec.phase = 'done';
    rec.releaseAudio();
    state.rec = null;
    renderMain();
    const denied = e.name === 'NotAllowedError' || e.name === 'SecurityError';
    toast(denied ? 'Microphone access is blocked. Allow it from the icon in the address bar.' : (e.message || 'Could not start recording'), 5000);
    return;
  }
  if ($('#mic').dataset.labels === 'missing') refreshMics();
  timerInterval = setInterval(() => {
    const t = (performance.now() - rec.startedAt) / 1000;
    $('#timer').textContent = fmtDuration(t);
  }, 250);
  tickMeter();
}

function onRecorderEvent(type, m) {
  const rec = state.rec;
  if (!rec) return;
  if (type === 'started') {
    state.current = m.session;
    $('#rec-state').textContent = 'Listening';
    renderMain();
    renderLive();
    refreshSessions();
  } else if (type === 'text') {
    renderLive();
  } else if (type === 'vad') {
    $('#rec-btn').classList.toggle('speaking', m.speech);
    $('#rec-state').textContent = m.speech ? 'Hearing you' : 'Listening';
  } else if (type === 'error') {
    toast(m.message, 5000);
    stopRecording(true);
  } else if (type === 'lost') {
    toast('Lost connection to the server. Text up to this point is saved.', 5000);
    stopRecording(true);
  }
}

async function stopRecording(aborted = false) {
  const rec = state.rec;
  if (!rec || rec.phase === 'finishing') return;
  clearInterval(timerInterval);
  $('#rec-state').textContent = 'Finishing…';
  const finishing = rec.stop();
  renderMain();
  renderLive();
  await finishing;
  $('#rec-btn').classList.remove('speaking');
  state.rec = null;
  const id = rec.sessionId;
  if (id) {
    try { state.current = await api(`/api/sessions/${id}`); } catch { state.current = null; }
  }
  renderMain();
  await refreshSessions();
  if (!aborted && state.settings?.copy_on_stop && state.current?.text) copyText();
}

function toggleRecording() {
  if (state.rec) stopRecording();
  else startRecording();
}

// ---------- wiring ----------
function bind() {
  $('#new-btn').addEventListener('click', newSession);
  $('#upload-btn').addEventListener('click', () => $('#file-input').click());
  $('#file-input').addEventListener('change', (e) => { uploadFiles(e.target.files); e.target.value = ''; });
  $('#rec-btn').addEventListener('click', toggleRecording);
  $('#big-mic').addEventListener('click', startRecording);
  $('#para-btn').addEventListener('click', () => { state.rec?.paragraph(); toast('New paragraph at your next sentence', 1500); });
  $('#copy-btn').addEventListener('click', copyText);

  let searchTimer;
  $('#search').addEventListener('input', (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.query = e.target.value.trim(); refreshSessions(); }, 200);
  });

  const editor = $('#editor');
  editor.addEventListener('input', () => {
    if (!state.current) return;
    state.current.text = editor.value;
    autosize();
    renderMeta();
    $('#copy-btn').disabled = !editor.value;
    scheduleSave();
  });
  editor.addEventListener('blur', flushSave);

  const title = $('#title');
  title.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); title.blur(); } if (e.key === 'Escape') { title.value = state.current?.custom_title ? state.current.title : ''; title.blur(); } });
  title.addEventListener('blur', saveTitle);

  const menu = $('#more-menu');
  $('#more-btn').addEventListener('click', (e) => { e.stopPropagation(); renderMenu(); menu.hidden = !menu.hidden; });
  document.addEventListener('click', () => { menu.hidden = true; });
  menu.addEventListener('click', (e) => {
    const action = e.target.closest('button')?.dataset.action;
    menu.hidden = true;
    ({ download: downloadText, retranscribe, 'delete-audio': deleteAudio, delete: deleteSession })[action]?.();
  });

  $('#settings-btn').addEventListener('click', () => { fillSettingsForm(); $('#settings').showModal(); });
  bindSettingsForm();
  $('#lang').addEventListener('change', (e) => saveSetting('language', e.target.value).catch((err) => toast(err.message)));
  $('#mic').addEventListener('change', (e) => { store.set('mic', e.target.value); e.target.title = e.target.selectedOptions[0]?.textContent || ''; });
  navigator.mediaDevices?.addEventListener?.('devicechange', refreshMics);

  document.addEventListener('keydown', (e) => {
    const typing = e.target.closest?.('input, textarea, select, [contenteditable]');
    if (e.ctrlKey && e.code === 'Space') { e.preventDefault(); toggleRecording(); return; }
    if (e.key === 'Enter' && state.rec && !typing && !e.ctrlKey) { e.preventDefault(); $('#para-btn').click(); }
  });

  // drag & drop anywhere
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { depth++; $('#drop').hidden = false; } });
  window.addEventListener('dragleave', (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; $('#drop').hidden = true; } });
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    $('#drop').hidden = true;
    uploadFiles(e.dataTransfer.files);
  });

  window.addEventListener('beforeunload', (e) => {
    if (state.dirty) {
      fetch(`/api/sessions/${state.dirty.id}`, { method: 'PATCH', keepalive: true, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: state.dirty.text }) });
    }
    if (state.rec) { e.preventDefault(); e.returnValue = ''; }
  });
}

async function init() {
  paintIcons();
  bind();
  try { state.settings = await api('/api/settings'); } catch { state.settings = {}; }
  applyTheme();
  renderLanguages();
  refreshMics();
  pollStatus();
  await refreshSessions();
  renderMain();
}

init();
