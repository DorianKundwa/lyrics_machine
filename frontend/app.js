/**
 * Croonify — AI Lyrics Synchronization Engine
 * Frontend Application (Vanilla ES6+)
 *
 * Architecture:
 *  - AppState          — singleton shared state
 *  - UIController      — show/hide panels, update DOM
 *  - AudioUploadHandler— drag-and-drop, file validation, display
 *  - LyricsHandler     — textarea management, line counting
 *  - AlignmentAPI      — fetch wrappers for all API endpoints
 *  - JobPoller         — polls /api/status/{job_id} every 1.5s
 *  - ResultsRenderer   — renders JSON, words, lines
 *  - JSONHighlighter   — syntax highlights JSON in <pre>
 *  - init()            — wire everything up on DOMContentLoaded
 */

'use strict';

/* =========================================================
   CONFIG
   ========================================================= */
const API_BASE = window.location.protocol === 'file:'
  ? 'http://localhost:8000'
  : window.location.origin;

const POLL_INTERVAL_MS    = 1500;
const MAX_BACKOFF_MS      = 30000;
const ACCEPTED_EXTENSIONS = new Set(['.mp3', '.wav', '.flac', '.m4a', '.ogg']);
const ACCEPTED_MIME_RE    = /^audio\//;

/* =========================================================
   APP STATE (singleton)
   ========================================================= */
const AppState = {
  audioFile:       null,    // File object
  lyricsText:      '',      // raw lyrics string
  language:        'auto',
  aligner:         'whisperx',
  vocalSeparation: false,
  jobId:           null,
  currentResult:   null,
  isPolling:       false,
  _pollTimer:      null,
  _backoffMs:      POLL_INTERVAL_MS,

  reset() {
    this.audioFile       = null;
    this.lyricsText      = '';
    this.jobId           = null;
    this.currentResult   = null;
    this.isPolling       = false;
    clearTimeout(this._pollTimer);
    this._pollTimer      = null;
    this._backoffMs      = POLL_INTERVAL_MS;
  },
};

/* =========================================================
   UTILITY HELPERS
   ========================================================= */
function formatBytes(bytes) {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1024 ** 2)  return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
}

function formatTimestamp(seconds) {
  if (typeof seconds !== 'number' || isNaN(seconds)) return '—';
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(2).padStart(5, '0');
  return `${String(m).padStart(2, '0')}:${s}`;
}

function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* =========================================================
   UI CONTROLLER
   ========================================================= */
const UIController = (() => {
  /* Panel IDs */
  const PANELS = {
    status:  document.getElementById('job-status-panel'),
    results: document.getElementById('results-panel'),
    error:   document.getElementById('error-panel'),
  };

  function showPanel(name) {
    Object.entries(PANELS).forEach(([key, el]) => {
      if (key === name) {
        el.hidden = false;
        // Trigger entrance animation
        el.classList.remove('panel-enter');
        void el.offsetWidth; // reflow
        el.classList.add('panel-enter');
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } else {
        el.hidden = true;
        el.classList.remove('panel-enter');
      }
    });
  }

  function hideAllPanels() {
    Object.values(PANELS).forEach(el => {
      el.hidden = true;
      el.classList.remove('panel-enter');
    });
  }

  function setSyncBtnState(state) {
    /* state: 'idle' | 'ready' | 'processing' | 'disabled' */
    const btn  = document.getElementById('sync-btn');
    const hint = document.getElementById('sync-btn-hint');

    btn.classList.remove('processing');

    switch (state) {
      case 'ready':
        btn.disabled = false;
        btn.querySelector('.sync-btn__text').textContent = 'Sync Now';
        hint.textContent = 'Ready — click to start alignment';
        break;
      case 'processing':
        btn.disabled = true;
        btn.classList.add('processing');
        btn.querySelector('.sync-btn__text').textContent = 'Processing…';
        hint.textContent = 'Alignment in progress — please wait';
        break;
      case 'disabled':
      default:
        btn.disabled = true;
        btn.querySelector('.sync-btn__text').textContent = 'Sync Now';
        hint.textContent = 'Upload audio and paste lyrics to enable';
    }
  }

  function setProgress(pct, message) {
    const fill    = document.getElementById('progress-fill');
    const pctEl   = document.getElementById('progress-pct');
    const msgEl   = document.getElementById('status-message');
    const wrapper = document.getElementById('progress-bar-wrapper');

    const clamped = clamp(pct, 0, 100);
    fill.style.width  = `${clamped}%`;
    pctEl.textContent = `${clamped}%`;
    wrapper.setAttribute('aria-valuenow', clamped);

    if (message) msgEl.textContent = message;
  }

  function setStatusBadge(status) {
    const badge = document.getElementById('status-badge');
    badge.className = `status-badge status-badge--${status}`;
    badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    badge.setAttribute('role', 'status');

    const title = document.querySelector('.status-panel__title');
    const spinner = document.getElementById('status-spinner');

    if (status === 'done') {
      title.textContent = 'Complete';
      spinner.style.opacity = '0';
    } else if (status === 'error') {
      title.textContent = 'Failed';
      spinner.style.opacity = '0';
    } else if (status === 'processing') {
      title.textContent = 'Processing';
      spinner.style.opacity = '1';
    } else {
      title.textContent = 'Queued';
      spinner.style.opacity = '1';
    }
  }

  function showError(message) {
    document.getElementById('error-message').textContent =
      message || 'An unexpected error occurred. Please try again.';
    showPanel('error');
    setSyncBtnState('ready');
  }

  function checkSubmitReady() {
    const hasFile   = Boolean(AppState.audioFile);
    const hasLyrics = AppState.lyricsText.trim().length > 0;
    setSyncBtnState(hasFile && hasLyrics ? 'ready' : 'disabled');
  }

  return {
    showPanel,
    hideAllPanels,
    setSyncBtnState,
    setProgress,
    setStatusBadge,
    showError,
    checkSubmitReady,
  };
})();

/* =========================================================
   TOAST NOTIFICATIONS
   ========================================================= */
const Toast = (() => {
  let _timer = null;
  const el   = document.getElementById('toast');

  function show(message, type = 'default', durationMs = 3000) {
    clearTimeout(_timer);
    el.textContent     = message;
    el.className       = `toast toast--${type}`;
    el.hidden          = false;

    _timer = setTimeout(hide, durationMs);
  }

  function hide() {
    el.hidden = true;
    el.className = 'toast';
  }

  return { show, hide };
})();

/* =========================================================
   AUDIO UPLOAD HANDLER
   ========================================================= */
const AudioUploadHandler = (() => {
  let dropZone  = null;
  let fileInput = null;

  function validateFile(file) {
    if (!file) return false;
    if (ACCEPTED_MIME_RE.test(file.type)) return true;
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    return ACCEPTED_EXTENSIONS.has(ext);
  }

  function displayFile(file) {
    AppState.audioFile = file;

    dropZone.classList.add('has-file');
    dropZone.classList.remove('drag-over');

    const fileInfo = document.getElementById('file-info');
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = `(${formatBytes(file.size)})`;
    fileInfo.hidden = false;

    // Update aria label
    dropZone.setAttribute('aria-label',
      `Audio file selected: ${file.name}. Press Enter to change.`);

    UIController.checkSubmitReady();
  }

  function clearFile() {
    AppState.audioFile = null;
    const fileInfo = document.getElementById('file-info');
    fileInfo.hidden = true;
    dropZone.classList.remove('has-file');
    dropZone.setAttribute('aria-label',
      'Upload audio file — drag and drop or press Enter to browse');
    fileInput.value = '';
    UIController.checkSubmitReady();
  }

  function handleFiles(files) {
    if (!files || files.length === 0) return;
    const file = files[0];

    if (!validateFile(file)) {
      Toast.show('❌ Unsupported file type. Please use MP3, WAV, FLAC, M4A, or OGG.', 'error', 4000);
      return;
    }

    displayFile(file);
  }

  function init() {
    dropZone  = document.getElementById('drop-zone');
    fileInput = document.getElementById('audio-file-input');

    /* File input change */
    fileInput.addEventListener('change', () => {
      handleFiles(fileInput.files);
    });

    /* Drag events */
    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', e => {
      e.preventDefault();
      e.stopPropagation();
      if (!dropZone.contains(e.relatedTarget)) {
        dropZone.classList.remove('drag-over');
      }
    });

    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
      handleFiles(e.dataTransfer.files);
    });

    /* Keyboard accessibility */
    dropZone.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });
  }

  return { init, clearFile };
})();

/* =========================================================
   LYRICS HANDLER
   ========================================================= */
const LyricsHandler = (() => {
  function countLines(text) {
    if (!text.trim()) return 0;
    return text.split('\n').filter(l => l.trim().length > 0).length;
  }

  function updateCounters(text) {
    const lines = countLines(text);
    const chars = text.length;
    document.getElementById('line-count').textContent = `${lines} line${lines !== 1 ? 's' : ''}`;
    document.getElementById('char-count').textContent = `${chars} char${chars !== 1 ? 's' : ''}`;
  }

  function init() {
    const textarea = document.getElementById('lyrics-input');

    textarea.addEventListener('input', () => {
      AppState.lyricsText = textarea.value;
      updateCounters(textarea.value);
      UIController.checkSubmitReady();
    });

    /* Paste event — update immediately */
    textarea.addEventListener('paste', () => {
      setTimeout(() => {
        AppState.lyricsText = textarea.value;
        updateCounters(textarea.value);
        UIController.checkSubmitReady();
      }, 0);
    });
  }

  return { init };
})();

/* =========================================================
   ALIGNMENT API
   ========================================================= */
const AlignmentAPI = (() => {
  async function submitJob(audioFile, lyrics, language, aligner, vocalSeparation) {
    const formData = new FormData();
    formData.append('audio', audioFile, audioFile.name);
    formData.append('lyrics', lyrics);
    formData.append('language', language);
    formData.append('aligner', aligner);
    formData.append('vocal_separation', String(vocalSeparation));

    const res = await fetch(`${API_BASE}/api/align`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = body.detail || body.message || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    return res.json(); // { job_id: string }
  }

  async function getStatus(jobId) {
    const res = await fetch(`${API_BASE}/api/status/${encodeURIComponent(jobId)}`);
    if (!res.ok) {
      throw new Error(`Status fetch failed: HTTP ${res.status}`);
    }
    return res.json(); // { status, progress, message }
  }

  async function getResult(jobId) {
    const res = await fetch(`${API_BASE}/api/result/${encodeURIComponent(jobId)}`);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const body = await res.json();
        detail = body.detail || body.message || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  }

  return { submitJob, getStatus, getResult };
})();

/* =========================================================
   JOB POLLER
   ========================================================= */
const JobPoller = (() => {
  function start(jobId) {
    AppState.isPolling  = true;
    AppState._backoffMs = POLL_INTERVAL_MS;
    poll(jobId);
  }

  function stop() {
    AppState.isPolling = false;
    clearTimeout(AppState._pollTimer);
    AppState._pollTimer = null;
  }

  async function poll(jobId) {
    if (!AppState.isPolling) return;

    try {
      const status = await AlignmentAPI.getStatus(jobId);
      AppState._backoffMs = POLL_INTERVAL_MS; // reset on success

      // Normalize
      const pct     = typeof status.progress === 'number' ? Math.round(status.progress) : 0;
      const message = status.message || '';
      const state   = (status.status || 'queued').toLowerCase();

      UIController.setStatusBadge(state);
      UIController.setProgress(pct, message);

      if (state === 'done') {
        stop();
        try {
          const result = await AlignmentAPI.getResult(jobId);
          AppState.currentResult = result;
          ResultsRenderer.render(result);
          UIController.showPanel('results');
          UIController.setSyncBtnState('ready');
        } catch (err) {
          UIController.showError(`Failed to fetch results: ${err.message}`);
        }
        return;
      }

      if (state === 'error') {
        stop();
        UIController.showError(message || 'The alignment job failed on the server.');
        UIController.setSyncBtnState('ready');
        return;
      }

      // Schedule next poll
      AppState._pollTimer = setTimeout(() => poll(jobId), AppState._backoffMs);

    } catch (err) {
      // Network error → exponential backoff
      console.warn('[Poller] Network error:', err.message);
      AppState._backoffMs = Math.min(AppState._backoffMs * 2, MAX_BACKOFF_MS);
      AppState._pollTimer = setTimeout(() => poll(jobId), AppState._backoffMs);
    }
  }

  return { start, stop };
})();

/* =========================================================
   JSON HIGHLIGHTER
   ========================================================= */
const JSONHighlighter = (() => {
  /**
   * highlight(jsonString) -> htmlString
   * Colors: keys=purple, strings=green, numbers=cyan,
   *         booleans=orange, null=red
   */
  function highlight(jsonString) {
    return jsonString
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(
        /("(?:\\u[0-9a-fA-F]{4}|\\[^u]|[^"\\])*")(\s*:)?|(\b(?:true|false)\b)|(\bnull\b)|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
        (match, str, colon, bool, nil, num) => {
          if (colon !== undefined) {
            // It's a key (string followed by colon)
            return `<span class="json-key">${str}</span>${colon}`;
          }
          if (str !== undefined) {
            return `<span class="json-string">${str}</span>`;
          }
          if (bool !== undefined) {
            return `<span class="json-bool">${bool}</span>`;
          }
          if (nil !== undefined) {
            return `<span class="json-null">${nil}</span>`;
          }
          if (num !== undefined) {
            return `<span class="json-number">${num}</span>`;
          }
          return match;
        }
      );
  }

  return { highlight };
})();

/* =========================================================
   RESULTS RENDERER
   ========================================================= */
const ResultsRenderer = (() => {
  function renderStats(result) {
    const words    = result.words || [];
    const lines    = result.lines || result.segments || [];
    const duration = result.duration || result.audio_duration || null;

    const avgConf = words.length > 0
      ? (words.reduce((s, w) => s + (w.confidence || w.score || 0), 0) / words.length)
      : null;

    document.getElementById('stat-words').textContent      = words.length;
    document.getElementById('stat-lines').textContent      = lines.length;
    document.getElementById('stat-confidence').textContent = avgConf !== null
      ? `${(avgConf * 100).toFixed(1)}%`
      : '—';
    document.getElementById('stat-duration').textContent   = duration !== null
      ? formatTimestamp(duration)
      : '—';
  }

  function renderWords(words) {
    const container = document.getElementById('word-confidence');
    container.innerHTML = '';

    if (!words || words.length === 0) {
      container.innerHTML = '<span style="color:var(--text-faint);font-size:0.82rem">No word data available</span>';
      return;
    }

    const fragment = document.createDocumentFragment();

    words.forEach((w, i) => {
      const conf  = w.confidence || w.score || 0;
      const word  = w.word || w.text || `word_${i}`;
      const start = w.start ?? null;
      const end   = w.end   ?? null;

      let cls = 'word-pill--low';
      if (conf >= 0.85)      cls = 'word-pill--high';
      else if (conf >= 0.6)  cls = 'word-pill--mid';

      const pill = document.createElement('span');
      pill.className       = `word-pill ${cls}`;
      pill.setAttribute('role', 'listitem');
      pill.setAttribute('aria-label', `Word: ${word}, confidence: ${(conf * 100).toFixed(0)}%`);
      pill.tabIndex = 0;

      const tooltipLines = [
        `<strong>${escapeHTML(word)}</strong>`,
        `Confidence: ${(conf * 100).toFixed(1)}%`,
        start !== null ? `Start: ${formatTimestamp(start)}` : null,
        end   !== null ? `End:   ${formatTimestamp(end)}` : null,
      ].filter(Boolean).join('<br/>');

      pill.innerHTML = `${escapeHTML(word)}<span class="tooltip">${tooltipLines}</span>`;

      fragment.appendChild(pill);
    });

    container.appendChild(fragment);
  }

  function renderLines(lines) {
    const container = document.getElementById('lines-list');
    container.innerHTML = '';

    if (!lines || lines.length === 0) {
      container.innerHTML = '<p style="color:var(--text-faint);font-size:0.85rem">No line data available</p>';
      return;
    }

    const fragment = document.createDocumentFragment();

    lines.forEach((line, i) => {
      const text  = line.text || line.lyric || line.line || `Line ${i + 1}`;
      const start = line.start ?? null;
      const end   = line.end   ?? null;

      const card = document.createElement('div');
      card.className = 'line-card';
      card.setAttribute('role', 'listitem');

      const timeStr = (start !== null && end !== null)
        ? `${formatTimestamp(start)} → ${formatTimestamp(end)}`
        : start !== null
          ? `${formatTimestamp(start)}`
          : '—';

      card.innerHTML = `
        <span class="line-card__time">${escapeHTML(timeStr)}</span>
        <span class="line-card__text">${escapeHTML(String(text))}</span>
      `;

      fragment.appendChild(card);
    });

    container.appendChild(fragment);
  }

  function renderJSON(result) {
    const pre = document.getElementById('json-output');
    try {
      const jsonStr = JSON.stringify(result, null, 2);
      pre.innerHTML = JSONHighlighter.highlight(jsonStr);
    } catch (e) {
      pre.textContent = '[Unable to render JSON]';
    }
  }

  function render(result) {
    renderStats(result);
    renderWords(result.words || []);
    renderLines(result.lines || result.segments || []);
    renderJSON(result);
  }

  return { render, renderStats, renderWords, renderLines, renderJSON };
})();

/* =========================================================
   DOWNLOAD & COPY
   ========================================================= */
function downloadJSON(result) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  const filename  = `croonify_result_${timestamp}.json`;
  const blob      = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
  const url       = URL.createObjectURL(blob);
  const a         = document.createElement('a');
  a.href          = url;
  a.download      = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    a.remove();
  }, 200);
  Toast.show(`📥 Downloaded ${filename}`, 'success');
}

async function copyJSON(result) {
  try {
    const text = JSON.stringify(result, null, 2);
    await navigator.clipboard.writeText(text);
    Toast.show('📋 JSON copied to clipboard!', 'success');
  } catch (err) {
    Toast.show('⚠ Could not access clipboard. Try downloading instead.', 'error', 4500);
  }
}

/* =========================================================
   FORM SUBMIT — MAIN WORKFLOW
   ========================================================= */
async function handleSync() {
  const { audioFile, lyricsText, language, aligner, vocalSeparation } = AppState;

  if (!audioFile || !lyricsText.trim()) {
    Toast.show('Please upload audio and paste lyrics first.', 'error');
    return;
  }

  /* Reset panels */
  UIController.hideAllPanels();
  UIController.setSyncBtnState('processing');
  UIController.setProgress(0, 'Submitting job…');
  UIController.setStatusBadge('queued');
  document.getElementById('status-spinner').style.opacity = '1';
  UIController.showPanel('status');

  try {
    const { job_id: jobId } = await AlignmentAPI.submitJob(
      audioFile,
      lyricsText,
      language,
      aligner,
      vocalSeparation,
    );

    if (!jobId) throw new Error('Server did not return a job ID.');

    AppState.jobId = jobId;
    UIController.setStatusBadge('processing');
    UIController.setProgress(0, 'Job accepted — alignment starting…');
    JobPoller.start(jobId);

  } catch (err) {
    UIController.showError(`Failed to submit job: ${err.message}`);
  }
}

/* =========================================================
   CONFIGURATION HANDLERS
   ========================================================= */
function initConfigHandlers() {
  /* Language select */
  document.getElementById('language-select').addEventListener('change', e => {
    AppState.language = e.target.value;
  });

  /* Aligner toggle */
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.toggle-btn').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
      AppState.aligner = btn.dataset.value;
    });
  });

  /* Vocal separation toggle */
  const vocalToggle = document.getElementById('vocal-sep-toggle');
  vocalToggle.addEventListener('change', () => {
    AppState.vocalSeparation = vocalToggle.checked;
    vocalToggle.setAttribute('aria-checked', String(vocalToggle.checked));
  });
}

/* =========================================================
   ACTION BUTTONS
   ========================================================= */
function initActionButtons() {
  /* Sync button */
  document.getElementById('sync-btn').addEventListener('click', handleSync);

  /* Download */
  document.getElementById('btn-download').addEventListener('click', () => {
    if (AppState.currentResult) downloadJSON(AppState.currentResult);
  });

  /* Copy */
  document.getElementById('btn-copy').addEventListener('click', () => {
    if (AppState.currentResult) copyJSON(AppState.currentResult);
  });

  /* Start New */
  document.getElementById('btn-new').addEventListener('click', resetApp);

  /* Retry */
  document.getElementById('btn-retry').addEventListener('click', handleSync);
}

/* =========================================================
   RESET APP
   ========================================================= */
function resetApp() {
  /* Stop any running poll */
  JobPoller.stop();
  AppState.reset();

  /* Clear UI */
  document.getElementById('lyrics-input').value = '';
  document.getElementById('line-count').textContent = '0 lines';
  document.getElementById('char-count').textContent = '0 chars';
  AudioUploadHandler.clearFile();

  UIController.hideAllPanels();
  UIController.setSyncBtnState('disabled');

  /* Scroll to top */
  window.scrollTo({ top: 0, behavior: 'smooth' });

  Toast.show('✨ Ready for a new sync', 'default', 2500);
}

/* =========================================================
   INIT
   ========================================================= */
function init() {
  AudioUploadHandler.init();
  LyricsHandler.init();
  initConfigHandlers();
  initActionButtons();

  /* Initial state */
  UIController.setSyncBtnState('disabled');

  /* Keyboard shortcut: Ctrl+Enter to sync */
  document.addEventListener('keydown', e => {
    const syncBtn = document.getElementById('sync-btn');
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !syncBtn.disabled) {
      e.preventDefault();
      handleSync();
    }
  });

  console.info(
    '%cCroonify AI Lyrics Sync Engine — Frontend Ready',
    'color: #7c3aed; font-size: 14px; font-weight: bold;'
  );
  console.info(`%cAPI base: ${API_BASE}`, 'color: #06b6d4; font-size: 12px;');
}

/* =========================================================
   BOOTSTRAP
   ========================================================= */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
