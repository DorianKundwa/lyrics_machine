/**
 * Lyric Editor — Save, Re-render, Auto-save (4 s debounce), Reset
 * Exposes: window.LE_Actions
 *
 * API endpoints used:
 *   GET  /alignment/<job_id>            → load segments
 *   PATCH /alignment/<job_id>           → save corrected segments
 *   POST  /rerender/<job_id>            → trigger re-render
 *   GET   /rerender_status/<rerender_id>→ poll progress
 */
(function () {
  'use strict';

  let _autoSaveTimer = null;
  let _rerenderJobId = null;
  let _pollActive    = false;

  // ── Auto-save (4 s debounce) ─────────────────────────────────────────────────

  function scheduleAutoSave() {
    clearTimeout(_autoSaveTimer);
    _setStatus('Unsaved changes…', 'pending');
    _autoSaveTimer = setTimeout(save, 4000);
  }

  // ── Save → PATCH /alignment/<job_id> ─────────────────────────────────────────

  async function save() {
    clearTimeout(_autoSaveTimer);
    if (!LE.jobId || !LE.segments.length) return;

    _setStatus('Saving…', 'pending');
    try {
      const res = await fetch(`/alignment/${LE.jobId}`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          segments:       LE.segments,
          active_color:   LE.activeColor   || '#FFFFFF',
          upcoming_color: LE.upcomingColor || '#FF0000',
          sung_color:     LE.sungColor     || '#FFFFFF',
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      _setStatus('Saved ✓', 'ok');
      setTimeout(() => _setStatus('', ''), 3000);
    } catch (err) {
      _setStatus('Save failed ✗', 'err');
      console.error('[LE] save:', err);
    }
  }

  // ── Re-render → POST /rerender/<job_id> ──────────────────────────────────────

  async function rerender() {
    if (!LE.jobId) return;

    const btn = document.getElementById('leRerender');
    if (btn) btn.disabled = true;

    await save();   // always persist first

    _showRerenderPanel();
    _setRerenderProgress(5, 'Queuing re-render…');

    try {
      const res = await fetch(`/rerender/${LE.jobId}`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { rerender_id } = await res.json();
      _rerenderJobId = rerender_id;
      _pollActive    = true;
      _pollRerender();
    } catch (err) {
      _setRerenderProgress(0, `⚠ Error: ${err.message}`);
      if (btn) btn.disabled = false;
    }
  }

  async function _pollRerender() {
    while (_pollActive) {
      await _sleep(1500);
      if (!_pollActive) break;
      try {
        const res  = await fetch(`/rerender_status/${_rerenderJobId}`);
        const data = await res.json();

        _setRerenderProgress(data.progress || 0, data.step || 'Working…');

        if (data.status === 'complete') {
          _pollActive = false;
          _setRerenderProgress(100, '✅ Re-render complete!');
          const dl = document.getElementById('leRerenderDl');
          if (dl && data.download_url) {
            dl.href = data.download_url;
            dl.style.display = 'inline-flex';
          }
          const btn = document.getElementById('leRerender');
          if (btn) btn.disabled = false;
          return;
        }
        if (data.status === 'error') {
          _pollActive = false;
          _setRerenderProgress(0, `⚠ ${data.error || 'Re-render failed'}`);
          const btn = document.getElementById('leRerender');
          if (btn) btn.disabled = false;
          return;
        }
      } catch (_) { /* transient — keep polling */ }
    }
  }

  // ── Reset ─────────────────────────────────────────────────────────────────────

  function reset() {
    if (!LE.originalSegments) return;
    if (!confirm('Discard all edits and restore the original alignment?')) return;
    LE.pushUndo();
    LE.segments = JSON.parse(JSON.stringify(LE.originalSegments));
    LE.notify();
    scheduleAutoSave();
  }

  // ── Init (called by showResults after a job completes) ───────────────────────

  async function initEditor(jobId) {
    LE.jobId = jobId;

    // Clear stale listeners from a previous job
    LE._changeListeners = [];
    LE._selectListeners = [];
    LE._undoStack       = [];
    LE._redoStack       = [];
    LE.selectedIndex    = -1;
    LE._syncUndoBtns();

    // Hide re-render panel from previous run
    const rrPanel = document.getElementById('leRerenderPanel');
    if (rrPanel) rrPanel.style.display = 'none';
    const dl = document.getElementById('leRerenderDl');
    if (dl) dl.style.display = 'none';

    try {
      const res = await fetch(`/alignment/${jobId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      LE.segments         = data.segments || [];
      LE.totalDuration    = data.total_duration || 0;
      LE.fontName         = data.font_name  || 'Arial';
      LE.fontSize         = data.font_size  || 72;
      LE.activeColor      = data.active_color   || '#FFFFFF';
      LE.upcomingColor    = data.upcoming_color || '#FF0000';
      LE.sungColor        = data.sung_color     || '#FFFFFF';
      LE.originalSegments = JSON.parse(JSON.stringify(LE.segments));

      // Sync color pickers to loaded values
      const _cp = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
      _cp('leActiveColor',   LE.activeColor);
      _cp('leUpcomingColor', LE.upcomingColor);
      _cp('leSungColor',     LE.sungColor);

      // Register sub-module listeners fresh
      LE_Table.init();
      LE_Timeline.init();

      // Show panel and populate UI
      const panel = document.getElementById('editorPanel');
      if (panel) {
        panel.style.display = 'block';
        setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 150);
      }

      LE.notify();
      _setStatus('', '');

    } catch (err) {
      console.warn('[LE] Could not load alignment:', err);
      const panel = document.getElementById('editorPanel');
      if (panel) panel.style.display = 'none';
    }
  }

  // ── UI helpers ────────────────────────────────────────────────────────────────

  function _setStatus(msg, cls) {
    const el = document.getElementById('leAutoSaveStatus');
    if (!el) return;
    el.textContent = msg;
    el.className   = `le-status${cls ? ' le-status--' + cls : ''}`;
  }

  function _showRerenderPanel() {
    const panel = document.getElementById('leRerenderPanel');
    if (panel) panel.style.display = 'block';
    const bar = document.getElementById('leRerenderBar');
    if (bar) { bar.style.width = '0%'; bar.style.background = ''; }
  }

  function _setRerenderProgress(pct, msg) {
    const bar  = document.getElementById('leRerenderBar');
    const step = document.getElementById('leRerenderStep');
    if (bar)  bar.style.width    = pct + '%';
    if (step) step.textContent   = msg;
  }

  function _sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Button wiring (runs after DOM is ready) ───────────────────────────────────

  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('leUndo')    ?.addEventListener('click', () => LE.undo());
    document.getElementById('leRedo')    ?.addEventListener('click', () => LE.redo());
    document.getElementById('leReset')   ?.addEventListener('click', reset);
    document.getElementById('leSave')    ?.addEventListener('click', save);
    document.getElementById('leRerender')?.addEventListener('click', rerender);

    // Color picker change listeners
    const _colorMap = {
      leActiveColor:   v => { LE.activeColor   = v; },
      leUpcomingColor: v => { LE.upcomingColor = v; },
      leSungColor:     v => { LE.sungColor     = v; },
    };
    Object.keys(_colorMap).forEach(id => {
      document.getElementById(id)?.addEventListener('input', e => {
        _colorMap[id](e.target.value);
        scheduleAutoSave();
      });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault(); LE.undo();
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault(); LE.redo();
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault(); save();
      }
    });
  });

  // ── Public API ────────────────────────────────────────────────────────────────

  window.LE_Actions = { initEditor, save, reset, scheduleAutoSave };
})();
