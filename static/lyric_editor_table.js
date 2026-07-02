/**
 * Lyric Editor — Editable Data Table
 * Exposes: window.LE_Table
 *
 * Columns: # | Start | End | Dur | Text | Y (px)
 * • Single-click row  → select (highlighted in timeline)
 * • Double-click cell → inline edit (Enter commits, Escape discards)
 */
(function () {
  'use strict';

  // ── Render ──────────────────────────────────────────────────────────────────

  function render(segments) {
    const el = document.getElementById('leTable');
    if (!el) return;

    if (!segments || !segments.length) {
      el.innerHTML = '<p class="le-empty">No lyric segments loaded.</p>';
      return;
    }

    const rows = segments.map((s, i) => {
      const sel = i === LE.selectedIndex ? ' le-row--sel' : '';
      return `<tr class="le-row${sel}" data-i="${i}">
        <td class="le-td le-idx">${i + 1}</td>
        <td class="le-td le-time" data-field="start" data-i="${i}">${LE.toDisplay(s.start)}</td>
        <td class="le-td le-time" data-field="end"   data-i="${i}">${LE.toDisplay(s.end)}</td>
        <td class="le-td le-dur">${(s.end - s.start).toFixed(2)}s</td>
        <td class="le-td le-text" data-field="text"     data-i="${i}">${esc(s.text)}</td>
        <td class="le-td le-yoff" data-field="y_offset" data-i="${i}">${s.y_offset || 0}</td>
      </tr>`;
    }).join('');

    el.innerHTML = `
      <div class="le-table-wrap">
        <table class="le-table">
          <thead>
            <tr>
              <th class="le-th">#</th>
              <th class="le-th">Start</th>
              <th class="le-th">End</th>
              <th class="le-th">Dur</th>
              <th class="le-th le-th-text">Text <span style="font-weight:400;opacity:.5">(dbl-click to edit)</span></th>
              <th class="le-th">Y (px)</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;

    // Row single-click → select
    el.querySelectorAll('.le-row').forEach(row => {
      row.addEventListener('click', e => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        LE.select(parseInt(row.dataset.i, 10));
      });
    });

    // Cell double-click → inline edit
    el.querySelectorAll('.le-td[data-field]').forEach(td => {
      td.addEventListener('dblclick', () => startEdit(td));
    });
  }

  // ── Inline Edit ─────────────────────────────────────────────────────────────

  function startEdit(td) {
    const field = td.dataset.field;
    const i     = parseInt(td.dataset.i, 10);
    const seg   = LE.segments[i];
    if (!seg) return;

    const rawVal =
      field === 'start'    ? LE.toDisplay(seg.start)
      : field === 'end'    ? LE.toDisplay(seg.end)
      : field === 'y_offset' ? (seg.y_offset || 0)
      : seg.text;

    const isText = (field === 'text');
    td.innerHTML = isText
      ? `<textarea class="le-inline-input le-inline-ta" rows="2">${esc(String(rawVal))}</textarea>`
      : `<input class="le-inline-input" value="${esc(String(rawVal))}">`;

    const inp = td.querySelector('input, textarea');
    inp.focus();
    if (inp.select) inp.select();

    const commit = () => {
      const val = inp.value;
      LE.pushUndo();
      if      (field === 'start')    seg.start    = Math.max(0, LE.fromDisplay(val));
      else if (field === 'end')      seg.end      = Math.max(seg.start + 0.05, LE.fromDisplay(val));
      else if (field === 'y_offset') seg.y_offset = parseInt(val, 10) || 0;
      else                           seg.text     = val;
      LE.notify();
      if (typeof LE_Actions !== 'undefined') LE_Actions.scheduleAutoSave();
    };

    inp.addEventListener('blur', commit);
    inp.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); inp.blur(); }
      if (e.key === 'Escape') { LE.notify(); }  // discard
    });
  }

  // ── Selection Highlight ──────────────────────────────────────────────────────

  function highlightRow(i) {
    document.querySelectorAll('#leTable .le-row').forEach(r =>
      r.classList.toggle('le-row--sel', parseInt(r.dataset.i, 10) === i)
    );
    const row = document.querySelector(`#leTable .le-row[data-i="${i}"]`);
    if (row) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // ── Utility ──────────────────────────────────────────────────────────────────

  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  function init() {
    LE.onChange(render);
    LE.onSelect(highlightRow);
  }

  window.LE_Table = { init, render };
})();
