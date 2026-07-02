/**
 * Lyric Editor — Canvas Timeline
 * Exposes: window.LE_Timeline
 *
 * Canvas layout (top → bottom):
 *   [0 .. RULER_H]              Time ruler with tick marks
 *   [RULER_H+4 .. +BAR_H]       Segment bars (colour-coded)
 *   [BAR_BOTTOM+6 .. +YHANDLE_H] Per-segment ↕ Y-offset handle
 *
 * Interactions:
 *   • Drag segment body      → move start+end (constant duration)
 *   • Drag left/right edge   → resize start / end
 *   • Drag ↕ handle up/down  → adjust y_offset (px from centre, + = up)
 *   • Click segment          → select (highlights table row)
 *   • Scroll wheel           → zoom in / out
 */
(function () {
  'use strict';

  // ── Layout constants ─────────────────────────────────────────────────────────
  const RULER_H    = 22;
  const BAR_Y      = RULER_H + 4;
  const BAR_H      = 64;
  const BAR_BOTTOM = BAR_Y + BAR_H;
  const YH_Y       = BAR_BOTTOM + 6;
  const YH_H       = 20;
  const CANVAS_H   = YH_Y + YH_H + 8;
  const EDGE_PX    = 8;   // px width of resize-handle zone at each edge

  // Colour palette for segments (cycles)
  const COLORS = [
    '#7c3aed', '#ff3c64', '#0ea5e9', '#10b981',
    '#f59e0b', '#e879f9', '#06b6d4', '#ef4444',
  ];

  // ── State ────────────────────────────────────────────────────────────────────
  let canvas, ctx, container;
  let pixPerSec  = 80;
  let drag       = null;
  let isPanning  = false;
  let panStartX  = 0;
  let panScrollX = 0;

  // ── Public: init ─────────────────────────────────────────────────────────────

  function init() {
    container = document.getElementById('leTimelineWrap');
    canvas    = document.getElementById('leTimeline');
    if (!canvas || !container) return;

    ctx = null;  // will be created on first draw
    canvas.height = CANVAS_H;

    _resize();
    window.addEventListener('resize', _resize);

    canvas.addEventListener('mousedown', _onDown);
    window.addEventListener('mousemove', _onMove);
    window.addEventListener('mouseup',   _onUp);
    canvas.addEventListener('wheel',     _onWheel, { passive: false });

    LE.onChange(draw);
    LE.onSelect(() => draw(LE.segments));
  }

  // ── Sizing ───────────────────────────────────────────────────────────────────

  function _resize() {
    if (!canvas || !container) return;
    const dur = LE.totalDuration || Math.max(...(LE.segments || []).map(s => s.end), 300);
    const newW = Math.max(container.clientWidth, dur * pixPerSec);
    if (canvas.width !== newW) {
      canvas.width = newW;
    }
    draw(LE.segments);
  }

  // ── Coordinate helpers ───────────────────────────────────────────────────────

  function _timeToX(t) { return t * pixPerSec; }
  function _xToTime(x) { return x / pixPerSec; }

  // ── Draw ─────────────────────────────────────────────────────────────────────

  function draw(segments) {
    if (!canvas || !canvas.width) return;
    if (!ctx) ctx = canvas.getContext('2d');
    const W = canvas.width;

    // Background
    ctx.clearRect(0, 0, W, CANVAS_H);
    ctx.fillStyle = '#0b0e1a';
    ctx.fillRect(0, 0, W, CANVAS_H);

    _drawRuler(W);

    if (!segments || !segments.length) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '12px DM Sans, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No segments', W / 2, BAR_Y + BAR_H / 2);
      return;
    }

    segments.forEach((seg, i) => _drawSegment(seg, i));
  }

  function _drawRuler(W) {
    ctx.fillStyle = '#131722';
    ctx.fillRect(0, 0, W, RULER_H);

    const dur = LE.totalDuration || Math.max(...(LE.segments || []).map(s => s.end), 300);
    const step = _niceStep((container.clientWidth || 800) / pixPerSec / 8);

    ctx.font = '9px DM Sans, monospace';
    ctx.textAlign = 'center';

    let t = 0;
    while (t <= dur) {
      const x = Math.round(_timeToX(t));
      ctx.fillStyle = '#1e2433';
      ctx.fillRect(x, RULER_H - 5, 1, 5);
      ctx.fillStyle = '#6b7280';
      ctx.fillText(LE.toDisplay(t), x, RULER_H - 7);
      t = Math.round((t + step) * 1000) / 1000;
    }
  }

  function _niceStep(raw) {
    for (const s of [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]) if (s >= raw) return s;
    return 300;
  }

  function _drawSegment(seg, i) {
    const color = COLORS[i % COLORS.length];
    const x1  = _timeToX(seg.start);
    const x2  = _timeToX(seg.end);
    const w   = Math.max(6, x2 - x1);
    const sel = (i === LE.selectedIndex);

    // ── Bar ──
    if (sel) { ctx.shadowColor = color; ctx.shadowBlur = 14; }
    ctx.fillStyle   = sel ? color : color + 'aa';
    _rr(ctx, x1, BAR_Y, w, BAR_H, 6);
    ctx.fill();
    ctx.shadowBlur  = 0;

    ctx.strokeStyle = sel ? '#ffffff' : color + 'cc';
    ctx.lineWidth   = sel ? 1.5 : 1;
    _rr(ctx, x1, BAR_Y, w, BAR_H, 6);
    ctx.stroke();

    // ── Label inside bar ──
    if (w > 28) {
      ctx.save();
      _rr(ctx, x1 + 1, BAR_Y + 1, w - 2, BAR_H - 2, 5);
      ctx.clip();

      ctx.fillStyle = 'rgba(255,255,255,0.92)';
      ctx.font      = `${sel ? 'bold ' : ''}11px DM Sans, sans-serif`;
      ctx.textAlign = 'left';
      ctx.fillText(_truncate(seg.text, Math.floor(w / 7)), x1 + 6, BAR_Y + 16);

      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font      = '9px monospace';
      ctx.fillText(`${LE.toDisplay(seg.start)} → ${LE.toDisplay(seg.end)}`, x1 + 6, BAR_Y + 29);
      ctx.restore();
    }

    // ── Resize edge handles ──
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.fillRect(x1,           BAR_Y + 4, 4, BAR_H - 8);
    ctx.fillRect(x1 + w - 4,  BAR_Y + 4, 4, BAR_H - 8);

    // ── Y-offset handle ──
    const yhW = Math.max(44, w * 0.55);
    const yhX = x1 + (w - yhW) / 2;
    const yOff = seg.y_offset || 0;

    ctx.fillStyle   = '#141928';
    _rr(ctx, yhX, YH_Y, yhW, YH_H, YH_H / 2);
    ctx.fill();
    ctx.strokeStyle = color + 'bb';
    ctx.lineWidth   = 1;
    _rr(ctx, yhX, YH_Y, yhW, YH_H, YH_H / 2);
    ctx.stroke();

    ctx.fillStyle = '#e8eaf0';
    ctx.font      = '9px DM Sans, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`↕ ${yOff > 0 ? '+' : ''}${yOff}px`, yhX + yhW / 2, YH_Y + 13);
  }

  // ── Rounded rect helper ───────────────────────────────────────────────────────

  function _rr(c, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    c.beginPath();
    c.moveTo(x + r, y);
    c.lineTo(x + w - r, y);
    c.arcTo(x + w, y,     x + w, y + r,     r);
    c.lineTo(x + w, y + h - r);
    c.arcTo(x + w, y + h, x + w - r, y + h, r);
    c.lineTo(x + r, y + h);
    c.arcTo(x,     y + h, x,     y + h - r, r);
    c.lineTo(x,     y + r);
    c.arcTo(x,     y,     x + r, y,          r);
    c.closePath();
  }

  function _truncate(s, maxChars) {
    if (!s) return '';
    s = String(s);
    return s.length > maxChars ? s.slice(0, maxChars - 1) + '…' : s;
  }

  // ── Hit testing ──────────────────────────────────────────────────────────────

  function _hit(x, y) {
    const segs = LE.segments;
    // Iterate in reverse so top-most (last drawn) wins
    for (let i = segs.length - 1; i >= 0; i--) {
      const seg = segs[i];
      const x1  = _timeToX(seg.start);
      const x2  = _timeToX(seg.end);
      const w   = Math.max(6, x2 - x1);

      // Y-handle hit
      const yhW = Math.max(44, w * 0.55);
      const yhX = x1 + (w - yhW) / 2;
      if (x >= yhX && x <= yhX + yhW && y >= YH_Y && y <= YH_Y + YH_H)
        return { type: 'yhandle', i };

      // Bar hit
      if (y >= BAR_Y && y <= BAR_BOTTOM && x >= x1 && x <= x1 + w) {
        if (x <= x1 + EDGE_PX)      return { type: 'left',  i };
        if (x >= x1 + w - EDGE_PX)  return { type: 'right', i };
        return { type: 'body', i };
      }
    }
    return null;
  }

  // ── Mouse events ─────────────────────────────────────────────────────────────

  function _onDown(e) {
    // Middle click or shift+click for panning
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      e.preventDefault();
      isPanning = true;
      panStartX = e.clientX;
      panScrollX = container.scrollLeft;
      canvas.style.cursor = 'grabbing';
      return;
    }

    if (e.button !== 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const hit = _hit(x, y);
    if (!hit) return;

    e.preventDefault();
    LE.select(hit.i);
    LE.pushUndo();

    const seg = LE.segments[hit.i];
    drag = {
      type:       hit.type,
      i:          hit.i,
      startX:     x,
      startY:     y,
      origStart:  seg.start,
      origEnd:    seg.end,
      origYOff:   seg.y_offset || 0,
    };
    canvas.style.cursor = _cursor(hit.type);
  }

  function _onMove(e) {
    if (isPanning) {
      const dx = e.clientX - panStartX;
      container.scrollLeft = panScrollX - dx;
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (!drag) {
      // Hover cursor hint
      if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
        const hit = _hit(x, y);
        canvas.style.cursor = hit ? _cursor(hit.type) : 'default';
      }
      return;
    }

    const dt  = _xToTime(x) - _xToTime(drag.startX);
    const dy  = y - drag.startY;
    const seg = LE.segments[drag.i];
    const MIN = 0.05;

    switch (drag.type) {
      case 'body': {
        const dur = drag.origEnd - drag.origStart;
        seg.start = Math.max(0, drag.origStart + dt);
        seg.end   = seg.start + dur;
        break;
      }
      case 'left':
        seg.start = Math.min(drag.origEnd - MIN, Math.max(0, drag.origStart + dt));
        break;
      case 'right':
        seg.end = Math.max(drag.origStart + MIN, drag.origEnd + dt);
        break;
      case 'yhandle':
        // Drag up (negative dy) → positive y_offset → text moves up on screen
        seg.y_offset = Math.round(Math.max(-400, Math.min(400, drag.origYOff - dy)));
        break;
    }

    // Update canvas only during drag (table refreshes on mouseup via LE.notify)
    draw(LE.segments);
  }

  function _onUp() {
    if (isPanning) {
      isPanning = false;
      canvas.style.cursor = 'default';
      return;
    }
    if (!drag) return;
    canvas.style.cursor = 'default';
    drag = null;
    LE.notify();
    if (typeof LE_Actions !== 'undefined') LE_Actions.scheduleAutoSave();
  }

  function _onWheel(e) {
    // Allow horizontal scrolling normally without zooming
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;

    e.preventDefault();
    
    // Zoom logic
    const oldPixPerSec = pixPerSec;
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    pixPerSec = Math.max(5, Math.min(800, pixPerSec * factor));

    // Keep the cursor position stable
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const timeAtCursor = x / oldPixPerSec;
    
    _resize();

    // Adjust scroll to keep timeAtCursor under the mouse
    const newX = timeAtCursor * pixPerSec;
    container.scrollLeft += (newX - x);
  }

  function _cursor(type) {
    return type === 'yhandle' ? 'ns-resize'
         : type === 'body'    ? 'grab'
         : 'ew-resize';
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  window.LE_Timeline = { init, draw };
})();
