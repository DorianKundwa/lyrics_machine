"""Word-to-line grouping for karaoke-style output.

Lyrics synchronization produces word-level timestamps.  For display in a
karaoke or subtitle player, these must be grouped into *lines* (display units)
that:

* Respect the original lyric line structure (priority 1)
* Break on long silences between words (priority 2)
* Stay within configurable word-count and duration limits (priority 3)
* Optionally snap boundaries to the nearest beat (priority 4)

The segmenter operates on a flat list of word dicts (output of the aligner
+ prosody refiner) and produces line dicts suitable for JSON output.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "max_words": 8,          # maximum words per line
    "max_duration_s": 4.0,   # maximum line duration in seconds
    "min_gap_s": 0.25,       # silence gap that forces a line break (seconds)
    "beat_snap": False,       # snap line start/end to nearest beat
}


class LineSegmenter:
    """Group aligned words into display lines.

    Parameters
    ----------
    config:
        Dictionary with segmentation settings (merged with :data:`DEFAULT_CONFIG`).

    Usage
    -----
    >>> seg = LineSegmenter(config={})
    >>> lines = seg.segment(words, beat_times=beat_times)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = {**DEFAULT_CONFIG, **config}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def segment(
        self,
        words: List[Dict[str, Any]],
        beat_times: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Segment *words* into lyric lines.

        Parameters
        ----------
        words:
            Flat list of word dicts.  Each dict must have keys
            ``text``, ``start``, ``end``.  An optional boolean
            ``line_break_after`` key (set by the normalizer from the original
            lyric structure) forces a break after that word.
        beat_times:
            Optional array of beat positions in seconds.  Used when
            ``config.beat_snap`` is ``True``.

        Returns
        -------
        list[dict]
            Each line dict::

                {
                    "start": float,       # onset of first word
                    "end":   float,       # offset of last word
                    "text":  str,         # space-joined word texts
                    "words": list[dict],  # constituent word dicts
                }
        """
        if not words:
            return []

        cfg = self.config
        beat_snap = cfg["beat_snap"] and beat_times is not None and len(beat_times) > 0

        lines: List[Dict[str, Any]] = []
        current_line: List[Dict[str, Any]] = []

        for i, word in enumerate(words):
            if not current_line:
                current_line.append(word)
                continue

            # Decide whether to break before this word
            if self._should_break(
                word_i=current_line[-1],
                word_j=word,
                current_line_words=current_line,
                config=cfg,
            ):
                lines.append(self._build_line(current_line, beat_snap, beat_times))
                current_line = [word]
            else:
                current_line.append(word)

        # Flush last line
        if current_line:
            lines.append(self._build_line(current_line, beat_snap, beat_times))

        logger.info(
            "Segmentation: %d words → %d lines (max_words=%d, min_gap=%.2f s)",
            len(words),
            len(lines),
            cfg["max_words"],
            cfg["min_gap_s"],
        )
        return lines

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_break(
        word_i: Dict[str, Any],
        word_j: Dict[str, Any],
        current_line_words: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> bool:
        """Return True if a line break should be inserted before *word_j*.

        Checks (in priority order):
        1. ``line_break_after`` flag on *word_i* (set from lyric structure).
        2. Silence gap between *word_i* and *word_j* exceeds ``min_gap_s``.
        3. Adding *word_j* would exceed ``max_words``.
        4. Adding *word_j* would exceed ``max_duration_s``.
        """
        # Priority 1: explicit line break marker from lyric structure
        if word_i.get("line_break_after", False):
            return True

        gap = float(word_j["start"]) - float(word_i["end"])

        # Priority 2: long silence gap
        if gap >= config["min_gap_s"]:
            return True

        # Priority 3: word count limit
        if len(current_line_words) >= config["max_words"]:
            return True

        # Priority 4: duration limit
        line_start = float(current_line_words[0]["start"])
        line_end_projected = float(word_j["end"])
        if line_end_projected - line_start > config["max_duration_s"]:
            return True

        return False

    def _build_line(
        self,
        word_list: List[Dict[str, Any]],
        beat_snap: bool,
        beat_times: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Build a line dict from *word_list*, optionally snapping to beats."""
        start = float(word_list[0]["start"])
        end = float(word_list[-1]["end"])
        text = " ".join(w["text"] for w in word_list)

        if beat_snap and beat_times is not None:
            start = self._snap_to_beat(start, beat_times)
            end = self._snap_to_beat(end, beat_times)
            if end <= start:
                end = float(word_list[-1]["end"])

        return {
            "start": round(start, 4),
            "end": round(end, 4),
            "text": text,
            "words": [dict(w) for w in word_list],
        }

    @staticmethod
    def _snap_to_beat(time_s: float, beat_times: np.ndarray) -> float:
        """Return the beat time in *beat_times* closest to *time_s*.

        Parameters
        ----------
        time_s:
            Target time in seconds.
        beat_times:
            Sorted array of beat positions in seconds.

        Returns
        -------
        float
            Beat time closest to *time_s*.
        """
        if len(beat_times) == 0:
            return time_s
        idx = int(np.argmin(np.abs(beat_times - time_s)))
        return float(beat_times[idx])
