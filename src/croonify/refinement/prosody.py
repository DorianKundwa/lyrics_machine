"""Prosody-aware word boundary refinement.

After forced alignment produces approximate word timestamps, this module
applies three post-processing steps that leverage raw acoustic signals to
improve temporal accuracy:

1. **Vowel / phoneme stretching** — Extends the end time of a word when RMS
   energy remains elevated beyond the initial alignment boundary, suggesting
   the singer is holding a note.

2. **Silence gap insertion** — Shrinks the end of a word when the VAD mask
   indicates silence in the inter-word gap, preventing one word from
   "eating into" the silence before the next.

3. **Zero-crossing snap** — Fine-tunes boundaries to the nearest waveform
   zero-crossing within a configurable window, minimizing click artifacts
   if the timestamps are used for audio slicing.

4. **Emphasis detection** — Marks words whose peak RMS significantly exceeds
   the track median as *emphasized* (useful for karaoke highlight effects).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from croonify.audio.features import AudioFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config values (overridden by pipeline config)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "vowel_stretch_threshold": 0.7,   # fraction of peak RMS that triggers stretch
    "min_silence_ms": 80,             # minimum silence to insert micro-pause (ms)
    "boundary_snap_ms": 20,           # ±window for zero-crossing snap (ms)
    "rms_extend_ratio": 0.15,         # max extension as fraction of word duration
}

EMPHASIS_RATIO: float = 1.5          # peak RMS / median RMS ≥ this → emphasized


class ProsodyRefiner:
    """Refine word timestamps using acoustic prosody signals.

    Parameters
    ----------
    config:
        Dictionary of prosody config keys (merged with :data:`DEFAULT_CONFIG`).

    Usage
    -----
    >>> refiner = ProsodyRefiner(config={})
    >>> refined = refiner.refine(words, audio_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = {**DEFAULT_CONFIG, **config}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refine(
        self,
        words: List[Dict[str, Any]],
        features: AudioFeatures,
    ) -> List[Dict[str, Any]]:
        """Apply all prosody refinements to *words* in-place (returns new list).

        Parameters
        ----------
        words:
            List of word dicts with ``text``, ``start``, ``end``, ``score``.
        features:
            Acoustic features extracted by :class:`~croonify.audio.features.FeatureExtractor`.

        Returns
        -------
        list[dict]
            Updated word list with refined timestamps and optional ``emphasized``
            flag.
        """
        if not words:
            return words

        cfg = self.config
        sr = features.sample_rate
        hop = sr // 100  # assuming 10 ms hop (160 @ 16 kHz)

        # Pre-compute global RMS statistics for emphasis detection
        global_median_rms = float(np.median(features.rms_energy))
        global_p10_rms = float(np.percentile(features.rms_energy, 10))

        snap_window_ms = cfg["boundary_snap_ms"]
        min_silence_ms = cfg["min_silence_ms"]
        rms_extend_ratio = cfg["rms_extend_ratio"]
        vowel_threshold = cfg["vowel_stretch_threshold"]

        refined: List[Dict[str, Any]] = []

        for i, word in enumerate(words):
            w = dict(word)  # shallow copy to avoid mutating caller's data
            start = float(w["start"])
            end = float(w["end"])

            # Skip degenerate windows
            if end <= start:
                refined.append(w)
                continue

            # --- Vowel stretch -----------------------------------------------
            end = self._apply_vowel_stretch(
                start=start,
                end=end,
                features=features,
                next_start=words[i + 1]["start"] if i + 1 < len(words) else features.duration,
                vowel_threshold=vowel_threshold,
                rms_extend_ratio=rms_extend_ratio,
                hop=hop,
                sr=sr,
            )

            # --- Silence gap -------------------------------------------------
            if i + 1 < len(words):
                next_start = float(words[i + 1]["start"])
                end = self._apply_silence_gap(
                    end=end,
                    next_start=next_start,
                    features=features,
                    min_silence_ms=min_silence_ms,
                    hop=hop,
                    sr=sr,
                )

            # --- Zero-crossing snap -----------------------------------------
            start = self._snap_to_zero_crossing(
                features.waveform, start, sr, snap_window_ms
            )
            end = self._snap_to_zero_crossing(
                features.waveform, end, sr, snap_window_ms
            )

            # Ensure monotonicity after snapping
            if end <= start:
                end = start + 0.01

            w["start"] = round(start, 4)
            w["end"] = round(end, 4)

            # --- Emphasis detection -----------------------------------------
            word_rms = features.rms_in_window(w["start"], w["end"])
            if len(word_rms) > 0:
                peak_rms = float(np.max(word_rms))
                if global_median_rms > 0 and (peak_rms / (global_median_rms + 1e-8)) >= EMPHASIS_RATIO:
                    w["emphasized"] = True

            refined.append(w)

        return refined

    # ------------------------------------------------------------------
    # Refinement steps
    # ------------------------------------------------------------------

    def _apply_vowel_stretch(
        self,
        start: float,
        end: float,
        features: AudioFeatures,
        next_start: float,
        vowel_threshold: float,
        rms_extend_ratio: float,
        hop: int,
        sr: int,
    ) -> float:
        """Extend *end* if energy remains elevated past the alignment boundary.

        The aligner often clips the end of held vowels early.  We extend by up
        to ``rms_extend_ratio * word_duration`` seconds while RMS > threshold.
        """
        word_duration = end - start
        max_extension = word_duration * rms_extend_ratio
        max_end = min(end + max_extension, next_start - 0.005)

        # Get RMS in a look-ahead window
        f_end = self._time_to_frame(end, hop, sr)
        f_max = self._time_to_frame(max_end, hop, sr)

        if f_max <= f_end:
            return end

        lookahead_rms = features.rms_energy[f_end: f_max + 1]
        if len(lookahead_rms) == 0:
            return end

        # Compute peak RMS in the word window to set threshold
        f_start = self._time_to_frame(start, hop, sr)
        word_rms = features.rms_energy[f_start: f_end + 1]
        if len(word_rms) == 0:
            return end

        peak_rms = float(np.max(word_rms))
        stretch_thresh = peak_rms * vowel_threshold

        # Find how far past f_end the energy stays above threshold
        new_f_end = f_end
        for fi, rms_val in enumerate(lookahead_rms):
            if rms_val >= stretch_thresh:
                new_f_end = f_end + fi
            else:
                break

        return min(self._frame_to_time(new_f_end, hop, sr), max_end)

    def _apply_silence_gap(
        self,
        end: float,
        next_start: float,
        features: AudioFeatures,
        min_silence_ms: float,
        hop: int,
        sr: int,
    ) -> float:
        """Shrink *end* if there is silence in the inter-word gap.

        If the VAD mask shows silence in the gap [end, next_start] for at
        least ``min_silence_ms`` milliseconds, we pull *end* back to the start
        of the silence region.
        """
        gap_s = next_start - end
        min_silence_s = min_silence_ms / 1000.0

        if gap_s < min_silence_s:
            return end

        f_end = self._time_to_frame(end, hop, sr)
        f_next = self._time_to_frame(next_start, hop, sr)

        if f_next <= f_end:
            return end

        gap_vad = features.vad_mask[f_end: f_next]
        if len(gap_vad) == 0:
            return end

        # Find first silent frame in the gap
        silent_indices = np.where(~gap_vad)[0]
        if len(silent_indices) == 0:
            return end  # no silence — no adjustment

        first_silent_frame = f_end + int(silent_indices[0])
        new_end = self._frame_to_time(first_silent_frame, hop, sr)

        # Only apply if adjustment is meaningful
        if end - new_end > 0.005:
            return new_end
        return end

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_to_frame(time_s: float, hop_length: int, sr: int) -> int:
        """Convert time in seconds to the nearest frame index."""
        return max(0, int(round(time_s * sr / hop_length)))

    @staticmethod
    def _frame_to_time(frame: int, hop_length: int, sr: int) -> float:
        """Convert frame index to time in seconds."""
        return float(frame * hop_length / sr)

    @staticmethod
    def _snap_to_zero_crossing(
        waveform: np.ndarray,
        time_s: float,
        sr: int,
        window_ms: float,
    ) -> float:
        """Snap *time_s* to the nearest zero-crossing within ±window_ms.

        Zero-crossings are where the waveform sign changes.  Snapping
        boundaries to these points eliminates click artifacts when slicing.

        Parameters
        ----------
        waveform:
            Raw mono waveform samples.
        time_s:
            Target time in seconds.
        sr:
            Sample rate.
        window_ms:
            Search window in milliseconds (±).

        Returns
        -------
        float
            Adjusted time in seconds (unchanged if no zero-crossing found).
        """
        center_sample = int(round(time_s * sr))
        window_samples = int(round(window_ms / 1000.0 * sr))

        lo = max(0, center_sample - window_samples)
        hi = min(len(waveform) - 1, center_sample + window_samples)

        if lo >= hi:
            return time_s

        segment = waveform[lo: hi + 1]

        # Detect zero-crossings (sign changes between adjacent samples)
        signs = np.sign(segment)
        crossings = np.where(np.diff(signs) != 0)[0]

        if len(crossings) == 0:
            return time_s

        # Find crossing closest to center_sample
        crossing_samples = lo + crossings  # absolute sample indices
        distances = np.abs(crossing_samples - center_sample)
        best_idx = int(np.argmin(distances))
        snapped_sample = int(crossing_samples[best_idx])

        return float(snapped_sample / sr)
