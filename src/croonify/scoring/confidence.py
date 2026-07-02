"""Composite confidence scoring for aligned words.

Each word produced by the aligner receives a raw ``score`` field from the
alignment model (0–1, where 1 = perfect confidence).  This module combines
that score with two additional acoustic quality signals:

1. **VAD coverage** — fraction of the word's time window where the Voice
   Activity Detector reports speech.  A word falling entirely in a silent
   region is almost certainly a mis-alignment.

2. **SNR estimate** — ratio of mean RMS in the word window to the global
   noise floor (10th-percentile RMS).  Words in noisy or low-energy regions
   receive a lower score.

The composite formula is a weighted average::

    composite = 0.5 * alignment_score
              + 0.3 * vad_coverage
              + 0.2 * min(snr_estimate, 1.0)

This heuristic has been empirically tuned so that words with
``composite < 0.5`` reliably identify alignment errors that benefit from
manual review or re-alignment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from croonify.audio.features import AudioFeatures

logger = logging.getLogger(__name__)

# Composite weight constants — must sum to 1.0
W_ALIGNMENT: float = 0.50
W_VAD: float = 0.30
W_SNR: float = 0.20

# SNR noise floor percentile
NOISE_FLOOR_PERCENTILE: int = 10


class ConfidenceScorer:
    """Compute composite confidence scores for aligned words.

    Parameters
    ----------
    config:
        Optional configuration dict (currently unused, reserved for future
        threshold overrides).

    Usage
    -----
    >>> scorer = ConfidenceScorer()
    >>> scored_words = scorer.score_words(words, features)
    >>> low_conf = scorer.get_low_confidence_words(scored_words, threshold=0.5)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def score_words(
        self,
        words: List[Dict[str, Any]],
        features: AudioFeatures,
    ) -> List[Dict[str, Any]]:
        """Compute and attach composite confidence scores to each word.

        This method modifies copies of the word dicts (does not mutate the
        originals) and returns the updated list.

        Parameters
        ----------
        words:
            List of word dicts with ``text``, ``start``, ``end``, ``score``.
        features:
            Acoustic features from :class:`~croonify.audio.features.FeatureExtractor`.

        Returns
        -------
        list[dict]
            Updated word dicts with:
            - ``score``: replaced by composite score
            - ``confidence``: dict with sub-component scores
              ``{alignment, vad_coverage, snr_estimate, composite}``
        """
        if not words:
            return words

        # Pre-compute global noise floor (10th-percentile RMS)
        noise_floor = float(np.percentile(features.rms_energy, NOISE_FLOOR_PERCENTILE)) + 1e-8

        result: List[Dict[str, Any]] = []
        for word in words:
            w = dict(word)
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start + 0.1))
            alignment_score = float(w.get("score", 0.5))

            # VAD coverage
            vad_cov = features.vad_coverage(start, end)

            # SNR estimate
            word_rms = features.rms_in_window(start, end)
            if len(word_rms) > 0:
                mean_rms = float(np.mean(word_rms))
                snr_raw = mean_rms / noise_floor
                snr_norm = min(1.0, snr_raw / 10.0)  # normalize: SNR=10 → 1.0
            else:
                snr_norm = 0.0

            # Composite
            composite = (
                W_ALIGNMENT * alignment_score
                + W_VAD * vad_cov
                + W_SNR * snr_norm
            )
            composite = float(np.clip(composite, 0.0, 1.0))

            w["score"] = round(composite, 4)
            w["confidence"] = {
                "alignment": round(alignment_score, 4),
                "vad_coverage": round(vad_cov, 4),
                "snr_estimate": round(snr_norm, 4),
                "composite": round(composite, 4),
            }
            result.append(w)

        logger.debug(
            "ConfidenceScorer: scored %d words, mean composite=%.3f",
            len(result),
            float(np.mean([w["score"] for w in result])) if result else 0.0,
        )
        return result

    def get_low_confidence_words(
        self,
        words: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Return words whose composite confidence is below *threshold*.

        Parameters
        ----------
        words:
            List of scored word dicts (output of :meth:`score_words`).
        threshold:
            Confidence threshold (0–1).  Words with ``score < threshold``
            are considered low confidence.

        Returns
        -------
        list[dict]
            Filtered subset of *words* with ``score < threshold``.
        """
        low = [w for w in words if float(w.get("score", 1.0)) < threshold]
        if low:
            logger.info(
                "%d / %d words below confidence threshold %.2f",
                len(low),
                len(words),
                threshold,
            )
        return low
