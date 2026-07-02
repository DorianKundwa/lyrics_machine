"""Croonify end-to-end synchronization pipeline.

This module is the main entry point for programmatic use of Croonify.
It orchestrates all processing stages in order:

1. (Optional) Vocal separation — Demucs
2. Acoustic feature extraction — librosa
3. Lyrics normalization — :class:`~croonify.text.normalizer.LyricsNormalizer`
4. Forced alignment — WhisperX (with Viterbi fallback)
5. Prosody refinement — :class:`~croonify.refinement.prosody.ProsodyRefiner`
6. Confidence scoring — :class:`~croonify.scoring.confidence.ConfidenceScorer`
7. Line segmentation — :class:`~croonify.segmentation.lines.LineSegmenter`

Each step is timed and logged at INFO level.  Any step may raise an exception
if its dependencies are missing; the pipeline handles these gracefully by
falling back to simpler alternatives where possible.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration (mirrors config/default.yaml)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "alignment": {
        "primary": "whisperx",
        "fallback": "viterbi",
        "model": "small",
        "language": "auto",
        "device": "cpu",
    },
    "vocal_separation": {
        "enabled": True,
        "model": "htdemucs",
        "fallback_to_original": True,
    },
    "prosody": {
        "vowel_stretch_threshold": 0.7,
        "min_silence_ms": 80,
        "boundary_snap_ms": 20,
        "rms_extend_ratio": 0.15,
    },
    "line_segmentation": {
        "max_words": 8,
        "max_duration_s": 4.0,
        "min_gap_s": 0.25,
        "beat_snap": False,
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "max_file_size_mb": 50,
        "job_ttl_s": 3600,
    },
}


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Container for the complete alignment output.

    Attributes
    ----------
    words:
        Flat list of word dicts with ``text``, ``start``, ``end``, ``score``,
        ``confidence``, and optional ``emphasized`` keys.
    lines:
        List of line dicts grouping words into display units.  Each dict has
        ``start``, ``end``, ``text``, and ``words``.
    metadata:
        Processing metadata including model name, language, audio duration,
        word count, low-confidence count, and per-step timing.
    """

    words: List[Dict[str, Any]]
    lines: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation (JSON-serializable)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# SyncPipeline
# ---------------------------------------------------------------------------

class SyncPipeline:
    """Orchestrates all Croonify alignment stages.

    Parameters
    ----------
    config_path:
        Optional path to a YAML configuration file.  Keys are merged over
        the built-in defaults.
    config:
        Optional dict to use directly (takes precedence over *config_path*).

    Usage
    -----
    >>> pipeline = SyncPipeline()
    >>> result = pipeline.align("song.wav", lyrics_text)
    >>> print(result.to_json())
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = self._merge_with_defaults(config)
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # Try the default config file location
            default_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
            if default_path.exists():
                self.config = self._load_config(str(default_path))
            else:
                self.config = self._get_default_config()

        logger.debug("Pipeline config: %s", self.config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def align(
        self,
        audio_path: str,
        lyrics_text: str,
        language: str = "auto",
        use_vocal_separation: Optional[bool] = None,
        aligner: Optional[str] = None,
    ) -> SyncResult:
        """Run the full alignment pipeline on *audio_path* with *lyrics_text*.

        Parameters
        ----------
        audio_path:
            Path to the audio file (WAV, MP3, FLAC, etc.).
        lyrics_text:
            Raw lyrics string (multi-line).
        language:
            ISO-639-1 language code or ``"auto"`` for auto-detection.
        use_vocal_separation:
            Override the config flag for vocal separation.
            ``None`` means use config value.
        aligner:
            Override the aligner choice (``"whisperx"`` or ``"viterbi"``).
            ``None`` means use config value.

        Returns
        -------
        SyncResult
        """
        pipeline_start = time.perf_counter()
        timing: Dict[str, float] = {}

        audio_path = str(Path(audio_path).resolve())
        logger.info("=== Croonify pipeline start: %s ===", audio_path)

        cfg = self.config
        align_cfg = cfg["alignment"]
        sep_cfg = cfg["vocal_separation"]

        # Resolve overrides
        do_separation = use_vocal_separation if use_vocal_separation is not None else sep_cfg["enabled"]
        primary_aligner = aligner or align_cfg.get("primary", "whisperx")
        fallback_aligner = align_cfg.get("fallback", "viterbi")
        model_size = align_cfg.get("model", "small")
        device = align_cfg.get("device", "cpu")

        # ------------------------------------------------------------------
        # Step 1: Vocal separation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        working_audio = audio_path
        if do_separation:
            try:
                from croonify.audio.separator import VocalSeparator
                separator = VocalSeparator(
                    model=sep_cfg.get("model", "htdemucs"),
                    device=device,
                    fallback_to_original=sep_cfg.get("fallback_to_original", True),
                )
                working_audio = separator.separate(audio_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Vocal separation failed: %s — using original audio.", exc)
                working_audio = audio_path
        timing["vocal_separation_s"] = time.perf_counter() - t0
        logger.info("Step 1 done (%.2f s): working audio = %s", timing["vocal_separation_s"], working_audio)

        # ------------------------------------------------------------------
        # Step 2: Feature extraction
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        from croonify.audio.features import FeatureExtractor
        extractor = FeatureExtractor()
        features = extractor.extract(working_audio)
        timing["feature_extraction_s"] = time.perf_counter() - t0
        logger.info("Step 2 done (%.2f s): duration=%.2f s", timing["feature_extraction_s"], features.duration)

        # ------------------------------------------------------------------
        # Step 3: Text normalization + line structure
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        from croonify.text.normalizer import LyricsNormalizer
        normalizer = LyricsNormalizer()
        line_structure = normalizer.get_line_structure(lyrics_text)
        timing["normalization_s"] = time.perf_counter() - t0
        total_words_normalized = sum(len(line) for line in line_structure)
        logger.info("Step 3 done (%.2f s): %d lines, %d words", timing["normalization_s"], len(line_structure), total_words_normalized)

        # ------------------------------------------------------------------
        # Step 4: Alignment (primary → fallback)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        words: List[Dict[str, Any]] = []
        detected_language = language
        aligner_used = primary_aligner

        try:
            words, detected_language = self._run_aligner(
                name=primary_aligner,
                audio_path=working_audio,
                lyrics_text=lyrics_text,
                language=language,
                model_size=model_size,
                device=device,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Primary aligner '%s' failed: %s — trying fallback '%s'.",
                primary_aligner, exc, fallback_aligner,
            )
            try:
                words, detected_language = self._run_aligner(
                    name=fallback_aligner,
                    audio_path=working_audio,
                    lyrics_text=lyrics_text,
                    language=language,
                    model_size=model_size,
                    device=device,
                )
                aligner_used = fallback_aligner
            except Exception as exc2:  # pylint: disable=broad-except
                logger.error("Fallback aligner also failed: %s", exc2, exc_info=True)
                raise RuntimeError(
                    f"Both primary ({primary_aligner}) and fallback ({fallback_aligner}) aligners failed."
                ) from exc2

        # Attach line-break metadata from lyric structure
        words = self._attach_line_breaks(words, line_structure)

        timing["alignment_s"] = time.perf_counter() - t0
        logger.info("Step 4 done (%.2f s): %d word timestamps, aligner=%s", timing["alignment_s"], len(words), aligner_used)

        # ------------------------------------------------------------------
        # Step 5: Prosody refinement
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        from croonify.refinement.prosody import ProsodyRefiner
        refiner = ProsodyRefiner(config=cfg.get("prosody", {}))
        words = refiner.refine(words, features)
        timing["prosody_refinement_s"] = time.perf_counter() - t0
        logger.info("Step 5 done (%.2f s): prosody refinement applied", timing["prosody_refinement_s"])

        # ------------------------------------------------------------------
        # Step 6: Confidence scoring
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        from croonify.scoring.confidence import ConfidenceScorer
        scorer = ConfidenceScorer(config=cfg)
        words = scorer.score_words(words, features)
        low_conf_words = scorer.get_low_confidence_words(words, threshold=0.5)
        timing["confidence_scoring_s"] = time.perf_counter() - t0
        logger.info(
            "Step 6 done (%.2f s): %d low-confidence words",
            timing["confidence_scoring_s"],
            len(low_conf_words),
        )

        # ------------------------------------------------------------------
        # Step 7: Line segmentation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        from croonify.segmentation.lines import LineSegmenter
        segmenter = LineSegmenter(config=cfg.get("line_segmentation", {}))
        lines = segmenter.segment(words, beat_times=features.beat_times)
        timing["line_segmentation_s"] = time.perf_counter() - t0
        logger.info("Step 7 done (%.2f s): %d lines", timing["line_segmentation_s"], len(lines))

        # ------------------------------------------------------------------
        # Step 8: Build SyncResult
        # ------------------------------------------------------------------
        total_time = time.perf_counter() - pipeline_start
        timing["total_s"] = total_time
        metadata: Dict[str, Any] = {
            "aligner_used": aligner_used,
            "model_size": model_size,
            "device": device,
            "language_detected": detected_language,
            "audio_path": audio_path,
            "audio_duration_s": round(features.duration, 3),
            "word_count": len(words),
            "line_count": len(lines),
            "low_confidence_count": len(low_conf_words),
            "vocal_separation": do_separation,
            "processing_time_s": round(total_time, 3),
            "timing": {k: round(v, 4) for k, v in timing.items()},
        }

        logger.info("=== Pipeline complete in %.2f s ===", total_time)
        return SyncResult(words=words, lines=lines, metadata=metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_aligner(
        self,
        name: str,
        audio_path: str,
        lyrics_text: str,
        language: str,
        model_size: str,
        device: str,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Instantiate the named aligner and run it.

        Returns (word_dicts, detected_language).
        """
        name_lower = name.lower()

        if name_lower == "whisperx":
            from croonify.alignment.whisperx_aligner import WhisperXAligner
            aligner_obj = WhisperXAligner(
                model_size=model_size,
                device=device,
                language=None if language in ("auto", "") else language,
            )
            words = aligner_obj.align(audio_path, lyrics_text, language=language)
            detected = language  # WhisperX may refine this internally

        elif name_lower == "viterbi":
            from croonify.alignment.viterbi_aligner import ViterbiAligner
            aligner_obj = ViterbiAligner(model_size=model_size, device=device)
            words = aligner_obj.align(audio_path, lyrics_text, language=language)
            detected = language

        else:
            raise ValueError(f"Unknown aligner: '{name}'. Choose 'whisperx' or 'viterbi'.")

        return words, detected

    def _attach_line_breaks(
        self,
        words: List[Dict[str, Any]],
        line_structure: List[List[str]],
    ) -> List[Dict[str, Any]]:
        """Attach ``line_break_after`` flags to words based on lyric structure.

        The lyric structure (list of lines, each a list of words) encodes where
        the original line breaks were.  We map these onto the aligned word list
        positionally and tag the last word of each lyric line with
        ``line_break_after = True``.

        If the word counts don't match (e.g. due to contraction expansion), we
        use a best-effort positional mapping.
        """
        if not words or not line_structure:
            return words

        # Flat list of lyric words from structure
        flat_lyric_words = [w for line in line_structure for w in line]
        # Positions of line-ending words in the flat list
        line_end_positions: set[int] = set()
        cursor = 0
        for line in line_structure:
            cursor += len(line)
            line_end_positions.add(cursor - 1)

        # Map positionally (both lists may differ in length due to expansions)
        n = min(len(words), len(flat_lyric_words))
        for i in range(n):
            if i in line_end_positions:
                words[i]["line_break_after"] = True

        # Tag the very last aligned word if last lyric word is a line end
        if words and (len(flat_lyric_words) - 1) in line_end_positions:
            words[-1]["line_break_after"] = True

        return words

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load a YAML config file and merge with built-in defaults."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        return SyncPipeline._merge_with_defaults(user_config)

    @staticmethod
    def _merge_with_defaults(user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge *user_config* over the built-in defaults."""
        import copy
        result = copy.deepcopy(_DEFAULT_CONFIG)

        def _deep_merge(base: Dict, override: Dict) -> Dict:
            for key, val in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                    _deep_merge(base[key], val)
                else:
                    base[key] = val
            return base

        return _deep_merge(result, user_config)

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Return the built-in default configuration dict."""
        import copy
        return copy.deepcopy(_DEFAULT_CONFIG)
