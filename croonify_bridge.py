"""
croonify_bridge.py — Adapter between LyricForge and the Croonify alignment engine.

Wraps SyncPipeline so the existing aligners/__init__.py interface is preserved:
    align(audio_path, lyrics_path, output_json, language) -> str (json path)
    parse_alignment_json(json_path) -> list[dict]

The returned segment dicts match the LyricForge schema:
    {start, end, text, words: [{start, end, text, score}], y_offset, index}
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure croonify package is importable ─────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
_SRC  = _HERE / "src"
for _p in [str(_HERE), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

# ── Default configuration ──────────────────────────────────────────────────────
_DEFAULT_CONFIG: Dict[str, Any] = {
    "alignment": {
        "primary":  "whisperx",
        "fallback": "viterbi",
        "model":    "small",
        "language": "auto",
        "device":   "cpu",
    },
    "vocal_separation": {
        "enabled":              False,   # done upstream by stem_separator.py
        "model":                "htdemucs",
        "fallback_to_original": True,
    },
    "prosody": {
        "vowel_stretch_threshold": 0.7,
        "min_silence_ms":          80,
        "boundary_snap_ms":        20,
        "rms_extend_ratio":        0.15,
    },
    "line_segmentation": {
        "max_words":      8,
        "max_duration_s": 4.0,
        "min_gap_s":      0.25,
        "beat_snap":      False,
    },
}


class CroonifyAligner:
    """
    Thin wrapper around SyncPipeline that speaks the LyricForge alignment protocol.

    Parameters
    ----------
    model_size : Whisper model size — 'tiny', 'base', 'small', 'medium'.
    device     : Torch device — 'cpu' or 'cuda'.
    aligner    : Primary aligner — 'whisperx' or 'viterbi'.
    """

    def __init__(
        self,
        model_size: str = "small",
        device:     str = "cpu",
        aligner:    str = "whisperx",
    ) -> None:
        self.model_size    = model_size
        self.device        = device
        self.aligner       = aligner
        self._pipeline     = None
        self._pipeline_key = None

    def _get_pipeline(self, model_size: str = None, aligner: str = None):
        ms  = model_size or self.model_size
        al  = aligner    or self.aligner
        key = (ms, al, self.device)
        if self._pipeline is None or self._pipeline_key != key:
            from croonify.pipeline import SyncPipeline
            cfg = {
                **_DEFAULT_CONFIG,
                "alignment": {**_DEFAULT_CONFIG["alignment"], "primary": al, "model": ms, "device": self.device},
            }
            self._pipeline     = SyncPipeline(config=cfg)
            self._pipeline_key = key
            logger.info("CroonifyAligner: pipeline ready (model=%s aligner=%s device=%s)", ms, al, self.device)
        return self._pipeline

    def align(
        self,
        audio_path:  str,
        lyrics_path: str,
        output_json: Optional[str] = None,
        language:    Optional[str] = None,
        model_size:  Optional[str] = None,
        aligner:     Optional[str] = None,
    ) -> str:
        lyrics_text = Path(lyrics_path).read_text(encoding="utf-8", errors="replace")
        if not lyrics_text.strip():
            raise ValueError(f"Lyrics file is empty: {lyrics_path}")

        if output_json is None:
            output_json = str(Path(audio_path).parent / "_croonify_alignment.json")

        pipeline = self._get_pipeline(model_size=model_size, aligner=aligner)
        result   = pipeline.align(
            audio_path           = str(audio_path),
            lyrics_text          = lyrics_text,
            language             = language or "auto",
            use_vocal_separation = False,
            aligner              = aligner or self.aligner,
        )

        payload = self._to_lyricforge(result)
        Path(output_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("CroonifyAligner: saved %d segments -> %s", len(payload["segments"]), output_json)
        return output_json

    @staticmethod
    def _to_lyricforge(result) -> Dict[str, Any]:
        segments: List[Dict[str, Any]] = []
        for i, line in enumerate(result.lines):
            words = [
                {
                    "start": round(float(w.get("start", 0.0)), 4),
                    "end":   round(float(w.get("end",   0.0)), 4),
                    "text":  str(w.get("text", "")),
                    "score": round(float(w.get("score", 1.0)), 4),
                }
                for w in line.get("words", [])
            ]
            segments.append({
                "index":    i,
                "start":    round(float(line.get("start", 0.0)), 4),
                "end":      round(float(line.get("end",   0.0)), 4),
                "text":     str(line.get("text", "")),
                "y_offset": 0,
                "words":    words,
            })
        return {
            "segments": segments,
            "metadata": result.metadata if hasattr(result, "metadata") else {},
        }


# ── Module helpers ─────────────────────────────────────────────────────────────

_default_aligner: Optional[CroonifyAligner] = None


def get_aligner(model_size: str = "small", device: str = "cpu", aligner: str = "whisperx") -> CroonifyAligner:
    global _default_aligner
    if (
        _default_aligner is None
        or _default_aligner.model_size != model_size
        or _default_aligner.device     != device
        or _default_aligner.aligner    != aligner
    ):
        _default_aligner = CroonifyAligner(model_size=model_size, device=device, aligner=aligner)
    return _default_aligner


def parse_alignment_json(json_path: str) -> List[Dict[str, Any]]:
    """Read a LyricForge alignment JSON and return the segment list."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "result" in data:
        return data["result"]
    return []
