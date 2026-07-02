"""WhisperX-based forced aligner for lyrics synchronization.

Alignment strategy
------------------
WhisperX provides two capabilities we exploit:

1. **Language detection** — running the Whisper transcription gives us the
   detected language code (``en``, ``es``, ``fr``, etc.) which is required
   by the ``whisperx.load_align_model`` call.

2. **Phone-level forced alignment** — ``whisperx.align()`` aligns a list of
   *word segments* against the audio using wav2vec2 / MMS models.  Because we
   want to align *our* lyrics (not Whisper's transcription), we replace the
   transcription output with a re-segmented version of the provided lyrics
   before calling ``whisperx.align()``.

Segment mapping heuristic
--------------------------
Whisper's transcription gives approximate sentence-level timing windows
(``[segment.start, segment.end]``).  We map the provided lyrics words onto
these windows proportionally by word count (``_map_lyrics_to_segments``).
This allows the wav2vec2 forced aligner to produce fine-grained per-word
timestamps within known temporal regions rather than across the entire track.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import
# ---------------------------------------------------------------------------
try:
    import whisperx  # type: ignore
    _WHISPERX_AVAILABLE = True
except ImportError:
    _WHISPERX_AVAILABLE = False
    whisperx = None  # type: ignore


class WhisperXAligner:
    """Force-align user-supplied lyrics against audio using WhisperX.

    Parameters
    ----------
    model_size:
        Whisper model size (``tiny``, ``base``, ``small``, ``medium``,
        ``large-v2``).  ``small`` is the recommended default for speed vs
        accuracy trade-off.
    device:
        PyTorch compute device (``cpu``, ``cuda``, ``mps``).
    language:
        ISO-639-1 language code to skip detection (e.g. ``"en"``).
        Pass ``None`` or ``"auto"`` to run language detection automatically.

    Raises
    ------
    ImportError
        If ``whisperx`` is not installed when :meth:`align` is called.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        language: Optional[str] = None,
    ) -> None:
        if not _WHISPERX_AVAILABLE:
            logger.warning(
                "whisperx is not installed.  WhisperXAligner will raise ImportError "
                "when align() is called.  Install with: pip install whisperx"
            )
        self.model_size = model_size
        self.device = device
        self.language = None if language in (None, "auto", "") else language
        self._whisper_model: Optional[Any] = None
        self._align_models: Dict[str, Any] = {}  # cache per language code

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def align(
        self,
        audio_path: str,
        lyrics_text: str,
        language: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Force-align *lyrics_text* against *audio_path*.

        Steps
        -----
        1. Load Whisper model (lazy, cached).
        2. Transcribe audio with WhisperX (FP32 on CPU, FP16 on CUDA).
           This provides language detection and rough segment timing.
        3. Normalize lyrics words from *lyrics_text*.
        4. Map normalized lyrics onto Whisper segment windows using
           :meth:`_map_lyrics_to_segments`.
        5. Load the wav2vec2/MMS alignment model for the detected language.
        6. Call ``whisperx.align()`` with the re-mapped segments.
        7. Flatten word-level results and return.

        Parameters
        ----------
        audio_path:
            Path to the audio file.
        lyrics_text:
            Raw lyrics string (multi-line).
        language:
            Override language code (ISO-639-1).  ``"auto"`` triggers
            auto-detection from the Whisper transcription.

        Returns
        -------
        list[dict]
            One dict per word::

                {
                    "text":  str,   # normalized word
                    "start": float, # onset in seconds
                    "end":   float, # offset in seconds
                    "score": float, # alignment confidence 0–1
                }
        """
        if not _WHISPERX_AVAILABLE:
            raise ImportError(
                "whisperx is required for WhisperXAligner.  "
                "Install it with:\n    pip install whisperx\n"
                "or use --aligner viterbi to fall back to the built-in aligner."
            )

        if not lyrics_text.strip():
            logger.warning("Empty lyrics_text passed to WhisperXAligner.align().")
            return []

        # --- Step 1: Load Whisper model ----------------------------------------
        whisper_model = self._get_whisper_model()

        # --- Step 2: Transcribe (for language detection + segment windows) ------
        logger.info("Running WhisperX transcription on %s", audio_path)
        compute_type = "float32" if self.device == "cpu" else "float16"
        transcription = whisperx.transcribe(
            whisper_model,
            audio_path,
            batch_size=16,
            compute_type=compute_type,
        )

        # --- Step 3: Resolve language -------------------------------------------
        effective_lang = self._resolve_language(language, transcription)
        logger.info("Effective language: %s", effective_lang)

        # --- Step 4: Map lyrics onto segment windows ----------------------------
        from croonify.text.normalizer import LyricsNormalizer  # avoid circular imports

        norm = LyricsNormalizer()
        lyrics_words = norm.flat_words(lyrics_text)

        if not lyrics_words:
            logger.warning("Normalized lyrics are empty.")
            return []

        mapped_segments = self._map_lyrics_to_segments(
            lyrics_words=lyrics_words,
            transcribed_segments=transcription.get("segments", []),
            audio_duration=self._get_audio_duration(audio_path),
        )

        # --- Step 5: Load alignment model ---------------------------------------
        align_model, align_metadata = self._get_align_model(effective_lang)

        # --- Step 6: Force-align -----------------------------------------------
        logger.info(
            "Running force-alignment: %d lyrics words across %d segments",
            len(lyrics_words),
            len(mapped_segments),
        )
        result = whisperx.align(
            mapped_segments,
            align_model,
            align_metadata,
            audio_path,
            self.device,
            return_char_alignments=False,
        )

        # --- Step 7: Flatten word-level results --------------------------------
        word_dicts = self._flatten_word_results(result)
        logger.info("WhisperX alignment produced %d word timestamps", len(word_dicts))
        return word_dicts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_whisper_model(self) -> Any:
        """Lazily load and cache the Whisper model."""
        if self._whisper_model is None:
            logger.info("Loading Whisper model: %s on %s", self.model_size, self.device)
            self._whisper_model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type="float32" if self.device == "cpu" else "float16",
            )
        return self._whisper_model

    def _get_align_model(self, language_code: str):
        """Lazily load and cache the wav2vec2/MMS alignment model."""
        if language_code not in self._align_models:
            logger.info("Loading alignment model for language: %s", language_code)
            model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
            )
            self._align_models[language_code] = (model, metadata)
        return self._align_models[language_code]

    def _resolve_language(self, requested: str, transcription: Dict[str, Any]) -> str:
        """Return the effective language code to use for alignment."""
        if self.language and self.language not in ("auto", ""):
            return self.language
        if requested and requested not in ("auto", ""):
            return requested
        # Try to extract from whisperx transcription result
        lang = transcription.get("language", "en")
        if not lang:
            lang = "en"
        return lang

    def _get_audio_duration(self, audio_path: str) -> float:
        """Return audio duration in seconds using librosa (fast header read)."""
        try:
            import librosa
            return float(librosa.get_duration(path=audio_path))
        except Exception:  # pylint: disable=broad-except
            return 0.0

    def _map_lyrics_to_segments(
        self,
        lyrics_words: List[str],
        transcribed_segments: List[Dict[str, Any]],
        audio_duration: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Map provided lyrics words onto Whisper-detected segment time windows.

        WhisperX's alignment model requires input in the form of ``{segments}``
        where each segment has a ``text`` field and ``start``/``end`` times.
        Because we want to align *our* lyrics (not Whisper's verbatim
        transcription), we:

        1. Compute the total word count in Whisper's segments.
        2. Distribute lyrics words into segments proportionally by word count.
        3. Assign Whisper segment start/end times to the corresponding lyrics
           sub-segments.

        If no Whisper segments are available (e.g. the audio is silent or very
        short) we fall back to a single segment spanning the full audio.

        Parameters
        ----------
        lyrics_words:
            Flat list of normalized lyrics words.
        transcribed_segments:
            Output from ``whisperx.transcribe()['segments']``.
        audio_duration:
            Total audio length in seconds (used for fallback).

        Returns
        -------
        list[dict]
            Segments with ``text``, ``start``, ``end``, ``words`` keys
            compatible with ``whisperx.align()``.
        """
        if not transcribed_segments:
            # Fallback: single segment spanning the full audio
            end = audio_duration if audio_duration > 0.0 else 60.0
            return [
                {
                    "start": 0.0,
                    "end": end,
                    "text": " ".join(lyrics_words),
                    "words": [
                        {"word": w, "start": 0.0, "end": end} for w in lyrics_words
                    ],
                }
            ]

        # Count words per Whisper segment
        seg_word_counts = []
        for seg in transcribed_segments:
            seg_text = seg.get("text", "")
            # rough word count
            count = max(1, len(seg_text.split()))
            seg_word_counts.append(count)

        total_whisper_words = sum(seg_word_counts)
        total_lyrics_words = len(lyrics_words)

        # Distribute lyrics words proportionally
        mapped_segments: List[Dict[str, Any]] = []
        lyrics_idx = 0

        for i, (seg, wcount) in enumerate(zip(transcribed_segments, seg_word_counts)):
            # Proportion of lyrics words for this segment
            proportion = wcount / total_whisper_words
            n_words = int(round(proportion * total_lyrics_words))
            n_words = max(1, n_words)

            # Last segment gets all remaining words
            if i == len(transcribed_segments) - 1:
                n_words = total_lyrics_words - lyrics_idx

            seg_words = lyrics_words[lyrics_idx: lyrics_idx + n_words]
            lyrics_idx += n_words

            if not seg_words:
                continue

            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start + 2.0))

            # Build word-level entries spread linearly within the window
            # (WhisperX will refine these with the phone model)
            word_duration = (seg_end - seg_start) / len(seg_words)
            words_for_seg = []
            for j, w in enumerate(seg_words):
                ws = seg_start + j * word_duration
                we = ws + word_duration
                words_for_seg.append({"word": w, "start": ws, "end": we})

            mapped_segments.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "text": " ".join(seg_words),
                    "words": words_for_seg,
                }
            )

        return mapped_segments

    def _flatten_word_results(
        self, align_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract and flatten word-level dicts from WhisperX align output.

        WhisperX returns ``{'segments': [...], 'word_segments': [...]}`` where
        ``word_segments`` contains per-word timing info.  We prefer the flat
        ``word_segments`` list when available, otherwise we iterate segments.

        Returns
        -------
        list[dict]
            List of ``{text, start, end, score}`` dicts.
        """
        words: List[Dict[str, Any]] = []

        # Prefer the flat word_segments key (WhisperX >= 3.x)
        word_segments = align_result.get("word_segments", [])
        if word_segments:
            for ws in word_segments:
                words.append(
                    {
                        "text": ws.get("word", ""),
                        "start": float(ws.get("start", 0.0)),
                        "end": float(ws.get("end", 0.0)),
                        "score": float(ws.get("score", 0.5)),
                    }
                )
            return words

        # Fallback: iterate segments → words
        for seg in align_result.get("segments", []):
            for w in seg.get("words", []):
                words.append(
                    {
                        "text": w.get("word", ""),
                        "start": float(w.get("start", seg.get("start", 0.0))),
                        "end": float(w.get("end", seg.get("end", 0.0))),
                        "score": float(w.get("score", 0.5)),
                    }
                )

        return words
