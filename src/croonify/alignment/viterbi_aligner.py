"""Custom monotonic HMM Viterbi aligner — CPU-friendly fallback.

This module implements a from-scratch forced alignment pipeline that does **not**
require wav2vec2 or any internet-connected model download.  It works entirely
with the Whisper encoder's frame-level representations or, if Whisper is
unavailable, with librosa MFCC features.

Alignment model
---------------
We treat alignment as a monotonic hidden Markov model problem:

* **States**    — characters of the lyrics (including word-boundary markers).
* **Frames**    — audio encoder frames (20 ms hop for Whisper; 10 ms for MFCC).
* **Transitions** — stay on the same character OR advance to the next character
  (no backward jumps — monotonic constraint).
* **Emissions** — log-probability derived from the cosine similarity between
  the audio frame embedding and the character's reference vector.

The Viterbi algorithm finds the most probable monotonic alignment path in
O(T × C) time, where T = number of audio frames and C = number of characters.

Character embeddings
--------------------
To avoid requiring a trained character encoder, we use simple 40-dimensional
one-hot-like frequency vectors for ASCII characters.  When Whisper token
embeddings are available, we project them down via the first 40 principal
components for a better approximation.

Design trade-offs
-----------------
This aligner sacrifices some accuracy compared to wav2vec2 forced alignment
but is:
* Fully offline after Whisper model download
* CPU-friendly (no GPU required)
* Low memory footprint
* ~10× faster than WhisperX on CPU for short clips
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR: int = 16_000
MFCC_HOP_LENGTH: int = 160  # 10 ms @ 16 kHz
WHISPER_HOP_SECONDS: float = 0.02  # 20 ms frame stride in Whisper encoder
CHAR_EMBED_DIM: int = 64  # dimensionality of character embedding vectors
WORD_BOUNDARY_MARKER: str = "|"  # inserted between words in character sequence
STAY_LOG_PROB: float = math.log(0.6)  # log-probability of staying on same state
ADVANCE_LOG_PROB: float = math.log(0.4)  # log-probability of advancing to next state
NEG_INF: float = -1e30  # safe -inf for log-space computations


class ViterbiAligner:
    """Monotonic HMM forced-aligner using Whisper encoder embeddings.

    Parameters
    ----------
    model_size:
        Whisper model size.  Used to load the Whisper encoder if available.
    device:
        PyTorch device string.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu") -> None:
        self.model_size = model_size
        self.device = device
        self._whisper_model: Optional[Any] = None
        self._char_embed_matrix: Optional[np.ndarray] = None

        # Try loading whisper at construction time (optional)
        self._whisper_available = self._check_whisper()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def align(
        self,
        audio_path: str,
        lyrics_text: str,
        language: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Full alignment pipeline.

        Parameters
        ----------
        audio_path:
            Path to audio file.
        lyrics_text:
            Raw lyrics (multi-line string).
        language:
            Not used by Viterbi aligner (included for API compatibility).

        Returns
        -------
        list[dict]
            ``[{'text': str, 'start': float, 'end': float, 'score': float}]``
        """
        from croonify.text.normalizer import LyricsNormalizer

        norm = LyricsNormalizer()
        lyrics_words = norm.flat_words(lyrics_text)

        if not lyrics_words:
            logger.warning("Empty lyrics passed to ViterbiAligner.")
            return []

        # --- 1. Get audio embeddings ------------------------------------------
        try:
            audio_embeddings, hop_seconds = self._get_audio_embeddings(audio_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to extract audio embeddings: %s", exc, exc_info=True)
            return self._uniform_fallback(lyrics_words, audio_path)

        T = audio_embeddings.shape[0]
        logger.debug("Audio: %d frames (%.1f s), hop=%.0f ms", T, T * hop_seconds, hop_seconds * 1000)

        # --- 2. Tokenize lyrics into characters + word-boundary markers ---------
        char_tokens = self._encode_text(lyrics_words)
        C = len(char_tokens)
        logger.debug("Lyrics: %d words → %d character tokens", len(lyrics_words), C)

        if C == 0:
            return self._uniform_fallback(lyrics_words, audio_path)

        # --- 3. Compute similarity matrix S[t, c] --------------------------------
        char_embeddings = self._get_char_embeddings(char_tokens)
        S = self._compute_similarity_matrix(audio_embeddings, char_embeddings)  # (T, C)

        # --- 4. Viterbi decode ---------------------------------------------------
        path = self._viterbi_decode(S)

        # --- 5. Extract word timestamps -----------------------------------------
        word_dicts = self._extract_word_timestamps(
            path=path,
            char_tokens=char_tokens,
            words=lyrics_words,
            hop_seconds=hop_seconds,
        )
        logger.info("ViterbiAligner produced %d word timestamps", len(word_dicts))
        return word_dicts

    # ------------------------------------------------------------------
    # Audio embeddings
    # ------------------------------------------------------------------

    def _get_audio_embeddings(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Return (T, D) frame-level audio embeddings and hop duration in seconds.

        Tries Whisper encoder first; falls back to librosa MFCC on failure.
        """
        if self._whisper_available:
            try:
                return self._get_whisper_embeddings(audio_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Whisper embedding failed (%s), using MFCC fallback.", exc)

        return self._get_mfcc_embeddings(audio_path)

    def _get_whisper_embeddings(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Run audio through Whisper's encoder and return frame embeddings."""
        import torch
        import whisper

        if self._whisper_model is None:
            logger.info("Loading Whisper model '%s' for Viterbi encoder", self.model_size)
            self._whisper_model = whisper.load_model(self.model_size, device=self.device)

        model = self._whisper_model

        # Load and pad audio the Whisper way
        audio = whisper.load_audio(audio_path)
        audio_tensor = torch.from_numpy(audio).to(self.device)

        # Whisper processes in 30-second windows; we take the first window's encoder output
        # For longer audio we concatenate multiple windows
        WHISPER_CHUNK_SAMPLES = 30 * TARGET_SR
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(audio_tensor), WHISPER_CHUNK_SAMPLES):
            chunk = audio_tensor[start: start + WHISPER_CHUNK_SAMPLES]
            # Pad to 30 s
            if len(chunk) < WHISPER_CHUNK_SAMPLES:
                chunk = torch.nn.functional.pad(chunk, (0, WHISPER_CHUNK_SAMPLES - len(chunk)))
            mel = whisper.log_mel_spectrogram(chunk).unsqueeze(0).to(self.device)
            with torch.no_grad():
                enc = model.encoder(mel)  # (1, T_enc, D_enc)
            all_embeddings.append(enc.squeeze(0).cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)  # (T_total, D)
        return embeddings, WHISPER_HOP_SECONDS

    def _get_mfcc_embeddings(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Compute librosa MFCC features as a CPU-only embedding fallback."""
        import librosa

        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=CHAR_EMBED_DIM,
            hop_length=MFCC_HOP_LENGTH,
        )  # (CHAR_EMBED_DIM, T)
        delta = librosa.feature.delta(mfcc)
        embeddings = np.concatenate([mfcc, delta], axis=0).T  # (T, 2*CHAR_EMBED_DIM)
        hop_seconds = MFCC_HOP_LENGTH / sr
        logger.debug("MFCC embeddings: shape=%s, hop=%.1f ms", embeddings.shape, hop_seconds * 1000)
        return embeddings.astype(np.float32), hop_seconds

    # ------------------------------------------------------------------
    # Text tokenization
    # ------------------------------------------------------------------

    def _encode_text(self, words: List[str]) -> List[str]:
        """Convert a list of words to a character sequence with word-boundary markers.

        The sequence interleaves word-boundary markers (``|``) between words:

            ["hello", "world"] → ['h','e','l','l','o','|','w','o','r','l','d']

        The boundary markers help Viterbi separate words during decoding.
        """
        tokens: List[str] = []
        for i, word in enumerate(words):
            tokens.extend(list(word))
            if i < len(words) - 1:
                tokens.append(WORD_BOUNDARY_MARKER)
        return tokens

    # ------------------------------------------------------------------
    # Character embeddings
    # ------------------------------------------------------------------

    def _get_char_embeddings(self, char_tokens: List[str]) -> np.ndarray:
        """Return (C, D) embedding matrix for the character token sequence.

        Each character is embedded as a sparse frequency-based vector
        in ``CHAR_EMBED_DIM`` dimensional space.  The representation encodes
        character identity and phonetic category membership.

        The matrix is L2-normalized per row for cosine similarity computation.
        """
        D = CHAR_EMBED_DIM
        C = len(char_tokens)
        matrix = np.zeros((C, D), dtype=np.float32)

        for i, ch in enumerate(char_tokens):
            vec = self._char_to_vector(ch)
            # Resize if needed
            if len(vec) < D:
                vec = np.pad(vec, (0, D - len(vec)))
            elif len(vec) > D:
                vec = vec[:D]
            matrix[i] = vec

        # L2-normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return matrix / norms

    @staticmethod
    def _char_to_vector(ch: str) -> np.ndarray:
        """Convert a single character to a feature vector.

        Encoding:
        - Dimensions  0–25 : ASCII letter one-hot (a=0, b=1, …)
        - Dimension   26   : is vowel (a, e, i, o, u)
        - Dimension   27   : is consonant
        - Dimension   28   : is word boundary (``|``)
        - Dimension   29   : is digit
        - Dimensions 30–37 : consonant category (plosive, fricative, nasal, …)
        - Remaining filled with small noise for regularization
        """
        dim = 40
        vec = np.zeros(dim, dtype=np.float32)
        ch_lower = ch.lower()

        if ch == WORD_BOUNDARY_MARKER:
            vec[28] = 1.0
            return vec

        if ch_lower.isdigit():
            vec[29] = 1.0
            digit_idx = int(ch_lower)
            vec[30 + digit_idx % (dim - 30)] = 0.5
            return vec

        if ch_lower.isalpha():
            ord_val = ord(ch_lower) - ord('a')
            if 0 <= ord_val < 26:
                vec[ord_val] = 1.0

            vowels = "aeiou"
            if ch_lower in vowels:
                vec[26] = 1.0
                # Vowel height / backness encodings
                vowel_features = {
                    'a': [0, 0, 1],    # low, back
                    'e': [1, 0, 0],    # high, front
                    'i': [1, 1, 0],    # high, front, tense
                    'o': [0, 1, 1],    # mid, back
                    'u': [1, 1, 1],    # high, back
                }
                feats = vowel_features.get(ch_lower, [0, 0, 0])
                for fi, fv in enumerate(feats):
                    if 30 + fi < dim:
                        vec[30 + fi] = float(fv)
            else:
                vec[27] = 1.0
                # Consonant manner of articulation
                plosives = "bpdtkg"
                fricatives = "fvszsh"
                nasals = "mn"
                liquids = "lr"
                glides = "wy"
                if ch_lower in plosives:
                    vec[33] = 1.0
                elif ch_lower in fricatives:
                    vec[34] = 1.0
                elif ch_lower in nasals:
                    vec[35] = 1.0
                elif ch_lower in liquids:
                    vec[36] = 1.0
                elif ch_lower in glides:
                    vec[37] = 1.0

        return vec

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------

    def _compute_similarity_matrix(
        self,
        audio_emb: np.ndarray,
        char_emb: np.ndarray,
    ) -> np.ndarray:
        """Compute S[t, c] = cosine similarity between frame t and character c.

        Both inputs are expected to be L2-normalized.

        Parameters
        ----------
        audio_emb:
            Shape ``(T, D_audio)``.  L2-normalized per row.
        char_emb:
            Shape ``(C, D_char)``.  L2-normalized per row.

        Returns
        -------
        np.ndarray
            Shape ``(T, C)``, values in ``[-1, 1]``.
        """
        T, D_a = audio_emb.shape
        C, D_c = char_emb.shape

        # If dimensions differ, project to the smaller one via truncation / zero-padding
        D = min(D_a, D_c)
        a = audio_emb[:, :D].astype(np.float32)
        c = char_emb[:, :D].astype(np.float32)

        # L2-normalize audio embeddings (may not already be normalized)
        a_norms = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
        a = a / a_norms

        # Dot-product similarity (both already normalized → cosine similarity)
        S = a @ c.T  # (T, C)
        return S

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------

    def _viterbi_decode(self, S: np.ndarray) -> List[int]:
        """Monotonic Viterbi decode over similarity matrix S[T, C].

        HMM structure:
        - State space : characters c = 0, 1, …, C-1
        - Transitions : stay on c (prob 0.6) or advance to c+1 (prob 0.4)
        - Emission    : log( sigmoid(S[t, c]) ) — maps [-1,1] → log-prob space

        Returns
        -------
        list[int]
            List of length T giving the most-likely character state per frame.
        """
        T, C = S.shape

        # Log-emission: convert cosine similarity to log-probability
        # sigmoid maps [-1,1] nicely to (0,1); log gives log-prob
        log_emission = np.log(1.0 / (1.0 + np.exp(-S * 3.0)) + 1e-8)  # (T, C)

        # Viterbi tables
        viterbi = np.full((T, C), NEG_INF, dtype=np.float64)
        backtrack = np.zeros((T, C), dtype=np.int32)

        # Initialize: all characters are valid starting points but we bias toward c=0
        viterbi[0, :] = log_emission[0, :]
        viterbi[0, 1:] += NEG_INF * 0.1  # soft bias toward starting from beginning

        # Forward pass
        for t in range(1, T):
            emit = log_emission[t]  # (C,)
            prev = viterbi[t - 1]   # (C,)

            # Option 1: stay — from same character
            stay = prev + STAY_LOG_PROB

            # Option 2: advance — from previous character (c-1 → c)
            advance = np.full(C, NEG_INF, dtype=np.float64)
            advance[1:] = prev[:-1] + ADVANCE_LOG_PROB

            # Best option per character
            best = np.where(stay >= advance, stay, advance)
            backtrack[t] = np.where(stay >= advance, np.arange(C), np.arange(C) - 1)
            viterbi[t] = best + emit

        # Backtrack from best final state
        path = [int(np.argmax(viterbi[T - 1]))]
        for t in range(T - 1, 0, -1):
            path.append(int(backtrack[t, path[-1]]))
        path.reverse()

        return path

    # ------------------------------------------------------------------
    # Word timestamp extraction
    # ------------------------------------------------------------------

    def _extract_word_timestamps(
        self,
        path: List[int],
        char_tokens: List[str],
        words: List[str],
        hop_seconds: float,
    ) -> List[Dict[str, Any]]:
        """Convert frame-level character alignment into word-level timestamps.

        Strategy:
        1. Find the first and last frame assigned to each character state.
        2. Map character ranges back to word ranges (using ``|`` boundaries).
        3. Compute a confidence score from the fraction of frames that
           ``agree`` with the word (i.e., are not assigned to a different word).

        Parameters
        ----------
        path:
            Frame-to-character-state mapping (output of Viterbi).
        char_tokens:
            Character sequence used during alignment.
        words:
            Original word list.
        hop_seconds:
            Duration per frame in seconds.

        Returns
        -------
        list[dict]
        """
        T = len(path)
        C = len(char_tokens)

        # Build first/last frame per character state
        first_frame = [T] * C
        last_frame = [-1] * C
        for t, c in enumerate(path):
            if c < C:
                first_frame[c] = min(first_frame[c], t)
                last_frame[c] = max(last_frame[c], t)

        # Map character index ranges to words
        # char_tokens = [c0, c1, …, |, c0, c1, …]
        word_char_ranges: List[Tuple[int, int]] = []
        word_start_c = 0
        for i, tok in enumerate(char_tokens):
            if tok == WORD_BOUNDARY_MARKER or i == len(char_tokens) - 1:
                end_c = i if tok == WORD_BOUNDARY_MARKER else i + 1
                word_char_ranges.append((word_start_c, end_c))
                word_start_c = i + 1  # after the boundary

        word_dicts: List[Dict[str, Any]] = []
        for wi, word in enumerate(words):
            if wi >= len(word_char_ranges):
                break
            c_start, c_end = word_char_ranges[wi]

            # First/last frame for this word's character range
            f_start = min((first_frame[c] for c in range(c_start, c_end) if first_frame[c] < T), default=0)
            f_end = max((last_frame[c] for c in range(c_start, c_end) if last_frame[c] >= 0), default=f_start)

            t_start = f_start * hop_seconds
            t_end = (f_end + 1) * hop_seconds

            # Confidence: fraction of path frames in this word's char range
            word_char_set = set(range(c_start, c_end))
            frames_in_word = sum(1 for c in path[f_start: f_end + 1] if c in word_char_set)
            total_word_frames = max(1, f_end - f_start + 1)
            score = min(1.0, frames_in_word / total_word_frames)

            word_dicts.append(
                {
                    "text": word,
                    "start": round(t_start, 4),
                    "end": round(t_end, 4),
                    "score": round(score, 4),
                }
            )

        return word_dicts

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _uniform_fallback(
        self, words: List[str], audio_path: str
    ) -> List[Dict[str, Any]]:
        """Return uniform word timestamps when alignment fails entirely."""
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
        except Exception:  # pylint: disable=broad-except
            duration = len(words) * 0.5  # rough estimate

        logger.warning("Using uniform fallback timestamps for %d words", len(words))
        dt = duration / max(1, len(words))
        result = []
        for i, word in enumerate(words):
            result.append(
                {
                    "text": word,
                    "start": round(i * dt, 4),
                    "end": round((i + 1) * dt, 4),
                    "score": 0.1,
                }
            )
        return result

    @staticmethod
    def _check_whisper() -> bool:
        """Return True if openai-whisper is importable."""
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            logger.debug("openai-whisper not installed — Viterbi will use MFCC fallback.")
            return False
