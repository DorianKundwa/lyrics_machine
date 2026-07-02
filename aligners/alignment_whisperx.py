"""
WhisperX-based forced alignment module — v3 (global DP + matched-anchor bounds).
Replaces alignment_aeneas.py entirely.

Architecture:
  1. Run the global DP across ALL asr_words vs ALL user lyric words.
     This gives the most accurate word↔timestamp correspondences.
  2. For each lyric line, determine fragment [begin, end] from the
     MATCHED word timestamps only — never from interpolated values.
     This prevents a single well-matched first word from stretching the
     fragment across a long silence.
  3. Interpolate missing word timestamps only WITHIN the matched-anchor
     budget for that line.
  4. Post-process: redistribute any silence larger than MAX_WORD_GAP_SEC
     inside a fragment, and enforce global monotonicity.

Public API (identical signatures to old module):
  align(audio_path, lyrics_path, output_json=None, language=None) -> str
  lrc_to_alignment(audio_path, lrc_path, output_json, language=None) -> str | None
  parse_alignment_json(json_path) -> list[dict]
"""

import os
import sys
import json
import re
import argparse
import unicodedata
from typing import List, Optional, Tuple

try:
    from .config import ALIGN_DIR
    from .audio_utils import convert_to_wav, get_duration
except Exception:
    from config import ALIGN_DIR
    from audio_utils import convert_to_wav, get_duration


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

MAX_WORD_GAP_SEC      = 6.0   # Internal word gap larger than this is redistributed
MAX_FRAGMENT_SEC      = 15.0  # Fragment longer than this is flagged / redistributed
MIN_LINE_DUR          = 0.15  # Minimum fragment duration
TYPICAL_WORD_DUR      = 0.35  # Fallback word duration when interpolating
INTRO_PUSH_STEP       = 5.0   # Max seconds to push unmatched intro words back

COST_MATCH   = 0.0
COST_PARTIAL = 0.5
COST_MISMATCH = 3.0
COST_MISS_USER = 2.0   # user lyric word absent from ASR
COST_SKIP_ASR  = 0.8   # extra ASR word (noise / ad-lib)


# ---------------------------------------------------------------------------
# Language normalisation
# ---------------------------------------------------------------------------

_LANG_MAP = {
    "en": "en", "eng": "en",
    "es": "es", "spa": "es",
    "fr": "fr", "fra": "fr",
    "de": "de", "deu": "de",
    "it": "it", "ita": "it",
    "pt": "pt", "por": "pt",
    "nl": "nl", "nld": "nl",
    "pl": "pl", "pol": "pl",
    "sv": "sv", "swe": "sv",
    "tr": "tr", "tur": "tr",
    "ru": "ru", "rus": "ru",
    "uk": "uk", "ukr": "uk",
    "cs": "cs", "ces": "cs",
    "el": "el", "ell": "el",
    "ar": "ar", "ara": "ar",
    "hi": "hi", "hin": "hi",
    "vi": "vi", "vie": "vi",
    "zh": "zh", "zho": "zh", "cmn": "zh",
    "ja": "ja", "jpn": "ja",
    "ko": "ko", "kor": "ko",
    "th": "th", "tha": "th",
}


def _normalize_language(lang: Optional[str]) -> Optional[str]:
    """Map a language hint string to a normalised 2-letter code, or None for auto-detect."""
    if not lang:
        return None  # None means: let WhisperX auto-detect from audio
    key = str(lang).lower()[:3]
    return _LANG_MAP.get(key, _LANG_MAP.get(key[:2], None))


def _detect_language_from_text(text: str) -> Optional[str]:
    """
    Detect the dominant language of a block of text using langdetect.
    Returns a 2-letter ISO code (e.g. 'en', 'zh', 'ja') or None if detection fails.
    Handles mixed-language lyrics gracefully — if multiple languages are detected
    we return the most probable one (langdetect already does this internally).
    """
    if not text or not text.strip():
        return None
    try:
        from langdetect import detect  # type: ignore
        return detect(text.strip())
    except Exception:
        pass
    # Character-script heuristic fallback (no external lib needed)
    cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff' or
                    '\u3040' <= ch <= '\u30ff' or
                    '\uac00' <= ch <= '\ud7a3' or
                    '\u0e00' <= ch <= '\u0e7f')
    if cjk_count > len(text) * 0.1:
        ja_count = sum(1 for ch in text if '\u3040' <= ch <= '\u30ff')
        ko_count = sum(1 for ch in text if '\uac00' <= ch <= '\ud7a3')
        th_count = sum(1 for ch in text if '\u0e00' <= ch <= '\u0e7f')
        zh_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        return max([('ja', ja_count), ('ko', ko_count), ('th', th_count), ('zh', zh_count)],
                   key=lambda x: x[1])[0]
    return None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _norm_word(w: str) -> str:
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", w.lower())


_CJK_LANGS = {"zh", "ja", "ko", "th"}


def _needs_char_tokenization(lang: str) -> bool:
    """Return True for languages that need character-level tokenisation (CJK, Thai)."""
    return _normalize_language(lang) in _CJK_LANGS


def _tokenize(line: str, lang: str = "en") -> List[str]:
    """Split a lyric line into tokens.
    For CJK/Thai: one character per token (skipping whitespace).
    For all others: whitespace split.
    """
    s = str(line or "").strip()
    if not s:
        return []
    if _needs_char_tokenization(lang):
        return [ch for ch in s if not ch.isspace()]
    return [t for t in re.split(r"\s+", s) if t]


# ---------------------------------------------------------------------------
# Audio-energy word boundary helper (ported from alignment_aeneas.py)
# ---------------------------------------------------------------------------

def _distribute_words_by_audio(
    audio_path: str,
    start_ms: int,
    end_ms: int,
    tokens: List[str],
) -> List[dict]:
    """Approximate per-word timings inside a line using audio energy minima
    near uniformly spaced boundaries (pydub-based, no ML required).
    Falls back to equal-length slices when pydub is unavailable.
    """
    try:
        from pydub import AudioSegment  # type: ignore
    except Exception:
        AudioSegment = None  # type: ignore

    start_ms = int(max(0, start_ms))
    end_ms   = int(max(start_ms, end_ms))

    def _equal_slices():
        dur = max(0, end_ms - start_ms)
        n   = max(1, len(tokens))
        return [
            {"text": tok,
             "start": (start_ms + int((i / n) * dur)) / 1000.0,
             "end":   (start_ms + int(((i + 1) / n) * dur)) / 1000.0}
            for i, tok in enumerate(tokens)
        ]

    if AudioSegment is None or end_ms <= start_ms or not tokens:
        return _equal_slices()

    try:
        audio = AudioSegment.from_file(audio_path)
        seg   = audio[start_ms:end_ms]
    except Exception:
        return _equal_slices()

    frame_ms   = 10
    frames     = max(1, int(len(seg) / frame_ms))
    energies   = []
    for i in range(frames):
        s = i * frame_ms
        e = min(len(seg), s + frame_ms)
        chunk = seg[s:e]
        energies.append(chunk.dBFS if len(chunk) > 0 else -90.0)

    n_words      = max(1, len(tokens))
    total_frames = len(energies)
    uniform_bounds = [int((i / n_words) * total_frames) for i in range(1, n_words)]

    radius = 8
    refined_bounds = []
    for ub in uniform_bounds:
        lo = max(0, ub - radius)
        hi = min(total_frames - 1, ub + radius)
        if lo >= hi:
            refined_bounds.append(ub)
            continue
        window  = energies[lo:hi + 1]
        min_idx = window.index(min(window))
        refined_bounds.append(lo + min_idx)

    bounds_ms = [start_ms] + [start_ms + b * frame_ms for b in refined_bounds] + [end_ms]
    return [
        {"text": tok,
         "start": bounds_ms[i]       / 1000.0,
         "end":   bounds_ms[i + 1]   / 1000.0}
        for i, tok in enumerate(tokens)
    ]


def _ensure_wav(audio_path: str) -> str:
    audio_path = os.path.abspath(audio_path)
    if os.path.splitext(audio_path)[1].lower() == ".wav":
        return audio_path
    wav = os.path.splitext(audio_path)[0] + "_wx.wav"
    try:
        if convert_to_wav(audio_path, wav):
            return os.path.abspath(wav)
    except Exception as e:
        print(f"[whisperx] WAV conversion failed: {e}")
    return audio_path


# ---------------------------------------------------------------------------
# Global DP aligner
# ---------------------------------------------------------------------------

def _dp_match(user_words: List[str], asr_words: List[dict]) -> dict:
    """
    Align user_words against asr_words using edit-distance DP.
    Returns mapping: user_word_index -> asr_word_index (matched pairs only).
    """
    n = len(user_words)
    m = len(asr_words)
    if n == 0 or m == 0:
        return {}

    INF = 1e9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + COST_MISS_USER
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + COST_SKIP_ASR

    for i in range(1, n + 1):
        uw = _norm_word(user_words[i - 1])
        for j in range(1, m + 1):
            aw = _norm_word(asr_words[j - 1].get("word", ""))
            if uw and aw and uw == aw:
                sub = COST_MATCH
            elif uw and aw and min(len(uw), len(aw)) >= 3 and (uw in aw or aw in uw):
                sub = COST_PARTIAL
            else:
                sub = COST_MISMATCH
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub,
                dp[i - 1][j] + COST_MISS_USER,
                dp[i][j - 1] + COST_SKIP_ASR,
            )

    mapping: dict = {}
    i, j = n, m
    while i > 0 and j > 0:
        uw = _norm_word(user_words[i - 1])
        aw = _norm_word(asr_words[j - 1].get("word", ""))
        if uw and aw and uw == aw:
            sub = COST_MATCH
        elif uw and aw and min(len(uw), len(aw)) >= 3 and (uw in aw or aw in uw):
            sub = COST_PARTIAL
        else:
            sub = COST_MISMATCH
        if abs(dp[i][j] - (dp[i - 1][j - 1] + sub)) < 1e-9 and sub <= COST_PARTIAL:
            mapping[i - 1] = j - 1
            i -= 1; j -= 1
        elif abs(dp[i][j] - (dp[i - 1][j] + COST_MISS_USER)) < 1e-9:
            i -= 1
        elif abs(dp[i][j] - (dp[i][j - 1] + COST_SKIP_ASR)) < 1e-9:
            j -= 1
        else:
            i -= 1; j -= 1
    return mapping


# ---------------------------------------------------------------------------
# Fragment boundary calculation from matched anchors
# ---------------------------------------------------------------------------

def _line_bounds_from_anchors(
    matched_starts: List[Optional[float]],
    matched_ends:   List[Optional[float]],
    prev_frag_end: float,
    next_frag_start: Optional[float],
    audio_duration: float,
    n_words: int,
) -> Tuple[float, float]:
    """
    Compute a fragment's [begin, end] using only the timestamps of
    actually-matched words (not fill-in interpolations).

    Key rule: the fragment begin/end is derived from the RANGE of
    matched timestamps, clamped to MAX_FRAGMENT_SEC. This prevents one
    matched word at 7s and another at 29s from creating a 22s fragment.
    """
    matched = [
        (s, e) for s, e in zip(matched_starts, matched_ends)
        if s is not None
    ]

    if not matched:
        # No matches — use a proportional slot after prev_frag_end
        slot = min(MAX_FRAGMENT_SEC, (next_frag_start or audio_duration) - prev_frag_end)
        slot = max(MIN_LINE_DUR * 2, slot / max(n_words, 1) * n_words)
        return (
            max(prev_frag_end + 0.05, prev_frag_end),
            min(prev_frag_end + slot, audio_duration),
        )

    raw_start = matched[0][0]
    raw_end   = matched[-1][1]

    # If the matched span is too long, there's likely a silence jump inside.
    # In that case, trust only the densely-matched cluster at the beginning,
    # trimming the end to raw_start + MAX_FRAGMENT_SEC.
    if raw_end - raw_start > MAX_FRAGMENT_SEC:
        # Walk through matched words; stop at first gap > MAX_WORD_GAP_SEC
        cluster_end = matched[0][1]
        for s, e in matched[1:]:
            if s - cluster_end > MAX_WORD_GAP_SEC:
                break
            cluster_end = e
        raw_end = cluster_end

    begin = max(prev_frag_end + 0.01, raw_start - 0.02)
    end   = max(begin + MIN_LINE_DUR, raw_end + 0.03)
    return begin, end


# ---------------------------------------------------------------------------
# Word-level gap filling (within a fragment's budget)
# ---------------------------------------------------------------------------

def _fill_word_gaps(
    words:       List[dict],
    frag_start:  float,
    frag_end:    float,
    matched_flags: List[bool],
) -> None:
    """
    Redistribute unmatched words inside a fragment.
    Matched words keep their real timestamps; unmatched words are
    interpolated in the gaps between matched neighbours, capped at
    MAX_WORD_GAP_SEC per gap.

    Mutates words in-place.
    """
    n = len(words)
    if n == 0:
        return

    # Ensure matched words are within [frag_start, frag_end]
    for i, (w, matched) in enumerate(zip(words, matched_flags)):
        if matched:
            w["start"] = max(frag_start, min(w["start"], frag_end - 0.05))
            w["end"]   = max(w["start"] + 0.05, min(w["end"],   frag_end))

    # Fill unmatched chunks between anchors
    i = 0
    while i < n:
        if matched_flags[i]:
            i += 1
            continue

        # Find extent of this unmatched chunk
        j = i
        while j < n and not matched_flags[j]:
            j += 1

        count = j - i
        # Previous anchor
        prev_anchor_end = words[i - 1]["end"] if i > 0 else frag_start
        # Next anchor
        next_anchor_start = words[j]["start"] if j < n else frag_end

        # Clamp raw gap to MAX_WORD_GAP_SEC
        available = min(next_anchor_start - prev_anchor_end, MAX_WORD_GAP_SEC)
        available = max(available, MIN_LINE_DUR * count)

        if i == 0 and not any(matched_flags):
            # Fully unmatched line — distribute evenly over frag budget
            span  = frag_end - frag_start
            step  = min(span / count, TYPICAL_WORD_DUR * 2)
            start = frag_start
        elif i == 0:
            # Unmatched block at beginning — push backwards from next anchor
            step  = min(INTRO_PUSH_STEP, available / count)
            start = max(frag_start, next_anchor_start - count * step)
        else:
            step  = min(available / count, TYPICAL_WORD_DUR * 2)
            start = prev_anchor_end + 0.03

        for k in range(i, j):
            words[k]["start"] = round(start + (k - i) * step, 3)
            words[k]["end"]   = round(words[k]["start"] + max(step * 0.85, MIN_LINE_DUR), 3)

        i = j


# ---------------------------------------------------------------------------
# Core mapping: global DP then per-line matched-anchor bounds
# ---------------------------------------------------------------------------

def _map_asr_words_to_lyrics(
    asr_words:   List[dict],
    lyric_lines: List[str],
    audio_duration: float,
) -> List[dict]:
    """
    1. Flatten all lyric lines into a word list; run global DP.
    2. For each lyric line, compute fragment [begin, end] from MATCHED
       timestamps only (preventing silence-stretch artefacts).
    3. Fill unmatched word gaps within the fragment budget.
    4. Enforce global monotonicity.
    """
    # Flatten user lyrics
    user_words:   List[str] = []
    word_to_line: List[int] = []
    for li, line in enumerate(lyric_lines):
        for tok in _tokenize(line):
            user_words.append(tok)
            word_to_line.append(li)

    n = len(user_words)
    m = len(asr_words)
    if n == 0:
        return []

    # Global DP
    mapping = _dp_match(user_words, asr_words) if m > 0 else {}
    print(f"[whisperx] DP matched {len(mapping)}/{n} user words")

    # Assign matched timestamps
    t_starts:       List[Optional[float]] = [None] * n
    t_ends:         List[Optional[float]] = [None] * n
    matched_flags:  List[bool]            = [False] * n
    for ui, ai in mapping.items():
        t_starts[ui]      = float(asr_words[ai].get("start", 0.0))
        t_ends[ui]        = float(asr_words[ai].get("end",   0.0))
        matched_flags[ui] = True

    # Group into per-line word buckets
    line_tstart: List[List[Optional[float]]] = [[] for _ in lyric_lines]
    line_tend:   List[List[Optional[float]]] = [[] for _ in lyric_lines]
    line_match:  List[List[bool]]            = [[] for _ in lyric_lines]
    line_toks:   List[List[str]]             = [[] for _ in lyric_lines]

    for wi, li in enumerate(word_to_line):
        line_tstart[li].append(t_starts[wi])
        line_tend[li].append(t_ends[wi])
        line_match[li].append(matched_flags[wi])
        line_toks[li].append(user_words[wi])

    # Build fragments
    fragments: List[dict] = []
    prev_end = 0.0

    for li, line in enumerate(lyric_lines):
        ms = line_tstart[li]
        me = line_tend[li]
        mf = line_match[li]
        toks = line_toks[li]

        if not toks:
            continue

        # Compute fragment bounds from matched anchors only
        next_start: Optional[float] = None
        for lj in range(li + 1, len(lyric_lines)):
            for ts, flag in zip(line_tstart[lj], line_match[lj]):
                if flag and ts is not None:
                    next_start = ts
                    break
            if next_start is not None:
                break

        f_begin, f_end = _line_bounds_from_anchors(
            ms, me, prev_end, next_start, audio_duration, len(toks)
        )

        # Build word list — initially keep matched timestamps, fill the rest
        word_objs: List[dict] = []
        for wi_loc, tok in enumerate(toks):
            ts = ms[wi_loc]
            te = me[wi_loc]
            if ts is not None and te is not None:
                word_objs.append({"text": tok, "start": float(ts), "end": float(te)})
            else:
                word_objs.append({"text": tok, "start": 0.0, "end": 0.0})

        # Fill unmatched word gaps within the fragment budget
        _fill_word_gaps(word_objs, f_begin, f_end, mf)

        # Recalculate fragment bounds after filling
        f_begin = min(w["start"] for w in word_objs)
        f_end   = max(w["end"]   for w in word_objs)
        f_begin = max(prev_end + 0.01, f_begin)
        f_end   = max(f_begin + MIN_LINE_DUR, f_end)

        fragments.append({
            "begin": f"{f_begin:.3f}",
            "end":   f"{f_end:.3f}",
            "lines": [line],
            "words": [
                {"text": w["text"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
                for w in word_objs
            ],
        })
        prev_end = f_end

    # Global monotonicity pass
    _enforce_global_monotonicity(fragments, audio_duration)
    return fragments


def _enforce_global_monotonicity(fragments: List[dict], audio_duration: float) -> None:
    prev_end = 0.0
    for frag in fragments:
        f_start = max(float(frag["begin"]), prev_end + 0.01)
        f_start = min(f_start, audio_duration - 0.05)
        f_end   = float(frag["end"])
        if f_end <= f_start:
            f_end = f_start + MIN_LINE_DUR

        cursor = f_start
        for w in frag.get("words", []):
            if w["start"] < cursor:
                w["start"] = round(cursor, 3)
            if w["end"] < w["start"] + 0.05:
                w["end"] = round(w["start"] + 0.10, 3)
            
            # Strict bound check
            w["start"] = round(min(w["start"], audio_duration - 0.02), 3)
            w["end"]   = round(min(w["end"], audio_duration - 0.01), 3)
            cursor = w["end"]

        frag["begin"] = f"{f_start:.3f}"
        frag["end"]   = f"{min(max(f_end, cursor), audio_duration):.3f}"
        prev_end = float(frag["end"])


# ---------------------------------------------------------------------------
# Heuristic fallback (no ML)
# ---------------------------------------------------------------------------

def _heuristic_align(
    audio_path:  str,
    lyrics_path: str,
    output_json: str,
    language:    Optional[str] = None,
) -> str:
    try:
        dur = float(get_duration(audio_path) or 0.0)
    except Exception:
        dur = 0.0

    with open(lyrics_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Auto-detect language from lyrics text if not provided
    if not language:
        all_text = " ".join(lines)
        language = _detect_language_from_text(all_text) or "en"
        print(f"[heuristic] Auto-detected language from lyrics text: {language}")
    else:
        language = _normalize_language(language) or "en"

    if dur <= 0.0:
        dur = max(1.0, len(lines) * 2.0)

    step = dur / max(1, len(lines))
    fragments = []
    for i, txt in enumerate(lines):
        b = i * step
        e = min(dur, (i + 1) * step)
        toks = _tokenize(txt, language)
        try:
            words = _distribute_words_by_audio(
                audio_path, int(b * 1000), int(e * 1000), toks
            )
        except Exception:
            w_step = (e - b) / max(1, len(toks))
            words = [
                {"text": t,
                 "start": round(b + j * w_step, 3),
                 "end":   round(b + (j + 1) * w_step, 3)}
                for j, t in enumerate(toks)
            ]
        fragments.append({
            "begin": f"{b:.3f}",
            "end":   f"{e:.3f}",
            "lines": [txt],
            "words": [{"text": w["text"], "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in words],
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"fragments": fragments}, f, indent=2, ensure_ascii=False)
    return output_json


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def align(
    audio_path: str,
    lyrics_path: str,
    output_json: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    if output_json is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_json = os.path.join(ALIGN_DIR, f"{base}_alignment.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    lang = _normalize_language(language)  # None when no hint provided = full auto-detect

    wav_path = _ensure_wav(audio_path)
    print(f"[whisperx] Audio:    {wav_path}")
    print(f"[whisperx] Lyrics:   {lyrics_path}")
    print(f"[whisperx] Output:   {output_json}")
    print(f"[whisperx] Language: {'auto-detect' if not lang else lang}")

    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyric_lines = [ln.strip() for ln in f if ln.strip()]

    # _lang_for_tokenize will be updated to the detected language after transcription
    _lang_for_tokenize = lang or "en"

    try:
        import whisperx
        import torch
        import inspect
        import functools

        # --- FIX: PyTorch 2.6+ & SpeechBrain Compatibility ---
        # 1. Globally disable weights_only=True to allow Pyannote VAD to load properly
        _old_load = torch.load
        _new_load = lambda *a, **kw: _old_load(*a, **{**kw, 'weights_only': False})
        torch.load = _new_load
        torch.serialization.load = _new_load
        
        # 2. Mock inspect.stack() during model load to prevent SpeechBrain from crashing
        # due to its importutils lazy-loading bug being triggered by pytorch_lightning's is_scripting check.
        _orig_stack = inspect.stack
        inspect.stack = lambda *a, **kw: []

        device       = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[whisperx] Device: {device}, compute: {compute_type}")

        print("[whisperx] Loading ASR model (small)...")
        model = whisperx.load_model("small", device=device, compute_type=compute_type)
        
        # Restore normal inspect.stack() functionality
        inspect.stack = _orig_stack
        # -----------------------------------------------------

        audio = whisperx.load_audio(wav_path)

        try:
            audio_duration = len(audio) / 16000.0
        except Exception:
            audio_duration = float(get_duration(wav_path) or 180.0)

        print("[whisperx] Transcribing (language auto-detected from audio)...")
        # Pass lang only if we have a reliable hint; otherwise let Whisper detect freely
        transcribe_kwargs: dict = {"batch_size": 4}
        if lang:
            transcribe_kwargs["language"] = lang
        result = model.transcribe(audio, **transcribe_kwargs)
        detected_lang = result.get("language") or lang or "en"
        print(f"[whisperx] Detected language: {detected_lang}")

        print(f"[whisperx] Loading alignment model ({detected_lang})...")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        print("[whisperx] Running forced alignment...")
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )

        asr_words: List[dict] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    asr_words.append(w)
        asr_words.sort(key=lambda x: x.get("start", 0.0))
        print(f"[whisperx] Got {len(asr_words)} word timestamps")

        print("[whisperx] Mapping to user lyrics (global DP + matched-anchor bounds)...")
        fragments = _map_asr_words_to_lyrics(asr_words, lyric_lines, audio_duration)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"fragments": fragments}, f, indent=2, ensure_ascii=False)

        print(f"[whisperx] Done. {len(fragments)} fragments -> {output_json}")
        return output_json

    except Exception as e:
        print(f"[whisperx] WhisperX failed: {e}. Using heuristic fallback.")
        import traceback; traceback.print_exc()
        return _heuristic_align(wav_path, lyrics_path, output_json, language=lang)


# ---------------------------------------------------------------------------
# LRC support
# ---------------------------------------------------------------------------

def _parse_lrc_time(tag: str) -> Optional[float]:
    try:
        s = tag.strip().strip("[]")
        parts = s.split(":")
        if len(parts) >= 2:
            mm = int(parts[0])
            ss = float(parts[1])
            return mm * 60.0 + ss
    except Exception:
        pass
    return None


def lrc_to_alignment(
    audio_path: str,
    lrc_path: str,
    output_json: str,
    language: Optional[str] = None,
) -> Optional[str]:
    try:
        with open(lrc_path, "r", encoding="utf-8") as f:
            raw_lines = [ln.rstrip("\n") for ln in f]
    except Exception:
        return None

    entries = []
    offset_sec = 0.0
    for ln in raw_lines:
        i, tags, times = 0, [], []
        while i < len(ln) and ln[i] == "[":
            j = ln.find("]", i + 1)
            if j == -1:
                break
            tag_body = ln[i + 1:j]
            tags.append(tag_body)
            i = j + 1
        text = ln[i:].strip()
        for tag in tags:
            if tag.startswith("offset:"):
                try:
                    offset_sec = float(tag.split(":", 1)[1]) / 1000.0
                except Exception:
                    pass
            t = _parse_lrc_time("[" + tag + "]")
            if t is not None:
                times.append(t)
        for tm in times:
            entries.append({"start": max(0.0, tm + offset_sec), "text": text})

    lang = _normalize_language(language)
    entries = sorted([e for e in entries if e["text"]], key=lambda x: x["start"])
    fragments = []
    for idx, e in enumerate(entries):
        s   = e["start"]
        nxt = entries[idx + 1]["start"] if idx + 1 < len(entries) else s + 2.0
        end = max(s + 0.1, float(nxt))
        toks = _tokenize(e["text"], lang)
        try:
            words = _distribute_words_by_audio(
                audio_path, int(s * 1000), int(end * 1000), toks
            )
            words = [
                {"text": w["text"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
                for w in words
            ]
        except Exception:
            dur    = end - s
            w_step = dur / max(1, len(toks))
            words  = [
                {"text": t, "start": round(s + k * w_step, 3), "end": round(s + (k + 1) * w_step, 3)}
                for k, t in enumerate(toks)
            ]
        fragments.append({"begin": f"{s:.3f}", "end": f"{end:.3f}", "lines": [e["text"]], "words": words})

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"fragments": fragments}, f, indent=2, ensure_ascii=False)
        return output_json
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_alignment_json(json_path: str) -> List[dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "fragments" not in data:
            raise ValueError("Alignment JSON must contain 'fragments'")
        result = []
        for i, frag in enumerate(data["fragments"]):
            lines_field = frag.get("lines")
            text = (
                lines_field[0]
                if isinstance(lines_field, list) and lines_field
                else (lines_field or "")
            )
            result.append({
                "index": i,
                "start": float(frag.get("begin", 0)),
                "end":   float(frag.get("end",   0)),
                "text":  text,
                "words": frag.get("words", []),
            })
        return result
    except Exception as e:
        print(f"[whisperx] Error parsing alignment JSON: {e}")
        return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align audio and lyrics using WhisperX v3")
    parser.add_argument("--audio",    required=True, help="Path to audio file")
    parser.add_argument("--lyrics",   required=True, help="Path to plain-text lyrics file")
    parser.add_argument("--output",   default=None,  help="Path to save alignment JSON")
    parser.add_argument("--language", default=None,  help="Language code (e.g. en, es, fr)")
    args = parser.parse_args()

    try:
        out     = align(args.audio, args.lyrics, args.output, args.language)
        entries = parse_alignment_json(out)
        print(f"\nSuccess: {len(entries)} aligned lines → {out}")
        for e in entries[:10]:
            print(f"  [{e['start']:7.2f}s -> {e['end']:7.2f}s]  {e['text'][:65]}")
        if len(entries) > 10:
            print(f"  ... ({len(entries) - 10} more)")
    except Exception as exc:
        print(f"Alignment failed: {exc}")
        sys.exit(1)
