"""
backend/alignment_hybrid.py

Hybrid Lyric Alignment Engine — 5-layer architecture.

Architecture:
  Layer 1: Audio Analysis     — VAD, onset detection, instrumental gap detection
  Layer 2: Text + Matching    — fuzzy DP with rapidfuzz + jellyfish phonetic matching
  Layer 3: Structural Tagging — verse / background / instrumental classification
  Layer 4: Interpolation      — cubic spline for unmatched words, energy-based refinement
  Layer 5: Quality Assurance  — monotonicity, smooth transitions, boundary clamping

Public API (identical signatures to alignment_whisperx.py):
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
from typing import List, Optional, Tuple, Dict

try:
    from .config import ALIGN_DIR
    from .audio_utils import convert_to_wav, get_duration
except Exception:
    from config import ALIGN_DIR
    from audio_utils import convert_to_wav, get_duration

# ---------------------------------------------------------------------------
# Optional dependencies — degrade gracefully when missing
# ---------------------------------------------------------------------------

try:
    import rapidfuzz.fuzz as _rf
except ImportError:
    _rf = None

try:
    import jellyfish as _jf
except ImportError:
    _jf = None

try:
    from scipy.interpolate import CubicSpline as _CubicSpline
except ImportError:
    _CubicSpline = None

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

MAX_FRAGMENT_SEC   = 5.5    # Fragment longer than this is suspect
MIN_LINE_DUR       = 0.15   # Minimum fragment duration
TYPICAL_WORD_DUR   = 0.25   # Fallback word duration when interpolating
MAX_WORD_GAP_SEC   = 2.0    # Internal word gap cap
SMOOTH_GAP_THRESH  = 0.20   # Gaps smaller than this are closed by _smooth_transitions

# DP cost model
COST_EXACT      = 0.0    # Normalised words match exactly
COST_PHONETIC   = 0.3    # Metaphone keys match
COST_FUZZY      = 0.5    # Levenshtein ratio ≥ 85 %
COST_FUZZY_LOW  = 1.0    # Levenshtein ratio ≥ 65 %
COST_MISMATCH   = 3.0    # Nothing matches
COST_MISS_USER  = 2.0    # User lyric word absent from ASR
COST_SKIP_ASR   = 0.8    # Extra ASR word (noise / ad-lib)
MATCH_THRESHOLD = 1.5    # Max substitution cost recorded as a "match"


# ===========================================================================
# LAYER 1 — Audio Analysis
# ===========================================================================

def _detect_instrumental_gaps(
    audio_path: str,
    min_gap_sec: float = 3.0,
    silence_thresh_db: float = -35.0,
) -> List[dict]:
    """Find long silent / instrumental sections using RMS energy."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
    except Exception:
        return []

    frame_ms = 50
    n_frames = max(1, int(len(audio) / frame_ms))

    # Compute per-frame dBFS
    energies = []
    for i in range(n_frames):
        s = i * frame_ms
        e = min(len(audio), s + frame_ms)
        chunk = audio[s:e]
        energies.append(chunk.dBFS if len(chunk) > 0 else -90.0)

    # Find runs below threshold
    gaps: List[dict] = []
    in_gap = False
    gap_start = 0.0
    for i, db in enumerate(energies):
        t = i * frame_ms / 1000.0
        if db < silence_thresh_db:
            if not in_gap:
                gap_start = t
                in_gap = True
        else:
            if in_gap:
                gap_end = t
                if gap_end - gap_start >= min_gap_sec:
                    gaps.append({"start": gap_start, "end": gap_end})
                in_gap = False
    # Close trailing gap
    if in_gap:
        gap_end = n_frames * frame_ms / 1000.0
        if gap_end - gap_start >= min_gap_sec:
            gaps.append({"start": gap_start, "end": gap_end})

    return gaps


# ===========================================================================
# LAYER 2 — Text Processing & Matching
# ===========================================================================

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

_CJK_LANGS = {"zh", "ja", "ko", "th"}


def _normalize_language(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    key = str(lang).lower()[:3]
    return _LANG_MAP.get(key, _LANG_MAP.get(key[:2], "en"))


def _norm_word(w: str) -> str:
    """Lowercase, strip accents and non-alphanumeric characters."""
    w = unicodedata.normalize("NFKD", str(w or ""))
    w = w.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", w.lower())


def _needs_char_tokenization(lang: str) -> bool:
    return _normalize_language(lang) in _CJK_LANGS


def _tokenize(line: str, lang: str = "en") -> List[str]:
    """Split a lyric line into tokens."""
    s = str(line or "").strip()
    if not s:
        return []
    if _needs_char_tokenization(lang):
        return [ch for ch in s if not ch.isspace()]
    return [t for t in re.split(r"\s+", s) if t]


def _is_background_token(text: str) -> bool:
    """Return True if the text is fully enclosed in () or []."""
    s = str(text or "").strip()
    if not s:
        return False
    return (
        (s.startswith("(") and s.endswith(")"))
        or (s.startswith("[") and s.endswith("]"))
    )


def _strip_bg_brackets(text: str) -> str:
    """Remove outer parentheses/brackets if fully enclosed."""
    s = str(text or "").strip()
    if _is_background_token(s):
        return s[1:-1].strip()
    return s


# --- Fuzzy & Phonetic helpers ---

def _levenshtein_ratio(a: str, b: str, rf=None) -> float:
    """Return similarity ratio (0–100) between two strings.
    Uses rapidfuzz if available, otherwise a manual DP implementation."""
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    # Prefer rapidfuzz (C-optimised)
    if rf is not None:
        try:
            return float(rf.ratio(a, b))
        except Exception:
            pass
    # Manual Levenshtein
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    dist = dp[m]
    max_len = max(n, m)
    return (1.0 - dist / max_len) * 100.0


def _phonetic_key(word: str, jf=None) -> str:
    """Return a phonetic key (Metaphone) for the word.
    Falls back to 3-char uppercase prefix when jellyfish is unavailable."""
    if jf is not None:
        try:
            return jf.metaphone(str(word))
        except Exception:
            pass
    # Fallback: first 3 characters uppercased
    return str(word)[:3].upper() if word else ""


def _sub_cost(un: str, an: str, up: Optional[str] = None, ap: Optional[str] = None) -> float:
    """Compute substitution cost for a single (user, ASR) word pair.
    Accepts pre-computed normalised words and phonetic keys."""
    if not un or not an:
        return COST_MISMATCH
    # Exact
    if un == an:
        return COST_EXACT
    # Phonetic
    if up and ap and up == ap:
        return COST_PHONETIC
    # Fuzzy (Levenshtein)
    ratio = _levenshtein_ratio(un, an, _rf)
    if ratio >= 85.0:
        return COST_FUZZY
    if ratio >= 65.0:
        return COST_FUZZY_LOW
    # Substring containment (abbreviations / contractions)
    if min(len(un), len(an)) >= 3 and (un in an or an in un):
        return COST_FUZZY_LOW
    return COST_MISMATCH


# --- Global DP aligner ---

def _hybrid_match(user_words: List[str], asr_words: List[dict]) -> Dict[int, int]:
    """Align user_words against asr_words using edit-distance DP enhanced
    with fuzzy + phonetic scoring.

    Returns mapping: user_word_index -> asr_word_index (matched pairs only).
    Unmatched user words are not in the dict.
    """
    n = len(user_words)
    m = len(asr_words)
    if n == 0 or m == 0:
        return {}

    # Pre-compute normalised forms and phonetic keys
    u_norms = [_norm_word(w) for w in user_words]
    a_norms = [_norm_word(w.get("word", "") if isinstance(w, dict) else str(w)) for w in asr_words]

    if _jf is not None:
        u_phones = [_phonetic_key(un, _jf) for un in u_norms]
        a_phones = [_phonetic_key(an, _jf) for an in a_norms]
    else:
        u_phones = [None] * n
        a_phones = [None] * m

    # Forward DP
    INF = 1e9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + COST_MISS_USER
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + COST_SKIP_ASR

    for i in range(1, n + 1):
        un = u_norms[i - 1]
        up = u_phones[i - 1]
        for j in range(1, m + 1):
            an = a_norms[j - 1]
            ap = a_phones[j - 1]
            sc = _sub_cost(un, an, up, ap)
            dp[i][j] = min(
                dp[i - 1][j - 1] + sc,
                dp[i - 1][j] + COST_MISS_USER,
                dp[i][j - 1] + COST_SKIP_ASR,
            )

    # Traceback — record a match only if sub cost ≤ MATCH_THRESHOLD
    mapping: Dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        un = u_norms[i - 1]
        an = a_norms[j - 1]
        up = u_phones[i - 1]
        ap = a_phones[j - 1]
        sc = _sub_cost(un, an, up, ap)

        diag = dp[i - 1][j - 1] + sc
        drop_u = dp[i - 1][j] + COST_MISS_USER
        drop_a = dp[i][j - 1] + COST_SKIP_ASR

        if abs(dp[i][j] - diag) < 1e-9 and sc <= MATCH_THRESHOLD:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif abs(dp[i][j] - drop_u) < 1e-9:
            i -= 1
        elif abs(dp[i][j] - drop_a) < 1e-9:
            j -= 1
        else:
            # Diagonal mismatch: consume both without recording a match
            i -= 1
            j -= 1

    return mapping


# ===========================================================================
# LAYER 3 — Structural Tagging
# ===========================================================================

def _tag_background_vocals(lines: List[str]) -> List[str]:
    """Classify each lyric line as 'verse' or 'background'.
    Lines fully enclosed in parentheses/brackets are background."""
    tags: List[str] = []
    for line in lines:
        s = str(line or "").strip()
        if _is_background_token(s):
            tags.append("background")
        else:
            tags.append("verse")
    return tags


# ===========================================================================
# LAYER 4 — Interpolation & Refinement
# ===========================================================================

def _cubic_spline_interpolate(
    anchor_times: List[float],
    anchor_indices: List[int],
    total_count: int,
    start_time: float,
    end_time: float,
) -> List[float]:
    """Interpolate timestamps for all positions [0, total_count) using
    known anchor points.  Uses scipy CubicSpline when available,
    otherwise falls back to piecewise linear interpolation.

    Returns a monotonically non-decreasing list of ``total_count`` floats
    clamped to [start_time, end_time].
    """
    if total_count <= 0:
        return []

    # Linear fallback when we have < 2 anchors or no scipy
    if len(anchor_times) < 2 or _CubicSpline is None:
        step = (end_time - start_time) / max(1, total_count)
        result = [start_time + i * step for i in range(total_count)]
        # If we have exactly one anchor, snap it
        if len(anchor_times) == 1 and 0 <= anchor_indices[0] < total_count:
            result[anchor_indices[0]] = anchor_times[0]
        # Enforce monotonicity and bounds
        for i in range(total_count):
            result[i] = max(start_time, min(end_time, result[i]))
        for i in range(1, total_count):
            if result[i] < result[i - 1]:
                result[i] = result[i - 1]
        return result

    # Cubic spline through anchors
    try:
        cs = _CubicSpline(anchor_indices, anchor_times, bc_type="clamped")
        xs = list(range(total_count))
        result = [float(cs(x)) for x in xs]
    except Exception:
        # Fallback to linear interpolation between successive anchors
        result = [0.0] * total_count
        pairs = sorted(zip(anchor_indices, anchor_times))
        for idx in range(total_count):
            # Find surrounding anchors
            prev_i, prev_t = 0, start_time
            next_i, next_t = total_count - 1, end_time
            for ai, at in pairs:
                if ai <= idx:
                    prev_i, prev_t = ai, at
                if ai >= idx:
                    next_i, next_t = ai, at
                    break
            if next_i == prev_i:
                result[idx] = prev_t
            else:
                frac = (idx - prev_i) / (next_i - prev_i)
                result[idx] = prev_t + frac * (next_t - prev_t)

    # Enforce monotonicity
    for i in range(1, total_count):
        if result[i] < result[i - 1]:
            result[i] = result[i - 1]

    # Clamp to [start_time, end_time]
    for i in range(total_count):
        result[i] = max(start_time, min(end_time, result[i]))

    return result


def _compute_fragment_bounds(
    matched_starts: List[Optional[float]],
    matched_ends: List[Optional[float]],
    prev_end: float,
    next_start: Optional[float],
    audio_duration: float,
    n_words: int = 1,
) -> Tuple[float, float]:
    """Compute a fragment's [begin, end] from matched-anchor timestamps.
    Uses only actually-matched word timestamps to prevent silence stretching."""
    matched = [
        (s, e)
        for s, e in zip(matched_starts, matched_ends)
        if s is not None
    ]

    if not matched:
        # No matches — allocate a proportional slot after prev_end
        budget = (next_start or audio_duration) - prev_end
        slot = min(MAX_FRAGMENT_SEC, max(MIN_LINE_DUR * 2, budget / max(n_words, 1) * n_words))
        begin = prev_end + 0.05
        end = min(begin + slot, audio_duration)
        return begin, max(begin + MIN_LINE_DUR, end)

    raw_start = matched[0][0]
    raw_end = matched[-1][1]

    # Trim overly long spans caused by silence jumps
    if raw_end - raw_start > MAX_FRAGMENT_SEC:
        cluster_end = matched[0][1]
        for s, e in matched[1:]:
            if s - cluster_end > MAX_WORD_GAP_SEC:
                break
            cluster_end = e
        raw_end = cluster_end

    begin = max(prev_end + 0.01, raw_start - 0.02)
    end = max(begin + MIN_LINE_DUR, raw_end + 0.03)
    return begin, end


# --- Audio-energy word boundary helper ---

def _distribute_words_by_audio(
    audio_path: str,
    start_ms: int,
    end_ms: int,
    tokens: List[str],
) -> List[dict]:
    """Approximate per-word timings inside a line using audio energy minima
    near uniformly spaced boundaries.  Falls back to equal slices."""
    try:
        from pydub import AudioSegment
    except Exception:
        AudioSegment = None  # type: ignore

    start_ms = int(max(0, start_ms))
    end_ms = int(max(start_ms, end_ms))

    def _equal_slices():
        dur = max(0, end_ms - start_ms)
        n = max(1, len(tokens))
        return [
            {"text": tok,
             "start": (start_ms + int((i / n) * dur)) / 1000.0,
             "end": (start_ms + int(((i + 1) / n) * dur)) / 1000.0}
            for i, tok in enumerate(tokens)
        ]

    if AudioSegment is None or end_ms <= start_ms or not tokens:
        return _equal_slices()

    try:
        audio = AudioSegment.from_file(audio_path)
        seg = audio[start_ms:end_ms]
    except Exception:
        return _equal_slices()

    frame_ms = 10
    frames = max(1, int(len(seg) / frame_ms))
    energies = []
    for i in range(frames):
        s = i * frame_ms
        e = min(len(seg), s + frame_ms)
        chunk = seg[s:e]
        energies.append(chunk.dBFS if len(chunk) > 0 else -90.0)

    n_words = max(1, len(tokens))
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
        window = energies[lo:hi + 1]
        min_idx = window.index(min(window))
        refined_bounds.append(lo + min_idx)

    bounds_ms = [start_ms] + [start_ms + b * frame_ms for b in refined_bounds] + [end_ms]
    return [
        {"text": tok,
         "start": bounds_ms[i] / 1000.0,
         "end": bounds_ms[i + 1] / 1000.0}
        for i, tok in enumerate(tokens)
    ]


# --- Silent block insertion ---

def _insert_silent_blocks(
    fragments: List[dict],
    gaps: List[dict],
) -> List[dict]:
    """Insert 'instrumental' placeholder fragments for detected silent gaps
    that are not already covered by existing fragments.
    Returns a NEW sorted list of fragments."""
    if not gaps:
        return list(fragments)

    result = list(fragments)

    for gap in gaps:
        gs = float(gap.get("start", 0))
        ge = float(gap.get("end", gs))
        if ge <= gs:
            continue

        # Check if this gap is already covered by an existing fragment
        covered = False
        for frag in fragments:
            fb = float(frag["begin"])
            fe = float(frag["end"])
            if fb <= gs and fe >= ge:
                covered = True
                break
        if covered:
            continue

        result.append({
            "begin": f"{gs:.3f}",
            "end": f"{ge:.3f}",
            "lines": ["♪"],
            "type": "instrumental",
            "words": [],
        })

    # Sort by begin time
    result.sort(key=lambda f: float(f["begin"]))
    return result


# ===========================================================================
# LAYER 5 — Quality Assurance
# ===========================================================================

def _enforce_global_monotonicity(fragments: List[dict], audio_duration: float = None) -> None:
    """Ensure all fragment and word timestamps are strictly monotonically
    ordered.  Mutates fragments in-place."""
    if not fragments:
        return

    if audio_duration is None:
        audio_duration = max((float(f["end"]) for f in fragments), default=0.0) + 1.0

    prev_end = 0.0
    for frag in fragments:
        f_start = max(float(frag["begin"]), prev_end + 0.01)
        f_start = min(f_start, audio_duration - 0.05)
        f_end = float(frag["end"])
        if f_end <= f_start:
            f_end = f_start + MIN_LINE_DUR

        cursor = f_start
        for w in frag.get("words", []):
            if w["start"] < cursor:
                w["start"] = round(cursor, 3)
            if w["end"] < w["start"] + 0.05:
                w["end"] = round(w["start"] + 0.10, 3)
            # Clamp to audio bounds
            w["start"] = round(min(w["start"], audio_duration - 0.02), 3)
            w["end"] = round(min(w["end"], audio_duration - 0.01), 3)
            cursor = w["end"]

        frag["begin"] = f"{f_start:.3f}"
        frag["end"] = f"{min(max(f_end, cursor), audio_duration):.3f}"
        prev_end = float(frag["end"])


def _smooth_transitions(fragments: List[dict]) -> None:
    """Close micro-gaps and resolve micro-overlaps between consecutive
    fragments.  Mutates fragments in-place."""
    for i in range(len(fragments) - 1):
        curr = fragments[i]
        nxt = fragments[i + 1]
        c_end = float(curr["end"])
        n_start = float(nxt["begin"])
        gap = n_start - c_end

        if abs(gap) <= SMOOTH_GAP_THRESH:
            # Small gap or small overlap — snap to midpoint
            mid = (c_end + n_start) / 2.0
            curr["end"] = f"{mid:.3f}"
            nxt["begin"] = f"{mid:.3f}"

            # Adjust trailing word of current fragment
            words_c = curr.get("words", [])
            if words_c and words_c[-1]["end"] > mid:
                words_c[-1]["end"] = round(mid, 3)
            # Adjust leading word of next fragment
            words_n = nxt.get("words", [])
            if words_n and words_n[0]["start"] < mid:
                words_n[0]["start"] = round(mid, 3)


# ===========================================================================
# PIPELINE — Combine all layers
# ===========================================================================

def _map_asr_words_to_lyrics(
    asr_words: List[dict],
    lyric_lines: List[str],
    audio_duration: float,
    audio_path: Optional[str] = None,
) -> List[dict]:
    """
    1. Flatten lyrics → word list; run global hybrid DP.
    2. Per line: compute fragment bounds from matched anchors.
    3. Interpolate unmatched words via cubic spline.
    4. Enforce monotonicity.
    """
    # Flatten user lyrics
    user_words: List[str] = []
    word_to_line: List[int] = []
    for li, line in enumerate(lyric_lines):
        for tok in _tokenize(line):
            user_words.append(tok)
            word_to_line.append(li)

    n = len(user_words)
    if n == 0:
        return []

    # Global hybrid DP
    mapping = _hybrid_match(user_words, asr_words) if asr_words else {}
    match_pct = len(mapping) / n * 100 if n > 0 else 0
    print(f"[hybrid] Matched {len(mapping)}/{n} words ({match_pct:.0f}%)")

    # Assign timestamps
    t_starts: List[Optional[float]] = [None] * n
    t_ends: List[Optional[float]] = [None] * n
    matched_flags: List[bool] = [False] * n
    match_types: List[str] = ["unmatched"] * n

    for ui, ai in mapping.items():
        t_starts[ui] = float(asr_words[ai].get("start", 0.0))
        t_ends[ui] = float(asr_words[ai].get("end", 0.0))
        matched_flags[ui] = True
        un = _norm_word(user_words[ui])
        an = _norm_word(asr_words[ai].get("word", ""))
        match_types[ui] = "exact" if un == an else "fuzzy"

    # Group by line
    line_starts: List[List[Optional[float]]] = [[] for _ in lyric_lines]
    line_ends: List[List[Optional[float]]] = [[] for _ in lyric_lines]
    line_flags: List[List[bool]] = [[] for _ in lyric_lines]
    line_toks: List[List[str]] = [[] for _ in lyric_lines]
    line_mtypes: List[List[str]] = [[] for _ in lyric_lines]

    for wi, li in enumerate(word_to_line):
        line_starts[li].append(t_starts[wi])
        line_ends[li].append(t_ends[wi])
        line_flags[li].append(matched_flags[wi])
        line_toks[li].append(user_words[wi])
        line_mtypes[li].append(match_types[wi])

    # Tag background vocals
    tags = _tag_background_vocals(lyric_lines)

    # Build fragments
    fragments: List[dict] = []
    prev_end = 0.0

    for li, line in enumerate(lyric_lines):
        toks = line_toks[li]
        if not toks:
            continue

        ms = line_starts[li]
        me = line_ends[li]
        mf = line_flags[li]

        # Find next matched start for bound computation
        next_start: Optional[float] = None
        for lj in range(li + 1, len(lyric_lines)):
            for ts, flag in zip(line_starts[lj], line_flags[lj]):
                if flag and ts is not None:
                    next_start = ts
                    break
            if next_start is not None:
                break

        f_begin, f_end = _compute_fragment_bounds(
            ms, me, prev_end, next_start, audio_duration, len(toks)
        )

        # Collect anchor points for cubic spline interpolation
        anchor_times: List[float] = []
        anchor_indices: List[int] = []
        for wi, (ts, flag) in enumerate(zip(ms, mf)):
            if flag and ts is not None:
                anchor_times.append(ts)
                anchor_indices.append(wi)

        # Interpolate timestamps for all words
        interp_starts = _cubic_spline_interpolate(
            anchor_times, anchor_indices, len(toks), f_begin, f_end
        )

        # Build word objects
        word_objs: List[dict] = []
        for wi, tok in enumerate(toks):
            if mf[wi] and ms[wi] is not None:
                w_start = max(f_begin, ms[wi])
                w_end = max(w_start + 0.05, me[wi] if me[wi] is not None else w_start + TYPICAL_WORD_DUR)
            else:
                w_start = interp_starts[wi]
                # End = next word's start or fragment end
                if wi + 1 < len(toks):
                    w_end = interp_starts[wi + 1]
                else:
                    w_end = f_end
                w_end = max(w_start + 0.05, min(w_end, f_end))

            word_objs.append({
                "text": tok,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "confidence": 1.0 if mf[wi] else 0.5,
                "matched_by": line_mtypes[li][wi],
            })

        # Recompute fragment bounds from word extents
        f_begin = min(w["start"] for w in word_objs)
        f_end = max(w["end"] for w in word_objs)
        f_begin = max(prev_end + 0.01, f_begin)
        f_end = max(f_begin + MIN_LINE_DUR, f_end)

        fragments.append({
            "begin": f"{f_begin:.3f}",
            "end": f"{f_end:.3f}",
            "lines": [line],
            "type": tags[li],
            "words": [
                {"text": w["text"], "start": round(w["start"], 3), "end": round(w["end"], 3),
                 "confidence": w.get("confidence", 0.5), "matched_by": w.get("matched_by", "unmatched")}
                for w in word_objs
            ],
        })
        prev_end = f_end

    # Global quality passes
    _enforce_global_monotonicity(fragments, audio_duration)
    _smooth_transitions(fragments)
    return fragments


# --- Heuristic fallback (no ML) ---

def _heuristic_align(
    audio_path: str,
    lyrics_path: str,
    output_json: str,
    language: str = "en",
) -> str:
    """Energy-based alignment without any ML model.
    Distributes lines uniformly across audio duration, then refines
    word boundaries using audio energy minima."""
    try:
        dur = float(get_duration(audio_path) or 0.0)
    except Exception:
        dur = 0.0

    with open(lyrics_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if dur <= 0.0:
        dur = max(1.0, len(lines) * 2.0)

    tags = _tag_background_vocals(lines)
    step = dur / max(1, len(lines))
    fragments: List[dict] = []
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
                {"text": t, "start": round(b + j * w_step, 3),
                 "end": round(b + (j + 1) * w_step, 3)}
                for j, t in enumerate(toks)
            ]
        fragments.append({
            "begin": f"{b:.3f}",
            "end": f"{e:.3f}",
            "lines": [txt],
            "type": tags[i],
            "words": [
                {"text": w["text"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
                for w in words
            ],
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"fragments": fragments}, f, indent=2, ensure_ascii=False)
    return output_json


def _ensure_wav(audio_path: str) -> str:
    """Convert to WAV if needed; return path to WAV file."""
    audio_path = os.path.abspath(audio_path)
    if os.path.splitext(audio_path)[1].lower() == ".wav":
        return audio_path
    wav = os.path.splitext(audio_path)[0] + "_hybrid.wav"
    try:
        if convert_to_wav(audio_path, wav):
            return os.path.abspath(wav)
    except Exception as e:
        print(f"[hybrid] WAV conversion failed: {e}")
    return audio_path


# ===========================================================================
# PUBLIC API
# ===========================================================================

def align(
    audio_path: str,
    lyrics_path: str,
    output_json: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Align audio with lyrics using the 5-layer hybrid engine.

    Pipeline:
      1. Convert to WAV, read lyrics
      2. Run WhisperX ASR (if available)
      3. Hybrid DP match ASR words to user lyrics
      4. Cubic spline interpolation for unmatched words
      5. Detect instrumental gaps → insert silent blocks
      6. Enforce monotonicity + smooth transitions
      7. Write JSON

    Falls back to heuristic (energy-based) alignment if WhisperX is unavailable.
    """
    if output_json is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_json = os.path.join(ALIGN_DIR, f"{base}_alignment.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

    lang = _normalize_language(language)
    wav_path = _ensure_wav(audio_path)
    print(f"[hybrid] Audio:    {wav_path}")
    print(f"[hybrid] Lyrics:   {lyrics_path}")
    print(f"[hybrid] Output:   {output_json}")
    print(f"[hybrid] Language: {lang}")

    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyric_lines = [ln.strip() for ln in f if ln.strip()]

    # Determine audio duration
    audio_duration = 0.0
    try:
        audio_duration = float(get_duration(wav_path) or 0.0)
    except Exception:
        pass

    # ----- Layer 2: Transcription -----
    asr_words: List[dict] = []
    try:
        import whisperx
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[hybrid] WhisperX device: {device}, compute: {compute_type}")

        model = whisperx.load_model("medium", device=device, compute_type=compute_type)
        audio = whisperx.load_audio(wav_path)

        if audio_duration <= 0:
            audio_duration = len(audio) / 16000.0

        result = model.transcribe(audio, batch_size=4, language=lang)
        detected_lang = result.get("language", lang)

        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )

        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    asr_words.append(w)
        asr_words.sort(key=lambda x: x.get("start", 0.0))
        print(f"[hybrid] Got {len(asr_words)} ASR word timestamps")

    except Exception as e:
        print(f"[hybrid] WhisperX unavailable: {e}. Using heuristic fallback.")
        if audio_duration <= 0:
            audio_duration = max(1.0, len(lyric_lines) * 2.0)
        return _heuristic_align(wav_path, lyrics_path, output_json, language=lang)

    # ----- Layers 2–4: Match, interpolate, build fragments -----
    fragments = _map_asr_words_to_lyrics(asr_words, lyric_lines, audio_duration, wav_path)

    # ----- Layer 1 + 4: Instrumental gap detection -----
    try:
        gaps = _detect_instrumental_gaps(wav_path)
        if gaps:
            fragments = _insert_silent_blocks(fragments, gaps)
            print(f"[hybrid] Inserted {len(gaps)} instrumental gap(s)")
    except Exception:
        pass

    # ----- Layer 5: Final QA -----
    _enforce_global_monotonicity(fragments, audio_duration)
    _smooth_transitions(fragments)

    # Write output
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"fragments": fragments}, f, indent=2, ensure_ascii=False)

    print(f"[hybrid] Done. {len(fragments)} fragments → {output_json}")
    return output_json


# ---------------------------------------------------------------------------
# LRC support
# ---------------------------------------------------------------------------

def _parse_lrc_time(tag: str) -> Optional[float]:
    """Parse a single LRC timestamp tag like [01:23.45] to seconds."""
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
    """Convert an LRC file to the alignment JSON format."""
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
    bg_tags = _tag_background_vocals([e["text"] for e in entries])

    fragments: List[dict] = []
    for idx, e in enumerate(entries):
        s = e["start"]
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
            dur = end - s
            w_step = dur / max(1, len(toks))
            words = [
                {"text": t, "start": round(s + k * w_step, 3), "end": round(s + (k + 1) * w_step, 3)}
                for k, t in enumerate(toks)
            ]
        fragments.append({
            "begin": f"{s:.3f}",
            "end": f"{end:.3f}",
            "lines": [e["text"]],
            "type": bg_tags[idx] if idx < len(bg_tags) else "verse",
            "words": words,
        })

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
    """Parse alignment JSON and return structured entries."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "fragments" not in data:
            print(f"[hybrid] Alignment JSON missing 'fragments' key")
            return []
        result = []
        for i, frag in enumerate(data["fragments"]):
            lines_field = frag.get("lines")
            text = (
                lines_field[0]
                if isinstance(lines_field, list) and lines_field
                else (lines_field or "")
            )
            entry = {
                "index": i,
                "start": float(frag.get("begin", 0)),
                "end": float(frag.get("end", 0)),
                "text": text,
            }
            # Include word-level data if present
            words = frag.get("words")
            if isinstance(words, list):
                entry["words"] = words
            result.append(entry)
        return result
    except Exception as e:
        print(f"[hybrid] Error parsing alignment JSON: {e}")
        return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align audio and lyrics using the 5-layer hybrid engine"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--lyrics", required=True, help="Path to plain-text lyrics file")
    parser.add_argument("--output", default=None, help="Path to save alignment JSON")
    parser.add_argument("--language", default=None, help="Language code (e.g. en, es, fr)")
    args = parser.parse_args()

    try:
        result_path = align(args.audio, args.lyrics, args.output, language=args.language)
        entries = parse_alignment_json(result_path)
        print(f"\nGenerated {len(entries)} fragments")
        for e in entries[:5]:
            print(f"  [{e['start']:.2f}s → {e['end']:.2f}s] {e['text'][:60]}")
        if len(entries) > 5:
            print("  ...")
    except Exception as e:
        print(f"Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
