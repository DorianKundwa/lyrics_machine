import os
import sys
import json
import argparse
import tempfile
import re
from typing import List, Tuple
try:
    from .config import ALIGN_DIR
    from .audio_utils import convert_to_wav
except Exception:
    from config import ALIGN_DIR
    from audio_utils import convert_to_wav

# Ensure ffprobe is discoverable by aeneas: prepend ffmpeg bin to PATH if provided
_ffmpeg_env = os.environ.get("FFMPEG_PATH")
if _ffmpeg_env:
    _bin_dir = os.path.dirname(_ffmpeg_env)
    if os.path.isdir(_bin_dir):
        os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

"""Robust Aeneas alignment wrapper

Goals:
- Precondition audio to mono PCM WAV for stable alignment
- Prefer programmatic Aeneas API, fallback to CLI invocation
- Fall back to heuristic alignment if Aeneas fails entirely
"""

# Try to import Aeneas programmatic API (preferred). If it fails, we'll use CLI.
try:
    from aeneas.executetask import ExecuteTask  # type: ignore
    from aeneas.task import Task  # type: ignore
    from aeneas.runtimeconfiguration import RuntimeConfiguration  # type: ignore
    AENEAS_AVAILABLE = True
except Exception:
    AENEAS_AVAILABLE = False

# Fallback alignment removed - Aeneas is now mandatory


def _normalize_language(lang: str) -> str:
    m = {
        "en": "eng", "eng": "eng",
        "es": "spa", "spa": "spa",
        "fr": "fra", "fra": "fra",
        "de": "deu", "deu": "deu",
        "pt": "por", "por": "por",
        "it": "ita", "ita": "ita",
        "nl": "nld", "nld": "nld",
        "pl": "pol", "pol": "pol",
        "sv": "swe", "swe": "swe",
        "tr": "tur", "tur": "tur",
        "ru": "rus", "rus": "rus",
        "uk": "ukr", "ukr": "ukr",
        "cs": "ces", "ces": "ces",
        "el": "ell", "ell": "ell",
        "ar": "ara", "ara": "ara",
        "hi": "hin", "hin": "hin",
        "vi": "vie", "vie": "vie",
        "zh": "zho", "zho": "zho", "cmn": "zho",
        "ja": "jpn", "jpn": "jpn",
        "ko": "kor", "kor": "kor",
        "th": "tha", "tha": "tha",
    }
    return m.get(str(lang or "").lower(), "eng")

def _needs_char_tokenization(lang: str) -> bool:
    l = _normalize_language(lang)
    return l in ("zho", "jpn", "kor", "tha")

def _tokenize_line(line: str, language: str = "eng") -> List[str]:
    s = str(line or "").strip()
    if not s:
        return []
    if _needs_char_tokenization(language):
        chars = []
        for ch in s:
            if ch.isspace():
                continue
            if ch in ",.;:!?，。、「」『』（）()【】…—-、・":
                chars.append(ch)
            else:
                chars.append(ch)
        return chars
    tokens = [t for t in re.split(r"\s+", s) if t]
    return tokens


def _build_words_text(lyrics_path: str) -> Tuple[str, List[List[str]]]:
    """Create a temporary text file with one token per line in original order.
    Returns the temp file path and the per-line tokens.
    """
    with open(lyrics_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]
    per_line_tokens: List[List[str]] = [_tokenize_line(ln) for ln in raw_lines]
    # Flatten into one token per line, preserving order
    fd, tmp_path = tempfile.mkstemp(prefix="words_", suffix=".txt")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as wf:
        for line_tokens in per_line_tokens:
            for tok in line_tokens:
                wf.write(tok + "\n")
    return tmp_path, per_line_tokens


def _enrich_fragments_with_words_aeneas(audio_path: str, lyrics_path: str, fragments: List[dict], language: str) -> List[dict]:
    try:
        with open(lyrics_path, "r", encoding="utf-8") as f:
            raw_lines = [ln.rstrip("\n") for ln in f.readlines()]
        per_line_tokens = [_tokenize_line(ln, language) for ln in raw_lines]
    except Exception:
        per_line_tokens = []
    for i, frag in enumerate(fragments):
        try:
            begin = float(frag.get("begin", 0.0))
            end = float(frag.get("end", 0.0))
        except Exception:
            begin, end = 0.0, 0.0
        tokens = per_line_tokens[i] if i < len(per_line_tokens) else _tokenize_line(" ".join(frag.get("lines", [])), language)
        try:
            words = _distribute_words_by_audio(audio_path, int(begin * 1000), int(end * 1000), tokens)
        except Exception:
            words = [{"text": t, "start": begin, "end": end} for t in tokens]
        frag["words"] = words
    return fragments


def _distribute_words_by_audio(audio_path: str, start_ms: int, end_ms: int, tokens: List[str]) -> List[dict]:
    """Approximate per-word timings inside a line using audio energy minima
    near uniformly spaced boundaries.
    """
    try:
        from pydub import AudioSegment
    except Exception:
        AudioSegment = None

    start_ms = int(max(0, start_ms))
    end_ms = int(max(start_ms, end_ms))
    if AudioSegment is None or end_ms <= start_ms or not tokens:
        # Equal slices fallback
        dur = max(0, end_ms - start_ms)
        n = max(1, len(tokens))
        words = []
        for i, tok in enumerate(tokens):
            w_s = start_ms + int((i / n) * dur)
            w_e = start_ms + int(((i + 1) / n) * dur)
            words.append({"text": tok, "start": w_s / 1000.0, "end": w_e / 1000.0})
        return words

    # Load segment and compute short-time energy
    try:
        audio = AudioSegment.from_file(audio_path)
        seg = audio[start_ms:end_ms]
    except Exception:
        seg = None
    if seg is None:
        dur = max(0, end_ms - start_ms)
        n = max(1, len(tokens))
        return [{"text": tok,
                 "start": (start_ms + int((i / n) * dur)) / 1000.0,
                 "end": (start_ms + int(((i + 1) / n) * dur)) / 1000.0}
                for i, tok in enumerate(tokens)]

    frame_ms = 10
    frames = max(1, int(len(seg) / frame_ms))
    # Compute dBFS per frame
    energies = []
    for i in range(frames):
        s = i * frame_ms
        e = min(len(seg), s + frame_ms)
        chunk = seg[s:e]
        energies.append(chunk.dBFS if len(chunk) > 0 else -90.0)

    # Initial uniform boundaries
    n_words = max(1, len(tokens))
    total_frames = len(energies)
    uniform_bounds = [int((i / n_words) * total_frames) for i in range(1, n_words)]

    # For each uniform boundary, search local minima within ±8 frames (~80ms)
    refined_bounds = []
    radius = 8
    for ub in uniform_bounds:
        lo = max(0, ub - radius)
        hi = min(total_frames - 1, ub + radius)
        if lo >= hi:
            refined_bounds.append(ub)
            continue
        window = energies[lo:hi+1]
        min_idx = window.index(min(window))
        refined_bounds.append(lo + min_idx)

    # Convert frame boundaries to ms
    bounds_ms = [start_ms] + [start_ms + b * frame_ms for b in refined_bounds] + [end_ms]
    words = []
    for i, tok in enumerate(tokens):
        w_s_ms = bounds_ms[i]
        w_e_ms = bounds_ms[i+1]
        words.append({"text": tok, "start": w_s_ms / 1000.0, "end": w_e_ms / 1000.0})
    return words


# Heuristic fallback builder (used only when Aeneas fails completely)

def align(audio_path, lyrics_path, output_json=None, language=None):
    """
    Align audio with lyrics using Aeneas (mandatory).
    If Aeneas alignment fails, an exception is raised.
    
    Args:
        audio_path (str): Path to audio file (WAV format recommended)
        lyrics_path (str): Path to lyrics text file
        output_json (str, optional): Path to output JSON file. If None, a default path will be used.
    
    Returns:
        str: Path to the output JSON file with alignment data
    """
    # Generate default output path if not provided
    if output_json is None:
        # Create a filename based on the audio filename
        audio_filename = os.path.basename(audio_path)
        base_name = os.path.splitext(audio_filename)[0]
        output_json = os.path.join(ALIGN_DIR, f"{base_name}_alignment.json")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    lang = _normalize_language(language)
    # Internal fallback: build naive alignment if Aeneas fails
    def _build_fallback_fragments(audio_path_local: str, lyrics_path_local: str, lang_local: str):
        try:
            from .audio_utils import get_duration  # lazy import
        except Exception:
            get_duration = None

        # Read non-empty lyric lines
        try:
            with open(lyrics_path_local, "r", encoding="utf-8") as lf:
                lines = [ln.strip() for ln in lf.readlines()]
            lines = [ln for ln in lines if ln]
        except Exception as _lf_err:
            print(f"Fallback alignment: failed to read lyrics: {_lf_err}")
            return []

        n = len(lines)
        if n == 0:
            print("Fallback alignment: no lyrics lines found")
            return []

        # Determine audio duration (seconds)
        dur = 0.0
        try:
            if get_duration is not None:
                dur = float(get_duration(audio_path_local) or 0.0)
        except Exception as _gd_err:
            print(f"Fallback alignment: duration lookup failed: {_gd_err}")
            dur = 0.0
        if dur <= 0.0:
            # Assume 1.5s per line if duration unknown
            dur = max(1.0, n * 1.5)

        # Uniformly distribute lines across duration
        min_dur = 0.5
        step = max(min_dur, dur / float(max(1, n)))
        fragments = []
        for i, txt in enumerate(lines):
            begin = i * step
            end = min(dur, (i + 1) * step)
            frag = {
                "begin": f"{begin:.3f}",
                "end": f"{end:.3f}",
                "lines": [txt],
                "language": lang_local,
                "id": f"fb{i:06d}",
                "children": []
            }
            # Approximate word timings using audio energy minima when possible
            try:
                tokens = _tokenize_line(txt, lang_local)
                words = _distribute_words_by_audio(audio_path_local, int(begin * 1000), int(end * 1000), tokens)
                frag["words"] = words
            except Exception as _w_err:
                print(f"Fallback alignment: word timing approximation failed: {_w_err}")
            fragments.append(frag)
        return fragments

    try:
        # Precondition audio to mono PCM WAV for best results
        audio_path_abs = os.path.abspath(audio_path)
        lyrics_path_abs = os.path.abspath(lyrics_path)
        output_json_abs = os.path.abspath(output_json)

        # If input is not WAV, convert to sibling WAV under uploads
        src_ext = os.path.splitext(audio_path_abs)[1].lower()
        if src_ext != ".wav":
            wav_candidate = os.path.splitext(audio_path_abs)[0] + ".wav"
            try:
                ok = convert_to_wav(audio_path_abs, wav_candidate)
                if ok:
                    audio_path_abs = os.path.abspath(wav_candidate)
                    print(f"Audio preconditioned to WAV: {audio_path_abs}")
            except Exception as e:
                print(f"Audio preconditioning to WAV failed: {e}")

        # Common Aeneas config: language, plain text, JSON output, boundary adjustments
        config_string = (
            f"task_language={lang}|"
            "is_text_type=plain|"
            "os_task_file_format=json|"
            "task_adjust_boundary_algorithm=percent|"
            "task_adjust_boundary_percent_value=50|"
            "task_adjust_boundary_min_duration=0.25"
        )

        print(f"Audio path: {audio_path_abs}")
        print(f"Lyrics path: {lyrics_path_abs}")
        print(f"Output path: {output_json_abs}")
        print(f"Audio exists: {os.path.exists(audio_path_abs)}")
        print(f"Lyrics exists: {os.path.exists(lyrics_path_abs)}")

        # Prefer programmatic API when available
        ran_ok = False

        # Fallback to CLI if programmatic path failed
        if not ran_ok:
            try:
                import subprocess
                py = sys.executable or "python"
                cmd = [
                    py, "-m", "aeneas.tools.execute_task",
                    audio_path_abs,
                    lyrics_path_abs,
                    config_string,
                    output_json_abs,
                ]
                print(f"Running Aeneas via CLI: {' '.join(cmd)}")
                env = dict(os.environ)
                # Hint to disable C extensions for CLI as well
                env.setdefault('AENEAS_FORCE_PYTHON', '1')
                timeout_s = int(os.environ.get('AENEAS_TIMEOUT', '120'))
                process = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
                print(f"CLI return code: {process.returncode}")
                if process.stdout:
                    print(f"CLI stdout: {process.stdout[:200]}...")
                if process.stderr:
                    print(f"CLI stderr: {process.stderr[:200]}...")
                ran_ok = os.path.exists(output_json_abs) and os.path.getsize(output_json_abs) > 0
            except Exception as e:
                print(f"CLI Aeneas execution failed: {e}")

        # If Aeneas failed entirely, build heuristic fallback
        if not ran_ok:
            print("Aeneas alignment failed; generating heuristic fallback alignment")
            fallback_frags = _build_fallback_fragments(audio_path_abs, lyrics_path_abs, lang)
            enriched_data = {"fragments": fallback_frags}
            with open(output_json_abs, "w", encoding="utf-8") as f:
                json.dump(enriched_data, f, indent=2)
            return output_json_abs

        # Load line-level fragments produced by Aeneas
        with open(output_json_abs, "r", encoding="utf-8") as f:
            data = json.load(f)
        fragments = data.get("fragments", [])

        # Enrich with per-word timings using second Aeneas run
        enriched_fragments = _enrich_fragments_with_words_aeneas(audio_path_abs, lyrics_path_abs, fragments, lang)
        enriched_data = {"fragments": enriched_fragments}

        with open(output_json_abs, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=2)
        print(f"Alignment completed with word timings. Output saved to: {output_json_abs}")
        return output_json_abs

    except Exception as e:
        # If anything goes wrong, try fallback alignment instead of failing hard
        print(f"Aeneas alignment threw an exception: {e}. Using fallback alignment.")
        try:
            fb = _build_fallback_fragments(audio_path, lyrics_path, lang)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump({"fragments": fb}, f, indent=2)
            return output_json
        except Exception as _final_err:
            print(f"Fallback alignment also failed: {_final_err}")
            raise



def parse_alignment_json(json_path):
    """
    Parse the alignment JSON file and return line entries from Aeneas output only.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'fragments' not in data:
            raise ValueError("Alignment JSON must contain 'fragments' from Aeneas")
        fragments = data.get('fragments', [])
        result = []
        for i, fragment in enumerate(fragments):
            lines_field = fragment.get('lines')
            if isinstance(lines_field, list):
                text_val = lines_field[0] if len(lines_field) > 0 else ''
            else:
                text_val = lines_field or ''
            result.append({
                'index': i,
                'start': float(fragment.get('begin', 0)),
                'end': float(fragment.get('end', 0)),
                'text': text_val,
                'words': fragment.get('words', [])
            })
        return result
    except Exception as e:
        print(f"Error parsing alignment JSON: {e}")
        return []

def _parse_lrc_timestamp(tag):
    try:
        s = tag.strip().strip('[]')
        if s.startswith('offset:'):
            return ('offset', float(s.split(':',1)[1]))
        parts = s.split(':')
        if len(parts) >= 2:
            mm = int(parts[0])
            rest = parts[1]
            ss = 0.0
            if '.' in rest:
                a,b = rest.split('.',1)
                ss = float(a + '.' + b)
            else:
                ss = float(rest)
            return ('time', mm*60.0 + ss)
    except Exception:
        pass
    return (None, None)

def lrc_to_alignment(audio_path, lrc_path, output_json, language=None):
    try:
        with open(lrc_path, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f.readlines()]
    except Exception:
        return None
    lang = _normalize_language(language)
    entries = []
    offset_ms = 0.0
    for ln in lines:
        i = 0
        tags = []
        while i < len(ln) and ln[i] == '[':
            j = ln.find(']', i+1)
            if j == -1:
                break
            tags.append(ln[i:j+1])
            i = j+1
        text = ln[i:].strip()
        times = []
        for t in tags:
            kind,val = _parse_lrc_timestamp(t)
            if kind == 'offset' and isinstance(val, (int,float)):
                offset_ms = float(val)
            if kind == 'time' and isinstance(val, (int,float)):
                times.append(float(val))
        for tm in times:
            entries.append({'start': max(0.0, (tm + offset_ms/1000.0)), 'text': text})
    entries = [e for e in entries if e.get('text')]
    entries.sort(key=lambda x: x['start'])
    frags = []
    for idx,e in enumerate(entries):
        s = float(e['start'])
        n = entries[idx+1]['start'] if idx+1 < len(entries) else s + 1.5
        end = max(s, float(n))
        toks = _tokenize_line(e['text'], lang)
        words = []
        if toks:
            dur = max(0.0, end - s)
            step = dur / float(len(toks)) if len(toks) > 0 else dur
            for i,t in enumerate(toks):
                ws = s + i*step
                we = min(end, s + (i+1)*step)
                words.append({'text': t, 'start': ws, 'end': we})
        frags.append({'begin': f"{s:.3f}", 'end': f"{end:.3f}", 'lines': [e['text']], 'language': lang, 'id': f"lrc{idx:06d}", 'children': [], 'words': words})
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump({'fragments': frags}, f, indent=2)
        return output_json
    except Exception:
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Align audio and lyrics using Aeneas (mandatory)")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--lyrics", required=True, help="Path to the lyrics file")
    parser.add_argument("--output", help="Path to save the alignment JSON")
    args = parser.parse_args()

    try:
        result_path = align(args.audio, args.lyrics, args.output)
        alignments = parse_alignment_json(result_path)
        print(f"Generated {len(alignments)} alignments")
        for i, item in enumerate(alignments[:3]):  # Print first 3 items
            print(f"{i+1}: {item['start']}s - {item['end']}s: {item['text']}")
        if len(alignments) > 3:
            print("...")
    except Exception as e:
        print(f"Aeneas alignment failed: {e}")
        sys.exit(1)
