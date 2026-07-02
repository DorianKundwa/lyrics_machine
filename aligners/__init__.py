import os
import traceback

def align(audio_path: str, lyrics_path: str, output_json: str = None) -> str:
    """
    Unified alignment function.
    Tries WhisperX -> Aeneas -> Heuristic Fallback.
    Language is auto-detected from the audio/lyrics — no manual selection needed.
    Returns the path to the generated JSON file.
    """
    if output_json is None:
        from .config import ALIGN_DIR
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_json = os.path.join(ALIGN_DIR, f"{base}_alignment.json")

    # 1. Try WhisperX (auto-detects language from audio)
    try:
        import whisperx
        import torch
        from .alignment_whisperx import align as whisperx_align
        print("WhisperX is available. Attempting WhisperX alignment (auto-detecting language)...")
        res = whisperx_align(audio_path, lyrics_path, output_json, language=None)
        if res and os.path.exists(res):
            print("WhisperX alignment succeeded.")
            return res
    except Exception as e:
        print(f"WhisperX alignment skipped/failed: {e}")

    # 2. Try Aeneas (passes None → falls back to 'eng' internally)
    try:
        import aeneas
        from .alignment_aeneas import align as aeneas_align
        print("Aeneas is available. Attempting Aeneas alignment...")
        res = aeneas_align(audio_path, lyrics_path, output_json, language=None)
        if res and os.path.exists(res):
            print("Aeneas alignment succeeded.")
            return res
    except Exception as e:
        print(f"Aeneas alignment skipped/failed: {e}")

    # 3. Fallback to Heuristic (auto-detects language from lyrics text)
    print("Falling back to Heuristic alignment...")
    try:
        from .alignment_whisperx import _heuristic_align
        res = _heuristic_align(audio_path, lyrics_path, output_json, language=None)
        if res and os.path.exists(res):
            print("Heuristic alignment succeeded.")
            return res
    except Exception as e:
        print(f"Heuristic alignment failed: {e}")
        traceback.print_exc()

    raise RuntimeError("All alignment engines failed.")

def parse_alignment_json(json_path: str):
    from .alignment_whisperx import parse_alignment_json as parse_json
    return parse_json(json_path)
