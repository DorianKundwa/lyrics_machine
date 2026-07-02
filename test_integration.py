"""
test_integration.py — Smoke-tests for the LyricForge x Croonify integration.
Run with: python test_integration.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC  = ROOT / "src"

# Inject paths
import site as _site
for p in [str(SRC), str(ROOT), _site.getusersitepackages()]:
    if p not in sys.path:
        sys.path.insert(0, p)

PASS = 0
FAIL = 0

def check(label, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS]  {label}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL]  {label}")
        print(f"          {e}")
        FAIL += 1

print()
print("=" * 60)
print("  LyricForge x Croonify  --  Integration Tests")
print("=" * 60)
print()

# ── 1. Core dependencies ───────────────────────────────────────────────────────
print("[1] Core dependencies")

check("fastapi importable",  lambda: __import__("fastapi"))
check("uvicorn importable",  lambda: __import__("uvicorn"))
check("PIL importable",      lambda: __import__("PIL"))
check("yaml importable",     lambda: __import__("yaml"))
check("numpy importable",    lambda: __import__("numpy"))
check("scipy importable",    lambda: __import__("scipy"))

print()
print("[2] Croonify engine packages")

check("croonify package",                 lambda: __import__("croonify"))
check("croonify.pipeline.SyncPipeline",  lambda: __import__("croonify.pipeline", fromlist=["SyncPipeline"]))
check("croonify.audio.features",          lambda: __import__("croonify.audio.features", fromlist=["FeatureExtractor"]))
check("croonify.text.normalizer",         lambda: __import__("croonify.text.normalizer", fromlist=["LyricsNormalizer"]))
check("croonify.alignment.viterbi_aligner", lambda: __import__("croonify.alignment.viterbi_aligner", fromlist=["ViterbiAligner"]))
check("croonify.scoring.confidence",      lambda: __import__("croonify.scoring.confidence", fromlist=["ConfidenceScorer"]))
check("croonify.segmentation.lines",      lambda: __import__("croonify.segmentation.lines", fromlist=["LineSegmenter"]))

print()
print("[3] LyricForge components")

check("processor importable",     lambda: __import__("processor"))
check("stem_separator importable", lambda: __import__("stem_separator"))
check("aligners package",         lambda: __import__("aligners"))
check("croonify_bridge importable", lambda: __import__("croonify_bridge"))

print()
print("[4] Bridge adapter")

def test_bridge_adapter():
    from croonify_bridge import CroonifyAligner, parse_alignment_json
    a = CroonifyAligner(model_size="tiny", device="cpu", aligner="viterbi")
    assert a.model_size == "tiny"
    assert a.aligner == "viterbi"

def test_parse_alignment_new_format():
    import json, tempfile
    from croonify_bridge import parse_alignment_json
    data = {"segments": [{"index":0,"start":0.0,"end":2.0,"text":"hello","words":[]}], "metadata":{}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        fname = f.name
    segs = parse_alignment_json(fname)
    assert len(segs) == 1
    assert segs[0]["text"] == "hello"
    os.unlink(fname)

def test_parse_alignment_legacy_format():
    import json, tempfile
    from croonify_bridge import parse_alignment_json
    data = [{"start":0.0,"end":2.0,"text":"world","words":[]}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        fname = f.name
    segs = parse_alignment_json(fname)
    assert len(segs) == 1
    assert segs[0]["text"] == "world"
    os.unlink(fname)

check("CroonifyAligner instantiation", test_bridge_adapter)
check("parse_alignment_json (new Croonify format)", test_parse_alignment_new_format)
check("parse_alignment_json (legacy list format)", test_parse_alignment_legacy_format)

print()
print("[5] LyricsNormalizer")

def test_normalizer():
    from croonify.text.normalizer import LyricsNormalizer
    n = LyricsNormalizer()
    words = n.normalize_lyrics("Don't stop the music!")
    assert len(words) > 0
    assert all(isinstance(w, str) for w in words if w)

def test_normalizer_contractions():
    from croonify.text.normalizer import LyricsNormalizer
    n = LyricsNormalizer()
    words = n.normalize_lyrics("I can't won't shouldn't")
    flat = " ".join(w for w in words if w)
    assert "cannot" in flat or "can" in flat

check("normalize_lyrics basic", test_normalizer)
check("expand contractions",    test_normalizer_contractions)

print()
print("[6] LineSegmenter")

def test_line_segmenter():
    from croonify.segmentation.lines import LineSegmenter
    import numpy as np
    cfg = {"max_words": 8, "max_duration_s": 4.0, "min_gap_s": 0.25, "beat_snap": False}
    seg = LineSegmenter(cfg)
    words = [
        {"start": 0.0, "end": 0.5, "text": "Hello", "score": 0.9},
        {"start": 0.5, "end": 1.0, "text": "world", "score": 0.9},
        {"start": 4.0, "end": 4.5, "text": "foo",   "score": 0.8},
    ]
    lines = seg.segment(words)
    assert len(lines) >= 1
    assert all("start" in l and "end" in l and "text" in l for l in lines)

check("LineSegmenter.segment()", test_line_segmenter)

print()
print("[7] ConfidenceScorer")

def test_confidence_scorer():
    from croonify.scoring.confidence import ConfidenceScorer
    from croonify.audio.features import AudioFeatures
    import numpy as np
    sr = 16000
    dur = 2.0
    n_samples = int(sr * dur)
    hop = 160
    n_frames = n_samples // hop
    feats = AudioFeatures(
        sample_rate    = sr,
        duration       = dur,
        waveform       = np.zeros(n_samples),
        rms_energy     = np.ones(n_frames) * 0.5,
        vad_mask       = np.ones(n_frames, dtype=bool),
        beat_frames    = np.array([10, 20, 30]),
        beat_times     = np.array([0.16, 0.32, 0.48]),
        mel_spectrogram= np.zeros((80, n_frames)),
    )
    words = [{"start":0.0,"end":1.0,"text":"hello","score":0.8}]
    scorer = ConfidenceScorer()
    result = scorer.score_words(words, feats)
    assert len(result) == 1
    assert "score" in result[0]
    assert 0.0 <= result[0]["score"] <= 1.0

check("ConfidenceScorer.score_words()", test_confidence_scorer)

print()
print("[8] Port finder")

def test_port_finder():
    from launch import find_free_port
    port = find_free_port(8000)
    assert isinstance(port, int)
    assert 8000 <= port < 8100

def test_port_finder_skips_busy():
    import socket
    from launch import find_free_port
    # Occupy port 8000 by actually listening on it
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", 8000))
            s.listen(1)   # must listen so connect_ex sees it as in use
            port = find_free_port(8000)
            assert port != 8000, f"Expected port != 8000, got {port}"
            assert port > 8000
        except OSError:
            # 8000 already busy — verify finder still returns a valid port
            port = find_free_port(8000)
            assert isinstance(port, int)

check("find_free_port() returns valid port",  test_port_finder)
check("find_free_port() skips busy ports",    test_port_finder_skips_busy)

print()
print("[9] FastAPI app startup (dry-run import)")

def test_lyricforge_app():
    import app as lf_app
    assert hasattr(lf_app, "app"), "app.py must export FastAPI 'app' instance"
    from fastapi import FastAPI
    assert isinstance(lf_app.app, FastAPI)

def test_croonify_api():
    from croonify.api.server import app as croonify_app
    from fastapi import FastAPI
    assert isinstance(croonify_app, FastAPI)

check("LyricForge app.py exports FastAPI app",  test_lyricforge_app)
check("Croonify api.server exports FastAPI app", test_croonify_api)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
total = PASS + FAIL
print(f"  Results: {PASS}/{total} passed", end="")
if FAIL:
    print(f"  ({FAIL} FAILED)")
else:
    print("  -- ALL PASS")
print("=" * 60)
print()

sys.exit(0 if FAIL == 0 else 1)
