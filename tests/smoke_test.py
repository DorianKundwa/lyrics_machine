"""Smoke tests — run with: python tests/smoke_test.py"""
import sys
sys.path.insert(0, "src")

print("=== Croonify Smoke Tests ===\n")

# ---- LyricsNormalizer ----
from croonify.text.normalizer import LyricsNormalizer
norm = LyricsNormalizer()

assert norm.normalize_word("don't") == "do not", "contraction expansion failed"
assert norm.normalize_word("I'm") == "i am", "I'm expansion failed"
assert norm.normalize_word("gonna") == "going to", "gonna expansion failed"
assert norm.normalize_word("can't") == "cannot", "can't expansion failed"
print("OK  LyricsNormalizer.normalize_word()")

lyrics_two_lines = "Hello there\nGoodnight moon"
result = norm.normalize_lyrics(lyrics_two_lines)
assert "hello" in result and "goodnight" in result, f"words missing: {result}"
# blank line between stanzas -> sentinel '' should appear
lyrics_stanzas = "Hello there\n\nGoodnight moon"
result2 = norm.normalize_lyrics(lyrics_stanzas)
assert "" in result2, f"sentinel missing between stanzas: {result2}"
assert "hello" in result2 and "goodnight" in result2
print("OK  LyricsNormalizer.normalize_lyrics()")

lines = norm.get_line_structure("Hello world\nGoodnight moon")
assert len(lines) == 2, f"expected 2 lines, got {len(lines)}"
assert lines[0] == ["hello", "world"]
print("OK  LyricsNormalizer.get_line_structure()")

words = LyricsNormalizer.flat_words("can't stop the feeling")
assert "cannot" in words, f"flat_words missing 'cannot': {words}"
print("OK  LyricsNormalizer.flat_words()")

# ---- LineSegmenter ----
from croonify.segmentation.lines import LineSegmenter
seg = LineSegmenter(config={})
mock_words = [
    {"text": "hello", "start": 0.0, "end": 0.5, "score": 0.9},
    {"text": "world", "start": 0.6, "end": 1.0, "score": 0.85},
    {"text": "foo",   "start": 2.0, "end": 2.4, "score": 0.7},  # gap > 0.25s -> new line
]
seg_lines = seg.segment(mock_words)
assert len(seg_lines) == 2, f"expected 2 lines, got {len(seg_lines)}: {seg_lines}"
assert seg_lines[0]["text"] == "hello world"
assert seg_lines[1]["text"] == "foo"
print("OK  LineSegmenter.segment()")

# Test max_words limit
long_words = [{"text": f"w{i}", "start": float(i)*0.3, "end": float(i)*0.3+0.2, "score": 0.8}
              for i in range(12)]
long_lines = seg.segment(long_words)
assert all(len(l["words"]) <= 8 for l in long_lines), "max_words not respected"
print("OK  LineSegmenter max_words limit")

# Test line_break_after flag
marked_words = [
    {"text": "hello", "start": 0.0, "end": 0.5, "score": 0.9, "line_break_after": True},
    {"text": "world", "start": 0.6, "end": 1.0, "score": 0.85},
]
marked_lines = seg.segment(marked_words)
assert len(marked_lines) == 2, f"line_break_after flag not respected: {marked_lines}"
print("OK  LineSegmenter line_break_after flag")

# ---- SyncPipeline config ----
from croonify.pipeline import SyncPipeline, SyncResult
p = SyncPipeline()
assert p.config["alignment"]["model"] == "small"
assert p.config["line_segmentation"]["max_words"] == 8
assert p.config["alignment"]["device"] == "cpu"
print("OK  SyncPipeline default config loaded")

# Test SyncResult serialization
sr = SyncResult(
    words=[{"text": "hello", "start": 0.0, "end": 0.5, "score": 0.9, "confidence": {"composite": 0.9}}],
    lines=[{"start": 0.0, "end": 0.5, "text": "hello", "words": []}],
    metadata={"word_count": 1}
)
j = sr.to_json()
import json
d = json.loads(j)
assert d["words"][0]["text"] == "hello"
assert "lines" in d
assert "metadata" in d
print("OK  SyncResult.to_json()")

# ---- ViterbiAligner (import + basic structure) ----
from croonify.alignment.viterbi_aligner import ViterbiAligner
va = ViterbiAligner(model_size="small", device="cpu")
# Test character encoding
tokens = va._encode_text(["hello", "world"])
assert "|" in tokens, "word boundary marker missing"
assert tokens[0] == "h"
assert len(tokens) == len("hello") + 1 + len("world")
print("OK  ViterbiAligner._encode_text()")

# Test char embedding shapes
import numpy as np
char_emb = va._get_char_embeddings(tokens)
assert char_emb.shape == (len(tokens), 64), f"bad char emb shape: {char_emb.shape}"
# Check L2-normalized
norms = np.linalg.norm(char_emb, axis=1)
assert np.all(norms > 0), "zero-norm char embeddings found"
print("OK  ViterbiAligner._get_char_embeddings()")

# Test Viterbi decode with tiny random matrix
rng = np.random.default_rng(42)
T, C = 20, len(tokens)
S = rng.uniform(-1, 1, (T, C)).astype(np.float32)
path = va._viterbi_decode(S)
assert len(path) == T, f"path length mismatch: {len(path)} != {T}"
# Verify monotonicity
for i in range(1, len(path)):
    assert path[i] >= path[i-1], f"non-monotonic at t={i}: {path[i-1]} -> {path[i]}"
print("OK  ViterbiAligner._viterbi_decode() — monotonic")

# ---- WhisperX aligner import guard ----
from croonify.alignment.whisperx_aligner import WhisperXAligner, _WHISPERX_AVAILABLE
wa = WhisperXAligner(model_size="small", device="cpu")
print(f"OK  WhisperXAligner instantiated (whisperx available: {_WHISPERX_AVAILABLE})")

print("\n=== All smoke tests PASSED ===")
