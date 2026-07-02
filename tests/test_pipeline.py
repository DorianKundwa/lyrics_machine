"""Croonify test suite.

Covers:
- LyricsNormalizer: contractions, punctuation, line structure
- LineSegmenter: gap detection, word limit, duration limit
- ConfidenceScorer: composite formula, low-confidence filtering
- SyncPipeline: config loading, default config
- API: health endpoint, job submission, status polling (via TestClient)

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing from src/ without installing
# ---------------------------------------------------------------------------
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_LYRICS = FIXTURE_DIR / "sample_lyrics.txt"


# ===========================================================================
# Helpers / Fixtures
# ===========================================================================

def _make_word(text: str, start: float, end: float, score: float = 0.8) -> Dict[str, Any]:
    return {"text": text, "start": start, "end": end, "score": score}


def _make_audio_features(
    n_frames: int = 200,
    sr: int = 16000,
    hop: int = 160,
    vad_fraction: float = 0.8,
) -> Any:
    """Build a mock AudioFeatures object with sensible values."""
    from croonify.audio.features import AudioFeatures

    duration = n_frames * hop / sr
    waveform = np.zeros(n_frames * hop, dtype=np.float32)
    # Simple sine wave as mock signal
    t = np.linspace(0, duration, len(waveform))
    waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    rms = np.abs(waveform[: n_frames * hop: hop])[:n_frames]
    rms = rms.astype(np.float32)

    vad_mask = np.zeros(n_frames, dtype=bool)
    active_frames = int(n_frames * vad_fraction)
    vad_mask[:active_frames] = True

    beat_frames = np.array([i * 20 for i in range(n_frames // 20)], dtype=np.int64)
    beat_times = beat_frames * hop / sr

    mel = np.random.randn(80, n_frames).astype(np.float32)

    return AudioFeatures(
        sample_rate=sr,
        duration=duration,
        waveform=waveform,
        rms_energy=rms,
        vad_mask=vad_mask,
        beat_frames=beat_frames,
        beat_times=beat_times,
        mel_spectrogram=mel,
    )


# ===========================================================================
# LyricsNormalizer tests
# ===========================================================================

class TestLyricsNormalizer:
    """Tests for croonify.text.normalizer.LyricsNormalizer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from croonify.text.normalizer import LyricsNormalizer
        self.norm = LyricsNormalizer()

    # --- Contraction expansion ------------------------------------------------

    def test_dont_expands(self):
        result = self.norm.normalize_word("don't")
        assert result == "do not"

    def test_cant_expands(self):
        result = self.norm.normalize_word("can't")
        assert result == "cannot"

    def test_wont_expands(self):
        result = self.norm.normalize_word("won't")
        assert result == "will not"

    def test_im_expands(self):
        result = self.norm.normalize_word("I'm")
        assert result == "i am"

    def test_youre_expands(self):
        result = self.norm.normalize_word("you're")
        assert result == "you are"

    def test_ive_expands(self):
        result = self.norm.normalize_word("I've")
        assert result == "i have"

    def test_ill_expands(self):
        result = self.norm.normalize_word("I'll")
        assert result == "i will"

    def test_id_expands(self):
        result = self.norm.normalize_word("I'd")
        assert result == "i would"

    def test_shouldnt_expands(self):
        result = self.norm.normalize_word("shouldn't")
        assert result == "should not"

    def test_gonna_expands(self):
        result = self.norm.normalize_word("gonna")
        assert result == "going to"

    def test_wanna_expands(self):
        result = self.norm.normalize_word("wanna")
        assert result == "want to"

    def test_aint_expands(self):
        result = self.norm.normalize_word("ain't")
        assert result == "is not"

    def test_lets_expands(self):
        result = self.norm.normalize_word("let's")
        assert result == "let us"

    def test_no_expansion_flag(self):
        result = self.norm.normalize_word("don't", expand_contractions=False)
        assert result == "don't"

    # --- Punctuation removal --------------------------------------------------

    def test_trailing_punctuation_removed(self):
        result = self.norm.normalize_word("hello!")
        assert result == "hello"

    def test_comma_removed(self):
        result = self.norm.normalize_word("world,")
        assert result == "world"

    # --- Case normalization ---------------------------------------------------

    def test_uppercase_lowercased(self):
        result = self.norm.normalize_word("HELLO")
        assert result == "hello"

    def test_mixed_case_lowercased(self):
        result = self.norm.normalize_word("TwInKlE")
        assert result == "twinkle"

    # --- normalize_line -------------------------------------------------------

    def test_normalize_line_basic(self):
        result = self.norm.normalize_line("Hello, World!")
        assert result == ["hello", "world"]

    def test_normalize_line_with_contraction(self):
        result = self.norm.normalize_line("Don't stop")
        assert result == ["do", "not", "stop"]

    def test_normalize_line_empty(self):
        result = self.norm.normalize_line("")
        assert result == []

    def test_normalize_line_whitespace_only(self):
        result = self.norm.normalize_line("   ")
        assert result == []

    # --- normalize_lyrics -----------------------------------------------------

    def test_normalize_lyrics_multiline(self):
        text = "Hello world\nGoodnight moon"
        result = self.norm.normalize_lyrics(text)
        assert "" in result, "Should contain line boundary sentinel"
        assert "hello" in result
        assert "world" in result
        assert "goodnight" in result
        assert "moon" in result

    def test_normalize_lyrics_line_boundary_position(self):
        text = "Hello world\nGoodnight moon"
        result = self.norm.normalize_lyrics(text)
        assert result == ["hello", "world", "", "goodnight", "moon"]

    def test_normalize_lyrics_no_trailing_sentinel(self):
        result = self.norm.normalize_lyrics("Hello\nWorld\n")
        assert result[-1] != "", "Should not end with sentinel"

    def test_normalize_lyrics_consecutive_blank_lines_collapsed(self):
        text = "Hello\n\n\nWorld"
        result = self.norm.normalize_lyrics(text)
        # Should only have one sentinel between hello and world
        sentinels = [i for i, t in enumerate(result) if t == ""]
        assert len(sentinels) == 1

    # --- get_line_structure ---------------------------------------------------

    def test_get_line_structure_basic(self):
        text = "Hello world\nGoodnight moon"
        result = self.norm.get_line_structure(text)
        assert result == [["hello", "world"], ["goodnight", "moon"]]

    def test_get_line_structure_skips_blank_lines(self):
        text = "Hello\n\nWorld"
        result = self.norm.get_line_structure(text)
        assert len(result) == 2
        assert result[0] == ["hello"]
        assert result[1] == ["world"]

    def test_get_line_structure_from_fixture(self):
        lyrics = SAMPLE_LYRICS.read_text(encoding="utf-8")
        structure = self.norm.get_line_structure(lyrics)
        assert len(structure) > 0, "Should parse fixture lyrics"
        for line in structure:
            assert len(line) > 0, "No empty lines in structure"

    # --- flat_words -----------------------------------------------------------

    def test_flat_words_no_sentinels(self):
        text = "Hello world\nGoodnight moon"
        result = self.norm.flat_words(text)
        assert "" not in result
        assert len(result) == 4


# ===========================================================================
# LineSegmenter tests
# ===========================================================================

class TestLineSegmenter:
    """Tests for croonify.segmentation.lines.LineSegmenter."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from croonify.segmentation.lines import LineSegmenter
        self.SegClass = LineSegmenter

    def _make_words_uniform(self, n: int, word_duration: float = 0.4, gap: float = 0.1) -> List[Dict]:
        """Create n words with uniform timing."""
        words = []
        t = 0.0
        for i in range(n):
            words.append(_make_word(f"word{i}", t, t + word_duration))
            t += word_duration + gap
        return words

    def test_basic_segmentation(self):
        seg = self.SegClass(config={})
        words = self._make_words_uniform(10)
        lines = seg.segment(words)
        assert len(lines) > 0
        # All words should be in some line
        total_words = sum(len(l["words"]) for l in lines)
        assert total_words == 10

    def test_respects_max_words(self):
        seg = self.SegClass(config={"max_words": 4, "min_gap_s": 999})
        words = self._make_words_uniform(12)
        lines = seg.segment(words)
        for line in lines:
            assert len(line["words"]) <= 4

    def test_gap_triggers_break(self):
        seg = self.SegClass(config={"min_gap_s": 0.5, "max_words": 100})
        words = [
            _make_word("hello", 0.0, 0.4),
            _make_word("world", 0.4, 0.8),
            # Large gap here
            _make_word("foo", 2.0, 2.4),
            _make_word("bar", 2.4, 2.8),
        ]
        lines = seg.segment(words)
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

    def test_max_duration_triggers_break(self):
        seg = self.SegClass(config={"max_duration_s": 1.0, "min_gap_s": 999, "max_words": 100})
        # Words spanning 3 seconds should be split
        words = self._make_words_uniform(10, word_duration=0.3, gap=0.0)
        lines = seg.segment(words)
        for line in lines:
            start = line["start"]
            end = line["end"]
            assert end - start <= 1.05, f"Line duration {end - start:.2f} exceeds max"

    def test_line_break_after_flag(self):
        seg = self.SegClass(config={"max_words": 100, "min_gap_s": 999})
        words = [
            {**_make_word("hello", 0.0, 0.4), "line_break_after": True},
            _make_word("world", 0.5, 0.9),
        ]
        lines = seg.segment(words)
        assert len(lines) == 2

    def test_empty_words_returns_empty(self):
        seg = self.SegClass(config={})
        assert seg.segment([]) == []

    def test_line_dict_has_required_keys(self):
        seg = self.SegClass(config={})
        words = self._make_words_uniform(3)
        lines = seg.segment(words)
        for line in lines:
            assert "start" in line
            assert "end" in line
            assert "text" in line
            assert "words" in line

    def test_beat_snap(self):
        seg = self.SegClass(config={"beat_snap": True, "min_gap_s": 999, "max_words": 100})
        words = self._make_words_uniform(4)
        beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        lines = seg.segment(words, beat_times=beat_times)
        assert len(lines) > 0
        for line in lines:
            assert line["start"] in beat_times or abs(line["start"] - beat_times).min() < 0.26


# ===========================================================================
# ConfidenceScorer tests
# ===========================================================================

class TestConfidenceScorer:
    """Tests for croonify.scoring.confidence.ConfidenceScorer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from croonify.scoring.confidence import ConfidenceScorer
        self.ScorerClass = ConfidenceScorer

    def test_score_adds_confidence_key(self):
        scorer = self.ScorerClass()
        features = _make_audio_features()
        words = [_make_word("hello", 0.0, 0.3)]
        scored = scorer.score_words(words, features)
        assert "confidence" in scored[0]
        assert "composite" in scored[0]["confidence"]

    def test_score_replaces_score_field(self):
        scorer = self.ScorerClass()
        features = _make_audio_features()
        words = [_make_word("hello", 0.0, 0.3, score=0.99)]
        scored = scorer.score_words(words, features)
        # Composite should differ from raw alignment score (0.99) because it blends in VAD/SNR
        assert 0.0 <= scored[0]["score"] <= 1.0

    def test_composite_weights_sum(self):
        """Verify composite is always in [0, 1]."""
        scorer = self.ScorerClass()
        features = _make_audio_features()
        words = [_make_word(f"w{i}", i * 0.5, i * 0.5 + 0.4, score=0.7) for i in range(10)]
        scored = scorer.score_words(words, features)
        for w in scored:
            assert 0.0 <= w["score"] <= 1.0

    def test_get_low_confidence_words(self):
        scorer = self.ScorerClass()
        words = [
            {"text": "good", "start": 0.0, "end": 0.3, "score": 0.8},
            {"text": "bad", "start": 0.5, "end": 0.8, "score": 0.2},
        ]
        low = scorer.get_low_confidence_words(words, threshold=0.5)
        assert len(low) == 1
        assert low[0]["text"] == "bad"

    def test_empty_words_returns_empty(self):
        scorer = self.ScorerClass()
        features = _make_audio_features()
        result = scorer.score_words([], features)
        assert result == []

    def test_score_keys_present(self):
        scorer = self.ScorerClass()
        features = _make_audio_features()
        words = [_make_word("star", 0.0, 0.5)]
        scored = scorer.score_words(words, features)
        conf = scored[0]["confidence"]
        assert "alignment" in conf
        assert "vad_coverage" in conf
        assert "snr_estimate" in conf
        assert "composite" in conf


# ===========================================================================
# SyncPipeline config tests
# ===========================================================================

class TestSyncPipelineConfig:
    """Tests for pipeline config loading."""

    def test_default_config_has_required_keys(self):
        from croonify.pipeline import SyncPipeline
        cfg = SyncPipeline._get_default_config()
        assert "alignment" in cfg
        assert "vocal_separation" in cfg
        assert "prosody" in cfg
        assert "line_segmentation" in cfg
        assert "api" in cfg

    def test_default_alignment_keys(self):
        from croonify.pipeline import SyncPipeline
        cfg = SyncPipeline._get_default_config()
        align = cfg["alignment"]
        assert "primary" in align
        assert "fallback" in align
        assert "model" in align
        assert "device" in align

    def test_load_config_from_file(self, tmp_path):
        import yaml
        from croonify.pipeline import SyncPipeline
        config_data = {"alignment": {"model": "large-v2", "device": "cpu"}}
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        cfg = SyncPipeline._load_config(str(config_file))
        assert cfg["alignment"]["model"] == "large-v2"
        # Non-overridden keys should have defaults
        assert "primary" in cfg["alignment"]

    def test_load_config_missing_file_raises(self):
        from croonify.pipeline import SyncPipeline
        with pytest.raises(FileNotFoundError):
            SyncPipeline._load_config("/nonexistent/config.yaml")

    def test_merge_with_defaults_deep(self):
        from croonify.pipeline import SyncPipeline
        override = {"alignment": {"model": "tiny"}}
        merged = SyncPipeline._merge_with_defaults(override)
        assert merged["alignment"]["model"] == "tiny"
        # Other alignment keys should remain
        assert merged["alignment"]["device"] == "cpu"

    def test_sync_result_to_json(self):
        from croonify.pipeline import SyncResult
        result = SyncResult(
            words=[{"text": "hello", "start": 0.0, "end": 0.5, "score": 0.9}],
            lines=[{"start": 0.0, "end": 0.5, "text": "hello", "words": []}],
            metadata={"word_count": 1},
        )
        j = result.to_json()
        import json
        data = json.loads(j)
        assert "words" in data
        assert "lines" in data
        assert "metadata" in data

    def test_sync_result_to_dict(self):
        from croonify.pipeline import SyncResult
        result = SyncResult(words=[], lines=[], metadata={"test": True})
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["metadata"]["test"] is True


# ===========================================================================
# API tests (FastAPI TestClient)
# ===========================================================================

class TestAPI:
    """HTTP API tests using FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fastapi.testclient import TestClient
        from croonify.api.server import create_app
        self.app = create_app(config={})
        self.client = TestClient(self.app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_align_missing_audio_returns_422(self):
        resp = self.client.post(
            "/api/align",
            data={"lyrics": "Hello world"},
        )
        assert resp.status_code == 422

    def test_align_missing_lyrics_returns_422(self):
        import io
        resp = self.client.post(
            "/api/align",
            files={"audio": ("test.wav", io.BytesIO(b"RIFF" + b"\x00" * 40), "audio/wav")},
        )
        assert resp.status_code == 422

    def test_align_empty_lyrics_returns_422(self):
        import io
        resp = self.client.post(
            "/api/align",
            data={"lyrics": "   ", "aligner": "viterbi"},
            files={"audio": ("test.wav", io.BytesIO(b"RIFF" + b"\x00" * 44), "audio/wav")},
        )
        assert resp.status_code == 422

    def test_align_invalid_aligner_returns_422(self):
        import io
        resp = self.client.post(
            "/api/align",
            data={"lyrics": "Hello world", "aligner": "invalid_aligner"},
            files={"audio": ("test.wav", io.BytesIO(b"RIFF" + b"\x00" * 44), "audio/wav")},
        )
        assert resp.status_code == 422

    def test_status_unknown_job_returns_404(self):
        resp = self.client.get("/api/status/nonexistent-job-id")
        assert resp.status_code == 404

    def test_result_unknown_job_returns_404(self):
        resp = self.client.get("/api/result/nonexistent-job-id")
        assert resp.status_code == 404

    def test_delete_unknown_job_returns_404(self):
        resp = self.client.delete("/api/job/nonexistent-job-id")
        assert resp.status_code == 404

    @patch("croonify.api.server.run_alignment_job")
    def test_align_submit_returns_job_id(self, mock_task):
        """Submitting a valid request returns a job_id without running alignment."""
        import io
        # Create minimal WAV bytes
        wav_bytes = _minimal_wav_bytes()
        resp = self.client.post(
            "/api/align",
            data={"lyrics": "Twinkle twinkle little star", "aligner": "viterbi"},
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    @patch("croonify.api.server.run_alignment_job")
    def test_status_after_submit(self, mock_task):
        """Polling status right after submit should return queued or running."""
        import io
        wav_bytes = _minimal_wav_bytes()
        submit_resp = self.client.post(
            "/api/align",
            data={"lyrics": "Hello world", "aligner": "viterbi"},
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert submit_resp.status_code == 202
        job_id = submit_resp.json()["job_id"]

        status_resp = self.client.get(f"/api/status/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ("queued", "running", "done", "error")

    @patch("croonify.api.server.run_alignment_job")
    def test_delete_job(self, mock_task):
        """Deleting a job removes it from the store."""
        import io
        wav_bytes = _minimal_wav_bytes()
        submit_resp = self.client.post(
            "/api/align",
            data={"lyrics": "Hello world", "aligner": "viterbi"},
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        job_id = submit_resp.json()["job_id"]

        del_resp = self.client.delete(f"/api/job/{job_id}")
        assert del_resp.status_code == 200

        # Now status should return 404
        status_resp = self.client.get(f"/api/status/{job_id}")
        assert status_resp.status_code == 404


# ===========================================================================
# Integration test (requires real audio file)
# ===========================================================================

@pytest.mark.skip(reason="Integration test requires a real audio file at tests/fixtures/sample.wav")
def test_full_pipeline_integration():
    """End-to-end pipeline test with a real audio file."""
    audio_path = FIXTURE_DIR / "sample.wav"
    if not audio_path.exists():
        pytest.skip("No sample audio file found.")

    lyrics = SAMPLE_LYRICS.read_text(encoding="utf-8")

    from croonify.pipeline import SyncPipeline
    pipeline = SyncPipeline()
    result = pipeline.align(
        audio_path=str(audio_path),
        lyrics_text=lyrics,
        use_vocal_separation=False,
        aligner="viterbi",
    )

    assert len(result.words) > 0
    assert len(result.lines) > 0
    assert result.metadata["word_count"] == len(result.words)
    assert result.metadata["line_count"] == len(result.lines)

    for word in result.words:
        assert "text" in word
        assert "start" in word
        assert "end" in word
        assert word["end"] >= word["start"]


# ===========================================================================
# Helpers
# ===========================================================================

def _minimal_wav_bytes(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    """Generate a minimal valid WAV file as bytes."""
    import struct, math
    n_samples = int(sr * duration_s)
    # 440 Hz sine wave
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n_samples)]
    data = struct.pack(f"<{n_samples}h", *samples)
    data_size = len(data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,        # chunk size
        1,         # PCM
        1,         # mono
        sr,        # sample rate
        sr * 2,    # byte rate
        2,         # block align
        16,        # bits per sample
        b"data",
        data_size,
    )
    return header + data
