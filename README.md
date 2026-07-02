# Croonify 🎵

**AI-powered lyrics synchronization engine** — produce word-level timestamps for any song in seconds.

Croonify combines neural vocal separation (Demucs), large-scale speech models (Whisper / WhisperX), and a custom monotonic Viterbi aligner to generate karaoke-ready JSON output that captures every word, beat, and breath of a performance.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Croonify Pipeline                           │
│                                                                     │
│  Audio File  ──►  [ Vocal Separation ]  ──►  Vocals WAV            │
│                        (Demucs)                                     │
│                            │                                        │
│                            ▼                                        │
│               [ Feature Extraction ]                                │
│                 RMS · VAD · Beats · Mel                             │
│                            │                                        │
│  Lyrics Text ─────────────►│                                        │
│  [ Normalizer ]            │                                        │
│  contractions · tokens     ▼                                        │
│                    [ Forced Alignment ]                             │
│                   WhisperX ── fallback ──► Viterbi HMM             │
│                            │                                        │
│                            ▼                                        │
│               [ Prosody Refinement ]                                │
│           vowel-stretch · silence-gap · zero-crossing              │
│                            │                                        │
│                            ▼                                        │
│               [ Confidence Scoring ]                                │
│            alignment · VAD · SNR → composite                       │
│                            │                                        │
│                            ▼                                        │
│               [ Line Segmentation ]                                 │
│           gap detection · word limit · beat snap                   │
│                            │                                        │
│                            ▼                                        │
│                  { SyncResult JSON }                                │
│            words[] · lines[] · metadata{}                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

- **Python 3.9+**
- **FFmpeg** (recommended — required for MP3/M4A input)
  - Windows: `winget install ffmpeg` or download from https://ffmpeg.org
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/croonify/croonify.git
cd croonify

# Install core dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Optional: install WhisperX for best accuracy
pip install whisperx

# Optional: install Demucs for vocal separation
pip install demucs
```

> **Note:** PyTorch installation varies by platform. Visit https://pytorch.org/get-started/locally/ for platform-specific instructions.

---

## Quick Start

### CLI — Align lyrics to audio

```bash
# Basic alignment (WhisperX)
croonify-cli align --audio song.mp3 --lyrics lyrics.txt

# Save result to JSON file
croonify-cli align --audio song.wav --lyrics lyrics.txt --output result.json

# Use Viterbi aligner (no WhisperX needed)
croonify-cli align --audio song.wav --lyrics lyrics.txt --aligner viterbi

# Specify language (skip auto-detection)
croonify-cli align --audio song.wav --lyrics lyrics.txt --language en

# Disable vocal separation (faster)
croonify-cli align --audio song.wav --lyrics lyrics.txt --no-vocal-separation

# Read lyrics from stdin
cat lyrics.txt | croonify-cli align --audio song.wav --lyrics -

# Verbose output (debug logging)
croonify-cli align --audio song.wav --lyrics lyrics.txt -v
```

### Python API

```python
from croonify.pipeline import SyncPipeline

pipeline = SyncPipeline()

result = pipeline.align(
    audio_path="song.wav",
    lyrics_text=open("lyrics.txt").read(),
    language="auto",
    use_vocal_separation=True,
    aligner="whisperx",   # or "viterbi"
)

print(result.to_json())
# {
#   "words": [
#     {"text": "twinkle", "start": 0.24, "end": 0.62, "score": 0.91, ...},
#     ...
#   ],
#   "lines": [
#     {"start": 0.24, "end": 2.18, "text": "twinkle twinkle little star", ...},
#     ...
#   ],
#   "metadata": { "word_count": 48, "line_count": 12, ... }
# }
```

---

## REST API Server

### Start the server

```bash
# Default (localhost:8000)
croonify-cli serve

# Custom host and port
croonify-cli serve --host 0.0.0.0 --port 9000

# With custom config
croonify-cli serve --config config/default.yaml
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/align` | Submit alignment job |
| `GET` | `/api/status/{job_id}` | Poll job status |
| `GET` | `/api/result/{job_id}` | Get full JSON result |
| `GET` | `/api/download/{job_id}` | Download result as `.json` |
| `DELETE` | `/api/job/{job_id}` | Delete job + temp files |
| `GET` | `/health` | Health check |

Interactive API docs: http://localhost:8000/docs

### curl Examples

```bash
# Submit a job
curl -X POST http://localhost:8000/api/align \
  -F "audio=@song.mp3" \
  -F "lyrics=Twinkle twinkle little star" \
  -F "aligner=whisperx" \
  -F "language=auto" \
  -F "use_vocal_separation=true"
# → {"job_id": "abc123...", "status": "queued"}

# Poll status
curl http://localhost:8000/api/status/abc123...
# → {"job_id": "abc123...", "status": "running", "progress": 0.45}

# Get result (when done)
curl http://localhost:8000/api/result/abc123...
# → { "words": [...], "lines": [...], "metadata": {...} }

# Download as file
curl -o result.json http://localhost:8000/api/download/abc123...

# Health check
curl http://localhost:8000/health
# → {"status": "ok", "version": "0.1.0"}
```

---

## Output Format

### Word-level timestamps

```json
{
  "words": [
    {
      "text": "twinkle",
      "start": 0.24,
      "end": 0.62,
      "score": 0.91,
      "emphasized": false,
      "confidence": {
        "alignment": 0.94,
        "vad_coverage": 0.88,
        "snr_estimate": 0.73,
        "composite": 0.91
      }
    }
  ]
}
```

### Line-level grouping

```json
{
  "lines": [
    {
      "start": 0.24,
      "end": 2.18,
      "text": "twinkle twinkle little star",
      "words": [ ... ]
    }
  ]
}
```

### Metadata

```json
{
  "metadata": {
    "aligner_used": "whisperx",
    "model_size": "small",
    "language_detected": "en",
    "audio_duration_s": 45.3,
    "word_count": 48,
    "line_count": 12,
    "low_confidence_count": 2,
    "processing_time_s": 8.7,
    "timing": {
      "vocal_separation_s": 3.2,
      "feature_extraction_s": 0.4,
      "alignment_s": 4.1,
      "prosody_refinement_s": 0.08,
      "confidence_scoring_s": 0.02,
      "line_segmentation_s": 0.01
    }
  }
}
```

---

## Configuration

Edit `config/default.yaml` or pass `--config path/to/config.yaml`:

```yaml
alignment:
  primary: whisperx      # "whisperx" or "viterbi"
  fallback: viterbi      # used when primary fails
  model: small           # Whisper model size: tiny/base/small/medium/large-v2
  language: auto         # ISO-639-1 or "auto"
  device: cpu            # "cpu", "cuda", "mps"

vocal_separation:
  enabled: true          # set false to skip Demucs
  model: htdemucs        # Demucs model name
  fallback_to_original: true

prosody:
  vowel_stretch_threshold: 0.7   # RMS fraction triggering end-extension
  min_silence_ms: 80             # silence gap that triggers gap insertion
  boundary_snap_ms: 20           # ±window for zero-crossing snap
  rms_extend_ratio: 0.15         # max extension as fraction of word duration

line_segmentation:
  max_words: 8           # max words per display line
  max_duration_s: 4.0    # max line duration in seconds
  min_gap_s: 0.25        # silence gap that forces a line break
  beat_snap: false       # snap boundaries to nearest beat

api:
  host: 0.0.0.0
  port: 8000
  max_file_size_mb: 50   # maximum upload size
  job_ttl_s: 3600        # seconds before job auto-expires
```

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=croonify --cov-report=term-missing

# Run only fast unit tests (skip integration)
pytest tests/ -v -k "not integration"
```

---

## Frontend

Place your frontend files in the `frontend/` directory at the project root.
The server will automatically serve them at `http://localhost:8000/`.

```
lyrics_machine/
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
```

---

## Project Structure

```
lyrics_machine/
├── cli.py                          # CLI entry point
├── requirements.txt
├── pyproject.toml
├── config/
│   └── default.yaml                # Default pipeline configuration
├── src/
│   └── croonify/
│       ├── pipeline.py             # SyncPipeline orchestrator
│       ├── audio/
│       │   ├── separator.py        # Demucs vocal separation
│       │   └── features.py         # RMS, VAD, beats, mel spectrogram
│       ├── text/
│       │   └── normalizer.py       # Contraction expansion, tokenization
│       ├── alignment/
│       │   ├── whisperx_aligner.py # WhisperX forced alignment
│       │   └── viterbi_aligner.py  # Custom HMM Viterbi aligner
│       ├── refinement/
│       │   └── prosody.py          # Boundary refinement
│       ├── scoring/
│       │   └── confidence.py       # Composite confidence scoring
│       ├── segmentation/
│       │   └── lines.py            # Word-to-line grouping
│       └── api/
│           └── server.py           # FastAPI REST server
└── tests/
    ├── test_pipeline.py
    └── fixtures/
        └── sample_lyrics.txt
```

---

## Aligner Comparison

| Feature | WhisperX | Viterbi (built-in) |
|---------|----------|-------------------|
| Accuracy | ★★★★★ | ★★★☆☆ |
| Speed (CPU) | ★★★☆☆ | ★★★★★ |
| Extra deps | whisperx, wav2vec2 | None |
| GPU required | No (but recommended) | No |
| Offline | After model download | Fully offline |
| Languages | 90+ | Any (character-level) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — base speech recognition
- [WhisperX](https://github.com/m-bain/whisperX) — forced alignment + diarization
- [Demucs](https://github.com/facebookresearch/demucs) — state-of-the-art source separation
- [librosa](https://librosa.org) — audio analysis toolkit
