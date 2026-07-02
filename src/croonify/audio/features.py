"""Acoustic feature extraction for the Croonify alignment pipeline.

Extracted features are used by downstream modules:

* ``ProsodyRefiner``   — uses ``rms_energy``, ``vad_mask``, ``waveform``
* ``LineSegmenter``    — uses ``beat_times``
* ``ConfidenceScorer`` — uses ``rms_energy``, ``vad_mask``

All processing is done at a fixed sample-rate of 16 000 Hz (Whisper's native
rate) with a hop-length of 160 samples (10 ms per frame).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SR: int = 16_000          # Whisper native sample rate
HOP_LENGTH: int = 160            # 10 ms per frame @ 16 kHz
FRAME_LENGTH: int = 512          # ~32 ms analysis window
N_MELS: int = 80                 # Mel filter-bank bins (matches Whisper)
VAD_MEDIAN_KERNEL: int = 5       # Frames for VAD smoothing (~50 ms)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatures:
    """Container for all acoustic features extracted from a single audio file.

    Attributes
    ----------
    sample_rate:
        Sample rate of the loaded waveform (always ``TARGET_SR`` = 16 000).
    duration:
        Total duration of the audio in seconds.
    waveform:
        Raw mono waveform as a 1-D float32 array of shape ``(N_samples,)``.
    rms_energy:
        Frame-level RMS energy, shape ``(T,)`` where ``T = ceil(N/hop_length)``.
    vad_mask:
        Boolean array of shape ``(T,)``; ``True`` where speech energy is
        detected (after median smoothing).
    beat_frames:
        Beat positions as frame indices (integer array).
    beat_times:
        Beat positions in seconds (float array).
    mel_spectrogram:
        Log-mel spectrogram of shape ``(N_MELS, T)`` in dB units.
    """

    sample_rate: int
    duration: float
    waveform: np.ndarray = field(repr=False)
    rms_energy: np.ndarray = field(repr=False)
    vad_mask: np.ndarray = field(repr=False)
    beat_frames: np.ndarray = field(repr=False)
    beat_times: np.ndarray = field(repr=False)
    mel_spectrogram: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        # Ensure consistent dtypes
        self.waveform = np.asarray(self.waveform, dtype=np.float32)
        self.rms_energy = np.asarray(self.rms_energy, dtype=np.float32)
        self.vad_mask = np.asarray(self.vad_mask, dtype=bool)
        self.beat_frames = np.asarray(self.beat_frames, dtype=np.int64)
        self.beat_times = np.asarray(self.beat_times, dtype=np.float64)
        self.mel_spectrogram = np.asarray(self.mel_spectrogram, dtype=np.float32)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def frame_to_time(self, frame: int) -> float:
        """Convert a frame index to time in seconds."""
        return float(frame * HOP_LENGTH / self.sample_rate)

    def time_to_frame(self, time_s: float) -> int:
        """Convert time in seconds to the nearest frame index (clamped)."""
        frame = int(round(time_s * self.sample_rate / HOP_LENGTH))
        return max(0, min(frame, len(self.rms_energy) - 1))

    def rms_in_window(self, start_s: float, end_s: float) -> np.ndarray:
        """Return RMS frames within a time window [start_s, end_s]."""
        f0 = self.time_to_frame(start_s)
        f1 = self.time_to_frame(end_s)
        if f1 <= f0:
            f1 = f0 + 1
        return self.rms_energy[f0:f1]

    def vad_coverage(self, start_s: float, end_s: float) -> float:
        """Return fraction of frames in [start_s, end_s] where VAD is True."""
        f0 = self.time_to_frame(start_s)
        f1 = self.time_to_frame(end_s)
        if f1 <= f0:
            return 0.0
        window = self.vad_mask[f0:f1]
        return float(np.mean(window)) if len(window) > 0 else 0.0


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extracts acoustic features from an audio file using librosa.

    All features are computed at ``TARGET_SR`` (16 kHz) with a hop-length of
    ``HOP_LENGTH`` (160 samples = 10 ms) for consistency with the Viterbi
    aligner's frame grid.

    Usage
    -----
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract("path/to/audio.wav")
    >>> print(features.duration, features.beat_times)
    """

    def __init__(self) -> None:
        try:
            import librosa  # noqa: F401
            self._librosa_available = True
        except ImportError:
            self._librosa_available = False
            logger.error(
                "librosa is not installed.  Install with: pip install librosa"
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, audio_path: str) -> AudioFeatures:
        """Load *audio_path* and compute all acoustic features.

        Parameters
        ----------
        audio_path:
            Path to any audio format supported by librosa/soundfile (WAV, MP3,
            FLAC, OGG, M4A …).

        Returns
        -------
        AudioFeatures
        """
        if not self._librosa_available:
            raise ImportError("librosa must be installed to extract audio features.")

        import librosa
        from scipy.signal import medfilt  # type: ignore

        logger.info("Extracting features from: %s", audio_path)

        # --- 1. Load waveform --------------------------------------------------
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        duration = float(len(y) / sr)
        logger.debug("Loaded audio: duration=%.2f s, samples=%d", duration, len(y))

        # --- 2. RMS energy -----------------------------------------------------
        rms = librosa.feature.rms(
            y=y,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )[0]  # shape (T,)

        # --- 3. VAD mask -------------------------------------------------------
        # Dynamic threshold: mean minus half-standard-deviation of log-RMS
        log_rms = np.log1p(rms.astype(np.float64))
        threshold = float(np.mean(log_rms) - 0.5 * np.std(log_rms))
        vad_raw = log_rms > threshold
        # Smooth with a median filter to remove isolated glitches
        vad_mask = medfilt(vad_raw.astype(np.float32), kernel_size=VAD_MEDIAN_KERNEL).astype(bool)

        # --- 4. Beat tracking --------------------------------------------------
        try:
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=HOP_LENGTH, units="frames"
            )
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
            logger.debug("Beat tracking: tempo=%.1f BPM, beats=%d", float(tempo), len(beat_frames))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Beat tracking failed: %s — using empty beat grid.", exc)
            beat_frames = np.array([], dtype=np.int64)
            beat_times = np.array([], dtype=np.float64)

        # --- 5. Mel spectrogram ------------------------------------------------
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=80.0,
            fmax=8000.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # shape (N_MELS, T)

        features = AudioFeatures(
            sample_rate=sr,
            duration=duration,
            waveform=y,
            rms_energy=rms.astype(np.float32),
            vad_mask=vad_mask,
            beat_frames=beat_frames,
            beat_times=beat_times,
            mel_spectrogram=mel_db,
        )
        logger.info(
            "Feature extraction complete: %d frames, vad_active=%.1f%%",
            len(rms),
            100.0 * np.mean(vad_mask),
        )
        return features
