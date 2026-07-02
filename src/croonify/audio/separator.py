"""Vocal source-separation using Demucs.

This module wraps the Demucs library to isolate vocal stems from mixed audio,
improving downstream alignment accuracy by removing accompaniment energy that
can confuse frame-level forced aligners.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — Demucs may not be installed in all environments
# ---------------------------------------------------------------------------
try:
    from demucs.api import Separator as _DemucsAPISeparator  # type: ignore
    _DEMUCS_API_AVAILABLE = True
except ImportError:
    _DEMUCSAPI_AVAILABLE = False
    _DemucsAPISeparator = None  # type: ignore

try:
    import demucs  # noqa: F401 — just checking availability for subprocess path
    _DEMUCS_AVAILABLE = True
except ImportError:
    _DEMUCS_AVAILABLE = False


class VocalSeparator:
    """Isolates the vocal stem from a mixed audio track using Demucs.

    The class attempts to use the Demucs Python API (``demucs.api.Separator``)
    for in-process separation.  If that is unavailable it falls back to
    launching Demucs as a subprocess (``python -m demucs``).  If both
    approaches fail, it logs a warning and returns the original audio path
    unchanged — the rest of the pipeline can still run on the mixed signal.

    Parameters
    ----------
    model:
        Name of the Demucs model to use.  ``htdemucs`` is the default
        hybrid-transformer model and gives the best vocal quality.
        Other options include ``mdx_extra``, ``mdx_extra_q``, ``hdemucs_mmi``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    fallback_to_original:
        When ``True`` (default) any separation failure returns ``audio_path``
        unchanged.  When ``False`` the exception is re-raised.
    """

    # Demucs writes stems under <output_dir>/<model>/<track_name>/<stem>.wav
    VOCAL_STEM_NAME = "vocals.wav"

    def __init__(
        self,
        model: str = "htdemucs",
        device: str = "cpu",
        fallback_to_original: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.fallback_to_original = fallback_to_original
        self._separator: Optional[object] = None  # lazy-loaded API separator

        if not _DEMUCS_AVAILABLE and not _DEMUCS_API_AVAILABLE:
            logger.warning(
                "Demucs is not installed.  Vocal separation will be skipped. "
                "Install it with: pip install demucs"
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def separate(self, audio_path: str) -> str:
        """Separate vocals from *audio_path* and return path to vocal stem.

        The vocal stem WAV file is written to a temporary directory that lives
        alongside the original file's parent directory.  Callers are
        responsible for cleaning up the returned path when done.

        Parameters
        ----------
        audio_path:
            Absolute or relative path to the input audio file (any format
            supported by Demucs / FFmpeg).

        Returns
        -------
        str
            Path to the isolated vocal stem WAV, or *audio_path* if
            separation could not be performed.
        """
        audio_path = str(Path(audio_path).resolve())
        logger.info("Starting vocal separation: model=%s device=%s", self.model, self.device)

        # Prefer the Python API (avoids subprocess overhead)
        if _DEMUCS_API_AVAILABLE:
            result = self._separate_via_api(audio_path)
        elif _DEMUCS_AVAILABLE:
            result = self._separate_via_subprocess(audio_path)
        else:
            logger.warning("Demucs unavailable — returning original audio unchanged.")
            return audio_path

        if result is None:
            if self.fallback_to_original:
                logger.warning("Vocal separation failed — falling back to original audio.")
                return audio_path
            raise RuntimeError("Vocal separation failed and fallback_to_original is False.")

        logger.info("Vocal separation complete: %s", result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_api_separator(self) -> object:
        """Lazily instantiate the Demucs API Separator (slow to load)."""
        if self._separator is None:
            logger.debug("Loading Demucs model '%s' on device '%s'", self.model, self.device)
            self._separator = _DemucsAPISeparator(
                model=self.model,
                device=self.device,
                progress=False,
            )
        return self._separator

    def _separate_via_api(self, audio_path: str) -> Optional[str]:
        """Run separation using ``demucs.api.Separator``."""
        try:
            import torch  # noqa: F401

            separator = self._get_api_separator()

            # Create a temp output directory
            out_dir = tempfile.mkdtemp(prefix="croonify_sep_")
            logger.debug("Demucs API output directory: %s", out_dir)

            # The API's separate_audio_file writes stems into out_dir
            origin, separated = separator.separate_audio_file(Path(audio_path))  # type: ignore[attr-defined]

            stem_path = Path(out_dir) / self.VOCAL_STEM_NAME

            # Save vocals tensor to wav
            if "vocals" not in separated:
                logger.warning("Demucs did not produce a 'vocals' stem.")
                shutil.rmtree(out_dir, ignore_errors=True)
                return None

            import torchaudio

            vocals_tensor = separated["vocals"]  # shape (C, T)
            sample_rate: int = separator._samplerate  # type: ignore[attr-defined]
            torchaudio.save(str(stem_path), vocals_tensor.cpu(), sample_rate)
            logger.debug("Vocals stem saved to %s", stem_path)
            return str(stem_path)

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Demucs API separation error: %s", exc, exc_info=True)
            return None

    def _separate_via_subprocess(self, audio_path: str) -> Optional[str]:
        """Run separation by invoking ``python -m demucs`` as a subprocess."""
        try:
            out_dir = tempfile.mkdtemp(prefix="croonify_sep_sub_")
            track_name = Path(audio_path).stem
            cmd = [
                "python", "-m", "demucs",
                "--name", self.model,
                "--device", self.device,
                "--out", out_dir,
                audio_path,
            ]
            logger.debug("Demucs subprocess command: %s", " ".join(cmd))
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10-minute hard timeout
            )
            if proc.returncode != 0:
                logger.warning("Demucs subprocess failed (rc=%d): %s", proc.returncode, proc.stderr)
                shutil.rmtree(out_dir, ignore_errors=True)
                return None

            # Demucs writes: <out_dir>/<model>/<track_name>/vocals.wav
            vocals_path = Path(out_dir) / self.model / track_name / self.VOCAL_STEM_NAME
            if not vocals_path.exists():
                # Search recursively in case directory structure differs between versions
                candidates = list(Path(out_dir).rglob("vocals.wav"))
                if not candidates:
                    logger.warning("Could not locate vocals.wav in demucs output dir %s", out_dir)
                    shutil.rmtree(out_dir, ignore_errors=True)
                    return None
                vocals_path = candidates[0]

            logger.debug("Vocals stem (subprocess) at %s", vocals_path)
            return str(vocals_path)

        except subprocess.TimeoutExpired:
            logger.warning("Demucs subprocess timed out.")
            return None
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Demucs subprocess error: %s", exc, exc_info=True)
            return None
