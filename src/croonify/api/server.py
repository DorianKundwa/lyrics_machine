"""FastAPI REST server for Croonify lyrics alignment.

Endpoints
---------
POST   /api/align              Submit an alignment job (audio + lyrics)
GET    /api/status/{job_id}    Poll job status
GET    /api/result/{job_id}    Retrieve full SyncResult JSON
GET    /api/download/{job_id}  Download SyncResult as a .json file
DELETE /api/job/{job_id}       Delete a job and its temp files
GET    /health                 Health check

Jobs are stored in an in-memory dict and cleaned up after ``job_ttl_s``
(configurable, default 1 hour).  Temp files written for job audio are cleaned
up when the job is deleted or expires.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

# Schema of each job entry:
# {
#   "status": str,        # "queued" | "running" | "done" | "error"
#   "progress": float,    # 0.0 – 1.0
#   "result": dict|None,  # SyncResult.to_dict() when done
#   "error": str|None,    # error message when failed
#   "created_at": float,  # time.time()
#   "audio_path": str,    # temp file path for cleanup
#   "result_path": str|None,  # temp JSON file path
# }

jobs: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    config:
        Optional Croonify config dict (passed through to :class:`SyncPipeline`).

    Returns
    -------
    FastAPI
    """
    app = FastAPI(
        title="Croonify Lyrics Alignment API",
        description=(
            "AI-powered lyrics synchronization engine. "
            "Submit audio + lyrics, poll for completion, retrieve word-level timestamps."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS — permissive for development; tighten for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config on app state for access in endpoints
    app.state.pipeline_config = config or {}

    # ------------------------------------------------------------------
    # Startup / shutdown events
    # ------------------------------------------------------------------

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("Croonify API server starting up.")
        # Launch background TTL cleaner
        asyncio.create_task(_job_ttl_cleaner(app.state.pipeline_config))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Croonify API server shutting down.")
        _cleanup_all_jobs()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", tags=["meta"])
    async def health() -> Dict[str, Any]:
        """Return server health and version."""
        return {
            "status": "ok",
            "version": "0.1.0",
            "jobs_active": len([j for j in jobs.values() if j["status"] in ("queued", "running")]),
            "jobs_total": len(jobs),
        }

    @app.post("/api/align", tags=["alignment"])
    async def submit_alignment(
        background_tasks: BackgroundTasks,
        audio: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, …)"),
        lyrics: str = Form(..., description="Raw lyrics text (multi-line)"),
        language: str = Form(default="auto", description="ISO-639-1 language code or 'auto'"),
        use_vocal_separation: bool = Form(default=True, description="Enable Demucs vocal separation"),
        aligner: str = Form(default="whisperx", description="Aligner: 'whisperx' or 'viterbi'"),
    ) -> JSONResponse:
        """Submit a new alignment job.

        Returns a ``job_id`` that can be used to poll status and retrieve results.
        """
        # Validate aligner choice
        if aligner not in ("whisperx", "viterbi"):
            raise HTTPException(status_code=422, detail=f"Invalid aligner '{aligner}'. Choose 'whisperx' or 'viterbi'.")

        if not lyrics.strip():
            raise HTTPException(status_code=422, detail="Lyrics text must not be empty.")

        # File size guard (default 50 MB)
        max_mb = app.state.pipeline_config.get("api", {}).get("max_file_size_mb", 50)
        max_bytes = max_mb * 1024 * 1024
        # Read audio into temp file
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="croonify_audio_")
        os.close(tmp_fd)

        try:
            bytes_written = 0
            async with aiofiles.open(tmp_path, "wb") as f:
                while True:
                    chunk = await audio.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Audio file exceeds maximum size of {max_mb} MB.",
                        )
                    await f.write(chunk)
        except HTTPException:
            _safe_unlink(tmp_path)
            raise
        except Exception as exc:  # pylint: disable=broad-except
            _safe_unlink(tmp_path)
            logger.error("Failed to save uploaded audio: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to save uploaded audio.") from exc

        # Create job record
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "audio_path": tmp_path,
            "result_path": None,
            "request": {
                "language": language,
                "use_vocal_separation": use_vocal_separation,
                "aligner": aligner,
                "original_filename": audio.filename,
            },
        }
        logger.info("Job created: %s (aligner=%s, lang=%s)", job_id, aligner, language)

        background_tasks.add_task(
            run_alignment_job,
            job_id=job_id,
            audio_path=tmp_path,
            lyrics=lyrics,
            language=language,
            use_vocal_separation=use_vocal_separation,
            aligner=aligner,
            pipeline_config=app.state.pipeline_config,
        )

        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "queued"},
        )

    @app.get("/api/status/{job_id}", tags=["alignment"])
    async def get_status(job_id: str) -> Dict[str, Any]:
        """Return the current status of an alignment job.

        Status values:
        - ``queued``  — waiting to start
        - ``running`` — actively processing
        - ``done``    — completed successfully
        - ``error``   — failed (see ``error`` field)
        """
        job = _get_job_or_404(job_id)
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "error": job.get("error"),
            "created_at": job["created_at"],
        }

    @app.get("/api/result/{job_id}", tags=["alignment"])
    async def get_result(job_id: str) -> Dict[str, Any]:
        """Return the full :class:`~croonify.pipeline.SyncResult` for a completed job."""
        job = _get_job_or_404(job_id)
        if job["status"] == "error":
            raise HTTPException(status_code=500, detail=f"Job failed: {job.get('error', 'Unknown error')}")
        if job["status"] != "done":
            raise HTTPException(status_code=202, detail=f"Job is not complete yet (status: {job['status']})")
        return job["result"]

    @app.get("/api/download/{job_id}", tags=["alignment"])
    async def download_result(job_id: str) -> FileResponse:
        """Download the alignment result as a JSON file."""
        job = _get_job_or_404(job_id)
        if job["status"] != "done":
            raise HTTPException(status_code=202, detail=f"Job not done yet (status: {job['status']})")
        result_path = job.get("result_path")
        if not result_path or not Path(result_path).exists():
            raise HTTPException(status_code=500, detail="Result file not found.")
        return FileResponse(
            path=result_path,
            media_type="application/json",
            filename=f"croonify_{job_id[:8]}.json",
        )

    @app.delete("/api/job/{job_id}", tags=["alignment"])
    async def delete_job(job_id: str) -> Dict[str, str]:
        """Delete a job and clean up its temporary files."""
        job = _get_job_or_404(job_id)
        _cleanup_job(job_id, job)
        return {"job_id": job_id, "status": "deleted"}

    # ------------------------------------------------------------------
    # Static frontend
    # ------------------------------------------------------------------
    frontend_dir = Path(__file__).parent.parent.parent.parent / "frontend"
    if frontend_dir.exists() and frontend_dir.is_dir():
        logger.info("Mounting frontend static files from %s", frontend_dir)
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


# ---------------------------------------------------------------------------
# Background alignment task
# ---------------------------------------------------------------------------

async def run_alignment_job(
    job_id: str,
    audio_path: str,
    lyrics: str,
    language: str,
    use_vocal_separation: bool,
    aligner: str,
    pipeline_config: Dict[str, Any],
) -> None:
    """Run the alignment pipeline in a background task and update the job store."""
    if job_id not in jobs:
        logger.warning("Job %s not found when background task started.", job_id)
        return

    jobs[job_id]["status"] = "running"
    jobs[job_id]["progress"] = 0.05
    logger.info("Background task started for job %s", job_id)

    try:
        # Import here to avoid circular import at module load time
        from croonify.pipeline import SyncPipeline

        pipeline = SyncPipeline(config=pipeline_config if pipeline_config else None)
        jobs[job_id]["progress"] = 0.1

        # Run alignment in an executor thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            lambda: pipeline.align(
                audio_path=audio_path,
                lyrics_text=lyrics,
                language=language,
                use_vocal_separation=use_vocal_separation,
                aligner=aligner,
            ),
        )

        jobs[job_id]["progress"] = 0.9

        # Serialize result to a temp file for /download endpoint
        result_dict = result.to_dict()
        tmp_fd, result_path = tempfile.mkstemp(suffix=".json", prefix=f"croonify_result_{job_id[:8]}_")
        os.close(tmp_fd)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        jobs[job_id]["result"] = result_dict
        jobs[job_id]["result_path"] = result_path
        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 1.0
        logger.info("Job %s completed successfully.", job_id)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)
        jobs[job_id]["progress"] = 0.0


# ---------------------------------------------------------------------------
# TTL cleaner
# ---------------------------------------------------------------------------

async def _job_ttl_cleaner(config: Dict[str, Any]) -> None:
    """Periodically remove jobs older than job_ttl_s."""
    ttl_s = config.get("api", {}).get("job_ttl_s", 3600)
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        now = time.time()
        expired = [
            jid for jid, job in list(jobs.items())
            if now - job["created_at"] > ttl_s
        ]
        for jid in expired:
            job = jobs.get(jid)
            if job:
                logger.info("Expiring job %s (age=%.0f s)", jid, now - job["created_at"])
                _cleanup_job(jid, job)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_job_or_404(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


def _cleanup_job(job_id: str, job: Dict[str, Any]) -> None:
    """Remove job from store and delete its temp files."""
    _safe_unlink(job.get("audio_path"))
    _safe_unlink(job.get("result_path"))
    jobs.pop(job_id, None)
    logger.debug("Cleaned up job %s", job_id)


def _cleanup_all_jobs() -> None:
    """Delete all jobs and their temp files (called on shutdown)."""
    for jid, job in list(jobs.items()):
        _cleanup_job(jid, job)


def _safe_unlink(path: Optional[str]) -> None:
    """Delete a file if it exists, silently ignoring errors."""
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:  # pylint: disable=broad-except
            pass


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

# Create default app instance for import by uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
