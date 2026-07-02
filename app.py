"""
Lyrics Video Maker — FastAPI Backend
"""

import asyncio
import json
import shutil
import uuid
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import processor as proc

# ─── Setup ───────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

for d in (UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR):
    d.mkdir(exist_ok=True)

app = FastAPI(title="Lyrics Video Maker", version="1.0.0")

# In-memory job store  {job_id: {...}}
_jobs: dict[str, dict] = {}

# ─── Alignment / Editor ──────────────────────────────────────────────────────

ALIGN_DIR  = Path(__file__).parent / "alignments"
ALIGN_DIR.mkdir(exist_ok=True)
JOBS_STORE = ALIGN_DIR / "_jobs_store.json"

_rerender_jobs: dict[str, dict] = {}


def _load_jobs_store() -> None:
    """Restore completed job metadata from disk on server restart."""
    if not JOBS_STORE.exists():
        return
    try:
        stored = json.loads(JOBS_STORE.read_text(encoding="utf-8"))
        for jid, data in stored.items():
            if jid not in _jobs:
                _jobs[jid] = data
    except Exception:
        pass


_load_jobs_store()


def _save_jobs_store() -> None:
    try:
        complete = {k: v for k, v in _jobs.items() if v.get("status") == "complete"}
        JOBS_STORE.write_text(
            json.dumps(complete, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        pass


def _srt_time(ts: str) -> float:
    """Parse SRT timestamp 'HH:MM:SS,mmm' to seconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt(srt_path: Path) -> list:
    """Parse an SRT file into a list of Lyric Editor segment dicts."""
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    segments = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)", lines[1])
        if not m:
            continue
        segments.append({
            "index":    len(segments),
            "start":    round(_srt_time(m.group(1)), 3),
            "end":      round(_srt_time(m.group(2)), 3),
            "text":     " ".join(lines[2:]),
            "y_offset": 0,
            "words":    [],
        })
    return segments


def _save_alignment_for_editor(
    job_id: str,
    audio_path,
    bg_path,
    out_dir: Path,
    title: str,
    artist: str,
    font_name: str,
    font_size: int,
    word_highlight: bool,
    language: str = "en",
    active_color: str = "#FFFFFF",
    upcoming_color: str = "#FF0000",
    sung_color: str = "#FFFFFF",
) -> None:
    """
    Called after a successful render (before cleanup).
    • Parses the SRT written by the processor into editor segments.
    • Copies source audio + background to ALIGN_DIR/<job_id>/ for re-renders.
    • Saves <job_id>_alignment.json consumed by the Lyric Editor.
    """
    srt_path = out_dir / "lyrics.srt"
    if not srt_path.exists():
        return

    segments       = _parse_srt(srt_path)
    total_duration = 0.0
    lv             = out_dir / "lyrics_video.mp4"
    if lv.exists():
        try:
            total_duration = proc.get_media_duration(str(lv))
        except Exception:
            pass

    # Persist source files for future re-renders
    align_job_dir = ALIGN_DIR / job_id
    align_job_dir.mkdir(exist_ok=True)

    # Prefer already-concatenated audio (includes outro) if available
    combined = out_dir / "_combined.m4a"
    if combined.exists():
        audio_dest = align_job_dir / "audio_combined.m4a"
        shutil.copy2(str(combined), str(audio_dest))
    else:
        audio_dest = align_job_dir / f"audio{Path(str(audio_path)).suffix}"
        shutil.copy2(str(audio_path), str(audio_dest))

    bg_dest = align_job_dir / f"bg{Path(str(bg_path)).suffix}"
    shutil.copy2(str(bg_path), str(bg_dest))

    alignment = {
        "job_id":         job_id,
        "segments":       segments,
        "total_duration": total_duration,
        "font_name":      font_name,
        "font_size":      font_size,
        "word_highlight": word_highlight,
        "language":       language,
        "active_color":   active_color,
        "upcoming_color": upcoming_color,
        "sung_color":     sung_color,
        "audio_path":     str(audio_dest),
        "bg_path":        str(bg_dest),
        "out_dir":        str(out_dir),
        "title":          title,
        "artist":         artist,
    }

    align_path = ALIGN_DIR / f"{job_id}_alignment.json"
    align_path.write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    _jobs[job_id]["alignment_path"] = str(align_path)


# ─── Static / Index ──────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(Path("static/index.html"))


@app.get("/editor")
def editor():
    return FileResponse(Path("static/editor.html"))


# ─── Upload & Process ────────────────────────────────────────────────────────

async def _save(upload: UploadFile, dest: Path) -> Path:
    max_size = 500 * 1024 * 1024 # 500 MB
    bytes_read = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 256):
            bytes_read += len(chunk)
            if bytes_read > max_size:
                raise ValueError("File exceeds maximum allowed size (500MB).")
            f.write(chunk)
    return dest


_semaphore = asyncio.Semaphore(2)

def _cleanup_old_outputs():
    now = time.time()
    for base_dir in (OUTPUT_DIR, UPLOAD_DIR):
        if not base_dir.exists():
            continue
        for d in base_dir.iterdir():
            if d.is_dir() and now - d.stat().st_mtime > 3600:
                shutil.rmtree(d, ignore_errors=True)

@app.post("/api/process")
async def start_process(
    background_tasks: BackgroundTasks,
    audio:       UploadFile = File(...),
    background:  Optional[UploadFile] = File(None),
    lyrics:      Optional[UploadFile] = File(None),
    lyrics_text: Optional[str] = Form(None),
    outro:       Optional[UploadFile] = File(None),
    title:       str = Form(""),
    artist:      str = Form(""),
    font_name:   str = Form("Arial"),
    font_size:   int = Form(72),
    bg_color:    str = Form("#000000"),
    stem_engine: str = Form("demucs"),
    word_highlight: bool = Form(True),
    language:    str = Form("en"),
    active_color:   str = Form("#FFFFFF"),
    upcoming_color: str = Form("#FF0000"),
    sung_color:     str = Form("#FFFFFF"),
):
    if artist and title:
        job_id = f"{_safe(artist, 'artist')}_{_safe(title, 'song')}"
    elif artist or title:
        job_id = _safe(artist or title, "job")
    else:
        job_id = str(uuid.uuid4())

    _cleanup_old_outputs()
    job_dir = UPLOAD_DIR / job_id
    out_dir = OUTPUT_DIR / job_id
    
    # Ensure empty directory if reusing name
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
        
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploads
        audio_path  = await _save(audio,      job_dir / _safe(audio.filename,      "audio"))
        if background and background.filename:
            bg_path = await _save(background, job_dir / _safe(background.filename, "background.jpg"))
        else:
            # Create solid color background
            bg_path = job_dir / "background.jpg"
            from PIL import Image
            img = Image.new("RGB", (1920, 1080), bg_color)
            img.save(bg_path, "JPEG")
            
        outro_path  = None
        if outro and outro.filename:
            outro_path = await _save(outro, job_dir / _safe(outro.filename, "outro"))
            
        if lyrics and lyrics.filename:
            lyrics_path = await _save(lyrics, job_dir / _safe(lyrics.filename, "lyrics.txt"))
        elif lyrics_text and lyrics_text.strip():
            lyrics_path = job_dir / "lyrics.txt"
            with open(lyrics_path, "w", encoding="utf-8") as f:
                f.write(lyrics_text)
        else:
            raise ValueError("Lyrics file or text must be provided.")
            
    except ValueError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=400)

    _jobs[job_id] = {"status": "queued", "step": "", "progress": 0, "error": None}

    background_tasks.add_task(
        _run_job, job_id, audio_path, lyrics_path, bg_path, outro_path, out_dir,
        title, artist, font_name, font_size, stem_engine, word_highlight, language,
        active_color, upcoming_color, sung_color,
    )
    return {"job_id": job_id}


# ─── Job Runner ──────────────────────────────────────────────────────────────

def _set(job_id, *, step="", progress=0):
    _jobs[job_id].update(status="running", step=step, progress=progress)


async def _run_job(
    job_id: str,
    audio_path, lyrics_path, bg_path, outro_path,
    out_dir: Path,
    title: str,
    artist: str,
    font_name: str = "Arial",
    font_size: int = 72,
    stem_engine: str = "demucs",
    word_highlight: bool = True,
    language: str = "en",
    active_color: str = "#FFFFFF",
    upcoming_color: str = "#FF0000",
    sung_color: str = "#FFFFFF",
):
    async with _semaphore:
        job = _jobs[job_id]
        loop = asyncio.get_running_loop()

    def run(fn, *args, **kwargs):
        return loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def check_cancelled():
        if _jobs.get(job_id, {}).get("status") == "cancelled":
            raise InterruptedError("Job cancelled by user")

    try:
        check_cancelled()
        # 1 — Thumbnail
        _set(job_id, step="Generating YouTube thumbnail…", progress=5)
        thumb_path = out_dir / "thumbnail.jpg"
        await run(proc.generate_thumbnail, bg_path, thumb_path, title, artist, (1280, 720), font_name)

        check_cancelled()
        # 1.5 — Stem Separation
        _set(job_id, step="Separating stems (this may take a few minutes)…", progress=15)
        from stem_separator import separate_audio
        vocals_path, inst_audio_path = await run(separate_audio, audio_path, stem_engine, str(out_dir))

        check_cancelled()
        # 2 — Instrumental video
        _set(job_id, step="Rendering instrumental video…", progress=35)
        inst_path = out_dir / "instrumental.mp4"
        await run(
            proc.generate_instrumental_video,
            bg_path, inst_audio_path, inst_path, outro_path, title, artist,
        )

        check_cancelled()
        # 3 — Lyrics video
        _set(job_id, step="Rendering lyrics video…", progress=65)
        lv_path = out_dir / "lyrics_video.mp4"
        await run(
            proc.generate_lyrics_video,
            bg_path, audio_path, vocals_path, lyrics_path, lv_path, outro_path, title, artist, font_name, font_size, word_highlight, language,
            active_color, upcoming_color, sung_color,
        )

        job.update(
            status="complete",
            step="All files ready!",
            progress=100,
            files={
                "lyrics_video":  str(lv_path),
                "instrumental":  str(inst_path),
                "thumbnail":     str(thumb_path),
            },
        )
        
        # ── Save alignment data for the Lyric Editor (non-fatal) ─────────
        try:
            _save_alignment_for_editor(
                job_id, audio_path, bg_path, out_dir,
                title, artist, font_name, font_size, word_highlight, language,
                active_color, upcoming_color, sung_color,
            )
            _save_jobs_store()
        except Exception:
            pass

        # Cleanup intermediate files in out_dir
        keep_files = {lv_path.name, inst_path.name, thumb_path.name}
        for f in out_dir.iterdir():
            if f.is_file() and f.name not in keep_files:
                try:
                    f.unlink()
                except Exception:
                    pass

    except InterruptedError:
        job.update(status="error", step="Cancelled", error="Job was cancelled by the user.")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        job.update(status="error", step="Failed", error="An internal error occurred during processing. Please try again.")
    finally:
        job_dir = UPLOAD_DIR / job_id
        shutil.rmtree(job_dir, ignore_errors=True)


# ─── Status & Download ───────────────────────────────────────────────────────

@app.get("/api/jobs")
def get_jobs():
    return _jobs

@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    if job_id in _jobs:
        _jobs[job_id]["status"] = "cancelled"
    return {"success": True}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({"error": "Not found"}, status_code=404)
    j = _jobs[job_id]
    resp = {k: v for k, v in j.items() if k != "files"}
    if j.get("status") == "complete":
        resp["files"] = list(j["files"].keys())
    return resp


@app.get("/api/download/{job_id}/{file_key}")
def download(job_id: str, file_key: str):
    if job_id not in _jobs or _jobs[job_id].get("status") != "complete":
        return JSONResponse({"error": "Not ready"}, status_code=404)
    files = _jobs[job_id].get("files", {})
    if file_key not in files:
        return JSONResponse({"error": "Unknown file"}, status_code=404)
    path = Path(files[file_key])
    media = "video/mp4" if path.suffix == ".mp4" else "image/jpeg"
    return FileResponse(path, media_type=media, filename=path.name)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe(name: Optional[str], fallback: str) -> str:
    if not name:
        return fallback
    # keep only the last part and sanitise
    stem = Path(name).name
    return re.sub(r"[^\w.\-]", "_", stem) or fallback


# ─── Lyric Editor Endpoints ───────────────────────────────────────────────────

@app.get("/alignment/{job_id}")
def get_alignment(job_id: str):
    """Return alignment segments for the Lyric Editor (corrected > original)."""
    corrected = ALIGN_DIR / f"{job_id}_alignment_corrected.json"
    original  = ALIGN_DIR / f"{job_id}_alignment.json"
    path = corrected if corrected.exists() else original
    if not path.exists():
        return JSONResponse({"error": "Alignment not found"}, status_code=404)
    try:
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/alignment_audio/{job_id}")
def get_alignment_audio(job_id: str):
    """Serve the full audio (with vocals) for the editor preview."""
    corrected = ALIGN_DIR / f"{job_id}_alignment_corrected.json"
    original  = ALIGN_DIR / f"{job_id}_alignment.json"
    path = corrected if corrected.exists() else original
    if not path.exists():
        return JSONResponse({"error": "Alignment not found"}, status_code=404)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        audio_path = Path(data.get("audio_path", ""))
        if not audio_path.exists():
            return JSONResponse({"error": "Audio file not found"}, status_code=404)
        return FileResponse(audio_path, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.patch("/alignment/{job_id}")
async def patch_alignment(job_id: str, request: Request):
    """Save editor-corrected segments. Body: {segments: [...], active_color?, upcoming_color?, sung_color?}."""
    original = ALIGN_DIR / f"{job_id}_alignment.json"
    if not original.exists():
        return JSONResponse({"error": "Original alignment not found"}, status_code=404)
    try:
        body     = await request.json()
        segments = body.get("segments", [])
        base     = json.loads(original.read_text(encoding="utf-8"))
        base["segments"] = segments
        # Persist any color overrides sent from the editor
        for key in ("active_color", "upcoming_color", "sung_color"):
            if key in body:
                base[key] = body[key]
        corrected = ALIGN_DIR / f"{job_id}_alignment_corrected.json"
        corrected.write_text(json.dumps(base, indent=2), encoding="utf-8")
        _save_jobs_store()
        return {"saved": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/rerender/{job_id}")
async def start_rerender(job_id: str, background_tasks: BackgroundTasks):
    """Trigger a re-render using the (possibly corrected) alignment."""
    corrected = ALIGN_DIR / f"{job_id}_alignment_corrected.json"
    original  = ALIGN_DIR / f"{job_id}_alignment.json"
    path      = corrected if corrected.exists() else original
    if not path.exists():
        return JSONResponse({"error": "Alignment not found"}, status_code=404)
    try:
        align_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    rerender_id = f"{job_id}_rr"
    _rerender_jobs[rerender_id] = {
        "status": "queued", "step": "Queued", "progress": 0, "error": None,
    }
    background_tasks.add_task(_run_rerender, rerender_id, align_data)
    return {"rerender_id": rerender_id}


@app.get("/rerender_status/{rerender_id}")
def get_rerender_status(rerender_id: str):
    if rerender_id not in _rerender_jobs:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return _rerender_jobs[rerender_id]


@app.get("/api/download_corrected/{rerender_id}")
def download_corrected(rerender_id: str):
    if rerender_id not in _rerender_jobs:
        return JSONResponse({"error": "Not found"}, status_code=404)
    j = _rerender_jobs[rerender_id]
    if j.get("status") != "complete":
        return JSONResponse({"error": "Not ready"}, status_code=404)
    path = Path(j["output_path"])
    if not path.exists():
        return JSONResponse({"error": "File missing"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename="lyrics_video_corrected.mp4")


async def _run_rerender(rerender_id: str, align_data: dict) -> None:
    """Background task: generate corrected ASS then call FFmpeg."""
    job  = _rerender_jobs[rerender_id]
    loop = asyncio.get_running_loop()

    def run(fn, *args, **kwargs):
        return loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    try:
        job.update(status="running", step="Rendering corrected lyrics video…", progress=10)
        segments       = align_data["segments"]
        out_dir        = Path(align_data["out_dir"])
        bg_path        = align_data["bg_path"]
        audio_path     = align_data["audio_path"]
        font_name      = align_data.get("font_name",      "Arial")
        font_size      = align_data.get("font_size",      72)
        active_color   = align_data.get("active_color",   "#FFFFFF")
        upcoming_color = align_data.get("upcoming_color", "#FF0000")
        sung_color     = align_data.get("sung_color",     "#FFFFFF")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "lyrics_video_corrected.mp4"

        job.update(step="Rendering corrected lyrics video…", progress=30)
        await run(
            proc.rerender_lyrics_video,
            bg_path, audio_path, segments, output_path, font_name, font_size,
            active_color, upcoming_color, sung_color,
        )

        job.update(
            status="complete", step="Done!", progress=100,
            download_url=f"/api/download_corrected/{rerender_id}",
            output_path=str(output_path),
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        job.update(status="error", step="Failed", error=str(exc))
