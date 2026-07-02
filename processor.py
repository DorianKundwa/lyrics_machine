"""
Lyrics Video Processor
Handles: LRC parsing, ASS subtitle generation, video rendering, thumbnail creation
"""

from __future__ import annotations

import subprocess
import re
import os
import sys
import json
import mimetypes
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

if sys.platform == "win32":
    FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
    FONT_REGULAR = "C:/Windows/Fonts/arial.ttf"
elif sys.platform == "darwin":
    FONT_BOLD = "/Library/Fonts/Arial Bold.ttf"
    FONT_REGULAR = "/Library/Fonts/Arial.ttf"
else:
    FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ─── Helpers ────────────────────────────────────────────────────────────────

def seconds_to_ass(t: float) -> str:
    """Convert float seconds → ASS timestamp  H:MM:SS.cc"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def get_media_duration(path: str | Path) -> float:
    """Return duration of audio/video file in seconds."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, check=True, encoding="utf-8", errors="replace"
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def concat_audio(files: list[Path], output: Path) -> Path:
    """Concatenate a list of audio files using ffmpeg concat filter to handle mixed formats."""
    if not files:
        return output
    if len(files) == 1:
        return files[0]
        
    cmd = ["ffmpeg", "-y"]
    filter_complex = ""
    for i, p in enumerate(files):
        cmd.extend(["-i", str(p)])
        filter_complex += f"[{i}:a]"
    filter_complex += f"concat=n={len(files)}:v=0:a=1[outa]"
    
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[outa]",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output)
    ])
    subprocess.run(cmd, check=True, capture_output=True)
    return output


# ─── Lyrics Parsing ─────────────────────────────────────────────────────────

def parse_lyrics(lyrics_path: str | Path) -> list[tuple[float | None, str]]:
    """
    Parse lyrics file.
    Supports:
      • LRC format  [mm:ss.xx] text
      • SRT-style   timestamp --> timestamp\\ntext
      • Plain text  (no timestamps, auto-timed later)
    Returns list of (timestamp_or_None, text).
    """
    text = Path(lyrics_path).read_text(encoding="utf-8", errors="replace")

    # LRC
    lrc_re = re.compile(r"\[(\d+):(\d+)[.:](\d+)\](.*)")
    matches = lrc_re.findall(text)
    if matches:
        result = []
        for mm, ss, cs, line in matches:
            ts = int(mm) * 60 + int(ss) + int(cs) / 100
            stripped = line.strip()
            if stripped:          # skip blank / metadata lines that are empty
                result.append((ts, stripped))
        if result:
            return sorted(result, key=lambda x: x[0])

    # Plain text – strip blank lines and section headers like [Chorus]
    lines = []
    for l in text.splitlines():
        s = l.strip()
        if not s: continue
        if re.match(r"^\[.*?\]$", s): continue
        lines.append(s)
    return [(None, line) for line in lines]


def assign_timestamps(
    lyrics: list[tuple[float | None, str]],
    total_duration: float,
    start_offset: float = 2.0,
) -> list[dict]:
    """
    Returns list of dicts with start, end, text, words.
    If timestamps already exist they are used directly.
    Otherwise the duration is split evenly with a brief lead-in.
    """
    if lyrics and lyrics[0][0] is not None:
        # already timed
        timed = [(t, txt) for t, txt in lyrics]
    else:
        n = len(lyrics)
        available = total_duration - start_offset
        interval = available / max(n, 1)
        timed = [(start_offset + i * interval, txt) for i, (_, txt) in enumerate(lyrics)]

    result = []
    for i, (start, txt) in enumerate(timed):
        end = timed[i + 1][0] if i + 1 < len(timed) else total_duration
        result.append({"start": start, "end": end, "text": txt, "words": []})
    return result


# ─── ASS Subtitle Generation ────────────────────────────────────────────────




def generate_ass(
    timed_lyrics: list[dict],
    title: str = "",
    artist: str = "",
    total_duration: float = 0,
    font_name: str = "Arial",
    font_size: int = 72,
    word_highlight: bool = True,
    active_color: str = "#FFFFFF",
    upcoming_color: str = "#FF0000",
    sung_color: str = "#FFFFFF",
) -> str:
    """Build an ASS subtitle file string with Karaoke tags."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{_hex_to_ass(active_color)},{_hex_to_ass(upcoming_color)},&H00000000,&HAA000000,-1,0,0,0,100,100,2,0,1,4,3,5,60,60,0,1
Style: Title,{font_name},42,&H00DDDDDD,{_hex_to_ass(upcoming_color)},&H00000000,&H88000000,0,0,0,0,100,100,1,0,1,2,1,8,60,60,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]

    # Removed title / artist text from the video output
    for item in timed_lyrics:
        start = item["start"]
        end = item["end"]
        words = item.get("words", [])
        
        if words and word_highlight:
            karaoke_text = ""
            current_time = start
            for w in words:
                w_start = w["start"]
                w_end = w["end"]
                if w_start > current_time:
                    gap_cs = int((w_start - current_time) * 100)
                    if gap_cs > 0:
                        karaoke_text += f"{{\\k{gap_cs}}}"
                duration_cs = int(max(0, w_end - w_start) * 100)
                word_text = _ass_escape(w["text"])
                karaoke_text += f"{{\\k{duration_cs}}}{word_text} "
                current_time = w_end
            line_text = karaoke_text.strip()
        else:
            line_text = _ass_escape(item.get("text", ""))

        # Apply smooth 150ms fade in and 300ms fade out
        line_text = f"{{\\fad(150,300)}}{line_text}"

        lines.append(
            f"Dialogue: 1,{seconds_to_ass(start)},{seconds_to_ass(end)},"
            f"Default,,0,0,0,,{line_text}"
        )

    return "\n".join(lines)


def _ass_escape(text: str) -> str:
    """Minimal ASS special-char escaping."""
    return text.replace("{", "\\{").replace("}", "\\}")


def _hex_to_ass(hex_color: str) -> str:
    """Convert #RRGGBB (or #RGB) hex color to ASS &H00BBGGRR format."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"&H00{b:02X}{g:02X}{r:02X}"


# ─── Thumbnail Generation ────────────────────────────────────────────────────

def generate_thumbnail(
    bg_path: str | Path,
    output_path: str | Path,
    title: str = "",
    artist: str = "",
    size: tuple[int, int] = (1280, 720),
    font_name: str = "Arial",
) -> None:
    """Create a YouTube thumbnail at 1280×720."""
    img = Image.open(bg_path).convert("RGB")

    # Smart crop to 16:9
    w, h = img.size
    target_ratio = size[0] / size[1]
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    img = img.resize(size, Image.LANCZOS)

    draw = ImageDraw.Draw(img)

    def load_font(path, size):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()

    # Determine font paths
    if font_name == "Edo":
        font_path_bold = str(Path(__file__).parent.resolve() / "edo.ttf")
        font_path_reg = font_path_bold
    else:
        font_path_bold = FONT_BOLD
        font_path_reg = FONT_REGULAR

    cx = size[0] // 2

    # Title (large, bold, centered)
    if title:
        max_width = size[0] - 60
        font_size_title = 280
        font_title = load_font(font_path_bold, font_size_title)
        
        while font_size_title > 40:
            bbox = draw.textbbox((0, 0), title, font=font_title)
            if (bbox[2] - bbox[0]) <= max_width:
                break
            font_size_title -= 10
            font_title = load_font(font_path_bold, font_size_title)

        ty = size[1] // 2 - 60
        
        # Drop shadow
        draw.text((cx + 12, ty + 12), title, font=font_title, fill=(0, 0, 0, 200), anchor="mm")
        # Main text with outline
        draw.text((cx, ty), title, font=font_title, fill=(255, 255, 255, 255), anchor="mm", stroke_width=6, stroke_fill=(0, 0, 0, 255))

    # Artist name (smaller, below title)
    if artist:
        max_width = size[0] - 80
        font_size_artist = 120
        font_artist = load_font(font_path_reg, font_size_artist)
        
        while font_size_artist > 30:
            bbox = draw.textbbox((0, 0), artist, font=font_artist)
            if (bbox[2] - bbox[0]) <= max_width:
                break
            font_size_artist -= 5
            font_artist = load_font(font_path_reg, font_size_artist)

        ay = size[1] // 2 + (font_size_title // 2) + 10
        
        # Drop shadow
        draw.text((cx + 6, ay + 6), artist, font=font_artist, fill=(0, 0, 0, 200), anchor="mm")
        # Main text with outline
        draw.text((cx, ay), artist, font=font_artist, fill=(255, 255, 255, 255), anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0, 255))

    img = img.convert("RGB")
    img.save(str(output_path), "JPEG", quality=95, optimize=True)


# ─── Video Generation ────────────────────────────────────────────────────────

_VIDEO_SCALE = "scale=1920:1080,setsar=1"

_FFMPEG_BASE_IMAGE = [
    "ffmpeg", "-y",
    "-loop", "1",          # still image → loop
]

_FFMPEG_BASE_VIDEO = [
    "ffmpeg", "-y",
    "-stream_loop", "-1",  # video → loop
]

def _get_ffmpeg_base(bg_path: str | Path) -> list[str]:
    mime, _ = mimetypes.guess_type(str(bg_path))
    if mime and mime.startswith('video/'):
        return _FFMPEG_BASE_VIDEO
    return _FFMPEG_BASE_IMAGE

_VIDEO_ENCODE = [
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "stillimage",
    "-crf", "23",
    "-c:a", "aac",
    "-b:a", "192k",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
]


def _build_audio(work_dir: Path, main_audio: Path, outro_audio: Path | None) -> Path:
    if outro_audio:
        combined = work_dir / "_combined.m4a"
        return concat_audio([main_audio, outro_audio], combined)
    return main_audio


def generate_instrumental_video(
    bg_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    outro_path: str | Path | None = None,
    title: str = "",
    artist: str = "",
) -> None:
    """Background image + audio (no lyrics overlay)."""
    work_dir = Path(output_path).parent
    audio = _build_audio(work_dir, Path(audio_path), Path(outro_path) if outro_path else None)

    # Optional title watermark using drawtext with textfile= to avoid special-char escaping issues
    vf = _VIDEO_SCALE
    textfile_path = None
    if title or artist:
        label = f"{artist} \u2014 {title}" if (artist and title) else (title or artist)
        textfile_path = work_dir / "_watermark.txt"
        textfile_path.write_text(label, encoding="utf-8")
        textfile_escaped = _ffmpeg_escape_path(str(textfile_path))
        fontfile_escaped = _ffmpeg_escape_path(FONT_REGULAR)
        vf += (
            f",drawtext=fontfile={fontfile_escaped}:textfile={textfile_escaped}"
            f":fontsize=36:fontcolor=white@0.7:bordercolor=black@0.5:borderw=2"
            f":x=40:y=40"
        )

    cmd = _get_ffmpeg_base(bg_path) + [
        "-i", str(bg_path),
        "-i", str(audio),
        "-vf", vf,
        "-shortest",
    ] + _VIDEO_ENCODE + [str(output_path)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd,
            output=result.stdout,
            stderr=result.stderr
        )


def generate_lyrics_video(
    bg_path: str | Path,
    audio_path: str | Path,
    vocals_path: str | Path,
    lyrics_path: str | Path,
    output_path: str | Path,
    outro_path: str | Path | None = None,
    title: str = "",
    artist: str = "",
    font_name: str = "Arial",
    font_size: int = 72,
    word_highlight: bool = True,
    language: str = "en",
    active_color: str = "#FFFFFF",
    upcoming_color: str = "#FF0000",
    sung_color: str = "#FFFFFF",
) -> None:
    """Background image + synced lyrics subtitles + audio."""
    work_dir = Path(output_path).parent
    audio = _build_audio(work_dir, Path(audio_path), Path(outro_path) if outro_path else None)
    duration = get_media_duration(audio)

    parsed = parse_lyrics(lyrics_path)
    
    needs_alignment = parsed and parsed[0][0] is None
    if needs_alignment:
        print("Un-timed lyrics detected. Running intelligent alignment on vocals...")
        try:
            from aligners import align, parse_alignment_json
            align_json = align(str(vocals_path), str(lyrics_path))
            timed = parse_alignment_json(align_json)
        except Exception as e:
            print(f"Alignment engine failed: {e}. Falling back to even spacing.")
            timed = assign_timestamps(parsed, duration)
    else:
        timed = assign_timestamps(parsed, duration)

    # Prevent the last lyric from staying on screen forever during an outro
    if timed:
        last_item = timed[-1]
        if "words" in last_item and last_item["words"]:
            # If we have word-level timestamps, cap the line end to shortly after the last word
            last_word_end = last_item["words"][-1]["end"]
            new_end = last_word_end + 1.0
            if new_end < last_item["end"]:
                last_item["end"] = new_end
        elif last_item["end"] - last_item["start"] > 8.0:
            # If no word-level timestamps, just cap at 8 seconds max
            last_item["end"] = last_item["start"] + 8.0

    if outro_path:
        main_dur = get_media_duration(audio_path)
        outro_dur = get_media_duration(outro_path)
        timed.append({
            "start": main_dur,
            "end": main_dur + outro_dur,
            "text": "Thanks for watching!",
            "words": []
        })

    # Write SRT
    srt_lines = []
    for i, item in enumerate(timed):
        s = item["start"]
        e = item["end"]
        sh, sm, ss, sms = int(s//3600), int((s%3600)//60), int(s%60), int((s%1)*1000)
        eh, em, es, ems = int(e//3600), int((e%3600)//60), int(e%60), int((e%1)*1000)
        srt_lines.append(f"{i+1}\n{sh:02d}:{sm:02d}:{ss:02d},{sms:03d} --> {eh:02d}:{em:02d}:{es:02d},{ems:03d}\n{item['text']}\n")
    (work_dir / "lyrics.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    # Write LRC
    lrc_lines = []
    for item in timed:
        s = item["start"]
        sm, ss, scs = int(s//60), int(s%60), int((s%1)*100)
        lrc_lines.append(f"[{sm:02d}:{ss:02d}.{scs:02d}]{item['text']}")
    (work_dir / "lyrics.lrc").write_text("\n".join(lrc_lines), encoding="utf-8")

    # Write ASS file
    ass_content = generate_ass(
        timed, title=title, artist=artist, total_duration=duration,
        font_name=font_name, font_size=font_size, word_highlight=word_highlight,
        active_color=active_color, upcoming_color=upcoming_color, sung_color=sung_color,
    )
    ass_path = work_dir / "_lyrics.ass"
    ass_path.write_text(ass_content, encoding="utf-8")

    ass_path_escaped = _ass_escape_path(str(ass_path))
    base_dir_escaped = _ass_escape_path(str(Path(__file__).parent.resolve()))
    vf = f"{_VIDEO_SCALE},subtitles={ass_path_escaped}:fontsdir={base_dir_escaped}"

    cmd = _get_ffmpeg_base(bg_path) + [
        "-i", str(bg_path),
        "-i", str(audio),
        "-vf", vf,
        "-shortest",
    ] + _VIDEO_ENCODE + [str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _ffmpeg_escape(text: str) -> str:
    """Escape text for use inside ffmpeg drawtext text= value."""
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _ffmpeg_escape_path(path: str) -> str:
    """Escape a file path for use in an ffmpeg filter option value (e.g. drawtext fontfile/textfile).
    Converts backslashes to forward slashes, escapes colons (Windows drive letters),
    and wraps the whole path in single quotes to handle spaces in directory names."""
    p = path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return f"'{p}'"


def _ass_escape_path(path: str) -> str:
    # FFmpeg subtitles filter path escaping (Windows drive letters, colons)
    # The entire path must be wrapped in single quotes to handle spaces correctly.
    p = path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    return f"'{p}'"


# ─── Editor Re-render Support ────────────────────────────────────────────────

def generate_ass_with_positions(
    segments: list,
    font_name: str = "Arial",
    font_size: int = 72,
    active_color: str = "#FFFFFF",
    upcoming_color: str = "#FF0000",
    sung_color: str = "#FFFFFF",
) -> str:
    """
    Supports per-segment x_offset and y_offset.
    y_offset: pixels from video centre; positive = up.
    x_offset: pixels from video centre; positive = right.
    Uses explicit \\pos() per dialogue line — no karaoke word-tagging.
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{_hex_to_ass(active_color)},{_hex_to_ass(upcoming_color)},&H00000000,&HAA000000,-1,0,0,0,100,100,2,0,1,4,3,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    cx, cy = 960, 540          # centre of 1920×1080
    lines = [header]

    for seg in segments:
        start    = float(seg.get("start", 0))
        end      = float(seg.get("end",   0))
        x_offset = int(seg.get("x_offset", 0))
        y_offset = int(seg.get("y_offset", 0))
        x_pos    = cx + x_offset   # +x_offset → text moves RIGHT
        y_pos    = cy - y_offset   # +y_offset → text moves UP (lower ASS y value)

        seg_text  = _ass_escape(seg.get("text", ""))
        line_text = f"{{\\fad(150,300)\\an5\\pos({x_pos},{y_pos})}}{seg_text}"
        lines.append(
            f"Dialogue: 1,{seconds_to_ass(start)},{seconds_to_ass(end)},"
            f"Default,,0,0,0,,{line_text}"
        )

    return "\n".join(lines)


def rerender_lyrics_video(
    bg_path,
    audio_path,
    segments: list,
    output_path,
    font_name: str = "Arial",
    font_size: int = 72,
    active_color: str = "#FFFFFF",
    upcoming_color: str = "#FF0000",
    sung_color: str = "#FFFFFF",
) -> None:
    """
    Re-render a lyrics video from Lyric Editor-corrected segments.
    Generates a fresh ASS file with per-segment x/y positioning
    then calls FFmpeg — no ML alignment step required.
    """
    work_dir = Path(output_path).parent
    work_dir.mkdir(parents=True, exist_ok=True)

    ass_content = generate_ass_with_positions(
        segments, font_name=font_name, font_size=font_size,
        active_color=active_color, upcoming_color=upcoming_color, sung_color=sung_color,
    )
    ass_path    = work_dir / "_corrected.ass"
    ass_path.write_text(ass_content, encoding="utf-8")

    ass_escaped      = _ass_escape_path(str(ass_path))
    base_dir_escaped = _ass_escape_path(str(Path(__file__).parent.resolve()))
    vf = f"{_VIDEO_SCALE},subtitles={ass_escaped}:fontsdir={base_dir_escaped}"

    cmd = _get_ffmpeg_base(bg_path) + [
        "-i", str(bg_path),
        "-i", str(audio_path),
        "-vf", vf,
        "-shortest",
    ] + _VIDEO_ENCODE + [str(output_path)]

    subprocess.run(cmd, check=True, capture_output=True)
