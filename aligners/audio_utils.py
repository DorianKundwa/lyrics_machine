import os
import json
import subprocess
import wave
import shutil
from shutil import which
from pydub import AudioSegment
try:
    from .config import FFMPEG_PATH
except Exception:
    FFMPEG_PATH = None

# Resolve ffmpeg/ffprobe paths robustly on Windows
from typing import Tuple
import threading

active_processes = {}  # tid -> proc

def _run_managed_subprocess(cmd, **kwargs):
    tid = threading.get_ident()
    check = kwargs.pop('check', False)
    
    proc = subprocess.Popen(cmd, **kwargs)
    active_processes[tid] = proc
    try:
        stdout, stderr = proc.communicate()
    finally:
        if tid in active_processes:
            del active_processes[tid]
            
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)

def kill_active_process(tid):
    proc = active_processes.get(tid)
    if proc:
        try:
            proc.kill()
            return True
        except Exception:
            pass
    return False

def _resolve_ffmpeg_tools(config_ffmpeg: str) -> Tuple[str, str]:
    candidates = []
    if config_ffmpeg:
        if config_ffmpeg.lower().endswith('.exe'):
            candidates.append(config_ffmpeg)
        else:
            candidates.extend([config_ffmpeg, config_ffmpeg + '.exe'])
    candidates.extend(['ffmpeg', 'ffmpeg.exe'])

    ffmpeg_path = None
    for cand in candidates:
        found = which(cand) or (os.path.exists(cand) and cand)
        if found:
            ffmpeg_path = found
            break

    if ffmpeg_path is None and os.name == 'nt':
        common_win = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\FFmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\FFmpeg\bin\ffmpeg.exe',
            r'C:\ProgramData\chocolatey\bin\ffmpeg.exe',
        ]
        for p in common_win:
            if os.path.exists(p):
                ffmpeg_path = p
                break

    ffprobe_path = which('ffprobe') or which('ffprobe.exe')
    if not ffprobe_path and ffmpeg_path:
        base = os.path.dirname(ffmpeg_path)
        cand_probe = os.path.join(base, 'ffprobe.exe')
        if os.path.exists(cand_probe):
            ffprobe_path = cand_probe

    return ffmpeg_path or config_ffmpeg, ffprobe_path or 'ffprobe'

_FFMPEG_PATH, _FFPROBE_PATH = _resolve_ffmpeg_tools(FFMPEG_PATH)
print(f"Using FFmpeg at '{_FFMPEG_PATH}', ffprobe at '{_FFPROBE_PATH}'")

AudioSegment.converter = _FFMPEG_PATH
try:
    AudioSegment.ffmpeg = _FFMPEG_PATH
    AudioSegment.ffprobe = _FFPROBE_PATH
except Exception:
    pass

def convert_to_wav(input_path, output_path):
    """
    Convert any audio file to mono 44.1kHz WAV using ffmpeg.
    If input is already WAV, copy it directly.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to output WAV file
    
    Returns:
        bool: True if conversion was successful
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # If already WAV, copy without conversion
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.wav':
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            shutil.copyfile(input_path, output_path)
        print("Input is already WAV; copied without conversion")
        return True
    
    # Check ffmpeg availability (PATH, env override, or common install paths)
    ffmpeg_path = _FFMPEG_PATH
    ffmpeg_available = (which(ffmpeg_path) is not None) or os.path.exists(ffmpeg_path)
    if not ffmpeg_available:
        print(f"FFmpeg not found or not executable at '{ffmpeg_path}'. Attempting pydub fallback...")
        try:
            # Ensure pydub uses the resolved tools
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffmpeg = ffmpeg_path
            AudioSegment.ffprobe = _FFPROBE_PATH

            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(44100)
            audio.export(output_path, format='wav')
            print("Conversion complete via pydub fallback")
            return True
        except Exception as e:
            print(f"Pydub fallback conversion failed: {e}")
            return False
    
    # Build ffmpeg command
    cmd = [
        ffmpeg_path,
        '-i', input_path,
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ar', '44100',          # 44.1kHz sample rate
        '-ac', '1',              # Mono
        '-threads', '0',         # Use all available CPU cores
        '-y',                    # Overwrite output file if it exists
        output_path
    ]
    
    try:
        _run_managed_subprocess(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Conversion complete")
        return True
    except subprocess.CalledProcessError as e:
        try:
            err = e.stderr.decode('utf-8', errors='ignore')
        except Exception:
            err = str(e)
        print(f"FFmpeg conversion error: {err}")
        # Attempt pydub fallback
        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(44100)
            audio.export(output_path, format='wav')
            print("Conversion complete via pydub fallback after ffmpeg error")
            return True
        except Exception as e2:
            print(f"Pydub fallback conversion failed: {e2}")
            return False
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        return False

def normalize_audio(wav_path, output_path=None):
    """
    Normalize audio volume using pydub
    
    Args:
        wav_path (str): Path to input WAV file
        output_path (str, optional): Explicit path to save normalized WAV. If None, appends "_normalized".
    
    Returns:
        str: Path to normalized WAV file
    """
    try:
        audio = AudioSegment.from_file(wav_path)
        
        # Normalize audio
        normalized_audio = audio.normalize()
        
        # Determine output path
        if output_path is None:
            filename, ext = os.path.splitext(wav_path)
            output_path = f"{filename}_normalized{ext}"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        normalized_audio.export(output_path, format="wav")
        
        return output_path
    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return wav_path

def separate_stems(input_path, output_dir=None, prefer='auto'):
    """
    Separate audio into vocals and instrumental stems.
    Tries Spleeter or Demucs if available; falls back to an FFmpeg mid/side approach.

    Args:
        input_path (str): Path to the source audio (any format supported by ffmpeg)
        output_dir (str, optional): Directory where stems will be written. Defaults to sibling folder.
        prefer (str): Hint for preferred engine ('spleeter', 'demucs', 'ffmpeg', or 'auto').

    Returns:
        tuple[str|None, str|None]: (vocals_path, instrumental_path) or (None, None) on failure
    """
    try:
        if not os.path.exists(input_path):
            print(f"Stem separation input not found: {input_path}")
            return None, None

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        root_dir = output_dir or os.path.dirname(input_path)
        os.makedirs(root_dir, exist_ok=True)
        stem_root = os.path.join(root_dir, f"{base_name}_stems")
        os.makedirs(stem_root, exist_ok=True)

        def _try_spleeter() -> tuple:
            sp = which('spleeter') or which('spleeter.exe')
            if not sp:
                return None, None
            try:
                cmd = [sp, 'separate', '-p', 'spleeter:2stems', '-o', stem_root, input_path]
                print(f"Running Spleeter: {' '.join(cmd)}")
                proc = _run_managed_subprocess(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                vocals = os.path.join(stem_root, base_name, 'vocals.wav')
                accomp = os.path.join(stem_root, base_name, 'accompaniment.wav')
                if os.path.exists(vocals) and os.path.exists(accomp):
                    return vocals, accomp, 'spleeter'
            except subprocess.CalledProcessError as e:
                try:
                    emsg = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr or '')
                    if ('cannot import name' in emsg) and ('click.termui' in emsg):
                        print("Spleeter CLI appears broken due to Click/Typer mismatch; skipping Spleeter.")
                    else:
                        print(f"Spleeter separation failed: {e}")
                except Exception:
                    print(f"Spleeter separation failed: {e}")
                return None, None, None
            except Exception as e:
                print(f"Spleeter separation failed: {e}")
                return None, None, None

        def _try_demucs() -> tuple:
            dm = which('demucs') or which('demucs.exe')
            if not dm:
                return None, None
            try:
                cmd = [dm, '--two-stems', 'vocals', '-o', stem_root, input_path]
                print(f"Running Demucs: {' '.join(cmd)}")
                _run_managed_subprocess(cmd, check=True)
                # Search for typical output filenames
                vocals, instrumental = None, None
                for root, dirs, files in os.walk(stem_root):
                    if 'vocals.wav' in files:
                        vocals = os.path.join(root, 'vocals.wav')
                    if 'no_vocals.wav' in files:
                        instrumental = os.path.join(root, 'no_vocals.wav')
                    elif 'accompaniment.wav' in files and instrumental is None:
                        instrumental = os.path.join(root, 'accompaniment.wav')
                if vocals and instrumental:
                    return vocals, instrumental, 'demucs'
            except Exception as e:
                print(f"Demucs separation failed: {e}")
            return None, None, None

        def _ffmpeg_fallback() -> tuple:
            ff = _FFMPEG_PATH or 'ffmpeg'
            # Resolve ffmpeg availability
            ff_ok = (which(ff) is not None) or os.path.exists(ff)
            if not ff_ok:
                print("FFmpeg not available for stem separation fallback.")
                return None, None
            vocals_out = os.path.join(stem_root, f"{base_name}_vocals_est.wav")
            instrumental_out = os.path.join(stem_root, f"{base_name}_instrumental_est.wav")
            ch = 2
            try:
                seg = AudioSegment.from_file(input_path)
                ch = int(getattr(seg, 'channels', 2) or 2)
            except Exception:
                ch = 2
            try:
                if ch >= 2:
                    cmd1 = [ff, '-y', '-i', input_path,
                            '-filter_complex', 'pan=mono|c0=0.5*c0+0.5*c1',
                            '-ar', '44100', '-ac', '1',
                            vocals_out]
                else:
                    cmd1 = [ff, '-y', '-i', input_path,
                            '-ar', '44100', '-ac', '1',
                            vocals_out]
                _run_managed_subprocess(cmd1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"FFmpeg vocals fallback failed: {e}")
                vocals_out = None

            try:
                if ch >= 2:
                    cmd2 = [ff, '-y', '-i', input_path,
                            '-filter_complex', 'pan=stereo|c0=c0-c1|c1=c1-c0',
                            '-ar', '44100',
                            instrumental_out]
                else:
                    cmd2 = [ff, '-y', '-i', input_path,
                            '-ar', '44100', '-ac', '2',
                            instrumental_out]
                _run_managed_subprocess(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"FFmpeg instrumental fallback failed: {e}")
                instrumental_out = None

            return vocals_out, instrumental_out, 'ffmpeg'

        # Selection logic (prefer Demucs first in 'auto' mode)
        vocals, instrumental, engine = None, None, None
        if prefer in ('demucs', 'auto'):
            vocals, instrumental, engine = _try_demucs()
        if (vocals is None or instrumental is None) and prefer in ('spleeter', 'auto'):
            v2, i2, e2 = _try_spleeter()
            vocals = vocals or v2
            instrumental = instrumental or i2
            engine = engine or e2
        if vocals is None or instrumental is None:
            vocals, instrumental, engine = _ffmpeg_fallback()

        if vocals and instrumental:
            print(f"Stems ready: vocals='{vocals}', instrumental='{instrumental}'")
        else:
            print("Stem separation failed; continuing without stems.")
        return vocals, instrumental, engine
    except Exception as e:
        print(f"Error separating stems: {e}")
        return None, None, None

def _load_wav_array(path):
    try:
        import numpy as np
        seg = AudioSegment.from_wav(path)
        sr = int(seg.frame_rate)
        ch = int(seg.channels)
        smp = np.array(seg.get_array_of_samples()).astype(np.float32)
        if ch > 1:
            smp = smp.reshape((-1, ch)).mean(axis=1)
        maxv = float(1 << (seg.sample_width * 8 - 1))
        smp = smp / maxv
        return smp, sr
    except Exception as e:
        print(f"Load wav array failed: {e}")
        return None, 0

def align_offset_crosscorr(mix_wav_path, vocal_wav_path, max_shift_sec=3.0):
    try:
        import numpy as np
        x, sr1 = _load_wav_array(mix_wav_path)
        y, sr2 = _load_wav_array(vocal_wav_path)
        if x is None or y is None or sr1 <= 0 or sr2 <= 0:
            return 0.0
        sr = min(sr1, sr2)
        def _resample(z, s):
            try:
                if s == sr:
                    return z
                import librosa
                return librosa.resample(z, orig_sr=s, target_sr=sr)
            except Exception:
                return z[:int(len(z) * sr / float(s))]
        x = _resample(x, sr1)
        y = _resample(y, sr2)
        n = min(len(x), len(y))
        if n <= 0:
            return 0.0
        x = x[:n]
        y = y[:n]
        win = int(max(1, sr * max_shift_sec))
        xp = x - np.mean(x)
        yp = y - np.mean(y)
        corr = np.correlate(xp, yp, mode='full')
        lags = np.arange(-n + 1, n)
        mask = (lags >= -win) & (lags <= win)
        corr = corr[mask]
        lags = lags[mask]
        k = int(np.argmax(corr))
        lag = int(lags[k])
        return float(lag) / float(sr)
    except Exception as e:
        print(f"Cross-corr offset failed: {e}")
        return 0.0

def shift_wav(input_wav, output_wav, delay_sec):
    try:
        seg = AudioSegment.from_wav(input_wav)
        if delay_sec >= 0:
            sil = AudioSegment.silent(duration=int(delay_sec * 1000))
            out = sil + seg
        else:
            cut_ms = int(abs(delay_sec) * 1000)
            out = seg[cut_ms:]
        os.makedirs(os.path.dirname(output_wav), exist_ok=True)
        out.export(output_wav, format='wav')
        return output_wav
    except Exception as e:
        print(f"Shift wav failed: {e}")
        return None

def shift_alignment_times(alignment_path, output_path, offset_sec):
    try:
        import json as _json
        with open(alignment_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        def _sh(v):
            try:
                return float(v) + float(offset_sec)
            except Exception:
                return v
        if isinstance(data, dict) and isinstance(data.get('fragments'), list):
            for frag in data['fragments']:
                if 'begin' in frag:
                    frag['begin'] = str(max(0.0, _sh(frag['begin'])))
                if 'end' in frag:
                    frag['end'] = str(max(0.0, _sh(frag['end'])))
                ws = frag.get('words')
                if isinstance(ws, list):
                    for w in ws:
                        if 'start' in w:
                            w['start'] = max(0.0, _sh(w['start']))
                        if 'end' in w:
                            w['end'] = max(0.0, _sh(w['end']))
        elif isinstance(data, list):
            for ln in data:
                if isinstance(ln, dict):
                    if 'start' in ln:
                        ln['start'] = max(0.0, _sh(ln['start']))
                    if 'end' in ln:
                        ln['end'] = max(0.0, _sh(ln['end']))
        with open(output_path, 'w', encoding='utf-8') as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception as e:
        print(f"Shift alignment failed: {e}")
        return None

def export_alignment_formats(alignment_path, lrc_out_path=None, srt_out_path=None):
    try:
        intervals = _load_alignment_intervals(alignment_path)
        if not intervals:
            return None, None
        def _fmt_ts(ts):
            m = int(ts // 60)
            s = ts - 60 * m
            return f"{m:02d}:{s:05.2f}"
        def _fmt_srt(ts):
            h = int(ts // 3600)
            m = int((ts % 3600) // 60)
            s = int(ts % 60)
            ms = int((ts - int(ts)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lrc_lines = []
        srt_blocks = []
        idx = 1
        for (s, e, txt) in intervals:
            ts = _fmt_ts(float(s))
            lrc_lines.append(f"[{ts}]{txt}")
            srt_blocks.append((idx, _fmt_srt(float(s)), _fmt_srt(float(e)), txt))
            idx += 1
        lrc_p = None
        srt_p = None
        if lrc_out_path:
            os.makedirs(os.path.dirname(lrc_out_path), exist_ok=True)
            with open(lrc_out_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lrc_lines))
            lrc_p = lrc_out_path
        if srt_out_path:
            os.makedirs(os.path.dirname(srt_out_path), exist_ok=True)
            with open(srt_out_path, 'w', encoding='utf-8') as f:
                for (i, s1, s2, t) in srt_blocks:
                    f.write(str(i) + "\n")
                    f.write(s1 + " --> " + s2 + "\n")
                    f.write(t + "\n\n")
            srt_p = srt_out_path
        return lrc_p, srt_p
    except Exception as e:
        print(f"Export alignment formats failed: {e}")
        return None, None

def extract_audio_from_video(video_path, audio_path):
    """
    Extract audio from a video file and save it as a WAV.
    """
    try:
        from backend.config import FFMPEG_PATH
        cmd = [
            FFMPEG_PATH, '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            audio_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None

def syllabify_word(word):
    try:
        import re
        w = re.sub(r"[^a-zA-Z]", "", str(word))
        if not w:
            return [str(word)]
        vowels = "aeiouyAEIOUY"
        syllables = []
        cur = ""
        for ch in w:
            cur += ch
            if ch in vowels:
                syllables.append(cur)
                cur = ""
        if cur:
            if syllables:
                syllables[-1] += cur
            else:
                syllables.append(cur)
        return [s if s else str(word) for s in syllables]
    except Exception:
        return [str(word)]

def refine_syllable_alignment(alignment_path, output_path, syllable_offset_ms=0.0):
    try:
        import json as _json
        with open(alignment_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        frags = []
        if isinstance(data, dict) and isinstance(data.get('fragments'), list):
            frags = data['fragments']
        else:
            return None
        for frag in frags:
            ws = frag.get('words') or []
            new_words = []
            for w in ws:
                s = float(w.get('start', frag.get('begin', 0)))
                e = float(w.get('end', s))
                toks = syllabify_word(w.get('text', ''))
                dur = max(0.0, e - s)
                step = dur / float(max(1, len(toks)))
                syls = []
                for i, t in enumerate(toks):
                    ss = s + i * step + float(syllable_offset_ms) / 1000.0
                    ee = min(e, s + (i + 1) * step + float(syllable_offset_ms) / 1000.0)
                    syls.append({'text': t, 'start': ss, 'end': ee})
                nw = dict(w)
                nw['syllables'] = syls
                new_words.append(nw)
            frag['words'] = new_words
        with open(output_path, 'w', encoding='utf-8') as f:
            _json.dump({'fragments': frags}, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception as e:
        print(f"Refine syllable alignment failed: {e}")
        return None

def detect_vocal_segments_from_stem(vocal_wav_path, min_segment_len=0.2):
    try:
        dur = get_duration(vocal_wav_path)
        if dur <= 0:
            return []
        sil = detect_silence_segments(vocal_wav_path, min_silence_dur=0.3)
        nons = []
        cur = 0.0
        for (s, e) in sil:
            if s > cur:
                nons.append((cur, s))
            cur = max(cur, e)
        if cur < dur:
            nons.append((cur, dur))
        out = []
        for (s, e) in nons:
            if (e - s) >= float(min_segment_len):
                out.append({'start': float(s), 'end': float(e)})
        return out
    except Exception as e:
        print(f"Detect vocal segments from stem failed: {e}")
        return []

def compute_separation_quality(vocals_path, instrumental_path, mix_path):
    try:
        import numpy as np
        xv, srv = _load_wav_array(vocals_path)
        xi, sri = _load_wav_array(instrumental_path)
        xm, srm = _load_wav_array(mix_path)
        if xv is None or xi is None or xm is None:
            return {'vocal_energy_ratio': None, 'reconstruction_corr': None}
        sr = min(srv, sri, srm)
        def rs(z, s):
            try:
                if s == sr:
                    return z
                import librosa
                return librosa.resample(z, orig_sr=s, target_sr=sr)
            except Exception:
                return z[:int(len(z) * sr / float(s))]
        xv = rs(xv, srv)
        xi = rs(xi, sri)
        xm = rs(xm, srm)
        n = min(len(xv), len(xi), len(xm))
        xv = xv[:n]
        xi = xi[:n]
        xm = xm[:n]
        ve = float(np.sqrt(np.mean(xv * xv)))
        me = float(np.sqrt(np.mean(xm * xm)))
        ratio = ve / me if me > 1e-9 else None
        recon = xv + xi
        try:
            c = float(np.corrcoef(recon, xm)[0, 1])
        except Exception:
            c = None
        return {'vocal_energy_ratio': ratio, 'reconstruction_corr': c}
    except Exception as e:
        print(f"Compute separation quality failed: {e}")
        return {'vocal_energy_ratio': None, 'reconstruction_corr': None}

# (removed duplicate _load_wav_array)

def _spectral_flux(x, sr, frame=1024, hop=256):
    try:
        import numpy as np
        n = len(x)
        hann = np.hanning(frame)
        prev_mag = None
        flux = []
        for i in range(0, max(0, n - frame), hop):
            seg = x[i:i+frame] * hann
            spec = np.fft.rfft(seg)
            mag = np.abs(spec)
            if prev_mag is None:
                flux.append(0.0)
            else:
                diff = np.clip(mag - prev_mag, 0, None)
                flux.append(float(np.sum(diff)))
            prev_mag = mag
        flux = np.array(flux, dtype=np.float32)
        # Normalize
        if np.max(flux) > 0:
            flux /= float(np.max(flux))
        times = np.arange(len(flux)) * (hop / float(sr))
        return flux, times
    except Exception as e:
        print(f"Spectral flux failed: {e}")
        return None, None

def refine_word_alignment(wav_path, alignment_path, window_sec=0.25):
    try:
        x, sr = _load_wav_array(wav_path)
        if x is None:
            return None
        flux, times = _spectral_flux(x, sr)
        if flux is None:
            return None
        import json as _json
        with open(alignment_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        # Normalize format to fragments with words
        fragments = []
        if isinstance(data, dict) and 'fragments' in data:
            fragments = data.get('fragments') or []
        elif isinstance(data, list):
            # Fallback format: lines only
            fragments = [
                {
                    'begin': str(max(0.0, float(d.get('start_ms', 0))/1000.0)),
                    'end': str(max(0.0, float(d.get('end_ms', 0))/1000.0)),
                    'lines': [str(d.get('text', ''))],
                    'words': []
                } for d in data
            ]
        # Build refined words per fragment
        import numpy as np
        wsize = float(window_sec)
        def _nearest_peak(target_t):
            try:
                # Search within +/- window for local maxima
                lo = max(0.0, target_t - wsize)
                hi = min(times[-1] if len(times)>0 else target_t + wsize, target_t + wsize)
                idx_lo = int(max(0, np.searchsorted(times, lo) - 1))
                idx_hi = int(min(len(times) - 1, np.searchsorted(times, hi)))
                if idx_hi <= idx_lo:
                    return target_t, 0.0
                seg_flux = flux[idx_lo:idx_hi+1]
                if seg_flux.size == 0:
                    return target_t, 0.0
                k = int(np.argmax(seg_flux))
                peak_t = float(times[idx_lo + k])
                conf = float(seg_flux[k])
                return peak_t, conf
            except Exception:
                return target_t, 0.0
        for frag in fragments:
            try:
                line_txt = ''
                lines = frag.get('lines')
                if isinstance(lines, list) and lines:
                    line_txt = str(lines[0])
                elif isinstance(lines, str):
                    line_txt = str(lines)
                begin = float(frag.get('begin', 0) or 0)
                end = float(frag.get('end', 0) or 0)
                words = frag.get('words')
                if not (isinstance(words, list) and words):
                    # Split naive words
                    toks = [w for w in (line_txt.split() if line_txt else [])]
                    # Distribute uniformly across fragment
                    dur = max(0.0, end - begin)
                    if dur <= 0 or not toks:
                        frag['words'] = []
                        continue
                    step = dur / float(len(toks))
                    words = []
                    for i, t in enumerate(toks):
                        s0 = begin + i * step
                        e0 = begin + min(dur, (i + 1) * step)
                        words.append({'text': t, 'start': s0, 'end': e0})
                    frag['words'] = words
                # Refine each word start by nearest flux peak; end by next peak or original end
                refined = []
                last_peak = None
                for i, w in enumerate(words):
                    ws = float(w.get('start', begin))
                    we = float(w.get('end', ws))
                    peak_t, conf = _nearest_peak(ws)
                    # Ensure monotonic non-decreasing starts
                    if last_peak is not None and peak_t < last_peak:
                        peak_t = last_peak
                    last_peak = peak_t
                    refined.append({
                        'text': str(w.get('text', '')),
                        'start': float(max(begin, min(peak_t, end))),
                        'end': float(min(end, max(we, peak_t + 0.05))),
                        'confidence': float(conf)
                    })
                frag['words'] = refined
            except Exception:
                continue
        # Persist refined structure to a sibling file
        out_path = os.path.splitext(alignment_path)[0] + '_refined.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            _json.dump({'fragments': fragments}, f, ensure_ascii=False, indent=2)
        return out_path
    except Exception as e:
        print(f"Refine word alignment failed: {e}")
        return None

def track_pitch_autocorr(wav_path, frame_ms=40, hop_ms=10, fmin=70.0, fmax=800.0):
    try:
        import numpy as np
        x, sr = _load_wav_array(wav_path)
        if x is None:
            return []
        frame = int(sr * (frame_ms / 1000.0))
        hop = int(sr * (hop_ms / 1000.0))
        out = []
        for i in range(0, max(0, len(x) - frame), hop):
            seg = x[i:i+frame]
            seg = seg - np.mean(seg)
            if np.max(np.abs(seg)) <= 1e-6:
                out.append({'t': i/float(sr), 'f0': 0.0, 'conf': 0.0})
                continue
            ac = np.correlate(seg, seg, mode='full')[len(seg)-1:]
            ac /= (np.max(ac) or 1.0)
            # Search lags for f0 between fmin and fmax
            lag_min = int(sr / float(fmax))
            lag_max = int(sr / float(fmin))
            if lag_max >= len(ac):
                lag_max = len(ac) - 1
            k = int(np.argmax(ac[lag_min:lag_max+1])) + lag_min
            f0 = float(sr / float(max(1, k)))
            conf = float(ac[k])
            out.append({'t': i/float(sr), 'f0': f0 if conf > 0.2 else 0.0, 'conf': conf})
        return out
    except Exception as e:
        print(f"Pitch tracking failed: {e}")
        return []

def track_formants_lpc(wav_path, frame_ms=40, hop_ms=20, order=12):
    try:
        import numpy as np
        x, sr = _load_wav_array(wav_path)
        if x is None:
            return []
        frame = int(sr * (frame_ms / 1000.0))
        hop = int(sr * (hop_ms / 1000.0))
        def _lpc_coeffs(sig, p):
            # Autocorrelation method with simple Levinson-Durbin
            r = np.array([np.sum(sig[:len(sig)-k] * sig[k:]) for k in range(p+1)], dtype=np.float64)
            a = np.zeros(p+1, dtype=np.float64)
            e = r[0]
            if e <= 1e-12:
                return a, e
            a[0] = 1.0
            refl = np.zeros(p, dtype=np.float64)
            for i in range(1, p+1):
                acc = r[i]
                for j in range(1, i):
                    acc += a[j] * r[i-j]
                k = -acc / (e or 1.0)
                refl[i-1] = k
                a_prev = a.copy()
                a[i] = k
                for j in range(1, i):
                    a[j] = a_prev[j] + k * a_prev[i-j]
                e *= (1.0 - k*k)
            return a, e
        out = []
        for i in range(0, max(0, len(x) - frame), hop):
            seg = x[i:i+frame]
            seg = seg * np.hanning(len(seg))
            a, e = _lpc_coeffs(seg, order)
            if e <= 1e-12:
                out.append({'t': i/float(sr), 'formants': []})
                continue
            den = np.concatenate(([1.0], -a[1:]))
            roots = np.roots(den)
            formants = []
            for r in roots:
                if np.imag(r) >= 0:
                    freq = np.angle(r) * (sr / (2.0 * np.pi))
                    if 200.0 <= freq <= 4000.0:
                        formants.append(freq)
            formants = sorted(formants)[:3]
            out.append({'t': i/float(sr), 'formants': formants})
        return out
    except Exception as e:
        print(f"Formant tracking failed: {e}")
        return []

def classify_vocal_segments_enhanced(wav_path, alignment_path, vocals=None, instrumental=None):
    try:
        duration = get_duration(wav_path)
        if duration <= 0:
            return {'segments': [], 'main_vocal_onset': None}
        if vocals and not os.path.exists(vocals):
            vocals = None
        if instrumental and not os.path.exists(instrumental):
            instrumental = None
        # Base classification using alignment
        base = classify_vocal_segments(wav_path, alignment_path)
        segs = base.get('segments', [])
        onset = base.get('main_vocal_onset')
        # Lead vs background using energy ratio on vocals stem
        if vocals and os.path.exists(vocals):
            xv, sr = _load_wav_array(vocals)
            import numpy as np
            def seg_energy(s, e):
                lo = int(max(0, s * sr))
                hi = int(min(len(xv), e * sr))
                if hi <= lo:
                    return 0.0
                seg = xv[lo:hi]
                return float(np.sqrt(np.mean(seg*seg)))
            for s in segs:
                if s['label'] == 'main_vocal':
                    er = seg_energy(s['start'], s['end'])
                    s['label'] = 'lead_vocal' if er >= 0.02 else 'background_vocal'
        # Speech vs singing heuristic using pitch continuity
        pitch = track_pitch_autocorr(vocals or wav_path)
        voiced = sum(1 for p in pitch if p.get('f0', 0.0) > 0)
        total = max(1, len(pitch))
        voicing_rate = voiced / float(total)
        # Tag segments with speech when voicing_rate low
        if voicing_rate < 0.35:
            for s in segs:
                if s['label'] in ('lead_vocal', 'background_vocal', 'main_vocal'):
                    s['label'] = 'speech'
        return {'segments': segs, 'main_vocal_onset': onset}
    except Exception as e:
        print(f"Enhanced vocal classification failed: {e}")
        return {'segments': [], 'main_vocal_onset': None}

def get_duration(wav_path):
    """
    Get the duration of a WAV file in seconds
    
    Args:
        wav_path (str): Path to WAV file
    
    Returns:
        float: Duration in seconds
    """
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            # Get audio parameters
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            
            # Calculate duration
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error getting duration: {e}")
        return 0.0

def mix_audio_tracks(tracks, output_path=None, sample_rate=44100, normalize=False):
    """
    Mix multiple audio files into a single track.
    Uses FFmpeg amix when available; falls back to pydub overlay.

    Args:
        tracks (List[str]): Paths to audio files to mix (2+ recommended)
        output_path (str, optional): Output path. Defaults to sibling mixed WAV.
        sample_rate (int): Target sample rate (Hz)
        normalize (bool): Enable FFmpeg amix normalize

    Returns:
        str|None: Path to mixed audio file or None on failure
    """
    try:
        tracks = [t for t in (tracks or []) if t and os.path.exists(t)]
        if len(tracks) < 2:
            print("Mix requires at least two valid tracks; skipping.")
            return tracks[0] if tracks else None

        base = os.path.splitext(os.path.basename(tracks[0]))[0]
        out_dir = os.path.dirname(tracks[0])
        if output_path is None:
            output_path = os.path.join(out_dir, f"{base}_mixed.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Try FFmpeg amix
        ff = _FFMPEG_PATH or 'ffmpeg'
        ff_ok = (which(ff) is not None) or os.path.exists(ff)
        if ff_ok:
            try:
                # Build input args and amix filter
                args = [ff, '-y']
                for t in tracks:
                    args += ['-i', t]
                amix = f"amix=inputs={len(tracks)}:duration=longest:normalize={'1' if normalize else '0'}"
                args += ['-filter_complex', amix, '-ar', str(int(sample_rate)), output_path]
                print(f"Running FFmpeg amix: {' '.join(args[:8])} ...")
                _run_managed_subprocess(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(output_path):
                    return output_path
            except Exception as e:
                print(f"FFmpeg amix failed: {e}")

        # Fallback: pydub overlay
        try:
            from pydub import AudioSegment
            mixed = None
            for t in tracks:
                seg = AudioSegment.from_file(t)
                seg = seg.set_frame_rate(sample_rate)
                mixed = seg if mixed is None else mixed.overlay(seg)
            mixed.export(output_path, format='wav')
            return output_path
        except Exception as e:
            print(f"Pydub mix failed: {e}")
            return None
    except Exception as e:
        print(f"Error mixing audio tracks: {e}")
        return None

def detect_vocal_onset(
    wav_path,
    noise_threshold_db='-25dB',
    min_silence_dur=0.5,
    hp_freq=300,
    lp_freq=3400,
    min_voiced_hold=0.4,
):
    """
    Detect the first significant non-silent moment (vocal onset proxy) using FFmpeg
    `silencedetect` with a narrower voice band and then verify a short voiced-hold
    window to avoid triggering on loud drums.

    Args:
        wav_path (str): Path to mono WAV file
        noise_threshold_db (str): Silence threshold (e.g., '-25dB')
        min_silence_dur (float): Minimum silence duration to consider (seconds)
        hp_freq (int): High-pass cutoff for band-limiting (Hz)
        lp_freq (int): Low-pass cutoff for band-limiting (Hz)
        min_voiced_hold (float): Required continuous voiced duration after onset (seconds)

    Returns:
        float: Detected onset time in seconds, or 0.0 if not found
    """
    try:
        if not os.path.exists(wav_path):
            return 0.0

        def _db_to_float(db_str):
            try:
                s = str(db_str).lower().replace('db', '').strip()
                return float(s)
            except Exception:
                return -25.0

        def _voiced_hold_ok(audio: AudioSegment, start_sec: float, hold_sec: float, thresh_dbf: float) -> bool:
            try:
                start_ms = int(max(0.0, start_sec) * 1000)
                end_ms = int((max(0.0, start_sec) + max(0.1, hold_sec)) * 1000)
                seg = audio[start_ms:end_ms]
                if len(seg) <= 0:
                    return False
                # Inspect in short windows (20ms) and require majority above threshold + margin
                window_ms = 20
                margin_db = 3.0
                thresh = (thresh_dbf + margin_db)
                total = max(1, int(len(seg) / window_ms))
                ok = 0
                for i in range(0, len(seg), window_ms):
                    frame = seg[i:i + window_ms]
                    try:
                        val = frame.dBFS
                        if val is None:
                            val = -90.0
                    except Exception:
                        val = -90.0
                    if val > thresh:
                        ok += 1
                ratio = ok / float(total)
                return ratio >= 0.6
            except Exception:
                return False

        ffmpeg_path = _FFMPEG_PATH or 'ffmpeg'
        # Build filter graph: stricter band-limit + silencedetect
        bandpass = f"highpass=f={int(hp_freq)},lowpass=f={int(lp_freq)}"
        sd = f"silencedetect=noise={noise_threshold_db}:d={float(min_silence_dur)}"
        filtergraph = f"{bandpass},{sd}"

        cmd = [
            ffmpeg_path,
            '-hide_banner', '-nostats',
            '-i', wav_path,
            '-af', filtergraph,
            '-f', 'null', '-'
        ]

        proc = _run_managed_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log = proc.stderr or ''

        # Parse all candidate silence_end events
        candidates = []
        for line in log.splitlines():
            line = line.strip()
            if 'silence_end:' in line:
                try:
                    parts = line.split('silence_end:')
                    if len(parts) > 1:
                        rest = parts[1].strip()
                        val_str = rest.split('|')[0].strip()
                        candidate = float(val_str)
                        candidates.append(candidate)
                except Exception:
                    continue

        if not candidates:
            return 0.0

        # Verify voiced-hold after each candidate; pick the first that passes
        try:
            audio = AudioSegment.from_wav(wav_path)
            thresh_dbf = _db_to_float(noise_threshold_db)
            for c in sorted({max(0.0, float(x)) for x in candidates}):
                if _voiced_hold_ok(audio, c, float(min_voiced_hold), thresh_dbf):
                    return c
        except Exception:
            # If pydub inspection fails, fall back to first candidate
            pass

        # Fallback: choose the earliest candidate even if hold check failed
        return max(0.0, float(min(candidates)))
    except Exception as e:
        print(f"Vocal onset detection failed: {e}")
        return 0.0

def detect_silence_segments(
    wav_path,
    noise_threshold_db='-25dB',
    min_silence_dur=0.8,
    hp_freq=300,
    lp_freq=3400,
):
    """
    Detect all silence segments across the track using FFmpeg silencedetect
    with optional voice-band limiting.

    Returns a list of (start_sec, end_sec) tuples. Trailing open silence is
    closed at the file duration.
    """
    try:
        if not os.path.exists(wav_path):
            return []

        # Build filter graph
        ffmpeg_path = _FFMPEG_PATH or 'ffmpeg'
        bandpass = f"highpass=f={int(hp_freq)},lowpass=f={int(lp_freq)}"
        sd = f"silencedetect=noise={noise_threshold_db}:d={float(min_silence_dur)}"
        filtergraph = f"{bandpass},{sd}"

        cmd = [
            ffmpeg_path,
            '-hide_banner', '-nostats',
            '-i', wav_path,
            '-af', filtergraph,
            '-f', 'null', '-'
        ]
        proc = _run_managed_subprocess(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        log = proc.stderr or ''

        segments = []
        current_start = None
        # Parse log lines for silence_start and silence_end
        for line in log.splitlines():
            ls = line.strip()
            if 'silence_start:' in ls:
                try:
                    val = float(ls.split('silence_start:')[1].strip())
                    current_start = max(0.0, val)
                except Exception:
                    continue
            elif 'silence_end:' in ls:
                try:
                    rest = ls.split('silence_end:')[1].strip()
                    end_str = rest.split('|')[0].strip()
                    end_val = max(0.0, float(end_str))
                    # If duration is present, compute start precisely
                    dur = None
                    if '| duration:' in ls:
                        try:
                            dur_str = ls.split('| duration:')[1].strip()
                            dur = float(dur_str)
                        except Exception:
                            dur = None
                    start_val = current_start if current_start is not None else (
                        (end_val - dur) if (dur is not None) else None
                    )
                    if start_val is None:
                        # If we cannot determine start, skip this segment
                        current_start = None
                        continue
                    segments.append((max(0.0, start_val), end_val))
                    current_start = None
                except Exception:
                    continue

        # Handle trailing open silence (start without explicit end)
        try:
            if current_start is not None:
                dur = get_duration(wav_path)
                segments.append((max(0.0, current_start), max(0.0, float(dur))))
        except Exception:
            pass

        # Normalize and sort
        cleaned = []
        for (s, e) in segments:
            if e <= s:
                continue
            cleaned.append((float(s), float(e)))
        cleaned.sort(key=lambda x: x[0])
        return cleaned
    except Exception as e:
        print(f"Silence segment detection failed: {e}")
        return []

def detect_tail_silence(
    wav_path,
    noise_threshold_db='-25dB',
    min_silence_dur=0.8,
    hp_freq=300,
    lp_freq=3400,
    min_tail_sec=1.5,
):
    """
    Detect trailing silence at the end of the file. Returns (start_sec, end_sec)
    if present and at least min_tail_sec long; otherwise returns None.
    """
    try:
        segments = detect_silence_segments(
            wav_path,
            noise_threshold_db=noise_threshold_db,
            min_silence_dur=min_silence_dur,
            hp_freq=hp_freq,
            lp_freq=lp_freq,
        )
        if not segments:
            return None
        dur = get_duration(wav_path)
        tail = None
        # Examine last segment that touches end
        for (s, e) in segments[::-1]:
            if abs(e - dur) <= 0.05 or e >= dur:
                if (e - s) >= float(min_tail_sec):
                    tail = (float(s), float(dur))
                break
        return tail
    except Exception as e:
        print(f"Tail silence detection failed: {e}")
        return None

def detect_breaks_and_pauses(
    wav_path,
    noise_threshold_db='-25dB',
    min_silence_dur=0.4,
    flux_window_sec=0.25,
    min_pause_sec=0.2,
    hp_freq=300,
    lp_freq=3400,
):
    try:
        if not os.path.exists(wav_path):
            return []
        silences = detect_silence_segments(
            wav_path,
            noise_threshold_db=noise_threshold_db,
            min_silence_dur=min_silence_dur,
            hp_freq=hp_freq,
            lp_freq=lp_freq,
        )
        x, sr = _load_wav_array(wav_path)
        flux, times = _spectral_flux(x, sr)
        pauses = []
        if flux is not None and times is not None:
            import numpy as np
            w = float(flux_window_sec)
            tmax = times[-1] if len(times) > 0 else 0.0
            tau = float(np.percentile(flux, 20)) if len(flux) > 0 else 0.0
            for i in range(1, len(flux) - 1):
                if flux[i] < tau and flux[i] <= flux[i-1] and flux[i] <= flux[i+1]:
                    ct = float(times[i])
                    s = max(0.0, ct - max(0.05, min_pause_sec))
                    e = min(tmax, ct + max(0.05, min_pause_sec))
                    if e > s:
                        pauses.append((s, e))
        merged = []
        all_segs = [(float(s), float(e), 'silence') for (s, e) in (silences or [])] + [(float(s), float(e), 'pause') for (s, e) in (pauses or [])]
        all_segs.sort(key=lambda x: x[0])
        for seg in all_segs:
            if not merged:
                merged.append(list(seg))
            else:
                ps, pe, pk = merged[-1]
                s, e, k = seg
                if s <= pe + 0.08:
                    merged[-1][1] = max(pe, e)
                    if pk != 'silence' and k == 'silence':
                        merged[-1][2] = 'silence'
                else:
                    merged.append([s, e, k])
        out = []
        for s, e, k in merged:
            if (e - s) >= float(min_pause_sec):
                out.append({'start': float(s), 'end': float(e), 'kind': str(k)})
        return out
    except Exception as e:
        print(f"Break/pause detection failed: {e}")
        return []

def _load_alignment_intervals(alignment_path):
    """
    Load alignment JSON and return a list of (start, end, text) tuples.
    Supports Aeneas JSON with top-level 'fragments', legacy top-level 'lines',
    or a plain list of line dicts.
    """
    try:
        if not alignment_path or not os.path.exists(alignment_path):
            return []
        with open(alignment_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        intervals = []
        # Aeneas format: 'fragments' with 'begin', 'end', and 'lines'
        if isinstance(data, dict) and isinstance(data.get('fragments'), list):
            for frag in data.get('fragments', []):
                try:
                    start = float(frag.get('begin') or frag.get('start') or 0)
                    end = float(frag.get('end') or 0)
                    lines = frag.get('lines') or []
                    text = " ".join([str(x) for x in lines]) if lines else str(frag.get('text', '') or '')
                    intervals.append((max(0.0, start), max(0.0, end), text))
                except Exception:
                    continue
            return intervals

        # Legacy dict format: top-level 'lines'
        lines = []
        if isinstance(data, dict):
            lines = data.get('lines', [])
        elif isinstance(data, list):
            lines = data
        for ln in lines:
            if isinstance(ln, dict):
                text = str(ln.get('text', '') or '')
                start = float(ln.get('start', 0) or 0)
                end = float(ln.get('end', 0) or 0)
            else:
                text = str(ln)
                start = 0.0
                end = 0.0
            intervals.append((max(0.0, start), max(0.0, end), text))
        return intervals
    except Exception as e:
        print(f"Failed to load alignment intervals: {e}")
        return []

def detect_main_vocal_onset_from_alignment(alignment_path):
    """
    Determine the earliest start time of a non-empty aligned lyric.
    This serves as the 'main vocal onset', ignoring samples not tied to lyrics.

    Returns float seconds or None if not available.
    """
    try:
        intervals = _load_alignment_intervals(alignment_path)
        onset = None
        for (s, e, txt) in intervals:
            if str(txt).strip():
                if onset is None or s < onset:
                    onset = s
        return onset
    except Exception as e:
        print(f"Main vocal onset detection from alignment failed: {e}")
        return None

def classify_vocal_segments(
    wav_path,
    alignment_path,
    noise_threshold_db='-25dB',
    min_silence_dur=0.6,
    hp_freq=300,
    lp_freq=3400,
    min_segment_len=0.15,
):
    """
    Classify non-silent segments into 'main_vocal' vs 'instrumental' using alignment.

    Returns a dict with:
      {
        'segments': [ {'start': s, 'end': e, 'label': 'main_vocal'|'instrumental'} ],
        'main_vocal_onset': float|None,
      }
    """
    try:
        duration = get_duration(wav_path)
        if duration <= 0:
            return {'segments': [], 'main_vocal_onset': None}

        # Silence → non-silence complement
        silences = detect_silence_segments(
            wav_path,
            noise_threshold_db=noise_threshold_db,
            min_silence_dur=min_silence_dur,
            hp_freq=hp_freq,
            lp_freq=lp_freq,
        )
        nons = []
        cur = 0.0
        for (s, e) in silences:
            if s > cur:
                nons.append((cur, s))
            cur = max(cur, e)
        if cur < duration:
            nons.append((cur, duration))

        # Alignment intervals with text
        intervals = _load_alignment_intervals(alignment_path)
        lyric_intervals = [ (s, e) for (s, e, txt) in intervals if str(txt).strip() ]
        main_onset = None
        if lyric_intervals:
            main_onset = min(s for (s, _e) in lyric_intervals)

        def _overlaps(a, b):
            (s1, e1), (s2, e2) = a, b
            return (min(e1, e2) - max(s1, s2)) > 0

        classified = []
        for (s, e) in nons:
            if (e - s) < float(min_segment_len):
                continue
            label = 'instrumental'
            for (ls, le) in lyric_intervals:
                if _overlaps((s, e), (ls, le)):
                    label = 'main_vocal'
                    break
            classified.append({'start': float(s), 'end': float(e), 'label': label})

        return {
            'segments': classified,
            'main_vocal_onset': main_onset,
        }
    except Exception as e:
        print(f"Vocal segment classification failed: {e}")
        return {'segments': [], 'main_vocal_onset': None}

# CLI test block
if __name__ == "__main__":
    # Test conversion
    convert_to_wav("tests/sample.mp3", "uploads/test.wav")
    
    # Test duration
    if os.path.exists("uploads/test.wav"):
        duration = get_duration("uploads/test.wav")
        print(f"Duration: {duration:.2f} seconds")
        
        # Test normalization
        normalized_path = normalize_audio("uploads/test.wav")
        print(f"Normalized audio saved to: {normalized_path}")
    else:
        print("Conversion failed, test.wav not found")
