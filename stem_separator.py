import os
import subprocess
import shutil
from pathlib import Path

def separate_audio(audio_path: str, engine: str, out_dir: str) -> tuple[str, str]:
    """
    Separates audio into vocals and instrumental tracks.
    Returns (vocals_path, instrumental_path).
    """
    audio_path = str(Path(audio_path).absolute())
    out_dir = str(Path(out_dir).absolute())
    os.makedirs(out_dir, exist_ok=True)
    
    base_name = Path(audio_path).stem
    vocals_out = os.path.join(out_dir, f"{base_name}_vocals.wav")
    inst_out = os.path.join(out_dir, f"{base_name}_instrumental.wav")
    
    if engine == "none":
        return audio_path, audio_path

    try:
        if engine == "demucs":
            print(f"Running Demucs on {audio_path}...")
            # demucs outputs to out_dir / htdemucs / base_name / {vocals.wav, no_vocals.wav}
            cmd = ["demucs", "-n", "htdemucs", "--two-stems=vocals", "-o", out_dir, audio_path]
            subprocess.run(cmd, check=True, capture_output=True)
            
            demucs_dir = os.path.join(out_dir, "htdemucs", base_name)
            v_path = os.path.join(demucs_dir, "vocals.wav")
            i_path = os.path.join(demucs_dir, "no_vocals.wav")
            
            if os.path.exists(v_path) and os.path.exists(i_path):
                shutil.copy(v_path, vocals_out)
                shutil.copy(i_path, inst_out)
                shutil.rmtree(os.path.join(out_dir, "htdemucs"), ignore_errors=True)
                return vocals_out, inst_out
            else:
                print("Demucs output not found, falling back...")

    except Exception as e:
        print(f"Engine {engine} failed: {e}. Falling back to FFmpeg.")

    # Fallback / FFmpeg engine
    print(f"Running FFmpeg phase cancellation on {audio_path}...")
    # Instrumental: subtract right from left to cancel center (vocals)
    inst_cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", "pan=stereo|c0=c0-c1|c1=c0-c1",
        inst_out
    ]
    subprocess.run(inst_cmd, check=True, capture_output=True)
    
    # Vocals: Since simple phase cancellation can't perfectly isolate vocals, 
    # we'll just return the original audio as the "vocals" track for alignment purposes
    # (Alignment engines work fine with original audio, just better with isolated vocals)
    shutil.copy(audio_path, vocals_out)
    
    return vocals_out, inst_out
