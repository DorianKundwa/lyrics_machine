"""
LyricForge x Croonify — Unified Launcher
=========================================
Automatically finds a free port, starts the server, waits for it
to be ready, then opens the browser.

Usage:
  python launch.py                  # auto-finds free port starting at 8000
  python launch.py --port 9000      # preferred port (still auto-skips if busy)
  python launch.py --no-browser     # skip auto-open
  python launch.py --verbose        # show full server logs
  python launch.py --mode croonify  # start Croonify API only (no video render)
  python launch.py --mode lyricforge # start LyricForge app (default)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
SRC  = ROOT / "src"          # croonify package lives here

# ── Console helpers (ASCII-safe for Windows cp1252) ───────────────────────────
DLINE = "=" * 56
LINE  = "-" * 56

def banner(port: int, mode: str) -> None:
    print(DLINE)
    print("  LyricForge  x  Croonify AI")
    print(f"  Mode: {mode}   |   Port: {port}")
    print(DLINE)
    print()

def ok(msg):   print(f"  [OK]  {msg}")
def info(msg): print(f"  [>>]  {msg}")
def warn(msg): print(f"  [!!]  {msg}", file=sys.stderr)
def fail(msg): print(f"  [XX]  {msg}", file=sys.stderr)


# ── Port utilities ─────────────────────────────────────────────────────────────

def _is_free(port: int) -> bool:
    """
    Return True if nothing is listening on `port`.
    Uses connect_ex which correctly detects any listening socket,
    regardless of SO_REUSEADDR.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        result = s.connect_ex(("127.0.0.1", port))
        # connect_ex returns 0 when connection succeeds (port is in use)
        return result != 0


def find_free_port(preferred: int = 8000, max_tries: int = 20) -> int:
    """
    Return the first free port starting at `preferred`.
    Tries preferred, preferred+1, ... preferred+max_tries.
    Raises RuntimeError if none found.
    """
    for port in range(preferred, preferred + max_tries):
        if _is_free(port):
            return port
    raise RuntimeError(
        f"No free port found in range {preferred}–{preferred + max_tries - 1}. "
        "Kill stale processes or use --port to specify a different range."
    )


def wait_for_server(url: str, timeout: float = 40.0) -> bool:
    """Poll `url` until HTTP 200 or timeout."""
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        attempt += 1
        sys.stdout.write(f"\r  [..] Waiting for server{'.' * (attempt % 4)}   ")
        sys.stdout.flush()
        time.sleep(0.8)
    print()
    return False


def open_browser(url: str) -> None:
    import webbrowser
    webbrowser.open(url)


# ── Dependency checks ─────────────────────────────────────────────────────────

def check_deps(mode: str) -> None:
    info("Checking dependencies...")

    # Ensure user site-packages visible
    import site as _site
    user_site = _site.getusersitepackages()
    for p in [str(SRC), user_site]:
        if p not in sys.path:
            sys.path.insert(0, p)

    required = ["fastapi", "uvicorn"]
    if mode == "croonify":
        required += ["librosa"]
    else:
        required += ["PIL"]   # Pillow for LyricForge thumbnails

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        fail(f"Missing packages: {', '.join(missing)}")
        fail("Run:  pip install -r requirements.txt")
        sys.exit(1)

    ok("All core dependencies found")

    # Soft-warn about Croonify neural deps
    neural_ok = True
    for pkg in ("whisperx", "librosa"):
        try:
            __import__(pkg)
        except ImportError:
            if pkg == "whisperx":
                warn("whisperx not found — alignment will use Viterbi/heuristic fallback")
            neural_ok = False

    if neural_ok:
        ok("Neural alignment (whisperx + librosa) available")

    print()


# ── Server command builders ───────────────────────────────────────────────────

def _build_env() -> dict:
    import site as _site
    env  = os.environ.copy()
    usp  = _site.getusersitepackages()
    existing = env.get("PYTHONPATH", "")
    parts = [str(ROOT), str(SRC), usp] + ([existing] if existing else [])
    env["PYTHONPATH"]       = os.pathsep.join(p for p in parts if p)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _server_cmd(mode: str, host: str, port: int, log_level: str) -> list[str]:
    """Build the python -c '...' command that starts the appropriate app."""
    src_str  = str(SRC).replace("\\", "/")
    root_str = str(ROOT).replace("\\", "/")

    if mode == "croonify":
        app_import = "from croonify.api.server import app"
    else:
        # LyricForge app.py lives at ROOT level
        app_import = "import sys; sys.path.insert(0, r'{root}'); import app as _app; app = _app.app".format(
            root=root_str
        )

    code = f"""
import sys, os, site
sys.path.insert(0, r'{src_str}')
sys.path.insert(0, r'{root_str}')
usp = site.getusersitepackages()
if usp not in sys.path:
    sys.path.insert(2, usp)
import uvicorn
{app_import}
uvicorn.run(app, host='{host}', port={port}, log_level='{log_level}')
"""
    return [sys.executable, "-c", code]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LyricForge x Croonify launcher — auto-finds a free port",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Preferred starting port (auto-increments if busy). Default: 8000"
    )
    parser.add_argument("--host",       default="0.0.0.0",       help="Bind host")
    parser.add_argument("--no-browser", action="store_true",      help="Skip auto browser open")
    parser.add_argument("--verbose",    action="store_true",      help="Show full server logs")
    parser.add_argument(
        "--mode", choices=["lyricforge", "croonify"], default="lyricforge",
        help="lyricforge = video generator (default) | croonify = alignment API only"
    )
    args = parser.parse_args()

    # ── Auto-find a free port ─────────────────────────────────────────────────
    try:
        port = find_free_port(preferred=args.port)
    except RuntimeError as e:
        fail(str(e))
        sys.exit(1)

    if port != args.port:
        warn(f"Port {args.port} is busy — using port {port} instead.")

    banner(port, args.mode)
    check_deps(args.mode)

    # ── Build subprocess env + command ────────────────────────────────────────
    env       = _build_env()
    log_level = "info" if args.verbose else "warning"
    cmd       = _server_cmd(args.mode, args.host, port, log_level)

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    stdout = None if args.verbose else subprocess.DEVNULL
    stderr = None if args.verbose else subprocess.DEVNULL

    info(f"Starting server on http://localhost:{port} ...")
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(ROOT),
        stdout=stdout, stderr=stderr,
        **kwargs,
    )

    # ── Choose the health-check URL based on mode ─────────────────────────────
    if args.mode == "croonify":
        health_url = f"http://localhost:{port}/health"
    else:
        health_url = f"http://localhost:{port}/"   # LyricForge serves / directly

    ready = wait_for_server(health_url, timeout=40)
    print()   # newline after animated dots

    if not ready:
        fail("Server did not start within 40 seconds.")
        proc.terminate()
        sys.exit(1)

    ok(f"Server is ready!  (PID {proc.pid})")
    print()

    base_url = f"http://localhost:{port}"
    print(LINE)
    if args.mode == "lyricforge":
        print(f"  Web UI   : {base_url}/")
        print(f"  Editor   : {base_url}/editor")
        print(f"  API docs : {base_url}/docs")
    else:
        print(f"  Web UI   : {base_url}/")
        print(f"  API docs : {base_url}/docs")
        print(f"  Health   : {base_url}/health")
    print(LINE)
    print()

    if not args.no_browser:
        info(f"Opening browser -> {base_url}")
        open_browser(base_url)

    print("  Press Ctrl+C to stop the server.")
    print()

    # ── Forward Ctrl+C cleanly to child process ───────────────────────────────
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n  Stopping server...")
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  Server stopped. Goodbye!")


if __name__ == "__main__":
    main()
