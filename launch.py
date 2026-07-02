"""
Croonify Launcher
=================
Starts the backend API server (which also serves the frontend SPA),
waits for it to be ready, then opens the browser automatically.

Usage:
  python launch.py              # default port 8000
  python launch.py --port 9000  # custom port
  python launch.py --no-browser # skip auto-open
"""

import argparse
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
ROOT    = Path(__file__).parent.resolve()
SRC     = str(ROOT / "src")
SERVE   = str(ROOT / "serve.py")

# ── Console helpers (no emoji — Windows cp1252 safe) ──────────────────────────
LINE  = "-" * 50
DLINE = "=" * 50

def banner():
    print(DLINE)
    print("  Croonify  --  AI Lyrics Sync Engine")
    print("  Neural lyrics-to-audio alignment")
    print(DLINE)
    print()

def ok(msg):  print(f"  [OK]  {msg}")
def info(msg): print(f"  [>>] {msg}")
def warn(msg): print(f"  [!!] {msg}", file=sys.stderr)
def fail(msg): print(f"  [XX] {msg}", file=sys.stderr)


# ── Port utilities ─────────────────────────────────────────────────────────────

def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Poll /health until the server responds or timeout is hit."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    attempt = 0
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


def open_browser(url: str):
    """Open the browser cross-platform."""
    import webbrowser
    webbrowser.open(url)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Croonify launcher")
    parser.add_argument("--port",       type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--host",       default="0.0.0.0",      help="Bind host")
    parser.add_argument("--no-browser", action="store_true",     help="Don't open the browser")
    parser.add_argument("--verbose",    action="store_true",     help="Show server logs")
    args = parser.parse_args()

    banner()

    # ── Check Python can find our packages ────────────────────────────────────
    info("Checking dependencies...")

    import site as _site
    user_site = _site.getusersitepackages()
    # Inject paths so imports below work
    for p in [SRC, user_site]:
        if p not in sys.path:
            sys.path.insert(0, p)

    missing = []
    for pkg in ("fastapi", "uvicorn", "librosa"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        fail(f"Missing packages: {', '.join(missing)}")
        fail("Run:  pip install -r requirements.txt")
        sys.exit(1)

    ok("All dependencies found")

    # ── Check if port is already busy ─────────────────────────────────────────
    if port_in_use(args.port):
        warn(f"Port {args.port} is already in use.")
        try:
            with urllib.request.urlopen(
                f"http://localhost:{args.port}/health", timeout=2
            ) as r:
                import json
                data = json.loads(r.read())
                if data.get("version") == "0.1.0":
                    ok(f"Croonify is already running on port {args.port}!")
                    url = f"http://localhost:{args.port}"
                    if not args.no_browser:
                        info(f"Opening {url}")
                        open_browser(url)
                    print()
                    info(f"  Web UI : {url}/")
                    info(f"  API    : {url}/docs")
                    info(f"  Health : {url}/health")
                    return
        except Exception:
            pass
        fail(f"Port {args.port} is busy with another service. Use --port <N>.")
        sys.exit(1)

    # ── Build PYTHONPATH for the subprocess ────────────────────────────────────
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    parts = [SRC, user_site] + ([existing_pypath] if existing_pypath else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # Force UTF-8 output in the child process
    env["PYTHONIOENCODING"] = "utf-8"

    # ── Start the server subprocess ────────────────────────────────────────────
    info(f"Starting Croonify server on http://localhost:{args.port} ...")

    log_level = "info" if args.verbose else "warning"
    cmd = [
        sys.executable, "-c",
        f"""
import sys, os, site
sys.path.insert(0, r'{SRC}')
sys.path.insert(1, site.getusersitepackages())
import uvicorn
from croonify.api.server import app
uvicorn.run(app, host='{args.host}', port={args.port}, log_level='{log_level}')
"""
    ]

    # Use CREATE_NEW_PROCESS_GROUP on Windows so Ctrl+C is handled properly
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    stdout = None if args.verbose else subprocess.DEVNULL
    stderr = None if args.verbose else subprocess.DEVNULL

    proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr, **kwargs)

    # ── Wait for the server to be ready ───────────────────────────────────────
    ready = wait_for_server(args.port, timeout=30)
    print()  # newline after the dots

    if not ready:
        fail("Server did not start within 30 seconds.")
        proc.terminate()
        sys.exit(1)

    ok(f"Server is ready!  (PID {proc.pid})")
    print()

    url = f"http://localhost:{args.port}"
    print(LINE)
    print(f"  Web UI   : {url}/")
    print(f"  API docs : {url}/docs")
    print(f"  Health   : {url}/health")
    print(LINE)
    print()

    # ── Open browser ──────────────────────────────────────────────────────────
    if not args.no_browser:
        info(f"Opening browser -> {url}")
        open_browser(url)

    print("  Press Ctrl+C to stop the server.")
    print()

    # ── Keep alive — forward Ctrl+C to child ──────────────────────────────────
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n  Stopping Croonify server...")
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
