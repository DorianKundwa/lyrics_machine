"""
Croonify server launcher.
Ensures src/ is on sys.path before uvicorn starts, so the croonify package
is importable in both the main process and any child processes.
Run with: python serve.py
"""
import sys
import os
from pathlib import Path

# ── Inject src/ and user site-packages into sys.path ──────────────────────────
ROOT = Path(__file__).parent
SRC = str(ROOT / "src")

# Prepend src so 'import croonify' works
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Also ensure user site-packages is available (in case PYTHONPATH was stripped)
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(1, user_site)

# ── Set PYTHONPATH so spawned sub-processes inherit the correct path ───────────
existing = os.environ.get("PYTHONPATH", "")
new_paths = [SRC, user_site]
os.environ["PYTHONPATH"] = os.pathsep.join(new_paths + ([existing] if existing else []))

# ── Verify imports before starting ────────────────────────────────────────────
try:
    import fastapi
    import uvicorn
    from croonify.api.server import app
    print(f"  fastapi  {fastapi.__version__}")
    print(f"  uvicorn  {uvicorn.__version__}")
    print(f"  croonify app loaded OK")
except ImportError as e:
    print(f"\nImport error: {e}", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# ── Start server ───────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

print(f"\n[Croonify] Server starting at http://{HOST}:{PORT}")
print(f"   Web UI : http://localhost:{PORT}/")
print(f"   API docs: http://localhost:{PORT}/docs")
print(f"   Health  : http://localhost:{PORT}/health\n")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

uvicorn.run(
    app,
    host=HOST,
    port=PORT,
    log_level="info",
    # No --reload: avoids subprocess path inheritance issues
)
