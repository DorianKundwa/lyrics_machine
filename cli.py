"""Croonify command-line interface.

Subcommands
-----------
align   — align a lyrics file to an audio file and print/save JSON timestamps
serve   — launch the FastAPI HTTP server

Usage examples
--------------
# Align using WhisperX (requires installation):
croonify-cli align --audio song.mp3 --lyrics lyrics.txt

# Align using built-in Viterbi aligner (no extra dependencies):
croonify-cli align --audio song.wav --lyrics lyrics.txt --aligner viterbi

# Save output to a file:
croonify-cli align --audio song.wav --lyrics lyrics.txt --output result.json

# Read lyrics from stdin:
echo "Hello world" | croonify-cli align --audio song.wav --lyrics -

# Start the API server:
croonify-cli serve --port 8000

# Start server with a custom config:
croonify-cli serve --config my_config.yaml --port 9000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_step(msg: str) -> None:
    """Print a progress message to stderr so it doesn't pollute stdout JSON."""
    print(f"  ▶  {msg}", file=sys.stderr, flush=True)


def _print_header() -> None:
    print(
        "\n╔══════════════════════════════════════╗\n"
        "║   🎵  Croonify  •  Lyrics Sync AI   ║\n"
        "╚══════════════════════════════════════╝\n",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Align subcommand
# ---------------------------------------------------------------------------

def cmd_align(args: argparse.Namespace) -> int:
    """Run the alignment pipeline from CLI arguments.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    _configure_logging(args.verbose)
    _print_header()
    logger = logging.getLogger("croonify.cli")

    # --- Validate audio path -------------------------------------------------
    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        return 1

    # --- Read lyrics ---------------------------------------------------------
    if args.lyrics == "-":
        _print_step("Reading lyrics from stdin…")
        lyrics_text = sys.stdin.read()
    else:
        lyrics_path = Path(args.lyrics)
        if not lyrics_path.exists():
            logger.error("Lyrics file not found: %s", lyrics_path)
            print(f"Error: lyrics file not found: {lyrics_path}", file=sys.stderr)
            return 1
        _print_step(f"Reading lyrics from {lyrics_path.name}…")
        lyrics_text = lyrics_path.read_text(encoding="utf-8")

    if not lyrics_text.strip():
        print("Error: lyrics text is empty.", file=sys.stderr)
        return 1

    # --- Build pipeline ------------------------------------------------------
    _print_step("Initializing Croonify pipeline…")
    try:
        from croonify.pipeline import SyncPipeline
    except ImportError as exc:
        print(f"Import error — is the croonify package installed? {exc}", file=sys.stderr)
        return 1

    pipeline_kwargs: dict = {}
    if args.config:
        pipeline_kwargs["config_path"] = str(args.config)

    try:
        pipeline = SyncPipeline(**pipeline_kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to initialize pipeline: %s", exc)
        print(f"Pipeline initialization error: {exc}", file=sys.stderr)
        return 1

    # --- Run alignment -------------------------------------------------------
    _print_step(
        f"Aligning '{audio_path.name}' with aligner={args.aligner}, "
        f"lang={args.language}, vocal_sep={not args.no_vocal_separation}…"
    )

    try:
        result = pipeline.align(
            audio_path=str(audio_path),
            lyrics_text=lyrics_text,
            language=args.language,
            use_vocal_separation=not args.no_vocal_separation,
            aligner=args.aligner,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Alignment failed: %s", exc, exc_info=args.verbose)
        print(f"\nAlignment error: {exc}", file=sys.stderr)
        return 1

    # --- Print summary -------------------------------------------------------
    meta = result.metadata
    print(
        f"\n✅  Done in {meta.get('processing_time_s', 0):.1f} s | "
        f"{meta.get('word_count', 0)} words | "
        f"{meta.get('line_count', 0)} lines | "
        f"language: {meta.get('language_detected', 'unknown')} | "
        f"low-conf: {meta.get('low_confidence_count', 0)}",
        file=sys.stderr,
    )

    # --- Output --------------------------------------------------------------
    output_json = result.to_json(indent=2)

    if args.output and args.output != "-":
        out_path = Path(args.output)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"   Result saved to: {out_path}", file=sys.stderr)
    else:
        print(output_json)

    return 0


# ---------------------------------------------------------------------------
# Serve subcommand
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the Croonify FastAPI server."""
    _configure_logging(getattr(args, "verbose", False))
    _print_header()
    logger = logging.getLogger("croonify.cli.serve")

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed.  Install with: pip install uvicorn[standard]", file=sys.stderr)
        return 1

    # Load config if provided
    pipeline_config: Optional[dict] = None
    if args.config:
        try:
            import yaml
            with open(args.config, encoding="utf-8") as f:
                pipeline_config = yaml.safe_load(f)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load config %s: %s", args.config, exc)

    # Create the app with the given config
    try:
        from croonify.api.server import create_app
        app = create_app(config=pipeline_config or {})
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to create app: {exc}", file=sys.stderr)
        return 1

    host = args.host
    port = args.port

    print(f"\n🚀  Croonify server starting at http://{host}:{port}", file=sys.stderr)
    print(f"   API docs: http://localhost:{port}/docs", file=sys.stderr)
    print(f"   Health:   http://localhost:{port}/health\n", file=sys.stderr)

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="croonify-cli",
        description="Croonify — AI-powered lyrics synchronization engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ---- align ---------------------------------------------------------------
    align_p = sub.add_parser(
        "align",
        help="Align lyrics to audio and output JSON timestamps",
        description="Run the full Croonify alignment pipeline.",
    )
    align_p.add_argument(
        "--audio",
        required=True,
        metavar="PATH",
        help="Path to the input audio file (WAV, MP3, FLAC, OGG, M4A …)",
    )
    align_p.add_argument(
        "--lyrics",
        required=True,
        metavar="PATH_OR_DASH",
        help="Path to the lyrics text file, or '-' to read from stdin",
    )
    align_p.add_argument(
        "--output", "-o",
        default="-",
        metavar="PATH",
        help="Output file for JSON result (default: stdout)",
    )
    align_p.add_argument(
        "--language", "-l",
        default="auto",
        metavar="LANG",
        help="ISO-639-1 language code (e.g. 'en', 'es') or 'auto' (default: auto)",
    )
    align_p.add_argument(
        "--aligner", "-a",
        choices=["whisperx", "viterbi"],
        default="whisperx",
        help="Alignment engine to use (default: whisperx)",
    )
    align_p.add_argument(
        "--no-vocal-separation",
        action="store_true",
        default=False,
        help="Disable Demucs vocal separation pre-processing",
    )
    align_p.add_argument(
        "--config", "-c",
        default=None,
        metavar="YAML_PATH",
        help="Optional YAML config file (overrides built-in defaults)",
    )
    align_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    align_p.set_defaults(func=cmd_align)

    # ---- serve ---------------------------------------------------------------
    serve_p = sub.add_parser(
        "serve",
        help="Start the Croonify REST API server",
        description="Launch the Croonify FastAPI / uvicorn server.",
    )
    serve_p.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="Bind host (default: 0.0.0.0)",
    )
    serve_p.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        metavar="PORT",
        help="Bind port (default: 8000)",
    )
    serve_p.add_argument(
        "--config", "-c",
        default=None,
        metavar="YAML_PATH",
        help="Optional YAML config file",
    )
    serve_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    serve_p.set_defaults(func=cmd_serve)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point — invoked by the ``croonify-cli`` console script."""
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
