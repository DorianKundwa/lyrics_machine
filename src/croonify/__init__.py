"""Croonify — AI-powered lyrics synchronization engine."""

__version__ = "0.1.0"
__author__ = "Croonify Authors"
__license__ = "MIT"

from pathlib import Path

# Package root — useful for locating bundled resources
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

__all__ = ["__version__", "PACKAGE_ROOT", "PROJECT_ROOT"]
