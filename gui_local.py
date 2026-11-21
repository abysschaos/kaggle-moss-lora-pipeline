"""Thin launcher entrypoint preserving full compatibility with existing startup scripts and .exe wrapper."""

from core import load_app_version, resolve_cli_python
from ui import ModernLocalTrainerGUI, run_gui

APP_VERSION = load_app_version()
_resolve_cli_python = resolve_cli_python

__all__ = [
    "APP_VERSION",
    "ModernLocalTrainerGUI",
    "_resolve_cli_python",
    "run_gui",
]

if __name__ == "__main__":
    run_gui()
