"""Modular Catppuccin Mocha dark Tkinter UI package for OpenMOSS MOSS-Image LoRA Trainer."""

from .app import ModernLocalTrainerGUI, run_gui
from .theme import PALETTE, adjust_hex_color, apply_dark_theme, apply_windows_dark_titlebar

__all__ = [
    "PALETTE",
    "adjust_hex_color",
    "apply_dark_theme",
    "apply_windows_dark_titlebar",
    "ModernLocalTrainerGUI",
    "run_gui",
]
