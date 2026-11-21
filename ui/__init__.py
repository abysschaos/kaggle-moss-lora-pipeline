"""Modular Catppuccin Mocha dark Tkinter UI package for OpenMOSS MOSS-Image LoRA Trainer."""

from .app import ModernLocalTrainerGUI, run_gui
from .theme import PALETTE, apply_dark_theme

__all__ = ["PALETTE", "apply_dark_theme", "ModernLocalTrainerGUI", "run_gui"]
