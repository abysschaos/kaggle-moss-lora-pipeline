"""Main orchestrator window (ModernLocalTrainerGUI) and desktop GUI entrypoint (run_gui)."""

import ctypes
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import traceback
import urllib.request
import webbrowser
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from core import (
    ADVANCED_DEFAULTS,
    NO_WINDOW,
    PROJECT_ROOT,
    build_train_command,
    compute_exposure_summary,
    count_images_in_dir,
    discover_default_comfy_models,
    force_kill_process_tree,
    get_preset_by_id,
    load_app_version,
    load_local_config,
    open_in_explorer,
    probe_gpu_quick,
    request_graceful_pause,
    rescue_latest_weights,
    resolve_max_tokens_budget,
    save_local_config,
    validate_training_params,
)
from .monitor_panel import MonitorPanel
from .tab_cloud import build_cloud_tab
from .tab_dataset import build_dataset_tab
from .tab_help import build_help_tab
from .tab_models import build_models_tab
from .tab_train import build_train_tab
from .theme import PALETTE, apply_dark_theme
from .widgets import show_text_viewer_dialog

APP_VERSION = load_app_version()


def _configure_windows_shell_ui():
    """在 Windows 下启用高分屏缩放与独立任务栏图标标识。"""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("OpenMOSS.MOSSImage.LoRATrainer.1")
    except Exception:
        pass


class ModernLocalTrainerGUI:
    """OpenMOSS MOSS-Image 5-Tab 分区式本地图形化 LoRA 训练控制台。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"OpenMOSS MOSS-Image LoRA 本地图形化训练控制台 v{APP_VERSION}")
        self.root.geometry("1420x900")
        self.root.minsize(1220, 760)

        icon_path = PROJECT_ROOT / "assets" / "moss_trainer.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except Exception:
                pass

        c = PALETTE
        self.bg_dark = c["bg_dark"]
        self.card_bg = c["card_bg"]
        self.input_bg = c["input_bg"]
        self.accent_blue = c["accent_blue"]
        self.accent_green = c["accent_green"]
        self.accent_red = c["accent_red"]
        self.accent_yellow = c["accent_yellow"]
        self.accent_purple = c["accent_purple"]
        self.accent_peach = c["accent_peach"]
        self.text_main = c["text_main"]
        self.text_sub = c["text_sub"]
        self.font_family = c["font_family"]

        apply_dark_theme(self.root)

        self.is_training = False
        self.process = None
        self._pause_requested = False
        self._user_stopped = False
        self._adv_expanded = False
        self.detected_vram_gb = 16.0

        self.config_file = PROJECT_ROOT / "local_config.json"
        saved_cfg = load_local_config(self.config_file)

        # ==================== 全局状态与训练超参变量 ====================
        default_dataset_dir = str((PROJECT_ROOT / "dataset").resolve())
        self.var_data_dir = tk.StringVar(value=saved_cfg.get("data_dir", default_dataset_dir))
        self.var_multi_subset = tk.BooleanVar(value=bool(saved_cfg.get("multi_subset", False)))

        raw_subsets = saved_cfg.get("subsets", [])
        if isinstance(raw_subsets, list) and len(raw_subsets) > 0:
            self.subsets = [dict(x) for x in raw_subsets if isinstance(x, dict)]
        else:
            self.subsets = []
        if not self.subsets:
            self.subsets = [{
                "name": "main_dataset",
                "path": self.var_data_dir.get() or default_dataset_dir,
                "repeats": int(saved_cfg.get("repeats", 6) or 6),
                "trigger": str(saved_cfg.get("trigger_word", "")),
                "enabled": True,
            }]

        self.var_comfy_dir = tk.StringVar(value=discover_default_comfy_models(saved_cfg.get("comfy_models_dir", "")))
        self.var_output_dir = tk.StringVar(value=saved_cfg.get("output_dir", str((PROJECT_ROOT / "output").resolve())))
        self.var_lora_name = tk.StringVar(value=saved_cfg.get("lora_name", "my_moss_lora"))
        self.var_trigger = tk.StringVar(value=saved_cfg.get("trigger_word", ""))

        self.var_transformer_path = tk.StringVar(value=saved_cfg.get("transformer_path", ""))
        self.var_text_encoder_path = tk.StringVar(value=saved_cfg.get("text_encoder_path", ""))
        self.var_vae_path = tk.StringVar(value=saved_cfg.get("vae_path", ""))
        self.var_copy_comfy = tk.BooleanVar(value=bool(saved_cfg.get("copy_to_comfy_loras", False)))

        self.var_precision = tk.StringVar(value=saved_cfg.get("precision", "bf16"))
        self.var_base_precision = tk.StringVar(value=saved_cfg.get("base_precision", "auto"))
        self.var_vram_preset = tk.StringVar(value=saved_cfg.get("vram_preset", "auto"))
        self.var_resolution = tk.StringVar(value=str(saved_cfg.get("resolution", "1024")))
        self.var_text_device = tk.StringVar(value=saved_cfg.get("text_device", "cpu"))
        self.var_vae_device = tk.StringVar(value=saved_cfg.get("vae_device", "cuda"))
        self.var_algo = tk.StringVar(value=saved_cfg.get("algo", "lora"))
        self.var_dora = tk.BooleanVar(value=bool(saved_cfg.get("dora", False)))
        self.var_target_preset = tk.StringVar(value=saved_cfg.get("target_preset", "attn-mlp"))
        self.var_loraplus_ratio = tk.StringVar(value=str(saved_cfg.get("loraplus_ratio", "1.0")))
        self.var_rank = tk.StringVar(value=str(saved_cfg.get("rank", "32")))
        self.var_alpha = tk.StringVar(value=str(saved_cfg.get("alpha", "32.0")))
        self.var_lr = tk.StringVar(value=str(saved_cfg.get("lr", "1e-4")))
        self.var_repeats = tk.StringVar(value=str(saved_cfg.get("repeats", "6")))
        self.var_epochs = tk.StringVar(value=str(saved_cfg.get("epochs", "4")))
        self.var_steps = tk.StringVar(value=str(saved_cfg.get("steps", "600")))
        self.var_save_every = tk.StringVar(value=str(saved_cfg.get("save_every", "100")))
        self.var_grad_acc = tk.StringVar(value=str(saved_cfg.get("grad_acc", "1")))
        self.var_timestep_mu = tk.StringVar(value=str(saved_cfg.get("timestep_mu", "-1.09")))
        self.var_tune = tk.StringVar(value=saved_cfg.get("tune", "off"))

        # 18 项折叠高阶专业参数
        self.var_lokr_factor = tk.StringVar(value=str(saved_cfg.get("lokr_factor", ADVANCED_DEFAULTS["lokr_factor"])))
        self.var_lokr_full_matrix = tk.BooleanVar(
            value=bool(saved_cfg.get("lokr_full_matrix", ADVANCED_DEFAULTS["lokr_full_matrix"]))
        )
        self.var_lokr_decompose_both = tk.BooleanVar(
            value=bool(saved_cfg.get("lokr_decompose_both", ADVANCED_DEFAULTS["lokr_decompose_both"]))
        )
        self.var_train_norms = tk.BooleanVar(value=bool(saved_cfg.get("train_norms", ADVANCED_DEFAULTS["train_norms"])))
        self.var_rank_dropout = tk.StringVar(value=str(saved_cfg.get("rank_dropout", ADVANCED_DEFAULTS["rank_dropout"])))
        self.var_module_dropout = tk.StringVar(value=str(saved_cfg.get("module_dropout", ADVANCED_DEFAULTS["module_dropout"])))
        self.var_input_dropout = tk.StringVar(value=str(saved_cfg.get("input_dropout", ADVANCED_DEFAULTS["input_dropout"])))
        self.var_optimizer = tk.StringVar(value=str(saved_cfg.get("optimizer", ADVANCED_DEFAULTS["optimizer"])))
        self.var_lr_scheduler = tk.StringVar(value=str(saved_cfg.get("lr_scheduler", ADVANCED_DEFAULTS["lr_scheduler"])))
        self.var_warmup_steps = tk.StringVar(value=str(saved_cfg.get("warmup_steps", ADVANCED_DEFAULTS["warmup_steps"])))
        self.var_weight_decay = tk.StringVar(value=str(saved_cfg.get("weight_decay", ADVANCED_DEFAULTS["weight_decay"])))
        self.var_wd_rules = tk.StringVar(value=str(saved_cfg.get("wd_rules", ADVANCED_DEFAULTS["wd_rules"])))
        self.var_ema = tk.StringVar(value=str(saved_cfg.get("ema", ADVANCED_DEFAULTS["ema"])))
        self.var_timestep = tk.StringVar(value=str(saved_cfg.get("timestep", ADVANCED_DEFAULTS["timestep"])))
        self.var_text_drop = tk.StringVar(value=str(saved_cfg.get("text_drop", ADVANCED_DEFAULTS["text_drop"])))
        self.var_flip = tk.BooleanVar(value=bool(saved_cfg.get("flip", ADVANCED_DEFAULTS["flip"])))
        self.var_seed = tk.StringVar(value=str(saved_cfg.get("seed", ADVANCED_DEFAULTS["seed"])))
        self.var_save_format = tk.StringVar(value=str(saved_cfg.get("save_format", ADVANCED_DEFAULTS["save_format"])))

        self.var_train_mode = tk.StringVar(value=saved_cfg.get("train_mode", "fresh"))
        self.var_resume_path = tk.StringVar(value=saved_cfg.get("resume_path", ""))

        # 状态文本标签变量
        self.var_trainer_status = tk.StringVar(value="正在检测 KohakuMossTrainer 引擎状态...")
        self.var_dataset_stat = tk.StringVar(value="当前共检测到 0 张图片")
        self.var_output_preview = tk.StringVar(value="")
        self.var_quick_status_summary = tk.StringVar(value="")
        self.var_math_info = tk.StringVar(value="数据曝光计算器就绪...")
        self.var_model_scan_stat = tk.StringVar(value="提示：可点击【🔍 自动扫描底模】从 ComfyUI 路径探测现有模型")
        self.var_resume_status = tk.StringVar(value="就绪：全新从头训练")

        # 变量联动监听
        self.var_lora_name.trace_add("write", lambda *_: (self.update_output_preview(), self.refresh_resume_status()))
        self.var_output_dir.trace_add("write", lambda *_: self.update_output_preview())
        self.var_resume_path.trace_add("write", lambda *_: self.refresh_resume_status())

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(120, self._apply_initial_sash)

        self.check_trainer_status()
        self.update_dataset_stats()
        self.refresh_subset_tree()
        self.update_output_preview()
        self.refresh_resume_status()
        self.recalculate_math()
        self.auto_scan_models(silent=True)
        self.root.after(200, lambda: self.monitor.load_history_metrics(silent=True))

    # ==================== 主界面与 5-Tab 架构搭建 ====================
    def build_ui(self):
        # 1. 顶部全局状态栏
        header_frame = tk.Frame(self.root, bg=self.card_bg, height=52)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 6))

        tk.Label(
            header_frame,
            text=f"OpenMOSS MOSS-Image 本地 LoRA 训练控制台  v{APP_VERSION}",
            bg=self.card_bg,
            fg=self.accent_blue,
            font=(self.font_family, 13, "bold"),
        ).pack(side=tk.LEFT, padx=14, pady=10)

        gpu_info, vram_gb = probe_gpu_quick()
        self.detected_vram_gb = vram_gb
        tk.Label(
            header_frame,
            text=f"设备状态: {gpu_info}",
            bg=self.card_bg,
            fg=self.accent_green,
            font=(self.font_family, 9, "bold"),
        ).pack(side=tk.RIGHT, padx=14, pady=10)

        # 2. 左右可拖拽分隔主面板
        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.bg_dark, sashwidth=6, sashrelief=tk.RAISED)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left_container = tk.Frame(self.main_paned, bg=self.bg_dark)
        self.main_paned.add(left_container, minsize=610, stretch="always")

        right_container = tk.Frame(self.main_paned, bg=self.card_bg)
        self.main_paned.add(right_container, minsize=480, stretch="always")

        # 3. 左侧 5 大功能选项卡 (ttk.Notebook)
        self.notebook = ttk.Notebook(left_container, style="Dark.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        tab_specs = [
            (" 🚀 快速训练 ", build_train_tab),
            (" 📂 数据与续炼 ", build_dataset_tab),
            (" 📦 底模与框架 ", build_models_tab),
            (" ☁️ 云端遥控 ", build_cloud_tab),
            (" 📖 指南与工具 ", build_help_tab),
        ]
        self.tab_frames = []
        self.tab_scrolls = []
        for title, builder in tab_specs:
            frame = ttk.Frame(self.notebook, style="TabContent.TFrame")
            self.notebook.add(frame, text=title)
            self.tab_frames.append(frame)
            self.tab_scrolls.append(builder(frame, self))

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 4. 右侧常驻监控看板 (并保留向后兼容引用属性)
        self.monitor = MonitorPanel(right_container, self.root, self)
        self.right_paned = self.monitor.right_paned
        self.loss_canvas = self.monitor.loss_canvas
        self.progress_canvas = self.monitor.progress_canvas
        self.console = self.monitor.console
        self.loss_history = self.monitor.loss_history
        self.btn_start = self.monitor.btn_start
        self.btn_pause = self.monitor.btn_pause
        self.btn_stop = self.monitor.btn_stop

    def _apply_initial_sash(self):
        """窗口首次布局稳定后设置左右 52%/48% 黄金比例。"""
        try:
            self.root.update_idletasks()
            total_w = self.main_paned.winfo_width()
            if total_w > 400:
                left_w = max(630, min(total_w - 490, int(total_w * 0.52)))
                self.main_paned.sash_place(0, left_w, 0)
            self.monitor.apply_initial_sash()
        except Exception:
            pass

    def _on_tab_changed(self, _event=None):
        """切换选项卡时解绑旧滚轮并刷新新卡片滚动区，避免空白边距或滚轮漂移。"""
        try:
            self.root.unbind_all("<MouseWheel>")
        except Exception:
            pass
        try:
            idx = self.notebook.index(self.notebook.select())
            if 0 <= idx < len(self.tab_scrolls):
                scroll = self.tab_scrolls[idx]
                self.root.after_idle(scroll.refresh_scroll_region)
        except Exception:
            pass

    def select_tab(self, index: int):
        """切换至指定索引的功能选项卡（0~4）。"""
        if 0 <= index < len(self.tab_frames):
            self.notebook.select(self.tab_frames[index])

    # ==================== 配置收集、持久化与关闭保护 ====================
    def _collect_raw_params(self) -> dict:
        """收集全量图形界面变量（单/多子集、基础超参、18 项高阶参数、底模与续炼路径）。"""
        return {
            "data_dir": self.var_data_dir.get().strip(),
            "multi_subset": bool(self.var_multi_subset.get()),
            "subsets": self.subsets,
            "comfy_models_dir": self.var_comfy_dir.get().strip(),
            "output_dir": self.var_output_dir.get().strip(),
            "lora_name": self.var_lora_name.get().strip() or "my_moss_lora",
            "trigger_word": self.var_trigger.get().strip(),
            "transformer_path": self.var_transformer_path.get().strip(),
            "text_encoder_path": self.var_text_encoder_path.get().strip(),
            "vae_path": self.var_vae_path.get().strip(),
            "copy_to_comfy_loras": bool(self.var_copy_comfy.get()),
            "precision": self.var_precision.get(),
            "base_precision": self.var_base_precision.get(),
            "vram_preset": self.var_vram_preset.get(),
            "resolution": self.var_resolution.get(),
            "text_device": self.var_text_device.get(),
            "vae_device": self.var_vae_device.get(),
            "algo": self.var_algo.get(),
            "dora": bool(self.var_dora.get()),
            "target_preset": self.var_target_preset.get(),
            "loraplus_ratio": self.var_loraplus_ratio.get().strip(),
            "rank": self.var_rank.get().strip(),
            "alpha": self.var_alpha.get().strip(),
            "lr": self.var_lr.get().strip(),
            "repeats": self.var_repeats.get().strip(),
            "epochs": self.var_epochs.get().strip(),
            "steps": self.var_steps.get().strip(),
            "save_every": self.var_save_every.get().strip(),
            "grad_acc": self.var_grad_acc.get().strip(),
            "timestep_mu": self.var_timestep_mu.get().strip(),
            "tune": self.var_tune.get(),
            "lokr_factor": self.var_lokr_factor.get().strip(),
            "lokr_full_matrix": bool(self.var_lokr_full_matrix.get()),
            "lokr_decompose_both": bool(self.var_lokr_decompose_both.get()),
            "train_norms": bool(self.var_train_norms.get()),
            "rank_dropout": self.var_rank_dropout.get().strip(),
            "module_dropout": self.var_module_dropout.get().strip(),
            "input_dropout": self.var_input_dropout.get().strip(),
            "optimizer": self.var_optimizer.get(),
            "lr_scheduler": self.var_lr_scheduler.get(),
            "warmup_steps": self.var_warmup_steps.get().strip(),
            "weight_decay": self.var_weight_decay.get().strip(),
            "wd_rules": self.var_wd_rules.get(),
            "ema": self.var_ema.get().strip(),
            "timestep": self.var_timestep.get(),
            "text_drop": self.var_text_drop.get().strip(),
            "flip": bool(self.var_flip.get()),
            "seed": self.var_seed.get().strip(),
            "save_format": self.var_save_format.get(),
            "train_mode": self.var_train_mode.get(),
            "resume_path": self.var_resume_path.get().strip(),
        }

    def save_local_config(self):
        save_local_config(self.config_file, self._collect_raw_params())

    def on_closing(self):
        """关闭主窗口前保存本机配置，并在训练未停机时弹窗防误触确认强杀。"""
        self.save_local_config()
        if self.is_training and self.process is not None and self.process.poll() is None:
            confirm = messagebox.askyesno(
                "训练任务正在后台运行",
                "检测到当前 LoRA 训练尚未完成！\n"
                "如果直接关闭窗口，将强制结束训练且可能导致当前未存盘的进度丢失。\n\n"
                "是否确认强制终止训练并退出？（建议先点击【保存当前进度并暂停】）",
            )
            if not confirm:
                return
            self._user_stopped = True
            self.monitor.stop_metrics_polling()
            force_kill_process_tree(self.process.pid)
        else:
            self.monitor.stop_metrics_polling()
        self.root.destroy()

    # ==================== 动态交互、折叠抽屉与 8 套预设应用 ====================
    def on_dynamic_controls_changed(self):
        """根据 Algo=LoKr、DoRA 开启、Timestep=uniform 动态显示或锁定对应控件。"""
        try:
            algo = (self.var_algo.get() or "lora").strip().lower()
            dora = bool(self.var_dora.get())
            ts_mode = (self.var_timestep.get() or "moss").strip().lower()

            if hasattr(self, "lokr_row"):
                if algo == "lokr":
                    self.lokr_row.pack(fill=tk.X, pady=(0, 4), before=self.reg_row)
                else:
                    self.lokr_row.pack_forget()

            if hasattr(self, "cb_input_dropout") and hasattr(self, "lbl_dora_drop_lock"):
                if dora:
                    if self.var_input_dropout.get() != "0.0":
                        self.var_input_dropout.set("0.0")
                    self.cb_input_dropout.config(state="disabled")
                    self.lbl_dora_drop_lock.config(text="DoRA开启时已自动锁0")
                else:
                    self.cb_input_dropout.config(state="normal")
                    self.lbl_dora_drop_lock.config(text="")

            if hasattr(self, "cb_mu"):
                if ts_mode == "uniform":
                    self.cb_mu.config(state="disabled")
                else:
                    self.cb_mu.config(state="normal")

            if hasattr(self, "tab_scrolls") and self.tab_scrolls:
                self.root.after_idle(self.tab_scrolls[0].refresh_scroll_region)
        except Exception:
            pass

    def toggle_advanced_drawer(self):
        """展开或折叠【高阶算法 / 防过拟合 / 优化器调优区】专业抽屉。"""
        if not hasattr(self, "adv_drawer_frame"):
            return
        if self._adv_expanded:
            self.adv_drawer_frame.pack_forget()
            self.btn_adv_toggle.config(text="▸ 展开高阶算法 / 防过拟合 / 优化器调优区 (18项专业参数)")
            self._adv_expanded = False
        else:
            self.adv_drawer_frame.pack(fill=tk.X, pady=(6, 2))
            self.btn_adv_toggle.config(text="▾ 收起高阶算法 / 防过拟合 / 优化器调优区")
            self._adv_expanded = True
            self.on_dynamic_controls_changed()
        if self.tab_scrolls:
            self.root.after_idle(self.tab_scrolls[0].refresh_scroll_region)

    def _reset_advanced_defaults(self):
        """加载常规基础预设前重置高阶参数默认态，防止上一模板残留特殊优化器或 Dropout。"""
        self.var_lokr_factor.set(ADVANCED_DEFAULTS["lokr_factor"])
        self.var_lokr_full_matrix.set(ADVANCED_DEFAULTS["lokr_full_matrix"])
        self.var_lokr_decompose_both.set(ADVANCED_DEFAULTS["lokr_decompose_both"])
        self.var_train_norms.set(ADVANCED_DEFAULTS["train_norms"])
        self.var_rank_dropout.set(ADVANCED_DEFAULTS["rank_dropout"])
        self.var_module_dropout.set(ADVANCED_DEFAULTS["module_dropout"])
        self.var_input_dropout.set(ADVANCED_DEFAULTS["input_dropout"])
        self.var_optimizer.set(ADVANCED_DEFAULTS["optimizer"])
        self.var_lr_scheduler.set(ADVANCED_DEFAULTS["lr_scheduler"])
        self.var_warmup_steps.set(ADVANCED_DEFAULTS["warmup_steps"])
        self.var_weight_decay.set(ADVANCED_DEFAULTS["weight_decay"])
        self.var_wd_rules.set(ADVANCED_DEFAULTS["wd_rules"])
        self.var_ema.set(ADVANCED_DEFAULTS["ema"])
        self.var_timestep.set(ADVANCED_DEFAULTS["timestep"])
        self.var_text_drop.set(ADVANCED_DEFAULTS["text_drop"])
        self.var_flip.set(ADVANCED_DEFAULTS["flip"])
        self.var_seed.set(ADVANCED_DEFAULTS["seed"])
        self.var_save_format.set(ADVANCED_DEFAULTS["save_format"])

    def apply_preset(self, preset_id: str):
        """应用 core.presets 中声明的 8 套场景预设并同步 UI 联动。"""
        preset = get_preset_by_id(preset_id)
        if not preset:
            return

        self._reset_advanced_defaults()
        self.var_multi_subset.set(bool(preset.get("multi_subset", False)))
        self._apply_subset_panel_visibility()

        var_map = {
            "repeats": self.var_repeats,
            "epochs": self.var_epochs,
            "rank": self.var_rank,
            "alpha": self.var_alpha,
            "algo": self.var_algo,
            "dora": self.var_dora,
            "target_preset": self.var_target_preset,
            "loraplus_ratio": self.var_loraplus_ratio,
            "lr": self.var_lr,
            "grad_acc": self.var_grad_acc,
            "timestep_mu": self.var_timestep_mu,
            "lora_name": self.var_lora_name,
            "trigger_word": self.var_trigger,
            "lokr_factor": self.var_lokr_factor,
            "lokr_full_matrix": self.var_lokr_full_matrix,
            "lokr_decompose_both": self.var_lokr_decompose_both,
            "train_norms": self.var_train_norms,
            "rank_dropout": self.var_rank_dropout,
            "module_dropout": self.var_module_dropout,
            "input_dropout": self.var_input_dropout,
            "text_drop": self.var_text_drop,
            "ema": self.var_ema,
            "optimizer": self.var_optimizer,
            "timestep": self.var_timestep,
        }
        for key, val in preset.get("values", {}).items():
            if key in var_map:
                var_map[key].set(val)

        if preset.get("align_steps", False):
            self.align_steps_by_formula(silent=True)

        self.on_dynamic_controls_changed()
        self.recalculate_math()
        self.update_quick_status_summary()
        messagebox.showinfo(preset["dialog_title"], preset["dialog_message"])

    # ==================== 引擎管理、底模扫描与输出预览 ====================
    def check_trainer_status(self):
        trainer_dir = PROJECT_ROOT / "KohakuMossTrainer"
        core_file = trainer_dir / "scripts" / "train_moss_lora.py"
        if core_file.exists():
            self.var_trainer_status.set("✅ 已就绪：KohakuMossTrainer 引擎完整存在（可直接启动）")
        elif trainer_dir.exists():
            self.var_trainer_status.set("⚠️ 目录存在但核心脚本缺失，请点击右方【一键安装 / 同步引擎】修复")
        else:
            self.var_trainer_status.set("❌ 未安装：尚未下载 KohakuMossTrainer 引擎，请点击右方按钮自动拉取")

    def open_trainer_folder(self):
        try:
            open_in_explorer(str(PROJECT_ROOT / "KohakuMossTrainer"))
        except Exception as err:
            messagebox.showerror("打开失败", str(err))

    def open_trainer_github(self):
        webbrowser.open("https://github.com/KohakuBlueleaf/KohakuMossTrainer")

    def patch_trainer_manually(self):
        trainer_dir = PROJECT_ROOT / "KohakuMossTrainer"
        if not trainer_dir.exists():
            messagebox.showwarning("引擎未找到", "请先点击【一键安装 / 同步引擎】下载 KohakuMossTrainer。")
            return
        try:
            import train_local

            train_local.apply_compatibility_patches(trainer_dir)
            self.monitor.log("[系统] 已完成 KohakuMossTrainer 全套兼容补丁修复。\n")
            messagebox.showinfo("修复完成", "已注入 PyTorch 兼容补丁与调度器防护！")
        except Exception as err:
            messagebox.showerror("修补失败", str(err))

    def download_or_clone_trainer(self):
        """后台线程下载或同步 KohakuMossTrainer（支持 git 镜像回退与 ZIP 解压 + 自动打补丁）。"""
        def _worker():
            trainer_dir = PROJECT_ROOT / "KohakuMossTrainer"
            core_file = trainer_dir / "scripts" / "train_moss_lora.py"
            self.root.after(0, lambda: self.var_trainer_status.set("⏳ 正在安装/同步 KohakuMossTrainer，请稍候..."))
            self.root.after(0, lambda: self.monitor.log("[引擎] 开始拉取 KohakuMossTrainer 仓库...\n"))

            success = False
            if shutil.which("git"):
                urls = [
                    "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git",
                    "https://ghfast.top/https://github.com/KohakuBlueleaf/KohakuMossTrainer.git",
                ]
                if (trainer_dir / ".git").exists():
                    res = subprocess.run(
                        ["git", "-C", str(trainer_dir), "pull", "--ff-only"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
                    )
                    if res.returncode == 0 and core_file.exists():
                        success = True
                if not success:
                    for url in urls:
                        if trainer_dir.exists() and not core_file.exists():
                            shutil.rmtree(trainer_dir, ignore_errors=True)
                        if not trainer_dir.exists():
                            res = subprocess.run(
                                ["git", "clone", "--depth", "1", url, str(trainer_dir)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
                            )
                            if res.returncode == 0 and core_file.exists():
                                success = True
                                break

            if not success:
                zip_urls = [
                    "https://github.com/KohakuBlueleaf/KohakuMossTrainer/archive/refs/heads/main.zip",
                    "https://ghfast.top/https://github.com/KohakuBlueleaf/KohakuMossTrainer/archive/refs/heads/main.zip",
                ]
                tmp_zip = PROJECT_ROOT / "workspace" / "trainer_main.zip"
                tmp_zip.parent.mkdir(parents=True, exist_ok=True)
                for zurl in zip_urls:
                    try:
                        req = urllib.request.Request(zurl, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=45) as resp, open(tmp_zip, "wb") as out_f:
                            out_f.write(resp.read())
                        extract_tmp = PROJECT_ROOT / "workspace" / "_trainer_extract"
                        shutil.rmtree(extract_tmp, ignore_errors=True)
                        with zipfile.ZipFile(tmp_zip, "r") as zf:
                            zf.extractall(extract_tmp)
                        sub = extract_tmp / "KohakuMossTrainer-main"
                        if sub.is_dir():
                            shutil.rmtree(trainer_dir, ignore_errors=True)
                            shutil.move(str(sub), str(trainer_dir))
                        shutil.rmtree(extract_tmp, ignore_errors=True)
                        tmp_zip.unlink(missing_ok=True)
                        if core_file.exists():
                            success = True
                            break
                    except Exception:
                        pass

            if success:
                try:
                    import train_local

                    train_local.apply_compatibility_patches(trainer_dir)
                except Exception:
                    pass
                self.root.after(0, self.check_trainer_status)
                self.root.after(0, lambda: self.monitor.log("[引擎] KohakuMossTrainer 安装同步与兼容补丁已就绪！\n"))
                self.root.after(0, lambda: messagebox.showinfo("同步成功", "KohakuMossTrainer 核心引擎已安装并完成兼容修补！"))
            else:
                self.root.after(0, self.check_trainer_status)
                self.root.after(0, lambda: self.monitor.log("❌ [引擎] 无法自动下载，请检查网络或手动克隆到项目根目录。\n"))
                self.root.after(0, lambda: messagebox.showerror("同步失败", "GitHub 与镜像通道均超时，请检查网络连接或手动下载。"))

        threading.Thread(target=_worker, daemon=True).start()

    def auto_scan_models(self, silent: bool = True):
        """调用 train_local.probe_local_models() 自动扫描本地 ComfyUI 与项目 models/ 底座权重。"""
        try:
            import train_local

            comfy_dir = self.var_comfy_dir.get().strip() or None
            prefer_bf16 = (self.var_base_precision.get() or "").lower() in ("compute", "fp8_compute")
            found = train_local.probe_local_models(comfy_dir, prefer_bf16_te=prefer_bf16)

            for key, var in (
                ("transformer", self.var_transformer_path),
                ("text_encoder", self.var_text_encoder_path),
                ("vae", self.var_vae_path),
            ):
                if found.get(key) and not var.get().strip():
                    var.set(str(found[key]))

            t_ok = bool(self.var_transformer_path.get().strip() or found.get("transformer"))
            te_ok = bool(self.var_text_encoder_path.get().strip() or found.get("text_encoder"))
            v_ok = bool(self.var_vae_path.get().strip() or found.get("vae"))
            ready_cnt = sum((t_ok, te_ok, v_ok))
            if ready_cnt == 3:
                self.var_model_scan_stat.set("✅ 扫描完成：Transformer / Qwen3.5-9B / FLUX.2 VAE 三核心底模已全部锁定！")
            else:
                self.var_model_scan_stat.set(
                    f"⚠️ 仅命中 {ready_cnt}/3 个底模文件（Missing: "
                    f"{'Transformer ' if not t_ok else ''}"
                    f"{'TextEnc ' if not te_ok else ''}"
                    f"{'VAE' if not v_ok else ''}），可点击下方浏览手动指定或放入 models/。"
                )
            if not silent:
                messagebox.showinfo(
                    "底模扫描结果",
                    f"扫描状态 ({ready_cnt}/3)：\n"
                    f"• Transformer: {self.var_transformer_path.get() or found.get('transformer') or '未找到'}\n"
                    f"• Text Encoder: {self.var_text_encoder_path.get() or found.get('text_encoder') or '未找到'}\n"
                    f"• VAE: {self.var_vae_path.get() or found.get('vae') or '未找到'}",
                )
        except Exception as err:
            self.var_model_scan_stat.set(f"⚠️ 自动扫描异常: {err}")

    def update_output_preview(self):
        name = self.var_lora_name.get().strip() or "my_lora"
        out_root = self.var_output_dir.get().strip() or str((PROJECT_ROOT / "output").resolve())
        dest_dir = Path(out_root) / name
        self.var_output_preview.set(f"最终归档输出位置：{dest_dir / (name + '_comfy.safetensors')}")

    def open_lora_output_folder(self):
        name = self.var_lora_name.get().strip() or "my_lora"
        out_root = self.var_output_dir.get().strip() or str((PROJECT_ROOT / "output").resolve())
        target = Path(out_root) / name
        try:
            open_in_explorer(str(target if target.exists() else Path(out_root)))
        except Exception as err:
            messagebox.showerror("打开失败", str(err))

    # ==================== 图集目录、多子集管理、健康体检与续炼状态 ====================
    def browse_data_dir(self):
        d = filedialog.askdirectory(title="选择训练图集目录（含图片与同名 .txt 标注）")
        if d:
            self.var_data_dir.set(d)
            self.update_dataset_stats()
            self.recalculate_math()

    def browse_comfy_dir(self):
        d = filedialog.askdirectory(title="选择 ComfyUI 根目录或 models 文件夹")
        if d:
            self.var_comfy_dir.set(d)
            self.auto_scan_models(silent=True)

    def browse_output_dir(self):
        d = filedialog.askdirectory(title="选择 LoRA 成果输出归档根目录")
        if d:
            self.var_output_dir.set(d)
            self.update_output_preview()

    def update_dataset_stats(self):
        cnt = count_images_in_dir(self.var_data_dir.get().strip())
        self.var_dataset_stat.set(f"当前有效图片数: {cnt} 张（目录: {self.var_data_dir.get().strip()}）")
        self.update_quick_status_summary()

    def update_quick_status_summary(self):
        """同步更新 Tab 1 快速状态条文本（数据集模式/总图数 + 续炼模式）。"""
        mode_labels = {"fresh": "全新训练", "resume": "断点续炼", "init_from": "继承权重再微调"}
        mode_txt = mode_labels.get((self.var_train_mode.get() or "fresh").strip(), "全新训练")
        if self.var_multi_subset.get():
            enabled_subs = [s for s in self.subsets if s.get("enabled", True)]
            total_imgs = sum(count_images_in_dir(s.get("path", "")) for s in enabled_subs)
            data_txt = f"[多子集] 启用 {len(enabled_subs)} 个 / 共 {total_imgs} 张"
        else:
            cnt = count_images_in_dir(self.var_data_dir.get().strip())
            data_txt = f"[单目录] {cnt} 张图片"
        self.var_quick_status_summary.set(f"📂 当前图集: {data_txt}  |  模式: {mode_txt}")

    def _apply_subset_panel_visibility(self):
        if not hasattr(self, "single_data_frame") or not hasattr(self, "multi_data_frame"):
            return
        if self.var_multi_subset.get():
            self.single_data_frame.pack_forget()
            self.multi_data_frame.pack(fill=tk.X, pady=(2, 0), before=self.output_dirs_anchor)
        else:
            self.multi_data_frame.pack_forget()
            self.single_data_frame.pack(fill=tk.X, pady=(2, 0), before=self.output_dirs_anchor)
        if hasattr(self, "tab_scrolls") and len(self.tab_scrolls) > 1:
            self.root.after_idle(self.tab_scrolls[1].refresh_scroll_region)

    def toggle_multi_subset_mode(self):
        self._apply_subset_panel_visibility()
        self.refresh_subset_tree()
        self.recalculate_math()
        self.update_quick_status_summary()

    def refresh_subset_tree(self):
        if not hasattr(self, "subset_tree"):
            return
        for iid in self.subset_tree.get_children():
            self.subset_tree.delete(iid)
        for idx, sub in enumerate(self.subsets):
            cnt = count_images_in_dir(sub.get("path", ""))
            en_flag = "✅" if sub.get("enabled", True) else "⏸️"
            self.subset_tree.insert(
                "", tk.END, iid=str(idx),
                values=(
                    en_flag,
                    sub.get("name", f"subset_{idx+1}"),
                    cnt,
                    sub.get("repeats", 1),
                    sub.get("trigger", ""),
                    sub.get("path", ""),
                ),
            )
        self.update_quick_status_summary()

    def _open_subset_editor_dialog(self, index=None):
        """打开暗色子集编辑弹窗（支持新增与编辑）。"""
        is_new = index is None
        existing = self.subsets[index] if not is_new else {
            "name": f"subset_{len(self.subsets) + 1}",
            "path": "",
            "repeats": 4,
            "trigger": "",
            "enabled": True,
        }

        dlg = tk.Toplevel(self.root)
        dlg.title("添加训练子集" if is_new else f"编辑子集：{existing.get('name', '')}")
        dlg.configure(bg=self.card_bg)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("540x255")

        v_name = tk.StringVar(value=str(existing.get("name", "")))
        v_path = tk.StringVar(value=str(existing.get("path", "")))
        v_rep = tk.StringVar(value=str(existing.get("repeats", 4)))
        v_trig = tk.StringVar(value=str(existing.get("trigger", "")))
        v_en = tk.BooleanVar(value=bool(existing.get("enabled", True)))

        body = tk.Frame(dlg, bg=self.card_bg, padx=14, pady=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)

        tk.Label(body, text="子集名称 (英文/数字):", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=0, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_name, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=4)

        tk.Label(body, text="本地图集路径:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=1, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_path, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT, font=("Consolas", 8)).grid(row=1, column=1, sticky="ew", padx=(8, 4), pady=4)
        tk.Button(
            body, text="浏览...", bg=self.accent_blue, fg="#11111b", relief=tk.FLAT, font=(self.font_family, 8, "bold"),
            command=lambda: v_path.set(filedialog.askdirectory(title="选择子集图片文件夹") or v_path.get()),
        ).grid(row=1, column=2, pady=4)

        tk.Label(body, text="独立重复次数 (Repeats):", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=2, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_rep, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=4)

        tk.Label(body, text="专属触发词 (可选前置):", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=3, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_trig, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=3, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=4)

        tk.Checkbutton(
            body, text="启用此子集参与训练", variable=v_en,
            bg=self.card_bg, fg=self.accent_green, selectcolor=self.input_bg, font=(self.font_family, 9),
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=6)

        def _save():
            try:
                rep_val = max(1, int(v_rep.get().strip()))
            except Exception:
                messagebox.showerror("格式错误", "Repeats 必须为 >= 1 的整数", parent=dlg)
                return
            entry = {
                "name": v_name.get().strip() or f"subset_{len(self.subsets)+1}",
                "path": v_path.get().strip(),
                "repeats": rep_val,
                "trigger": v_trig.get().strip(),
                "enabled": bool(v_en.get()),
            }
            if is_new:
                self.subsets.append(entry)
            else:
                self.subsets[index] = entry
            self.refresh_subset_tree()
            self.recalculate_math()
            self.save_local_config()
            dlg.destroy()

        btn_row = tk.Frame(body, bg=self.card_bg)
        btn_row.grid(row=5, column=0, columnspan=3, sticky="e", pady=(8, 0))
        tk.Button(btn_row, text="取消", bg=self.input_bg, fg=self.text_sub, relief=tk.FLAT, padx=14, pady=4, command=dlg.destroy).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_row, text="保存子集", bg=self.accent_green, fg="#11111b", relief=tk.FLAT, font=(self.font_family, 9, "bold"), padx=14, pady=4, command=_save).pack(side=tk.LEFT)

    def add_subset_dialog(self):
        self._open_subset_editor_dialog(index=None)

    def edit_selected_subset(self):
        sel = self.subset_tree.selection() if hasattr(self, "subset_tree") else ()
        if not sel:
            messagebox.showinfo("提示", "请先在子集列表中选中一行（或双击对应行）再编辑。")
            return
        self._open_subset_editor_dialog(index=int(sel[0]))

    def remove_selected_subset(self):
        sel = self.subset_tree.selection() if hasattr(self, "subset_tree") else ()
        if not sel:
            messagebox.showinfo("提示", "请先在子集列表中选中要删除的行。")
            return
        if len(self.subsets) <= 1:
            messagebox.showwarning("无法清空", "多子集模式下至少需保留 1 个子集。")
            return
        idx = int(sel[0])
        self.subsets.pop(idx)
        self.refresh_subset_tree()
        self.recalculate_math()
        self.save_local_config()

    def auto_balance_subset_repeats(self):
        """按最大曝光目标自动拉平各个启用子集的 Repeats（防小样本被大子集淹没）。"""
        valid = []
        for idx, sub in enumerate(self.subsets):
            if sub.get("enabled", True):
                cnt = count_images_in_dir(sub.get("path", ""))
                if cnt > 0:
                    valid.append((idx, sub, cnt))
        if not valid:
            messagebox.showwarning("无有效图片", "已启用的子集目录中均未检测到图片，无法配平。")
            return

        target_exp = max(max(cnt * max(1, int(sub.get("repeats", 1))) for _, sub, cnt in valid), max(cnt for _, _, cnt in valid) * 4)
        summary_lines = [f"目标单轮均衡曝光基准：每子集约 {target_exp} 次/Epoch\n"]
        for idx, sub, cnt in valid:
            old_r = int(sub.get("repeats", 1))
            new_r = max(1, round(target_exp / cnt))
            self.subsets[idx]["repeats"] = new_r
            summary_lines.append(f"• {sub.get('name', idx)}: {cnt} 图 × {old_r} → {new_r} (单轮 {cnt * new_r} 次)")

        self.refresh_subset_tree()
        self.recalculate_math()
        self.save_local_config()
        messagebox.showinfo("⚖️ 配平完成", "\n".join(summary_lines))

    def run_dataset_health_audit(self):
        """后台运行 train_local.audit_dataset_directories() 并弹出体检报告窗口。"""
        if self.var_multi_subset.get():
            targets = [
                {
                    "name": s.get("name", "subset"),
                    "path": s.get("path", ""),
                    "trigger": s.get("trigger", ""),
                    "enabled": s.get("enabled", True),
                }
                for s in self.subsets
            ]
        else:
            targets = [{
                "name": self.var_lora_name.get().strip() or "main_dataset",
                "path": self.var_data_dir.get().strip(),
                "trigger": self.var_trigger.get().strip(),
                "enabled": True,
            }]

        max_tokens = resolve_max_tokens_budget(self.var_vram_preset.get(), self.detected_vram_gb)
        try:
            res_px = int(self.var_resolution.get())
        except Exception:
            res_px = 1024

        self.monitor.log(f"\n[图集体检] 正在扫描 {len(targets)} 个目标目录（分辨率 {res_px}px | MaxTokens {max_tokens}）...\n")

        def _worker():
            try:
                import train_local

                res = train_local.audit_dataset_directories(targets, max_tokens=max_tokens, resolution=res_px)
                self.root.after(0, lambda r=res: self._on_audit_finished(r))
            except Exception as err:
                self.root.after(0, lambda e=err: messagebox.showerror("图集体检失败", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_audit_finished(self, result: dict):
        level = result.get("level", "ok")
        total_imgs = result.get("total_images", 0)
        total_caps = result.get("total_captions", 0)
        report_text = result.get("report_text", "")

        color_map = {"ok": self.accent_green, "warn": self.accent_yellow, "error": self.accent_red}
        header_map = {
            "ok": f"✅ 体检通过：共 {total_imgs} 张图 / {total_caps} 份有效标注，零风险",
            "warn": f"⚠️ 发现潜在优化项（共 {total_imgs} 张图 / {total_caps} 份标注），请核对下方明细",
            "error": "❌ 体检阻断：目录无效或未发现图片，请检查路径配置",
        }
        self.monitor.log(f"[图集体检完成] 状态={level} | 有效图={total_imgs} | 标注={total_caps}\n")
        show_text_viewer_dialog(
            self.root,
            "📊 训练图集健康体检报告",
            header_map.get(level, "图集体检结果"),
            color_map.get(level, self.accent_blue),
            report_text,
            size=(760, 520),
        )

    def browse_existing_lora_file(self):
        p = filedialog.askopenfilename(
            title="选择已有 LoRA 权重 (.safetensors) 以继承/续炼",
            filetypes=[("Safetensors 模型", "*.safetensors"), ("所有文件", "*.*")],
        )
        if p:
            self.var_resume_path.set(p)
            if self.var_train_mode.get() == "fresh":
                self.var_train_mode.set("init_from")
            self.refresh_resume_status()

    def browse_resume_ckpt_dir(self):
        d = filedialog.askdirectory(title="选择包含 state.json 与 trainable.safetensors 的断点目录")
        if d:
            self.var_resume_path.set(d)
            self.var_train_mode.set("resume")
            self.refresh_resume_status()

    def refresh_resume_status(self):
        """检测断点目录或 .safetensors 权重规格，并在 resume/init_from 下自动对齐算法/Rank/LoKr。"""
        mode = (self.var_train_mode.get() or "fresh").strip()
        manual_path = self.var_resume_path.get().strip()
        lora_name = self.var_lora_name.get().strip() or "my_moss_lora"

        if mode == "fresh":
            self.var_resume_status.set("🆕 当前处于【全新从头训练】模式，将忽略历史检查点并生成全新权重。")
            self.update_quick_status_summary()
            return

        try:
            import train_local

            trainer_dir = PROJECT_ROOT / "KohakuMossTrainer"
            inspect_target = None
            status_prefix = ""

            if manual_path and Path(manual_path).is_file() and manual_path.lower().endswith(".safetensors"):
                inspect_target = Path(manual_path)
                status_prefix = f"📦 已选中权重文件: {inspect_target.name}"
            else:
                ckpts = train_local.list_lora_checkpoints(trainer_dir, lora_name, manual_path)
                if ckpts:
                    step_num, ckpt_dir = ckpts[-1]
                    inspect_target = ckpt_dir / "trainable.safetensors"
                    status_prefix = f"⏩ 已发现最新断点: step_{step_num} ({ckpt_dir.name})"
                elif manual_path:
                    status_prefix = f"⚠️ 指定路径未检测到有效断点/权重: {manual_path}"
                else:
                    status_prefix = f"⚠️ 未在 runs/moss/{lora_name}/checkpoints/ 找到断点目录，请手动选择文件或目录"

            if inspect_target and inspect_target.is_file():
                info = train_local.inspect_safetensors_lora(inspect_target)
                if info.get("rank"):
                    self.var_rank.set(str(info["rank"]))
                if info.get("alpha") is not None:
                    self.var_alpha.set(str(info["alpha"]))
                if info.get("algo"):
                    self.var_algo.set(info["algo"])
                self.var_dora.set(bool(info.get("dora", False)))
                if info.get("preset"):
                    self.var_target_preset.set(info["preset"])
                if info.get("train_norms") is not None:
                    self.var_train_norms.set(bool(info["train_norms"]))
                if info.get("lokr_factor") is not None:
                    self.var_lokr_factor.set(str(info["lokr_factor"]))
                if info.get("lokr_full_matrix") is not None:
                    self.var_lokr_full_matrix.set(bool(info["lokr_full_matrix"]))
                if info.get("lokr_decompose_both") is not None:
                    self.var_lokr_decompose_both.set(bool(info["lokr_decompose_both"]))
                self.on_dynamic_controls_changed()
                self.var_resume_status.set(
                    f"{status_prefix}\n✅ 已自动对齐规格: Algo={info.get('algo')} | "
                    f"Rank={info.get('rank')} | Alpha={info.get('alpha')} | "
                    f"DoRA={info.get('dora')} | Preset={info.get('preset') or '默认'}"
                )
            else:
                self.var_resume_status.set(status_prefix)
        except Exception as err:
            self.var_resume_status.set(f"⚠️ 断点检测异常: {err}")
        self.update_quick_status_summary()

    # ==================== 数学算力估算与步数自动对齐 ====================
    def _compute_math_summary(self):
        return compute_exposure_summary(
            multi_subset=bool(self.var_multi_subset.get()),
            subsets=self.subsets,
            data_dir=self.var_data_dir.get().strip(),
            repeats=max(1, int(self.var_repeats.get() or 1)),
            epochs=max(1, int(self.var_epochs.get() or 1)),
            vram_preset=self.var_vram_preset.get(),
            detected_vram_gb=self.detected_vram_gb,
            resolution=int(self.var_resolution.get() or 1024),
            grad_acc=max(1, int(float(self.var_grad_acc.get() or 1))),
            timestep_mu=float(self.var_timestep_mu.get() or 0.0),
        )

    def recalculate_math(self):
        try:
            is_multi, img_count, factor, ep, total_exp, rec, eff_batch, imgs_per_pack, acc, mu, _ = self._compute_math_summary()
            if mu <= -0.8:
                mu_hint = "宏观画风(高噪)"
            elif mu >= 0.3:
                mu_hint = "人脸服饰细节(低噪)"
            else:
                mu_hint = "通用均衡"

            if is_multi:
                enabled_cnt = sum(1 for s in self.subsets if s.get("enabled", True))
                self.var_math_info.set(
                    f"📊 [多子集算力公式] ({enabled_cnt}启用子集 / 共{img_count}张) 单轮{factor}曝光 × {ep}Epochs = {total_exp}曝光\n"
                    f"📦 单步等效 Batch = {imgs_per_pack}图/包 × GradAcc {acc} = {eff_batch}张/步 → 建议 Max Steps ≈ {rec} 步 ({mu_hint})"
                )
            else:
                self.var_math_info.set(
                    f"📊 [单目录算力公式] {img_count}图 × {factor}Repeats × {ep}Epochs = {total_exp}次总曝光\n"
                    f"📦 单步等效 Batch = {imgs_per_pack}图/包 × GradAcc {acc} = {eff_batch}张/步 → 建议 Max Steps ≈ {rec} 步 ({mu_hint})"
                )
        except Exception:
            self.var_math_info.set("📊 请填入有效正整数（Repeats / Epochs / GradAcc）以实时演算推荐步数。")

    def align_steps_by_formula(self, silent: bool = False):
        try:
            _, _, _, _, total_exp, rec, eff_batch, _, _, _, _ = self._compute_math_summary()
            self.var_steps.set(str(rec))
            if not silent:
                messagebox.showinfo(
                    "📐 步数对齐成功",
                    f"总曝光 {total_exp} 次 ÷ 等效 Batch {eff_batch} 图/步\n"
                    f"已自动将【总步数 (Max Steps)】设为 {rec} 步。",
                )
        except Exception as err:
            if not silent:
                messagebox.showerror("计算失败", f"无法对齐步数，请检查数字格式：\n{err}")

    # ==================== 帮助/指南查看与一键环境诊断 ====================
    def open_project_file(self, rel_path: str):
        target = PROJECT_ROOT / rel_path
        if not target.exists():
            messagebox.showwarning("文件不存在", f"未在项目目录找到：{target}")
            return
        try:
            open_in_explorer(str(target), is_file_hint=True)
        except Exception as err:
            messagebox.showerror("打开失败", str(err))

    def open_doc_viewer(self, title: str, rel_path: str):
        doc_path = PROJECT_ROOT / rel_path
        if not doc_path.exists():
            messagebox.showwarning("文档未找到", f"未在本地目录发现文档：{doc_path}")
            return
        try:
            body = doc_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as err:
            messagebox.showerror("读取失败", str(err))
            return
        show_text_viewer_dialog(
            self.root,
            f"📖 {title}",
            f"文件来源：{rel_path}",
            self.accent_blue,
            body,
            size=(800, 580),
        )

    def run_quick_env_diagnostics(self):
        """检查 Python 解释器、PyTorch / CUDA 版本、GPU 显存、核心引擎及底模状态并打印至右侧控制台。"""
        from core.config_store import resolve_cli_python

        py_exe = resolve_cli_python()
        gpu_text, vram_gb = probe_gpu_quick()
        trainer_ok = (PROJECT_ROOT / "KohakuMossTrainer" / "scripts" / "train_moss_lora.py").exists()
        torch_info = "未知"
        try:
            import torch

            cuda_ver = getattr(torch.version, "cuda", None) or "CPU-Only"
            torch_info = f"PyTorch {torch.__version__} (CUDA {cuda_ver})"
        except Exception as err:
            torch_info = f"未检测到或导入异常 ({err})"

        report = (
            f"\n==================== 🩺 本机运行环境自检报告 (v{APP_VERSION}) ====================\n"
            f"• Python 执行器: {py_exe}\n"
            f"• PyTorch 运行库: {torch_info}\n"
            f"• GPU 算力探测: {gpu_text} (可用显存基线 {vram_gb:.1f} GB)\n"
            f"• 核心引擎就绪: {'✅ KohakuMossTrainer 正常' if trainer_ok else '❌ 引擎缺失（请到 Tab 3 一键同步）'}\n"
            f"• 底模锁状态:   {self.var_model_scan_stat.get()}\n"
            f"================================================================================\n"
        )
        self.monitor.log(report)
        messagebox.showinfo("本机环境自检完成", report.strip())

    # ==================== 启动、优雅暂停、强杀与收尾工作 ====================
    def start_training(self):
        if self.is_training:
            return

        if not (PROJECT_ROOT / "KohakuMossTrainer" / "scripts" / "train_moss_lora.py").exists():
            messagebox.showerror("引擎未安装", "请先切换到【📦 底模与框架】选项卡并点击【一键安装 / 同步引擎】！")
            self.select_tab(2)
            return

        multi_mode = bool(self.var_multi_subset.get())
        if multi_mode:
            enabled_subs = [s for s in self.subsets if s.get("enabled", True)]
            if not enabled_subs:
                messagebox.showwarning("未启用子集", "多子集模式下至少需要启用 1 个包含有效图片的子集！")
                self.select_tab(1)
                return
            for sub in enabled_subs:
                if not Path(sub.get("path", "")).is_dir() or count_images_in_dir(sub.get("path", "")) == 0:
                    messagebox.showwarning("子集无图片", f"子集 [{sub.get('name', '')}] 目录无效或无图片，请前往【📂 数据与续炼】修正！")
                    self.select_tab(1)
                    return
        else:
            d_dir = self.var_data_dir.get().strip()
            if not Path(d_dir).exists() or count_images_in_dir(d_dir) == 0:
                messagebox.showwarning("图集为空", f"目录 {d_dir} 中未找到图片（支持 jpg/png/webp/bmp），请先选择包含图片与 .txt 标注的文件夹！")
                self.select_tab(1)
                return

        cleaned, err_msg = validate_training_params(self._collect_raw_params())
        if err_msg:
            messagebox.showerror("参数格式错误", err_msg)
            return

        mode_code = (self.var_train_mode.get() or "fresh").strip()
        if mode_code == "fresh":
            try:
                import train_local

                existing = train_local.list_lora_checkpoints(PROJECT_ROOT / "KohakuMossTrainer", cleaned["lora_name"], "")
                if existing:
                    latest_step, _ = existing[-1]
                    choice = messagebox.askyesnocancel(
                        "发现同名历史断点！",
                        f"在任务 [{cleaned['lora_name']}] 下发现已存在【第 {latest_step} 步】的完整断点！\n\n"
                        "• 点击【是】：改为接着历史进度继续训练 (Resume，强烈推荐)\n"
                        "• 点击【否】：按当前【全新训练】覆盖清空旧日志与断点重新开始\n"
                        "• 点击【取消】：暂不启动，返回检查设置",
                    )
                    if choice is None:
                        return
                    if choice is True:
                        mode_code = "resume"
                        self.var_train_mode.set("resume")
                        self.refresh_resume_status()
            except Exception:
                pass

        self.save_local_config()
        (PROJECT_ROOT / "workspace" / ".pause_request").unlink(missing_ok=True)

        subsets_manifest_path = None
        if multi_mode:
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", cleaned["lora_name"])
            subsets_manifest_path = (PROJECT_ROOT / "workspace" / f"{safe_name}_subsets.json").resolve()
            subsets_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            subsets_manifest_path.write_text(json.dumps(self.subsets, indent=2, ensure_ascii=False), encoding="utf-8")

        cmd = build_train_command(cleaned, mode_code, subsets_manifest_path)

        self.is_training = True
        self._pause_requested = False
        self._user_stopped = False
        self.monitor.set_running_state(True)
        self.monitor.log(
            f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🚀 触发训练指令 (模式: {mode_code})：\n"
            f"{' '.join(cmd)}\n" + "=" * 75 + "\n"
        )
        self.monitor.start_metrics_polling(reset_chart=(mode_code != "resume"))

        threading.Thread(target=self.run_subprocess_worker, args=(cmd,), daemon=True).start()

    def run_subprocess_worker(self, cmd: list):
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            self.process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=NO_WINDOW,
            )

            for line in iter(self.process.stdout.readline, ""):
                if line:
                    self.root.after(0, lambda l=line: self.monitor.log(l))

            self.process.stdout.close()
            rc = self.process.wait()
            self.root.after(0, lambda code=rc: self.training_finished(code))
        except Exception as err:
            self.root.after(0, lambda e=err: self.monitor.log(f"\n❌ 运行异常崩溃: {e}\n"))
            self.root.after(0, lambda: self.training_finished(-1))

    def pause_and_save_training(self):
        if not self.is_training:
            return
        self._pause_requested = True
        request_graceful_pause(PROJECT_ROOT / "workspace")
        self.monitor.set_pause_pending()
        self.monitor.log("\n⏳ [安全暂停指令已下发] 正在等待当前步计算完成并完整保存断点，请勿强行关窗...\n")

    def stop_training(self):
        if not self.is_training or self.process is None:
            return
        if not messagebox.askyesno("确认强制终止", "强制终止会立即杀死训练进程！建议改用【保存当前进度并暂停】。\n确定仍要强行结束吗？"):
            return
        self._user_stopped = True
        self.monitor.log("\n⏹ 正在强制终止训练进程树...\n")
        try:
            force_kill_process_tree(self.process.pid)
        except Exception as err:
            self.monitor.log(f"⚠️ 结束进程异常: {err}\n")

    def training_finished(self, returncode: int):
        self.is_training = False
        self.process = None
        self.monitor.stop_metrics_polling()
        self.monitor.set_running_state(False)
        self.monitor.load_history_metrics(silent=True)

        lora_name = self.var_lora_name.get().strip() or "my_moss_lora"
        output_dir = self.var_output_dir.get().strip() or str((PROJECT_ROOT / "output").resolve())

        if self._pause_requested:
            self._pause_requested = False
            self.monitor.log("\n✅ [已安全保存并暂停] 下次选择【断点续炼 (Resume)】即可接着继续训练！\n")
            self.refresh_resume_status()
            messagebox.showinfo("⏸️ 已安全保存进度", "当前步检查点与状态已完整保存！随时可在【📂 数据与续炼】选 Resume 接着练。")
        elif self._user_stopped:
            self._user_stopped = False
            rescued = rescue_latest_weights(lora_name, output_dir, self.var_comfy_dir.get().strip(), bool(self.var_copy_comfy.get()))
            msg = f"⏹ 训练已被用户强制中断（已抢救至第 {rescued} 步权重）。" if rescued else "⏹ 训练已被用户强制中断。"
            self.monitor.log(f"\n{msg}\n")
            self.refresh_resume_status()
        elif returncode == 0:
            self.monitor.log("\n🎉 [任务圆满完成] 成品 LoRA 与合并格式已自动同步到输出归档文件夹！\n")
            self.refresh_resume_status()
            messagebox.showinfo("🎉 训练圆满完成", f"任务 [{lora_name}] 训练与成品导出已完成！")
        else:
            self.monitor.log(f"\n❌ 训练进程异常退出 (Exit Code={returncode})，请检查控制台末尾报错。\n")
            messagebox.showerror("训练异常结束", f"训练进程退出码：{returncode}\n请查看右侧终端日志排查问题，或到【📖 指南与工具】查看排障矩阵。")


def run_gui():
    """启动桌面 GUI 控制台并记录未捕获崩溃日志到 workspace/gui_crash.log。"""
    _configure_windows_shell_ui()
    try:
        root = tk.Tk()
        ModernLocalTrainerGUI(root)
        root.mainloop()
    except Exception:
        crash_log = PROJECT_ROOT / "workspace" / "gui_crash.log"
        crash_log.parent.mkdir(parents=True, exist_ok=True)
        crash_log.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            messagebox.showerror("启动异常", f"界面初始化异常，崩溃详情已写入：\n{crash_log}\n\n{traceback.format_exc()}")
        except Exception:
            traceback.print_exc(file=sys.stderr)
        raise
