"""Main orchestrator window (ModernLocalTrainerGUI) and desktop GUI entrypoint (run_gui)."""

import ctypes
import datetime
import json
import os
import queue
import re
import secrets
import shutil
import stat
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
    MACHINE_ENV_KEYS,
    NO_WINDOW,
    PRESETS,
    PROFILE_CATEGORIES,
    PROJECT_ROOT,
    build_config_from_preset,
    build_train_command,
    cleanup_workspace_staging,
    compute_exposure_summary,
    compute_profile_diff,
    count_images_in_dir,
    create_continue_task_profile,
    delete_kaggle_kernel,
    delete_task_profile,
    derive_interactive_kernel_slug,
    diagnose_kaggle_queue_blockers,
    discover_default_comfy_models,
    duplicate_task_profile,
    export_manual_upload_bundle,
    export_task_profile_to_file,
    extract_kaggle_outputs_from_dir,
    fetch_kaggle_kernel_logs,
    fetch_kaggle_kernel_status,
    find_kaggle_cli,
    force_kill_process_tree,
    format_profile_summary,
    get_preset_by_id,
    import_kaggle_json,
    import_task_profile_from_file,
    install_kaggle_cli,
    is_trainer_ready,
    list_task_profiles,
    load_app_version,
    load_local_config,
    load_task_profile,
    normalize_hf_dataset_repo,
    open_in_explorer,
    probe_gpu_quick,
    pull_and_extract_kaggle_outputs,
    push_kaggle_kernel,
    read_kaggle_credentials,
    read_r2_credentials,
    recommend_save_every_and_ema,
    rename_task_profile,
    request_graceful_pause,
    rescue_latest_weights,
    resolve_compute_target_budget,
    resolve_effective_vram_label,
    resolve_max_tokens_budget,
    resolve_preset_for_target,
    sanitize_external_dataset_url,
    sanitize_kaggle_slug,
    sanitize_profile_config,
    save_kaggle_credentials,
    save_local_config,
    save_r2_credentials,
    save_task_profile,
    stage_dataset_bundle,
    stage_interactive_notebook_bundle,
    stage_kernel_bundle,
    test_kaggle_connection,
    toggle_pin_task_profile,
    touch_profile_loaded,
    upload_dataset_to_cloudflare_r2,
    upload_dataset_to_hf_repo,
    upload_external_dataset,
    upload_kaggle_dataset,
    validate_training_params,
)
from .monitor_panel import MonitorPanel
from .tab_cloud import build_cloud_tab
from .tab_dataset import build_dataset_tab
from .tab_help import build_help_tab
from .tab_models import build_models_tab
from .tab_profiles import build_profiles_tab
from .tab_train import build_train_tab
from .theme import PALETTE, apply_dark_theme, apply_windows_dark_titlebar
from .widgets import FlatButton, HoverTooltip, SegmentedSelector, ToggleButton, show_text_viewer_dialog

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
        self._closing = False
        self._cloud_busy = False
        self._cloud_poll_job = None
        self._kaggle_cli_info = None
        self._ui_queue = queue.SimpleQueue()
        self._ui_queue_job = None
        self._dirty_check_job = None
        self._suppress_var_traces = False
        self.preset_buttons = {}
        self._preset_base_styles = {}
        self._active_preset_id = None
        self.cloud_action_buttons = []
        self.detected_vram_gb = 8.0
        self.profiles_dir = PROJECT_ROOT / "task_profiles"
        self._active_profile_record = None
        self._active_profile_snapshot = None
        self._profile_records_cache = []
        self._profile_quick_choices = {}
        self.tab_index_map = {}
        self.tab_scroll_map = {}

        self.config_file = PROJECT_ROOT / "local_config.json"
        saved_cfg = load_local_config(self.config_file)
        self.active_profile_id = str(saved_cfg.get("active_profile_id", "") or "").strip()

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

        saved_comfy_dir = str(saved_cfg.get("comfy_models_dir", "") or "").strip()
        self.var_comfy_dir = tk.StringVar(
            value=saved_comfy_dir if (saved_comfy_dir and os.path.exists(saved_comfy_dir)) else ""
        )
        self.var_output_dir = tk.StringVar(value=saved_cfg.get("output_dir", str((PROJECT_ROOT / "output").resolve())))
        self.var_lora_name = tk.StringVar(value=saved_cfg.get("lora_name", "my_moss_lora"))
        self.var_trigger = tk.StringVar(value=saved_cfg.get("trigger_word", ""))

        self.var_transformer_path = tk.StringVar(value=saved_cfg.get("transformer_path", ""))
        self.var_text_encoder_path = tk.StringVar(value=saved_cfg.get("text_encoder_path", ""))
        self.var_vae_path = tk.StringVar(value=saved_cfg.get("vae_path", ""))
        self.var_copy_comfy = tk.BooleanVar(value=bool(saved_cfg.get("copy_to_comfy_loras", False)))

        self.var_precision = tk.StringVar(value=saved_cfg.get("precision", "bf16"))
        self.var_base_precision = tk.StringVar(value=saved_cfg.get("base_precision", "auto"))
        raw_vram_cfg = str(saved_cfg.get("vram_preset", "auto") or "auto").strip().lower()
        if raw_vram_cfg not in ("auto", "6g", "8g", "12g", "16g", "24g"):
            raw_vram_cfg = "auto"
        self.var_vram_preset = tk.StringVar(value=raw_vram_cfg)
        self._has_saved_resolution = "resolution" in saved_cfg
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
        self.var_caption_template = tk.StringVar(
            value=str(saved_cfg.get("caption_template", ADVANCED_DEFAULTS["caption_template"]))
        )
        self.var_caption_p_nl = tk.StringVar(value=str(saved_cfg.get("caption_p_nl", ADVANCED_DEFAULTS["caption_p_nl"])))
        self.var_caption_variants = tk.StringVar(
            value=str(saved_cfg.get("caption_variants", ADVANCED_DEFAULTS["caption_variants"]))
        )
        self.var_int8_compute = tk.BooleanVar(
            value=bool(saved_cfg.get("int8_compute", ADVANCED_DEFAULTS["int8_compute"]))
        )
        self.var_compile_mode = tk.StringVar(value=str(saved_cfg.get("compile_mode", ADVANCED_DEFAULTS["compile_mode"])))
        self.var_step_release = tk.StringVar(value=str(saved_cfg.get("step_release", ADVANCED_DEFAULTS["step_release"])))
        self.var_sample_every = tk.StringVar(value=str(saved_cfg.get("sample_every", ADVANCED_DEFAULTS["sample_every"])))
        self.var_sample_prompt = tk.StringVar(value=str(saved_cfg.get("sample_prompt", ADVANCED_DEFAULTS["sample_prompt"])))
        self.var_teacher_mode = tk.StringVar(value=str(saved_cfg.get("teacher_mode", ADVANCED_DEFAULTS["teacher_mode"])))
        self.var_distill_weight = tk.StringVar(
            value=str(saved_cfg.get("distill_weight", ADVANCED_DEFAULTS["distill_weight"]))
        )

        self.var_train_mode = tk.StringVar(value=saved_cfg.get("train_mode", "fresh"))
        self.var_resume_path = tk.StringVar(value=saved_cfg.get("resume_path", ""))

        # 云端遥控配置变量（注意：var_kaggle_api_key 绝不落盘至 local_config.json）
        creds_snap = read_kaggle_credentials()
        default_lora_slug = str(saved_cfg.get("lora_name", "my_moss_lora") or "my_moss_lora").strip()
        k_user_init = str(saved_cfg.get("kaggle_username") or creds_snap.get("username", "") or "").strip()
        k_acc_init = str(saved_cfg.get("kaggle_accelerator", "NvidiaTeslaT4") or "NvidiaTeslaT4").strip()
        if k_acc_init not in ("NvidiaTeslaT4", "default_gpu", "NvidiaTeslaP100"):
            k_acc_init = "NvidiaTeslaT4"
        k_var_init = str(saved_cfg.get("kaggle_variant", "heal") or "heal").strip().lower()
        if k_var_init not in ("heal", "base", "sft"):
            k_var_init = "heal"
        k_quant_init = str(saved_cfg.get("kaggle_base_quant", "auto") or "auto").strip().lower()
        if k_quant_init not in ("auto", "int8", "full"):
            k_quant_init = "auto"

        self.var_kaggle_username = tk.StringVar(value=k_user_init)
        self.var_kaggle_api_key = tk.StringVar(value="")
        self.var_kaggle_dataset_slug = tk.StringVar(
            value=str(saved_cfg.get("kaggle_dataset_slug") or sanitize_kaggle_slug(f"{default_lora_slug}-data"))
        )
        self.var_kaggle_kernel_slug = tk.StringVar(
            value=str(saved_cfg.get("kaggle_kernel_slug") or sanitize_kaggle_slug(f"{default_lora_slug}-train", prefix="moss-train"))
        )
        self.var_kaggle_accelerator = tk.StringVar(value=k_acc_init)
        self.var_kaggle_variant = tk.StringVar(value=k_var_init)
        self.var_kaggle_base_quant = tk.StringVar(value=k_quant_init)
        k_src_init = str(saved_cfg.get("kaggle_dataset_source", "local") or "local").strip().lower()
        if k_src_init not in ("local", "hf", "url", "r2"):
            k_src_init = "local"
        self.var_kaggle_dataset_source = tk.StringVar(value=k_src_init)
        self.var_external_dataset_url = tk.StringVar(
            value=sanitize_external_dataset_url(saved_cfg.get("external_dataset_url", ""))
        )
        self.var_hf_dataset_repo = tk.StringVar(value=str(saved_cfg.get("hf_dataset_repo", "") or "").strip())
        self.var_hf_dataset_subdir = tk.StringVar(value=str(saved_cfg.get("hf_dataset_subdir", "") or "").strip())
        self.var_hf_upload_local = tk.BooleanVar(value=bool(saved_cfg.get("hf_upload_local", True)))
        # HF Access Token 仅驻留内存，绝不序列化写入 local_config.json 或导出包
        self.var_hf_token = tk.StringVar(value="")

        # Cloudflare R2 对象存储变量（Secret 仅保存在 ~/.kaggle/r2_credentials.json，绝不写入 local_config.json）
        r2_snap = read_r2_credentials()
        self.var_r2_account_id = tk.StringVar(value=str(saved_cfg.get("r2_account_id") or r2_snap.get("account_id", "") or "").strip())
        self.var_r2_bucket = tk.StringVar(value=str(saved_cfg.get("r2_bucket") or r2_snap.get("bucket", "") or "").strip())
        self.var_r2_prefix = tk.StringVar(value=str(saved_cfg.get("r2_prefix", "moss-datasets") or "moss-datasets").strip())
        self.var_r2_access_key_id = tk.StringVar(value=str(saved_cfg.get("r2_access_key_id") or r2_snap.get("access_key_id", "") or "").strip())
        self.var_r2_secret_access_key = tk.StringVar(value=str(r2_snap.get("secret_access_key", "") or "").strip())
        self.var_r2_dataset_url = tk.StringVar(value="")
        self.var_kaggle_max_run_min = tk.StringVar(
            value=str(saved_cfg.get("kaggle_max_run_min", "540") or "540").strip()
        )
        tc_init = str(saved_cfg.get("target_compute", "local") or "local").strip().lower()
        if tc_init not in ("local", "kaggle_t4x2", "kaggle_t4x1"):
            tc_init = "local"
        self.var_target_compute = tk.StringVar(value=tc_init)
        self.var_manual_img_count = tk.StringVar(value=str(saved_cfg.get("manual_img_count", "") or "").strip())
        self.var_cloud_auto_poll = tk.BooleanVar(value=False)

        # 状态文本标签变量
        self.var_gpu_status = tk.StringVar(value="设备状态: ⏳ 正在探测 GPU...")
        self.var_trainer_status = tk.StringVar(value="正在检测 KohakuMossTrainer 引擎状态...")
        self.var_dataset_stat = tk.StringVar(value="当前共检测到 0 张图片")
        self.var_output_preview = tk.StringVar(value="")
        self.var_quick_status_summary = tk.StringVar(value="")
        self.var_math_info = tk.StringVar(value="数据曝光计算器就绪...")
        self.var_model_scan_stat = tk.StringVar(value="提示：可点击【🔍 自动扫描底模】从 ComfyUI 路径探测现有模型")
        self.var_resume_status = tk.StringVar(value="就绪：全新从头训练")
        self.var_kaggle_status = tk.StringVar(value="⏳ 正在后台自检 Kaggle CLI 与 API 凭证状态...")
        self.var_cloud_sync_summary = tk.StringVar(value="就绪：将同步 Tab 1/2 当前超参数与图集")
        self.var_cloud_job_status = tk.StringVar(value="⚪ 尚未查询云端作业状态")

        # 任务档案管理状态标签变量
        self.var_active_profile_badge = tk.StringVar(value="📌 当前任务：未绑定（草稿）")
        self.var_quick_profile_select = tk.StringVar(value="（未绑定任务档案）")
        self.var_profile_search = tk.StringVar(value="")
        self.var_profile_compute_filter = tk.StringVar(value="all")
        self.var_profile_cat_filter = tk.StringVar(value="全部类型")
        self.var_profile_count_stat = tk.StringVar(value="共 0 份档案")
        self.var_profile_detail_title = tk.StringVar(value="📄 档案详情透视")

        # 变量联动监听（受 _suppress_var_traces 保护，避免切换档案时中途触发局部联动覆写）
        self.var_lora_name.trace_add("write", lambda *_: self._on_lora_name_changed())
        self.var_output_dir.trace_add("write", lambda *_: None if self._suppress_var_traces else self.update_output_preview())
        self.var_resume_path.trace_add("write", lambda *_: None if self._suppress_var_traces else self.refresh_resume_status())
        self.var_steps.trace_add("write", lambda *_: self._on_steps_var_changed())

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(120, self._apply_initial_sash)

        self.check_trainer_status()
        self.update_dataset_stats()
        self.refresh_subset_tree()
        self.update_output_preview()
        self.refresh_resume_status()
        self.on_target_compute_changed(silent=True, from_cloud_tab=True)
        self.on_dynamic_controls_changed()
        self.recalculate_math()
        self.refresh_cloud_sync_summary()
        self._restore_active_profile_state()
        self.refresh_profile_list(select_id=self.active_profile_id or None)
        self._bind_dirty_check_traces()
        self._update_active_profile_bar()
        self._ui_queue_job = self.root.after(20, self._drain_ui_queue)
        self.root.after(20, self._start_background_probes)
        self.root.after(180, lambda: self.monitor.load_history_metrics(silent=True))

    def _on_lora_name_changed(self):
        if self._suppress_var_traces:
            return
        self.update_output_preview()
        self.refresh_resume_status()
        self.refresh_cloud_sync_summary()

    def _on_steps_var_changed(self):
        if self._suppress_var_traces:
            return
        self.on_dynamic_controls_changed()
        self.recalculate_math()
        self.refresh_cloud_sync_summary()

    def _enqueue_ui(self, fn):
        """将后台线程产生的界面更新任务安全投递至主线程队列，彻底避免 Python 3.13+ 跨线程 after 报错。"""
        if not self._closing:
            self._ui_queue.put(fn)

    def _drain_ui_queue(self):
        """在主线程周期性清空并执行来自后台工作线程的界面回调。"""
        if self._closing:
            return
        try:
            for _ in range(200):
                try:
                    fn = self._ui_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn()
                except Exception:
                    pass
        finally:
            if not self._closing:
                try:
                    self._ui_queue_job = self.root.after(25, self._drain_ui_queue)
                except Exception:
                    pass

    def _start_background_probes(self):
        """后台异步执行毫秒级 GPU 探测、Kaggle CLI 自检与本地底模复检，彻底消除启动白屏卡顿。"""
        def _gpu_worker():
            gpu_info, vram_gb = probe_gpu_quick()

            def _apply_gpu():
                self.detected_vram_gb = vram_gb
                self.var_gpu_status.set(f"设备状态: {gpu_info}")
                if not getattr(self, "_has_saved_resolution", False) and float(vram_gb or 8.0) <= 10.0:
                    self.var_resolution.set("896")
                self.on_dynamic_controls_changed()
                self.recalculate_math()

            self._enqueue_ui(_apply_gpu)

        threading.Thread(target=_gpu_worker, daemon=True).start()
        self.refresh_kaggle_env_status()
        self.auto_scan_models(silent=True)

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

        tk.Label(
            header_frame,
            textvariable=self.var_gpu_status,
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

        # 3. 顶部紧凑任务档案快捷切换条（全 Tab 常驻）
        profile_bar = tk.Frame(
            left_container,
            bg=self.card_bg,
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=8,
            pady=4,
        )
        profile_bar.pack(fill=tk.X, pady=(0, 4))

        lbl_prof_badge = tk.Label(
            profile_bar,
            textvariable=self.var_active_profile_badge,
            bg=self.card_bg,
            fg=self.accent_purple,
            font=(self.font_family, 8, "bold"),
            anchor="w",
        )
        lbl_prof_badge.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        HoverTooltip(
            lbl_prof_badge,
            "显示当前绑定的任务档案；若出现 *未保存，可点右侧【💾 保存】写入当前档案，或点【➕ 另存】新建一份独立 LoRA 任务配置。",
        )

        for text, bg, fg, cmd in (
            ("🗂️ 管理", PALETTE["input_bg"], self.accent_purple, lambda: self.select_tab("profiles")),
            ("➕ 另存", PALETTE["input_bg"], self.accent_green, lambda: self.open_create_profile_dialog("current")),
            ("💾 保存", self.accent_blue, "#11111b", self.save_active_profile),
        ):
            FlatButton(
                profile_bar,
                text=text,
                bg=bg,
                fg=fg,
                font=(self.font_family, 8, "bold"),
                padx=7,
                pady=2,
                command=cmd,
            ).pack(side=tk.RIGHT, padx=(4, 0))

        self.cb_quick_profile = ttk.Combobox(
            profile_bar,
            textvariable=self.var_quick_profile_select,
            values=["（未绑定任务档案）"],
            state="readonly",
            width=22,
        )
        self.cb_quick_profile.pack(side=tk.RIGHT, padx=(0, 4))
        self.cb_quick_profile.bind("<<ComboboxSelected>>", self._on_quick_profile_selected)
        HoverTooltip(self.cb_quick_profile, "快速切换已保存的任务档案（切换前自动检查未保存修改）。")

        # 4. 左侧 6 大功能选项卡 (ttk.Notebook)
        self.notebook = ttk.Notebook(left_container, style="Dark.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        tab_specs = [
            ("train", " 🚀 快速训练 ", build_train_tab),
            ("dataset", " 📂 数据与续炼 ", build_dataset_tab),
            ("models", " 📦 底模与框架 ", build_models_tab),
            ("cloud", " ☁️ 云端遥控 ", build_cloud_tab),
            ("profiles", " 🗂️ 任务档案 ", build_profiles_tab),
            ("help", " 📖 指南与工具 ", build_help_tab),
        ]
        self.tab_frames = []
        self.tab_scrolls = []
        self.tab_index_map = {}
        self.tab_scroll_map = {}
        for idx, (key, title, builder) in enumerate(tab_specs):
            frame = ttk.Frame(self.notebook, style="TabContent.TFrame")
            self.notebook.add(frame, text=title)
            scroll = builder(frame, self)
            self.tab_frames.append(frame)
            self.tab_scrolls.append(scroll)
            self.tab_index_map[key] = idx
            self.tab_scroll_map[key] = scroll

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        for tab_idx in range(len(self.tab_frames)):
            self.root.bind(f"<Control-Key-{tab_idx + 1}>", lambda _e, i=tab_idx: self.select_tab(i))

        # 5. 右侧常驻监控看板 (并保留向后兼容引用属性)
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
            if idx == self.tab_index_map.get("cloud", 3):
                self.refresh_cloud_sync_summary()
            elif idx == self.tab_index_map.get("profiles", 4):
                self.refresh_profile_list()
            if 0 <= idx < len(self.tab_scrolls):
                scroll = self.tab_scrolls[idx]
                self.root.after_idle(scroll.refresh_scroll_region)
        except Exception:
            pass

    def select_tab(self, index_or_key):
        """切换至指定索引或具名键的功能选项卡（例如 0~5 或 'profiles'）。"""
        if isinstance(index_or_key, str):
            index = self.tab_index_map.get(index_or_key, -1)
        else:
            index = int(index_or_key)
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
            "caption_template": self.var_caption_template.get(),
            "caption_p_nl": self.var_caption_p_nl.get().strip(),
            "caption_variants": self.var_caption_variants.get().strip(),
            "int8_compute": bool(self.var_int8_compute.get()),
            "compile_mode": self.var_compile_mode.get(),
            "step_release": self.var_step_release.get().strip(),
            "sample_every": self.var_sample_every.get().strip(),
            "sample_prompt": self.var_sample_prompt.get().strip(),
            "teacher_mode": self.var_teacher_mode.get(),
            "distill_weight": self.var_distill_weight.get().strip(),
            "train_mode": self.var_train_mode.get(),
            "resume_path": self.var_resume_path.get().strip(),
            "kaggle_username": self.var_kaggle_username.get().strip(),
            "kaggle_dataset_slug": self.var_kaggle_dataset_slug.get().strip(),
            "kaggle_kernel_slug": self.var_kaggle_kernel_slug.get().strip(),
            "kaggle_accelerator": self.var_kaggle_accelerator.get().strip(),
            "kaggle_variant": self.var_kaggle_variant.get().strip(),
            "kaggle_base_quant": self.var_kaggle_base_quant.get().strip(),
            "kaggle_dataset_source": self.var_kaggle_dataset_source.get().strip().lower() or "local",
            "external_dataset_url": self.var_external_dataset_url.get().strip(),
            "hf_dataset_repo": self.var_hf_dataset_repo.get().strip(),
            "hf_dataset_subdir": self.var_hf_dataset_subdir.get().strip(),
            "hf_upload_local": bool(self.var_hf_upload_local.get()),
            "r2_account_id": self.var_r2_account_id.get().strip(),
            "r2_bucket": self.var_r2_bucket.get().strip(),
            "r2_prefix": self.var_r2_prefix.get().strip(),
            "r2_access_key_id": self.var_r2_access_key_id.get().strip(),
            "kaggle_max_run_min": self.var_kaggle_max_run_min.get().strip() or "540",
            "target_compute": self.var_target_compute.get().strip().lower() or "local",
            "manual_img_count": self.var_manual_img_count.get().strip(),
        }

    def save_local_config(self):
        payload = self._collect_raw_params()
        payload["active_profile_id"] = getattr(self, "active_profile_id", "")
        save_local_config(self.config_file, payload)

    def on_closing(self):
        """关闭主窗口前保存本机配置，并在训练未停机时弹窗防误触确认强杀。"""
        self.save_local_config()
        if self._cloud_poll_job is not None:
            try:
                self.root.after_cancel(self._cloud_poll_job)
            except Exception:
                pass
            self._cloud_poll_job = None
        if getattr(self, "_dirty_check_job", None) is not None:
            try:
                self.root.after_cancel(self._dirty_check_job)
            except Exception:
                pass
            self._dirty_check_job = None
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
            self._closing = True
            if self._ui_queue_job is not None:
                try:
                    self.root.after_cancel(self._ui_queue_job)
                except Exception:
                    pass
            self.monitor.stop_metrics_polling()
            force_kill_process_tree(self.process.pid)
        else:
            self._closing = True
            if self._ui_queue_job is not None:
                try:
                    self.root.after_cancel(self._ui_queue_job)
                except Exception:
                    pass
            self.monitor.stop_metrics_polling()
        self.root.destroy()

    # ==================== 动态交互、折叠抽屉与 9 套预设应用 ====================
    def on_dynamic_controls_changed(self):
        """根据 Algo=LoKr、DoRA 开启、Timestep=uniform 及显存档位动态显示或锁定对应控件。"""
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

            from core.config_store import can_enable_teacher_distill, is_low_vram_preview_blocked

            vram_p = self.var_vram_preset.get() if hasattr(self, "var_vram_preset") else "auto"
            det_gb = float(getattr(self, "detected_vram_gb", 8.0) or 8.0)
            tc_val = self.var_target_compute.get() if hasattr(self, "var_target_compute") else "local"

            preview_blocked = is_low_vram_preview_blocked(vram_p, det_gb, tc_val)
            if preview_blocked and hasattr(self, "var_sample_every"):
                if str(self.var_sample_every.get() or "0").strip() != "0":
                    self.var_sample_every.set("0")
            if hasattr(self, "cb_sample_every") and hasattr(self, "ent_sample_prompt") and hasattr(self, "lbl_preview_guard"):
                if preview_blocked:
                    self.cb_sample_every.config(state="disabled")
                    self.ent_sample_prompt.config(state="disabled")
                    self.lbl_preview_guard.config(text="🛡️ 6G/8G默认关预览(防OOM)")
                else:
                    self.cb_sample_every.config(state="normal")
                    self.ent_sample_prompt.config(state="normal")
                    self.lbl_preview_guard.config(text="")

            teacher_ok, t_prec, _t_reason = can_enable_teacher_distill(vram_p, det_gb, tc_val)
            if not teacher_ok and hasattr(self, "var_teacher_mode"):
                if str(self.var_teacher_mode.get() or "none").strip().lower() != "none":
                    self.var_teacher_mode.set("none")
            if hasattr(self, "cb_teacher_mode") and hasattr(self, "ent_distill_weight") and hasattr(self, "lbl_teacher_guard"):
                if not teacher_ok:
                    self.cb_teacher_mode.config(state="disabled")
                    self.ent_distill_weight.config(state="disabled")
                    self.lbl_teacher_guard.config(text="⚠️ <=12G防OOM禁用")
                else:
                    self.cb_teacher_mode.config(state="readonly")
                    self.ent_distill_weight.config(state="normal")
                    self.lbl_teacher_guard.config(
                        text="💡 16G锁int8" if t_prec == "int8" else "✅ >=24G可用auto"
                    )

            train_scroll = getattr(self, "tab_scroll_map", {}).get("train")
            if train_scroll is None and hasattr(self, "tab_scrolls") and self.tab_scrolls:
                train_scroll = self.tab_scrolls[0]
            if train_scroll is not None:
                self.root.after_idle(train_scroll.refresh_scroll_region)
        except Exception:
            pass

    def toggle_advanced_drawer(self):
        """展开或折叠【高阶算法 / 防过拟合 / 优化器调优区】专业抽屉。"""
        if not hasattr(self, "adv_drawer_frame"):
            return
        if self._adv_expanded:
            self.adv_drawer_frame.pack_forget()
            self.btn_adv_toggle.config(text="▸ 展开高阶算法 / 防过拟合 / 引擎调优区 (v1.9 专业参数)")
            self._adv_expanded = False
        else:
            self.adv_drawer_frame.pack(fill=tk.X, pady=(6, 2))
            self.btn_adv_toggle.config(text="▾ 收起高阶算法 / 防过拟合 / 引擎调优区")
            self._adv_expanded = True
            self.on_dynamic_controls_changed()
        train_scroll = getattr(self, "tab_scroll_map", {}).get("train")
        if train_scroll is None and hasattr(self, "tab_scrolls") and self.tab_scrolls:
            train_scroll = self.tab_scrolls[0]
        if train_scroll is not None:
            self.root.after_idle(train_scroll.refresh_scroll_region)

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
        if hasattr(self, "var_caption_template"):
            self.var_caption_template.set(ADVANCED_DEFAULTS["caption_template"])
            self.var_caption_p_nl.set(ADVANCED_DEFAULTS["caption_p_nl"])
            self.var_caption_variants.set(ADVANCED_DEFAULTS["caption_variants"])
            self.var_int8_compute.set(ADVANCED_DEFAULTS["int8_compute"])
            self.var_compile_mode.set(ADVANCED_DEFAULTS["compile_mode"])
            self.var_step_release.set(ADVANCED_DEFAULTS["step_release"])
            self.var_sample_every.set(ADVANCED_DEFAULTS["sample_every"])
            self.var_sample_prompt.set(ADVANCED_DEFAULTS["sample_prompt"])
            self.var_teacher_mode.set(ADVANCED_DEFAULTS["teacher_mode"])
            self.var_distill_weight.set(ADVANCED_DEFAULTS["distill_weight"])

    def highlight_active_preset(self, preset_id: str):
        """高亮当前激活的黄金预设按钮，并恢复其余预设的基准暗色外观。"""
        self._active_preset_id = preset_id
        for pid, btn in self.preset_buttons.items():
            base_bg, base_fg, active_bg = self._preset_base_styles.get(pid, (self.input_bg, self.text_main, self.accent_blue))
            if pid == preset_id:
                btn.set_colors(active_bg, "#11111b", active_bg, active_bg)
            else:
                hover_bg = self.input_bg if base_bg != "#45475a" else "#52556b"
                btn.set_colors(base_bg, base_fg, hover_bg, active_bg)

    def apply_preset(self, preset_id: str):
        """应用 core.presets 中声明的 8 套场景预设，并按当前目标算力自动派生 cloud/local 参数与同步联动。"""
        preset = get_preset_by_id(preset_id)
        if not preset:
            return

        self._reset_advanced_defaults()
        self.var_multi_subset.set(bool(preset.get("multi_subset", False)))
        self._apply_subset_panel_visibility()

        target_compute = (self.var_target_compute.get() or "local").strip().lower()
        resolved_vals, cloud_notes = resolve_preset_for_target(preset_id, target_compute)

        from core.presets import DEFAULT_LEGACY_TRIGGERS, DEFAULT_PLACEHOLDER_LORA_NAMES

        cur_lora_name = (self.var_lora_name.get() or "").strip()
        cur_trigger = (self.var_trigger.get() or "").strip()

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
            "weight_decay": self.var_weight_decay,
            "timestep": self.var_timestep,
        }
        for key, val in resolved_vals.items():
            if key == "lora_name" and cur_lora_name not in DEFAULT_PLACEHOLDER_LORA_NAMES:
                continue
            if key == "trigger_word" and cur_trigger not in DEFAULT_LEGACY_TRIGGERS:
                continue
            if key in var_map:
                var_map[key].set(val)

        if preset.get("align_steps", False):
            self.align_steps_by_formula(silent=True)

        if target_compute != "local":
            cloud_notes.extend(self._apply_target_save_and_ema(target_compute))
        else:
            self.var_save_every.set("100")

        self.on_dynamic_controls_changed()
        self.recalculate_math()
        self.update_quick_status_summary()
        self.highlight_active_preset(preset_id)
        if hasattr(self, "_schedule_dirty_check"):
            self._schedule_dirty_check()
        dialog_msg = preset["dialog_message"]
        if cloud_notes:
            dialog_msg += "\n\n【☁️ 云端算力自动适配】\n" + "\n".join(f"• {n}" for n in cloud_notes)
        messagebox.showinfo(preset["dialog_title"], dialog_msg)

    # ==================== 引擎管理、底模扫描与输出预览 ====================
    def check_trainer_status(self):
        trainer_dir = PROJECT_ROOT / "KohakuMossTrainer"
        if is_trainer_ready(trainer_dir):
            self.var_trainer_status.set("✅ 已就绪：KohakuMossTrainer 引擎完整存在（可直接启动）")
        elif trainer_dir.exists():
            self.var_trainer_status.set("⚠️ 目录存在但未发现 pyproject.toml / src/kohakumoss，请点击右方【一键安装 / 同步引擎】修复")
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
        if not is_trainer_ready(trainer_dir):
            messagebox.showwarning("引擎未找到", "未发现完整的 KohakuMossTrainer（缺少 pyproject.toml 或 src/kohakumoss），请先点击【一键安装 / 同步引擎】。")
            return
        try:
            import train_local

            train_local.apply_compatibility_patches(trainer_dir)
            self.check_trainer_status()
            self.monitor.log("[系统] 已完成 KohakuMossTrainer 全套兼容补丁修复。\n")
            messagebox.showinfo("修复完成", "已注入 PyTorch 兼容补丁与调度器防护！")
        except Exception as err:
            messagebox.showerror("修补失败", str(err))

    @staticmethod
    def _force_rmtree(target: Path):
        """清除只读属性后安全删除临时或残缺目录（兼容 Windows .git pack 只读文件）。"""
        if not target.exists():
            return
        for item in target.rglob("*"):
            try:
                os.chmod(item, stat.S_IWRITE)
            except Exception:
                pass
        try:
            os.chmod(target, stat.S_IWRITE)
        except Exception:
            pass
        shutil.rmtree(target, ignore_errors=True)

    @staticmethod
    def _backup_trainer_user_data(trainer_dir: Path, backup_dir: Path):
        """在替换损坏目录前备份用户训练产物与自定义配置，防止历史 runs/ 丢失。"""
        if backup_dir.exists():
            ModernLocalTrainerGUI._force_rmtree(backup_dir)
        for rel in ("runs", os.path.join("configs", "user")):
            src = trainer_dir / rel
            if src.exists():
                dst = backup_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)

    @staticmethod
    def _restore_trainer_user_data(backup_dir: Path, trainer_dir: Path):
        """恢复之前备份的 runs/ 与 configs/user/ 目录。"""
        if not backup_dir.exists():
            return
        for rel in ("runs", os.path.join("configs", "user")):
            src = backup_dir / rel
            if src.exists():
                dst = trainer_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)

    def download_or_clone_trainer(self):
        """无损下载或同步 KohakuMossTrainer（先拉取到临时暂存区验真再切换，绝不误删现有完整引擎）。"""
        trainer_dir = PROJECT_ROOT / "KohakuMossTrainer"
        already_ready = is_trainer_ready(trainer_dir)
        if already_ready:
            proceed = messagebox.askyesno(
                "引擎已就绪",
                "检测到 KohakuMossTrainer 已完整安装！\n"
                "是否尝试在线拉取最新更新？（全程无损更新，绝不会删除 runs/ 断点与 configs/user/ 配置）",
            )
            if not proceed:
                return

        def _worker():
            self._enqueue_ui(lambda: self.var_trainer_status.set("⏳ 正在无损安装/同步 KohakuMossTrainer，请稍候..."))
            self._enqueue_ui(lambda: self.monitor.log("[引擎] 开始执行无损检查与 KohakuMossTrainer 同步...\n"))

            # 若现有目录已就绪且具备 .git，仅执行安全快进更新，不删除现有目录
            if is_trainer_ready(trainer_dir) and (trainer_dir / ".git").exists() and shutil.which("git"):
                res = subprocess.run(
                    ["git", "-C", str(trainer_dir), "pull", "--ff-only"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
                )
                if res.returncode == 0 and is_trainer_ready(trainer_dir):
                    try:
                        import train_local

                        train_local.apply_compatibility_patches(trainer_dir)
                    except Exception:
                        pass
                    self._enqueue_ui(self.check_trainer_status)
                    self._enqueue_ui(lambda: self.monitor.log("[引擎] KohakuMossTrainer 已无损拉取更新并完成兼容修补。\n"))
                    self._enqueue_ui(lambda: messagebox.showinfo("同步成功", "KohakuMossTrainer 已更新并完成兼容修补（现有数据完全保留）！"))
                    return
                self._enqueue_ui(self.check_trainer_status)
                self._enqueue_ui(lambda: self.monitor.log("⚠️ [引擎] 在线 git pull 未成功，已完整保留本地现有引擎不做任何删除。\n"))
                self._enqueue_ui(lambda: messagebox.showwarning("更新未完成", "在线拉取超时或存在本地修改，现有 KohakuMossTrainer 引擎已完整保留不受影响。"))
                return

            stage_dir = PROJECT_ROOT / "workspace" / "_trainer_stage"
            backup_dir = PROJECT_ROOT / "workspace" / "_trainer_preserve"
            self._force_rmtree(stage_dir)
            stage_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_ok = False

            if shutil.which("git"):
                git_cmds = [
                    ["git", "clone", "--depth", "1", "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git", str(stage_dir)],
                    ["git", "-c", "http.proxy=http://127.0.0.1:7890", "clone", "--depth", "1", "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git", str(stage_dir)],
                    ["git", "clone", "--depth", "1", "https://ghfast.top/https://github.com/KohakuBlueleaf/KohakuMossTrainer.git", str(stage_dir)],
                ]
                for gcmd in git_cmds:
                    self._force_rmtree(stage_dir)
                    res = subprocess.run(
                        gcmd, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
                    )
                    if res.returncode == 0 and is_trainer_ready(stage_dir):
                        staged_ok = True
                        break

            if not staged_ok:
                zip_urls = [
                    "https://github.com/KohakuBlueleaf/KohakuMossTrainer/archive/refs/heads/main.zip",
                    "https://ghfast.top/https://github.com/KohakuBlueleaf/KohakuMossTrainer/archive/refs/heads/main.zip",
                ]
                tmp_zip = PROJECT_ROOT / "workspace" / "trainer_main.zip"
                extract_tmp = PROJECT_ROOT / "workspace" / "_trainer_extract"
                for zurl in zip_urls:
                    try:
                        self._force_rmtree(extract_tmp)
                        self._force_rmtree(stage_dir)
                        req = urllib.request.Request(zurl, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=45) as resp, open(tmp_zip, "wb") as out_f:
                            out_f.write(resp.read())
                        with zipfile.ZipFile(tmp_zip, "r") as zf:
                            zf.extractall(extract_tmp)
                        sub = extract_tmp / "KohakuMossTrainer-main"
                        if sub.is_dir():
                            shutil.move(str(sub), str(stage_dir))
                        if is_trainer_ready(stage_dir):
                            staged_ok = True
                            break
                    except Exception:
                        pass
                    finally:
                        self._force_rmtree(extract_tmp)
                        tmp_zip.unlink(missing_ok=True)

            if staged_ok:
                try:
                    self._backup_trainer_user_data(trainer_dir, backup_dir)
                    self._force_rmtree(trainer_dir)
                    shutil.move(str(stage_dir), str(trainer_dir))
                    self._restore_trainer_user_data(backup_dir, trainer_dir)
                    self._force_rmtree(backup_dir)
                    import train_local

                    train_local.apply_compatibility_patches(trainer_dir)
                except Exception as err:
                    self._enqueue_ui(self.check_trainer_status)
                    self._enqueue_ui(lambda e=err: self.monitor.log(f"❌ [引擎] 切换目录或修补异常: {e}\n"))
                    self._enqueue_ui(lambda e=err: messagebox.showerror("安装异常", f"切换目录或写入补丁失败：{e}"))
                    return
                self._enqueue_ui(self.check_trainer_status)
                self._enqueue_ui(lambda: self.monitor.log("[引擎] KohakuMossTrainer 无损安装/修复与兼容补丁已就绪！\n"))
                self._enqueue_ui(lambda: messagebox.showinfo("同步成功", "KohakuMossTrainer 核心引擎已安装并完成兼容修补！"))
            else:
                self._force_rmtree(stage_dir)
                self._enqueue_ui(self.check_trainer_status)
                self._enqueue_ui(lambda: self.monitor.log("❌ [引擎] 全部通道下载均未成功，原有目录保持原样未删。\n"))
                self._enqueue_ui(lambda: messagebox.showerror("同步失败", "GitHub 与镜像通道均超时，原有文件未作删除。请检查网络后重试。"))

        threading.Thread(target=_worker, daemon=True).start()

    def auto_scan_models(self, silent: bool = True):
        """异步/极速自检本地 ComfyUI 与 models/ 底座权重；三路径均已存在时免全盘扫描，失效路径自动清空重探。"""
        path_vars = (
            ("transformer", self.var_transformer_path),
            ("text_encoder", self.var_text_encoder_path),
            ("vae", self.var_vae_path),
        )
        for _k, var in path_vars:
            val = var.get().strip()
            if val and not Path(val).is_file():
                var.set("")

        cur_comfy = self.var_comfy_dir.get().strip()
        if cur_comfy and not os.path.exists(cur_comfy):
            self.var_comfy_dir.set("")
            cur_comfy = ""

        all_files_ready = all(bool(var.get().strip() and Path(var.get().strip()).is_file()) for _k, var in path_vars)
        if silent and all_files_ready and cur_comfy:
            self.var_model_scan_stat.set("✅ 已锁定本地三核心底模（极速复检通过）")
            return

        self.var_model_scan_stat.set("⏳ 正在后台检索本地 ComfyUI / models 底模...")

        def _scan_worker():
            try:
                import train_local

                resolved_comfy = cur_comfy or discover_default_comfy_models("")
                prefer_bf16 = (self.var_base_precision.get() or "").lower() in ("compute", "fp8_compute")
                found = train_local.probe_local_models(resolved_comfy or None, prefer_bf16_te=prefer_bf16)

                def _apply_found():
                    if resolved_comfy and not self.var_comfy_dir.get().strip():
                        self.var_comfy_dir.set(resolved_comfy)
                    for key, var in path_vars:
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

                self._enqueue_ui(_apply_found)
            except Exception as err:
                self._enqueue_ui(lambda e=err: self.var_model_scan_stat.set(f"⚠️ 自动扫描异常: {e}"))

        threading.Thread(target=_scan_worker, daemon=True).start()

    def update_output_preview(self):
        name = self.var_lora_name.get().strip() or "my_lora"
        out_root = self.var_output_dir.get().strip() or str((PROJECT_ROOT / "output").resolve())
        dest_dir = Path(out_root) / name
        self.var_output_preview.set(f"最终归档输出位置：{dest_dir / (name + '-comfy.safetensors')}")

    def open_lora_output_folder(self):
        name = self.var_lora_name.get().strip() or "my_lora"
        out_root = self.var_output_dir.get().strip() or str((PROJECT_ROOT / "output").resolve())
        target = Path(out_root) / name
        has_weights = target.is_dir() and any(
            f.is_file() and f.stat().st_size > 0 for f in target.rglob("*.safetensors")
        )
        if not has_weights:
            cloud_cache = PROJECT_ROOT / "workspace" / "kaggle_outputs" / name
            if cloud_cache.is_dir() and any(
                f.is_file() and f.stat().st_size > 0 and f.suffix.lower() in (".safetensors", ".zip")
                for f in cloud_cache.rglob("*")
            ):
                try:
                    res = extract_kaggle_outputs_from_dir(
                        cloud_cache,
                        lora_name=name,
                        output_dir=out_root,
                        comfy_models_dir=self.var_comfy_dir.get().strip(),
                        copy_to_comfy=bool(self.var_copy_comfy.get()),
                        on_log=lambda txt: self.monitor.log(txt),
                    )
                    if res.get("ok"):
                        self.monitor.log(f"[输出自愈] 已将未归档云端成品自动解压整理至：{res['output_dir']}\n")
                except Exception:
                    pass
        try:
            open_in_explorer(str(target if target.exists() else Path(out_root)))
        except Exception as err:
            messagebox.showerror("打开失败", str(err))

    def clean_workspace_caches_ui(self):
        """一键安全清理 workspace/ 过期中转与 output/ 空目录（自动抢救未归档云端成品，严格保护 models/ 硬链接）。"""
        if self.is_training:
            messagebox.showwarning("训练进行中", "当前本地训练正在运行，为防止误删活跃中转缓存，请等待训练结束或先暂停后再执行清理。")
            return
        out_root = self.var_output_dir.get().strip() or str((PROJECT_ROOT / "output").resolve())
        try:
            res = cleanup_workspace_staging(PROJECT_ROOT, output_dir=out_root)
            rescued = res.get("rescued_loras", [])
            removed = res.get("removed_items", [])
            freed_mb = res.get("freed_mb", 0.0)
            self.monitor.log(
                f"\n🧹 [工作区清理完成] 共释放 {freed_mb:.2f} MB | 自动抢救归档: {len(rescued)} 项 | 清理条目: {len(removed)} 项\n"
            )
            for item in removed:
                self.monitor.log(f"  • {item}\n")
            rescue_note = f"\n自动抢救归档云端成品：{', '.join(rescued)}" if rescued else ""
            if not removed and not rescued:
                messagebox.showinfo("工作区已很整洁", "未发现需要清理的过期中转残余或空输出目录（workspace/models 硬链接与成品库均受保护）。")
            else:
                messagebox.showinfo(
                    "🧹 临时工作区清理完成",
                    f"共释放磁盘空间：约 {freed_mb:.2f} MB{rescue_note}\n"
                    f"已清理条目数：{len(removed)} 项（详情见右侧日志；workspace/models 硬链接与已有成品未受影响）。",
                )
        except Exception as err:
            messagebox.showerror("清理失败", f"清理临时工作区时发生异常：\n{err}")

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
        self.refresh_cloud_sync_summary()

    def _apply_subset_panel_visibility(self):
        if not hasattr(self, "single_data_frame") or not hasattr(self, "multi_data_frame"):
            return
        if self.var_multi_subset.get():
            self.single_data_frame.pack_forget()
            self.multi_data_frame.pack(fill=tk.X, pady=(2, 0), before=self.output_dirs_anchor)
        else:
            self.multi_data_frame.pack_forget()
            self.single_data_frame.pack(fill=tk.X, pady=(2, 0), before=self.output_dirs_anchor)
        ds_scroll = getattr(self, "tab_scroll_map", {}).get("dataset")
        if ds_scroll is None and hasattr(self, "tab_scrolls") and len(self.tab_scrolls) > 1:
            ds_scroll = self.tab_scrolls[1]
        if ds_scroll is not None:
            self.root.after_idle(ds_scroll.refresh_scroll_region)

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
        if hasattr(self, "_schedule_dirty_check") and not getattr(self, "_suppress_var_traces", False):
            self._schedule_dirty_check()

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
        dlg.geometry("540x265")
        icon_path = PROJECT_ROOT / "assets" / "moss_trainer.ico"
        if icon_path.exists():
            try:
                dlg.iconbitmap(str(icon_path))
            except Exception:
                pass
        apply_windows_dark_titlebar(dlg)

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
        FlatButton(
            body, text="浏览...", bg=self.accent_blue, fg="#11111b", font=(self.font_family, 8, "bold"), padx=10, pady=2,
            command=lambda: v_path.set(filedialog.askdirectory(title="选择子集图片文件夹") or v_path.get()),
        ).grid(row=1, column=2, pady=4)

        tk.Label(body, text="独立重复次数 (Repeats):", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=2, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_rep, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=4)

        tk.Label(body, text="专属触发词 (可选前置):", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=3, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_trig, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=3, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=4)

        ToggleButton(
            body, variable=v_en,
            text_on="已启用此子集参与训练", text_off="此子集已暂停（不参与训练）",
            on_bg=self.accent_green, off_fg=self.accent_green, font_size=9,
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
        FlatButton(btn_row, text="取消", bg=self.input_bg, fg=self.text_sub, padx=14, pady=4, command=dlg.destroy).pack(side=tk.LEFT, padx=(0, 8))
        FlatButton(btn_row, text="保存子集", bg=self.accent_green, fg="#11111b", font=(self.font_family, 9, "bold"), padx=14, pady=4, command=_save).pack(side=tk.LEFT)

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
                self._enqueue_ui(lambda r=res: self._on_audit_finished(r))
            except Exception as err:
                self._enqueue_ui(lambda e=err: messagebox.showerror("图集体检失败", str(e)))

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

    def refresh_resume_status(self, auto_align_arch: bool = True):
        """检测断点目录或 .safetensors 权重规格，并在 resume/init_from 下按需自动对齐算法/Rank/LoKr。"""
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
            elif mode == "init_from" and not manual_path:
                out_root = self.var_output_dir.get().strip() or str((PROJECT_ROOT / "output").resolve())
                cand_roots = [Path(out_root) / lora_name, PROJECT_ROOT / "workspace" / "kaggle_outputs" / lora_name]
                found_sfs = []
                for cr in cand_roots:
                    if cr.is_dir():
                        found_sfs.extend(
                            f for f in cr.rglob("*.safetensors")
                            if f.is_file()
                            and f.stat().st_size > 0
                            and "moss_image" not in f.name.lower()
                            and "qwen" not in f.name.lower()
                            and "flux2" not in f.name.lower()
                        )
                if found_sfs:
                    def _sf_rank(p: Path):
                        ln = p.name.lower()
                        prio = 2 if ln.endswith("-trainable.safetensors") else (1 if "-comfy" in ln else 0)
                        return (prio, p.stat().st_mtime)

                    found_sfs.sort(key=_sf_rank)
                    inspect_target = found_sfs[-1]
                    status_prefix = f"📦 已自动匹配成品权重: {inspect_target.name}"
                else:
                    status_prefix = f"⚠️ 未在 output/{lora_name} 找到已有 .safetensors 权重，请先训练或手动选择文件"
            else:
                ckpts = train_local.list_lora_checkpoints(trainer_dir, lora_name, manual_path)
                if ckpts:
                    step_num, ckpt_dir = ckpts[-1]
                    inspect_target = ckpt_dir / "trainable.safetensors"
                    status_prefix = f"⏩ 已发现最新断点: step_{step_num} ({ckpt_dir.name})"
                elif manual_path:
                    status_prefix = f"⚠️ 指定路径未检测到有效断点/权重: {manual_path}"
                else:
                    status_prefix = f"⚠️ 未在 runs/moss/{lora_name}/checkpoints/ 或云端回传目录找到断点，请手动选择文件或目录"

            if inspect_target and inspect_target.is_file():
                info = train_local.inspect_safetensors_lora(inspect_target)
                if auto_align_arch:
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
                align_tag = "✅ 已自动对齐规格" if auto_align_arch else "🔎 检测到权重规格"
                self.var_resume_status.set(
                    f"{status_prefix}\n{align_tag}: Algo={info.get('algo')} | "
                    f"Rank={info.get('rank')} | Alpha={info.get('alpha')} | "
                    f"DoRA={info.get('dora')} | Preset={info.get('preset') or '默认'}"
                )
            else:
                self.var_resume_status.set(status_prefix)
        except Exception as err:
            self.var_resume_status.set(f"⚠️ 断点检测异常: {err}")
        self.update_quick_status_summary()

    # ==================== 数学算力估算与步数自动对齐 ====================
    def _resolve_manual_img_override(self) -> int:
        raw = (self.var_manual_img_count.get() or "").strip()
        if not raw:
            return 0
        try:
            manual = int(float(raw))
        except Exception:
            return 0
        if manual <= 0:
            return 0
        if self.var_multi_subset.get():
            return 0
        local_cnt = count_images_in_dir(self.var_data_dir.get().strip())
        tc = (self.var_target_compute.get() or "local").strip().lower()
        if (self._is_hf_cloud_source() and tc != "local") or local_cnt == 0:
            return manual
        return 0

    def _get_active_compute_meta(self, target_override: str = None) -> tuple:
        tc = (target_override or self.var_target_compute.get() or "local").strip().lower()
        eff_key, tok_budget, active_gpus, short_badge = resolve_compute_target_budget(
            tc, self.var_vram_preset.get(), self.detected_vram_gb
        )
        return tc, eff_key, tok_budget, active_gpus, short_badge

    def _compute_math_summary(self, target_override: str = None):
        _tc, _eff_key, tok_budget, active_gpus, _badge = self._get_active_compute_meta(target_override)
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
            active_gpus=active_gpus,
            max_tokens_override=tok_budget,
            img_count_override=self._resolve_manual_img_override(),
            algo=self.var_algo.get(),
        )

    def _apply_target_save_and_ema(self, target_compute: str, preset_id: str = None) -> list:
        """按目标算力与推荐步数同步存档间隔与短步数 EMA；切回本机时恢复预设基准值。"""
        notes = []
        if target_compute != "local":
            try:
                cur_steps = int(float(self.var_steps.get() or 0))
            except Exception:
                cur_steps = 0
            save_ev, ema_adj, ema_note = recommend_save_every_and_ema(cur_steps, self.var_ema.get())
            self.var_save_every.set(save_ev)
            self.var_ema.set(ema_adj)
            if ema_note:
                notes.append(ema_note)
            if cur_steps > 0 and save_ev != "100":
                notes.append(f"推荐步数 {cur_steps} 偏短，已自动将【每 N 步存盘】细化为 {save_ev} 步。")
        else:
            self.var_save_every.set("100")
            if preset_id:
                preset = get_preset_by_id(preset_id)
                base_ema = (preset or {}).get("values", {}).get("ema", ADVANCED_DEFAULTS["ema"])
                self.var_ema.set(str(base_ema))
        return notes

    def on_target_compute_changed(self, silent: bool = False, from_cloud_tab: bool = False):
        """在 Tab 1 切换目标算力（本机 / Kaggle双卡T4 / Kaggle单卡T4），同步 Tab 4、提示条及激活预设派生值。"""
        tc = (self.var_target_compute.get() or "local").strip().lower()
        if tc not in ("local", "kaggle_t4x2", "kaggle_t4x1"):
            tc = "local"
            self.var_target_compute.set(tc)

        if not from_cloud_tab:
            if tc == "kaggle_t4x2":
                self.var_kaggle_accelerator.set("NvidiaTeslaT4")
            elif tc == "kaggle_t4x1":
                self.var_kaggle_accelerator.set("default_gpu")

        banner = getattr(self, "cloud_hw_banner", None)
        hp_grid = getattr(self, "hp_grid", None)
        if banner is not None and hp_grid is not None:
            try:
                banner.pack_forget()
                if tc != "local":
                    banner.pack(fill=tk.X, pady=(0, 6), before=hp_grid)
            except Exception:
                pass

        active_pid = getattr(self, "_active_preset_id", None)
        if not silent and not from_cloud_tab and active_pid:
            vals, _notes = resolve_preset_for_target(active_pid, tc)
            if vals:
                if "grad_acc" in vals:
                    self.var_grad_acc.set(str(vals["grad_acc"]))
                if "lr" in vals:
                    self.var_lr.set(str(vals["lr"]))
                _, cur_img_cnt, *_ = self._compute_math_summary()
                if cur_img_cnt > 0:
                    self.align_steps_by_formula(silent=True)
                self._apply_target_save_and_ema(tc, preset_id=active_pid)
                self.monitor.log(
                    f"[目标算力切换] {tc} -> 已重新校准激活预设 [{active_pid}]："
                    f"GradAcc={self.var_grad_acc.get()}, LR={self.var_lr.get()}, "
                    f"Steps={self.var_steps.get()}, SaveEvery={self.var_save_every.get()}, EMA={self.var_ema.get()}\n"
                )

        self.on_dynamic_controls_changed()
        self.recalculate_math()
        self.refresh_cloud_sync_summary()
        train_scroll = getattr(self, "tab_scroll_map", {}).get("train")
        if train_scroll is None and hasattr(self, "tab_scrolls") and self.tab_scrolls:
            train_scroll = self.tab_scrolls[0]
        if train_scroll is not None:
            try:
                self.root.after_idle(train_scroll.refresh_scroll_region)
            except Exception:
                pass

    def on_cloud_accelerator_changed(self):
        """Tab 4 切换云端算力队列时，反向同步 Tab 1 目标算力与数学估算。"""
        acc = (self.var_kaggle_accelerator.get() or "NvidiaTeslaT4").strip()
        mapped = "kaggle_t4x2" if acc == "NvidiaTeslaT4" else "kaggle_t4x1"
        if self.var_target_compute.get() != mapped:
            self.var_target_compute.set(mapped)
        self.on_target_compute_changed(silent=True, from_cloud_tab=True)

    def recalculate_math(self):
        try:
            tc, _eff_key, tok_budget, active_gpus, short_badge = self._get_active_compute_meta()
            is_multi, img_count, factor, ep, total_exp, rec, eff_batch, imgs_per_pack, acc, mu, _ = self._compute_math_summary()
            tok_desc = f"{tok_budget}×{active_gpus}卡" if active_gpus > 1 else f"{tok_budget}"
            batch_expr = (
                f"{imgs_per_pack}图/卡 × {active_gpus}卡 × GradAcc {acc} = {eff_batch}张/步"
                if active_gpus > 1
                else f"{imgs_per_pack}图/包 × GradAcc {acc} = {eff_batch}张/步"
            )
            if mu <= -0.8:
                mu_hint = "宏观画风(高噪)"
            elif mu >= 0.3:
                mu_hint = "人脸服饰细节(低噪)"
            else:
                mu_hint = "通用均衡"

            manual_used = self._resolve_manual_img_override() > 0
            cnt_note = f"(预估{img_count}图)" if manual_used else f"{img_count}图"
            import math as _math
            raw_formula = _math.ceil(total_exp / max(1, eff_batch)) if total_exp > 0 else 1
            floor_tag = f" [含防欠拟合保底，原始公式 {raw_formula} 步]" if (img_count > 0 and rec > raw_formula) else ""
            if is_multi:
                enabled_cnt = sum(1 for s in self.subsets if s.get("enabled", True))
                base_msg = (
                    f"📊 [{short_badge} | MAX_TOKENS={tok_desc}] ({enabled_cnt}子集/{img_count}图) 单轮{factor}曝光 × {ep}Epochs = {total_exp}\n"
                    f"📦 单步等效 Batch = {batch_expr} → 建议 Max Steps ≈ {rec} 步{floor_tag} ({mu_hint})"
                )
            else:
                base_msg = (
                    f"📊 [{short_badge} | MAX_TOKENS={tok_desc}] {cnt_note} × {factor}Repeats × {ep}Epochs = {total_exp}次总曝光\n"
                    f"📦 单步等效 Batch = {batch_expr} → 建议 Max Steps ≈ {rec} 步{floor_tag} ({mu_hint})"
                )

            warn_color = self.accent_green
            try:
                cur_steps = int(float(self.var_steps.get().strip()))
            except Exception:
                cur_steps = 0

            if img_count > 0 and cur_steps > 0 and cur_steps >= max(rec * 1.8, rec + 80):
                ratio = cur_steps / max(1, rec)
                base_msg += f"\n⚠️ 当前 Max Steps ({cur_steps}) 约为该算力推荐值 ({rec}) 的 {ratio:.1f} 倍，建议点【📐 一键对齐推荐步数】防过拟合！"
                warn_color = self.accent_yellow
            elif tc == "kaggle_t4x2" and img_count > 0 and 0 < rec < 120:
                base_msg += f"\n⚠️ 双卡大 Batch 下推荐步数仅 {rec} 步(<120)，易更新不足，建议略提高 Epochs 使步数达到 150+ 步。"
                warn_color = self.accent_yellow

            self.var_math_info.set(base_msg)
            if hasattr(self, "lbl_math_info"):
                self.lbl_math_info.config(fg=warn_color)
        except Exception:
            self.var_math_info.set("📊 请填入有效正整数（Repeats / Epochs / GradAcc）以实时演算推荐步数。")
            if hasattr(self, "lbl_math_info"):
                try:
                    self.lbl_math_info.config(fg=self.accent_green)
                except Exception:
                    pass

    def preview_cli_command(self):
        """校验当前全部面板参数并弹窗预览对应的完整 train_local.py 启动命令（支持一键复制）。"""
        cleaned, err_msg = validate_training_params(self._collect_raw_params())
        if err_msg:
            messagebox.showerror("参数格式错误", err_msg)
            return None

        mode_code = (self.var_train_mode.get() or "fresh").strip()
        subsets_manifest_path = None
        if self.var_multi_subset.get():
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", cleaned["lora_name"])
            subsets_manifest_path = (PROJECT_ROOT / "workspace" / f"{safe_name}_subsets.json").resolve()

        cmd = build_train_command(cleaned, mode_code, subsets_manifest_path)
        formatted = (
            "# 单行直接复制运行命令：\n"
            f"{subprocess.list2cmdline(cmd)}\n\n"
            "# 逐参数拆解清单：\n"
            f'"{cmd[0]}" {cmd[1]} {cmd[2]}\n'
            + "\n".join(f"  {tok}" for tok in cmd[3:])
        )
        return show_text_viewer_dialog(
            self.root,
            "📋 等效 CLI 训练命令预览",
            f"任务 [{cleaned['lora_name']}] | 模式: {mode_code} | 共 {len(cmd)} 项参数片段",
            self.accent_green,
            formatted,
            size=(780, 520),
            copy_button=True,
        )

    def align_steps_by_formula(self, silent: bool = False):
        try:
            import math as _math
            _, img_count, _, _, total_exp, rec, eff_batch, _, _, _, _ = self._compute_math_summary()
            self.var_steps.set(str(rec))
            if not silent:
                raw_formula = _math.ceil(total_exp / max(1, eff_batch)) if total_exp > 0 else 1
                floor_note = (
                    f"\n（注：原始公式仅 {raw_formula} 步，已触发防欠拟合安全下限保底至 {rec} 步）"
                    if (img_count > 0 and rec > raw_formula)
                    else ""
                )
                messagebox.showinfo(
                    "📐 步数对齐成功",
                    f"总曝光 {total_exp} 次 ÷ 等效 Batch {eff_batch} 图/步\n"
                    f"已自动将【总步数 (Max Steps)】设为 {rec} 步。{floor_note}",
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
        trainer_ok = is_trainer_ready(PROJECT_ROOT / "KohakuMossTrainer")
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

    # ==================== Kaggle 云端 API 一键遥控方法区 ====================
    def _cloud_source(self) -> str:
        src = self.var_kaggle_dataset_source.get().strip().lower()
        return src if src in ("local", "hf", "url", "r2") else "local"

    def _is_hf_cloud_source(self) -> bool:
        return self._cloud_source() == "hf"

    def _is_url_cloud_source(self) -> bool:
        return self._cloud_source() == "url"

    def _is_r2_cloud_source(self) -> bool:
        return self._cloud_source() == "r2"

    def on_cloud_hf_upload_toggle(self):
        self._sync_cloud_source_buttons()
        self.refresh_cloud_sync_summary()

    def clear_external_dataset_url(self):
        self.var_external_dataset_url.set("")
        self.save_local_config()
        self.refresh_cloud_sync_summary()

    def clear_r2_dataset_url(self):
        self.var_r2_dataset_url.set("")
        self.refresh_cloud_sync_summary()

    def _collect_r2_config(self) -> dict:
        saved = read_r2_credentials()
        return {
            "account_id": self.var_r2_account_id.get().strip() or saved.get("account_id", ""),
            "bucket": self.var_r2_bucket.get().strip() or saved.get("bucket", ""),
            "prefix": self.var_r2_prefix.get().strip() or "moss-datasets",
            "access_key_id": self.var_r2_access_key_id.get().strip() or saved.get("access_key_id", ""),
            "secret_access_key": self.var_r2_secret_access_key.get().strip() or saved.get("secret_access_key", ""),
            "public_base_url": saved.get("public_base_url", ""),
        }

    def save_r2_credentials_ui(self):
        r2_cfg = self._collect_r2_config()
        ok, msg = save_r2_credentials(
            r2_cfg["account_id"],
            r2_cfg["access_key_id"],
            r2_cfg["secret_access_key"],
            bucket=r2_cfg["bucket"],
            public_base_url=r2_cfg.get("public_base_url", ""),
        )
        if ok:
            self.save_local_config()
            self.monitor.log(f"[Cloudflare R2] {msg}\n")
            messagebox.showinfo("R2 密钥保存成功", msg)
        else:
            messagebox.showwarning("R2 配置不完整", msg)

    def _validate_r2_config_ui(self) -> dict:
        cfg = self._collect_r2_config()
        if not cfg["account_id"] or not cfg["bucket"] or not cfg["access_key_id"] or not cfg["secret_access_key"]:
            messagebox.showwarning(
                "缺少 Cloudflare R2 凭据",
                "请完整填写 Cloudflare R2 的 Account ID、Bucket 桶名、Access Key ID 与 Secret Access Key！\n"
                "（点击右侧【💾 保存 R2 密钥】可将密钥安全存入 ~/.kaggle/r2_credentials.json 以便下次免填）",
            )
            return {}
        return cfg

    def _has_local_dataset_images(self) -> bool:
        if bool(self.var_multi_subset.get()):
            enabled_subs = [s for s in self.subsets if s.get("enabled", True)]
            return bool(enabled_subs) and all(
                Path(s.get("path", "")).is_dir() and count_images_in_dir(s.get("path", "")) > 0
                for s in enabled_subs
            )
        d_dir = self.var_data_dir.get().strip()
        return bool(d_dir) and Path(d_dir).exists() and count_images_in_dir(d_dir) > 0

    def _parse_max_run_minutes(self) -> int:
        raw = str(self.var_kaggle_max_run_min.get() or "540").strip()
        token = raw.split()[0] if raw else "540"
        try:
            return max(0, int(float(token)))
        except Exception:
            return 540

    def _validate_hf_repo_ui(self, require_token: bool = False):
        """校验并规范化 Hugging Face 数据集仓库输入，返回 (repo_id, subdir, token) 或 None。"""
        repo = normalize_hf_dataset_repo(self.var_hf_dataset_repo.get())
        if not repo or "/" not in repo:
            messagebox.showwarning(
                "缺少 Hugging Face 数据集仓库",
                "请填写有效的 Hugging Face Dataset 仓库 ID 或链接：\n"
                "例如 `owner/my-dataset` 或 `https://huggingface.co/datasets/owner/my-dataset`。",
            )
            return None
        self.var_hf_dataset_repo.set(repo)
        subdir = self.var_hf_dataset_subdir.get().strip().strip("/")
        token = self.var_hf_token.get().strip()
        if require_token and not token:
            messagebox.showwarning(
                "缺少 HF Write Token",
                "自动创建并上传至 Hugging Face 私有仓库需要填写具备 Write 权限的 HF Access Token！\n"
                "（若图集已提前传好，请取消勾选下方【开炼前自动上传】Toggle 开关。）",
            )
            return None
        return repo, subdir, token

    def _cloud_opts(self) -> dict:
        src = self._cloud_source()
        opts = {
            "accelerator": self.var_kaggle_accelerator.get().strip() or "NvidiaTeslaT4",
            "variant": self.var_kaggle_variant.get().strip() or "heal",
            "base_quant": self.var_kaggle_base_quant.get().strip() or "auto",
            "max_run_minutes": self._parse_max_run_minutes(),
            "dataset_source": src,
        }
        if src == "hf":
            opts["hf_dataset_repo"] = normalize_hf_dataset_repo(self.var_hf_dataset_repo.get())
            opts["hf_dataset_subdir"] = self.var_hf_dataset_subdir.get().strip().strip("/")
            opts["hf_dataset_revision"] = "main"
            tok = self.var_hf_token.get().strip()
            if tok:
                opts["hf_token"] = tok
        elif src == "url":
            cleaned_url = sanitize_external_dataset_url(self.var_external_dataset_url.get().strip())
            if cleaned_url != self.var_external_dataset_url.get().strip():
                self.var_external_dataset_url.set(cleaned_url)
            opts["dataset_url"] = cleaned_url
        elif src == "r2":
            opts["dataset_url"] = self.var_r2_dataset_url.get().strip()
        return opts

    def _sync_cloud_source_buttons(self):
        """非忙碌状态下同步主按钮文案及【上传图集】按钮启用状态。"""
        src = self._cloud_source()
        hf_auto_up = bool(self.var_hf_upload_local.get())
        if hasattr(self, "btn_cloud_master"):
            try:
                if src == "url":
                    master_txt = "🚀 一键全自动：图集上传 5 站外链 + 推送 Kaggle 直拉开炼 (免审核)"
                elif src == "hf":
                    master_txt = (
                        "🚀 一键全自动：上传 HF 私有仓库 + 推送 Kaggle 云端直拉开炼"
                        if hf_auto_up
                        else "🚀 云端直拉 HF 图集 + 一键推送开炼（免传本地图）"
                    )
                elif src == "r2":
                    master_txt = "🚀 一键全自动：上传 Cloudflare R2 私有桶 + 推送 Kaggle 直拉开炼"
                else:
                    master_txt = "🚀 一键全自动：打包上传图集 + 同步当前参数推送云端开炼"
                self.btn_cloud_master.config(text=master_txt)
            except Exception:
                pass
        if not self._cloud_busy and hasattr(self, "btn_cloud_upload_ds"):
            try:
                if src == "url":
                    ds_btn_txt, ds_state = "📤 1. 仅打包图集上传外链", tk.NORMAL
                elif src == "hf":
                    ds_btn_txt = "📤 1. 上传图集至 HF 私有仓"
                    ds_state = tk.NORMAL if hf_auto_up else tk.DISABLED
                elif src == "r2":
                    ds_btn_txt, ds_state = "📤 1. 上传图集至 R2 私有桶", tk.NORMAL
                else:
                    ds_btn_txt, ds_state = "1️⃣ 打包并上传图集", tk.NORMAL
                self.btn_cloud_upload_ds.config(text=ds_btn_txt, state=ds_state)
            except Exception:
                pass

    def on_cloud_dataset_source_changed(self):
        """在【临时外链(5站)】、【HF私有仓库】、【Cloudflare R2私有存储】与【Kaggle私有Dataset】间切换面板。"""
        src = self._cloud_source()
        for attr in ("cloud_url_frame", "cloud_local_slug_frame", "cloud_hf_frame", "cloud_r2_frame"):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.pack_forget()
                except Exception:
                    pass

        anchor = getattr(self, "cloud_kernel_slug_anchor", None)
        if anchor is not None:
            try:
                if src == "url" and hasattr(self, "cloud_url_frame"):
                    self.cloud_url_frame.pack(fill=tk.X, pady=(0, 8), before=anchor)
                elif src == "hf" and hasattr(self, "cloud_hf_frame"):
                    self.cloud_hf_frame.pack(fill=tk.X, pady=(0, 8), before=anchor)
                elif src == "r2" and hasattr(self, "cloud_r2_frame"):
                    self.cloud_r2_frame.pack(fill=tk.X, pady=(0, 8), before=anchor)
                elif hasattr(self, "cloud_local_slug_frame"):
                    self.cloud_local_slug_frame.pack(fill=tk.X, pady=(0, 4), before=anchor)
            except Exception:
                pass

        self._sync_cloud_source_buttons()
        self.refresh_cloud_sync_summary()
        cloud_scroll = getattr(self, "tab_scroll_map", {}).get("cloud")
        if cloud_scroll is None and hasattr(self, "tab_scrolls") and len(self.tab_scrolls) > 3:
            cloud_scroll = self.tab_scrolls[3]
        if cloud_scroll is not None:
            try:
                self.root.after_idle(cloud_scroll.refresh_scroll_region)
            except Exception:
                pass

    def _set_cloud_busy(self, busy: bool):
        self._cloud_busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in getattr(self, "cloud_action_buttons", []):
            try:
                btn.config(state=state)
            except Exception:
                pass
        if not busy:
            self._sync_cloud_source_buttons()

    def _run_cloud_worker(self, start_status: str, fn):
        """在后台守护线程安全执行云端耗时操作并防重入，所有 UI 更新均经由 _enqueue_ui 回流主线程。"""
        if self._cloud_busy:
            messagebox.showinfo("云端操作进行中", "当前已有云端任务正在上传、推送或拉取中，请等待右侧日志提示完成。")
            return
        self._set_cloud_busy(True)
        if start_status:
            self.var_cloud_job_status.set(start_status)

        def _worker():
            try:
                fn()
            except Exception as err:
                self._enqueue_ui(lambda e=err: self.monitor.log(f"❌ [云端操作异常] {e}\n"))
                self._enqueue_ui(lambda e=err: self.var_cloud_job_status.set(f"🔴 云端操作失败: {e}"))
                self._enqueue_ui(lambda e=err: messagebox.showerror("云端操作异常", str(e)))
            finally:
                self._enqueue_ui(lambda: self._set_cloud_busy(False))

        threading.Thread(target=_worker, daemon=True).start()

    def refresh_kaggle_env_status(self):
        """异步自检本机 Kaggle CLI 与 ~/.kaggle/kaggle.json 凭证状态（零阻塞启动）。"""
        def _probe():
            cli = find_kaggle_cli()
            creds = read_kaggle_credentials()

            def _apply():
                self._kaggle_cli_info = cli
                if creds.get("username") and not self.var_kaggle_username.get().strip():
                    self.var_kaggle_username.set(creds["username"])
                cli_txt = f"✅ {cli['version']} ({cli['source']})" if cli.get("found") else "⚠️ 未安装 Kaggle CLI"
                if creds.get("username") and creds.get("has_key"):
                    cred_txt = f"✅ 已接入账号 [{creds['username']}] (Key: {creds['key_masked']})"
                else:
                    cred_txt = "⚠️ 尚未配置 kaggle.json（点下方【📥 导入 kaggle.json】或填入 Token 保存）"
                self.var_kaggle_status.set(f"CLI 状态：{cli_txt}  |  凭据：{cred_txt}")

            self._enqueue_ui(_apply)

        threading.Thread(target=_probe, daemon=True).start()

    def refresh_cloud_sync_summary(self):
        """实时在 Tab 4 展示即将推送至 Kaggle 的 Tab 1/2 参数与算力摘要。"""
        try:
            lora_name = self.var_lora_name.get().strip() or "my_moss_lora"
            src = self._cloud_source()
            if self.var_multi_subset.get():
                enabled_subs = [s for s in self.subsets if s.get("enabled", True)]
                imgs = sum(count_images_in_dir(s.get("path", "")) for s in enabled_subs)
                base_ds = f"{len(enabled_subs)}子集/{imgs}图"
            else:
                imgs = count_images_in_dir(self.var_data_dir.get().strip())
                base_ds = f"单图集/{imgs}图"

            if src == "hf":
                repo_norm = normalize_hf_dataset_repo(self.var_hf_dataset_repo.get()) or "(待填 HF 仓库)"
                subdir = self.var_hf_dataset_subdir.get().strip().strip("/")
                sub_note = f"/{subdir}" if subdir else ""
                if self.var_hf_upload_local.get():
                    ds_txt = f"🤗 HF 私有上传→直拉 ({base_ds} → {repo_norm}{sub_note})"
                else:
                    ds_txt = f"🤗 HF 直拉 ({repo_norm}{sub_note})"
            elif src == "r2":
                bkt = self.var_r2_bucket.get().strip() or "(待填 Bucket)"
                r2_ready = "已签发7天链" if self.var_r2_dataset_url.get().strip() else "待上传R2"
                ds_txt = f"☁️ R2 私有 ({base_ds} → {bkt} · {r2_ready})"
            elif src == "url":
                cur_url = self.var_external_dataset_url.get().strip()
                url_state = "已就绪外链" if cur_url else "自动传5站"
                ds_txt = f"🔗 外链直拉 ({base_ds} · {url_state})"
            else:
                ds_txt = f"📦 Kaggle私有 ({base_ds})"
            acc_val = self.var_kaggle_accelerator.get().strip()
            acc_map = {"NvidiaTeslaT4": "双卡 T4 DDP", "default_gpu": "单卡 GPU", "NvidiaTeslaP100": "兼容队列"}
            acc_txt = acc_map.get(acc_val, "双卡 T4 DDP")
            sim_tc = "kaggle_t4x2" if acc_val == "NvidiaTeslaT4" else "kaggle_t4x1"
            _, _, _, _, _, _, cloud_batch, _, _, _, _ = self._compute_math_summary(target_override=sim_tc)
            dora_txt = "+DoRA" if self.var_dora.get() else ""
            guard_min = self._parse_max_run_minutes()
            guard_txt = f"{guard_min}m存盘" if guard_min > 0 else "无时限"
            self.var_cloud_sync_summary.set(
                f"🔄 将同步推送：[{lora_name}] | {ds_txt} | {self.var_steps.get().strip()}步 | "
                f"{self.var_algo.get().upper()}{dora_txt} R{self.var_rank.get().strip()} LR={self.var_lr.get().strip()} @ {self.var_resolution.get()}px "
                f"| 云端 {acc_txt} ({cloud_batch}张/步) / {self.var_kaggle_variant.get().upper()} / {guard_txt}"
            )
        except Exception:
            pass

    def _resolve_kaggle_identity(self, require_creds: bool = True) -> tuple:
        """校验并返回规范化后的 (username, dataset_slug, kernel_slug, dataset_id, kernel_id)。"""
        creds = read_kaggle_credentials()
        username = (self.var_kaggle_username.get().strip() or creds.get("username", "")).strip()
        if not username:
            messagebox.showwarning("缺少 Kaggle 用户名", "请先在第 1 步导入 kaggle.json 或填写 Kaggle 用户名！")
            return None
        self.var_kaggle_username.set(username)
        if require_creds and not (creds.get("exists") and creds.get("has_key")):
            messagebox.showwarning("尚未保存 API 凭据", "请先点击【📥 导入 kaggle.json】或填入 API Key 后点击【💾 保存凭证】！")
            return None

        lora_name = self.var_lora_name.get().strip() or "my-moss-lora"
        if self._cloud_source() in ("url", "r2", "hf"):
            raw_ds = f"{lora_name}-data"
        else:
            raw_ds = self.var_kaggle_dataset_slug.get().strip() or f"{lora_name}-data"
        raw_kn = self.var_kaggle_kernel_slug.get().strip() or f"{lora_name}-train"
        if "/" in raw_ds:
            raw_ds = raw_ds.split("/", 1)[1]
        if "/" in raw_kn:
            raw_kn = raw_kn.split("/", 1)[1]
        ds_slug = sanitize_kaggle_slug(raw_ds, prefix="moss-data")
        kn_slug = sanitize_kaggle_slug(raw_kn, prefix="moss-train")
        self.var_kaggle_dataset_slug.set(ds_slug)
        self.var_kaggle_kernel_slug.set(kn_slug)
        return username, ds_slug, kn_slug, f"{username}/{ds_slug}", f"{username}/{kn_slug}"

    def auto_generate_cloud_slugs(self):
        lora_name = self.var_lora_name.get().strip() or "my_moss_lora"
        ds_slug = sanitize_kaggle_slug(f"{lora_name}-data", prefix="moss-data")
        base_kn = sanitize_kaggle_slug(f"{lora_name}-train", prefix="moss-train")
        cur_kn = self.var_kaggle_kernel_slug.get().strip()
        if cur_kn.startswith(base_kn):
            kn_slug = sanitize_kaggle_slug(f"{lora_name}-train-{secrets.token_hex(2)[:3]}", prefix="moss-train")
        else:
            kn_slug = base_kn
        self.var_kaggle_dataset_slug.set(ds_slug)
        self.var_kaggle_kernel_slug.set(kn_slug)
        self.save_local_config()
        self.refresh_cloud_sync_summary()
        self.monitor.log(
            f"[云端 Slug] 已生成合法 Kaggle 标识：图集={self.var_kaggle_dataset_slug.get()} | 任务={self.var_kaggle_kernel_slug.get()}\n"
        )

    def import_kaggle_json_dialog(self):
        src = filedialog.askopenfilename(
            title="选择从 Kaggle 下载的 kaggle.json 凭证文件",
            filetypes=[("Kaggle Token JSON", "*.json"), ("所有文件", "*.*")],
        )
        if not src:
            return
        ok, username, msg = import_kaggle_json(src)
        if ok:
            self.var_kaggle_username.set(username)
            self.var_kaggle_api_key.set("")
            self.save_local_config()
            self.refresh_kaggle_env_status()
            self.monitor.log(f"[Kaggle 凭证] {msg}\n")
            messagebox.showinfo("导入成功", msg)
        else:
            messagebox.showerror("导入失败", msg)

    def save_kaggle_credentials_ui(self):
        user = self.var_kaggle_username.get().strip()
        key = self.var_kaggle_api_key.get().strip()
        ok, msg = save_kaggle_credentials(user, key)
        if ok:
            self.var_kaggle_api_key.set("")
            self.save_local_config()
            self.refresh_kaggle_env_status()
            self.monitor.log(f"[Kaggle 凭证] {msg}\n")
            messagebox.showinfo("保存成功", msg)
        else:
            messagebox.showwarning("凭证信息不完整", msg)

    def test_kaggle_connection_ui(self):
        def _worker():
            self._enqueue_ui(lambda: self.monitor.log("[Kaggle 诊断] 正在测试 Kaggle API 连通性与账户权限...\n"))
            ok, msg = test_kaggle_connection(cli_info=self._kaggle_cli_info)
            self._enqueue_ui(lambda m=msg: self.monitor.log(f"[Kaggle 诊断] {m}\n"))
            if ok:
                self._enqueue_ui(lambda m=msg: self.var_kaggle_status.set(m))
                self._enqueue_ui(lambda m=msg: messagebox.showinfo("连通性检测通过", m))
            else:
                self._enqueue_ui(lambda m=msg: messagebox.showwarning("连通性检测未通过", m))

        self._run_cloud_worker("⏳ 正在检测 Kaggle API 连通性...", _worker)

    def install_kaggle_cli_ui(self):
        def _worker():
            ok, msg = install_kaggle_cli(on_log=lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt)))
            self._enqueue_ui(self.refresh_kaggle_env_status)
            self._enqueue_ui(lambda m=msg: self.monitor.log(f"[Kaggle CLI] {m}\n"))
            if ok:
                self._enqueue_ui(lambda m=msg: messagebox.showinfo("安装成功", m))
            else:
                self._enqueue_ui(lambda m=msg: messagebox.showerror("安装失败", m))

        self._run_cloud_worker("⏳ 正在安装官方 Kaggle CLI...", _worker)

    def _upload_external_step(self, raw_params: dict, ds_slug: str, log_cb) -> tuple:
        """打包本地图集并按序上传至 5 站候选外链（成功 1 站即止，失败自动顺延），成功后回填直链并持久化配置。"""
        log_cb("\n[云端流水线 · 外链中转] 正在打包本地图集并按序上传候选外链站（成功 1 站即止）...\n")
        ok, url_or_err, stats = upload_external_dataset(raw_params, dataset_slug=ds_slug, on_log=log_cb)
        if ok:
            def _save_url():
                self.var_external_dataset_url.set(url_or_err)
                self.save_local_config()
                self.refresh_cloud_sync_summary()

            self._enqueue_ui(_save_url)
        return ok, url_or_err, stats

    def _upload_r2_step(self, raw_params: dict, r2_cfg: dict, ds_slug: str, log_cb) -> tuple:
        """打包本地图集并上传至私有 Cloudflare R2 桶，成功后在 UI 线程回填 7 天预签名直链。"""
        log_cb("\n[云端流水线 · Cloudflare R2] 正在打包本地图集并上传至 R2 私有存储桶...\n")
        ok, url_or_err, stats = upload_dataset_to_cloudflare_r2(raw_params, r2_cfg, dataset_slug=ds_slug, on_log=log_cb)
        if ok:
            def _save_r2_url():
                self.var_r2_dataset_url.set(url_or_err)
                self.save_local_config()
                self.refresh_cloud_sync_summary()

            self._enqueue_ui(_save_r2_url)
        return ok, url_or_err, stats

    def _upload_hf_step(self, raw_params: dict, repo_id: str, hf_token: str, subdir: str, log_cb) -> tuple:
        """创建或更新 Hugging Face Private Dataset 永久私有仓库。"""
        log_cb(f"\n[云端流水线 · HF 私有仓库] 正在创建并上传至私有数据集 {repo_id}...\n")
        ok, repo_or_err, stats = upload_dataset_to_hf_repo(
            raw_params,
            repo_id=repo_id,
            hf_token=hf_token,
            subdir=subdir,
            private=True,
            on_log=log_cb,
        )
        if ok:
            self._enqueue_ui(lambda r=repo_or_err: self.var_hf_dataset_repo.set(r))
            self._enqueue_ui(self.save_local_config)
            self._enqueue_ui(self.refresh_cloud_sync_summary)
        return ok, repo_or_err, stats

    def cloud_upload_dataset_only(self):
        src = self._cloud_source()
        if src == "hf":
            if not self.var_hf_upload_local.get():
                messagebox.showinfo(
                    "当前为 HF 免上传直拉模式",
                    "您取消勾选了【自动上传本地图集至 HF 私有仓】。\n"
                    "若需将当前本地图集上传至 Hugging Face Private 仓库，请先勾选该开关后重试，或直接点【🚀 云端直拉 HF 图集】。",
                )
                return
            if not self._check_dataset_inputs_ready():
                return
            hf_check = self._validate_hf_repo_ui(require_token=True)
            if not hf_check:
                return
            repo_id, subdir, hf_tok = hf_check
            raw_params = self._collect_raw_params()
            self.save_local_config()

            def _hf_worker():
                log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
                ok, repo_or_err, stats = self._upload_hf_step(raw_params, repo_id, hf_tok, subdir, log_cb)
                if ok:
                    self._enqueue_ui(
                        lambda r=repo_or_err: self.var_cloud_job_status.set(
                            f"✅ 已同步至 HF 私有仓库 {r}（下一步可点【2️⃣ 仅推送参数开炼】）"
                        )
                    )
                    self._enqueue_ui(
                        lambda r=repo_or_err, st=stats: messagebox.showinfo(
                            "🤗 HF 私有仓库同步成功",
                            f"已将 {st['images']} 张图 / {st['captions']} 份标注（{st['total_mb']} MB）"
                            f"同步至 Hugging Face Private 数据集仓库：\n\nhttps://huggingface.co/datasets/{r}",
                        )
                    )
                else:
                    self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 上传至 HF 私有仓库失败，请查看控制台报错"))
                    self._enqueue_ui(lambda m=repo_or_err: messagebox.showerror("HF 私有仓上传失败", m))

            self._run_cloud_worker(f"⏳ 正在上传本地图集至 HF 私有仓 {repo_id}...", _hf_worker)
            return

        if not self._check_dataset_inputs_ready():
            return

        if src == "r2":
            r2_cfg = self._validate_r2_config_ui()
            if not r2_cfg:
                return
            raw_params = self._collect_raw_params()
            ds_slug = sanitize_kaggle_slug(
                self.var_kaggle_dataset_slug.get().strip() or raw_params.get("lora_name", "") or "moss-data",
                prefix="moss-data",
            )
            self.save_local_config()

            def _r2_worker():
                log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
                ok, url_or_err, stats = self._upload_r2_step(raw_params, r2_cfg, ds_slug, log_cb)
                if ok:
                    self._enqueue_ui(
                        lambda: self.var_cloud_job_status.set(
                            "✅ 已上传至 Cloudflare R2 私有桶并生成 7 天预签名直链（下一步可点【2️⃣ 仅推送参数开炼】）"
                        )
                    )
                    self._enqueue_ui(
                        lambda st=stats: messagebox.showinfo(
                            "☁️ Cloudflare R2 私有上传成功",
                            f"已将 {st['images']} 张图 / {st['captions']} 份标注（{st['zip_mb']} MB）"
                            f"上传至私有 R2 对象：\nr2://{st.get('r2_bucket')}/{st.get('r2_object_key')}\n\n"
                            f"已自动回填 7 天有效预签名直链至面板！",
                        )
                    )
                else:
                    self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 上传至 Cloudflare R2 失败，请查看控制台报错"))
                    self._enqueue_ui(lambda m=url_or_err: messagebox.showerror("Cloudflare R2 上传失败", m))

            self._run_cloud_worker("⏳ 正在打包并上传至 Cloudflare R2 私有存储...", _r2_worker)
            return

        if src == "url":
            raw_params = self._collect_raw_params()
            ds_slug = sanitize_kaggle_slug(
                self.var_kaggle_dataset_slug.get().strip() or raw_params.get("lora_name", "") or "moss-data",
                prefix="moss-data",
            )
            self.save_local_config()

            def _url_worker():
                log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
                ok, url_or_err, stats = self._upload_external_step(raw_params, ds_slug, log_cb)
                if ok:
                    mirrors_ok = stats.get("mirrors", [])
                    host_desc = (
                        f"站 [{mirrors_ok[0].get('host', '外链')}]"
                        if len(mirrors_ok) == 1
                        else f"至 {len(mirrors_ok)} 条外链中转站"
                    )
                    self._enqueue_ui(
                        lambda: self.var_cloud_job_status.set(
                            "✅ 外链图集已上传并回填（下一步可点【2️⃣ 仅推送参数开炼】）"
                        )
                    )
                    self._enqueue_ui(
                        lambda u=url_or_err, st=stats, hd=host_desc: messagebox.showinfo(
                            "🔗 外链图集上传成功",
                            f"已打包 {st['images']} 张图 / {st['captions']} 份标注（{st['zip_mb']} MB），\n"
                            f"并已成功上传{hd}：\n\n{u}",
                        )
                    )
                else:
                    self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 外链图集上传失败，请查看控制台报错"))
                    self._enqueue_ui(lambda m=url_or_err: messagebox.showerror("外链上传失败", m))

            self._run_cloud_worker("⏳ 正在打包并上传图集至外链中转站（成功 1 站即止）...", _url_worker)
            return

        ident = self._resolve_kaggle_identity(require_creds=True)
        if not ident:
            return
        username, ds_slug, _kn_slug, ds_id, _kn_id = ident
        raw_params = self._collect_raw_params()
        self.save_local_config()

        def _worker():
            log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
            log_cb(f"\n[云端流水线 1/2] 正在打包本地图集为 Kaggle 私有数据集 {ds_id}...\n")
            stage_dir, stats = stage_dataset_bundle(raw_params, username, ds_slug)
            log_cb(
                f"[图集打包完成] 模式={stats['mode']} | {stats['images']} 张图 / {stats['captions']} 份标注 "
                f"| 约 {stats['total_mb']} MB -> {stage_dir}\n"
            )
            ok, msg = upload_kaggle_dataset(
                stage_dir,
                ds_id,
                version_notes=f"MOSS-Image GUI sync ({stats['images']} images)",
                cli_info=self._kaggle_cli_info,
                on_log=log_cb,
            )
            if ok:
                self._enqueue_ui(lambda m=msg: self.var_cloud_job_status.set(f"✅ {m}（下一步可点【2️⃣ 仅推送参数开炼】）"))
                self._enqueue_ui(lambda m=msg: messagebox.showinfo("图集上传成功", f"{m}\n\n共打包 {stats['images']} 张图 / {stats['captions']} 份标注。"))
            else:
                self._enqueue_ui(lambda m=msg: self.var_cloud_job_status.set("🔴 图集上传失败，请查看控制台报错"))
                self._enqueue_ui(lambda m=msg: messagebox.showerror("图集上传失败", m))

        self._run_cloud_worker(f"⏳ 正在打包并上传图集 {ds_id}...", _worker)

    def _confirm_cloud_batch_sanity(self, cleaned: dict, cloud_opts: dict) -> bool:
        """推送/导出至 Kaggle 双卡 T4 DDP 前拦截潜在过曝配置，提供一键双卡重算或强推选择。"""
        if (cloud_opts.get("accelerator") or "NvidiaTeslaT4").strip() != "NvidiaTeslaT4":
            return True

        res_int = max(256, int(cleaned.get("resolution", 1024)))
        est_tok = (res_int // 16) ** 2 + 160
        imgs_per_pack = max(1, 9216 // est_tok)
        cur_ga = max(1, int(cleaned.get("grad_acc", 1)))
        cur_batch = imgs_per_pack * 2 * cur_ga
        ga1_batch = imgs_per_pack * 2
        cur_steps = max(1, int(cleaned.get("steps", 1)))

        import math as _math
        from core.presets import recommend_min_steps

        _, img_count, _, _, total_exp, _, _, _, _, _, _ = self._compute_math_summary(target_override="kaggle_t4x2")
        floor_steps = recommend_min_steps(algo=str(cleaned.get("algo", "lora")), multi_subset=bool(self.var_multi_subset.get()))
        rec_ga1 = max(floor_steps, _math.ceil(total_exp / ga1_batch)) if img_count > 0 else 0
        baseline_exp = max(total_exp, rec_ga1 * ga1_batch)

        high_ga = cur_ga >= 2 and cur_batch >= 8
        over_exposed = img_count > 0 and (cur_steps * cur_batch >= baseline_exp * 1.8)
        if not (high_ga or over_exposed):
            return True

        plan_steps_desc = f"{rec_ga1} 步" if rec_ga1 > 0 else f"保持 {cur_steps} 步但 GradAcc 归 1"
        exposure_desc = (
            f"• 预期曝光 {total_exp} 次（{img_count} 图）→ 按当前参数将跑出 {cur_steps * cur_batch} 次（过曝隐患）！\n"
            if img_count > 0
            else ""
        )
        ans = messagebox.askyesnocancel(
            "⚠️ 云端双卡 T4 曝光倍率提醒",
            f"当前准备推送至【Kaggle 双卡 T4 DDP】，容器实测并发吞吐较高：\n"
            f"• 双卡 DDP 下 MAX_TOKENS=9216×2卡（{res_int}px 单步并发 = {imgs_per_pack}×2×GradAcc {cur_ga} = {cur_batch} 张/步）\n"
            f"{exposure_desc}"
            f"\n是否按双卡标准（GradAcc=1 -> {ga1_batch} 张/步，{plan_steps_desc}）自动重算并同步左侧界面后推送？\n\n"
            f"• [是 (Y)]：自动按双卡 T4 重算后继续推送/导出\n"
            f"• [否 (N)]：保持当前 {cur_steps} 步 / GradAcc={cur_ga} 原参数强推\n"
            f"• [取消]：中止本次操作返回修改",
            icon="warning",
        )
        if ans is None:
            return False
        if ans is True:
            if cur_ga > 1:
                cleaned["grad_acc"] = 1
                self.var_grad_acc.set("1")
            if rec_ga1 > 0:
                cleaned["steps"] = rec_ga1
                self.var_steps.set(str(rec_ga1))
                save_ev, ema_adj, _ = recommend_save_every_and_ema(rec_ga1, self.var_ema.get())
                cleaned["save_every"] = int(save_ev)
                self.var_save_every.set(save_ev)
                cleaned["ema"] = float(ema_adj)
                self.var_ema.set(ema_adj)
            self.var_target_compute.set("kaggle_t4x2")
            self.on_target_compute_changed(silent=True, from_cloud_tab=True)
            self.save_local_config()
            self.monitor.log(
                f"[双卡防呆校准] 已按 T4×2 DDP 重算：GradAcc={cleaned['grad_acc']}, Steps={cleaned['steps']}, SaveEvery={cleaned['save_every']}\n"
            )
        return True

    def _push_kernel_and_report(
        self,
        cleaned: dict,
        cloud_opts: dict,
        username: str,
        ds_id: str,
        kn_slug: str,
        src_note: str,
        queued_prefix: str,
        success_title: str,
        success_summary: str,
        fail_status: str,
        log_cb,
    ):
        """统一执行阶段化内核组装、推送至 Kaggle 队列及界面状态回执弹窗。"""
        stage_dir, info = stage_kernel_bundle(cleaned, cloud_opts, username, ds_id, kn_slug)
        log_cb(
            f"\n[云端流水线 2/2] 已注入 Tab 1/2 全量超参数，准备推送 {info['kernel_id']} "
            f"(算力: {cloud_opts['accelerator']} | {src_note} | 自动存盘时限: {cloud_opts['max_run_minutes']} 分钟)...\n"
        )
        ok, url, msg = push_kaggle_kernel(
            stage_dir,
            accelerator=cloud_opts["accelerator"],
            cli_info=self._kaggle_cli_info,
            on_log=log_cb,
        )
        if ok:
            self._enqueue_ui(lambda m=msg, p=queued_prefix: self.var_cloud_job_status.set(f"{p}{m}"))
            self._enqueue_ui(
                lambda u=url, m=msg, t=success_title, body=success_summary: messagebox.showinfo(
                    t,
                    f"{body}\n\n{m}\n\n可在第 4 步开启【自动轮询 (30s)】或点【📜 拉取云端日志】追踪进度：\n{u}",
                )
            )
        else:
            self._enqueue_ui(lambda fs=fail_status: self.var_cloud_job_status.set(fs))
            self._enqueue_ui(lambda m=msg: messagebox.showerror("推送训练任务失败", m))

    def cloud_push_kernel_only(self):
        src = self._cloud_source()
        if src == "hf" and not self._validate_hf_repo_ui(require_token=False):
            return
        r2_cfg = {}
        if src == "url" and not self.var_external_dataset_url.get().strip():
            if not self._check_dataset_inputs_ready():
                return
        elif src == "r2" and not self.var_r2_dataset_url.get().strip():
            if not self._check_dataset_inputs_ready():
                return
            r2_cfg = self._validate_r2_config_ui()
            if not r2_cfg:
                return

        ident = self._resolve_kaggle_identity(require_creds=True)
        if not ident:
            return
        username, ds_slug, kn_slug, ds_id, _kn_id = ident
        raw_params = self._collect_raw_params()
        cleaned, err_msg = validate_training_params(raw_params)
        if err_msg:
            messagebox.showerror("参数格式错误", err_msg)
            return
        cloud_opts = self._cloud_opts()
        if not self._confirm_cloud_batch_sanity(cleaned, cloud_opts):
            return
        self.save_local_config()

        def _worker():
            log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
            if src == "url" and not cloud_opts.get("dataset_url"):
                ok_ext, url_or_err, _ext_stats = self._upload_external_step(raw_params, ds_slug, log_cb)
                if not ok_ext:
                    self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 自动上传外链图集失败，已中止推送"))
                    self._enqueue_ui(lambda m=url_or_err: messagebox.showerror("外链上传失败", m))
                    return
                cloud_opts["dataset_url"] = url_or_err
            elif src == "r2" and not cloud_opts.get("dataset_url"):
                ok_r2, url_or_err, _r2_stats = self._upload_r2_step(raw_params, r2_cfg, ds_slug, log_cb)
                if not ok_r2:
                    self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 自动上传 R2 图集失败，已中止推送"))
                    self._enqueue_ui(lambda m=url_or_err: messagebox.showerror("R2 上传失败", m))
                    return
                cloud_opts["dataset_url"] = url_or_err

            if src == "hf":
                src_note = f"🤗 云端直拉 HF: {cloud_opts.get('hf_dataset_repo')}"
            elif src == "r2":
                src_note = "☁️ Cloudflare R2 私有预签名直拉 (dataset_sources=[])"
            elif src == "url":
                src_note = f"🔗 5站外链直拉 ({len(cloud_opts.get('dataset_url', '').split('|'))}条线路, dataset_sources=[])"
            else:
                src_note = f"📦 挂载 Kaggle 图集: {ds_id}"

            self._push_kernel_and_report(
                cleaned, cloud_opts, username, ds_id, kn_slug,
                src_note=src_note,
                queued_prefix="🟡 已提交至 Kaggle 云端队列：",
                success_title="云端开炼已启动",
                success_summary=f"数据来源：{src_note}",
                fail_status="🔴 推送任务失败，请检查权限或 CLI 输出",
                log_cb=log_cb,
            )

        self._run_cloud_worker("⏳ 正在推送训练内核至 Kaggle...", _worker)

    def _prepare_cloud_deploy_request(self):
        """校验数据源、Kaggle 身份与训练超参数，并确认云端等效 Batch 后返回部署上下文。"""
        src = self._cloud_source()
        hf_auto_up = bool(self.var_hf_upload_local.get())
        has_local = self._has_local_dataset_images()
        r2_cfg = {}
        hf_check = None

        if src == "hf":
            if hf_auto_up and not self._check_dataset_inputs_ready():
                return None
            hf_check = self._validate_hf_repo_ui(require_token=hf_auto_up)
            if not hf_check:
                return None
        elif src == "r2":
            has_r2_url = bool(self.var_r2_dataset_url.get().strip())
            if not has_local and not has_r2_url:
                self._check_dataset_inputs_ready()
                return None
            if has_local:
                r2_cfg = self._validate_r2_config_ui()
                if not r2_cfg:
                    return None
        elif src == "url":
            if not has_local and not bool(self.var_external_dataset_url.get().strip()):
                self._check_dataset_inputs_ready()
                return None
        elif not self._check_dataset_inputs_ready():
            return None

        ident = self._resolve_kaggle_identity(require_creds=True)
        if not ident:
            return None
        username, ds_slug, kn_slug, ds_id, kn_id = ident
        raw_params = self._collect_raw_params()
        cleaned, err_msg = validate_training_params(raw_params)
        if err_msg:
            messagebox.showerror("参数格式错误", err_msg)
            return None
        cloud_opts = self._cloud_opts()
        if not self._confirm_cloud_batch_sanity(cleaned, cloud_opts):
            return None
        self.save_local_config()
        return {
            "src": src,
            "hf_auto_up": hf_auto_up,
            "has_local": has_local,
            "r2_cfg": r2_cfg,
            "hf_check": hf_check,
            "username": username,
            "ds_slug": ds_slug,
            "kn_slug": kn_slug,
            "ds_id": ds_id,
            "kn_id": kn_id,
            "raw_params": raw_params,
            "cleaned": cleaned,
            "cloud_opts": cloud_opts,
        }

    def _sync_cloud_dataset_for_deploy(self, ctx: dict, log_cb) -> tuple:
        """按当前选择的数据源完成打包上传或直拉地址注入，返回 `(ok, summary_meta)`。"""
        src = ctx["src"]
        raw_params = ctx["raw_params"]
        cloud_opts = ctx["cloud_opts"]
        username = ctx["username"]
        ds_slug = ctx["ds_slug"]
        ds_id = ctx["ds_id"]
        kn_id = ctx["kn_id"]

        if src == "hf":
            hf_repo_target, hf_sub_target, hf_tok_target = ctx["hf_check"]
            sub_hint = f" (子目录: {hf_sub_target})" if hf_sub_target else ""
            if ctx["hf_auto_up"]:
                log_cb(f"\n{'='*70}\n🤗 [一键 HF 私有仓库流水线] 1/2 创建并同步图集至私有仓 `{hf_repo_target}`{sub_hint}\n")
                ok_hf, repo_or_err, hf_stats = self._upload_hf_step(
                    raw_params, hf_repo_target, hf_tok_target, hf_sub_target, log_cb
                )
                if not ok_hf:
                    self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 上传 HF 私有仓失败，已中止推送"))
                    self._enqueue_ui(lambda m=repo_or_err: messagebox.showerror("HF 私有仓上传失败", m))
                    return False, {}
                cloud_opts["hf_dataset_repo"] = repo_or_err
                summary_txt = f"已将 {hf_stats['images']} 张图同步至 Hugging Face Private 仓库 `{repo_or_err}`{sub_hint}，\n并跳过 Kaggle Dataset 审核推入云端！"
            else:
                summary_txt = f"已跳过本地上传，并配置容器开机直拉 Hugging Face `{hf_repo_target}`{sub_hint}！"
            return True, {
                "src_note": f"🤗 云端直拉 HF: {cloud_opts.get('hf_dataset_repo')}{sub_hint}",
                "queued_prefix": "🟡 云端 HF 直拉任务已排队：",
                "success_title": "🚀 云端 HF 私有直拉部署成功！",
                "success_summary": summary_txt,
                "fail_status": "🔴 推送 HF 直拉训练任务失败",
            }

        if src == "r2":
            r2_stats = None
            if ctx["has_local"]:
                log_cb(f"\n{'='*70}\n☁️ [一键 R2 私有存储流水线] 1/2 打包图集并上传至 Cloudflare R2 私有桶\n")
                ok_r2, url_or_err, r2_stats = self._upload_r2_step(raw_params, ctx["r2_cfg"], ds_slug, log_cb)
                if not ok_r2:
                    self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 上传至 Cloudflare R2 失败，已中止推送"))
                    self._enqueue_ui(lambda m=url_or_err: messagebox.showerror("R2 上传失败", m))
                    return False, {}
                cloud_opts["dataset_url"] = url_or_err
            summary_txt = (
                f"已将 {r2_stats['images']} 张图上传至私有 Cloudflare R2（r2://{r2_stats.get('r2_bucket')}/{r2_stats.get('r2_object_key')}），\n并签发 7 天免密预签名直链推送 Kaggle 开炼（dataset_sources=[]）！"
                if r2_stats
                else "已使用当前 R2 预签名直链推送无数据集挂载训练任务至 Kaggle！"
            )
            return True, {
                "src_note": "☁️ Cloudflare R2 私有预签名直拉 (dataset_sources=[])",
                "queued_prefix": "🟡 R2 私有直拉训练已排队：",
                "success_title": "🚀 Cloudflare R2 私有直拉部署成功！",
                "success_summary": summary_txt,
                "fail_status": "🔴 R2 已上传，但推送训练任务失败",
            }

        if src == "url":
            ext_stats = None
            if ctx["has_local"]:
                log_cb(f"\n{'='*70}\n🔗 [一键外链直拉流水线] 1/2 打包图集并按序上传候选外链（成功 1 站即止，免 Kaggle 审核）\n")
                ok_ext, url_or_err, ext_stats = self._upload_external_step(raw_params, ds_slug, log_cb)
                if not ok_ext:
                    self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 第一步上传外链失败，已中止推送"))
                    self._enqueue_ui(lambda m=url_or_err: messagebox.showerror("外链上传失败", m))
                    return False, {}
                cloud_opts["dataset_url"] = url_or_err
            url_parts = [u.strip() for u in cloud_opts.get("dataset_url", "").split("|") if u.strip()]
            mirror_cnt = len(url_parts)
            mirrors_list = (ext_stats or {}).get("mirrors", [])
            host_tag = (
                mirrors_list[0].get("host", "外链直拉")
                if len(mirrors_list) == 1
                else f"{mirror_cnt}条镜像"
            )
            if ext_stats:
                summary_txt = (
                    f"已将 {ext_stats['images']} 张图上传至 [{host_tag}]，并推送无数据集挂载（dataset_sources=[]）训练任务！"
                    if mirror_cnt == 1
                    else f"已将 {ext_stats['images']} 张图上传至 {mirror_cnt} 条外链镜像，并推送无数据集挂载（dataset_sources=[]）训练任务！"
                )
            else:
                summary_txt = (
                    "已注入外链直拉地址并推送任务！"
                    if mirror_cnt == 1
                    else f"已注入外链直拉地址（{mirror_cnt} 条镜像）并推送任务！"
                )
            note_desc = host_tag if mirror_cnt == 1 else f"{mirror_cnt}条镜像"
            return True, {
                "src_note": f"🔗 外链直拉 ({note_desc}, dataset_sources=[])",
                "queued_prefix": "🟡 外链直拉训练已排队：",
                "success_title": "🚀 外链直拉云端部署成功！",
                "success_summary": summary_txt,
                "fail_status": "🔴 外链已就绪，但推送训练任务失败",
            }

        log_cb(f"\n{'='*70}\n🚀 [一键云端流水线] 1/2 打包并上传图集 -> {ds_id}\n")
        ds_stage, stats = stage_dataset_bundle(raw_params, username, ds_slug)
        ok_ds, msg_ds = upload_kaggle_dataset(
            ds_stage,
            ds_id,
            version_notes=f"Auto-sync {stats['images']} imgs from GUI",
            cli_info=self._kaggle_cli_info,
            on_log=log_cb,
        )
        if not ok_ds:
            self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 第一步上传图集失败，已中止推送"))
            self._enqueue_ui(lambda m=msg_ds: messagebox.showerror("上传图集失败", m))
            return False, {}
        return True, {
            "src_note": f"📦 挂载 Kaggle 图集: {ds_id}",
            "queued_prefix": "🟡 云端训练已排队：",
            "success_title": "🚀 云端全自动部署成功！",
            "success_summary": f"已上传 {stats['images']} 张图至私有数据集 {ds_id} 并推送任务 {kn_id}！",
            "fail_status": "🔴 图集已上传，但推送训练任务失败",
        }

    def cloud_one_click_deploy(self):
        ctx = self._prepare_cloud_deploy_request()
        if not ctx:
            return

        def _worker():
            log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
            ok_sync, meta = self._sync_cloud_dataset_for_deploy(ctx, log_cb)
            if not ok_sync:
                return
            self._push_kernel_and_report(
                ctx["cleaned"],
                ctx["cloud_opts"],
                ctx["username"],
                ctx["ds_id"],
                ctx["kn_slug"],
                src_note=meta["src_note"],
                queued_prefix=meta["queued_prefix"],
                success_title=meta["success_title"],
                success_summary=meta["success_summary"],
                fail_status=meta["fail_status"],
                log_cb=log_cb,
            )

        title_map = {
            "hf": f"⏳ 正在执行 HF 私有仓流水线 ({ctx['cloud_opts'].get('hf_dataset_repo', '')})...",
            "r2": "⏳ 正在执行 Cloudflare R2 私有上传 + 推云端...",
            "url": "⏳ 正在执行外链中转（成功 1 站即止）+ 推云端直拉开炼...",
            "local": "⏳ 全自动流水线执行中：传图集 + 推云端...",
        }
        self._run_cloud_worker(title_map.get(ctx["src"], title_map["local"]), _worker)

    def cloud_push_interactive_notebook(self):
        """打包并推送为 `--no-run` 交互式 `.ipynb` Notebook，绕过 Kaggle 后台批处理排队池并打开编辑页直接 Run All。"""
        ctx = self._prepare_cloud_deploy_request()
        if not ctx:
            return

        nb_slug = derive_interactive_kernel_slug(ctx["kn_slug"])
        nb_id = f"{ctx['username']}/{nb_slug}"

        def _worker():
            log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
            ok_sync, meta = self._sync_cloud_dataset_for_deploy(ctx, log_cb)
            if not ok_sync:
                return

            diag = diagnose_kaggle_queue_blockers(nb_id, cli_info=self._kaggle_cli_info)
            if diag.get("hint"):
                log_cb(f"[免排队前置巡检] ⚠️ {diag['hint']}\n")

            stage_dir, info = stage_interactive_notebook_bundle(
                ctx["cleaned"],
                ctx["cloud_opts"],
                ctx["username"],
                ctx["ds_id"],
                nb_slug,
            )
            log_cb(
                f"\n[云端流水线 2/2 · 免排队交互模式] 已注入全量超参数，准备 Quick-Save 为交互式 Notebook：{info['kernel_id']} "
                f"(算力: {ctx['cloud_opts']['accelerator']} | {meta['src_note']} | --no-run 绕过后台排队)...\n"
            )
            ok, url, msg = push_kaggle_kernel(
                stage_dir,
                accelerator=ctx["cloud_opts"]["accelerator"],
                cli_info=self._kaggle_cli_info,
                on_log=log_cb,
                no_run=True,
            )
            if ok:
                accel_lbl = ctx["cloud_opts"]["accelerator"]
                blocker_line = f"\n\n⚠️ 检测到同账号有其它占槽任务：{diag['hint']}" if diag.get("hint") else ""
                self._enqueue_ui(lambda k=nb_id: self.var_cloud_job_status.set(f"🟢 免排队交互 Notebook 已保存：{k}（请在网页点击 Run All）"))
                self._enqueue_ui(lambda u=url: webbrowser.open(u))
                self._enqueue_ui(
                    lambda u=url, m=msg, s=meta["success_summary"], a=accel_lbl, b=blocker_line: messagebox.showinfo(
                        "⚡ 免排队交互式 Notebook 已就绪！",
                        f"{s}\n\n{m}\n\n已绕过拥堵的后台批处理排队（--no-run Quick Save），并已在浏览器为你打开页面：\n{u}"
                        f"{b}\n\n"
                        f"⚠️ 重要提示：若网页显示 Version 1 of 1、Runtime: 9s、代码未执行 (In [ ]:) 且右上角有黑色【Edit】按钮，"
                        f"这表示快速保存已成功（尚未开始训练，绝非出错！）。\n\n"
                        f"请按以下 4 步立即开炼：\n"
                        f"1. 先点击网页右上角黑色 【Edit】 按钮进入编辑器；\n"
                        f"2. 看左下角 View Active Events，确认没有其它占满 GPU 的旧会话或正在跑的任务；\n"
                        f"3. 确认右侧 Settings -> Accelerator = {a} 且 Internet = ON；\n"
                        f"4. 点击顶部 【Run All】 按钮即可直接在交互式 GPU 池开机训练！",
                    )
                )
            else:
                self._enqueue_ui(lambda fs=meta["fail_status"]: self.var_cloud_job_status.set(fs))
                self._enqueue_ui(lambda m=msg: messagebox.showerror("推送免排队交互 Notebook 失败", m))

        self._run_cloud_worker("⏳ 正在同步数据并免排队保存为交互式 Notebook...", _worker)

    def cloud_refresh_status(self, silent: bool = False):
        ident = self._resolve_kaggle_identity(require_creds=True) if not silent else None
        if silent:
            creds = read_kaggle_credentials()
            user = (self.var_kaggle_username.get().strip() or creds.get("username", "")).strip()
            kn_slug = sanitize_kaggle_slug(
                self.var_kaggle_kernel_slug.get().strip() or self.var_lora_name.get().strip() or "moss-train",
                prefix="moss-train",
            )
            if not user:
                return
            kn_id = f"{user}/{kn_slug}"
        else:
            if not ident:
                return
            _u, _ds, _kn, _ds_id, kn_id = ident

        def _worker():
            st = fetch_kaggle_kernel_status(kn_id, cli_info=self._kaggle_cli_info)
            self._enqueue_ui(lambda s=st: self.var_cloud_job_status.set(s["label"]))
            if not silent:
                self._enqueue_ui(lambda s=st: self.monitor.log(f"[Kaggle 状态] {kn_id} -> {s['label']}\n"))

        threading.Thread(target=_worker, daemon=True).start()

    def toggle_cloud_auto_poll(self):
        if self._cloud_poll_job is not None:
            try:
                self.root.after_cancel(self._cloud_poll_job)
            except Exception:
                pass
            self._cloud_poll_job = None

        if not self.var_cloud_auto_poll.get():
            self.monitor.log("[Kaggle 监控] 已停止 30 秒自动轮询。\n")
            return

        self.monitor.log("[Kaggle 监控] 已开启每 30 秒自动查询云端作业状态。\n")
        self.cloud_refresh_status(silent=True)

        def _tick():
            if self._closing or not self.var_cloud_auto_poll.get():
                self._cloud_poll_job = None
                return
            self.cloud_refresh_status(silent=True)
            self._cloud_poll_job = self.root.after(30000, _tick)

        self._cloud_poll_job = self.root.after(30000, _tick)

    def open_kaggle_kernel_page(self):
        creds = read_kaggle_credentials()
        user = (self.var_kaggle_username.get().strip() or creds.get("username", "")).strip()
        kn_slug = self.var_kaggle_kernel_slug.get().strip()
        if user and kn_slug:
            webbrowser.open(f"https://www.kaggle.com/code/{user}/{sanitize_kaggle_slug(kn_slug, prefix='moss-train')}")
        else:
            webbrowser.open("https://www.kaggle.com/code")

    def cloud_fetch_logs(self):
        """从 Kaggle CLI 拉取云端任务最新日志尾部并直接打印至右侧实时控制台。"""
        ident = self._resolve_kaggle_identity(require_creds=True)
        if not ident:
            return
        _u, _ds, _kn, _ds_id, kn_id = ident

        def _worker():
            res = fetch_kaggle_kernel_logs(kn_id, cli_info=self._kaggle_cli_info, tail_lines=220)
            if not res.get("ok"):
                err = res.get("error", "拉取云端日志失败")
                self._enqueue_ui(lambda m=err: self.monitor.log(f"⚠️ [云端日志] {kn_id}: {m}\n"))
                self._enqueue_ui(lambda m=err: messagebox.showwarning("暂未获取到云端日志", m))
                return

            tail = res.get("tail", "")
            total = res.get("line_count", 0)

            def _apply():
                self.monitor.log(
                    f"\n{'='*70}\n📜 [Kaggle 云端日志] {kn_id}（共 {total} 行，展示末尾片段）\n"
                    f"{'-'*70}\n{tail}\n{'='*70}\n"
                )
                self.var_cloud_job_status.set(f"📜 已拉取 {kn_id} 最新云端日志（共 {total} 行，见右侧控制台）")

            self._enqueue_ui(_apply)

        self._run_cloud_worker(f"📜 正在拉取 {kn_id} 云端执行日志...", _worker)

    def _stop_cloud_poll_timer(self):
        if self._cloud_poll_job is not None:
            try:
                self.root.after_cancel(self._cloud_poll_job)
            except Exception:
                pass
            self._cloud_poll_job = None
        self.var_cloud_auto_poll.set(False)

    def cloud_stop_and_open_kaggle(self):
        """立即关闭本地 30 秒自动轮询，并在浏览器打开 Kaggle 任务页指导点 Cancel Run 安全停机。"""
        self._stop_cloud_poll_timer()
        creds = read_kaggle_credentials()
        user = (self.var_kaggle_username.get().strip() or creds.get("username", "")).strip()
        kn_slug = sanitize_kaggle_slug(
            self.var_kaggle_kernel_slug.get().strip() or self.var_lora_name.get().strip() or "moss-train",
            prefix="moss-train",
        )
        kn_id = f"{user}/{kn_slug}" if user else kn_slug
        self.var_cloud_job_status.set("🛑 已停本地监控：请在打开的 Kaggle 网页点 Cancel Run（已设超时则到点自动存盘）")
        self.monitor.log(
            f"\n🛑 [停机引导] 已取消本地 30s 自动轮询（目标任务：{kn_id}）。\n"
            f" • Kaggle 公开 CLI 未开放按 slug 中止正在跑的 Batch 会话（`cancel_kernel_session` 仅接受运行时动态 session_id）；\n"
            f" • 请在刚刚打开的 Kaggle 任务页点击顶部/右侧【Cancel Run】或【Stop Session】立刻释放云端 GPU！\n"
            f" • 若训练已配置【自动存盘时限】，容器到达设定分钟数也会自动触发优雅存盘并归档最新权重。\n"
        )
        self.open_kaggle_kernel_page()
        messagebox.showinfo(
            "🛑 已停止本地监控并打开任务页",
            f"已停止本地 30 秒自动轮询。\n\n"
            f"请在浏览器打开的 Kaggle 任务页中点击【Cancel Run】/【Stop Session】即可立刻终止云端作业：\n"
            f"（注意：直接网页 Cancel 通常不会留存 ZIP，若希望留存最新检查点，建议等待设定的【自动存盘时限】自动收尾。）",
        )

    def cloud_delete_kernel_ui(self):
        """高危二次确认后，执行 `kaggle kernels delete <owner>/<slug> --yes` 彻底清除云端任务。"""
        ident = self._resolve_kaggle_identity(require_creds=True)
        if not ident:
            return
        _u, _ds, _kn, _ds_id, kn_id = ident
        confirmed = messagebox.askyesno(
            "⚠️ 高危确认：彻底删除云端任务",
            f"确定要永久删除 Kaggle 云端任务 `{kn_id}` 吗？\n\n"
            f"• 该操作会清空该任务在 Kaggle 的全部历史版本与未拉回的输出文件，且无法撤销！\n"
            f"• 若仅想中止本轮运行并保留已完成权重，建议改用【🛑 停监控 & 去网页点 Stop】或先拉回权重再删除。\n\n"
            f"仍要立即执行 `kaggle kernels delete {kn_id} --yes` 吗？",
            icon="warning",
            default="no",
        )
        if not confirmed:
            return

        self._stop_cloud_poll_timer()

        def _worker():
            log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
            ok, msg = delete_kaggle_kernel(kn_id, cli_info=self._kaggle_cli_info, on_log=log_cb)
            if ok:
                self._enqueue_ui(lambda m=msg: self.var_cloud_job_status.set(f"🗑️ {m}"))
                self._enqueue_ui(lambda m=msg: messagebox.showinfo("删除成功", m))
            else:
                self._enqueue_ui(lambda: self.var_cloud_job_status.set("🔴 删除云端任务失败，请查看右侧日志"))
                self._enqueue_ui(lambda m=msg: messagebox.showerror("删除失败", m))

        self._run_cloud_worker(f"🗑️ 正在彻底删除云端任务 {kn_id}...", _worker)

    def cloud_pull_outputs(self):
        ident = self._resolve_kaggle_identity(require_creds=True)
        if not ident:
            return
        _u, _ds, _kn, _ds_id, kn_id = ident
        lora_name = self.var_lora_name.get().strip() or "my_moss_lora"
        output_dir = self.var_output_dir.get().strip() or str((PROJECT_ROOT / "output").resolve())
        comfy_dir = self.var_comfy_dir.get().strip()
        copy_comfy = bool(self.var_copy_comfy.get())
        self.save_local_config()

        def _worker():
            log_cb = lambda s: self._enqueue_ui(lambda txt=s: self.monitor.log(txt))
            res = pull_and_extract_kaggle_outputs(
                kn_id,
                lora_name=lora_name,
                output_dir=output_dir,
                comfy_models_dir=comfy_dir,
                copy_to_comfy=copy_comfy,
                cli_info=self._kaggle_cli_info,
                on_log=log_cb,
            )
            if not res.get("ok"):
                err = res.get("error", "拉回失败，请先点击【刷新云端状态】确认云端任务是否显示 COMPLETE。")
                self._enqueue_ui(lambda m=err: self.var_cloud_job_status.set("🔴 拉回云端产物失败"))
                self._enqueue_ui(lambda m=err: messagebox.showerror("拉回产物失败", m))
                return

            def _on_ok():
                warn_line = f"\n\n{res['partial_warning']}" if res.get("partial_warning") else ""
                status_prefix = "🟡 已恢复拉回" if res.get("partial_warning") else "🟢 已成功拉回"
                self.var_cloud_job_status.set(f"{status_prefix} {len(res['safetensors'])} 个权重至 {res['output_dir']}")
                if res.get("metrics_path"):
                    self.monitor.load_metrics_from_path(Path(res["metrics_path"]), label=f"云端 {lora_name}", silent=False)
                comfy_note = f"\n已自动同步 ComfyUI loras：{len(res['comfy_synced'])} 个文件" if res["comfy_synced"] else ""
                messagebox.showinfo(
                    "🎉 云端成品拉回成功！",
                    f"已完成解压并归档至：\n{res['output_dir']}\n\n"
                    f"权重文件数：{len(res['safetensors'])} 个{comfy_note}\n"
                    f"{'已同步在右侧绘出云端 Loss 收敛曲线！' if res.get('metrics_path') else ''}{warn_line}",
                )

            self._enqueue_ui(_on_ok)

        self._run_cloud_worker(f"⏳ 正在拉回并解压 {kn_id} 成果...", _worker)

    def cloud_export_manual_bundle(self):
        src = self._cloud_source()
        is_hf = (src == "hf")
        is_url = (src == "url")
        is_r2 = (src == "r2")
        if is_hf:
            if not self._validate_hf_repo_ui(require_token=False):
                return
        elif is_url:
            if not self.var_external_dataset_url.get().strip():
                messagebox.showwarning(
                    "尚未填写或生成外链直拉地址",
                    "请先点击【📤 1. 仅打包图集上传外链】生成外链，或手动填入 .zip 直拉链接后再导出免本地传图 Notebook；\n"
                    "若需导出传统本地 ZIP 上传包，请切回【📦 Kaggle 私有集】。",
                )
                return
        elif is_r2:
            if not self.var_r2_dataset_url.get().strip():
                messagebox.showwarning(
                    "尚未生成 R2 预签名直链",
                    "请先点击【📤 1. 上传图集至 R2 私有桶】生成 7 天预签名直链，或粘贴有效 R2 直链后再导出 Notebook。",
                )
                return
        elif not self._check_dataset_inputs_ready():
            return

        raw_params = self._collect_raw_params()
        cleaned, err_msg = validate_training_params(raw_params)
        if err_msg:
            messagebox.showerror("参数格式错误", err_msg)
            return
        cloud_opts = self._cloud_opts()
        if not self._confirm_cloud_batch_sanity(cleaned, cloud_opts):
            return
        self.save_local_config()
        try:
            res = export_manual_upload_bundle(cleaned, raw_params, cloud_opts)
            if res.get("zip_path"):
                zip_note = f" • 图集 ZIP: {res['zip_path']}\n"
            elif is_r2:
                zip_note = " • 图集模式: ☁️ Cloudflare R2 预签名直拉（请注意预签名直链有效期为 7 天）\n"
            elif is_url:
                zip_note = " • 图集模式: 🔗 外链 ZIP 云端直拉（免导出本地图集 ZIP）\n"
            else:
                zip_note = " • 图集模式: 🤗 HF 云端直拉（免导出本地图集 ZIP）\n"
            tok_warn = (
                " • ⚠️ 安全提示：导出包未内置您填写的 HF Token；若为私有仓库，请在 Kaggle Add-ons -> Secrets 中配置 `HF_TOKEN`。\n"
                if (is_hf and self.var_hf_token.get().strip())
                else ""
            )
            self.monitor.log(
                f"\n[免 API 导出包] 已生成手动上传包：\n"
                f"{zip_note}"
                f" • 预填参数本: {res['notebook_path']}\n"
                f" • 操作指南: {res['guide_path']}\n"
                f"{tok_warn}"
            )
            open_in_explorer(res["export_dir"])
            if res.get("zip_path"):
                files_desc = f"{Path(res['zip_path']).name}、{Path(res['notebook_path']).name} 与 README_操作步骤.txt"
            else:
                direct_desc = "Cloudflare R2 直拉地址" if is_r2 else ("外链 ZIP 直拉地址" if is_url else "HF 云端直拉仓库")
                files_desc = f"{Path(res['notebook_path']).name} 与 README_操作步骤.txt（已预填{direct_desc}，无需上传图片）"
            messagebox.showinfo(
                "📦 免 API 手动上传包已就绪",
                f"已自动打开导出目录：\n{res['export_dir']}\n\n"
                f"目录内含：{files_desc}。\n"
                f"无需配置 Kaggle API，按文本里的步骤直接导入 Kaggle 网页即可开炼！",
            )
        except Exception as err:
            messagebox.showerror("导出失败", str(err))

    # ==================== 启动、优雅暂停、强杀与收尾工作 ====================
    def _check_dataset_inputs_ready(self) -> bool:
        """校验单图集或多子集目录与有效图片就绪状态，不通过时自动跳转并提示。"""
        if bool(self.var_multi_subset.get()):
            enabled_subs = [s for s in self.subsets if s.get("enabled", True)]
            if not enabled_subs:
                messagebox.showwarning("未启用子集", "多子集模式下至少需要启用 1 个包含有效图片的子集！")
                self.select_tab(1)
                return False
            for sub in enabled_subs:
                if not Path(sub.get("path", "")).is_dir() or count_images_in_dir(sub.get("path", "")) == 0:
                    messagebox.showwarning("子集无图片", f"子集 [{sub.get('name', '')}] 目录无效或无图片，请前往【📂 数据与续炼】修正！")
                    self.select_tab(1)
                    return False
            return True

        d_dir = self.var_data_dir.get().strip()
        if not Path(d_dir).exists() or count_images_in_dir(d_dir) == 0:
            messagebox.showwarning("图集为空", f"目录 {d_dir} 中未找到图片（支持 jpg/png/webp/bmp），请先选择包含图片与 .txt 标注的文件夹！")
            self.select_tab(1)
            return False
        return True

    def start_training(self):
        if self.is_training:
            return

        if not is_trainer_ready(PROJECT_ROOT / "KohakuMossTrainer"):
            messagebox.showerror("引擎未安装", "未发现完整的 KohakuMossTrainer（缺少 pyproject.toml 或 src/kohakumoss），请先切换到【📦 底模与框架】选项卡并点击【一键安装 / 同步引擎】！")
            self.select_tab(2)
            return

        if not self._check_dataset_inputs_ready():
            return

        multi_mode = bool(self.var_multi_subset.get())

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
                    self._enqueue_ui(lambda l=line: self.monitor.log(l))

            self.process.stdout.close()
            rc = self.process.wait()
            self._enqueue_ui(lambda code=rc: self.training_finished(code))
        except Exception as err:
            self._enqueue_ui(lambda e=err: self.monitor.log(f"\n❌ 运行异常崩溃: {e}\n"))
            self._enqueue_ui(lambda: self.training_finished(-1))

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

    # ==================== 任务档案管理（每 LoRA 独立本地 + 云端配置） ====================
    def _bind_dirty_check_traces(self):
        """为所有配置变量挂载轻量防抖脏检查监听。"""
        tracked_vars = [
            self.var_data_dir, self.var_multi_subset, self.var_comfy_dir, self.var_output_dir,
            self.var_lora_name, self.var_trigger, self.var_transformer_path, self.var_text_encoder_path,
            self.var_vae_path, self.var_copy_comfy, self.var_precision, self.var_base_precision,
            self.var_vram_preset, self.var_resolution, self.var_text_device, self.var_vae_device,
            self.var_algo, self.var_dora, self.var_target_preset, self.var_loraplus_ratio,
            self.var_rank, self.var_alpha, self.var_lr, self.var_repeats, self.var_epochs,
            self.var_steps, self.var_save_every, self.var_grad_acc, self.var_timestep_mu,
            self.var_tune, self.var_lokr_factor, self.var_lokr_full_matrix, self.var_lokr_decompose_both,
            self.var_train_norms, self.var_rank_dropout, self.var_module_dropout, self.var_input_dropout,
            self.var_optimizer, self.var_lr_scheduler, self.var_warmup_steps, self.var_weight_decay,
            self.var_wd_rules, self.var_ema, self.var_timestep, self.var_text_drop, self.var_flip,
            self.var_seed, self.var_save_format,
            self.var_caption_template, self.var_caption_p_nl, self.var_caption_variants,
            self.var_int8_compute, self.var_compile_mode, self.var_step_release,
            self.var_sample_every, self.var_sample_prompt,
            self.var_teacher_mode, self.var_distill_weight,
            self.var_train_mode, self.var_resume_path,
            self.var_kaggle_username, self.var_kaggle_dataset_slug, self.var_kaggle_kernel_slug,
            self.var_kaggle_accelerator, self.var_kaggle_variant, self.var_kaggle_base_quant,
            self.var_kaggle_dataset_source, self.var_external_dataset_url, self.var_hf_dataset_repo,
            self.var_hf_dataset_subdir, self.var_hf_upload_local, self.var_r2_account_id,
            self.var_r2_bucket, self.var_r2_prefix, self.var_r2_access_key_id,
            self.var_kaggle_max_run_min, self.var_target_compute, self.var_manual_img_count,
        ]
        for var in tracked_vars:
            var.trace_add("write", lambda *_: self._schedule_dirty_check())

    def _schedule_dirty_check(self):
        if getattr(self, "_suppress_var_traces", False) or getattr(self, "_closing", False):
            return
        dirty_job = getattr(self, "_dirty_check_job", None)
        if dirty_job is not None:
            try:
                self.root.after_cancel(dirty_job)
            except Exception:
                pass
        if hasattr(self, "root"):
            try:
                self._dirty_check_job = self.root.after(150, self._update_active_profile_bar)
            except Exception:
                pass

    def has_unsaved_profile_changes(self) -> bool:
        """判断当前控制台参数相对于已绑定的任务档案是否存在未保存修改。"""
        if not self.active_profile_id or self._active_profile_snapshot is None:
            return False
        return bool(compute_profile_diff(self._active_profile_snapshot, self._collect_raw_params()))

    def _confirm_unsaved_profile_before_switch(self) -> bool:
        """切换任务档案前拦截未保存改动；返回 True 表示允许继续切换，False 表示取消切换。"""
        if not self.has_unsaved_profile_changes():
            return True
        name = (self._active_profile_record or {}).get("name") or self.active_profile_id
        ans = messagebox.askyesnocancel(
            "当前任务档案有未保存修改",
            f"当前任务档案「{name}」的参数已被修改但尚未保存。\n\n"
            "• 点击【是】：先保存修改到当前档案，再切换\n"
            "• 点击【否】：放弃当前修改，直接切换\n"
            "• 点击【取消】：留在本任务继续编辑",
            icon="warning",
        )
        if ans is None:
            return False
        if ans is True:
            return bool(self.save_active_profile(silent=True))
        return True

    def _update_active_profile_bar(self):
        """刷新左侧顶部快捷任务条的当前任务名称、未保存标记及快捷下拉选项。"""
        self._dirty_check_job = None
        if self.active_profile_id and self._active_profile_record:
            name = self._active_profile_record.get("name", self.active_profile_id)
            dirty = self.has_unsaved_profile_changes()
            dirty_suffix = "  *未保存" if dirty else ""
            self.var_active_profile_badge.set(f"📌 当前任务：{name}{dirty_suffix}")
            target_display = self._format_quick_choice_label(self._active_profile_record)
            if self.var_quick_profile_select.get() != target_display:
                self.var_quick_profile_select.set(target_display)
        else:
            self.var_active_profile_badge.set("📌 当前任务：未绑定（草稿，点右方【➕ 另存】保存）")
            if self.var_quick_profile_select.get() != "（未绑定任务档案）":
                self.var_quick_profile_select.set("（未绑定任务档案）")

    @staticmethod
    def _format_quick_choice_label(record: dict) -> str:
        prefix = "⭐ " if record.get("pinned") else ""
        tc = (record.get("config") or {}).get("target_compute", "local")
        badge = {"local": "💻", "kaggle_t4x2": "☁️", "kaggle_t4x1": "⚡"}.get(tc, "🧩")
        return f"{prefix}{badge} {record.get('name', record.get('id', 'task'))} [{record.get('id', '')}]"

    def _sync_quick_profile_combobox(self, all_records: list):
        choices = ["（未绑定任务档案）"]
        mapping = {}
        for rec in all_records:
            label = self._format_quick_choice_label(rec)
            choices.append(label)
            mapping[label] = rec["id"]
        self._profile_quick_choices = mapping
        if hasattr(self, "cb_quick_profile"):
            self.cb_quick_profile.configure(values=choices)
        if self.active_profile_id and self._active_profile_record:
            cur_lbl = self._format_quick_choice_label(self._active_profile_record)
            if cur_lbl in mapping:
                self.var_quick_profile_select.set(cur_lbl)
                return
        self.var_quick_profile_select.set("（未绑定任务档案）")

    def _restore_active_profile_state(self):
        """启动时恢复上次绑定的档案元信息与基准快照（档案缺失时自动退回未绑定状态）。"""
        if not self.active_profile_id:
            self._active_profile_record = None
            self._active_profile_snapshot = None
            return
        try:
            rec = load_task_profile(self.active_profile_id, self.profiles_dir)
            self._active_profile_record = rec
            self._active_profile_snapshot = sanitize_profile_config(rec.get("config", {}))
        except Exception:
            self.active_profile_id = ""
            self._active_profile_record = None
            self._active_profile_snapshot = None

    def _resolve_profile_category_filter_key(self) -> str:
        label = (self.var_profile_cat_filter.get() or "全部类型").strip()
        for key, txt in PROFILE_CATEGORIES:
            if txt == label or key == label:
                return key
        return "all"

    def refresh_profile_list(self, select_id: str = None):
        """刷新任务档案列表、快捷下拉框以及选中档案的详情透视区。"""
        warnings = []
        all_records = list_task_profiles(self.profiles_dir, warnings=warnings)
        if warnings:
            self.monitor.log(f"⚠️ [任务档案] 跳过 {len(warnings)} 个损坏 JSON 文件：{'; '.join(warnings)}\n")

        self._sync_quick_profile_combobox(all_records)
        filtered = list_task_profiles(
            self.profiles_dir,
            query=self.var_profile_search.get().strip(),
            compute_filter=self.var_profile_compute_filter.get().strip() or "all",
            category_filter=self._resolve_profile_category_filter_key(),
        )
        self._profile_records_cache = filtered
        self.var_profile_count_stat.set(f"显示 {len(filtered)} / 共 {len(all_records)} 份档案")

        if not hasattr(self, "profile_tree"):
            return

        cur_sel = select_id or (self.profile_tree.selection()[0] if self.profile_tree.selection() else None)
        for iid in self.profile_tree.get_children():
            self.profile_tree.delete(iid)

        for rec in filtered:
            sm = format_profile_summary(rec)
            name_disp = rec["name"]
            if rec["id"] == self.active_profile_id:
                name_disp = f"▶ {name_disp}"
            self.profile_tree.insert(
                "",
                tk.END,
                iid=rec["id"],
                values=(
                    sm["pinned_icon"],
                    name_disp,
                    sm["compute_label"],
                    sm["algo_spec"],
                    sm["steps_lr"],
                    sm["cloud_src"],
                    sm["updated_short"],
                ),
            )

        target_iid = None
        if cur_sel and self.profile_tree.exists(cur_sel):
            target_iid = cur_sel
        elif self.active_profile_id and self.profile_tree.exists(self.active_profile_id):
            target_iid = self.active_profile_id
        elif filtered:
            target_iid = filtered[0]["id"]

        if target_iid:
            self.profile_tree.selection_set(target_iid)
            self.profile_tree.focus(target_iid)
            self.on_profile_tree_select()
        else:
            self._render_profile_detail(None)

    def reset_profile_filters(self):
        self.var_profile_search.set("")
        self.var_profile_compute_filter.set("all")
        self.var_profile_cat_filter.set("全部类型")
        self.refresh_profile_list()

    def _get_selected_profile_id(self) -> str:
        if not hasattr(self, "profile_tree"):
            return ""
        sel = self.profile_tree.selection()
        return str(sel[0]) if sel else ""

    def _find_cached_profile(self, profile_id: str):
        for rec in self._profile_records_cache:
            if rec.get("id") == profile_id:
                return rec
        return None

    def _render_profile_detail(self, record):
        if not hasattr(self, "profile_detail_text"):
            return
        self.profile_detail_text.config(state=tk.NORMAL)
        self.profile_detail_text.delete("1.0", tk.END)
        if not record:
            self.var_profile_detail_title.set("📄 档案详情透视")
            self.profile_detail_text.insert(
                "1.0",
                "当前暂无选中档案。点击上方【➕ 新建任务档案】可将当前配置保存为独立任务，或基于 8 套黄金预设快速创建新模板。",
            )
        else:
            sm = format_profile_summary(record)
            active_tag = " [当前激活]" if record.get("id") == self.active_profile_id else ""
            self.var_profile_detail_title.set(f"📄 档案详情：{record.get('name', '')}{active_tag}")
            self.profile_detail_text.insert("1.0", sm["detail_text"])
        self.profile_detail_text.config(state=tk.DISABLED)

    def on_profile_tree_select(self):
        pid = self._get_selected_profile_id()
        rec = self._find_cached_profile(pid)
        if rec is None and pid:
            try:
                rec = load_task_profile(pid, self.profiles_dir)
            except Exception:
                rec = None
        self._render_profile_detail(rec)

    def apply_config_dict(
        self,
        cfg: dict,
        *,
        preserve_missing_env_paths: bool = True,
        preset_id: str = None,
        clear_ephemeral_r2_url: bool = True,
    ):
        """将任务配置字典原子化回填到全部本地/云端控件，并在结束时统一刷新联动面板。"""
        clean_cfg = sanitize_profile_config(cfg)
        if preserve_missing_env_paths:
            cur_raw = self._collect_raw_params()
            if not clean_cfg.get("comfy_models_dir") or not os.path.exists(clean_cfg["comfy_models_dir"]):
                if cur_raw.get("comfy_models_dir") and os.path.exists(cur_raw["comfy_models_dir"]):
                    clean_cfg["comfy_models_dir"] = cur_raw["comfy_models_dir"]
            for model_key in ("transformer_path", "text_encoder_path", "vae_path"):
                val = clean_cfg.get(model_key, "")
                if (not val or not Path(val).is_file()) and cur_raw.get(model_key) and Path(cur_raw[model_key]).is_file():
                    clean_cfg[model_key] = cur_raw[model_key]
            if not clean_cfg.get("output_dir"):
                clean_cfg["output_dir"] = cur_raw.get("output_dir") or str((PROJECT_ROOT / "output").resolve())
            for acct_key in ("kaggle_username", "r2_account_id", "r2_bucket", "r2_access_key_id"):
                if not clean_cfg.get(acct_key) and cur_raw.get(acct_key):
                    clean_cfg[acct_key] = cur_raw[acct_key]

        default_dataset_dir = str((PROJECT_ROOT / "dataset").resolve())
        self._suppress_var_traces = True
        try:
            self.var_data_dir.set(clean_cfg.get("data_dir") or default_dataset_dir)
            self.var_multi_subset.set(bool(clean_cfg.get("multi_subset", False)))
            raw_subs = clean_cfg.get("subsets", [])
            if isinstance(raw_subs, list) and len(raw_subs) > 0:
                self.subsets = [dict(x) for x in raw_subs if isinstance(x, dict)]
            else:
                self.subsets = [
                    {
                        "name": "main_dataset",
                        "path": self.var_data_dir.get(),
                        "repeats": int(float(clean_cfg.get("repeats", "6") or 6)),
                        "trigger": clean_cfg.get("trigger_word", ""),
                        "enabled": True,
                    }
                ]

            self.var_comfy_dir.set(clean_cfg.get("comfy_models_dir", ""))
            self.var_output_dir.set(clean_cfg.get("output_dir") or str((PROJECT_ROOT / "output").resolve()))
            self.var_lora_name.set(clean_cfg.get("lora_name", "my_moss_lora"))
            self.var_trigger.set(clean_cfg.get("trigger_word", ""))
            self.var_transformer_path.set(clean_cfg.get("transformer_path", ""))
            self.var_text_encoder_path.set(clean_cfg.get("text_encoder_path", ""))
            self.var_vae_path.set(clean_cfg.get("vae_path", ""))
            self.var_copy_comfy.set(bool(clean_cfg.get("copy_to_comfy_loras", False)))

            self.var_precision.set(clean_cfg.get("precision", "bf16"))
            self.var_base_precision.set(clean_cfg.get("base_precision", "auto"))
            self.var_vram_preset.set(clean_cfg.get("vram_preset", "auto"))
            self.var_resolution.set(str(clean_cfg.get("resolution", "1024")))
            self.var_text_device.set(clean_cfg.get("text_device", "cpu"))
            self.var_vae_device.set(clean_cfg.get("vae_device", "cuda"))
            self.var_algo.set(clean_cfg.get("algo", "lora"))
            self.var_dora.set(bool(clean_cfg.get("dora", False)))
            self.var_target_preset.set(clean_cfg.get("target_preset", "attn-mlp"))
            self.var_loraplus_ratio.set(str(clean_cfg.get("loraplus_ratio", "1.0")))
            self.var_rank.set(str(clean_cfg.get("rank", "32")))
            self.var_alpha.set(str(clean_cfg.get("alpha", "32.0")))
            self.var_lr.set(str(clean_cfg.get("lr", "1e-4")))
            self.var_repeats.set(str(clean_cfg.get("repeats", "6")))
            self.var_epochs.set(str(clean_cfg.get("epochs", "4")))
            self.var_steps.set(str(clean_cfg.get("steps", "600")))
            self.var_save_every.set(str(clean_cfg.get("save_every", "100")))
            self.var_grad_acc.set(str(clean_cfg.get("grad_acc", "1")))
            self.var_timestep_mu.set(str(clean_cfg.get("timestep_mu", "-0.85")))
            self.var_tune.set(clean_cfg.get("tune", "off"))

            self.var_lokr_factor.set(str(clean_cfg.get("lokr_factor", ADVANCED_DEFAULTS["lokr_factor"])))
            self.var_lokr_full_matrix.set(bool(clean_cfg.get("lokr_full_matrix", ADVANCED_DEFAULTS["lokr_full_matrix"])))
            self.var_lokr_decompose_both.set(bool(clean_cfg.get("lokr_decompose_both", ADVANCED_DEFAULTS["lokr_decompose_both"])))
            self.var_train_norms.set(bool(clean_cfg.get("train_norms", ADVANCED_DEFAULTS["train_norms"])))
            self.var_rank_dropout.set(str(clean_cfg.get("rank_dropout", ADVANCED_DEFAULTS["rank_dropout"])))
            self.var_module_dropout.set(str(clean_cfg.get("module_dropout", ADVANCED_DEFAULTS["module_dropout"])))
            self.var_input_dropout.set(str(clean_cfg.get("input_dropout", ADVANCED_DEFAULTS["input_dropout"])))
            self.var_optimizer.set(str(clean_cfg.get("optimizer", ADVANCED_DEFAULTS["optimizer"])))
            self.var_lr_scheduler.set(str(clean_cfg.get("lr_scheduler", ADVANCED_DEFAULTS["lr_scheduler"])))
            self.var_warmup_steps.set(str(clean_cfg.get("warmup_steps", ADVANCED_DEFAULTS["warmup_steps"])))
            self.var_weight_decay.set(str(clean_cfg.get("weight_decay", ADVANCED_DEFAULTS["weight_decay"])))
            self.var_wd_rules.set(str(clean_cfg.get("wd_rules", ADVANCED_DEFAULTS["wd_rules"])))
            self.var_ema.set(str(clean_cfg.get("ema", ADVANCED_DEFAULTS["ema"])))
            self.var_timestep.set(str(clean_cfg.get("timestep", ADVANCED_DEFAULTS["timestep"])))
            self.var_text_drop.set(str(clean_cfg.get("text_drop", ADVANCED_DEFAULTS["text_drop"])))
            self.var_flip.set(bool(clean_cfg.get("flip", ADVANCED_DEFAULTS["flip"])))
            self.var_seed.set(str(clean_cfg.get("seed", ADVANCED_DEFAULTS["seed"])))
            self.var_save_format.set(str(clean_cfg.get("save_format", ADVANCED_DEFAULTS["save_format"])))
            self.var_caption_template.set(str(clean_cfg.get("caption_template", ADVANCED_DEFAULTS["caption_template"])))
            self.var_caption_p_nl.set(str(clean_cfg.get("caption_p_nl", ADVANCED_DEFAULTS["caption_p_nl"])))
            self.var_caption_variants.set(str(clean_cfg.get("caption_variants", ADVANCED_DEFAULTS["caption_variants"])))
            self.var_int8_compute.set(bool(clean_cfg.get("int8_compute", ADVANCED_DEFAULTS["int8_compute"])))
            self.var_compile_mode.set(str(clean_cfg.get("compile_mode", ADVANCED_DEFAULTS["compile_mode"])))
            self.var_step_release.set(str(clean_cfg.get("step_release", ADVANCED_DEFAULTS["step_release"])))
            self.var_sample_every.set(str(clean_cfg.get("sample_every", ADVANCED_DEFAULTS["sample_every"])))
            self.var_sample_prompt.set(str(clean_cfg.get("sample_prompt", ADVANCED_DEFAULTS["sample_prompt"])))
            self.var_teacher_mode.set(str(clean_cfg.get("teacher_mode", ADVANCED_DEFAULTS["teacher_mode"])))
            self.var_distill_weight.set(str(clean_cfg.get("distill_weight", ADVANCED_DEFAULTS["distill_weight"])))

            self.var_train_mode.set(clean_cfg.get("train_mode", "fresh"))
            self.var_resume_path.set(clean_cfg.get("resume_path", ""))

            self.var_kaggle_username.set(clean_cfg.get("kaggle_username", ""))
            self.var_kaggle_dataset_slug.set(clean_cfg.get("kaggle_dataset_slug", ""))
            self.var_kaggle_kernel_slug.set(clean_cfg.get("kaggle_kernel_slug", ""))
            self.var_kaggle_accelerator.set(clean_cfg.get("kaggle_accelerator", "NvidiaTeslaT4"))
            self.var_kaggle_variant.set(clean_cfg.get("kaggle_variant", "heal"))
            self.var_kaggle_base_quant.set(clean_cfg.get("kaggle_base_quant", "auto"))
            self.var_kaggle_dataset_source.set(clean_cfg.get("kaggle_dataset_source", "local"))
            self.var_external_dataset_url.set(clean_cfg.get("external_dataset_url", ""))
            self.var_hf_dataset_repo.set(clean_cfg.get("hf_dataset_repo", ""))
            self.var_hf_dataset_subdir.set(clean_cfg.get("hf_dataset_subdir", ""))
            self.var_hf_upload_local.set(bool(clean_cfg.get("hf_upload_local", True)))
            self.var_r2_account_id.set(clean_cfg.get("r2_account_id", ""))
            self.var_r2_bucket.set(clean_cfg.get("r2_bucket", ""))
            self.var_r2_prefix.set(clean_cfg.get("r2_prefix", "moss-datasets"))
            self.var_r2_access_key_id.set(clean_cfg.get("r2_access_key_id", ""))
            if clear_ephemeral_r2_url:
                self.var_r2_dataset_url.set("")
            self.var_kaggle_max_run_min.set(clean_cfg.get("kaggle_max_run_min", "540") or "540")
            self.var_target_compute.set(clean_cfg.get("target_compute", "local"))
            self.var_manual_img_count.set(clean_cfg.get("manual_img_count", ""))
        finally:
            self._suppress_var_traces = False

        self._apply_subset_panel_visibility()
        self.refresh_subset_tree()
        self.on_dynamic_controls_changed()
        self.on_cloud_dataset_source_changed()
        self.on_target_compute_changed(silent=True, from_cloud_tab=True)
        self.update_dataset_stats()
        self.update_output_preview()
        self.refresh_resume_status(auto_align_arch=False)
        self.on_dynamic_controls_changed()
        self.recalculate_math()
        self.refresh_cloud_sync_summary()
        self.update_quick_status_summary()
        self.highlight_active_preset(preset_id or None)

    def load_profile_by_id(self, profile_id: str, *, switch_to_tab=None, skip_dirty_check: bool = False) -> bool:
        """载入指定任务档案到主控制台（包含运行态拦截与未保存修改提示）。"""
        if self.is_training or self._cloud_busy:
            messagebox.showwarning("任务正在运行", "当前本地训练或云端操作正在进行中，请结束后再切换任务档案。")
            self._update_active_profile_bar()
            return False

        if not skip_dirty_check and not self._confirm_unsaved_profile_before_switch():
            self._update_active_profile_bar()
            return False

        try:
            rec = load_task_profile(profile_id, self.profiles_dir)
            rec = touch_profile_loaded(rec["id"], self.profiles_dir)
        except Exception as err:
            messagebox.showerror("载入档案失败", f"无法读取任务档案 [{profile_id}]：\n{err}")
            self.refresh_profile_list()
            return False

        self.apply_config_dict(
            rec["config"],
            preserve_missing_env_paths=True,
            preset_id=rec.get("preset_id") or None,
            clear_ephemeral_r2_url=True,
        )
        self.active_profile_id = rec["id"]
        self._active_profile_record = rec
        self._active_profile_snapshot = sanitize_profile_config(self._collect_raw_params())
        self.save_local_config()
        self.refresh_profile_list(select_id=rec["id"])
        self._update_active_profile_bar()
        self.monitor.log(f"[任务档案] 已载入「{rec['name']}」(ID={rec['id']}, 算力={rec['config'].get('target_compute')})\n")
        if switch_to_tab is not None:
            self.select_tab(switch_to_tab)
        return True

    def load_selected_profile(self, switch_to_tab=None):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先在列表中选中要载入的任务档案。")
            return
        self.load_profile_by_id(pid, switch_to_tab=switch_to_tab)

    def _on_quick_profile_selected(self, _event=None):
        sel_label = self.var_quick_profile_select.get()
        pid = self._profile_quick_choices.get(sel_label)
        if not pid:
            if not self.active_profile_id:
                return
            if not self._confirm_unsaved_profile_before_switch():
                self._update_active_profile_bar()
                return
            self.active_profile_id = ""
            self._active_profile_record = None
            self._active_profile_snapshot = None
            self.save_local_config()
            self.refresh_profile_list()
            self._update_active_profile_bar()
            return
        if pid == self.active_profile_id:
            return
        self.load_profile_by_id(pid)

    def save_active_profile(self, silent: bool = False) -> bool:
        """将当前控制台参数保存回当前激活档案；若尚未绑定档案，引导打开另存弹窗。"""
        if not self.active_profile_id or not self._active_profile_record:
            if silent:
                return False
            self.open_create_profile_dialog("current")
            return False

        cur = self._active_profile_record
        preset_id = getattr(self, "_active_preset_id", None) or cur.get("preset_id", "")
        try:
            updated = save_task_profile(
                name=cur["name"],
                config=self._collect_raw_params(),
                profiles_dir=self.profiles_dir,
                profile_id=cur["id"],
                category=cur.get("category", "custom"),
                tags=cur.get("tags", []),
                notes=cur.get("notes", ""),
                preset_id=preset_id or "",
                pinned=bool(cur.get("pinned", False)),
                overwrite=True,
            )
            self._active_profile_record = updated
            self._active_profile_snapshot = sanitize_profile_config(self._collect_raw_params())
            self.save_local_config()
            self.refresh_profile_list(select_id=updated["id"])
            self._update_active_profile_bar()
            self.monitor.log(f"[任务档案] 已保存当前参数到「{updated['name']}」\n")
            if not silent:
                messagebox.showinfo("保存成功", f"已将当前本地与云端参数保存至任务档案「{updated['name']}」。")
            return True
        except Exception as err:
            messagebox.showerror("保存失败", str(err))
            return False

    def save_current_to_selected_profile(self):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先在列表中选中一个目标档案。")
            return
        rec = self._find_cached_profile(pid) or load_task_profile(pid, self.profiles_dir)
        if pid != self.active_profile_id:
            if not messagebox.askyesno(
                "确认覆盖保存",
                f"确定将当前控制台的全部参数覆盖保存到档案「{rec['name']}」吗？",
            ):
                return
        preset_id = getattr(self, "_active_preset_id", None) or rec.get("preset_id", "")
        updated = save_task_profile(
            name=rec["name"],
            config=self._collect_raw_params(),
            profiles_dir=self.profiles_dir,
            profile_id=rec["id"],
            category=rec.get("category", "custom"),
            tags=rec.get("tags", []),
            notes=rec.get("notes", ""),
            preset_id=preset_id or "",
            pinned=bool(rec.get("pinned", False)),
            overwrite=True,
        )
        self.active_profile_id = updated["id"]
        self._active_profile_record = updated
        self._active_profile_snapshot = sanitize_profile_config(self._collect_raw_params())
        self.save_local_config()
        self.refresh_profile_list(select_id=updated["id"])
        self._update_active_profile_bar()
        self.monitor.log(f"[任务档案] 已覆盖保存至「{updated['name']}」\n")
        messagebox.showinfo("保存成功", f"已将当前参数保存至档案「{updated['name']}」并设为当前激活任务。")

    def _create_themed_dialog(self, title: str, geometry: str) -> tk.Toplevel:
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg=self.card_bg)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry(geometry)
        icon_path = PROJECT_ROOT / "assets" / "moss_trainer.ico"
        if icon_path.exists():
            try:
                dlg.iconbitmap(str(icon_path))
            except Exception:
                pass
        apply_windows_dark_titlebar(dlg)
        return dlg

    def open_create_profile_dialog(self, default_source: str = "current"):
        """弹出新建任务档案弹窗（支持从当前参数保存 / 基于黄金预设+算力创建 / 复制已有档案）。"""
        dlg = self._create_themed_dialog("➕ 新建 / 另存任务档案", "560x455")
        body = tk.Frame(dlg, bg=self.card_bg, padx=14, pady=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)

        cur_lora = self.var_lora_name.get().strip() or "my_moss_lora"
        v_name = tk.StringVar(value=f"{cur_lora}_{self.var_target_compute.get().strip() or 'local'}")
        cat_choices = [(k, lbl) for k, lbl in PROFILE_CATEGORIES if k != "all"]
        v_cat = tk.StringVar(value=cat_choices[-1][1])
        v_tags = tk.StringVar(value="")
        v_notes = tk.StringVar(value="")
        v_source = tk.StringVar(value=default_source if default_source in ("current", "preset", "copy") else "current")
        preset_choices = [f"{p['id']} | {p['label']}" for p in PRESETS]
        v_preset = tk.StringVar(value=preset_choices[0])
        v_preset_compute = tk.StringVar(value=self.var_target_compute.get().strip() or "local")
        all_existing = list_task_profiles(self.profiles_dir)
        copy_choices = [f"{r['id']} | {r['name']}" for r in all_existing] or ["（暂无现有档案）"]
        v_copy_src = tk.StringVar(value=copy_choices[0])
        v_sync_lora = tk.BooleanVar(value=True)
        v_load_now = tk.BooleanVar(value=True)

        tk.Label(body, text="档案名称:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9, "bold")).grid(row=0, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_name, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=4)

        tk.Label(body, text="任务分类:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Combobox(body, textvariable=v_cat, values=[lbl for _k, lbl in cat_choices], state="readonly").grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)

        tk.Label(body, text="标签 (逗号隔开):", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=2, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_tags, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)

        tk.Label(body, text="初始化来源:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9, "bold")).grid(row=3, column=0, sticky="w", pady=6)
        src_frame = tk.Frame(body, bg=self.card_bg)
        src_frame.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=6)

        preset_row = tk.Frame(body, bg=self.card_bg)
        preset_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=4)
        preset_row.columnconfigure(1, weight=1)
        tk.Label(preset_row, text="黄金预设:", bg=self.card_bg, fg=self.accent_blue, font=(self.font_family, 8, "bold")).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(preset_row, textvariable=v_preset, values=preset_choices, state="readonly").grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        tk.Label(preset_row, text="目标算力:", bg=self.card_bg, fg=self.accent_blue, font=(self.font_family, 8, "bold")).grid(row=1, column=0, sticky="w", pady=4)
        SegmentedSelector(
            preset_row,
            variable=v_preset_compute,
            options=[
                ("local", "💻 本机", self.accent_blue),
                ("kaggle_t4x2", "☁️ 双卡T4", self.accent_green),
                ("kaggle_t4x1", "⚡ 单卡T4", self.accent_yellow),
            ],
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)

        copy_row = tk.Frame(body, bg=self.card_bg)
        copy_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)
        copy_row.columnconfigure(1, weight=1)
        tk.Label(copy_row, text="复制源档案:", bg=self.card_bg, fg=self.accent_purple, font=(self.font_family, 8, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Combobox(copy_row, textvariable=v_copy_src, values=copy_choices, state="readonly").grid(row=0, column=1, sticky="ew", padx=(8, 0))

        def _toggle_src_rows():
            mode = v_source.get()
            if mode == "preset":
                preset_row.grid()
                copy_row.grid_remove()
            elif mode == "copy":
                preset_row.grid_remove()
                copy_row.grid()
            else:
                preset_row.grid_remove()
                copy_row.grid_remove()

        SegmentedSelector(
            src_frame,
            variable=v_source,
            options=[
                ("current", "保存当前控制台参数", self.accent_green),
                ("preset", "基于 8 套黄金预设", self.accent_blue),
                ("copy", "复制已有档案", self.accent_purple),
            ],
            command=_toggle_src_rows,
        ).pack(fill=tk.X)
        _toggle_src_rows()

        tk.Label(body, text="任务备注:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=6, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_notes, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=6, column=1, sticky="ew", padx=(8, 0), pady=4)

        ToggleButton(
            body,
            variable=v_sync_lora,
            text_on="同步将 LoRA 名称与云端 Slug 对齐此任务名（防产出撞名）",
            text_off="保持现有 LoRA 名称与云端 Slug 不变",
            on_bg=self.accent_blue,
            pady=3,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 4))

        ToggleButton(
            body,
            variable=v_load_now,
            text_on="创建后立即切换并绑定为当前激活任务",
            text_off="仅保存到档案库（保持当前控制台不变）",
            on_bg=self.accent_green,
            pady=3,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(0, 8))

        def _submit():
            name = v_name.get().strip()
            if not name:
                messagebox.showerror("名称不能为空", "请填写任务档案名称。", parent=dlg)
                return
            cat_key = next((k for k, lbl in cat_choices if lbl == v_cat.get()), "custom")
            src_mode = v_source.get()
            chosen_preset_id = getattr(self, "_active_preset_id", None) or ""

            if src_mode == "preset":
                chosen_preset_id = v_preset.get().split("|", 1)[0].strip()
                cfg = build_config_from_preset(
                    chosen_preset_id,
                    v_preset_compute.get(),
                    self._collect_raw_params(),
                    lora_name=sanitize_profile_id(name, "my_moss_lora") if v_sync_lora.get() else "",
                )
            elif src_mode == "copy":
                if not all_existing:
                    messagebox.showerror("暂无档案", "当前档案库为空，请选择“保存当前控制台参数”或“基于 8 套黄金预设”。", parent=dlg)
                    return
                src_id = v_copy_src.get().split("|", 1)[0].strip()
                src_rec = load_task_profile(src_id, self.profiles_dir)
                cfg = dict(src_rec["config"])
                chosen_preset_id = src_rec.get("preset_id", "")
            else:
                cfg = self._collect_raw_params()

            if v_sync_lora.get() and src_mode != "preset":
                safe_lora = sanitize_profile_id(name, fallback=cfg.get("lora_name", "my_moss_lora"))
                cfg["lora_name"] = safe_lora
                cfg["kaggle_dataset_slug"] = sanitize_kaggle_slug(f"{safe_lora}-data", prefix="moss-data")
                cfg["kaggle_kernel_slug"] = sanitize_kaggle_slug(f"{safe_lora}-train", prefix="moss-train")

            rec = save_task_profile(
                name=name,
                config=cfg,
                profiles_dir=self.profiles_dir,
                category=cat_key,
                tags=v_tags.get(),
                notes=v_notes.get(),
                preset_id=chosen_preset_id or "",
                pinned=False,
            )
            dlg.destroy()
            if v_load_now.get():
                self.load_profile_by_id(rec["id"], skip_dirty_check=True)
            else:
                self.refresh_profile_list(select_id=rec["id"])
                self._update_active_profile_bar()
            self.monitor.log(f"[任务档案] 已创建新档案「{rec['name']}」(ID={rec['id']})\n")

        btn_bar = tk.Frame(body, bg=self.card_bg)
        btn_bar.grid(row=9, column=0, columnspan=2, sticky="e", pady=(6, 0))
        FlatButton(btn_bar, text="取消", bg=self.input_bg, fg=self.text_sub, padx=14, pady=5, command=dlg.destroy).pack(side=tk.LEFT, padx=(0, 8))
        FlatButton(btn_bar, text="创建档案", bg=self.accent_green, fg="#11111b", font=(self.font_family, 9, "bold"), padx=16, pady=5, command=_submit).pack(side=tk.LEFT)

    def open_edit_profile_meta_dialog(self):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先在列表中选中要重命名或修改备注的档案。")
            return
        rec = load_task_profile(pid, self.profiles_dir)

        dlg = self._create_themed_dialog(f"✏️ 编辑任务档案：{rec['name']}", "500x310")
        body = tk.Frame(dlg, bg=self.card_bg, padx=14, pady=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)

        cat_choices = [(k, lbl) for k, lbl in PROFILE_CATEGORIES if k != "all"]
        cur_cat_lbl = next((lbl for k, lbl in cat_choices if k == rec.get("category")), cat_choices[-1][1])
        v_name = tk.StringVar(value=rec["name"])
        v_cat = tk.StringVar(value=cur_cat_lbl)
        v_tags = tk.StringVar(value=", ".join(rec.get("tags", [])))
        v_notes = tk.StringVar(value=rec.get("notes", ""))
        v_sync = tk.BooleanVar(value=False)

        tk.Label(body, text="档案名称:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9, "bold")).grid(row=0, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_name, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=4)
        tk.Label(body, text="分类:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Combobox(body, textvariable=v_cat, values=[lbl for _k, lbl in cat_choices], state="readonly").grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        tk.Label(body, text="标签:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=2, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_tags, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)
        tk.Label(body, text="备注:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=3, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_notes, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=4)

        ToggleButton(
            body,
            variable=v_sync,
            text_on="同时将档案内的 LoRA 名与云端 Slug 改为新名称",
            text_off="仅修改档案名称与备注（不动 LoRA 名与 Slug）",
            on_bg=self.accent_blue,
            pady=3,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=8)

        def _save():
            new_name = v_name.get().strip()
            if not new_name:
                messagebox.showerror("名称不能为空", "请输入有效任务名称。", parent=dlg)
                return
            cat_key = next((k for k, lbl in cat_choices if lbl == v_cat.get()), "custom")
            was_active = (pid == self.active_profile_id)
            updated = rename_task_profile(
                pid,
                new_name,
                self.profiles_dir,
                update_lora_name=bool(v_sync.get()),
                update_cloud_slugs=bool(v_sync.get()),
                category=cat_key,
                tags=v_tags.get(),
                notes=v_notes.get(),
            )
            dlg.destroy()
            if was_active:
                self.active_profile_id = updated["id"]
                self._active_profile_record = updated
                if v_sync.get():
                    self.apply_config_dict(updated["config"], preserve_missing_env_paths=True, preset_id=updated.get("preset_id") or None)
                    self._active_profile_snapshot = sanitize_profile_config(self._collect_raw_params())
                self.save_local_config()
            self.refresh_profile_list(select_id=updated["id"])
            self._update_active_profile_bar()

        btn_bar = tk.Frame(body, bg=self.card_bg)
        btn_bar.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))
        FlatButton(btn_bar, text="取消", bg=self.input_bg, fg=self.text_sub, padx=14, pady=4, command=dlg.destroy).pack(side=tk.LEFT, padx=(0, 8))
        FlatButton(btn_bar, text="保存修改", bg=self.accent_blue, fg="#11111b", font=(self.font_family, 9, "bold"), padx=14, pady=4, command=_save).pack(side=tk.LEFT)

    def duplicate_selected_profile(self):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先选中要复制的任务档案。")
            return
        try:
            dup = duplicate_task_profile(pid, profiles_dir=self.profiles_dir, sync_lora_and_slugs=True)
            self.refresh_profile_list(select_id=dup["id"])
            self.monitor.log(f"[任务档案] 已复制为新档案「{dup['name']}」(ID={dup['id']})\n")
        except Exception as err:
            messagebox.showerror("复制失败", str(err))

    def open_continue_profile_dialog(self):
        """基于选中档案派生独立的断点续炼 / 权重继承二训任务（继承架构与云端配置，自动分配新 Slug）。"""
        pid = self._get_selected_profile_id() or self.active_profile_id
        if not pid:
            messagebox.showinfo("提示", "请先在列表中选中一个源任务档案作为续炼基底。")
            return
        src = load_task_profile(pid, self.profiles_dir)

        dlg = self._create_themed_dialog(f"🔁 派生续炼/二训任务（源：{src['name']}）", "560x360")
        body = tk.Frame(dlg, bg=self.card_bg, padx=14, pady=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)

        v_new_name = tk.StringVar(value=f"{src['name']}_v2")
        v_mode = tk.StringVar(value="init_from")
        v_path = tk.StringVar(value=src["config"].get("resume_path", ""))
        v_steps = tk.StringVar(value=str(src["config"].get("steps", "300")))
        v_lr = tk.StringVar(value=str(src["config"].get("lr", "8e-5")))
        v_notes = tk.StringVar(value=f"继承「{src['name']}」二训")

        tk.Label(body, text="新任务名称:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9, "bold")).grid(row=0, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_new_name, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=4)

        tk.Label(body, text="续炼方式:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=1, column=0, sticky="w", pady=5)
        SegmentedSelector(
            body,
            variable=v_mode,
            options=[
                ("init_from", "🧬 继承权重再微调 (Init-From)", self.accent_purple),
                ("resume", "⏩ 断点续炼 (Resume)", self.accent_green),
            ],
        ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=5)

        tk.Label(body, text="权重或断点路径:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=2, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_path, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT, font=("Consolas", 8)).grid(row=2, column=1, sticky="ew", padx=(8, 4), pady=4)

        def _pick_path():
            if v_mode.get() == "resume":
                p = filedialog.askdirectory(title="选择包含 state.json 与 trainable.safetensors 的断点目录", parent=dlg)
            else:
                p = filedialog.askopenfilename(title="选择已有 .safetensors 权重", filetypes=[("Safetensors", "*.safetensors"), ("All", "*.*")], parent=dlg)
            if p:
                v_path.set(p)

        FlatButton(body, text="浏览...", bg=self.accent_blue, fg="#11111b", font=(self.font_family, 8, "bold"), padx=10, pady=2, command=_pick_path).grid(row=2, column=2, pady=4)

        tk.Label(body, text="目标步数 / 学习率:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=3, column=0, sticky="w", pady=4)
        hp_row = tk.Frame(body, bg=self.card_bg)
        hp_row.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=4)
        tk.Entry(hp_row, textvariable=v_steps, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT, width=10).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(hp_row, text="LR:", bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8)).pack(side=tk.LEFT)
        tk.Entry(hp_row, textvariable=v_lr, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT, width=12).pack(side=tk.LEFT, padx=(4, 0))

        tk.Label(body, text="备注:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9)).grid(row=4, column=0, sticky="w", pady=4)
        tk.Entry(body, textvariable=v_notes, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main, relief=tk.FLAT).grid(row=4, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=4)

        tk.Label(
            body,
            text="说明：将自动锁定源档案的 Algo/Rank/Alpha/DoRA/LoKr 与云端设置，并生成新 LoRA 名与独立 Kaggle Slug。",
            bg=self.card_bg,
            fg=self.accent_yellow,
            font=(self.font_family, 8),
            wraplength=520,
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 6))

        def _create():
            nm = v_new_name.get().strip()
            if not nm:
                messagebox.showerror("名称不能为空", "请填写派生续炼任务名称。", parent=dlg)
                return
            rec = create_continue_task_profile(
                pid,
                new_name=nm,
                train_mode=v_mode.get(),
                resume_path=v_path.get(),
                steps=v_steps.get(),
                lr=v_lr.get(),
                notes=v_notes.get(),
                profiles_dir=self.profiles_dir,
            )
            dlg.destroy()
            self.load_profile_by_id(rec["id"])

        btn_bar = tk.Frame(body, bg=self.card_bg)
        btn_bar.grid(row=6, column=0, columnspan=3, sticky="e", pady=(6, 0))
        FlatButton(btn_bar, text="取消", bg=self.input_bg, fg=self.text_sub, padx=14, pady=4, command=dlg.destroy).pack(side=tk.LEFT, padx=(0, 8))
        FlatButton(btn_bar, text="创建并载入", bg=self.accent_purple, fg="#11111b", font=(self.font_family, 9, "bold"), padx=14, pady=4, command=_create).pack(side=tk.LEFT)

    def toggle_pin_selected_profile(self):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先选中要置顶/取消置顶的档案。")
            return
        try:
            updated = toggle_pin_task_profile(pid, self.profiles_dir)
            if pid == self.active_profile_id:
                self._active_profile_record = updated
            self.refresh_profile_list(select_id=updated["id"])
            self._update_active_profile_bar()
        except Exception as err:
            messagebox.showerror("操作失败", str(err))

    def show_selected_profile_diff(self):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先选中要对比的任务档案。")
            return
        rec = load_task_profile(pid, self.profiles_dir)
        diffs = compute_profile_diff(rec["config"], self._collect_raw_params())
        if not diffs:
            messagebox.showinfo("完全一致", f"档案「{rec['name']}」与当前控制台参数完全一致，无差异项。")
            return
        lines = [f"档案「{rec['name']}」与当前控制台共有 {len(diffs)} 项差异（左侧=档案保存值，右侧=当前控制台）：\n"]
        for key, old_v, new_v in diffs:
            lines.append(f"• {key}:\n    档案: {old_v!r}\n    当前: {new_v!r}")
        show_text_viewer_dialog(
            self.root,
            f"🔍 参数差异对比：{rec['name']}",
            f"共发现 {len(diffs)} 项不同配置",
            self.accent_peach,
            "\n".join(lines),
            size=(760, 520),
        )

    def show_selected_profile_json(self):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先选中要查看 JSON 的任务档案。")
            return
        rec = load_task_profile(pid, self.profiles_dir)
        show_text_viewer_dialog(
            self.root,
            f"🧾 任务档案 JSON：{rec['name']}",
            f"文件路径：{self.profiles_dir / (rec['id'] + '.json')}",
            self.accent_yellow,
            json.dumps(rec, indent=2, ensure_ascii=False),
            size=(760, 540),
        )

    def delete_selected_profile(self):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先选中要删除的任务档案。")
            return
        rec = self._find_cached_profile(pid) or load_task_profile(pid, self.profiles_dir)
        if not messagebox.askyesno(
            "确认删除任务档案",
            f"确定删除任务档案「{rec['name']}」({pid}.json) 吗？\n\n"
            "安全提示：仅删除该配置档案 JSON 文件，您的本地图集和 output/ 训练成果权重将完整保留。",
            icon="warning",
        ):
            return
        delete_task_profile(pid, self.profiles_dir)
        if pid == self.active_profile_id:
            self.active_profile_id = ""
            self._active_profile_record = None
            self._active_profile_snapshot = None
            self.save_local_config()
        self.refresh_profile_list()
        self._update_active_profile_bar()
        self.monitor.log(f"[任务档案] 已删除配置档案「{rec['name']}」(图集与权重未受影响)\n")

    def import_profile_dialog(self):
        src = filedialog.askopenfilename(
            title="导入任务档案或 local_config.json",
            filetypes=[("JSON 配置文件", "*.json"), ("所有文件", "*.*")],
        )
        if not src:
            return
        try:
            rec = import_task_profile_from_file(Path(src), self.profiles_dir)
            self.refresh_profile_list(select_id=rec["id"])
            self.monitor.log(f"[任务档案] 已导入档案「{rec['name']}」(ID={rec['id']})\n")
            if messagebox.askyesno("导入成功", f"已导入任务档案「{rec['name']}」并过滤敏感字段。\n是否立即载入到控制台？"):
                self.load_profile_by_id(rec["id"])
        except Exception as err:
            messagebox.showerror("导入失败", f"无法解析配置文件：\n{err}")

    def export_selected_profile_dialog(self):
        pid = self._get_selected_profile_id()
        if not pid:
            messagebox.showinfo("提示", "请先选中要导出的任务档案。")
            return
        rec = load_task_profile(pid, self.profiles_dir)
        portable_choice = messagebox.askyesnocancel(
            "选择导出模式",
            f"准备导出任务档案「{rec['name']}」：\n\n"
            "• 点击【是】：导出【便携分享模板】（自动剥离本机路径、Kaggle/R2 账号与临时外链，保留全部训练与云端架构参数）\n"
            "• 点击【否】：导出【完整本机备份】（保留本机路径与非密钥云端配置，仍绝不含 API Key/Token）\n"
            "• 点击【取消】：中止导出",
        )
        if portable_choice is None:
            return
        suffix = "_portable" if portable_choice else "_backup"
        dest = filedialog.asksaveasfilename(
            title="保存导出的任务档案 JSON",
            defaultextension=".json",
            initialfile=f"{rec['id']}{suffix}.json",
            filetypes=[("JSON 文件", "*.json")],
        )
        if not dest:
            return
        try:
            out_fp = export_task_profile_to_file(rec, Path(dest), portable=bool(portable_choice))
            messagebox.showinfo("导出成功", f"任务档案已成功导出至：\n{out_fp}")
        except Exception as err:
            messagebox.showerror("导出失败", str(err))

    def open_profiles_folder(self):
        try:
            open_in_explorer(str(self.profiles_dir))
        except Exception as err:
            messagebox.showerror("打开失败", str(err))


def run_gui():
    """启动桌面 GUI 控制台并记录未捕获崩溃日志到 workspace/gui_crash.log。"""
    _configure_windows_shell_ui()
    try:
        root = tk.Tk()
        # 隐藏至首次布局与样式计算完毕后再显示，消除冷启动白屏闪烁
        root.withdraw()
        ModernLocalTrainerGUI(root)
        root.update_idletasks()
        root.deiconify()
        apply_windows_dark_titlebar(root)
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
