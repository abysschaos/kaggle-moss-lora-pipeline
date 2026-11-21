"""Tab 3 (📦 底模与框架): ComfyUI base model auto-scan/manual override and KohakuMossTrainer engine manager."""

import tkinter as tk

from core.config_store import open_in_explorer
from .theme import PALETTE
from .widgets import ScrollableFrame, create_file_picker_block


def build_models_tab(parent, app) -> ScrollableFrame:
    """构建 Tab 3【📦 底模与框架】面板并返回其 ScrollableFrame。"""
    c = PALETTE
    scroll = ScrollableFrame(parent, bg=c["card_bg"])
    scroll.pack(fill=tk.BOTH, expand=True)
    body = scroll.content

    # ==================== 1. 三核心底模寻址卡 ====================
    model_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    model_card.pack(fill=tk.X, pady=(0, 8))

    mdl_hdr = tk.Frame(model_card, bg=c["bg_dark"])
    mdl_hdr.pack(fill=tk.X, pady=(0, 5))
    tk.Label(
        mdl_hdr,
        text="📦 底层大模型路径 (优先检索本地目录与 ComfyUI/models)",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
    ).pack(side=tk.LEFT)

    tk.Button(
        mdl_hdr,
        text="🔍 自动扫描底模",
        bg=c["input_bg"],
        fg=c["accent_green"],
        relief=tk.FLAT,
        font=(c["font_family"], 8, "bold"),
        cursor="hand2",
        padx=8,
        pady=2,
        command=lambda: app.auto_scan_models(silent=False),
    ).pack(side=tk.RIGHT)

    scan_lbl = tk.Label(
        model_card,
        textvariable=app.var_model_scan_stat,
        bg=c["card_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        padx=8,
        pady=5,
        wraplength=560,
    )
    scan_lbl.pack(fill=tk.X, pady=(0, 6))
    scroll.register_wrap_label(scan_lbl)

    model_pick_wrap = tk.Frame(model_card, bg=c["card_bg"], padx=8, pady=6)
    model_pick_wrap.pack(fill=tk.X)

    create_file_picker_block(
        model_pick_wrap,
        "主模型 Transformer (MOSS_image_base_bf16 / int8 / fp8.safetensors):",
        app.var_transformer_path,
        "选择 MOSS-Image Transformer 主模型",
        "models/diffusion_models",
        open_in_explorer,
    )
    create_file_picker_block(
        model_pick_wrap,
        "文本编码器 Text Encoder (Qwen3.5-9B / int8 .safetensors):",
        app.var_text_encoder_path,
        "选择 Qwen3.5 Text Encoder 权重",
        "models/text_encoders",
        open_in_explorer,
    )
    create_file_picker_block(
        model_pick_wrap,
        "变分自编码器 VAE (moss_image_vae.safetensors):",
        app.var_vae_path,
        "选择 MOSS-Image VAE 权重",
        "models/vae",
        open_in_explorer,
    )

    # ==================== 2. KohakuMossTrainer 引擎管理卡 ====================
    trainer_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    trainer_card.pack(fill=tk.X, pady=(0, 6))

    trn_hdr = tk.Frame(trainer_card, bg=c["bg_dark"])
    trn_hdr.pack(fill=tk.X, pady=(0, 6))
    tk.Label(
        trn_hdr,
        text="🧩 底层训练核心引擎 (KohakuMossTrainer)",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
    ).pack(side=tk.LEFT)

    for text, fg, cmd in (
        ("🌐 GitHub", c["text_sub"], app.open_trainer_github),
        ("📂 目录", c["accent_green"], app.open_trainer_folder),
        ("🔧 补丁修复", c["accent_yellow"], app.patch_trainer_manually),
    ):
        tk.Button(
            trn_hdr,
            text=text,
            bg=c["input_bg"],
            fg=fg,
            relief=tk.FLAT,
            font=(c["font_family"], 8),
            cursor="hand2",
            padx=7,
            pady=2,
            command=cmd,
        ).pack(side=tk.RIGHT, padx=(4, 0))

    trn_body = tk.Frame(trainer_card, bg=c["card_bg"], padx=8, pady=6)
    trn_body.pack(fill=tk.X)

    tk.Button(
        trn_body,
        text="⬇️ 一键安装 / 同步引擎",
        bg=c["accent_blue"],
        fg="#11111b",
        relief=tk.FLAT,
        font=(c["font_family"], 8, "bold"),
        cursor="hand2",
        padx=10,
        pady=4,
        command=app.download_or_clone_trainer,
    ).pack(side=tk.RIGHT, padx=(8, 0))

    trn_stat_lbl = tk.Label(
        trn_body,
        textvariable=app.var_trainer_status,
        bg=c["card_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=400,
    )
    trn_stat_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    scroll.register_wrap_label(trn_stat_lbl)

    return scroll
