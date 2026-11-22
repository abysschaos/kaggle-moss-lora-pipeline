"""Tab 4 (☁️ 云端遥控): Foolproof Kaggle Cloud API remote workbench & offline manual bundle exporter."""

import tkinter as tk
import webbrowser

from .theme import PALETTE
from .widgets import FlatButton, HoverTooltip, ScrollableFrame, SegmentedSelector, ToggleButton


def build_cloud_tab(parent, app) -> ScrollableFrame:
    """构建 Tab 4【☁️ 云端遥控】无代码四步 Kaggle 云端遥控台面板。"""
    c = PALETTE
    scroll = ScrollableFrame(parent, bg=c["card_bg"])
    scroll.pack(fill=tk.BOTH, expand=True)
    body = scroll.content
    app.cloud_action_buttons = []

    # ==================== 卡片 1：Kaggle API 凭证与 CLI 状态 ====================
    auth_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    auth_card.pack(fill=tk.X, pady=(0, 7))

    tk.Label(
        auth_card,
        text="🔐 第 1 步 · 连接 Kaggle API（一次配置永久生效 · 仅存 ~/.kaggle/kaggle.json）",
        bg=c["bg_dark"],
        fg=c["accent_purple"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 5))

    status_lbl = tk.Label(
        auth_card,
        textvariable=app.var_kaggle_status,
        bg=c["card_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        padx=8,
        pady=5,
        wraplength=560,
    )
    status_lbl.pack(fill=tk.X, pady=(0, 6))
    scroll.register_wrap_label(status_lbl)

    cred_grid = tk.Frame(auth_card, bg=c["bg_dark"])
    cred_grid.pack(fill=tk.X, pady=(0, 6))
    cred_grid.columnconfigure(1, weight=1)
    cred_grid.columnconfigure(3, weight=2)

    lbl_user = tk.Label(cred_grid, text="Kaggle 用户名:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8))
    lbl_user.grid(row=0, column=0, sticky="w", padx=(0, 6))
    ent_user = tk.Entry(
        cred_grid,
        textvariable=app.var_kaggle_username,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
    )
    ent_user.grid(row=0, column=1, sticky="ew", padx=(0, 10))

    lbl_key = tk.Label(cred_grid, text="API Key (隐藏):", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8))
    lbl_key.grid(row=0, column=2, sticky="w", padx=(0, 6))
    ent_key = tk.Entry(
        cred_grid,
        textvariable=app.var_kaggle_api_key,
        show="*",
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
    )
    ent_key.grid(row=0, column=3, sticky="ew")
    HoverTooltip(
        ent_key,
        "安全提示：API Key 仅用于保存至本机用户目录 ~/.kaggle/kaggle.json，绝不写入项目配置文件或日志。\n"
        "如果您已点【导入 kaggle.json】或系统已有旧凭证，无需重复填写。",
    )

    auth_btns = tk.Frame(auth_card, bg=c["bg_dark"])
    auth_btns.pack(fill=tk.X)
    for text, bg, fg, cmd in (
        ("📥 导入 kaggle.json", c["accent_blue"], "#11111b", app.import_kaggle_json_dialog),
        ("💾 保存凭证", c["input_bg"], c["accent_green"], app.save_kaggle_credentials_ui),
        ("🔍 检测连通性", c["input_bg"], c["accent_yellow"], app.test_kaggle_connection_ui),
        ("🌐 获取 Token", c["input_bg"], c["text_main"], lambda: webbrowser.open("https://www.kaggle.com/settings")),
        ("⬇️ 安装 CLI", c["input_bg"], c["accent_peach"], app.install_kaggle_cli_ui),
    ):
        btn = FlatButton(
            auth_btns,
            text=text,
            bg=bg,
            fg=fg,
            font=(c["font_family"], 8, "bold"),
            padx=7,
            pady=4,
            command=cmd,
        )
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        app.cloud_action_buttons.append(btn)

    # ==================== 卡片 2：云端算力、底模变体与自动 Slug 配置 ====================
    cfg_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    cfg_card.pack(fill=tk.X, pady=(0, 7))

    tk.Label(
        cfg_card,
        text="⚡ 第 2 步 · 云端算力与任务配置（实时继承 Tab 1 超参数 & Tab 2 图集/多子集）",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 5))

    sync_lbl = tk.Label(
        cfg_card,
        textvariable=app.var_cloud_sync_summary,
        bg=c["card_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        padx=8,
        pady=5,
        wraplength=560,
    )
    sync_lbl.pack(fill=tk.X, pady=(0, 6))
    scroll.register_wrap_label(sync_lbl)

    acc_row = tk.Frame(cfg_card, bg=c["bg_dark"])
    acc_row.pack(fill=tk.X, pady=(0, 5))
    tk.Label(acc_row, text="云端算力队列:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8), width=13, anchor="w").pack(side=tk.LEFT)
    SegmentedSelector(
        acc_row,
        variable=app.var_kaggle_accelerator,
        options=[
            ("NvidiaTeslaT4", "🔥 双卡 T4 DDP (推荐)", c["accent_green"]),
            ("default_gpu", "⚡ 单卡默认 GPU", c["accent_blue"]),
            ("NvidiaTeslaP100", "🟦 兼容队列", c["accent_yellow"]),
        ],
        command=app.refresh_cloud_sync_summary,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    var_row = tk.Frame(cfg_card, bg=c["bg_dark"])
    var_row.pack(fill=tk.X, pady=(0, 5))
    tk.Label(var_row, text="云端底模分支:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8), width=13, anchor="w").pack(side=tk.LEFT)
    SegmentedSelector(
        var_row,
        variable=app.var_kaggle_variant,
        options=[
            ("heal", "🩺 Heal (防溢出推荐)", c["accent_green"]),
            ("base", "🧱 Base 原版", c["accent_blue"]),
            ("sft", "🎯 SFT 对齐版", c["accent_purple"]),
        ],
        command=app.refresh_cloud_sync_summary,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    quant_row = tk.Frame(cfg_card, bg=c["bg_dark"])
    quant_row.pack(fill=tk.X, pady=(0, 6))
    tk.Label(quant_row, text="底模下载量化:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8), width=13, anchor="w").pack(side=tk.LEFT)
    SegmentedSelector(
        quant_row,
        variable=app.var_kaggle_base_quant,
        options=[
            ("auto", "auto (双卡自动 int8)", c["accent_blue"]),
            ("int8", "int8 (省显存快拉)", c["accent_green"]),
            ("full", "full (全精度 BF16)", c["accent_yellow"]),
        ],
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    slug_grid = tk.Frame(cfg_card, bg=c["bg_dark"])
    slug_grid.pack(fill=tk.X)
    slug_grid.columnconfigure(1, weight=1)
    slug_grid.columnconfigure(3, weight=1)

    lbl_ds = tk.Label(slug_grid, text="私有图集 Slug:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8))
    lbl_ds.grid(row=0, column=0, sticky="w", padx=(0, 5))
    ent_ds = tk.Entry(
        slug_grid,
        textvariable=app.var_kaggle_dataset_slug,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
    )
    ent_ds.grid(row=0, column=1, sticky="ew", padx=(0, 8))

    lbl_kn = tk.Label(slug_grid, text="云端任务 Slug:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8))
    lbl_kn.grid(row=0, column=2, sticky="w", padx=(0, 5))
    ent_kn = tk.Entry(
        slug_grid,
        textvariable=app.var_kaggle_kernel_slug,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
    )
    ent_kn.grid(row=0, column=3, sticky="ew", padx=(0, 6))

    btn_auto_slug = FlatButton(
        slug_grid,
        text="🔄 自动生成",
        bg=c["input_bg"],
        fg=c["accent_blue"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=3,
        command=app.auto_generate_cloud_slugs,
    )
    btn_auto_slug.grid(row=0, column=4)
    slug_tip = "Kaggle 规则：仅允许小写字母、数字和连字符(-)，长度 6~50。留空或包含中文时点【自动生成】会自动转为合法英文 Slug。"
    HoverTooltip(ent_ds, slug_tip)
    HoverTooltip(ent_kn, slug_tip)

    # ==================== 卡片 3：傻瓜式一键云端流水线操作区 ====================
    pipe_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    pipe_card.pack(fill=tk.X, pady=(0, 7))

    tk.Label(
        pipe_card,
        text="🚀 第 3 步 · 一键全自动云端炼丹与回传到本地",
        bg=c["bg_dark"],
        fg=c["accent_green"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 6))

    master_btn = FlatButton(
        pipe_card,
        text="🚀 一键全自动：打包上传图集 + 同步当前参数推送云端开炼",
        bg="#2e7d32",
        fg="white",
        hover_bg="#388e3c",
        pressed_bg="#1b5e20",
        font=(c["font_family"], 9, "bold"),
        pady=8,
        command=app.cloud_one_click_deploy,
    )
    master_btn.pack(fill=tk.X, pady=(0, 6))
    app.cloud_action_buttons.append(master_btn)

    step_row = tk.Frame(pipe_card, bg=c["bg_dark"])
    step_row.pack(fill=tk.X, pady=(0, 5))
    for text, bg, fg, cmd in (
        ("1️⃣ 打包并上传图集", c["input_bg"], c["accent_blue"], app.cloud_upload_dataset_only),
        ("2️⃣ 仅推送参数开炼", c["input_bg"], c["accent_yellow"], app.cloud_push_kernel_only),
        ("3️⃣ 一键拉回成品权重", c["accent_purple"], "#11111b", app.cloud_pull_outputs),
    ):
        btn = FlatButton(
            step_row,
            text=text,
            bg=bg,
            fg=fg,
            font=(c["font_family"], 8, "bold"),
            padx=8,
            pady=6,
            command=cmd,
        )
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        app.cloud_action_buttons.append(btn)

    fallback_row = tk.Frame(pipe_card, bg=c["bg_dark"])
    fallback_row.pack(fill=tk.X)
    btn_export = FlatButton(
        fallback_row,
        text="📦 导出免 API 手动上传包 (图集 ZIP + 预填参数 Notebook)",
        bg=c["input_bg"],
        fg=c["accent_peach"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=5,
        command=app.cloud_export_manual_bundle,
    )
    btn_export.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    app.cloud_action_buttons.append(btn_export)

    FlatButton(
        fallback_row,
        text="📂 打开输出目录",
        bg=c["input_bg"],
        fg=c["text_main"],
        font=(c["font_family"], 8),
        padx=10,
        pady=5,
        command=app.open_lora_output_folder,
    ).pack(side=tk.RIGHT)

    # ==================== 卡片 4：云端进度轮询与资产快捷入口 ====================
    mon_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    mon_card.pack(fill=tk.X, pady=(0, 4))

    tk.Label(
        mon_card,
        text="📡 第 4 步 · 云端作业状态监控与手动资产入口",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 5))

    job_lbl = tk.Label(
        mon_card,
        textvariable=app.var_cloud_job_status,
        bg=c["card_bg"],
        fg=c["text_main"],
        font=(c["font_family"], 8, "bold"),
        anchor="w",
        justify=tk.LEFT,
        padx=8,
        pady=5,
        wraplength=560,
    )
    job_lbl.pack(fill=tk.X, pady=(0, 6))
    scroll.register_wrap_label(job_lbl)

    poll_row = tk.Frame(mon_card, bg=c["bg_dark"])
    poll_row.pack(fill=tk.X, pady=(0, 6))

    btn_refresh = FlatButton(
        poll_row,
        text="🔄 刷新云端状态",
        bg=c["input_bg"],
        fg=c["accent_blue"],
        font=(c["font_family"], 8, "bold"),
        padx=10,
        pady=4,
        command=lambda: app.cloud_refresh_status(silent=False),
    )
    btn_refresh.pack(side=tk.LEFT, padx=(0, 6))
    app.cloud_action_buttons.append(btn_refresh)

    ToggleButton(
        poll_row,
        variable=app.var_cloud_auto_poll,
        text_on="⏱️ 自动轮询 (30s) 已开",
        text_off="⏱️ 自动轮询 (30s)",
        on_bg=c["accent_green"],
        off_fg=c["accent_green"],
        padx=8,
        pady=4,
        command=app.toggle_cloud_auto_poll,
    ).pack(side=tk.LEFT, padx=(0, 6))

    FlatButton(
        poll_row,
        text="🌐 在浏览器打开 Kaggle 任务页",
        bg=c["input_bg"],
        fg=c["accent_peach"],
        font=(c["font_family"], 8, "bold"),
        padx=10,
        pady=4,
        command=app.open_kaggle_kernel_page,
    ).pack(side=tk.RIGHT)

    quick_row = tk.Frame(mon_card, bg=c["bg_dark"])
    quick_row.pack(fill=tk.X)
    for text, fg, cmd in (
        ("📓 Notebook 源码", c["text_sub"], lambda: app.open_project_file("notebooks/kaggle_moss_lora_training.ipynb")),
        ("📖 云端部署指南", c["accent_purple"], lambda: app.open_doc_viewer("Kaggle / 云端部署实操指南", "docs/kaggle_deployment_guide.md")),
        ("⚙️ kernel-metadata.json", c["accent_yellow"], lambda: app.open_project_file("kernel-metadata.json")),
        ("🔐 .env.example", c["text_sub"], lambda: app.open_doc_viewer("环境变量凭据模板 (.env.example)", ".env.example")),
    ):
        FlatButton(
            quick_row,
            text=text,
            bg=c["card_bg"],
            fg=fg,
            font=(c["font_family"], 8),
            padx=6,
            pady=4,
            command=cmd,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

    return scroll
