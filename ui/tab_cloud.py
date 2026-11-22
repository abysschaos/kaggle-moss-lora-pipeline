"""Tab 4 (☁️ 云端遥控): Foolproof Kaggle Cloud API remote workbench (Local / HF direct pull, live logs, runtime guard & stop controls)."""

import tkinter as tk
from tkinter import ttk
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

    tk.Label(cred_grid, text="Kaggle 用户名:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8)).grid(row=0, column=0, sticky="w", padx=(0, 6))
    tk.Entry(
        cred_grid,
        textvariable=app.var_kaggle_username,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
    ).grid(row=0, column=1, sticky="ew", padx=(0, 10))

    tk.Label(cred_grid, text="API Key (隐藏):", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8)).grid(row=0, column=2, sticky="w", padx=(0, 6))
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

    # ==================== 卡片 2：云端算力、数据源、超时存盘与 Slug 配置 ====================
    cfg_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    cfg_card.pack(fill=tk.X, pady=(0, 7))

    tk.Label(
        cfg_card,
        text="⚡ 第 2 步 · 云端算力、图集来源与防超时设置（继承 Tab 1 超参）",
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
        command=app.on_cloud_accelerator_changed,
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
    quant_row.pack(fill=tk.X, pady=(0, 5))
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

    guard_row = tk.Frame(cfg_card, bg=c["bg_dark"])
    guard_row.pack(fill=tk.X, pady=(0, 5))
    lbl_guard = tk.Label(guard_row, text="自动存盘时限(分):", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8), width=13, anchor="w")
    lbl_guard.pack(side=tk.LEFT)
    cb_guard = ttk.Combobox(
        guard_row,
        textvariable=app.var_kaggle_max_run_min,
        values=["0", "180", "300", "480", "540", "660"],
        width=8,
    )
    cb_guard.pack(side=tk.LEFT, padx=(0, 8))
    tk.Label(
        guard_row,
        text="到点自动保存并打包退出（默认 540分 = 9小时，赶在 Kaggle 12h 硬超时前保住成果；0=不限制）",
        bg=c["bg_dark"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
        anchor="w",
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    HoverTooltip(
        cb_guard,
        "Kaggle 后台任务最长运行 12 小时后会被强杀且不会打包输出。\n"
        "设为 540 分钟（9小时）时，云端会在超时前自动触发保存当前步检查点并导出 results.zip！",
    )

    src_row = tk.Frame(cfg_card, bg=c["bg_dark"])
    src_row.pack(fill=tk.X, pady=(0, 6))
    tk.Label(src_row, text="训练图集来源:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8), width=13, anchor="w").pack(side=tk.LEFT)
    SegmentedSelector(
        src_row,
        variable=app.var_kaggle_dataset_source,
        options=[
            ("local", "📦 本地打包上传到 Kaggle", c["accent_blue"]),
            ("hf", "🤗 云端直拉 Hugging Face 图集", c["accent_green"]),
        ],
        command=app.on_cloud_dataset_source_changed,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # 动态源区 A：本地打包上传时的私有图集 Slug
    app.cloud_local_slug_frame = tk.Frame(cfg_card, bg=c["bg_dark"])
    app.cloud_local_slug_frame.columnconfigure(1, weight=1)
    tk.Label(app.cloud_local_slug_frame, text="私有图集 Slug:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8), width=13, anchor="w").grid(row=0, column=0, sticky="w", pady=(0, 4))
    ent_ds = tk.Entry(
        app.cloud_local_slug_frame,
        textvariable=app.var_kaggle_dataset_slug,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
    )
    ent_ds.grid(row=0, column=1, sticky="ew", pady=(0, 4))
    HoverTooltip(ent_ds, "Kaggle 私有数据集标识：仅小写字母/数字/连字符(-)，6~50位；点下方【自动生成】自动转换。")

    # 动态源区 B：免本地上传直拉 Hugging Face Dataset
    app.cloud_hf_frame = tk.Frame(cfg_card, bg=c["card_bg"], padx=8, pady=6)
    app.cloud_hf_frame.columnconfigure(1, weight=2)
    app.cloud_hf_frame.columnconfigure(3, weight=1)

    tk.Label(app.cloud_hf_frame, text="HF 仓库 / 链接:", bg=c["card_bg"], fg=c["accent_green"], font=(c["font_family"], 8, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 5), pady=(0, 4))
    ent_hf_repo = tk.Entry(
        app.cloud_hf_frame,
        textvariable=app.var_hf_dataset_repo,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 8),
    )
    ent_hf_repo.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=(0, 4))
    ent_hf_repo.bind("<KeyRelease>", lambda _e: app.refresh_cloud_sync_summary())
    HoverTooltip(
        ent_hf_repo,
        "填写 Hugging Face Dataset 仓库 ID（如 owner/my-dataset）或完整网页 URL。\n"
        "仓库内放图片+同名 .txt（也可直接放多子集文件夹 + subsets_cloud.json），云端开机直拉免本地上传！",
    )

    tk.Label(app.cloud_hf_frame, text="仓内子目录:", bg=c["card_bg"], fg=c["text_main"], font=(c["font_family"], 8)).grid(row=0, column=2, sticky="w", padx=(0, 5), pady=(0, 4))
    ent_hf_sub = tk.Entry(
        app.cloud_hf_frame,
        textvariable=app.var_hf_dataset_subdir,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 8),
    )
    ent_hf_sub.grid(row=0, column=3, sticky="ew", pady=(0, 4))
    HoverTooltip(ent_hf_sub, "可选：如果一个 HF 仓库下分了好几个子文件夹，填写其中一个（如 chars/makima）；整个仓库都是图集则留空。")

    tk.Label(app.cloud_hf_frame, text="HF 只读 Token:", bg=c["card_bg"], fg=c["text_main"], font=(c["font_family"], 8)).grid(row=1, column=0, sticky="w", padx=(0, 5))
    ent_hf_tok = tk.Entry(
        app.cloud_hf_frame,
        textvariable=app.var_hf_token,
        show="*",
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 8),
    )
    ent_hf_tok.grid(row=1, column=1, columnspan=3, sticky="ew")
    HoverTooltip(
        ent_hf_tok,
        "公开仓库留空即可；私有 HF 数据集请填只读 Access Token（绝不写入 local_config.json，仅注入私有 Kaggle 脚本运行期使用；也可改在 Kaggle Secrets 配置 HF_TOKEN）。",
    )

    # 共享云端任务 Slug 行（锚点位于动态区下方）
    app.cloud_kernel_slug_anchor = tk.Frame(cfg_card, bg=c["bg_dark"])
    app.cloud_kernel_slug_anchor.pack(fill=tk.X)
    app.cloud_kernel_slug_anchor.columnconfigure(1, weight=1)

    tk.Label(app.cloud_kernel_slug_anchor, text="云端任务 Slug:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8), width=13, anchor="w").grid(row=0, column=0, sticky="w")
    ent_kn = tk.Entry(
        app.cloud_kernel_slug_anchor,
        textvariable=app.var_kaggle_kernel_slug,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
    )
    ent_kn.grid(row=0, column=1, sticky="ew", padx=(0, 6))
    HoverTooltip(ent_kn, "Kaggle 脚本任务唯一英文标识（6~50 位小写字母/数字/连字符）；同名再推会覆盖重跑，改新名则另起新任务。")

    FlatButton(
        app.cloud_kernel_slug_anchor,
        text="🔄 自动生成 Slug",
        bg=c["input_bg"],
        fg=c["accent_blue"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=3,
        command=app.auto_generate_cloud_slugs,
    ).grid(row=0, column=2)

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

    app.btn_cloud_master = FlatButton(
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
    app.btn_cloud_master.pack(fill=tk.X, pady=(0, 6))
    app.cloud_action_buttons.append(app.btn_cloud_master)

    step_row = tk.Frame(pipe_card, bg=c["bg_dark"])
    step_row.pack(fill=tk.X, pady=(0, 5))

    app.btn_cloud_upload_ds = FlatButton(
        step_row,
        text="1️⃣ 打包并上传图集",
        bg=c["input_bg"],
        fg=c["accent_blue"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=6,
        command=app.cloud_upload_dataset_only,
    )
    app.btn_cloud_upload_ds.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
    app.cloud_action_buttons.append(app.btn_cloud_upload_ds)

    for text, bg, fg, cmd in (
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
        text="📦 导出免 API 手动上传包 (Notebook / 图集 ZIP)",
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

    # ==================== 卡片 4：云端进度监控、实时日志与停机控制 ====================
    mon_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    mon_card.pack(fill=tk.X, pady=(0, 4))

    tk.Label(
        mon_card,
        text="📡 第 4 步 · 云端进度监控、实时日志与停机控制",
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

    poll_row1 = tk.Frame(mon_card, bg=c["bg_dark"])
    poll_row1.pack(fill=tk.X, pady=(0, 5))

    btn_refresh = FlatButton(
        poll_row1,
        text="🔄 刷新云端状态",
        bg=c["input_bg"],
        fg=c["accent_blue"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=4,
        command=lambda: app.cloud_refresh_status(silent=False),
    )
    btn_refresh.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
    app.cloud_action_buttons.append(btn_refresh)

    btn_logs = FlatButton(
        poll_row1,
        text="📜 拉取云端日志",
        bg=c["input_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=4,
        command=app.cloud_fetch_logs,
    )
    btn_logs.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
    app.cloud_action_buttons.append(btn_logs)

    ToggleButton(
        poll_row1,
        variable=app.var_cloud_auto_poll,
        text_on="⏱️ 自动轮询 (30s) 已开",
        text_off="⏱️ 自动轮询 (30s)",
        on_bg=c["accent_green"],
        off_fg=c["accent_green"],
        padx=8,
        pady=4,
        command=app.toggle_cloud_auto_poll,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    poll_row2 = tk.Frame(mon_card, bg=c["bg_dark"])
    poll_row2.pack(fill=tk.X, pady=(0, 6))

    btn_stop_web = FlatButton(
        poll_row2,
        text="🛑 停监控 & 去网页点 Stop",
        bg=c["input_bg"],
        fg=c["accent_yellow"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=4,
        command=app.cloud_stop_and_open_kaggle,
    )
    btn_stop_web.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
    HoverTooltip(
        btn_stop_web,
        "因 Kaggle 官方公共 API 不开放无损 Session Cancel ID，\n"
        "点击此按钮将立刻停止本地轮询并打开云端任务页面，您在网页点一下 Stop 即可安全释放 GPU 配额。",
    )

    btn_del = FlatButton(
        poll_row2,
        text="🗑️ 强删云端任务",
        bg="#5a232e",
        fg=c["accent_red"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=4,
        command=app.cloud_delete_kernel_ui,
    )
    btn_del.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
    app.cloud_action_buttons.append(btn_del)
    HoverTooltip(
        btn_del,
        "⚠️ 高危强杀（带二次确认）：调用 kaggle kernels delete 强制彻底销毁该任务及远端全部尚未拉回的产出与日志！",
    )

    FlatButton(
        poll_row2,
        text="🌐 打开任务页",
        bg=c["input_bg"],
        fg=c["accent_peach"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=4,
        command=app.open_kaggle_kernel_page,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

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

    app.on_cloud_dataset_source_changed()
    return scroll
