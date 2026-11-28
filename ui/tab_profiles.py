"""Tab 5 (🗂️ 任务档案): per-LoRA independent task profile manager (local + cloud)."""

import tkinter as tk
from tkinter import ttk

from core.task_profiles import PROFILE_CATEGORIES
from .theme import PALETTE
from .widgets import FlatButton, HoverTooltip, ScrollableFrame, SegmentedSelector


def build_profiles_tab(parent, app) -> ScrollableFrame:
    """构建 Tab 5【🗂️ 任务档案】独立任务配置管理面板并返回其 ScrollableFrame。"""
    c = PALETTE
    scroll = ScrollableFrame(parent, bg=c["card_bg"])
    scroll.pack(fill=tk.BOTH, expand=True)
    body = scroll.content

    # ==================== 1. 顶部搜索筛选与主操作卡 ====================
    hdr_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    hdr_card.pack(fill=tk.X, pady=(0, 8))

    tk.Label(
        hdr_card,
        text="🗂️ 任务档案管理中心（每个 LoRA 任务独立保存本地 + 云端全量参数 · 切换即用）",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 5))

    intro_lbl = tk.Label(
        hdr_card,
        text="说明：每份档案以独立 JSON 保存在 task_profiles/ 中，涵盖本地显卡、多子集、18 项高阶参数及 Kaggle/HF/R2 云端设置；"
             "所有 API Key / HF Token / R2 Secret 均严禁写入档案。删除档案仅移除配置文件，绝不删除图集与 output/ 成品权重。",
        bg=c["card_bg"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        padx=8,
        pady=5,
        wraplength=560,
    )
    intro_lbl.pack(fill=tk.X, pady=(0, 6))
    scroll.register_wrap_label(intro_lbl)

    filter_row = tk.Frame(hdr_card, bg=c["bg_dark"])
    filter_row.pack(fill=tk.X, pady=(0, 6))
    filter_row.columnconfigure(1, weight=1)

    tk.Label(filter_row, text="关键字:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8)).grid(
        row=0, column=0, sticky="w", padx=(0, 6)
    )
    ent_search = tk.Entry(
        filter_row,
        textvariable=app.var_profile_search,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
    )
    ent_search.grid(row=0, column=1, sticky="ew", padx=(0, 8))
    ent_search.bind("<KeyRelease>", lambda _e: app.refresh_profile_list())
    HoverTooltip(ent_search, "按任务名、LoRA 名、触发词、标签、备注、Kaggle Slug 或 HF 仓库实时检索。")

    tk.Label(filter_row, text="分类:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8)).grid(
        row=0, column=2, sticky="w", padx=(0, 4)
    )
    cat_labels = [label for _k, label in PROFILE_CATEGORIES]
    cb_cat = ttk.Combobox(filter_row, textvariable=app.var_profile_cat_filter, values=cat_labels, state="readonly", width=13)
    cb_cat.grid(row=0, column=3, sticky="w", padx=(0, 6))
    cb_cat.bind("<<ComboboxSelected>>", lambda _e: app.refresh_profile_list())

    FlatButton(
        filter_row,
        text="🧹 重置",
        bg=c["input_bg"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
        padx=8,
        pady=3,
        command=app.reset_profile_filters,
    ).grid(row=0, column=4)

    comp_row = tk.Frame(hdr_card, bg=c["bg_dark"])
    comp_row.pack(fill=tk.X, pady=(0, 6))
    tk.Label(comp_row, text="算力筛选:", bg=c["bg_dark"], fg=c["text_main"], font=(c["font_family"], 8), width=9, anchor="w").pack(
        side=tk.LEFT
    )
    SegmentedSelector(
        comp_row,
        variable=app.var_profile_compute_filter,
        options=[
            ("all", "全部", c["accent_purple"]),
            ("local", "💻 本机", c["accent_blue"]),
            ("kaggle_t4x2", "☁️ 双卡T4", c["accent_green"]),
            ("kaggle_t4x1", "⚡ 单卡T4", c["accent_yellow"]),
        ],
        command=app.refresh_profile_list,
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    action_bar = tk.Frame(hdr_card, bg=c["bg_dark"])
    action_bar.pack(fill=tk.X)
    for text, bg, fg, cmd in (
        ("➕ 新建任务档案", c["accent_green"], "#11111b", lambda: app.open_create_profile_dialog("current")),
        ("💾 保存当前参数", c["accent_blue"], "#11111b", app.save_active_profile),
        ("📥 导入 JSON", c["input_bg"], c["accent_yellow"], app.import_profile_dialog),
        ("📤 导出模板", c["input_bg"], c["accent_peach"], app.export_selected_profile_dialog),
        ("📂 打开档案目录", c["input_bg"], c["text_main"], app.open_profiles_folder),
    ):
        FlatButton(
            action_bar,
            text=text,
            bg=bg,
            fg=fg,
            font=(c["font_family"], 8, "bold"),
            padx=7,
            pady=5,
            command=cmd,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

    # ==================== 2. 任务档案列表与管理操作卡 ====================
    list_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    list_card.pack(fill=tk.X, pady=(0, 8))

    list_hdr = tk.Frame(list_card, bg=c["bg_dark"])
    list_hdr.pack(fill=tk.X, pady=(0, 5))
    tk.Label(
        list_hdr,
        text="📋 已保存任务档案列表（双击行直接载入 · 选中行查看下方全量参数）",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
    ).pack(side=tk.LEFT)
    tk.Label(
        list_hdr,
        textvariable=app.var_profile_count_stat,
        bg=c["bg_dark"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
    ).pack(side=tk.RIGHT)

    tree_wrap = tk.Frame(list_card, bg=c["input_bg"])
    tree_wrap.pack(fill=tk.X, pady=(0, 6))

    cols = ("pinned", "name", "compute", "spec", "steps_lr", "cloud_src", "updated")
    app.profile_tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=7, style="Dark.Treeview")
    for col_id, heading, width, anchor in (
        ("pinned", "★", 28, "center"),
        ("name", "任务档案名称", 150, "w"),
        ("compute", "目标算力", 78, "center"),
        ("spec", "算法 / 规格", 105, "w"),
        ("steps_lr", "步数 / LR", 96, "w"),
        ("cloud_src", "云端源", 80, "center"),
        ("updated", "更新时间", 112, "center"),
    ):
        app.profile_tree.heading(col_id, text=heading)
        app.profile_tree.column(col_id, width=width, anchor=anchor)

    app.profile_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    app.profile_tree.bind("<<TreeviewSelect>>", lambda _e: app.on_profile_tree_select())
    app.profile_tree.bind("<Double-1>", lambda _e: app.load_selected_profile())

    tree_sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=app.profile_tree.yview)
    app.profile_tree.configure(yscrollcommand=tree_sb.set)
    tree_sb.pack(side=tk.RIGHT, fill=tk.Y)

    row_ops1 = tk.Frame(list_card, bg=c["bg_dark"])
    row_ops1.pack(fill=tk.X, pady=(0, 4))
    for text, bg, fg, cmd in (
        ("▶ 载入到控制台", c["accent_blue"], "#11111b", app.load_selected_profile),
        ("💾 用当前参数覆盖", c["input_bg"], c["accent_green"], app.save_current_to_selected_profile),
        ("📋 复制副本", c["input_bg"], c["text_main"], app.duplicate_selected_profile),
        ("🔁 派生续炼任务", c["input_bg"], c["accent_purple"], app.open_continue_profile_dialog),
    ):
        FlatButton(
            row_ops1,
            text=text,
            bg=bg,
            fg=fg,
            font=(c["font_family"], 8, "bold"),
            padx=7,
            pady=4,
            command=cmd,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

    row_ops2 = tk.Frame(list_card, bg=c["bg_dark"])
    row_ops2.pack(fill=tk.X)
    for text, bg, fg, cmd in (
        ("✏️ 重命名 / 备注", c["input_bg"], c["accent_blue"], app.open_edit_profile_meta_dialog),
        ("⭐ 置顶 / 取消", c["input_bg"], c["accent_yellow"], app.toggle_pin_selected_profile),
        ("🔍 对比当前差异", c["input_bg"], c["accent_peach"], app.show_selected_profile_diff),
        ("🗑️ 删除档案", "#4a222b", c["accent_red"], app.delete_selected_profile),
    ):
        FlatButton(
            row_ops2,
            text=text,
            bg=bg,
            fg=fg,
            font=(c["font_family"], 8, "bold"),
            padx=7,
            pady=4,
            command=cmd,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

    # ==================== 3. 选中档案双轨参数透视详情卡 ====================
    detail_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    detail_card.pack(fill=tk.X, pady=(0, 4))

    detail_hdr = tk.Frame(detail_card, bg=c["bg_dark"])
    detail_hdr.pack(fill=tk.X, pady=(0, 5))
    tk.Label(
        detail_hdr,
        textvariable=app.var_profile_detail_title,
        bg=c["bg_dark"],
        fg=c["accent_purple"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    for text, fg, cmd in (
        ("🚀 载入并去训练页", c["accent_green"], lambda: app.load_selected_profile(switch_to_tab="train")),
        ("☁️ 载入并去云端页", c["accent_blue"], lambda: app.load_selected_profile(switch_to_tab="cloud")),
        ("🧾 查看 JSON", c["accent_yellow"], app.show_selected_profile_json),
    ):
        FlatButton(
            detail_hdr,
            text=text,
            bg=c["input_bg"],
            fg=fg,
            font=(c["font_family"], 8, "bold"),
            padx=7,
            pady=3,
            command=cmd,
        ).pack(side=tk.RIGHT, padx=(4, 0))

    detail_wrap = tk.Frame(detail_card, bg=c["card_bg"])
    detail_wrap.pack(fill=tk.X)
    detail_sb = ttk.Scrollbar(detail_wrap, orient="vertical")
    app.profile_detail_text = tk.Text(
        detail_wrap,
        height=13,
        bg=c["card_bg"],
        fg=c["text_main"],
        font=("Consolas", 8),
        wrap=tk.WORD,
        yscrollcommand=detail_sb.set,
        relief=tk.FLAT,
        padx=8,
        pady=6,
    )
    detail_sb.config(command=app.profile_detail_text.yview)
    app.profile_detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    detail_sb.pack(side=tk.RIGHT, fill=tk.Y)
    app.profile_detail_text.insert("1.0", "请点击上方列表中的任意任务档案，或点击【➕ 新建任务档案】创建您的第一个专属 LoRA 任务配置模板。")
    app.profile_detail_text.config(state=tk.DISABLED)

    return scroll
