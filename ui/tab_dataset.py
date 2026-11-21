"""Tab 2 (📂 数据与续炼): dataset/subset management, output paths, health audit, and resume/init_from."""

import tkinter as tk
from tkinter import ttk

from core.config_store import open_in_explorer
from .theme import PALETTE
from .widgets import ScrollableFrame, create_dir_picker_block


def build_dataset_tab(parent, app) -> ScrollableFrame:
    """构建 Tab 2【📂 数据与续炼】面板并返回其 ScrollableFrame。"""
    c = PALETTE
    scroll = ScrollableFrame(parent, bg=c["card_bg"])
    scroll.pack(fill=tk.BOTH, expand=True)
    body = scroll.content

    # ==================== 1. 训练数据与输出路径卡 ====================
    path_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    path_card.pack(fill=tk.X, pady=(0, 8))

    path_hdr = tk.Frame(path_card, bg=c["bg_dark"])
    path_hdr.pack(fill=tk.X, pady=(0, 6))

    tk.Label(
        path_hdr,
        text="📂 训练图集、多子集配比与输出归档路径",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
    ).pack(side=tk.LEFT)

    tk.Checkbutton(
        path_hdr,
        text="多子集配比模式",
        variable=app.var_multi_subset,
        bg=c["bg_dark"],
        fg=c["accent_purple"],
        selectcolor=c["input_bg"],
        activebackground=c["bg_dark"],
        font=(c["font_family"], 8, "bold"),
        command=app.toggle_multi_subset_mode,
    ).pack(side=tk.RIGHT, padx=(6, 0))

    tk.Button(
        path_hdr,
        text="🩺 一键体检图集",
        bg=c["input_bg"],
        fg=c["accent_peach"],
        relief=tk.FLAT,
        font=(c["font_family"], 8, "bold"),
        cursor="hand2",
        padx=8,
        pady=2,
        command=app.run_dataset_health_audit,
    ).pack(side=tk.RIGHT)

    # 1A. 单目录图集面板
    app.single_data_frame = tk.Frame(path_card, bg=c["card_bg"], padx=8, pady=6)
    create_dir_picker_block(
        app.single_data_frame,
        "1. 训练图集文件夹 (含图片与同名 .txt 自然语言标注):",
        app.var_data_dir,
        app.browse_data_dir,
        open_in_explorer,
        stat_var=app.var_dataset_stat,
    )

    # 1B. 多子集混合配比面板 (仅勾选多子集配比模式时展示)
    app.multi_data_frame = tk.Frame(path_card, bg=c["card_bg"], padx=8, pady=6)
    sub_note = tk.Label(
        app.multi_data_frame,
        text="多子集混训说明：可同时混合多角色、多画风或正则化图集目录；每个子集支持独立 Repeats 与自动前置触发词。",
        bg=c["card_bg"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=540,
    )
    sub_note.pack(fill=tk.X, pady=(0, 5))
    scroll.register_wrap_label(sub_note)

    tree_wrap = tk.Frame(app.multi_data_frame, bg=c["input_bg"])
    tree_wrap.pack(fill=tk.X, pady=(0, 6))

    cols = ("enabled", "name", "images", "repeats", "trigger", "path")
    app.subset_tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=5, style="Dark.Treeview")
    for col_id, heading, width in (
        ("enabled", "启用", 42),
        ("name", "子集名称", 95),
        ("images", "图数", 46),
        ("repeats", "重复", 46),
        ("trigger", "专属触发词", 130),
        ("path", "本地文件夹路径", 190),
    ):
        app.subset_tree.heading(col_id, text=heading)
        app.subset_tree.column(col_id, width=width, anchor="center" if col_id in ("enabled", "images", "repeats") else "w")
    app.subset_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    app.subset_tree.bind("<Double-1>", lambda _e: app.edit_selected_subset())

    tree_sb = ttk.Scrollbar(tree_wrap, orient="vertical", command=app.subset_tree.yview)
    app.subset_tree.configure(yscrollcommand=tree_sb.set)
    tree_sb.pack(side=tk.RIGHT, fill=tk.Y)

    sub_btn_bar = tk.Frame(app.multi_data_frame, bg=c["card_bg"])
    sub_btn_bar.pack(fill=tk.X)
    for text, fg, cmd in (
        ("➕ 添加子集", c["accent_green"], app.add_subset_dialog),
        ("✏️ 编辑选中", c["accent_blue"], app.edit_selected_subset),
        ("🗑️ 删除选中", c["accent_red"], app.remove_selected_subset),
        ("⚖️ 一键配平 Repeat", c["accent_yellow"], app.auto_balance_subset_repeats),
    ):
        tk.Button(
            sub_btn_bar,
            text=text,
            bg=c["input_bg"],
            fg=fg,
            relief=tk.FLAT,
            font=(c["font_family"], 8, "bold"),
            cursor="hand2",
            padx=8,
            pady=3,
            command=cmd,
        ).pack(side=tk.LEFT, padx=(0, 6))

    # 1C. 固定下方的 ComfyUI 与 Output 目录锚点
    app.output_dirs_anchor = tk.Frame(path_card, bg=c["card_bg"], padx=8, pady=6)
    app.output_dirs_anchor.pack(fill=tk.X, pady=(6, 0))

    create_dir_picker_block(
        app.output_dirs_anchor,
        "2. ComfyUI 根目录或 models 文件夹 (用于扫描底模与同步 loras):",
        app.var_comfy_dir,
        app.browse_comfy_dir,
        open_in_explorer,
    )
    create_dir_picker_block(
        app.output_dirs_anchor,
        "3. LoRA 成果归档根目录 (实际产出保存在 <此目录>/<LoRA名称>/):",
        app.var_output_dir,
        app.browse_output_dir,
        open_in_explorer,
        open_target_cmd=app.open_lora_output_folder,
    )

    # 根据当前模式挂载单图集或多子集面板
    app._apply_subset_panel_visibility()

    # ==================== 2. 断点续炼、再训练与权重继承卡 ====================
    resume_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    resume_card.pack(fill=tk.X, pady=(0, 6))

    res_hdr = tk.Frame(resume_card, bg=c["bg_dark"])
    res_hdr.pack(fill=tk.X, pady=(0, 5))
    tk.Label(
        res_hdr,
        text="🔁 断点续炼、再训练与权重继承 (Resume / Init-From)",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
    ).pack(side=tk.LEFT)
    tk.Button(
        res_hdr,
        text="🔍 检测最新存档与规格对齐",
        bg=c["input_bg"],
        fg=c["accent_green"],
        relief=tk.FLAT,
        font=(c["font_family"], 8, "bold"),
        cursor="hand2",
        padx=8,
        pady=2,
        command=app.refresh_resume_status,
    ).pack(side=tk.RIGHT)

    mode_row = tk.Frame(resume_card, bg=c["bg_dark"])
    mode_row.pack(fill=tk.X, pady=(0, 4))
    for val, text, fg in (
        ("fresh", "🆕 全新从头训练", c["text_main"]),
        ("resume", "⏩ 断点续炼 (延续步数/优化器)", c["accent_green"]),
        ("init_from", "🧬 继承权重再微调 (Step 归零)", c["accent_purple"]),
    ):
        tk.Radiobutton(
            mode_row,
            text=text,
            variable=app.var_train_mode,
            value=val,
            bg=c["bg_dark"],
            fg=fg,
            selectcolor=c["input_bg"],
            activebackground=c["bg_dark"],
            font=(c["font_family"], 8, "bold"),
            command=app.refresh_resume_status,
        ).pack(side=tk.LEFT, padx=(0, 10))

    res_stat_lbl = tk.Label(
        resume_card,
        textvariable=app.var_resume_status,
        bg=c["card_bg"],
        fg=c["accent_yellow"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        padx=8,
        pady=5,
        wraplength=560,
    )
    res_stat_lbl.pack(fill=tk.X, pady=(0, 6))
    scroll.register_wrap_label(res_stat_lbl)

    res_pick = tk.Frame(resume_card, bg=c["bg_dark"])
    res_pick.pack(fill=tk.X)
    tk.Label(
        res_pick,
        text="手动来源 (可选，留空按上方状态自动匹配):",
        bg=c["bg_dark"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
    ).pack(anchor="w", pady=(0, 2))

    res_entry_row = tk.Frame(res_pick, bg=c["bg_dark"])
    res_entry_row.pack(fill=tk.X)
    tk.Entry(
        res_entry_row,
        textvariable=app.var_resume_path,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.SOLID,
        bd=1,
        font=("Consolas", 8),
    ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

    for text, fg, cmd in (
        ("📁 选模型.safetensors", c["accent_blue"], app.browse_existing_lora_file),
        ("📂 选状态目录", c["accent_green"], app.browse_resume_ckpt_dir),
        ("✖ 清空", c["accent_red"], lambda: (app.var_resume_path.set(""), app.refresh_resume_status())),
    ):
        tk.Button(
            res_entry_row,
            text=text,
            bg=c["input_bg"],
            fg=fg,
            relief=tk.FLAT,
            font=(c["font_family"], 8),
            cursor="hand2",
            padx=6,
            pady=2,
            command=cmd,
        ).pack(side=tk.LEFT, padx=(2, 0))

    return scroll
