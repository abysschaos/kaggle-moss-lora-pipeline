"""Tab 1 (🚀 快速训练): zero-flicker presets, toggle pills, hover tooltips, core hyperparameters & drawer."""

import tkinter as tk
from tkinter import ttk

from core.presets import PRESETS
from .theme import PALETTE
from .widgets import FlatButton, HoverTooltip, ScrollableFrame, SegmentedSelector, ToggleButton


def _add_labeled_entry(parent, row: int, col_base: int, label_text: str, var: tk.Variable, pady: int = 3, tip: str = None):
    """在网格中创建统一风格的标签 + 输入框组合，支持可选悬浮提示。"""
    c = PALETTE
    lbl = ttk.Label(parent, text=label_text)
    lbl.grid(row=row, column=col_base, sticky="w", padx=4, pady=pady)
    entry = tk.Entry(
        parent,
        textvariable=var,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 9),
        width=18,
    )
    entry.grid(row=row, column=col_base + 1, sticky="ew", padx=(4, 10), pady=pady)
    if tip:
        HoverTooltip(lbl, tip)
        HoverTooltip(entry, tip)
    return entry


def _add_labeled_combo(
    parent,
    row: int,
    col_base: int,
    label_text: str,
    var: tk.Variable,
    values: list,
    state: str = "readonly",
    pady: int = 3,
    tip: str = None,
):
    """在网格中创建统一风格的标签 + 下拉框组合，支持可选悬浮提示。"""
    lbl = ttk.Label(parent, text=label_text)
    lbl.grid(row=row, column=col_base, sticky="w", padx=4, pady=pady)
    cb = ttk.Combobox(parent, textvariable=var, values=values, state=state, width=16)
    cb.grid(row=row, column=col_base + 1, sticky="ew", padx=(4, 10), pady=pady)
    if tip:
        HoverTooltip(lbl, tip)
        HoverTooltip(cb, tip)
    return cb


def build_train_tab(parent, app) -> ScrollableFrame:
    """构建 Tab 1【🚀 快速训练】面板并返回其 ScrollableFrame。"""
    c = PALETTE
    scroll = ScrollableFrame(parent, bg=c["card_bg"])
    scroll.pack(fill=tk.BOTH, expand=True)
    body = scroll.content

    app.preset_buttons = {}
    app._preset_base_styles = {}

    # ==================== 1. 黄金场景模板预设区 ====================
    preset_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    preset_card.pack(fill=tk.X, pady=(0, 8))

    tk.Label(
        preset_card,
        text="💡 一键载入黄金调参场景模板 (8 套预设即点即用 · 悬浮参数名可看中文说明)",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 6))

    target_row = tk.Frame(preset_card, bg=c["bg_dark"])
    target_row.pack(fill=tk.X, pady=(0, 6))
    lbl_target = tk.Label(
        target_row,
        text="目标算力:",
        bg=c["bg_dark"],
        fg=c["text_main"],
        font=(c["font_family"], 8, "bold"),
        width=9,
        anchor="w",
    )
    lbl_target.pack(side=tk.LEFT)
    seg_target = SegmentedSelector(
        target_row,
        variable=app.var_target_compute,
        options=[
            ("local", "💻 本机显卡", c["accent_blue"]),
            ("kaggle_t4x2", "☁️ Kaggle 双卡 T4", c["accent_green"]),
            ("kaggle_t4x1", "⚡ Kaggle 单卡 T4", c["accent_yellow"]),
        ],
        command=app.on_target_compute_changed,
    )
    seg_target.pack(side=tk.LEFT, fill=tk.X, expand=True)
    target_tip = (
        "切换下方 8 套预设及公式算步的目标硬件环境：\n"
        "• 本机显卡：按本机探测 VRAM 与单卡打包能力核算；\n"
        "• Kaggle 双卡 T4：按 DDP 并发（MAX_TOKENS=9216×2卡, 1024px=4图/步）自动校准 GradAcc/LR/Steps；\n"
        "• Kaggle 单卡 T4：按单卡 16GB（MAX_TOKENS=8192）核算步数。"
    )
    HoverTooltip(lbl_target, target_tip)
    HoverTooltip(seg_target, target_tip)

    row1 = tk.Frame(preset_card, bg=c["bg_dark"])
    row1.pack(fill=tk.X, pady=(0, 4))
    row2 = tk.Frame(preset_card, bg=c["bg_dark"])
    row2.pack(fill=tk.X)

    for preset in PRESETS:
        target_row = row1 if preset["row"] == 1 else row2
        fg_color = c.get(preset["fg_key"], c["text_main"])
        active_bg = c.get(preset["active_bg_key"], c["accent_blue"])
        btn = FlatButton(
            target_row,
            text=preset["label"],
            bg=preset["bg"],
            fg=fg_color,
            hover_bg=c["input_bg"] if preset["bg"] != "#45475a" else "#52556b",
            pressed_bg=active_bg,
            font=(c["font_family"], 8, "bold"),
            pady=4,
            command=lambda pid=preset["id"]: app.apply_preset(pid),
        )
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        app.preset_buttons[preset["id"]] = btn
        app._preset_base_styles[preset["id"]] = (preset["bg"], fg_color, active_bg)

    preset_note = tk.Label(
        preset_card,
        text="提示：多子集配比预设会自动开启【多子集配比模式】，请到【📂 数据与续炼】选项卡添加各目录并一键配平 Repeat。",
        bg=c["bg_dark"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
    )
    preset_note.pack(fill=tk.X, pady=(5, 0))
    scroll.register_wrap_label(preset_note)

    # ==================== 2. 任务命名、触发词与状态摘要卡 ====================
    name_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    name_card.pack(fill=tk.X, pady=(0, 8))

    tk.Label(
        name_card,
        text="🏷️ 任务命名与触发词",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 4))

    name_grid = tk.Frame(name_card, bg=c["bg_dark"])
    name_grid.pack(fill=tk.X)
    name_grid.columnconfigure(1, weight=1)
    name_grid.columnconfigure(3, weight=1)

    _add_labeled_entry(
        name_grid, 0, 0, "LoRA 权重名称:", app.var_lora_name,
        tip="输出文件及归档文件夹名称（仅限英文/数字/下划线，避免空格与中文）。",
    )
    _add_labeled_entry(
        name_grid, 0, 2, "触发词 (可选注入):", app.var_trigger,
        tip="若 .txt 中未含此词，预处理时自动前置到句首；若已包含则自动跳过防重复。",
    )

    preview_lbl = tk.Label(
        name_card,
        textvariable=app.var_output_preview,
        bg=c["bg_dark"],
        fg=c["accent_green"],
        font=("Consolas", 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
    )
    preview_lbl.pack(fill=tk.X, pady=(4, 4))
    scroll.register_wrap_label(preview_lbl)

    status_strip = tk.Frame(name_card, bg=c["card_bg"], padx=8, pady=5)
    status_strip.pack(fill=tk.X)

    FlatButton(
        status_strip,
        text="📂 数据/续炼 ↗",
        bg=c["input_bg"],
        fg=c["accent_blue"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=3,
        command=lambda: app.select_tab(1),
    ).pack(side=tk.RIGHT, padx=(4, 0))

    FlatButton(
        status_strip,
        text="🩺 体检图集",
        bg=c["input_bg"],
        fg=c["accent_peach"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=3,
        command=app.run_dataset_health_audit,
    ).pack(side=tk.RIGHT)

    quick_stat_lbl = tk.Label(
        status_strip,
        textvariable=app.var_quick_status_summary,
        bg=c["card_bg"],
        fg=c["accent_yellow"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=420,
    )
    quick_stat_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
    scroll.register_wrap_label(quick_stat_lbl)

    # ==================== 3. 硬件与超参数配置卡 ====================
    hp_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=10, pady=8)
    hp_card.pack(fill=tk.X, pady=(0, 6))

    hp_hdr = tk.Frame(hp_card, bg=c["bg_dark"])
    hp_hdr.pack(fill=tk.X, pady=(0, 6))
    tk.Label(
        hp_hdr,
        text="⚙️ 硬件与核心超参数配置",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
    ).pack(side=tk.LEFT)

    btn_align = FlatButton(
        hp_hdr,
        text="📐 一键对齐推荐步数",
        bg="#45475a",
        fg=c["accent_blue"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=3,
        command=app.align_steps_by_formula,
    )
    btn_align.pack(side=tk.RIGHT, padx=(4, 0))
    HoverTooltip(btn_align, "按【有效总曝光 ÷ 单步等效Batch】一键把 Max Steps 设到公式推荐值。")

    btn_cli = FlatButton(
        hp_hdr,
        text="📋 预览命令",
        bg=c["input_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=3,
        command=app.preview_cli_command,
    )
    btn_cli.pack(side=tk.RIGHT)
    HoverTooltip(btn_cli, "查看当前所有界面选项对应的完整 train_local.py CLI 命令行，支持一键复制用于排查或脚本自动化。")

    app.cloud_hw_banner = tk.Label(
        hp_card,
        text="☁️ 云端算力模式：Precision(T4自动FP16)、VRAM、Base Prec 与 TextEnc 将由 Kaggle 自动托管；步数与预设已按云端实际并发吞吐校准。",
        bg=c["card_bg"],
        fg=c["accent_blue"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        padx=8,
        pady=5,
        wraplength=560,
    )
    scroll.register_wrap_label(app.cloud_hw_banner)

    hp_grid = tk.Frame(hp_card, bg=c["bg_dark"])
    app.hp_grid = hp_grid
    hp_grid.pack(fill=tk.X)
    hp_grid.columnconfigure(1, weight=1)
    hp_grid.columnconfigure(3, weight=1)

    _add_labeled_combo(
        hp_grid, 0, 0, "训练精度 (Precision):", app.var_precision, ["bf16", "fp16"],
        tip="RTX 30/40/50 系强烈推荐 bf16；老卡（如 Turing/T4）后端自动降级 fp16。",
    )
    _add_labeled_combo(
        hp_grid, 0, 2, "底座量化 (Base Prec):", app.var_base_precision,
        ["auto", "int8", "fp8", "fp8_compute", "compute"],
        tip="默认 auto 即可（int8 底模自动防溢出）；全精度底模在 RTX 40+ 可尝试 fp8_compute 提速。",
    )

    cb_vram = _add_labeled_combo(
        hp_grid, 1, 0, "显存档位 (VRAM):", app.var_vram_preset, ["auto", "6g", "8g", "12g", "16g"],
        tip="默认 auto 会自动匹配当前显卡（如 8GB 显卡自动映射至 8g / MAX_TOKENS=4352）；如仍爆显存可手动切 6g。",
    )
    cb_vram.bind("<<ComboboxSelected>>", lambda _e: app.recalculate_math())
    cb_res = _add_labeled_combo(
        hp_grid, 1, 2, "训练分辨率 (Res):", app.var_resolution, ["512", "768", "896", "1024"],
        tip="8GB 显卡跑 1024px 文本上限仅约 256 tok；若长自然语言较多或想提速，建议切 896（甜点档）！",
    )
    cb_res.bind("<<ComboboxSelected>>", lambda _e: app.recalculate_math())

    _add_labeled_combo(
        hp_grid, 2, 0, "TextEnc 运行设备:", app.var_text_device, ["cpu", "cuda"],
        tip="8GB~12GB 显卡务必选 cpu 离线提取文本，用完释放内存；>=16GB 大显存才考虑 cuda。",
    )
    _add_labeled_combo(
        hp_grid, 2, 2, "VAE 编码设备:", app.var_vae_device, ["cuda", "cpu"],
        tip="前置潜向量提取阶段，默认 cuda 几秒即完成并释放。",
    )

    cb_algo = _add_labeled_combo(
        hp_grid, 3, 0, "微调算法 (Algo):", app.var_algo, ["lora", "loha", "lokr"],
        tip="通用选 lora；高密度/复杂多风格可试 loha；追求极小产物文件且保留精细笔触可选 lokr。",
    )
    cb_algo.bind("<<ComboboxSelected>>", lambda _e: app.on_dynamic_controls_changed())
    _add_labeled_entry(
        hp_grid, 3, 2, "LoRA Rank (维度秩):", app.var_rank,
        tip="单画风/单人角色通常 16~32；多人物或多画风混合建议 48。",
    )

    _add_labeled_entry(
        hp_grid, 4, 0, "LoRA Alpha (缩放):", app.var_alpha,
        tip="一般等于 Rank（画风/标准）或 Rank/2（人物小图防死锁降强度）。",
    )
    tgl_dora = ToggleButton(
        hp_grid,
        variable=app.var_dora,
        text_on="DoRA 权重分解：已开启",
        text_off="DoRA 权重分解：已关闭",
        on_bg=c["accent_purple"],
        off_fg=c["accent_purple"],
        command=app.on_dynamic_controls_changed,
    )
    tgl_dora.grid(row=4, column=2, columnspan=2, sticky="ew", padx=(4, 10), pady=3)
    HoverTooltip(tgl_dora, "DoRA (ICML 2024) 拆解方向与幅度提升人物/纹理精度；开启时引擎会强制把 InputDropout 锁为 0.0。")

    _add_labeled_combo(
        hp_grid, 5, 0, "作用层预设 (Target):", app.var_target_preset,
        ["attn", "attn-mlp", "attn-mlp+modulation", "attn-mlp+text-refiner", "full-block", "all"],
        tip="标准首选 attn-mlp；多画风推荐 +modulation；多人物/概念绑定推荐 +text-refiner。",
    )
    _add_labeled_combo(
        hp_grid, 5, 2, "LoRA+ 速率倍增:", app.var_loraplus_ratio, ["1.0", "1.5", "2.0", "3.0", "4.0"], state="normal",
        tip="B 矩阵学习率倍率（1.0=关闭，2.0=人物与概念收敛显著更快）。",
    )

    _add_labeled_combo(
        hp_grid, 6, 0, "学习率 (LR):", app.var_lr, ["1e-4", "1.2e-4", "1.3e-4", "8e-5", "6e-5", "1.5e-4", "1.8e-4", "2e-4"], state="normal",
        tip="画风常用 1e-4（双卡 T4 推荐 1.2e-4 / LoKr 1.8e-4 / Muon 1.3e-4）；人物五官常用 6e-5~8e-5（双卡 T4 推荐 1e-4）。",
    )
    ent_rep = _add_labeled_entry(
        hp_grid, 6, 2, "每图重复 (Repeats):", app.var_repeats,
        tip="仅用于按公式换算推荐 Max Steps；多子集模式下可在【📂 数据与续炼】单独为每组配比。",
    )
    ent_rep.bind("<KeyRelease>", lambda _e: app.recalculate_math())

    ent_ep = _add_labeled_entry(
        hp_grid, 7, 0, "训练轮数 (Epochs):", app.var_epochs,
        tip="用于计算总曝光次数（实际停止条件以右栏 Max Steps 为准）。",
    )
    ent_ep.bind("<KeyRelease>", lambda _e: app.recalculate_math())
    _add_labeled_entry(
        hp_grid, 7, 2, "总步数 (Max Steps):", app.var_steps,
        tip="优化器真实更新总步数；建议修改图数或 GradAcc 后点击上方【一键对齐推荐步数】。",
    )

    cb_acc = _add_labeled_combo(
        hp_grid, 8, 0, "梯度累积 (GradAcc):", app.var_grad_acc, ["1", "2", "4", "8"], state="normal",
        tip="画风选 1 极速推进；人物五官与多子集混合强烈建议 2（零额外显存占用）。",
    )
    cb_acc.bind("<<ComboboxSelected>>", lambda _e: app.recalculate_math())
    cb_acc.bind("<KeyRelease>", lambda _e: app.recalculate_math())

    app.cb_mu = _add_labeled_combo(
        hp_grid, 8, 2, "时间步重心 (Mu):", app.var_timestep_mu, ["-1.09", "-0.60", "0.0", "0.35", "0.50"], state="normal",
        tip="官方 moss 模式下：-1.09 偏高噪画风；0.0 对称；0.35~0.50 专攻低噪面部五官与服饰细节（注：logit_normal 模式正负号相反）。",
    )
    app.cb_mu.bind("<<ComboboxSelected>>", lambda _e: app.recalculate_math())
    app.cb_mu.bind("<KeyRelease>", lambda _e: app.recalculate_math())

    _add_labeled_entry(
        hp_grid, 9, 0, "每 N 步存盘:", app.var_save_every,
        tip="每隔多少步保存一次检查点并自动同步至 output/<LoRA名>/。",
    )
    _add_labeled_combo(
        hp_grid, 9, 2, "导出格式 (Format):", app.var_save_format, ["both", "comfy", "lycoris"],
        tip="推荐 both 同时保存 ComfyUI 原生键名与 LyCORIS/HF 格式。",
    )

    tgl_comfy = ToggleButton(
        hp_grid,
        variable=app.var_copy_comfy,
        text_on="完成后自动同步副本至 ComfyUI/models/loras：已开启",
        text_off="同步导出副本至 ComfyUI/models/loras：关",
        on_bg=c["accent_green"],
        off_fg=c["accent_green"],
    )
    tgl_comfy.grid(row=10, column=0, columnspan=4, sticky="ew", padx=4, pady=(5, 2))
    HoverTooltip(tgl_comfy, "开启后，每当训练存盘新步数权重时后台都会同步拷贝一份 _comfy.safetensors 到你的 ComfyUI/models/loras/ 方便随时试跑。")

    math_lbl = tk.Label(
        hp_card,
        textvariable=app.var_math_info,
        bg=c["card_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        padx=8,
        pady=5,
        wraplength=560,
    )
    math_lbl.pack(fill=tk.X, pady=(4, 4))
    scroll.register_wrap_label(math_lbl)
    app.lbl_math_info = math_lbl

    est_row = tk.Frame(hp_card, bg=c["bg_dark"])
    est_row.pack(fill=tk.X, pady=(0, 6))
    lbl_est = tk.Label(
        est_row,
        text="预估图数(HF/空目录算步):",
        bg=c["bg_dark"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
        anchor="w",
    )
    lbl_est.pack(side=tk.LEFT)
    ent_est = tk.Entry(
        est_row,
        textvariable=app.var_manual_img_count,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.FLAT,
        font=("Consolas", 8),
        width=6,
    )
    ent_est.pack(side=tk.LEFT, padx=(6, 8))
    ent_est.bind("<KeyRelease>", lambda _e: app.recalculate_math())
    tk.Label(
        est_row,
        text="留空自动统计本地目录",
        bg=c["bg_dark"],
        fg=c["text_sub"],
        font=(c["font_family"], 8),
        anchor="w",
    ).pack(side=tk.LEFT)
    est_tip = "使用【🤗 云端直拉 Hugging Face 图集】且本机未放图时，填入 HF 数据集张数即可精准核算双卡/单卡推荐步数；本地有图时留空自动数图。"
    HoverTooltip(lbl_est, est_tip)
    HoverTooltip(ent_est, est_tip)

    # ==================== 可折叠高阶算法/防过拟合/优化器抽屉 ====================
    app.btn_adv_toggle = FlatButton(
        hp_card,
        text="▸ 展开高阶算法 / 防过拟合 / 优化器调优区 (18项专业参数)",
        bg=c["input_bg"],
        fg=c["accent_purple"],
        font=(c["font_family"], 8, "bold"),
        anchor="w",
        padx=8,
        pady=6,
        command=app.toggle_advanced_drawer,
    )
    app.btn_adv_toggle.pack(fill=tk.X)

    app.adv_drawer_frame = tk.Frame(hp_card, bg=c["card_bg"], padx=8, pady=8)

    # LoKr 专属参数行 (仅 algo == "lokr" 显示)
    app.lokr_row = tk.Frame(app.adv_drawer_frame, bg=c["card_bg"])
    tk.Label(app.lokr_row, text="LoKr Factor:", bg=c["card_bg"], fg=c["accent_peach"], font=(c["font_family"], 8, "bold")).pack(side=tk.LEFT)
    ttk.Combobox(app.lokr_row, textvariable=app.var_lokr_factor, values=["-1", "4", "8", "16"], width=6).pack(side=tk.LEFT, padx=(4, 10))
    ToggleButton(
        app.lokr_row, variable=app.var_lokr_full_matrix,
        text_on="Full Matrix: 开", text_off="Full Matrix: 关",
        on_bg=c["accent_peach"], pady=2, padx=8,
    ).pack(side=tk.LEFT, padx=(0, 8))
    ToggleButton(
        app.lokr_row, variable=app.var_lokr_decompose_both,
        text_on="Decompose Both: 开", text_off="Decompose Both: 关",
        on_bg=c["accent_peach"], pady=2, padx=8,
    ).pack(side=tk.LEFT)

    # 归一化层微调 & 正则化行 (固定在 drop_row 之前)
    app.reg_row = tk.Frame(app.adv_drawer_frame, bg=c["card_bg"])
    app.reg_row.pack(fill=tk.X, pady=(0, 5))
    tgl_norms = ToggleButton(
        app.reg_row, variable=app.var_train_norms,
        text_on="Train Norms: 开", text_off="Train Norms (归一化): 关",
        on_bg=c["accent_yellow"], off_fg=c["accent_yellow"], pady=2, padx=8,
    )
    tgl_norms.pack(side=tk.LEFT, padx=(0, 10))
    HoverTooltip(tgl_norms, "协同微调 AdaLN/RMS/LayerNorm 缩放权重，适合重度画风色彩迁移；日常单人角色建议保持关闭。")

    tk.Label(app.reg_row, text="RankDrop:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    ttk.Combobox(app.reg_row, textvariable=app.var_rank_dropout, values=["0.0", "0.05", "0.1", "0.2"], width=6).pack(side=tk.LEFT, padx=(3, 10))
    tk.Label(app.reg_row, text="ModDrop:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    ttk.Combobox(app.reg_row, textvariable=app.var_module_dropout, values=["0.0", "0.05", "0.1", "0.2"], width=6).pack(side=tk.LEFT, padx=(3, 0))

    # InputDropout & TextDropout 行
    drop_row = tk.Frame(app.adv_drawer_frame, bg=c["card_bg"])
    drop_row.pack(fill=tk.X, pady=(0, 5))
    tk.Label(drop_row, text="InDrop:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    app.cb_input_dropout = ttk.Combobox(
        drop_row, textvariable=app.var_input_dropout, values=["0.0", "0.05", "0.1", "0.15", "0.2"], width=6
    )
    app.cb_input_dropout.pack(side=tk.LEFT, padx=(3, 6))
    app.lbl_dora_drop_lock = tk.Label(
        drop_row, text="", bg=c["card_bg"], fg=c["accent_yellow"], font=(c["font_family"], 8)
    )
    app.lbl_dora_drop_lock.pack(side=tk.LEFT, padx=(0, 10))
    tk.Label(drop_row, text="TextDrop (无字丢弃):", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    ttk.Combobox(drop_row, textvariable=app.var_text_drop, values=["0.0", "0.05", "0.1", "0.15", "0.2"], width=6).pack(side=tk.LEFT, padx=(3, 0))

    # 优化器与学习率调度行 1
    opt_row1 = tk.Frame(app.adv_drawer_frame, bg=c["card_bg"])
    opt_row1.pack(fill=tk.X, pady=(0, 5))
    tk.Label(opt_row1, text="优化器:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    cb_opt = ttk.Combobox(opt_row1, textvariable=app.var_optimizer, values=["adamw", "adamw8bit_kernel", "muon"], state="readonly", width=16)
    cb_opt.pack(side=tk.LEFT, padx=(3, 8))
    HoverTooltip(cb_opt, "adamw 最省显存兼容性最好；muon/8bit_kernel 需由后端切至 GPU 流式显存档位，6G 卡慎用。")
    tk.Label(opt_row1, text="LR调度:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    ttk.Combobox(opt_row1, textvariable=app.var_lr_scheduler, values=["cosine", "constant"], state="readonly", width=9).pack(side=tk.LEFT, padx=(3, 8))
    tk.Label(opt_row1, text="Warmup(-1自动):", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    tk.Entry(
        opt_row1, textvariable=app.var_warmup_steps, bg=c["input_bg"], fg=c["text_main"],
        insertbackground=c["text_main"], relief=tk.FLAT, font=("Consolas", 8), width=6,
    ).pack(side=tk.LEFT, padx=(3, 8))
    tk.Label(opt_row1, text="WD:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    tk.Entry(
        opt_row1, textvariable=app.var_weight_decay, bg=c["input_bg"], fg=c["text_main"],
        insertbackground=c["text_main"], relief=tk.FLAT, font=("Consolas", 8), width=6,
    ).pack(side=tk.LEFT, padx=(3, 0))

    # 优化器行 2 (WD规则 / EMA / 时间步模式)
    opt_row2 = tk.Frame(app.adv_drawer_frame, bg=c["card_bg"])
    opt_row2.pack(fill=tk.X, pady=(0, 5))
    tk.Label(opt_row2, text="WD规则:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    ttk.Combobox(opt_row2, textvariable=app.var_wd_rules, values=["moss", "none", "all"], state="readonly", width=7).pack(side=tk.LEFT, padx=(3, 8))
    tk.Label(opt_row2, text="EMA平滑:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    cb_ema = ttk.Combobox(opt_row2, textvariable=app.var_ema, values=["0.0", "0.95", "0.99", "0.995"], width=7)
    cb_ema.pack(side=tk.LEFT, padx=(3, 8))
    HoverTooltip(cb_ema, "500+ 步长程训练可开 0.99 平滑权重抖动；短步数 (<300) 请保持 0.0 关闭。")
    tk.Label(opt_row2, text="时间步采样:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    cb_ts = ttk.Combobox(opt_row2, textvariable=app.var_timestep, values=["moss", "uniform", "logit_normal"], state="readonly", width=12)
    cb_ts.pack(side=tk.LEFT, padx=(3, 0))
    cb_ts.bind("<<ComboboxSelected>>", lambda _e: app.on_dynamic_controls_changed())
    HoverTooltip(cb_ts, "绝大多数情况锁定 moss；uniform 会禁用 Mu；logit_normal 的 Mu 正负号与 moss 相反。")

    # 增强与种子行
    aug_row = tk.Frame(app.adv_drawer_frame, bg=c["card_bg"])
    aug_row.pack(fill=tk.X)
    tgl_flip = ToggleButton(
        aug_row, variable=app.var_flip,
        text_on="随机水平翻转 (Flip): 开", text_off="随机水平翻转 (Flip): 关",
        on_bg=c["accent_peach"], pady=2, padx=8,
    )
    tgl_flip.pack(side=tk.LEFT, padx=(0, 10))
    HoverTooltip(tgl_flip, "人物含不对称服饰、胸章、发饰或画面含文字时严禁开启！")

    tk.Label(aug_row, text="Seed:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    tk.Entry(
        aug_row, textvariable=app.var_seed, bg=c["input_bg"], fg=c["text_main"],
        insertbackground=c["text_main"], relief=tk.FLAT, font=("Consolas", 8), width=7,
    ).pack(side=tk.LEFT, padx=(3, 10))
    tk.Label(aug_row, text="Tune表:", bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8)).pack(side=tk.LEFT)
    ttk.Combobox(aug_row, textvariable=app.var_tune, values=["off", "table"], state="readonly", width=7).pack(side=tk.LEFT, padx=(3, 0))

    app.on_dynamic_controls_changed()
    return scroll
