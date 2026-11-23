"""Tab 5 (📖 指南与工具): built-in offline doc viewer buttons, env diagnostics, and quick folder shortcuts."""

import tkinter as tk
import webbrowser

from core.config_store import PROJECT_ROOT, open_in_explorer
from .theme import PALETTE
from .widgets import FlatButton, ScrollableFrame


def build_help_tab(parent, app) -> ScrollableFrame:
    """构建 Tab 5【📖 指南与工具】内置说明书阅读与环境诊断面板。"""
    c = PALETTE
    scroll = ScrollableFrame(parent, bg=c["card_bg"])
    scroll.pack(fill=tk.BOTH, expand=True)
    body = scroll.content

    # ==================== 1. 内置离线实战指南阅读器 ====================
    docs_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=12, pady=10)
    docs_card.pack(fill=tk.X, pady=(0, 8))

    tk.Label(
        docs_card,
        text="📚 内置离线实战指南与参数百科 (弹窗直接阅读)",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 6))

    doc_items = [
        ("🟢 傻瓜式入门与参数实战说明书 (docs/beginner_guide.md)", c["accent_green"], "新手傻瓜式操作与参数手册", "docs/beginner_guide.md"),
        ("📐 54 项参数全景字典对照表 (docs/parameter_configuration_reference.md)", c["accent_blue"], "全量参数权威配置字典", "docs/parameter_configuration_reference.md"),
        ("🎯 超参数科学调优与防过拟合手册 (docs/hyperparameter_tuning_guide.md)", c["accent_purple"], "超参数调优、LoRA+ / DoRA / LoKr / Muon 实战", "docs/hyperparameter_tuning_guide.md"),
        ("🎨 ComfyUI 推理挂载与生图对齐指南 (docs/comfyui_inference_guide.md)", c["accent_yellow"], "ComfyUI 原生加载与推理排查", "docs/comfyui_inference_guide.md"),
        ("🚑 显存爆掉 / NaN / 报错排障矩阵 (docs/troubleshooting_matrix.md)", c["accent_peach"], "常见故障排查对照表", "docs/troubleshooting_matrix.md"),
    ]

    for label, fg, title, rel_path in doc_items:
        FlatButton(
            docs_card,
            text=label,
            bg=c["input_bg"],
            fg=fg,
            font=(c["font_family"], 8, "bold"),
            anchor="w",
            padx=10,
            pady=6,
            command=lambda t=title, p=rel_path: app.open_doc_viewer(t, p),
        ).pack(fill=tk.X, pady=(0, 4))

    # ==================== 2. 本机环境体检与目录快捷直达 ====================
    tools_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=12, pady=10)
    tools_card.pack(fill=tk.X, pady=(0, 6))

    tk.Label(
        tools_card,
        text="🛠️ 本机环境快速自检与常用目录直达 (支持 Ctrl+1~5 快捷切页)",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 6))

    FlatButton(
        tools_card,
        text="🩺 一键运行本机环境全面自检 (Python / CUDA / PyTorch / 底模 / 引擎状态)",
        bg=c["accent_green"],
        fg="#11111b",
        font=(c["font_family"], 8, "bold"),
        pady=7,
        command=app.run_quick_env_diagnostics,
    ).pack(fill=tk.X, pady=(0, 8))

    folder_rows = [
        (
            ("📂 打开输出成果目录 (output/)", c["accent_blue"], lambda: open_in_explorer(app.var_output_dir.get() or str(PROJECT_ROOT / "output"))),
            ("📂 打开临时工作区 (workspace/)", c["accent_yellow"], lambda: open_in_explorer(str(PROJECT_ROOT / "workspace"))),
        ),
        (
            ("📂 打开引擎用户配置 (configs/user/)", c["accent_purple"], lambda: open_in_explorer(str(PROJECT_ROOT / "KohakuMossTrainer/configs/user"))),
            ("📂 打开项目根目录", c["text_main"], lambda: open_in_explorer(str(PROJECT_ROOT))),
        ),
        (
            ("🌐 访问本项目 GitHub 仓库", c["accent_peach"], lambda: webbrowser.open("https://github.com/abysschaos/kaggle-moss-lora-pipeline")),
            ("📝 查看 CHANGELOG 版本历史", c["accent_green"], lambda: app.open_doc_viewer("版本更新日志 (CHANGELOG.md)", "CHANGELOG.md")),
        ),
    ]

    for left_btn, right_btn in folder_rows:
        r = tk.Frame(tools_card, bg=c["bg_dark"])
        r.pack(fill=tk.X, pady=(0, 5))
        for text, fg, cmd in (left_btn, right_btn):
            FlatButton(
                r,
                text=text,
                bg=c["input_bg"],
                fg=fg,
                font=(c["font_family"], 8, "bold"),
                pady=5,
                command=cmd,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

    FlatButton(
        tools_card,
        text="🧹 一键清理临时工作区与空目录 (释放过期中转空间并保护底模硬链接)",
        bg=c["input_bg"],
        fg=c["accent_yellow"],
        font=(c["font_family"], 8, "bold"),
        pady=6,
        command=app.clean_workspace_caches_ui,
    ).pack(fill=tk.X, pady=(2, 0))

    return scroll
