"""Tab 4 (☁️ 云端遥控): v1.6.0 cloud remote-control roadmap and shortcuts to existing cloud assets."""

import tkinter as tk
import webbrowser

from core.config_store import PROJECT_ROOT, open_in_explorer
from .theme import PALETTE
from .widgets import ScrollableFrame


def build_cloud_tab(parent, app) -> ScrollableFrame:
    """构建 Tab 4【☁️ 云端遥控】预览与现有云端资产快捷入口面板。"""
    c = PALETTE
    scroll = ScrollableFrame(parent, bg=c["card_bg"])
    scroll.pack(fill=tk.BOTH, expand=True)
    body = scroll.content

    # ==================== 1. v1.6.0 云端遥控演进路线图卡 ====================
    roadmap_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=12, pady=10)
    roadmap_card.pack(fill=tk.X, pady=(0, 8))

    tk.Label(
        roadmap_card,
        text="☁️ 云端算力无头遥控中心 (v1.6.0 规划中 · 当前保留只读入口)",
        bg=c["bg_dark"],
        fg=c["accent_purple"],
        font=(c["font_family"], 10, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 6))

    intro = tk.Label(
        roadmap_card,
        text=(
            "本选项卡专为即将到来的 v1.6.0【本地 UI 直控云端 GPU 实例】架构预留。"
            "完成当前 v1.5.1 模块化架构升级后，你将无需手敲终端或忍受浏览器卡顿，直接在本界面遥控远程训练。"
        ),
        bg=c["bg_dark"],
        fg=c["text_main"],
        font=(c["font_family"], 8),
        anchor="w",
        justify=tk.LEFT,
        wraplength=560,
    )
    intro.pack(fill=tk.X, pady=(0, 8))
    scroll.register_wrap_label(intro)

    for title, desc, badge_fg in (
        (
            "1. Kaggle API 零浏览器一键推送与成果回传",
            "自动基于 Tab 1/2 图形化参数生成训练包，经由 Kaggle CLI 静默推送至 T4/P100 双卡队列，实时轮询状态并下载成品权重。",
            c["accent_blue"],
        ),
        (
            "2. 通用 SSH / 腾讯云 / AutoDL / RunPod 遥控通道",
            "基于 Paramiko/OpenSSH 建立轻量隧道，支持数据集增量 rsync 上传、tmux 守护进程拉起、实时流式回传 metrics.jsonl 驱动右侧 Loss 曲线。",
            c["accent_green"],
        ),
        (
            "3. 本地与云端断点双同步",
            "在云端点击【保存并暂停】即可将中间检查点自动压缩归档或拉回本地，随时切回本机继续补步。",
            c["accent_yellow"],
        ),
    ):
        box = tk.Frame(roadmap_card, bg=c["card_bg"], padx=10, pady=7)
        box.pack(fill=tk.X, pady=(0, 5))
        tk.Label(box, text=title, bg=c["card_bg"], fg=badge_fg, font=(c["font_family"], 8, "bold"), anchor="w").pack(fill=tk.X)
        lbl = tk.Label(
            box, text=desc, bg=c["card_bg"], fg=c["text_sub"], font=(c["font_family"], 8),
            anchor="w", justify=tk.LEFT, wraplength=540,
        )
        lbl.pack(fill=tk.X, pady=(2, 0))
        scroll.register_wrap_label(lbl)

    # ==================== 2. 现有云端脚本与文档快捷操作区 ====================
    asset_card = tk.Frame(body, bg=c["bg_dark"], highlightbackground=c["input_bg"], highlightthickness=1, padx=12, pady=10)
    asset_card.pack(fill=tk.X, pady=(0, 6))

    tk.Label(
        asset_card,
        text="🧳 现有 Kaggle / Colab 云端资产快捷入口 (立即可用)",
        bg=c["bg_dark"],
        fg=c["accent_blue"],
        font=(c["font_family"], 9, "bold"),
        anchor="w",
    ).pack(fill=tk.X, pady=(0, 8))

    btn_rows = [
        (
            ("📓 打开 Kaggle Notebook 文件", c["accent_blue"], lambda: app.open_project_file("notebooks/kaggle_moss_lora_training.ipynb")),
            ("📂 打开 Notebooks 目录", c["accent_green"], lambda: open_in_explorer(str(PROJECT_ROOT / "notebooks"))),
        ),
        (
            ("⚙️ 打开 kernel-metadata.json", c["accent_yellow"], lambda: app.open_project_file("kernel-metadata.json")),
            ("🔐 查看 .env.example 凭据模板", c["text_main"], lambda: app.open_doc_viewer("环境变量凭据模板 (.env.example)", ".env.example")),
        ),
        (
            ("📖 阅读 Kaggle 部署与云端指南", c["accent_purple"], lambda: app.open_doc_viewer("Kaggle / 云端部署实操指南", "docs/kaggle_deployment_guide.md")),
            ("🌐 访问 Kaggle Code 官网", c["accent_peach"], lambda: webbrowser.open("https://www.kaggle.com/code")),
        ),
    ]

    for left_btn, right_btn in btn_rows:
        r = tk.Frame(asset_card, bg=c["bg_dark"])
        r.pack(fill=tk.X, pady=(0, 5))
        for text, fg, cmd in (left_btn, right_btn):
            tk.Button(
                r,
                text=text,
                bg=c["input_bg"],
                fg=fg,
                relief=tk.FLAT,
                font=(c["font_family"], 8, "bold"),
                cursor="hand2",
                pady=6,
                command=cmd,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

    return scroll
