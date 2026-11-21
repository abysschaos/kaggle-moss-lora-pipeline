"""Catppuccin Mocha dark color palette and ttk styling (Combobox, Treeview, Notebook)."""

from tkinter import ttk

PALETTE = {
    "bg_dark": "#181825",
    "card_bg": "#1e1e2e",
    "input_bg": "#313244",
    "accent_blue": "#89b4fa",
    "accent_green": "#a6e3a1",
    "accent_red": "#f38ba8",
    "accent_yellow": "#f9e2af",
    "accent_purple": "#cba6f7",
    "accent_peach": "#fab387",
    "text_main": "#cdd6f4",
    "text_sub": "#a6adc8",
    "font_family": "Microsoft YaHei UI",
}


def apply_dark_theme(root) -> ttk.Style:
    """配置 Catppuccin Mocha 全局 ttk 暗色样式（含 Combobox/Treeview/多标签 Notebook）。"""
    c = PALETTE
    root.configure(bg=c["bg_dark"])

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=c["bg_dark"], foreground=c["text_main"], font=(c["font_family"], 9))
    style.configure("Card.TFrame", background=c["card_bg"])
    style.configure("TLabel", background=c["card_bg"], foreground=c["text_main"], font=(c["font_family"], 9))
    style.configure("Sub.TLabel", background=c["card_bg"], foreground=c["text_sub"], font=(c["font_family"], 8))
    style.configure(
        "Header.TLabel",
        background=c["card_bg"],
        foreground=c["accent_blue"],
        font=(c["font_family"], 10, "bold"),
    )
    style.configure("TCheckbutton", background=c["card_bg"], foreground=c["text_main"], font=(c["font_family"], 9))

    # 解决 ttk.Combobox 在 Windows clam 主题下的 readonly 对比度异常
    style.configure(
        "TCombobox",
        fieldbackground=c["input_bg"],
        background="#45475a",
        foreground=c["text_main"],
        darkcolor=c["card_bg"],
        lightcolor=c["card_bg"],
        arrowcolor=c["text_main"],
        bordercolor=c["card_bg"],
        padding=3,
        font=(c["font_family"], 9),
    )
    style.map(
        "TCombobox",
        fieldbackground=[
            ("readonly", c["input_bg"]),
            ("disabled", c["card_bg"]),
            ("focus", c["input_bg"]),
            ("active", c["input_bg"]),
        ],
        foreground=[
            ("readonly", c["text_main"]),
            ("disabled", c["text_sub"]),
            ("focus", c["text_main"]),
            ("active", c["text_main"]),
        ],
        selectbackground=[("readonly", c["input_bg"]), ("focus", c["accent_blue"])],
        selectforeground=[("readonly", c["text_main"]), ("focus", "#11111b")],
        background=[("readonly", "#45475a"), ("active", c["accent_blue"])],
        arrowcolor=[("readonly", c["text_main"]), ("active", "#11111b")],
    )

    root.option_add("*TCombobox*Listbox*Background", c["input_bg"])
    root.option_add("*TCombobox*Listbox*Foreground", c["text_main"])
    root.option_add("*TCombobox*Listbox*selectBackground", c["accent_blue"])
    root.option_add("*TCombobox*Listbox*selectForeground", "#11111b")
    root.option_add("*TCombobox*Listbox*font", (c["font_family"], 9))

    # 多子集暗色 Treeview 表格样式
    style.configure(
        "Dark.Treeview",
        background=c["input_bg"],
        fieldbackground=c["input_bg"],
        foreground=c["text_main"],
        bordercolor="#45475a",
        rowheight=24,
        font=(c["font_family"], 8),
    )
    style.map(
        "Dark.Treeview",
        background=[("selected", c["accent_blue"])],
        foreground=[("selected", "#11111b")],
    )
    style.configure(
        "Dark.Treeview.Heading",
        background=c["bg_dark"],
        foreground=c["accent_blue"],
        relief="flat",
        font=(c["font_family"], 8, "bold"),
    )
    style.map("Dark.Treeview.Heading", background=[("active", c["input_bg"])])

    # 暗色多标签页 TNotebook 与无虚线焦点 TNotebook.Tab
    style.configure(
        "Dark.TNotebook",
        background=c["bg_dark"],
        borderwidth=0,
        tabmargins=[2, 4, 2, 0],
        darkcolor=c["bg_dark"],
        lightcolor=c["bg_dark"],
    )
    style.configure(
        "Dark.TNotebook.Tab",
        background=c["bg_dark"],
        foreground=c["text_sub"],
        padding=(10, 6),
        borderwidth=0,
        focuscolor="",
        font=(c["font_family"], 9),
    )
    style.map(
        "Dark.TNotebook.Tab",
        background=[("selected", c["card_bg"]), ("active", "#252538")],
        foreground=[("selected", c["accent_blue"]), ("active", c["text_main"])],
        font=[("selected", (c["font_family"], 9, "bold"))],
        focuscolor=[("focus", c["card_bg"])],
    )
    style.configure("TabContent.TFrame", background=c["card_bg"])
    return style
