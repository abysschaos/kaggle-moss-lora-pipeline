"""Reusable dark-themed Tkinter widgets (ScrollableFrame, file/dir pickers, text viewer)."""

import tkinter as tk
from tkinter import filedialog, ttk

from core.config_store import PROJECT_ROOT
from .theme import PALETTE


class ScrollableFrame(tk.Frame):
    """带按需自动显隐滚动条、宽屏自适应折行与安全滚轮绑定的容器面板。"""

    def __init__(self, parent, bg=None):
        bg_color = bg or PALETTE["card_bg"]
        super().__init__(parent, bg=bg_color)
        self._wrap_labels = []
        self._vscroll_visible = False
        self._scroll_check_job = None

        self.canvas = tk.Canvas(self, bg=bg_color, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = tk.Frame(self.canvas, bg=bg_color, padx=8, pady=6)
        self._canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.content.bind("<Configure>", lambda _e: self.refresh_scroll_region())

        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        self.content.bind("<Enter>", self._bind_wheel)

        self.canvas.pack(side="left", fill="both", expand=True)

    def register_wrap_label(self, widget):
        """注册需要随容器宽度变化自适应 wraplength 的文本标签或复选框。"""
        self._wrap_labels.append(widget)
        return widget

    def _schedule_scrollbar_check(self):
        """防抖调度滚动条显隐校验，避免布局尚未落定时的抖动。"""
        if self._scroll_check_job is not None:
            try:
                self.after_cancel(self._scroll_check_job)
            except Exception:
                pass
        try:
            self._scroll_check_job = self.after_idle(self._update_scrollbar_visibility)
        except Exception:
            pass

    def _update_scrollbar_visibility(self):
        """仅当内容总高度确实超出可视高度时显示垂直滚动条，否则彻底隐藏并归零视图。"""
        self._scroll_check_job = None
        try:
            canvas_h = self.canvas.winfo_height()
            content_h = self.content.winfo_reqheight()
            if canvas_h <= 1:
                return
            need_scroll = content_h > (canvas_h + 4)
            if need_scroll:
                if not self.scrollbar.winfo_ismapped():
                    self.scrollbar.pack(side="right", fill="y")
                self._vscroll_visible = True
            else:
                if self.scrollbar.winfo_ismapped():
                    self.scrollbar.pack_forget()
                self.canvas.yview_moveto(0.0)
                self._vscroll_visible = False
        except Exception:
            pass

    def refresh_scroll_region(self):
        try:
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)
        except Exception:
            pass
        self._schedule_scrollbar_check()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self._canvas_window, width=event.width)
        new_wrap = max(440, event.width - 42)
        for lbl in self._wrap_labels:
            try:
                lbl.configure(wraplength=new_wrap)
            except Exception:
                pass
        self._schedule_scrollbar_check()

    def _on_mousewheel(self, event):
        if not self._vscroll_visible or not self.canvas.winfo_ismapped():
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_wheel(self, _event):
        if self._vscroll_visible and self.canvas.winfo_ismapped():
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_wheel(self, _event):
        try:
            self.canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass


def create_file_picker_block(
    parent,
    label_text: str,
    var: tk.StringVar,
    browse_title: str,
    folder_path: str,
    on_open_folder,
    on_selected=None,
    filetypes=None,
):
    """防遮挡两行式文件选择器：第 1 行右侧优先放置按钮，第 2 行全宽路径输入框。"""
    c = PALETTE
    if filetypes is None:
        filetypes = [("Safetensors模型", "*.safetensors"), ("所有模型", "*.safetensors *.pt *.bin"), ("所有文件", "*.*")]

    block = tk.Frame(parent, bg=c["card_bg"])
    block.pack(fill=tk.X, pady=3)

    top_bar = tk.Frame(block, bg=c["card_bg"])
    top_bar.pack(fill=tk.X, pady=(0, 2))

    def _browse_file():
        selected = filedialog.askopenfilename(title=browse_title, filetypes=filetypes)
        if selected:
            var.set(selected)
            if on_selected:
                on_selected(selected)

    tk.Button(
        top_bar,
        text="📁 点此自选文件...",
        bg=c["accent_blue"],
        fg="#11111b",
        relief=tk.FLAT,
        font=(c["font_family"], 8, "bold"),
        cursor="hand2",
        padx=8,
        pady=1,
        command=_browse_file,
    ).pack(side=tk.RIGHT, padx=(4, 0))

    tk.Button(
        top_bar,
        text="📂 打开放置目录",
        bg=c["input_bg"],
        fg=c["accent_green"],
        relief=tk.FLAT,
        font=(c["font_family"], 8),
        cursor="hand2",
        padx=6,
        pady=1,
        command=lambda: on_open_folder(folder_path),
    ).pack(side=tk.RIGHT)

    tk.Label(
        top_bar,
        text=label_text,
        bg=c["card_bg"],
        fg=c["text_main"],
        font=(c["font_family"], 8, "bold"),
        anchor="w",
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    tk.Entry(
        block,
        textvariable=var,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.SOLID,
        bd=1,
        font=("Consolas", 8),
    ).pack(fill=tk.X, pady=(1, 2))


def create_dir_picker_block(
    parent,
    label_text: str,
    var: tk.StringVar,
    browse_cmd,
    on_open_folder,
    open_target_cmd=None,
    stat_var=None,
):
    """防遮挡两行式文件夹选择器，支持可选状态指示行。"""
    c = PALETTE
    block = tk.Frame(parent, bg=c["card_bg"])
    block.pack(fill=tk.X, pady=3)

    top_bar = tk.Frame(block, bg=c["card_bg"])
    top_bar.pack(fill=tk.X, pady=(0, 2))

    tk.Button(
        top_bar,
        text="📁 点此自选文件夹...",
        bg=c["accent_blue"],
        fg="#11111b",
        relief=tk.FLAT,
        font=(c["font_family"], 8, "bold"),
        cursor="hand2",
        padx=8,
        pady=1,
        command=browse_cmd,
    ).pack(side=tk.RIGHT, padx=(4, 0))

    target_cmd = open_target_cmd if open_target_cmd else (lambda: on_open_folder(var.get()))
    tk.Button(
        top_bar,
        text="📂 打开此目录",
        bg=c["input_bg"],
        fg=c["accent_green"],
        relief=tk.FLAT,
        font=(c["font_family"], 8),
        cursor="hand2",
        padx=6,
        pady=1,
        command=target_cmd,
    ).pack(side=tk.RIGHT)

    tk.Label(
        top_bar,
        text=label_text,
        bg=c["card_bg"],
        fg=c["text_main"],
        font=(c["font_family"], 8, "bold"),
        anchor="w",
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    tk.Entry(
        block,
        textvariable=var,
        bg=c["input_bg"],
        fg=c["text_main"],
        insertbackground=c["text_main"],
        relief=tk.SOLID,
        bd=1,
        font=("Consolas", 8),
    ).pack(fill=tk.X, pady=(1, 2))

    if stat_var is not None:
        tk.Label(
            block,
            textvariable=stat_var,
            bg=c["card_bg"],
            fg=c["accent_yellow"],
            font=(c["font_family"], 8),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 2))


def show_text_viewer_dialog(parent, title: str, header_text: str, header_color: str, body_text: str, size=(740, 500)):
    """弹出统一的暗色可滚动只读文本窗口（用于图集体检报告与内置指南阅读）。"""
    c = PALETTE
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.configure(bg=c["card_bg"])
    dlg.transient(parent)
    icon_path = PROJECT_ROOT / "assets" / "moss_trainer.ico"
    if icon_path.exists():
        try:
            dlg.iconbitmap(str(icon_path))
        except Exception:
            pass
    w, h = size
    rx = parent.winfo_rootx() + max(40, (parent.winfo_width() - w) // 2)
    ry = parent.winfo_rooty() + max(40, (parent.winfo_height() - h) // 2)
    dlg.geometry(f"{w}x{h}+{rx}+{ry}")

    tk.Label(
        dlg,
        text=header_text,
        bg=c["card_bg"],
        fg=header_color,
        font=(c["font_family"], 10, "bold"),
        anchor="w",
        padx=12,
        pady=8,
    ).pack(fill=tk.X)

    box_wrap = tk.Frame(dlg, bg="#11111b", padx=8, pady=4)
    box_wrap.pack(fill=tk.BOTH, expand=True, padx=12)
    sb = ttk.Scrollbar(box_wrap, orient="vertical")
    txt = tk.Text(
        box_wrap,
        bg="#11111b",
        fg=c["text_main"],
        font=("Consolas", 9),
        wrap=tk.WORD,
        yscrollcommand=sb.set,
        relief=tk.FLAT,
        padx=8,
        pady=8,
    )
    sb.config(command=txt.yview)
    txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    txt.insert("1.0", body_text)
    txt.config(state=tk.DISABLED)

    btn_bar = tk.Frame(dlg, bg=c["card_bg"], pady=8, padx=12)
    btn_bar.pack(fill=tk.X)
    tk.Button(
        btn_bar,
        text="知道了 / 关闭",
        bg=c["accent_blue"],
        fg="#11111b",
        relief=tk.FLAT,
        font=(c["font_family"], 9, "bold"),
        padx=16,
        pady=4,
        cursor="hand2",
        command=dlg.destroy,
    ).pack(side=tk.RIGHT)
    return dlg
