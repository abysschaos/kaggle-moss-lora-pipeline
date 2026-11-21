"""Reusable dark-themed zero-flicker Tkinter widgets (FlatButton, ToggleButton, SegmentedSelector, HoverTooltip, ScrollableFrame)."""

import tkinter as tk
from tkinter import filedialog, ttk

from core.config_store import PROJECT_ROOT
from .theme import PALETTE, adjust_hex_color, apply_windows_dark_titlebar


def _smart_press_bg(bg: str) -> str:
    """为高饱和浅亮按钮微降亮度、为深底按钮微提亮度，防止按压瞬间泛白闪烁。"""
    s = (bg or "").strip()
    if len(s) == 7 and s.startswith("#"):
        try:
            lum = (int(s[1:3], 16) + int(s[3:5], 16) + int(s[5:7], 16)) / 3.0
            return adjust_hex_color(s, -22 if lum > 110 else 18)
        except Exception:
            pass
    return adjust_hex_color(s, 12)


class FlatButton(tk.Button):
    """零闪烁暗色扁平按钮：锁定无边框、无焦点虚线、自定义悬停与按压稳态背景色。"""

    def __init__(self, parent, bg=None, fg=None, hover_bg=None, pressed_bg=None, **kwargs):
        c = PALETTE
        base_bg = bg or c["input_bg"]
        base_fg = fg or c["text_main"]
        self._base_bg = base_bg
        self._base_fg = base_fg
        self._hover_bg = hover_bg or adjust_hex_color(base_bg, 14)
        self._pressed_bg = pressed_bg or _smart_press_bg(base_bg)

        kwargs.setdefault("relief", tk.FLAT)
        kwargs.setdefault("overrelief", tk.FLAT)
        kwargs.setdefault("bd", 0)
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("takefocus", 0)
        kwargs.setdefault("cursor", "hand2")
        kwargs["bg"] = base_bg
        kwargs["fg"] = base_fg
        kwargs["activebackground"] = self._pressed_bg
        kwargs["activeforeground"] = base_fg
        super().__init__(parent, **kwargs)

        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")

    def set_colors(self, bg: str, fg: str = None, hover_bg: str = None, pressed_bg: str = None):
        """运行期无闪烁更新按钮颜色（用于运行态切换与当前激活预设高亮）。"""
        self._base_bg = bg
        if fg is not None:
            self._base_fg = fg
        self._hover_bg = hover_bg or adjust_hex_color(bg, 14)
        self._pressed_bg = pressed_bg or _smart_press_bg(bg)
        self.configure(
            bg=self._base_bg,
            fg=self._base_fg,
            activebackground=self._pressed_bg,
            activeforeground=self._base_fg,
        )

    def _on_enter(self, _event=None):
        if str(self.cget("state")) != "disabled":
            self.configure(bg=self._hover_bg)

    def _on_leave(self, _event=None):
        self.configure(bg=self._base_bg)


class ToggleButton(tk.Label):
    """无闪烁胶囊式双态开关按钮（替代旧版带勾方框 Checkbutton），支持 BooleanVar 双向同步。"""

    def __init__(
        self,
        parent,
        variable: tk.BooleanVar,
        text_on: str,
        text_off: str = None,
        command=None,
        on_bg: str = None,
        on_fg: str = "#11111b",
        off_bg: str = None,
        off_fg: str = None,
        font_family: str = None,
        font_size: int = 8,
        padx: int = 10,
        pady: int = 4,
    ):
        c = PALETTE
        self.variable = variable
        self._text_on = text_on
        self._text_off = text_off if text_off is not None else text_on
        self._command = command
        self._on_bg = on_bg or c["accent_purple"]
        self._on_fg = on_fg
        self._off_bg = off_bg or c["surface_dim"]
        self._off_fg = off_fg or c["text_sub"]
        self._font_family = font_family or c["font_family"]
        self._font_size = font_size
        self._hovered = False

        super().__init__(
            parent,
            text="",
            bg=self._off_bg,
            fg=self._off_fg,
            highlightbackground=c["border"],
            highlightthickness=1,
            cursor="hand2",
            padx=padx,
            pady=pady,
            anchor="w",
        )
        self._trace_id = self.variable.trace_add("write", lambda *_: self._refresh_visual())
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Destroy>", self._on_destroy)
        self._refresh_visual()

    def _on_destroy(self, event=None):
        if event is None or event.widget is self:
            try:
                self.variable.trace_remove("write", self._trace_id)
            except Exception:
                pass

    def _on_click(self, _event=None):
        self.variable.set(not bool(self.variable.get()))
        if self._command:
            self._command()

    def _on_enter(self, _event=None):
        self._hovered = True
        self._refresh_visual()

    def _on_leave(self, _event=None):
        self._hovered = False
        self._refresh_visual()

    def _refresh_visual(self):
        c = PALETTE
        active = bool(self.variable.get())
        if active:
            base = self._on_bg
            bg = adjust_hex_color(base, -12 if self._hovered else 0)
            self.configure(
                text=f"● {self._text_on}",
                bg=bg,
                fg=self._on_fg,
                highlightbackground=self._on_bg,
                font=(self._font_family, self._font_size, "bold"),
            )
        else:
            base = self._off_bg
            bg = adjust_hex_color(base, 12 if self._hovered else 0)
            self.configure(
                text=f"○ {self._text_off}",
                bg=bg,
                fg=c["text_main"] if self._hovered else self._off_fg,
                highlightbackground=c["border"],
                font=(self._font_family, self._font_size),
            )


class SegmentedSelector(tk.Frame):
    """无闪烁分段按钮组（替代旧版点状 Radiobutton），与 StringVar 双向实时联动。"""

    def __init__(
        self,
        parent,
        variable: tk.StringVar,
        options: list,
        command=None,
        font_family: str = None,
        font_size: int = 8,
    ):
        c = PALETTE
        super().__init__(parent, bg="#11111b", highlightbackground=c["border"], highlightthickness=1, padx=2, pady=2)
        self.variable = variable
        self._items = options
        self._command = command
        self._font_family = font_family or c["font_family"]
        self._font_size = font_size
        self._segments = {}
        self._hovered_val = None

        for idx, (val, label, active_bg) in enumerate(options):
            seg = tk.Label(
                self,
                text=label,
                bg=c["card_bg"],
                fg=c["text_sub"],
                font=(self._font_family, self._font_size),
                padx=10,
                pady=4,
                cursor="hand2",
            )
            seg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0 if idx == 0 else 2, 0))
            seg.bind("<Button-1>", lambda _e, v=val: self.select(v))
            seg.bind("<Enter>", lambda _e, v=val: self._set_hover(v))
            seg.bind("<Leave>", lambda _e, v=val: self._clear_hover(v))
            self._segments[val] = (seg, active_bg or c["accent_blue"])

        self._trace_id = self.variable.trace_add("write", lambda *_: self._refresh_visual())
        self.bind("<Destroy>", self._on_destroy)
        self._refresh_visual()

    def _on_destroy(self, event=None):
        if event is None or event.widget is self:
            try:
                self.variable.trace_remove("write", self._trace_id)
            except Exception:
                pass

    def select(self, value: str):
        if self.variable.get() != value:
            self.variable.set(value)
        if self._command:
            self._command()

    def _set_hover(self, value: str):
        self._hovered_val = value
        self._refresh_visual()

    def _clear_hover(self, value: str):
        if self._hovered_val == value:
            self._hovered_val = None
            self._refresh_visual()

    def _refresh_visual(self):
        c = PALETTE
        cur = self.variable.get()
        for val, (seg, active_bg) in self._segments.items():
            if val == cur:
                seg.configure(bg=active_bg, fg="#11111b", font=(self._font_family, self._font_size, "bold"))
            elif val == self._hovered_val:
                seg.configure(bg=c["input_bg"], fg=c["text_main"], font=(self._font_family, self._font_size))
            else:
                seg.configure(bg=c["card_bg"], fg=c["text_sub"], font=(self._font_family, self._font_size))


class HoverTooltip:
    """轻量级暗色悬浮参数说明条（延迟弹出，自动避让屏幕边界，零焦点干扰）。"""

    def __init__(self, widget, text: str, delay_ms: int = 380):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def _schedule(self, _event=None):
        self._cancel_timer()
        if self.text:
            self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_timer(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._tip or not self.widget.winfo_ismapped():
            return
        c = PALETTE
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        try:
            tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        lbl = tk.Label(
            tip,
            text=self.text,
            bg="#11111b",
            fg=c["text_main"],
            highlightbackground=c["border"],
            highlightthickness=1,
            font=(c["font_family"], 8),
            justify=tk.LEFT,
            wraplength=320,
            padx=9,
            pady=6,
        )
        lbl.pack()
        tip.update_idletasks()
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        sw, sh = self.widget.winfo_screenwidth(), self.widget.winfo_screenheight()
        tw, th = tip.winfo_reqwidth(), tip.winfo_reqheight()
        if x + tw > sw - 12:
            x = max(12, sw - tw - 12)
        if y + th > sh - 12:
            y = max(12, self.widget.winfo_rooty() - th - 4)
        tip.wm_geometry(f"+{x}+{y}")
        self._tip = tip

    def _hide(self, _event=None):
        self._cancel_timer()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None

    def _on_destroy(self, event=None):
        if event is None or event.widget is self.widget:
            self._hide()


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

    FlatButton(
        top_bar,
        text="📁 点此自选文件...",
        bg=c["accent_blue"],
        fg="#11111b",
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=2,
        command=_browse_file,
    ).pack(side=tk.RIGHT, padx=(4, 0))

    FlatButton(
        top_bar,
        text="📂 打开放置目录",
        bg=c["input_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8),
        padx=6,
        pady=2,
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

    FlatButton(
        top_bar,
        text="📁 点此自选文件夹...",
        bg=c["accent_blue"],
        fg="#11111b",
        font=(c["font_family"], 8, "bold"),
        padx=8,
        pady=2,
        command=browse_cmd,
    ).pack(side=tk.RIGHT, padx=(4, 0))

    target_cmd = open_target_cmd if open_target_cmd else (lambda: on_open_folder(var.get()))
    FlatButton(
        top_bar,
        text="📂 打开此目录",
        bg=c["input_bg"],
        fg=c["accent_green"],
        font=(c["font_family"], 8),
        padx=6,
        pady=2,
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


def show_text_viewer_dialog(
    parent,
    title: str,
    header_text: str,
    header_color: str,
    body_text: str,
    size=(740, 500),
    copy_button: bool = True,
):
    """弹出统一的暗色可滚动只读文本窗口（支持暗色标题栏与一键复制到剪贴板）。"""
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
    apply_windows_dark_titlebar(dlg)

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
    if copy_button:
        copy_btn = FlatButton(
            btn_bar,
            text="📋 复制全文",
            bg=c["input_bg"],
            fg=c["accent_green"],
            font=(c["font_family"], 9, "bold"),
            padx=12,
            pady=4,
        )

        def _do_copy():
            try:
                dlg.clipboard_clear()
                dlg.clipboard_append(body_text)
                copy_btn.configure(text="✓ 已复制")
                dlg.after(1400, lambda: copy_btn.configure(text="📋 复制全文") if copy_btn.winfo_exists() else None)
            except Exception:
                pass

        copy_btn.configure(command=_do_copy)
        copy_btn.pack(side=tk.LEFT)

    FlatButton(
        btn_bar,
        text="知道了 / 关闭",
        bg=c["accent_blue"],
        fg="#11111b",
        font=(c["font_family"], 9, "bold"),
        padx=16,
        pady=4,
        command=dlg.destroy,
    ).pack(side=tk.RIGHT)
    return dlg
