"""Persistent right-hand monitor panel: KPI strip, native Canvas Loss/EMA chart, console log, and controls."""

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .theme import PALETTE


class MonitorPanel:
    """右侧常驻监控看板（KPI 卡片 + 实时 Loss/EMA 折线图 + 终端控制台 + 启动/暂停/强杀按钮栏）。"""

    def __init__(self, parent, root, app):
        self.root = root
        self.app = app
        c = PALETTE
        self.card_bg = c["card_bg"]
        self.bg_dark = c["bg_dark"]
        self.text_main = c["text_main"]
        self.text_sub = c["text_sub"]
        self.accent_blue = c["accent_blue"]
        self.accent_green = c["accent_green"]
        self.accent_purple = c["accent_purple"]
        self.accent_yellow = c["accent_yellow"]
        self.input_bg = c["input_bg"]
        self.font_family = c["font_family"]

        self.loss_history = []
        self._loss_step_index = {}
        self._metrics_file_offset = 0
        self._metrics_poll_job = None
        self._chart_resize_job = None

        self.var_show_ema = tk.BooleanVar(value=True)
        self.var_kpi_step = tk.StringVar(value="Step 0 / 0 (0.0%)")
        self.var_kpi_loss = tk.StringVar(value="-- (EMA: --)")
        self.var_kpi_lr = tk.StringVar(value="--")
        self.var_kpi_eta = tk.StringVar(value="--")

        outer = tk.Frame(parent, bg=self.card_bg, padx=10, pady=8)
        outer.pack(fill=tk.BOTH, expand=True)

        self.right_paned = tk.PanedWindow(outer, orient=tk.VERTICAL, bg=self.bg_dark, sashwidth=6, sashrelief=tk.RAISED)
        self.right_paned.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # ==================== 上区：实时 Loss 折线图看板 ====================
        chart_card = tk.Frame(self.right_paned, bg=self.card_bg)
        self.right_paned.add(chart_card, minsize=250, stretch="always")

        chart_hdr = tk.Frame(chart_card, bg=self.card_bg)
        chart_hdr.pack(fill=tk.X, pady=(0, 5))

        tk.Label(
            chart_hdr, text="📈 实时训练 Loss 曲线与收敛监控",
            bg=self.card_bg, fg=self.accent_blue, font=(self.font_family, 10, "bold"),
        ).pack(side=tk.LEFT)

        tk.Button(
            chart_hdr, text="清空曲线", bg=self.input_bg, fg=self.text_sub, relief=tk.FLAT,
            font=(self.font_family, 8), padx=7, pady=1, cursor="hand2", command=self.clear_loss_chart,
        ).pack(side=tk.RIGHT, padx=(4, 0))

        tk.Button(
            chart_hdr, text="📊 加载当前 LoRA 历史曲线", bg=self.input_bg, fg=self.accent_green,
            relief=tk.FLAT, font=(self.font_family, 8, "bold"), padx=8, pady=1, cursor="hand2",
            command=lambda: self.load_history_metrics(silent=False),
        ).pack(side=tk.RIGHT, padx=(4, 0))

        tk.Checkbutton(
            chart_hdr, text="平滑趋势 (EMA)", variable=self.var_show_ema,
            bg=self.card_bg, fg=self.accent_green, selectcolor=self.input_bg,
            activebackground=self.card_bg, font=(self.font_family, 8), command=self.redraw_loss_chart,
        ).pack(side=tk.RIGHT, padx=(0, 6))

        kpi_bar = tk.Frame(chart_card, bg=self.card_bg)
        kpi_bar.pack(fill=tk.X, pady=(0, 6))
        for col_idx, (title, var, color) in enumerate([
            ("训练进度", self.var_kpi_step, self.accent_blue),
            ("实时 Loss / EMA", self.var_kpi_loss, self.accent_green),
            ("当前学习率 (LR)", self.var_kpi_lr, self.accent_purple),
            ("速度 / 预估剩余", self.var_kpi_eta, self.accent_yellow),
        ]):
            kpi_bar.columnconfigure(col_idx, weight=1)
            box = tk.Frame(kpi_bar, bg=self.bg_dark, highlightbackground=self.input_bg, highlightthickness=1, padx=8, pady=4)
            box.grid(row=0, column=col_idx, sticky=tk.EW, padx=(0 if col_idx == 0 else 4, 0))
            tk.Label(box, text=title, bg=self.bg_dark, fg=self.text_sub, font=(self.font_family, 7)).pack(anchor="w")
            tk.Label(box, textvariable=var, bg=self.bg_dark, fg=color, font=("Consolas", 9, "bold")).pack(anchor="w")

        self.progress_canvas = tk.Canvas(chart_card, height=6, bg="#11111b", highlightthickness=0)
        self.progress_canvas.pack(fill=tk.X, pady=(0, 5))

        self.loss_canvas = tk.Canvas(chart_card, bg="#11111b", highlightbackground=self.input_bg, highlightthickness=1)
        self.loss_canvas.pack(fill=tk.BOTH, expand=True)
        self.loss_canvas.bind("<Configure>", self._on_chart_resize)

        # ==================== 下区：实时日志终端控制台 ====================
        log_card = tk.Frame(self.right_paned, bg=self.card_bg)
        self.right_paned.add(log_card, minsize=180, stretch="always")

        header_box = tk.Frame(log_card, bg=self.card_bg)
        header_box.pack(fill=tk.X, pady=(4, 4))

        tk.Label(
            header_box, text="🖥️ 训练实时日志与终端控制台",
            bg=self.card_bg, fg=self.accent_blue, font=(self.font_family, 10, "bold"),
        ).pack(side=tk.LEFT)

        tk.Button(
            header_box, text="清空控制台", bg=self.input_bg, fg=self.text_sub, relief=tk.FLAT,
            font=(self.font_family, 8), padx=7, pady=1, cursor="hand2", command=self.clear_console,
        ).pack(side=tk.RIGHT)

        console_wrap = tk.Frame(log_card, bg="#11111b")
        console_wrap.pack(fill=tk.BOTH, expand=True)

        console_scroll = ttk.Scrollbar(console_wrap, orient="vertical")
        self.console = tk.Text(
            console_wrap, bg="#11111b", fg="#a6e3a1", insertbackground="#a6e3a1",
            font=("Consolas", 9), wrap=tk.WORD, yscrollcommand=console_scroll.set, relief=tk.FLAT, padx=6, pady=4,
        )
        console_scroll.config(command=self.console.yview)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ==================== 底部三按钮操作栏 ====================
        btn_bar = tk.Frame(outer, bg=self.card_bg)
        btn_bar.pack(fill=tk.X)

        self.btn_start = tk.Button(
            btn_bar, text="🚀 启动本地极速训练", bg="#2e7d32", fg="white",
            disabledforeground="#cdd6f4", activebackground="#388e3c",
            font=(self.font_family, 10, "bold"), relief=tk.FLAT, pady=8, cursor="hand2",
            command=self.app.start_training,
        )
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.btn_pause = tk.Button(
            btn_bar, text="⏸️ 保存当前进度并暂停", bg="#45475a", fg="white",
            disabledforeground="#9399b2", activebackground="#b45309",
            font=(self.font_family, 9, "bold"), relief=tk.FLAT, pady=8, state=tk.DISABLED, cursor="hand2",
            command=self.app.pause_and_save_training,
        )
        self.btn_pause.pack(side=tk.LEFT, padx=4, ipadx=12)

        self.btn_stop = tk.Button(
            btn_bar, text="⏹ 强制终止", bg="#45475a", fg="white",
            disabledforeground="#9399b2", activebackground="#d32f2f",
            font=(self.font_family, 9, "bold"), relief=tk.FLAT, pady=8, state=tk.DISABLED, cursor="hand2",
            command=self.app.stop_training,
        )
        self.btn_stop.pack(side=tk.RIGHT, padx=(5, 0), ipadx=12)

    def apply_initial_sash(self):
        try:
            total_h = self.right_paned.winfo_height()
            if total_h > 200:
                top_h = max(290, int(total_h * 0.48))
                self.right_paned.sash_place(0, 0, top_h)
            self.redraw_loss_chart()
        except Exception:
            pass

    def set_running_state(self, running: bool):
        if running:
            self.btn_start.config(state=tk.DISABLED, bg="#45475a")
            self.btn_pause.config(state=tk.NORMAL, text="⏸️ 保存当前进度并暂停", bg="#d97706")
            self.btn_stop.config(state=tk.NORMAL, bg="#c62828")
        else:
            self.btn_start.config(state=tk.NORMAL, bg="#2e7d32")
            self.btn_pause.config(state=tk.DISABLED, text="⏸️ 保存当前进度并暂停", bg="#45475a")
            self.btn_stop.config(state=tk.DISABLED, bg="#45475a")

    def set_pause_pending(self):
        self.btn_pause.config(state=tk.DISABLED, text="⏳ 正在保存当前步...", bg="#45475a")

    def clear_console(self):
        self.console.delete("1.0", tk.END)

    def log(self, text: str):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        if self.app.is_training:
            try:
                self._try_parse_stdout_metrics(text)
            except Exception:
                pass

    def _on_chart_resize(self, _event=None):
        if self._chart_resize_job is not None:
            try:
                self.root.after_cancel(self._chart_resize_job)
            except Exception:
                pass
        self._chart_resize_job = self.root.after(45, self.redraw_loss_chart)

    def clear_loss_chart(self):
        self.loss_history.clear()
        self._loss_step_index.clear()
        self._metrics_file_offset = 0
        self.var_kpi_step.set("Step 0 / 0 (0.0%)")
        self.var_kpi_loss.set("-- (EMA: --)")
        self.var_kpi_lr.set("--")
        self.var_kpi_eta.set("--")
        self.redraw_loss_chart()

    def record_loss_point(self, step: int, loss: float, lr: float = None, step_time: float = None, redraw: bool = True):
        if loss is None or step is None or step < 0:
            return
        try:
            loss_f = float(loss)
            if not (loss_f == loss_f) or loss_f > 1e6:
                return
        except Exception:
            return

        if step in self._loss_step_index:
            idx = self._loss_step_index[step]
            _s, _l, old_lr, old_t = self.loss_history[idx]
            self.loss_history[idx] = (
                step,
                loss_f,
                lr if lr is not None else old_lr,
                step_time if step_time is not None else old_t,
            )
        else:
            self._loss_step_index[step] = len(self.loss_history)
            self.loss_history.append((step, loss_f, lr, step_time))
            if len(self.loss_history) >= 2 and self.loss_history[-1][0] < self.loss_history[-2][0]:
                self.loss_history.sort(key=lambda x: x[0])
                self._loss_step_index = {pt[0]: i for i, pt in enumerate(self.loss_history)}

        if redraw:
            self._update_kpi_and_chart()

    def _compute_ema_series(self, alpha: float = 0.15) -> list:
        ema_vals = []
        ema = None
        for _, loss, _, _ in self.loss_history:
            ema = loss if ema is None else (alpha * loss + (1.0 - alpha) * ema)
            ema_vals.append(ema)
        return ema_vals

    def _update_kpi_and_chart(self):
        if not self.loss_history:
            self.redraw_loss_chart()
            return

        last_step, last_loss, last_lr, last_time = self.loss_history[-1]
        ema_vals = self._compute_ema_series()
        last_ema = ema_vals[-1] if ema_vals else last_loss

        try:
            target_steps = max(last_step, int(self.app.var_steps.get()))
        except Exception:
            target_steps = max(1, last_step)

        pct = min(100.0, (last_step / max(1, target_steps)) * 100.0)
        self.var_kpi_step.set(f"Step {last_step} / {target_steps} ({pct:.1f}%)")
        self.var_kpi_loss.set(f"{last_loss:.4f} (EMA: {last_ema:.4f})")

        if last_lr is not None:
            self.var_kpi_lr.set(f"{last_lr:.2e}")
        if last_time is not None and last_time > 0:
            rem_steps = max(0, target_steps - last_step)
            rem_sec = rem_steps * last_time
            if rem_steps == 0:
                eta_txt = f"{last_time:.1f}s/步 | 已完工"
            elif rem_sec >= 3600:
                eta_txt = f"{last_time:.1f}s/步 | 约 {rem_sec/3600:.1f}h"
            else:
                eta_txt = f"{last_time:.1f}s/步 | 约 {rem_sec/60:.1f}m"
            self.var_kpi_eta.set(eta_txt)

        try:
            pw = self.progress_canvas.winfo_width()
            self.progress_canvas.delete("all")
            if pw > 10:
                self.progress_canvas.create_rectangle(0, 0, pw, 6, fill=self.input_bg, outline="")
                bar_w = int(pw * (pct / 100.0))
                if bar_w > 0:
                    fill_c = self.accent_green if pct >= 100 else self.accent_blue
                    self.progress_canvas.create_rectangle(0, 0, bar_w, 6, fill=fill_c, outline="")
        except Exception:
            pass

        self.redraw_loss_chart(ema_vals=ema_vals)

    def redraw_loss_chart(self, ema_vals=None):
        c = self.loss_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 120 or h < 90:
            return

        pad_l, pad_r, pad_t, pad_b = 58, 22, 26, 30
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)
        c.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline=self.input_bg, width=1)

        if not self.loss_history:
            c.create_text(
                w // 2, h // 2 - 8,
                text="等待训练 Loss 数据流输入...", fill=self.text_sub, font=(self.font_family, 10, "bold"),
            )
            c.create_text(
                w // 2, h // 2 + 14,
                text="启动训练后将实时绘制收敛折线图；也可点击右上角【📊 加载当前 LoRA 历史曲线】复盘",
                fill="#6c7086", font=(self.font_family, 8),
            )
            return

        steps = [pt[0] for pt in self.loss_history]
        losses = [pt[1] for pt in self.loss_history]
        if ema_vals is None:
            ema_vals = self._compute_ema_series()

        min_s, max_s = min(steps), max(steps)
        if min_s == max_s:
            max_s = min_s + 10

        min_l, max_l = min(losses), max(losses)
        if ema_vals:
            min_l = min(min_l, min(ema_vals))
            max_l = max(max_l, max(ema_vals))
        span_l = max_l - min_l
        if span_l < 1e-4:
            span_l = max(0.1, abs(max_l) * 0.2)
        y_min = max(0.0, min_l - span_l * 0.12)
        y_max = max_l + span_l * 0.12
        if y_max <= y_min:
            y_max = y_min + 0.1

        grid_rows = 4
        for i in range(grid_rows + 1):
            frac = i / grid_rows
            y = pad_t + plot_h * frac
            val = y_max - (y_max - y_min) * frac
            if 0 < i < grid_rows:
                c.create_line(pad_l, y, w - pad_r, y, fill="#24273a", dash=(3, 4))
            c.create_text(pad_l - 8, y, text=f"{val:.3f}", anchor="e", fill="#9399b2", font=("Consolas", 8))

        grid_cols = min(6, max(2, plot_w // 95))
        for j in range(grid_cols + 1):
            frac = j / grid_cols
            x = pad_l + plot_w * frac
            s_val = int(round(min_s + (max_s - min_s) * frac))
            if 0 < j < grid_cols:
                c.create_line(x, pad_t, x, h - pad_b, fill="#24273a", dash=(3, 4))
            c.create_text(x, h - pad_b + 12, text=f"{s_val}", anchor="n", fill="#9399b2", font=("Consolas", 8))

        def to_xy(s, l_val):
            px = pad_l + ((s - min_s) / (max_s - min_s)) * plot_w
            py = pad_t + ((y_max - l_val) / (y_max - y_min)) * plot_h
            return px, py

        raw_coords = []
        for s, l_val in zip(steps, losses):
            px, py = to_xy(s, l_val)
            raw_coords.extend([px, py])

        if len(raw_coords) >= 4:
            c.create_line(*raw_coords, fill="#587bb8" if self.var_show_ema.get() else self.accent_blue, width=1.6)
        elif len(raw_coords) == 2:
            px, py = raw_coords
            c.create_oval(px - 3, py - 3, px + 3, py + 3, fill=self.accent_blue, outline="")

        if self.var_show_ema.get() and len(ema_vals) >= 2:
            ema_coords = []
            for s, e_val in zip(steps, ema_vals):
                px, py = to_xy(s, e_val)
                ema_coords.extend([px, py])
            c.create_line(*ema_coords, fill=self.accent_green, width=2.4, smooth=True)

        lx, ly = to_xy(steps[-1], ema_vals[-1] if (self.var_show_ema.get() and ema_vals) else losses[-1])
        c.create_oval(lx - 6, ly - 6, lx + 6, ly + 6, fill="", outline=self.accent_green, width=1.5)
        c.create_oval(lx - 3, ly - 3, lx + 3, ly + 3, fill=self.accent_green, outline="#11111b", width=1)

        badge_text = f"Step {steps[-1]} | Loss {losses[-1]:.4f}"
        bx = min(w - pad_r - 78, max(pad_l + 78, lx - 10))
        by = ly - 16 if ly > pad_t + 28 else ly + 18
        c.create_rectangle(bx - 74, by - 10, bx + 74, by + 10, fill=self.card_bg, outline=self.accent_green, width=1)
        c.create_text(bx, by, text=badge_text, fill=self.text_main, font=("Consolas", 8, "bold"))

        leg_x = w - pad_r - 6
        leg_y = pad_t - 12
        if self.var_show_ema.get():
            c.create_text(leg_x, leg_y, text="━━ 平滑趋势 (EMA)", anchor="e", fill=self.accent_green, font=(self.font_family, 8, "bold"))
            c.create_text(leg_x - 118, leg_y, text="── 原始 Loss", anchor="e", fill="#749be0", font=(self.font_family, 8))
        else:
            c.create_text(leg_x, leg_y, text="── 原始 Loss", anchor="e", fill=self.accent_blue, font=(self.font_family, 8, "bold"))
        c.create_text(pad_l, leg_y, text="Loss", anchor="w", fill="#9399b2", font=("Consolas", 8, "bold"))

    def load_history_metrics(self, silent: bool = False):
        lora_name = self.app.var_lora_name.get().strip() or "my_lora"
        metrics_path = Path("./KohakuMossTrainer/runs/moss") / lora_name / "metrics.jsonl"
        if not metrics_path.exists():
            if not silent:
                messagebox.showinfo(
                    "未找到历史曲线",
                    f"尚未在运行目录中找到 [{lora_name}] 的 metrics.jsonl 指标记录。\n启动训练后折线图将实时自动绘制！",
                )
            return
        try:
            self.loss_history.clear()
            self._loss_step_index.clear()
            raw_text = metrics_path.read_text(encoding="utf-8", errors="ignore")
            self._metrics_file_offset = len(raw_text.encode("utf-8"))
            count = 0
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "step" in row and "loss" in row:
                        self.record_loss_point(
                            int(row["step"]),
                            float(row["loss"]),
                            float(row["lr"]) if "lr" in row else None,
                            float(row["step_time"]) if "step_time" in row else None,
                            redraw=False,
                        )
                        count += 1
                except Exception:
                    pass
            self._update_kpi_and_chart()
            if not silent and count > 0:
                self.log(f"[监控面板] 已加载 [{lora_name}] 历史收敛曲线，共 {count} 个记录点。\n")
        except Exception as err:
            if not silent:
                messagebox.showerror("加载曲线失败", f"读取指标文件异常: {err}")

    def start_metrics_polling(self, reset_chart: bool = True):
        self.stop_metrics_polling()
        if reset_chart:
            self.clear_loss_chart()
        else:
            self.load_history_metrics(silent=True)

        def _poll():
            if not self.app.is_training:
                return
            try:
                lora_name = self.app.var_lora_name.get().strip() or "my_lora"
                metrics_path = Path("./KohakuMossTrainer/runs/moss") / lora_name / "metrics.jsonl"
                if metrics_path.exists():
                    fsize = metrics_path.stat().st_size
                    if fsize < self._metrics_file_offset:
                        self._metrics_file_offset = 0
                    if fsize > self._metrics_file_offset:
                        with open(metrics_path, "rb") as f:
                            f.seek(self._metrics_file_offset)
                            chunk = f.read()
                            self._metrics_file_offset = f.tell()
                        updated = False
                        for line in chunk.decode("utf-8", errors="ignore").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                                if "step" in row and "loss" in row:
                                    self.record_loss_point(
                                        int(row["step"]),
                                        float(row["loss"]),
                                        float(row["lr"]) if "lr" in row else None,
                                        float(row["step_time"]) if "step_time" in row else None,
                                        redraw=False,
                                    )
                                    updated = True
                            except Exception:
                                pass
                        if updated:
                            self._update_kpi_and_chart()
            except Exception:
                pass
            self._metrics_poll_job = self.root.after(2000, _poll)

        self._metrics_poll_job = self.root.after(2000, _poll)

    def stop_metrics_polling(self):
        if self._metrics_poll_job is not None:
            try:
                self.root.after_cancel(self._metrics_poll_job)
            except Exception:
                pass
            self._metrics_poll_job = None

    def _try_parse_stdout_metrics(self, line: str):
        if "loss" not in line.lower():
            return
        m_step = re.search(r"step\s*[=:]?\s*(\d+)", line, re.IGNORECASE)
        m_loss = re.search(r"loss\s*[=:]\s*([0-9]+\.[0-9]+(?:[eE][+-]?\d+)?)", line, re.IGNORECASE)
        if m_step and m_loss:
            m_lr = re.search(r"\blr\s*[=:]\s*([0-9]+\.[0-9]+(?:[eE][+-]?\d+)?)", line, re.IGNORECASE)
            self.record_loss_point(
                int(m_step.group(1)),
                float(m_loss.group(1)),
                float(m_lr.group(1)) if m_lr else None,
                redraw=True,
            )
