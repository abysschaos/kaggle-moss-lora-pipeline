import os
import re
import sys
import json
import threading
import subprocess
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 启用 Windows 高 DPI 适配与独立任务栏应用分组 ID（必须在创建 tk.Tk() 之前调用）
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("OpenMOSS.MOSSImage.LoRATrainer.1")
except Exception:
    pass

# Windows 下彻底禁止后台子进程弹出 CMD/PowerShell 黑框
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _resolve_cli_python() -> str:
    """当 GUI 通过 pythonw.exe 无黑框运行时，将子进程解释器安全映射回同目录 python.exe 以保障完整 stdout 管道。"""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        cli_py = exe.with_name("python.exe")
        if cli_py.is_file():
            return str(cli_py)
    return sys.executable

class ModernLocalTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMOSS MOSS-Image 本地 LoRA 训练引擎控制台")

        # 挂载自定义应用图标（与 MOSS_LoRA_Trainer.exe 保持任务栏/标题栏一致）
        try:
            icon_path = Path(__file__).resolve().parent / "assets" / "moss_trainer.ico"
            if icon_path.is_file():
                self.root.iconbitmap(default=str(icon_path))
        except Exception:
            pass
        
        # 窗口大小与屏幕居中适配 (宽屏自适应，保证左右两栏均有宽裕空间)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1480, max(1240, int(screen_w * 0.88)))
        win_h = min(920, max(820, int(screen_h * 0.86)))
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(1160, 740)

        # 调色盘与字体定义 (Catppuccin Mocha 现代深色主题)
        self.bg_dark = "#181825"
        self.card_bg = "#1e1e2e"
        self.input_bg = "#313244"
        self.accent_blue = "#89b4fa"
        self.accent_green = "#a6e3a1"
        self.accent_red = "#f38ba8"
        self.accent_yellow = "#f9e2af"
        self.accent_purple = "#cba6f7"
        self.accent_peach = "#fab387"
        self.text_main = "#cdd6f4"
        self.text_sub = "#a6adc8"
        self.font_family = "Microsoft YaHei UI"

        # 动态换行标签注册表 (随左侧宽度自适应更新 wraplength)
        self._wrap_labels = []

        # 实时 Loss 曲线数据结构与渲染控制
        self.loss_history = []          # [(step, loss, lr, step_time), ...]
        self._loss_step_index = {}      # step -> index in loss_history
        self._metrics_file_offset = 0
        self._metrics_poll_job = None
        self._chart_resize_job = None
        self.var_show_ema = tk.BooleanVar(value=True)
        self.var_kpi_step = tk.StringVar(value="Step 0 / 0 (0.0%)")
        self.var_kpi_loss = tk.StringVar(value="-- (EMA: --)")
        self.var_kpi_lr = tk.StringVar(value="--")
        self.var_kpi_eta = tk.StringVar(value="--")

        self.root.configure(bg=self.bg_dark)
        self.setup_styles()

        # 本地持久化配置（已在 .gitignore 中忽略，保障隐私且自动记忆用户选择）
        self.config_file = Path("./local_config.json").resolve()
        local_cfg = self.load_local_config()

        # 核心变量初始化（通用默认值 + 宽域自动探测，杜绝任何个人信息硬编码）
        self.current_process = None
        self.is_training = False

        self.var_lora_name = tk.StringVar(value=local_cfg.get("lora_name", "my_lora"))
        self.var_trigger_word = tk.StringVar(value=local_cfg.get("trigger_word", ""))
        self.var_data_dir = tk.StringVar(value=local_cfg.get("data_dir", str(Path("./dataset").resolve())))
        
        # 优先读取本地保存的 ComfyUI 模型库，若无则自动宽域探测
        default_comfy_models = local_cfg.get("comfy_models_dir", "")
        if not default_comfy_models or not os.path.exists(default_comfy_models):
            try:
                import train_local
                discovered_models, _ = train_local.find_broad_comfy_paths()
                if discovered_models:
                    default_comfy_models = str(discovered_models[0])
            except Exception:
                default_comfy_models = ""
        self.var_comfy_models = tk.StringVar(value=default_comfy_models)
        self.var_output_dir = tk.StringVar(value=local_cfg.get("output_dir", str(Path("./output").resolve())))
        self.var_copy_comfy = tk.BooleanVar(value=True)

        # 自选底模与组件权重路径
        self.var_transformer_path = tk.StringVar(value="")
        self.var_text_encoder_path = tk.StringVar(value="")
        self.var_vae_path = tk.StringVar(value="")
        self.var_models_status = tk.StringVar(value="点击【🔍 自动扫描】或手动点选文件")
        self.var_trainer_status = tk.StringVar(value="检查中...")

        self.var_repeats = tk.IntVar(value=10)
        self.var_epochs = tk.IntVar(value=6)
        self.var_steps = tk.IntVar(value=600)
        self.var_rank = tk.IntVar(value=32)
        self.var_alpha = tk.DoubleVar(value=32.0)
        self.var_lr = tk.StringVar(value="1e-4")
        self.var_precision = tk.StringVar(value="bf16")
        self.var_vram_preset = tk.StringVar(value=local_cfg.get("vram_preset", "auto"))
        self.var_resolution = tk.StringVar(value=str(local_cfg.get("resolution", "1024")))
        self.var_text_device = tk.StringVar(value="cpu")
        self.var_vae_device = tk.StringVar(value="cuda")

        # 续炼与进度恢复相关状态变量
        self.train_mode_map = {
            "全新训练 (从头开始)": "fresh",
            "断点续炼 (接着上次暂停或已有进度继续)": "resume",
            "基于已有 LoRA 继续训练 (加载 .safetensors 二阶段微调)": "init_from"
        }
        self.train_mode_rev_map = {v: k for k, v in self.train_mode_map.items()}
        saved_mode_code = local_cfg.get("train_mode", "fresh")
        default_mode_label = self.train_mode_rev_map.get(saved_mode_code, "全新训练 (从头开始)")
        self.var_train_mode_label = tk.StringVar(value=default_mode_label)
        self.var_resume_path = tk.StringVar(value=local_cfg.get("resume_path", ""))
        self.var_save_every = tk.IntVar(value=int(local_cfg.get("save_every", 100)))
        self.var_resume_status = tk.StringVar(value="正在检测历史训练进度...")
        self.pause_requested = False

        self.var_dataset_stat = tk.StringVar(value="未检测")
        self.var_calc_stat = tk.StringVar(value="曝光计算：就绪")
        self.var_output_preview = tk.StringVar(value="")

        self.build_ui()
        self.update_dataset_stats()
        self.recalculate_math()
        self.update_output_preview()
        self.check_trainer_status()
        self.refresh_resume_status()

        # 延迟动态分配黄金左右分栏比例（左侧约占 46%，确保高分屏与全屏最大化下按钮均宽裕完整）
        self.root.after(120, self._apply_initial_sash)

        # 启动时执行一次静默自动底模探测并加载当前 LoRA 的历史 Loss 曲线（如有）
        self.auto_scan_models(silent=True)
        self.root.after(200, lambda: self.load_history_metrics(silent=True))

        # 监听名称变动实时刷新产出路径预览与历史断点状态
        self.var_lora_name.trace_add("write", lambda *args: (self.update_output_preview(), self.refresh_resume_status()))
        self.var_resume_path.trace_add("write", lambda *args: self.refresh_resume_status())

        # 拦截窗口右上角关闭按钮事件，彻底杜绝训练中途直接关窗导致的后台僵尸 GPU 进程
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _apply_initial_sash(self):
        """按窗口当前实际宽度应用 46% / 54% 黄金左右分栏，以及右侧 46% / 54% 上下分栏。"""
        try:
            self.root.update_idletasks()
            total_w = self.main_paned.winfo_width()
            if total_w > 100:
                left_w = max(650, min(860, int(total_w * 0.46)))
                self.main_paned.sash_place(0, left_w, 0)
            if hasattr(self, "right_paned"):
                total_h = self.right_paned.winfo_height()
                if total_h > 200:
                    top_h = max(290, int(total_h * 0.48))
                    self.right_paned.sash_place(0, 0, top_h)
            self.redraw_loss_chart()
        except Exception:
            pass

    def on_closing(self):
        """安全退出拦截器：若后台仍在训练，引导用户优雅保存暂停或级联强杀释放 GPU。"""
        if self.is_training and self.current_process:
            ans = messagebox.askyesnocancel(
                "训练正在进行中",
                "当前后台仍有 LoRA 训练进程正在占用 GPU！\n\n"
                "• 点击【是 (Yes)】：发送【保存当前进度并暂停】指令，等待当前步保存完毕（不强制退出）；\n"
                "• 点击【否 (No)】：立即强制终止训练进程树、回收已有断点并关闭窗口；\n"
                "• 点击【取消 (Cancel)】：返回继续训练。"
            )
            if ans is None:
                return
            elif ans is True:
                self.pause_and_save_training()
                return
            else:
                try:
                    pid = self.current_process.pid
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=_NO_WINDOW,
                        check=False
                    )
                    self.current_process.kill()
                except Exception:
                    pass
                try:
                    import train_local
                    lora_name = self.var_lora_name.get().strip() or "my_lora"
                    dest_dir = Path(self.var_output_dir.get()) / lora_name
                    train_local.export_and_sync_weights(
                        Path("./KohakuMossTrainer").resolve(),
                        lora_name,
                        dest_dir,
                        self.var_comfy_models.get(),
                        self.var_copy_comfy.get(),
                        quiet=True
                    )
                except Exception:
                    pass
        self.save_local_config()
        self.root.destroy()

    def load_local_config(self) -> dict:
        """从本地私有配置文件读取用户历史偏好路径（该文件不入 Git）。"""
        if self.config_file.exists():
            try:
                import json
                return json.loads(self.config_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save_local_config(self):
        """将当前用户选定的本地路径记忆到本地私有配置文件中。"""
        try:
            import json
            mode_code = self.train_mode_map.get(self.var_train_mode_label.get(), "fresh")
            cfg = {
                "data_dir": self.var_data_dir.get(),
                "comfy_models_dir": self.var_comfy_models.get(),
                "output_dir": self.var_output_dir.get(),
                "lora_name": self.var_lora_name.get(),
                "trigger_word": self.var_trigger_word.get(),
                "vram_preset": self.var_vram_preset.get(),
                "resolution": self.var_resolution.get(),
                "train_mode": mode_code,
                "resume_path": self.var_resume_path.get(),
                "save_every": self.var_save_every.get()
            }
            self.config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.bg_dark, foreground=self.text_main, font=(self.font_family, 9))
        style.configure("Card.TFrame", background=self.card_bg)
        style.configure("TLabel", background=self.card_bg, foreground=self.text_main, font=(self.font_family, 9))
        style.configure("Sub.TLabel", background=self.card_bg, foreground=self.text_sub, font=(self.font_family, 8))
        style.configure("Header.TLabel", background=self.card_bg, foreground=self.accent_blue, font=(self.font_family, 10, "bold"))
        style.configure("TCheckbutton", background=self.card_bg, foreground=self.text_main, font=(self.font_family, 9))
        
        # 彻底解决 ttk.Combobox 在 Windows 下 readonly 状态出现浅米黄底色导致白字隐形的问题
        style.configure(
            "TCombobox",
            fieldbackground=self.input_bg,
            background="#45475a",
            foreground=self.text_main,
            darkcolor=self.card_bg,
            lightcolor=self.card_bg,
            arrowcolor=self.text_main,
            bordercolor=self.card_bg,
            padding=3,
            font=(self.font_family, 9)
        )
        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", self.input_bg),
                ("disabled", "#1e1e2e"),
                ("focus", self.input_bg),
                ("active", self.input_bg)
            ],
            foreground=[
                ("readonly", self.text_main),
                ("disabled", self.text_sub),
                ("focus", self.text_main),
                ("active", self.text_main)
            ],
            selectbackground=[
                ("readonly", self.input_bg),
                ("focus", self.accent_blue)
            ],
            selectforeground=[
                ("readonly", self.text_main),
                ("focus", "#11111b")
            ],
            background=[
                ("readonly", "#45475a"),
                ("active", self.accent_blue)
            ],
            arrowcolor=[
                ("readonly", self.text_main),
                ("active", "#11111b")
            ]
        )

        # 修复下拉弹出菜单 (Listbox) 的背景与高亮对比度
        self.root.option_add("*TCombobox*Listbox*Background", self.input_bg)
        self.root.option_add("*TCombobox*Listbox*Foreground", self.text_main)
        self.root.option_add("*TCombobox*Listbox*selectBackground", self.accent_blue)
        self.root.option_add("*TCombobox*Listbox*selectForeground", "#11111b")
        self.root.option_add("*TCombobox*Listbox*font", (self.font_family, 9))

    def build_ui(self):
        # 1. 顶部 Header 状态条
        header_frame = tk.Frame(self.root, bg=self.card_bg, height=55)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 6))

        title_lbl = tk.Label(header_frame, text="OpenMOSS MOSS-Image 本地 LoRA 训练引擎",
                             bg=self.card_bg, fg=self.accent_blue, font=(self.font_family, 13, "bold"))
        title_lbl.pack(side=tk.LEFT, padx=14, pady=10)

        gpu_info = self.probe_gpu_quick()
        gpu_lbl = tk.Label(header_frame, text=f"设备状态: {gpu_info}",
                           bg=self.card_bg, fg=self.accent_green, font=(self.font_family, 9))
        gpu_lbl.pack(side=tk.RIGHT, padx=14, pady=10)

        # 2. 主体左右分栏 (双侧 stretch='always'，最大化全屏时左右按黄金比例同步扩展)
        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.bg_dark, sashwidth=6, sashrelief=tk.RAISED)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 左侧控制面板 (宽裕 minsize=620，彻底杜绝按钮被挤压截断)
        left_outer = tk.Frame(self.main_paned, bg=self.card_bg)
        self.main_paned.add(left_outer, minsize=620, stretch="always")

        # 右侧实时监控 (Loss 折线图看板 + 实时日志控制台)
        right_container = tk.Frame(self.main_paned, bg=self.card_bg)
        self.main_paned.add(right_container, minsize=500, stretch="always")

        self.build_left_scrollable(left_outer)
        self.build_right_panel(right_container)

    def probe_gpu_quick(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                bf16 = "支持 BF16" if torch.cuda.is_bf16_supported() else "仅 FP16"
                return f"{name} ({vram:.1f} GB, {bf16})"
            return "仅 CPU (未检测到 CUDA)"
        except Exception:
            return "PyTorch 未加载"

    def open_in_explorer(self, target_path: str):
        """在 Windows 资源管理器中打开指定文件夹，不存在则先自动创建。"""
        try:
            p = Path(target_path).resolve()
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            os.startfile(str(p))
        except Exception as e:
            messagebox.showerror("打开文件夹失败", f"无法打开指定路径:\n{target_path}\n错误信息: {e}")

    def build_left_scrollable(self, parent):
        canvas = tk.Canvas(parent, bg=self.card_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.card_bg)

        # 核心关键：记录 window 实例并在 canvas 大小变动时动态同步绑定框架宽度与多行标签折行宽度
        self.canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def on_canvas_configure(event):
            canvas.itemconfig(self.canvas_window, width=event.width)
            new_wrap = max(480, event.width - 42)
            for lbl in self._wrap_labels:
                try:
                    lbl.configure(wraplength=new_wrap)
                except Exception:
                    pass

        canvas.bind("<Configure>", on_canvas_configure)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(_event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(_event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        scrollable_frame.bind("<Enter>", _bind_wheel)

        canvas.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        scrollbar.pack(side="right", fill="y", pady=4)

        self.build_left_controls(scrollable_frame)

    def build_left_controls(self, parent):
        # 1. 快捷预设按钮栏
        preset_frame = tk.Frame(parent, bg=self.card_bg)
        preset_frame.pack(fill=tk.X, pady=(0, 6))
        
        tk.Label(preset_frame, text="快速模板预设：", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9, "bold")).pack(side=tk.LEFT, padx=(4, 6), pady=4)

        btn_style = tk.Button(preset_frame, text="🎨 艺术画风模板", bg="#45475a", fg=self.text_main,
                              activebackground=self.accent_blue, font=(self.font_family, 8, "bold"),
                              relief=tk.FLAT, padx=10, pady=3, cursor="hand2", command=self.apply_style_preset)
        btn_style.pack(side=tk.LEFT, padx=4)

        btn_char = tk.Button(preset_frame, text="👤 人物角色模板", bg="#45475a", fg=self.text_main,
                             activebackground=self.accent_blue, font=(self.font_family, 8, "bold"),
                             relief=tk.FLAT, padx=10, pady=3, cursor="hand2", command=self.apply_character_preset)
        btn_char.pack(side=tk.LEFT, padx=4)

        # 2. 命名与触发词卡片
        name_card = tk.LabelFrame(parent, text=" 🏷️ 自定义 LoRA 命名与触发词 ", bg=self.card_bg, fg=self.accent_blue,
                                  font=(self.font_family, 9, "bold"), padx=10, pady=6)
        name_card.pack(fill=tk.X, pady=(0, 6))

        tk.Label(name_card, text="LoRA与文件夹名称:", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").grid(row=0, column=0, sticky=tk.W, pady=2)
        ent_lora_name = tk.Entry(name_card, textvariable=self.var_lora_name, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main)
        ent_lora_name.grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        name_card.columnconfigure(1, weight=1)

        tk.Label(name_card, text="自定义触发词(Trigger):", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").grid(row=1, column=0, sticky=tk.W, pady=2)
        ent_trigger = tk.Entry(name_card, textvariable=self.var_trigger_word, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main)
        ent_trigger.grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)

        lbl_trig_hint = tk.Label(name_card, text="说明: 若填写触发词，系统自动前置注入每张图片标注；留空则直接使用原标注。",
                                 bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8), wraplength=560, justify=tk.LEFT)
        lbl_trig_hint.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(1, 2))
        self._wrap_labels.append(lbl_trig_hint)

        lbl_preview = tk.Label(name_card, textvariable=self.var_output_preview, bg=self.card_bg, fg=self.accent_green,
                               font=(self.font_family, 8), wraplength=560, justify=tk.LEFT)
        lbl_preview.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        self._wrap_labels.append(lbl_preview)

        # 2.5 续炼与进度恢复卡片 (分层宽裕布局，按钮 100% 清晰可见绝不截断)
        resume_card = tk.LabelFrame(parent, text=" 🔄 续炼与进度恢复 (支持中途断点续炼 / 已完工 LoRA 继续训练) ", bg=self.card_bg, fg=self.accent_blue,
                                    font=(self.font_family, 9, "bold"), padx=10, pady=6)
        resume_card.pack(fill=tk.X, pady=(0, 6))

        # 第1行：先 pack 右侧固定按钮，再 pack 左侧标签与自适应下拉框
        rm_row0 = tk.Frame(resume_card, bg=self.card_bg)
        rm_row0.pack(fill=tk.X, pady=2)
        btn_chk_prog = tk.Button(
            rm_row0, text="🔍 检测现有进度", bg="#313244", fg=self.accent_blue,
            relief=tk.FLAT, font=(self.font_family, 8, "bold"), cursor="hand2", padx=8, pady=2,
            command=self.refresh_resume_status
        )
        btn_chk_prog.pack(side=tk.RIGHT, padx=(6, 0))
        tk.Label(rm_row0, text="训练/续炼模式:", bg=self.card_bg, fg=self.text_main, width=14, anchor="w").pack(side=tk.LEFT)
        cb_tmode = ttk.Combobox(
            rm_row0,
            textvariable=self.var_train_mode_label,
            values=list(self.train_mode_map.keys()),
            state="readonly"
        )
        cb_tmode.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        cb_tmode.bind("<<ComboboxSelected>>", lambda e: (self.refresh_resume_status(), self.save_local_config()))

        # 第2行：提示标题独占一行
        tk.Label(resume_card, text="指定已有 LoRA / 断点路径 (留空则自动匹配当前 LoRA 名称的最新进度):",
                 bg=self.card_bg, fg=self.text_main, font=(self.font_family, 8, "bold"), anchor="w").pack(fill=tk.X, pady=(6, 2))

        # 第3行：独立按钮操作条独占一行，三个按钮清晰并列绝不挤压
        rm_btn_bar = tk.Frame(resume_card, bg=self.card_bg)
        rm_btn_bar.pack(fill=tk.X, pady=(0, 4))

        btn_pick_lora = tk.Button(
            rm_btn_bar, text="📁 选择已训练 LoRA (.safetensors)", bg=self.accent_blue, fg="#11111b",
            relief=tk.FLAT, font=(self.font_family, 8, "bold"), cursor="hand2", padx=10, pady=3,
            command=self.browse_existing_lora_file
        )
        btn_pick_lora.pack(side=tk.LEFT, padx=(0, 6))

        btn_pick_ckpt = tk.Button(
            rm_btn_bar, text="📂 选择断点目录 (step_xxx)", bg="#313244", fg=self.accent_green,
            relief=tk.FLAT, font=(self.font_family, 8, "bold"), cursor="hand2", padx=10, pady=3,
            command=self.browse_resume_ckpt_dir
        )
        btn_pick_ckpt.pack(side=tk.LEFT, padx=(0, 6))

        btn_clear_res = tk.Button(
            rm_btn_bar, text="✖ 清空路径", bg="#45475a", fg=self.text_main,
            relief=tk.FLAT, font=(self.font_family, 8), cursor="hand2", padx=8, pady=3,
            command=lambda: (self.var_resume_path.set(""), self.refresh_resume_status(), self.save_local_config())
        )
        btn_clear_res.pack(side=tk.LEFT)

        # 第4行：全宽路径输入框
        ent_resume_p = tk.Entry(resume_card, textvariable=self.var_resume_path, bg=self.input_bg, fg=self.text_main,
                                insertbackground=self.text_main, font=("Consolas", 8))
        ent_resume_p.pack(fill=tk.X, pady=(0, 4))

        # 第5行：自动存盘间隔
        rm_row2 = tk.Frame(resume_card, bg=self.card_bg)
        rm_row2.pack(fill=tk.X, pady=2)
        tk.Label(rm_row2, text="自动存盘间隔:", bg=self.card_bg, fg=self.text_main, width=14, anchor="w").pack(side=tk.LEFT)
        sp_save_ev = tk.Spinbox(rm_row2, from_=10, to=2000, increment=10, textvariable=self.var_save_every, width=7,
                                bg=self.input_bg, fg=self.text_main, command=self.save_local_config)
        sp_save_ev.pack(side=tk.LEFT, padx=(4, 6))
        tk.Label(rm_row2, text="步/次  (训练期间可随时点击右下角【⏸️ 保存当前进度并暂停】)",
                 bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8)).pack(side=tk.LEFT)

        lbl_rstat = tk.Label(resume_card, textvariable=self.var_resume_status, bg=self.card_bg, fg=self.accent_yellow,
                             font=(self.font_family, 8), wraplength=560, justify=tk.LEFT)
        lbl_rstat.pack(fill=tk.X, pady=(3, 0))
        self._wrap_labels.append(lbl_rstat)

        # 3. 训练底模与组件权重设置卡片
        model_card = tk.LabelFrame(parent, text=" 📦 训练底模与组件权重设置 (支持自选或自动探测) ", bg=self.card_bg, fg=self.accent_blue,
                                   font=(self.font_family, 9, "bold"), padx=10, pady=6)
        model_card.pack(fill=tk.X, pady=(0, 6))

        scan_bar = tk.Frame(model_card, bg=self.card_bg)
        scan_bar.pack(fill=tk.X, pady=(0, 4))
        btn_scan = tk.Button(scan_bar, text="🔍 自动扫描识别本地模型", bg="#313244", fg=self.accent_blue,
                             activebackground=self.accent_blue, font=(self.font_family, 8, "bold"),
                             relief=tk.FLAT, padx=8, pady=2, cursor="hand2", command=lambda: self.auto_scan_models(silent=False))
        btn_scan.pack(side=tk.LEFT)
        lbl_mstat = tk.Label(scan_bar, textvariable=self.var_models_status, bg=self.card_bg, fg=self.accent_yellow, font=(self.font_family, 8))
        lbl_mstat.pack(side=tk.LEFT, padx=8)

        # (1) DiT 底模
        self.create_file_picker_block(
            model_card, "DiT 底模文件 (Diffusion Model):", self.var_transformer_path,
            browse_title="选择 MOSS-Image 底模文件 (支持 Base/Heal/SFT，如 moss_image_heal_bf16.safetensors 等)",
            folder_path="models/diffusion_models"
        )

        # (2) 文本编码器
        self.create_file_picker_block(
            model_card, "文本编码器文件 (Text Encoder):", self.var_text_encoder_path,
            browse_title="选择 Qwen3.5-9B 文本编码器 (qwen3.5_9b_bf16.safetensors 等)",
            folder_path="models/text_encoders"
        )

        # (3) VAE
        self.create_file_picker_block(
            model_card, "图像 VAE 文件 (Autoencoder):", self.var_vae_path,
            browse_title="选择 FLUX.2 VAE 文件 (flux2-vae_small.safetensors)",
            folder_path="models/vae"
        )

        # 4. 训练器框架管理卡片
        trainer_card = tk.LabelFrame(parent, text=" 🛠️ 训练器框架管理 (KohakuMossTrainer) ", bg=self.card_bg, fg=self.accent_blue,
                                     font=(self.font_family, 9, "bold"), padx=10, pady=6)
        trainer_card.pack(fill=tk.X, pady=(0, 6))

        t_status_bar = tk.Frame(trainer_card, bg=self.card_bg)
        t_status_bar.pack(fill=tk.X, pady=(0, 4))
        tk.Label(t_status_bar, text="框架状态:", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 8, "bold")).pack(side=tk.LEFT)
        lbl_tstat = tk.Label(t_status_bar, textvariable=self.var_trainer_status, bg=self.card_bg, fg=self.accent_yellow, font=(self.font_family, 8, "bold"))
        lbl_tstat.pack(side=tk.LEFT, padx=6)

        t_row1 = tk.Frame(trainer_card, bg=self.card_bg)
        t_row1.pack(fill=tk.X, pady=2)
        btn_open_trainer = tk.Button(t_row1, text="📂 打开训练器放置目录", bg="#313244", fg=self.accent_green, relief=tk.FLAT,
                                     font=(self.font_family, 8, "bold"), cursor="hand2", command=self.open_trainer_folder)
        btn_open_trainer.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_clone_trainer = tk.Button(t_row1, text="⬇️ 一键自动克隆/下载", bg=self.accent_blue, fg="#11111b", relief=tk.FLAT,
                                      font=(self.font_family, 8, "bold"), cursor="hand2", command=self.download_or_clone_trainer)
        btn_clone_trainer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        t_row2 = tk.Frame(trainer_card, bg=self.card_bg)
        t_row2.pack(fill=tk.X, pady=2)
        btn_patch_trainer = tk.Button(t_row2, text="🩹 检查并注入补丁", bg="#313244", fg=self.accent_yellow, relief=tk.FLAT,
                                      font=(self.font_family, 8), cursor="hand2", command=self.patch_trainer_manually)
        btn_patch_trainer.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_web_trainer = tk.Button(t_row2, text="🌐 访问官方 GitHub", bg="#45475a", fg=self.text_main, relief=tk.FLAT,
                                    font=(self.font_family, 8), cursor="hand2", command=self.open_trainer_github)
        btn_web_trainer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        lbl_trainer_note = tk.Label(trainer_card, text="说明: 推荐点击【⬇️ 一键自动克隆/下载】全自动安装；若网络受限，可点击【📂 打开训练器放置目录】手动解压源码入内，并点击【🩹 检查并注入补丁】自愈修复。",
                                    bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8), wraplength=560, justify=tk.LEFT)
        lbl_trainer_note.pack(fill=tk.X, pady=(2, 0))
        self._wrap_labels.append(lbl_trainer_note)

        # 5. 数据集与产出目录卡片
        path_card = tk.LabelFrame(parent, text=" 📂 数据集与产出目录 ", bg=self.card_bg, fg=self.accent_blue,
                                  font=(self.font_family, 9, "bold"), padx=10, pady=6)
        path_card.pack(fill=tk.X, pady=(0, 6))

        self.create_dir_picker_block(
            path_card, "训练图片目录 (成对 jpg/png + txt):", self.var_data_dir,
            browse_cmd=self.browse_data_dir, stat_var=self.var_dataset_stat
        )

        self.create_dir_picker_block(
            path_card, "结果存放根目录 (LoRA 归档位置):", self.var_output_dir,
            browse_cmd=self.browse_output_dir
        )

        self.create_dir_picker_block(
            path_card, "ComfyUI 模型库目录 (自动复用):", self.var_comfy_models,
            browse_cmd=self.browse_comfy_dir
        )

        chk_comfy = tk.Checkbutton(path_card, text="训练完成后自动同步复制权重至 ComfyUI models/loras 专属子目录",
                                   variable=self.var_copy_comfy, bg=self.card_bg, fg=self.text_main,
                                   selectcolor=self.input_bg, activebackground=self.card_bg,
                                   font=(self.font_family, 8), wraplength=560, justify=tk.LEFT, anchor="w")
        chk_comfy.pack(fill=tk.X, pady=(4, 2))
        self._wrap_labels.append(chk_comfy)

        # 6. 超参数配置卡片
        hp_card = tk.LabelFrame(parent, text=" ⚙️ 核心训练超参数 ", bg=self.card_bg, fg=self.accent_blue,
                                font=(self.font_family, 9, "bold"), padx=10, pady=6)
        hp_card.pack(fill=tk.X, pady=(0, 6))

        r_f = tk.Frame(hp_card, bg=self.card_bg)
        r_f.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(r_f, text="单图重复 (Repeats):", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").pack(side=tk.LEFT)
        sp_rep = tk.Spinbox(r_f, from_=1, to=100, textvariable=self.var_repeats, width=7, bg=self.input_bg, fg=self.text_main, command=self.recalculate_math)
        sp_rep.pack(side=tk.LEFT, padx=(0, 14))
        sp_rep.bind("<KeyRelease>", lambda e: self.recalculate_math())

        tk.Label(r_f, text="训练轮数 (Epochs):", bg=self.card_bg, fg=self.text_main).pack(side=tk.LEFT)
        sp_ep = tk.Spinbox(r_f, from_=1, to=100, textvariable=self.var_epochs, width=7, bg=self.input_bg, fg=self.text_main, command=self.recalculate_math)
        sp_ep.pack(side=tk.LEFT, padx=(4, 0))
        sp_ep.bind("<KeyRelease>", lambda e: self.recalculate_math())

        s_f = tk.Frame(hp_card, bg=self.card_bg)
        s_f.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(s_f, text="核心步数 (Steps):", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").pack(side=tk.LEFT)
        ent_steps = tk.Entry(s_f, textvariable=self.var_steps, width=9, bg=self.input_bg, fg=self.text_main)
        ent_steps.pack(side=tk.LEFT)
        btn_calc = tk.Button(s_f, text="⚡ 按公式对齐", bg="#45475a", fg=self.text_main, relief=tk.FLAT, font=(self.font_family, 8),
                             cursor="hand2", command=self.align_steps_by_formula)
        btn_calc.pack(side=tk.LEFT, padx=6)

        math_lbl = tk.Label(hp_card, textvariable=self.var_calc_stat, bg=self.card_bg, fg=self.accent_yellow, font=(self.font_family, 8))
        math_lbl.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 3))

        ra_f = tk.Frame(hp_card, bg=self.card_bg)
        ra_f.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(ra_f, text="LoRA 矩阵秩 (Rank):", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").pack(side=tk.LEFT)
        cb_rank = ttk.Combobox(ra_f, textvariable=self.var_rank, values=[16, 32, 64, 128], width=7, state="readonly")
        cb_rank.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(ra_f, text="缩放系数 (Alpha):", bg=self.card_bg, fg=self.text_main).pack(side=tk.LEFT)
        ent_alpha = tk.Entry(ra_f, textvariable=self.var_alpha, width=7, bg=self.input_bg, fg=self.text_main)
        ent_alpha.pack(side=tk.LEFT, padx=(4, 0))

        lr_f = tk.Frame(hp_card, bg=self.card_bg)
        lr_f.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(lr_f, text="基础学习率 (LR):", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").pack(side=tk.LEFT)
        ent_lr = tk.Entry(lr_f, textvariable=self.var_lr, width=9, bg=self.input_bg, fg=self.text_main)
        ent_lr.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(lr_f, text="计算精度:", bg=self.card_bg, fg=self.text_main).pack(side=tk.LEFT)
        cb_prec = ttk.Combobox(lr_f, textvariable=self.var_precision, values=["bf16", "fp16"], width=7, state="readonly")
        cb_prec.pack(side=tk.LEFT, padx=(4, 0))

        preset_f = tk.Frame(hp_card, bg=self.card_bg)
        preset_f.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=2)
        tk.Label(preset_f, text="显存优化档位:", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").pack(side=tk.LEFT)
        cb_vram = ttk.Combobox(preset_f, textvariable=self.var_vram_preset, values=["auto", "6g", "8g", "12g", "16g"], width=7, state="readonly")
        cb_vram.pack(side=tk.LEFT, padx=(0, 14))
        cb_vram.bind("<<ComboboxSelected>>", lambda e: self.save_local_config())

        tk.Label(preset_f, text="训练分辨率:", bg=self.card_bg, fg=self.text_main).pack(side=tk.LEFT)
        cb_res = ttk.Combobox(preset_f, textvariable=self.var_resolution, values=["1024", "768", "512"], width=7, state="readonly")
        cb_res.pack(side=tk.LEFT, padx=(4, 0))
        cb_res.bind("<<ComboboxSelected>>", lambda e: self.save_local_config())

        dev_f = tk.Frame(hp_card, bg=self.card_bg)
        dev_f.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(2, 4))
        tk.Label(dev_f, text="文本提取设备:", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").pack(side=tk.LEFT)
        cb_txt_dev = ttk.Combobox(dev_f, textvariable=self.var_text_device, values=["cpu", "cuda"], width=7, state="readonly")
        cb_txt_dev.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(dev_f, text="(8GB 显卡请锁定 CPU 避免 OOM；显存档位 auto 智能推荐)", bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8)).pack(side=tk.LEFT)

    def create_file_picker_block(self, parent, label_text, var, browse_title, folder_path, filetypes=None):
        """
        优雅防遮挡的两行式文件选择组件：
        第1行：标签名 + 【📁 点此自选文件...】醒目大按钮 + 【📂 打开放置目录】
        第2行：全宽单行输入框 (展示当前选定文件的绝对路径)
        """
        if filetypes is None:
            filetypes = [("Safetensors模型", "*.safetensors"), ("所有模型", "*.safetensors *.pt *.bin"), ("所有文件", "*.*")]
            
        block = tk.Frame(parent, bg=self.card_bg)
        block.pack(fill=tk.X, pady=3)

        top_bar = tk.Frame(block, bg=self.card_bg)
        top_bar.pack(fill=tk.X, pady=(0, 2))

        def browse_file():
            selected = filedialog.askopenfilename(
                title=browse_title,
                filetypes=filetypes
            )
            if selected:
                var.set(selected)
                self.var_models_status.set("已手动自选模型权重")
                self.update_output_preview()

        # 黄金布局原则：先 pack 右侧固定宽度按钮，再 pack 左侧标题，彻底杜绝窄屏截断按钮
        btn_browse = tk.Button(top_bar, text="📁 点此自选文件...", bg=self.accent_blue, fg="#11111b",
                               relief=tk.FLAT, font=(self.font_family, 8, "bold"), cursor="hand2", padx=8, pady=1,
                               command=browse_file)
        btn_browse.pack(side=tk.RIGHT, padx=(4, 0))

        btn_open = tk.Button(top_bar, text="📂 打开放置目录", bg="#313244", fg=self.accent_green,
                             relief=tk.FLAT, font=(self.font_family, 8), cursor="hand2", padx=6, pady=1,
                             command=lambda: self.open_in_explorer(folder_path))
        btn_open.pack(side=tk.RIGHT)

        tk.Label(top_bar, text=label_text, bg=self.card_bg, fg=self.text_main,
                 font=(self.font_family, 8, "bold"), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        ent = tk.Entry(block, textvariable=var, bg=self.input_bg, fg=self.text_main,
                       insertbackground=self.text_main, relief=tk.SOLID, bd=1, font=("Consolas", 8))
        ent.pack(fill=tk.X, pady=(1, 2))

    def create_dir_picker_block(self, parent, label_text, var, browse_cmd, open_target_cmd=None, stat_var=None):
        """
        优雅防遮挡的两行式目录选择组件：
        第1行：标签名 + 【📁 点此自选文件夹...】醒目大按钮 + 【📂 打开目录】
        第2行：全宽单行输入框 (展示选定目录绝对路径)
        可选第3行：状态指示标签
        """
        block = tk.Frame(parent, bg=self.card_bg)
        block.pack(fill=tk.X, pady=3)

        top_bar = tk.Frame(block, bg=self.card_bg)
        top_bar.pack(fill=tk.X, pady=(0, 2))

        # 黄金布局原则：先 pack 右侧操作按钮，确保文字 100% 完整可见
        btn_browse = tk.Button(top_bar, text="📁 点此自选文件夹...", bg=self.accent_blue, fg="#11111b",
                               relief=tk.FLAT, font=(self.font_family, 8, "bold"), cursor="hand2", padx=8, pady=1,
                               command=browse_cmd)
        btn_browse.pack(side=tk.RIGHT, padx=(4, 0))

        target = open_target_cmd if open_target_cmd else lambda: self.open_in_explorer(var.get())
        btn_open = tk.Button(top_bar, text="📂 打开此目录", bg="#313244", fg=self.accent_green,
                             relief=tk.FLAT, font=(self.font_family, 8), cursor="hand2", padx=6, pady=1,
                             command=target)
        btn_open.pack(side=tk.RIGHT)

        tk.Label(top_bar, text=label_text, bg=self.card_bg, fg=self.text_main,
                 font=(self.font_family, 8, "bold"), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        ent = tk.Entry(block, textvariable=var, bg=self.input_bg, fg=self.text_main,
                       insertbackground=self.text_main, relief=tk.SOLID, bd=1, font=("Consolas", 8))
        ent.pack(fill=tk.X, pady=(1, 2))

        if stat_var:
            lbl_stat = tk.Label(block, textvariable=stat_var, bg=self.card_bg, fg=self.accent_yellow,
                                font=(self.font_family, 8), anchor="w")
            lbl_stat.pack(fill=tk.X, pady=(0, 2))

    # 训练器管理方法
    def check_trainer_status(self):
        trainer_dir = Path("./KohakuMossTrainer").resolve()
        pyproject = trainer_dir / "pyproject.toml"
        src_dir = trainer_dir / "src" / "kohakumoss"
        if pyproject.exists() and src_dir.exists():
            self.var_trainer_status.set("✅ 已就绪 (包含完整训练架构)")
        elif trainer_dir.exists():
            self.var_trainer_status.set("⚠️ 目录已创建，但尚未发现 pyproject.toml 文件")
        else:
            self.var_trainer_status.set("❌ 尚未就绪 (请点击下载或放置)")

    def open_trainer_folder(self):
        trainer_dir = Path("./KohakuMossTrainer").resolve()
        if not trainer_dir.exists():
            trainer_dir.mkdir(parents=True, exist_ok=True)
            readme = trainer_dir / "训练器放置说明.txt"
            readme.write_text(
                "[KohakuMossTrainer 训练器放置说明]\n\n"
                "请将从 GitHub 下载并解压的训练器源码文件全部放入本目录。\n"
                "放置后，请确保直接打开本目录时能看到以下核心文件:\n"
                " - pyproject.toml\n"
                " - src/ (包含 kohakumoss 子目录)\n"
                " - configs/\n"
                " - scripts/\n\n"
                "官方仓库地址: https://github.com/KohakuBlueleaf/KohakuMossTrainer\n"
                "放置完成后，返回 GUI 界面点击【🩹 检查并注入补丁】即可就绪！\n",
                encoding="utf-8"
            )
        self.open_in_explorer(str(trainer_dir))

    def open_trainer_github(self):
        webbrowser.open("https://github.com/KohakuBlueleaf/KohakuMossTrainer")

    def patch_trainer_manually(self):
        trainer_dir = Path("./KohakuMossTrainer").resolve()
        if not (trainer_dir / "pyproject.toml").exists():
            messagebox.showwarning("未发现训练器", "未在 KohakuMossTrainer 目录下找到 pyproject.toml！\n请先下载并放置源码文件后再注入补丁。")
            return
        try:
            import train_local
            train_local.apply_compatibility_patches(trainer_dir)
            self.check_trainer_status()
            messagebox.showinfo("补丁就绪", "自愈兼容补丁注入成功！\n已放宽 PyTorch 限制并完成接口适配。")
        except Exception as e:
            messagebox.showerror("补丁失败", f"注入补丁异常:\n{e}")

    def download_or_clone_trainer(self):
        trainer_dir = Path("./KohakuMossTrainer").resolve()
        if (trainer_dir / "pyproject.toml").exists():
            if not messagebox.askyesno("已存在", "检测到 KohakuMossTrainer 已经存在且完整，是否强制重新拉取覆盖？"):
                return

        def worker():
            self.log(f"\n{'='*75}\n>>> 正在拉取 KohakuMossTrainer 训练器框架...\n{'='*75}\n")
            try:
                # 1. 核心关键清理：若目录存在且仅包含说明文件或未完成内容，必须先清除，防止 Git 报目标目录非空错误
                if trainer_dir.exists() and not (trainer_dir / "pyproject.toml").exists():
                    shutil.rmtree(trainer_dir, ignore_errors=True)

                # 2. 尝试使用 Git 克隆 (检测并附加常见代理参数)
                git_success = False
                git_cmds = [
                    ["git", "-c", "http.proxy=http://127.0.0.1:7890", "clone", "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git", str(trainer_dir)],
                    ["git", "clone", "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git", str(trainer_dir)]
                ]
                for cmd in git_cmds:
                    self.log(f"执行命令: {' '.join(cmd)}\n")
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
                    self.log(p.stdout + "\n")
                    if p.returncode == 0 and (trainer_dir / "pyproject.toml").exists():
                        git_success = True
                        break

                # 3. 容灾方案：若系统未装 Git 或 Git 网络超时，使用 Python urllib 自动拉取 ZIP 压缩包并扁平化解压
                if not git_success:
                    self.log("[提示] Git 拉取未成功，自动切换至 Python 直链下载 ZIP 容灾通道...\n")
                    zip_url = "https://github.com/KohakuBlueleaf/KohakuMossTrainer/archive/refs/heads/main.zip"
                    import urllib.request
                    import zipfile
                    import io
                    
                    proxies = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
                    try:
                        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
                        req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
                        resp = opener.open(req, timeout=30)
                        zip_bytes = resp.read()
                    except Exception:
                        req = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
                        resp = urllib.request.urlopen(req, timeout=30)
                        zip_bytes = resp.read()
                        
                    trainer_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                        # 扁平化解压，去除第一层 KohakuMossTrainer-main/ 避免套娃
                        for member in zf.infolist():
                            parts = Path(member.filename).parts
                            if len(parts) > 1:
                                target_path = trainer_dir / Path(*parts[1:])
                                if member.is_dir():
                                    target_path.mkdir(parents=True, exist_ok=True)
                                else:
                                    target_path.parent.mkdir(parents=True, exist_ok=True)
                                    with zf.open(member) as source, open(target_path, "wb") as target:
                                        shutil.copyfileobj(source, target)
                    if (trainer_dir / "pyproject.toml").exists():
                        git_success = True

                if (trainer_dir / "pyproject.toml").exists():
                    import train_local
                    train_local.apply_compatibility_patches(trainer_dir)
                    self.root.after(0, self.check_trainer_status)
                    self.log("[成功] 训练器框架已成功就绪并打入全部自愈补丁！\n")
                    self.root.after(0, lambda: messagebox.showinfo("成功", "训练器框架下载并补丁成功，已完全就绪！"))
                else:
                    self.log("[失败] 自动拉取失败。请点击【📂 打开训练器放置目录】，手动下载 ZIP 解压放入。\n")
                    self.root.after(0, lambda: messagebox.showerror("下载失败", "无法自动拉取训练器。建议点击【📂 打开训练器放置目录】手动下载 ZIP 解压放入。"))
            except Exception as e:
                self.log(f"[异常] {e}\n")

        threading.Thread(target=worker, daemon=True).start()

    def auto_scan_models(self, silent=False):
        """自动从 ComfyUI 模型库或项目 ./models 探测并预填模型路径。"""
        try:
            import train_local
            found = train_local.probe_local_models(self.var_comfy_models.get())
            all_found = True
            if found["transformer"]:
                self.var_transformer_path.set(str(found["transformer"]))
            else:
                all_found = False
            
            if found["text_encoder"]:
                self.var_text_encoder_path.set(str(found["text_encoder"]))
            else:
                all_found = False

            if found["vae"]:
                self.var_vae_path.set(str(found["vae"]))
            else:
                all_found = False

            if all_found:
                m_name = Path(str(found["transformer"])).name.lower()
                arch_tag = "SFT" if "sft" in m_name else "Base/Heal"
                self.var_models_status.set(f"✅ 全部自动就绪！[{arch_tag}]")
                self.save_local_config()
                if not silent:
                    messagebox.showinfo("扫描成功", f"已成功自动检索并匹配到全部三大核心模型权重！\n识别架构变体: {arch_tag}\n路径已自动填入输入框。")
            else:
                missing = []
                if not found["transformer"]: missing.append("DiT 底模")
                if not found["text_encoder"]: missing.append("Qwen3.5 文本编码器")
                if not found["vae"]: missing.append("FLUX.2 VAE")
                self.var_models_status.set(f"⚠️ 尚缺: {', '.join(missing)}")
                if not silent:
                    messagebox.showwarning("部分缺失", f"未找到以下组件: {', '.join(missing)}\n请点击【📂 打开放置目录】将模型放入对应文件夹，或手动选择文件。")
        except Exception as e:
            self.var_models_status.set("扫描异常")
            if not silent:
                messagebox.showerror("扫描失败", f"自动探测异常: {e}")

    def update_output_preview(self):
        name = self.var_lora_name.get().strip()
        if not name:
            name = "my_lora"
        out_root = self.var_output_dir.get().strip()
        preview_text = f"产出归档: {out_root}/{name}/ -> [{name}-comfy.safetensors, ...]"
        self.var_output_preview.set(preview_text)

    def build_right_panel(self, parent):
        outer = tk.Frame(parent, bg=self.card_bg, padx=10, pady=8)
        outer.pack(fill=tk.BOTH, expand=True)

        # 右侧上下可拖拽分栏：上方为实时 Loss 曲线与 KPI 看板，下方为终端实时日志
        self.right_paned = tk.PanedWindow(outer, orient=tk.VERTICAL, bg=self.bg_dark, sashwidth=6, sashrelief=tk.RAISED)
        self.right_paned.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # ==================== 上区：实时 Loss 折线图看板 ====================
        chart_card = tk.Frame(self.right_paned, bg=self.card_bg)
        self.right_paned.add(chart_card, minsize=250, stretch="always")

        chart_hdr = tk.Frame(chart_card, bg=self.card_bg)
        chart_hdr.pack(fill=tk.X, pady=(0, 5))

        tk.Label(chart_hdr, text="📈 实时训练 Loss 曲线与收敛监控", bg=self.card_bg, fg=self.accent_blue,
                 font=(self.font_family, 10, "bold")).pack(side=tk.LEFT)

        btn_clear_chart = tk.Button(chart_hdr, text="清空曲线", bg="#313244", fg=self.text_sub, relief=tk.FLAT,
                                    font=(self.font_family, 8), padx=7, pady=1, cursor="hand2",
                                    command=self.clear_loss_chart)
        btn_clear_chart.pack(side=tk.RIGHT, padx=(4, 0))

        btn_load_hist = tk.Button(chart_hdr, text="📊 加载当前 LoRA 历史曲线", bg="#313244", fg=self.accent_green,
                                  relief=tk.FLAT, font=(self.font_family, 8, "bold"), padx=8, pady=1, cursor="hand2",
                                  command=lambda: self.load_history_metrics(silent=False))
        btn_load_hist.pack(side=tk.RIGHT, padx=(4, 0))

        chk_ema = tk.Checkbutton(chart_hdr, text="平滑趋势 (EMA)", variable=self.var_show_ema,
                                 bg=self.card_bg, fg=self.accent_green, selectcolor=self.input_bg,
                                 activebackground=self.card_bg, font=(self.font_family, 8),
                                 command=self.redraw_loss_chart)
        chk_ema.pack(side=tk.RIGHT, padx=(0, 6))

        # 4 枚实时 KPI 卡片条
        kpi_bar = tk.Frame(chart_card, bg=self.card_bg)
        kpi_bar.pack(fill=tk.X, pady=(0, 6))
        for col_idx, (title, var, color) in enumerate([
            ("训练进度", self.var_kpi_step, self.accent_blue),
            ("实时 Loss / EMA", self.var_kpi_loss, self.accent_green),
            ("当前学习率 (LR)", self.var_kpi_lr, self.accent_purple),
            ("速度 / 预估剩余", self.var_kpi_eta, self.accent_yellow),
        ]):
            kpi_bar.columnconfigure(col_idx, weight=1)
            box = tk.Frame(kpi_bar, bg="#181825", highlightbackground="#313244", highlightthickness=1, padx=8, pady=4)
            box.grid(row=0, column=col_idx, sticky=tk.EW, padx=(0 if col_idx == 0 else 4, 0))
            tk.Label(box, text=title, bg="#181825", fg=self.text_sub, font=(self.font_family, 7)).pack(anchor="w")
            tk.Label(box, textvariable=var, bg="#181825", fg=color, font=("Consolas", 9, "bold")).pack(anchor="w")

        # 实时纤细进度条
        self.progress_canvas = tk.Canvas(chart_card, height=6, bg="#11111b", highlightthickness=0)
        self.progress_canvas.pack(fill=tk.X, pady=(0, 5))

        # 折线图主画布 (防抖响应式渲染)
        self.loss_canvas = tk.Canvas(chart_card, bg="#11111b", highlightbackground="#313244", highlightthickness=1)
        self.loss_canvas.pack(fill=tk.BOTH, expand=True)
        self.loss_canvas.bind("<Configure>", self._on_chart_resize)

        # ==================== 下区：实时日志终端控制台 ====================
        log_card = tk.Frame(self.right_paned, bg=self.card_bg)
        self.right_paned.add(log_card, minsize=180, stretch="always")

        header_box = tk.Frame(log_card, bg=self.card_bg)
        header_box.pack(fill=tk.X, pady=(4, 4))

        tk.Label(header_box, text="🖥️ 训练实时日志与终端控制台", bg=self.card_bg, fg=self.accent_blue,
                 font=(self.font_family, 10, "bold")).pack(side=tk.LEFT)

        btn_clear = tk.Button(header_box, text="清空控制台", bg="#313244", fg=self.text_sub, relief=tk.FLAT,
                              font=(self.font_family, 8), padx=7, pady=1, cursor="hand2", command=self.clear_console)
        btn_clear.pack(side=tk.RIGHT)

        console_wrap = tk.Frame(log_card, bg="#11111b")
        console_wrap.pack(fill=tk.BOTH, expand=True)

        console_scroll = ttk.Scrollbar(console_wrap, orient="vertical")
        self.console = tk.Text(console_wrap, bg="#11111b", fg="#a6e3a1", insertbackground="#a6e3a1",
                               font=("Consolas", 9), wrap=tk.WORD, yscrollcommand=console_scroll.set,
                               relief=tk.FLAT, padx=6, pady=4)
        console_scroll.config(command=self.console.yview)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ==================== 底部三按钮操作栏 ====================
        btn_bar = tk.Frame(outer, bg=self.card_bg)
        btn_bar.pack(fill=tk.X)

        self.btn_start = tk.Button(btn_bar, text="🚀 启动本地极速训练", bg="#2e7d32", fg="white",
                                   disabledforeground="#cdd6f4", activebackground="#388e3c",
                                   font=(self.font_family, 10, "bold"),
                                   relief=tk.FLAT, pady=8, cursor="hand2", command=self.start_training)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.btn_pause = tk.Button(btn_bar, text="⏸️ 保存当前进度并暂停", bg="#45475a", fg="white",
                                   disabledforeground="#9399b2", activebackground="#b45309",
                                   font=(self.font_family, 9, "bold"),
                                   relief=tk.FLAT, pady=8, state=tk.DISABLED, cursor="hand2",
                                   command=self.pause_and_save_training)
        self.btn_pause.pack(side=tk.LEFT, padx=4, ipadx=12)

        self.btn_stop = tk.Button(btn_bar, text="⏹ 强制终止", bg="#45475a", fg="white",
                                  disabledforeground="#9399b2", activebackground="#d32f2f",
                                  font=(self.font_family, 9, "bold"),
                                  relief=tk.FLAT, pady=8, state=tk.DISABLED, cursor="hand2", command=self.stop_training)
        self.btn_stop.pack(side=tk.RIGHT, padx=(5, 0), ipadx=12)

    # ==================== 实时 Loss 折线图数据与渲染引擎 ====================
    def _on_chart_resize(self, _event=None):
        """防抖处理画布尺寸变更，避免拖拽分栏时频繁重绘卡顿。"""
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
        """写入或更新指定 step 的 Loss 数据点并刷新 KPI 与折线图。"""
        if loss is None or step is None or step < 0:
            return
        try:
            loss_f = float(loss)
            if not (loss_f == loss_f) or loss_f > 1e6:  # 过滤 NaN / 异常无穷大
                return
        except Exception:
            return

        if step in self._loss_step_index:
            idx = self._loss_step_index[step]
            old_s, old_l, old_lr, old_t = self.loss_history[idx]
            self.loss_history[idx] = (
                step,
                loss_f,
                lr if lr is not None else old_lr,
                step_time if step_time is not None else old_t
            )
        else:
            self._loss_step_index[step] = len(self.loss_history)
            self.loss_history.append((step, loss_f, lr, step_time))
            # 保持按 step 升序
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
            target_steps = max(last_step, int(self.var_steps.get()))
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

        # 绘制顶部纤细进度条
        try:
            pw = self.progress_canvas.winfo_width()
            self.progress_canvas.delete("all")
            if pw > 10:
                self.progress_canvas.create_rectangle(0, 0, pw, 6, fill="#313244", outline="")
                bar_w = int(pw * (pct / 100.0))
                if bar_w > 0:
                    self.progress_canvas.create_rectangle(0, 0, bar_w, 6, fill=self.accent_green if pct >= 100 else self.accent_blue, outline="")
        except Exception:
            pass

        self.redraw_loss_chart(ema_vals=ema_vals)

    def redraw_loss_chart(self, ema_vals=None):
        """原生 Tkinter Canvas 暗色科技风实时 Loss 折线图渲染器 (零外部库依赖)。"""
        if not hasattr(self, "loss_canvas"):
            return
        c = self.loss_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 120 or h < 90:
            return

        pad_l, pad_r, pad_t, pad_b = 58, 22, 26, 30
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)

        # 绘制坐标系外框
        c.create_rectangle(pad_l, pad_t, w - pad_r, h - pad_b, outline="#313244", width=1)

        if not self.loss_history:
            c.create_text(
                w // 2, h // 2 - 8,
                text="等待训练 Loss 数据流输入...",
                fill=self.text_sub, font=(self.font_family, 10, "bold")
            )
            c.create_text(
                w // 2, h // 2 + 14,
                text="启动训练后将实时绘制收敛折线图；也可点击右上角【📊 加载当前 LoRA 历史曲线】复盘",
                fill="#6c7086", font=(self.font_family, 8)
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

        # 1. 绘制 5 级水平虚线网格与 Y 轴 Loss 刻度
        grid_rows = 4
        for i in range(grid_rows + 1):
            frac = i / grid_rows
            y = pad_t + plot_h * frac
            val = y_max - (y_max - y_min) * frac
            if 0 < i < grid_rows:
                c.create_line(pad_l, y, w - pad_r, y, fill="#24273a", dash=(3, 4))
            c.create_text(pad_l - 8, y, text=f"{val:.3f}", anchor="e", fill="#9399b2", font=("Consolas", 8))

        # 2. 绘制垂直虚线网格与 X 轴 Step 刻度
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

        # 3. 绘制原始 Loss 折线 (亮青蓝色)
        raw_coords = []
        for s, l_val in zip(steps, losses):
            px, py = to_xy(s, l_val)
            raw_coords.extend([px, py])

        if len(raw_coords) >= 4:
            c.create_line(*raw_coords, fill="#587bb8" if self.var_show_ema.get() else self.accent_blue, width=1.6)
        elif len(raw_coords) == 2:
            px, py = raw_coords
            c.create_oval(px - 3, py - 3, px + 3, py + 3, fill=self.accent_blue, outline="")

        # 4. 绘制平滑趋势线 EMA (高对比度翡翠绿粗线)
        if self.var_show_ema.get() and len(ema_vals) >= 2:
            ema_coords = []
            for s, e_val in zip(steps, ema_vals):
                px, py = to_xy(s, e_val)
                ema_coords.extend([px, py])
            c.create_line(*ema_coords, fill=self.accent_green, width=2.4, smooth=True)

        # 5. 绘制最新点的发光锚点与悬浮标签
        lx, ly = to_xy(steps[-1], ema_vals[-1] if (self.var_show_ema.get() and ema_vals) else losses[-1])
        c.create_oval(lx - 6, ly - 6, lx + 6, ly + 6, fill="", outline=self.accent_green, width=1.5)
        c.create_oval(lx - 3, ly - 3, lx + 3, ly + 3, fill=self.accent_green, outline="#11111b", width=1)

        badge_text = f"Step {steps[-1]} | Loss {losses[-1]:.4f}"
        bx = min(w - pad_r - 78, max(pad_l + 78, lx - 10))
        by = ly - 16 if ly > pad_t + 28 else ly + 18
        c.create_rectangle(bx - 74, by - 10, bx + 74, by + 10, fill="#1e1e2e", outline=self.accent_green, width=1)
        c.create_text(bx, by, text=badge_text, fill=self.text_main, font=("Consolas", 8, "bold"))

        # 6. 右上角图例
        leg_x = w - pad_r - 6
        leg_y = pad_t - 12
        if self.var_show_ema.get():
            c.create_text(leg_x, leg_y, text="━━ 平滑趋势 (EMA)", anchor="e", fill=self.accent_green, font=(self.font_family, 8, "bold"))
            c.create_text(leg_x - 118, leg_y, text="── 原始 Loss", anchor="e", fill="#749be0", font=(self.font_family, 8))
        else:
            c.create_text(leg_x, leg_y, text="── 原始 Loss", anchor="e", fill=self.accent_blue, font=(self.font_family, 8, "bold"))
        c.create_text(pad_l, leg_y, text="Loss", anchor="w", fill="#9399b2", font=("Consolas", 8, "bold"))

    def load_history_metrics(self, silent: bool = False):
        """读取当前 LoRA 的 metrics.jsonl 历史指标，立即渲染完整收敛曲线。"""
        lora_name = self.var_lora_name.get().strip() or "my_lora"
        metrics_path = Path("./KohakuMossTrainer/runs/moss") / lora_name / "metrics.jsonl"
        if not metrics_path.exists():
            if not silent:
                messagebox.showinfo("未找到历史曲线", f"尚未在运行目录中找到 [{lora_name}] 的 metrics.jsonl 指标记录。\n启动训练后折线图将实时自动绘制！")
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
                            redraw=False
                        )
                        count += 1
                except Exception:
                    pass
            self._update_kpi_and_chart()
            if not silent and count > 0:
                self.log(f"[监控面板] 已加载 [{lora_name}] 历史收敛曲线，共 {count} 个记录点。\n")
        except Exception as e:
            if not silent:
                messagebox.showerror("加载曲线失败", f"读取指标文件异常: {e}")

    def _start_metrics_polling(self, reset_chart: bool = True):
        """训练启动时激活每 2 秒一次的 metrics.jsonl 增量轮询。"""
        self._stop_metrics_polling()
        if reset_chart:
            self.clear_loss_chart()
        else:
            self.load_history_metrics(silent=True)

        def _poll():
            if not self.is_training:
                return
            try:
                lora_name = self.var_lora_name.get().strip() or "my_lora"
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
                                        redraw=False
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

    def _stop_metrics_polling(self):
        if self._metrics_poll_job is not None:
            try:
                self.root.after_cancel(self._metrics_poll_job)
            except Exception:
                pass
            self._metrics_poll_job = None

    def _try_parse_stdout_metrics(self, line: str):
        """从实时终端日志中兜底捕获 step / loss / lr，保证尚未 flush jsonl 时也能实时绘点。"""
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
                redraw=True
            )

    # 续炼与进度恢复辅助方法
    def browse_existing_lora_file(self):
        init_dir = self.var_output_dir.get() if os.path.exists(self.var_output_dir.get()) else "."
        selected = filedialog.askopenfilename(
            initialdir=init_dir,
            title="选择已训练结束的 LoRA 权重文件 (.safetensors) 以继续训练",
            filetypes=[("Safetensors LoRA 权重", "*.safetensors"), ("所有文件", "*.*")]
        )
        if selected:
            self.var_resume_path.set(selected)
            self.var_train_mode_label.set(self.train_mode_rev_map["init_from"])
            self.refresh_resume_status(align_hyperparams=True)
            self.save_local_config()

    def browse_resume_ckpt_dir(self):
        trainer_runs = Path("./KohakuMossTrainer/runs/moss").resolve()
        init_dir = str(trainer_runs) if trainer_runs.exists() else "."
        selected = filedialog.askdirectory(
            initialdir=init_dir,
            title="选择训练断点目录 (如 step_00000310 或 checkpoints 目录)"
        )
        if selected:
            self.var_resume_path.set(selected)
            self.var_train_mode_label.set(self.train_mode_rev_map["resume"])
            self.refresh_resume_status(align_hyperparams=True)
            self.save_local_config()

    def refresh_resume_status(self, align_hyperparams=False):
        """实时探测当前 LoRA 名称或自选路径下的历史断点与已完工 .safetensors 权重。"""
        try:
            import train_local
            mode_code = self.train_mode_map.get(self.var_train_mode_label.get(), "fresh")
            lora_name = self.var_lora_name.get().strip() or "my_lora"
            trainer_dir = Path("./KohakuMossTrainer").resolve()
            out_dir = Path(self.var_output_dir.get()).resolve()
            custom_p = self.var_resume_path.get().strip()

            ckpts = train_local.list_lora_checkpoints(trainer_dir, lora_name, custom_p if (mode_code == "resume" and custom_p) else "")

            # 查找是否有已导出的 .safetensors
            existing_sf = None
            if custom_p and os.path.isfile(custom_p) and custom_p.lower().endswith(".safetensors"):
                existing_sf = Path(custom_p)
            else:
                lora_out_folder = out_dir / lora_name
                if lora_out_folder.exists():
                    for cand_name in [f"{lora_name}-comfy.safetensors", f"{lora_name}-lycoris.safetensors", f"{lora_name}-trainable.safetensors"]:
                        if (lora_out_folder / cand_name).exists():
                            existing_sf = lora_out_folder / cand_name
                            break

            if mode_code == "fresh":
                if ckpts:
                    last_step, last_dir = ckpts[-1]
                    self.var_resume_status.set(
                        f"💡 检测到 [{lora_name}] 已有历史进度 ({last_dir.name}, 已完成 {last_step} 步)。"
                        f"当前为【全新训练】(启动时将自动备份旧断点并从0开始)；若想接着练请切换上方模式为【断点续炼】或【基于已有 LoRA 继续训练】。"
                    )
                elif existing_sf:
                    info = train_local.inspect_safetensors_lora(existing_sf)
                    self.var_resume_status.set(
                        f"💡 检测到 [{lora_name}] 已有完工权重 {existing_sf.name} (Rank={info.get('rank')})。"
                        f"当前为【全新训练】；若想在该 LoRA 基础上加练，请切换为【基于已有 LoRA 继续训练】。"
                    )
                else:
                    self.var_resume_status.set("模式状态: 全新训练 (将从 Step 0 全新开始训练)")

            elif mode_code == "resume":
                if ckpts:
                    last_step, last_dir = ckpts[-1]
                    t_sf = last_dir / "trainable.safetensors"
                    r_str = ""
                    if t_sf.exists():
                        info = train_local.inspect_safetensors_lora(t_sf)
                        if info.get("rank"):
                            r_str = f", Rank={info['rank']}"
                            if align_hyperparams:
                                self.var_rank.set(info["rank"])
                                if info.get("alpha"):
                                    self.var_alpha.set(info["alpha"])
                    self.var_resume_status.set(
                        f"✅ 已锁定断点: {last_dir.name} (已训 {last_step} 步{r_str})。"
                        f"若下方【核心步数】> {last_step} 则补齐剩余步数；若 <= {last_step} (如填 200) 将自动作为【追加步数】顺延训练至 {last_step} + N 步！"
                    )
                elif existing_sf:
                    info = train_local.inspect_safetensors_lora(existing_sf)
                    self.var_resume_status.set(
                        f"✅ 未找到中间断点目录，已自动回退匹配完工权重: {existing_sf.name} (Rank={info.get('rank')})，将在此权重基础上继续训练！"
                    )
                else:
                    self.var_resume_status.set(
                        f"⚠️ 未检测到 [{lora_name}] 的历史断点！请点击【📂 选断点目录】或【📁 选已训 LoRA】，或先切换回【全新训练】。"
                    )

            elif mode_code == "init_from":
                target_sf = existing_sf
                if not target_sf and ckpts:
                    target_sf = ckpts[-1][1] / "trainable.safetensors"
                if target_sf and target_sf.exists():
                    info = train_local.inspect_safetensors_lora(target_sf)
                    if info.get("rank"):
                        self.var_rank.set(info["rank"])
                    if info.get("alpha"):
                        self.var_alpha.set(info["alpha"])
                    step_tag = f"原步数 {info['step']}, " if info.get("step") else ""
                    self.var_resume_status.set(
                        f"✅ 已加载基底 LoRA: {target_sf.name} ({step_tag}已自动对齐 Rank={info.get('rank')}, Alpha={info.get('alpha')})。"
                        f"将以全新学习率周期在该权重基础上继续训练下方设定的【核心步数】！"
                    )
                else:
                    self.var_resume_status.set(
                        "⚠️ 请点击上方【📁 选已训 LoRA (.safetensors)】选择一个已训练完成的 LoRA 权重文件作为继续训练的起点！"
                    )
        except Exception as e:
            self.var_resume_status.set(f"进度检测提示: {e}")

    # 路径浏览与检测
    def browse_data_dir(self):
        selected = filedialog.askdirectory(initialdir=self.var_data_dir.get(), title="选择训练数据集目录")
        if selected:
            self.var_data_dir.set(selected)
            self.update_dataset_stats()
            self.recalculate_math()
            self.save_local_config()

    def browse_comfy_dir(self):
        selected = filedialog.askdirectory(initialdir=self.var_comfy_models.get(), title="选择 ComfyUI 模型库目录")
        if selected:
            self.var_comfy_models.set(selected)
            self.auto_scan_models(silent=True)
            self.save_local_config()

    def browse_output_dir(self):
        selected = filedialog.askdirectory(initialdir=self.var_output_dir.get(), title="选择权重产出保存根目录")
        if selected:
            self.var_output_dir.set(selected)
            self.update_output_preview()
            self.save_local_config()

    def update_dataset_stats(self):
        dpath = Path(self.var_data_dir.get())
        if dpath.exists():
            img_count = len([f for f in dpath.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
            txt_count = len([f for f in dpath.iterdir() if f.suffix.lower() == ".txt"])
            self.var_dataset_stat.set(f"已就绪：发现 {img_count} 张图像，{txt_count} 份配套标注")
        else:
            self.var_dataset_stat.set("⚠️ 目录不存在，请检查路径")

    def recalculate_math(self):
        dpath = Path(self.var_data_dir.get())
        img_count = 0
        if dpath.exists():
            img_count = len([f for f in dpath.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        try:
            rep = self.var_repeats.get()
            ep = self.var_epochs.get()
        except tk.TclError:
            self.var_calc_stat.set("换算：请输入有效的正整数重复次数与轮数")
            return
        total_exposures = img_count * rep * ep
        recommended_steps = max(1, total_exposures // 4)
        self.var_calc_stat.set(f"换算：{img_count}图 × {rep}重复 × {ep}轮 = {total_exposures}次曝光 (等效推荐约 {recommended_steps} 步)")

    def align_steps_by_formula(self):
        dpath = Path(self.var_data_dir.get())
        if not dpath.exists():
            return
        try:
            rep = self.var_repeats.get()
            ep = self.var_epochs.get()
        except tk.TclError:
            messagebox.showwarning("参数输入提示", "请先填写有效的正整数【单图重复】和【训练轮数】。")
            return
        img_count = len([f for f in dpath.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        recommended = max(1, (img_count * rep * ep) // 4)
        self.var_steps.set(recommended)
        self.log(f"[提示] 步数已根据换算公式对齐为: {recommended} 步\n")

    # 预设模板应用
    def apply_style_preset(self):
        self.var_repeats.set(10)
        self.var_epochs.set(6)
        self.var_rank.set(32)
        self.var_alpha.set(32.0)
        self.var_lr.set("1e-4")
        self.var_lora_name.set("my_art_style")
        self.var_trigger_word.set("in the style of my_art_style")
        self.align_steps_by_formula()
        self.update_output_preview()
        messagebox.showinfo("模板切换", "已成功加载【艺术画风模板】：\nLoRA名称: my_art_style\n触发词: in the style of my_art_style\nRepeats=10, Epochs=6, Rank=32, Alpha=32.0, LR=1e-4")

    def apply_character_preset(self):
        self.var_repeats.set(6)
        self.var_epochs.set(3)
        self.var_rank.set(32)
        self.var_alpha.set(16.0)
        self.var_lr.set("8e-5")
        self.var_lora_name.set("my_character")
        self.var_trigger_word.set("my_character_char")
        self.align_steps_by_formula()
        self.update_output_preview()
        messagebox.showinfo("模板切换", "已成功加载【人物角色模板】：\nLoRA名称: my_character\n触发词: my_character_char\nRepeats=6, Epochs=3, Rank=32, Alpha=16.0, LR=8e-5")

    def clear_console(self):
        self.console.delete("1.0", tk.END)

    def log(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        if self.is_training:
            try:
                self._try_parse_stdout_metrics(text)
            except Exception:
                pass

    # 训练执行控制
    def start_training(self):
        if self.is_training:
            return

        # 检查训练器框架就绪状态
        trainer_dir = Path("./KohakuMossTrainer").resolve()
        if not (trainer_dir / "pyproject.toml").exists():
            messagebox.showwarning("训练器未就绪", "检测到 KohakuMossTrainer 训练器尚未就绪！\n系统将在启动时自动尝试从 GitHub 克隆。若您网络无法访问 GitHub，请先点击【📂 打开训练器放置目录】手动解压源码。")

        # 检查数据集
        dpath = Path(self.var_data_dir.get())
        if not dpath.exists():
            messagebox.showerror("错误", f"训练数据集路径不存在: {dpath}")
            return

        # 数值型超参数合法性校验，防御空输入引发 TclError 崩溃
        try:
            repeats_val = int(self.var_repeats.get())
            epochs_val = int(self.var_epochs.get())
            steps_val = int(self.var_steps.get())
            rank_val = int(self.var_rank.get())
            alpha_val = float(self.var_alpha.get())
            lr_val = float(self.var_lr.get().strip())
            save_every_val = int(self.var_save_every.get())
            if repeats_val < 1 or epochs_val < 1 or steps_val < 1 or rank_val < 1 or lr_val <= 0:
                raise ValueError("超参数必须为正数")
        except Exception as e:
            messagebox.showerror("参数格式错误", f"请检查核心超参数是否填写了合法数字：\n{e}")
            return

        lora_name_clean = self.var_lora_name.get().strip()
        if not lora_name_clean:
            lora_name_clean = "my_lora"

        mode_code = self.train_mode_map.get(self.var_train_mode_label.get(), "fresh")
        custom_resume_p = self.var_resume_path.get().strip()

        # init_from 模式前置校验：确保确实存在可用的基底 LoRA 文件，避免空跑环境检查后报错
        if mode_code == "init_from":
            has_valid_init = bool(custom_resume_p and os.path.exists(custom_resume_p))
            if not has_valid_init:
                out_folder = Path(self.var_output_dir.get()) / lora_name_clean
                if out_folder.exists() and any(out_folder.glob("*.safetensors")):
                    has_valid_init = True
                elif (trainer_dir / "runs" / "moss" / lora_name_clean).exists() and any((trainer_dir / "runs" / "moss" / lora_name_clean).rglob("*.safetensors")):
                    has_valid_init = True
            if not has_valid_init:
                messagebox.showerror(
                    "缺少基底 LoRA 文件",
                    "当前处于【基于已有 LoRA 继续训练】模式，但未检测到可用的 .safetensors 文件！\n\n"
                    "请点击【📁 选已训 LoRA (.safetensors)】手动选择一个已训练完成的权重文件，或切换回【全新训练】。"
                )
                return

        # 若用户在 fresh 模式下发现已有历史断点，友好询问是要全新覆盖还是断点续炼
        if mode_code == "fresh":
            try:
                import train_local
                existing_ckpts = train_local.list_lora_checkpoints(trainer_dir, lora_name_clean)
                if existing_ckpts:
                    last_s, last_d = existing_ckpts[-1]
                    ans = messagebox.askyesnocancel(
                        "发现已有训练进度",
                        f"检测到【{lora_name_clean}】已有历史断点 {last_d.name} (已完成 {last_s} 步)！\n\n"
                        f"• 点击【是 (Yes)】：自动切换为【断点续炼】接着第 {last_s} 步继续训练（推荐）；\n"
                        f"• 点击【否 (No)】：备份旧断点，从第 0 步开始【全新训练】；\n"
                        f"• 点击【取消 (Cancel)】：返回界面手动调整。"
                    )
                    if ans is None:
                        return
                    elif ans is True:
                        mode_code = "resume"
                        self.var_train_mode_label.set(self.train_mode_rev_map["resume"])
                        self.refresh_resume_status(align_hyperparams=True)
            except Exception:
                pass

        self.save_local_config()
        self.pause_requested = False

        cmd = [
            _resolve_cli_python(), "-u", "train_local.py",
            "--data_dir", self.var_data_dir.get(),
            "--comfy_models_dir", self.var_comfy_models.get(),
            "--output_dir", self.var_output_dir.get(),
            "--lora_name", lora_name_clean,
            "--trigger_word", self.var_trigger_word.get().strip(),
            "--repeats", str(self.var_repeats.get()),
            "--epochs", str(self.var_epochs.get()),
            "--steps", str(self.var_steps.get()),
            "--rank", str(self.var_rank.get()),
            "--alpha", str(self.var_alpha.get()),
            "--lr", str(self.var_lr.get()),
            "--precision", self.var_precision.get(),
            "--vram_preset", self.var_vram_preset.get(),
            "--resolution", str(self.var_resolution.get()),
            "--variant", "auto",
            "--train_mode", mode_code,
            "--save_every", str(self.var_save_every.get()),
            "--text_device", self.var_text_device.get(),
            "--vae_device", self.var_vae_device.get()
        ]

        if mode_code == "resume" and custom_resume_p:
            if os.path.isfile(custom_resume_p) and custom_resume_p.lower().endswith(".safetensors"):
                # 用户在 resume 模式下选了单个 .safetensors 文件，智能路由至 init_from
                idx = cmd.index("--train_mode")
                cmd[idx + 1] = "init_from"
                cmd.extend(["--init_from", custom_resume_p])
            else:
                cmd.extend(["--resume_from", custom_resume_p])
        elif mode_code == "init_from" and custom_resume_p:
            cmd.extend(["--init_from", custom_resume_p])

        if self.var_transformer_path.get().strip():
            cmd.extend(["--transformer_path", self.var_transformer_path.get().strip()])
        if self.var_text_encoder_path.get().strip():
            cmd.extend(["--text_encoder_path", self.var_text_encoder_path.get().strip()])
        if self.var_vae_path.get().strip():
            cmd.extend(["--vae_path", self.var_vae_path.get().strip()])

        if self.var_copy_comfy.get():
            cmd.append("--copy_to_comfy_loras")

        self.log(f"\n{'='*75}\n>>> 正在启动训练引擎 (模式: {mode_code.upper()})...\n执行命令: {' '.join(cmd)}\n{'='*75}\n")
        self.is_training = True
        self.btn_start.config(state=tk.DISABLED, bg="#45475a")
        self.btn_pause.config(state=tk.NORMAL, text="⏸️ 保存当前进度并暂停", bg="#d97706")
        self.btn_stop.config(state=tk.NORMAL, bg="#c62828")

        # 启动实时 Loss 曲线轮询 (若是 resume 模式则保留并接续历史曲线)
        self._start_metrics_polling(reset_chart=(mode_code != "resume"))

        threading.Thread(target=self.run_subprocess_worker, args=(cmd,), daemon=True).start()

    def run_subprocess_worker(self, cmd):
        project_root = Path(__file__).resolve().parent
        try:
            self.current_process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                creationflags=_NO_WINDOW
            )

            for line in iter(self.current_process.stdout.readline, ""):
                if line:
                    self.root.after(0, self.log, line)

            self.current_process.stdout.close()
            return_code = self.current_process.wait()
            self.root.after(0, self.training_finished, return_code)
        except Exception as e:
            self.root.after(0, self.log, f"\n[异常退出] 执行失败: {e}\n")
            self.root.after(0, self.training_finished, -1)

    def pause_and_save_training(self):
        """在当前训练步完成时，立即触发完整 Checkpoint 保存、导出当前版本 LoRA 并安全暂停退出。"""
        if not self.current_process or not self.is_training:
            return
        workspace_dir = Path("./workspace").resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)
        pause_flag_file = workspace_dir / ".pause_request"
        try:
            pause_flag_file.write_text("pause", encoding="utf-8")
            self.pause_requested = True
            self.btn_pause.config(state=tk.DISABLED, text="⏳ 正在保存当前步...", bg="#45475a")
            self.log(
                f"\n{'='*75}\n"
                f">>> [保存并暂停] 已向训练内核发送即时存盘指令！\n"
                f">>> 系统将在当前训练步（约 5~8 秒内）完成后，自动保存完整断点并导出截止至当前的 LoRA 权重，请稍候...\n"
                f"{'='*75}\n"
            )
        except Exception as e:
            messagebox.showerror("暂停异常", f"无法写入暂停信号文件: {e}")

    def stop_training(self):
        if self.current_process and self.is_training:
            reply = messagebox.askyesno(
                "确认强制终止",
                "确定要立刻强制杀死训练进程吗？\n\n"
                "💡 提示：如果您希望【保存当前正在训练的步数进度】以便稍后继续训练，请点击【否】改用旁边的橙色按钮【⏸️ 保存当前进度并暂停】！"
            )
            if reply:
                pid = self.current_process.pid
                self.log(f"\n[用户操作] 正在强制级联终止训练进程树 (PID: {pid})...\n")
                try:
                    # Windows 下使用 taskkill /F /T 强杀目标进程及其派生的整个子孙进程树 (彻底清理 kogine 和主训练 python 进程)
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=_NO_WINDOW,
                        check=False
                    )
                except Exception as e:
                    self.log(f"taskkill 异常: {e}\n")

                try:
                    self.current_process.kill()
                except Exception:
                    pass

                # 兜底回收强杀前已存盘的阶段性 checkpoint 权重至 output/<lora_name>
                try:
                    import train_local
                    lora_name = self.var_lora_name.get().strip() or "my_lora"
                    dest_dir = Path(self.var_output_dir.get()) / lora_name
                    files, latest_s = train_local.export_and_sync_weights(
                        Path("./KohakuMossTrainer").resolve(),
                        lora_name,
                        dest_dir,
                        self.var_comfy_models.get(),
                        self.var_copy_comfy.get(),
                        quiet=True
                    )
                    if latest_s > 0:
                        self.log(f"[自动抢救] 已将强制终止前最新保存的第 {latest_s} 步权重同步至: {dest_dir}\n")
                except Exception:
                    pass

                self.log("[用户操作] 训练进程树已全部强制中断，GPU 与内存资源已安全释放。\n")
                self.root.after(0, self.training_finished, -9)

    def training_finished(self, return_code):
        self.is_training = False
        self.current_process = None
        self._stop_metrics_polling()
        self.load_history_metrics(silent=True)
        self.btn_start.config(state=tk.NORMAL, bg="#2e7d32")
        self.btn_pause.config(state=tk.DISABLED, text="⏸️ 保存当前进度并暂停", bg="#45475a")
        self.btn_stop.config(state=tk.DISABLED, bg="#45475a")
        self.refresh_resume_status()

        if return_code == 0:
            lora_name = self.var_lora_name.get().strip() or "my_lora"
            dest_dir = Path(self.var_output_dir.get()) / lora_name
            if self.pause_requested:
                self.pause_requested = False
                # 自动将模式切换为【断点续炼】，方便用户下次打开直接点击启动继续训练
                self.var_train_mode_label.set(self.train_mode_rev_map["resume"])
                self.refresh_resume_status()
                self.save_local_config()
                self.log(f"\n{'='*75}\n>>> [已保存并暂停] 截止至当前步数的断点与 LoRA 权重已完整归档至: {dest_dir} <<<\n{'='*75}\n")
                messagebox.showinfo(
                    "已保存当前进度并暂停",
                    f"训练已安全暂停，截止至当前步数的 LoRA 权重已导出至：\n{dest_dir}\n\n"
                    f"• 您现在可以直接在 ComfyUI 中测试当前版本的生图效果；\n"
                    f"• 界面已自动为您切换为【断点续炼】模式，之后随时点击【🚀 启动本地极速训练】即可无缝接着当前步数继续！"
                )
            else:
                self.log(f"\n{'='*75}\n>>> [成功] 本地训练已圆满完成！权重已归档至专属子文件夹: {dest_dir} <<<\n{'='*75}\n")
                messagebox.showinfo("训练成功", f"LoRA 训练已圆满完成！\n专属产出文件夹:\n{dest_dir}\n包含: {lora_name}-comfy.safetensors 及各阶段 checkpoints。")
        else:
            self.pause_requested = False
            self.log(f"\n>>> [提示] 训练进程已结束，退出码: {return_code} <<<\n")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ModernLocalTrainerGUI(root)
        root.mainloop()
    except Exception as _fatal_err:
        import traceback
        _tb = traceback.format_exc()
        try:
            _crash_dir = Path("./workspace").resolve()
            _crash_dir.mkdir(parents=True, exist_ok=True)
            (_crash_dir / "gui_crash.log").write_text(_tb, encoding="utf-8")
        except Exception:
            pass
        try:
            _err_root = tk.Tk()
            _err_root.withdraw()
            messagebox.showerror("启动异常", f"训练器 GUI 启动失败：\n{_fatal_err}\n\n详细日志已保存至 workspace/gui_crash.log")
            _err_root.destroy()
        except Exception:
            pass
        raise
