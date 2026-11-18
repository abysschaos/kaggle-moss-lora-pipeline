import os
import sys
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 启用 Windows 高 DPI 适配，防止界面模糊
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

class ModernLocalTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMOSS MOSS-Image 本地 LoRA 训练引擎控制台")
        
        # 窗口大小与屏幕居中适配 (适配 1080P 125%/150% 笔记本屏幕)
        win_w, win_h = 1080, 760
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(960, 680)

        # 调色盘与字体定义 (现代深色主题)
        self.bg_dark = "#181825"
        self.card_bg = "#1e1e2e"
        self.input_bg = "#313244"
        self.accent_blue = "#89b4fa"
        self.accent_green = "#a6e3a1"
        self.accent_red = "#f38ba8"
        self.accent_yellow = "#f9e2af"
        self.text_main = "#cdd6f4"
        self.text_sub = "#a6adc8"
        self.font_family = "Microsoft YaHei UI"

        self.root.configure(bg=self.bg_dark)
        self.setup_styles()

        # 核心变量初始化
        self.current_process = None
        self.is_training = False

        self.var_lora_name = tk.StringVar(value="my_lora")
        self.var_trigger_word = tk.StringVar(value="")
        self.var_data_dir = tk.StringVar(value="C:/Users/白音华/Pictures/ness")
        self.var_comfy_models = tk.StringVar(value="C:/dowmload idm/ComfyUI-aki-v3.2(1)/ComfyUI-aki-v3.2/ComfyUI/models")
        self.var_output_dir = tk.StringVar(value=str(Path("./output").resolve()))
        self.var_copy_comfy = tk.BooleanVar(value=True)

        self.var_repeats = tk.IntVar(value=10)
        self.var_epochs = tk.IntVar(value=6)
        self.var_steps = tk.IntVar(value=600)
        self.var_rank = tk.IntVar(value=32)
        self.var_alpha = tk.DoubleVar(value=32.0)
        self.var_lr = tk.StringVar(value="1e-4")
        self.var_precision = tk.StringVar(value="bf16")
        self.var_text_device = tk.StringVar(value="cpu")
        self.var_vae_device = tk.StringVar(value="cuda")

        self.var_dataset_stat = tk.StringVar(value="未检测")
        self.var_calc_stat = tk.StringVar(value="曝光计算：就绪")
        self.var_output_preview = tk.StringVar(value="")

        self.build_ui()
        self.update_dataset_stats()
        self.recalculate_math()
        self.update_output_preview()

        # 监听名称变动实时刷新产出路径预览
        self.var_lora_name.trace_add("write", lambda *args: self.update_output_preview())

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.bg_dark, foreground=self.text_main, font=(self.font_family, 9))
        style.configure("Card.TFrame", background=self.card_bg)
        style.configure("TLabel", background=self.card_bg, foreground=self.text_main, font=(self.font_family, 9))
        style.configure("Sub.TLabel", background=self.card_bg, foreground=self.text_sub, font=(self.font_family, 8))
        style.configure("Header.TLabel", background=self.card_bg, foreground=self.accent_blue, font=(self.font_family, 11, "bold"))
        style.configure("TCheckbutton", background=self.card_bg, foreground=self.text_main, font=(self.font_family, 9))
        style.configure("TCombobox", fieldbackground=self.input_bg, background=self.card_bg, foreground=self.text_main)

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

        # 2. 主体左右分栏
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.bg_dark, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 左侧控制面板 (带滚动支持，防止在小屏幕笔记本身上被挤压)
        left_outer = tk.Frame(main_paned, bg=self.card_bg, width=540)
        main_paned.add(left_outer, minsize=480)

        # 右侧实时监控与日志控制台
        right_container = tk.Frame(main_paned, bg=self.card_bg)
        main_paned.add(right_container, minsize=420)

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

    def build_left_scrollable(self, parent):
        canvas = tk.Canvas(parent, bg=self.card_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.card_bg)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 绑定鼠标滚轮
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side="right", fill="y", pady=8)

        self.build_left_controls(scrollable_frame)

    def build_left_controls(self, parent):
        # 1. 快捷预设按钮栏
        preset_frame = tk.Frame(parent, bg=self.card_bg)
        preset_frame.pack(fill=tk.X, pady=(0, 6))
        
        tk.Label(preset_frame, text="快速模板预设：", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9, "bold")).pack(side=tk.LEFT, padx=(4, 6), pady=6)

        btn_style = tk.Button(preset_frame, text="🎨 艺术画风模板", bg="#45475a", fg=self.text_main,
                              activebackground=self.accent_blue, font=(self.font_family, 8, "bold"),
                              relief=tk.FLAT, padx=8, pady=3, cursor="hand2", command=self.apply_style_preset)
        btn_style.pack(side=tk.LEFT, padx=4)

        btn_char = tk.Button(preset_frame, text="👤 人物角色模板", bg="#45475a", fg=self.text_main,
                             activebackground=self.accent_blue, font=(self.font_family, 8, "bold"),
                             relief=tk.FLAT, padx=8, pady=3, cursor="hand2", command=self.apply_character_preset)
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

        tk.Label(name_card, text="说明: 若填写触发词，系统自动前置注入每张图片标注；留空则直接使用原标注。",
                 bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8)).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(1, 2))

        lbl_preview = tk.Label(name_card, textvariable=self.var_output_preview, bg=self.card_bg, fg=self.accent_green, font=(self.font_family, 8))
        lbl_preview.grid(row=3, column=0, columnspan=2, sticky=tk.W)

        # 3. 资产与路径卡片
        path_card = tk.LabelFrame(parent, text=" 📂 资产与目录设置 ", bg=self.card_bg, fg=self.accent_blue,
                                  font=(self.font_family, 9, "bold"), padx=10, pady=6)
        path_card.pack(fill=tk.X, pady=(0, 6))

        self.create_path_row(path_card, "训练图片目录:", self.var_data_dir, self.browse_data_dir, 0)
        stat_lbl = tk.Label(path_card, textvariable=self.var_dataset_stat, bg=self.card_bg, fg=self.accent_yellow, font=(self.font_family, 8))
        stat_lbl.grid(row=1, column=1, sticky=tk.W, pady=(0, 3))

        self.create_path_row(path_card, "ComfyUI模型库:", self.var_comfy_models, self.browse_comfy_dir, 2)
        self.create_path_row(path_card, "结果存放根目录:", self.var_output_dir, self.browse_output_dir, 3)

        chk_comfy = tk.Checkbutton(path_card, text="训练完成后自动复制权重至 ComfyUI models/loras 专属子目录",
                                   variable=self.var_copy_comfy, bg=self.card_bg, fg=self.text_main,
                                   selectcolor=self.input_bg, activebackground=self.card_bg, font=(self.font_family, 8))
        chk_comfy.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(3, 1))

        # 4. 超参数配置卡片
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

        dev_f = tk.Frame(hp_card, bg=self.card_bg)
        dev_f.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(2, 4))
        tk.Label(dev_f, text="文本提取设备:", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").pack(side=tk.LEFT)
        cb_txt_dev = ttk.Combobox(dev_f, textvariable=self.var_text_device, values=["cpu", "cuda"], width=7, state="readonly")
        cb_txt_dev.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(dev_f, text="(8GB 显卡请锁定 CPU 避免 OOM)", bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8)).pack(side=tk.LEFT)

    def create_path_row(self, parent, label_text, var, browse_cmd, row_idx):
        tk.Label(parent, text=label_text, bg=self.card_bg, fg=self.text_main, width=14, anchor="w").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        ent = tk.Entry(parent, textvariable=var, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main)
        ent.grid(row=row_idx, column=1, sticky=tk.EW, padx=4, pady=2)
        parent.columnconfigure(1, weight=1)
        btn = tk.Button(parent, text="浏览...", bg="#45475a", fg=self.text_main, relief=tk.FLAT, font=(self.font_family, 8),
                        cursor="hand2", command=browse_cmd)
        btn.grid(row=row_idx, column=2, padx=2, pady=2)

    def update_output_preview(self):
        name = self.var_lora_name.get().strip()
        if not name:
            name = "my_lora"
        out_root = self.var_output_dir.get().strip()
        preview_text = f"产出归档: {out_root}/{name}/ -> [{name}-comfy.safetensors, ...]"
        self.var_output_preview.set(preview_text)

    def build_right_panel(self, parent):
        card = tk.Frame(parent, bg=self.card_bg, padx=10, pady=8)
        card.pack(fill=tk.BOTH, expand=True)

        header_box = tk.Frame(card, bg=self.card_bg)
        header_box.pack(fill=tk.X, pady=(0, 4))

        tk.Label(header_box, text="🖥️ 训练实时日志与终端控制台", bg=self.card_bg, fg=self.accent_blue,
                 font=(self.font_family, 10, "bold")).pack(side=tk.LEFT)

        btn_clear = tk.Button(header_box, text="清空控制台", bg="#313244", fg=self.text_sub, relief=tk.FLAT,
                              font=(self.font_family, 8), command=self.clear_console)
        btn_clear.pack(side=tk.RIGHT)

        # 终端文本框 (黑底绿字 Terminal 风格)
        self.console = tk.Text(card, bg="#11111b", fg="#a6e3a1", insertbackground="#a6e3a1",
                               font=("Consolas", 9), wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # 底部大按钮操作区
        btn_bar = tk.Frame(card, bg=self.card_bg)
        btn_bar.pack(fill=tk.X)

        self.btn_start = tk.Button(btn_bar, text="🚀 启动本地极速训练", bg="#2e7d32", fg="white",
                                   activebackground="#388e3c", font=(self.font_family, 10, "bold"),
                                   relief=tk.FLAT, pady=7, cursor="hand2", command=self.start_training)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_stop = tk.Button(btn_bar, text="⏹ 终止训练", bg="#c62828", fg="white",
                                  activebackground="#d32f2f", font=(self.font_family, 10, "bold"),
                                  relief=tk.FLAT, pady=7, state=tk.DISABLED, cursor="hand2", command=self.stop_training)
        self.btn_stop.pack(side=tk.RIGHT, padx=(4, 0), ipadx=14)

    # 路径浏览与检测
    def browse_data_dir(self):
        selected = filedialog.askdirectory(initialdir=self.var_data_dir.get(), title="选择训练数据集目录")
        if selected:
            self.var_data_dir.set(selected)
            self.update_dataset_stats()
            self.recalculate_math()

    def browse_comfy_dir(self):
        selected = filedialog.askdirectory(initialdir=self.var_comfy_models.get(), title="选择 ComfyUI 模型库目录")
        if selected:
            self.var_comfy_models.set(selected)

    def browse_output_dir(self):
        selected = filedialog.askdirectory(initialdir=self.var_output_dir.get(), title="选择权重产出保存根目录")
        if selected:
            self.var_output_dir.set(selected)
            self.update_output_preview()

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
        
        rep = self.var_repeats.get()
        ep = self.var_epochs.get()
        total_exposures = img_count * rep * ep
        recommended_steps = max(1, total_exposures // 4)
        
        self.var_calc_stat.set(f"换算：{img_count}图 × {rep}重复 × {ep}轮 = {total_exposures}次曝光 (等效推荐约 {recommended_steps} 步)")

    def align_steps_by_formula(self):
        dpath = Path(self.var_data_dir.get())
        if not dpath.exists():
            return
        img_count = len([f for f in dpath.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        rep = self.var_repeats.get()
        ep = self.var_epochs.get()
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

    # 训练执行控制
    def start_training(self):
        if self.is_training:
            return

        # 检查数据集
        dpath = Path(self.var_data_dir.get())
        if not dpath.exists():
            messagebox.showerror("错误", f"训练数据集路径不存在: {dpath}")
            return

        lora_name_clean = self.var_lora_name.get().strip()
        if not lora_name_clean:
            lora_name_clean = "my_lora"

        cmd = [
            sys.executable, "-u", "train_local.py",
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
            "--text_device", self.var_text_device.get(),
            "--vae_device", self.var_vae_device.get()
        ]
        if self.var_copy_comfy.get():
            cmd.append("--copy_to_comfy_loras")

        self.log(f"\n{'='*75}\n>>> 正在启动训练引擎...\n执行命令: {' '.join(cmd)}\n{'='*75}\n")
        self.is_training = True
        self.btn_start.config(state=tk.DISABLED, bg="#45475a")
        self.btn_stop.config(state=tk.NORMAL)

        # 启动后台守护线程读取输出
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
                errors="replace"
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

    def stop_training(self):
        if self.current_process and self.is_training:
            reply = messagebox.askyesno("确认终止", "确定要强制中断当前正在运行的训练进程吗？")
            if reply:
                try:
                    self.current_process.terminate()
                    self.log("\n[用户操作] 正在向训练进程发送终止信号...\n")
                except Exception as e:
                    self.log(f"\n终止进程异常: {e}\n")

    def training_finished(self, return_code):
        self.is_training = False
        self.current_process = None
        self.btn_start.config(state=tk.NORMAL, bg="#2e7d32")
        self.btn_stop.config(state=tk.DISABLED)

        if return_code == 0:
            lora_name = self.var_lora_name.get().strip() or "my_lora"
            dest_dir = Path(self.var_output_dir.get()) / lora_name
            self.log(f"\n{'='*75}\n>>> [成功] 本地训练已圆满完成！权重已归档至专属子文件夹: {dest_dir} <<<\n{'='*75}\n")
            messagebox.showinfo("训练成功", f"LoRA 训练已圆满完成！\n专属产出文件夹:\n{dest_dir}\n包含: {lora_name}-comfy.safetensors 等。")
        else:
            self.log(f"\n>>> [提示] 训练进程已结束，退出码: {return_code} <<<\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernLocalTrainerGUI(root)
    root.mainloop()
