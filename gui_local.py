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
        self.root.title("OpenMOSS MOSS-Image 本地训练控制面板")
        self.root.geometry("1120x820")
        self.root.minsize(980, 720)

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

        self.var_run_name = tk.StringVar(value="ness_style")
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

        self.build_ui()
        self.update_dataset_stats()
        self.recalculate_math()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.bg_dark, foreground=self.text_main, font=(self.font_family, 10))
        style.configure("Card.TFrame", background=self.card_bg)
        style.configure("TLabel", background=self.card_bg, foreground=self.text_main, font=(self.font_family, 10))
        style.configure("Sub.TLabel", background=self.card_bg, foreground=self.text_sub, font=(self.font_family, 9))
        style.configure("Header.TLabel", background=self.card_bg, foreground=self.accent_blue, font=(self.font_family, 12, "bold"))
        style.configure("TCheckbutton", background=self.card_bg, foreground=self.text_main, font=(self.font_family, 10))
        style.configure("TCombobox", fieldbackground=self.input_bg, background=self.card_bg, foreground=self.text_main)

    def build_ui(self):
        # 1. 顶部 Header
        header_frame = tk.Frame(self.root, bg=self.card_bg, height=65)
        header_frame.pack(fill=tk.X, padx=12, pady=(12, 8))

        title_lbl = tk.Label(header_frame, text="OpenMOSS MOSS-Image 本地 LoRA 训练引擎",
                             bg=self.card_bg, fg=self.accent_blue, font=(self.font_family, 15, "bold"))
        title_lbl.pack(side=tk.LEFT, padx=16, pady=12)

        # 硬件状态探测展示
        gpu_info = self.probe_gpu_quick()
        gpu_lbl = tk.Label(header_frame, text=f"当前算力设备: {gpu_info}",
                           bg=self.card_bg, fg=self.accent_green, font=(self.font_family, 9))
        gpu_lbl.pack(side=tk.RIGHT, padx=16, pady=12)

        # 2. 主体左右分栏
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.bg_dark, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # 左侧控制面板 (固定宽度)
        left_container = tk.Frame(main_paned, bg=self.bg_dark, width=540)
        main_paned.add(left_container, minsize=480)

        # 右侧实时监控与日志面板
        right_container = tk.Frame(main_paned, bg=self.bg_dark)
        main_paned.add(right_container, minsize=450)

        self.build_left_panel(left_container)
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

    def build_left_panel(self, parent):
        # 快捷预设按钮栏
        preset_frame = tk.Frame(parent, bg=self.card_bg)
        preset_frame.pack(fill=tk.X, pady=(0, 8))
        
        p_title = tk.Label(preset_frame, text="快速模板预设：", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 10, "bold"))
        p_title.pack(side=tk.LEFT, padx=12, pady=10)

        btn_style = tk.Button(preset_frame, text="🎨 艺术画风模板", bg="#45475a", fg=self.text_main,
                              activebackground=self.accent_blue, font=(self.font_family, 9, "bold"),
                              relief=tk.FLAT, padx=10, pady=4, cursor="hand2", command=self.apply_style_preset)
        btn_style.pack(side=tk.LEFT, padx=6)

        btn_char = tk.Button(preset_frame, text="👤 人物角色模板", bg="#45475a", fg=self.text_main,
                             activebackground=self.accent_blue, font=(self.font_family, 9, "bold"),
                             relief=tk.FLAT, padx=10, pady=4, cursor="hand2", command=self.apply_character_preset)
        btn_char.pack(side=tk.LEFT, padx=6)

        # 资产与路径卡片
        path_card = tk.LabelFrame(parent, text=" 📂 资产与目录设置 ", bg=self.card_bg, fg=self.accent_blue,
                                  font=(self.font_family, 10, "bold"), padx=12, pady=10)
        path_card.pack(fill=tk.X, pady=(0, 8))

        # 数据集路径
        self.create_path_row(path_card, "训练图片目录:", self.var_data_dir, self.browse_data_dir, 0)
        # 数据集文件统计展示
        stat_lbl = tk.Label(path_card, textvariable=self.var_dataset_stat, bg=self.card_bg, fg=self.accent_yellow, font=(self.font_family, 9))
        stat_lbl.grid(row=1, column=1, sticky=tk.W, pady=(0, 6))

        # ComfyUI 模型库路径
        self.create_path_row(path_card, "ComfyUI模型库:", self.var_comfy_models, self.browse_comfy_dir, 2)
        # 结果保存路径
        self.create_path_row(path_card, "产出保存目录:", self.var_output_dir, self.browse_output_dir, 3)

        chk_comfy = tk.Checkbutton(path_card, text="训练完成后自动同步复制权重至 ComfyUI models/loras 目录",
                                   variable=self.var_copy_comfy, bg=self.card_bg, fg=self.text_main,
                                   selectcolor=self.input_bg, activebackground=self.card_bg, font=(self.font_family, 9))
        chk_comfy.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))

        # 超参数配置卡片
        hp_card = tk.LabelFrame(parent, text=" ⚙️ 核心训练超参数 ", bg=self.card_bg, fg=self.accent_blue,
                                font=(self.font_family, 10, "bold"), padx=12, pady=10)
        hp_card.pack(fill=tk.BOTH, expand=True)

        # 任务名称
        self.create_input_row(hp_card, "产出名称 (run_name):", self.var_run_name, 0)

        # 重复次数 与 Epoch
        r_f = tk.Frame(hp_card, bg=self.card_bg)
        r_f.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=4)
        tk.Label(r_f, text="单图重复 (Repeats):", bg=self.card_bg, fg=self.text_main, width=18, anchor="w").pack(side=tk.LEFT)
        sp_rep = tk.Spinbox(r_f, from_=1, to=100, textvariable=self.var_repeats, width=8, bg=self.input_bg, fg=self.text_main, command=self.recalculate_math)
        sp_rep.pack(side=tk.LEFT, padx=(0, 16))
        sp_rep.bind("<KeyRelease>", lambda e: self.recalculate_math())

        tk.Label(r_f, text="训练轮数 (Epochs):", bg=self.card_bg, fg=self.text_main).pack(side=tk.LEFT)
        sp_ep = tk.Spinbox(r_f, from_=1, to=100, textvariable=self.var_epochs, width=8, bg=self.input_bg, fg=self.text_main, command=self.recalculate_math)
        sp_ep.pack(side=tk.LEFT, padx=(6, 0))
        sp_ep.bind("<KeyRelease>", lambda e: self.recalculate_math())

        # 总训练步数
        s_f = tk.Frame(hp_card, bg=self.card_bg)
        s_f.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=4)
        tk.Label(s_f, text="核心步数 (Steps):", bg=self.card_bg, fg=self.text_main, width=18, anchor="w").pack(side=tk.LEFT)
        ent_steps = tk.Entry(s_f, textvariable=self.var_steps, width=10, bg=self.input_bg, fg=self.text_main)
        ent_steps.pack(side=tk.LEFT)
        btn_calc = tk.Button(s_f, text="⚡ 按公式对齐", bg="#45475a", fg=self.text_main, relief=tk.FLAT, font=(self.font_family, 8),
                             cursor="hand2", command=self.align_steps_by_formula)
        btn_calc.pack(side=tk.LEFT, padx=8)

        # 换算公式联动展示
        math_lbl = tk.Label(hp_card, textvariable=self.var_calc_stat, bg=self.card_bg, fg=self.accent_yellow, font=(self.font_family, 9))
        math_lbl.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        # Rank 与 Alpha
        ra_f = tk.Frame(hp_card, bg=self.card_bg)
        ra_f.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=4)
        tk.Label(ra_f, text="LoRA 矩阵秩 (Rank):", bg=self.card_bg, fg=self.text_main, width=18, anchor="w").pack(side=tk.LEFT)
        cb_rank = ttk.Combobox(ra_f, textvariable=self.var_rank, values=[16, 32, 64, 128], width=8, state="readonly")
        cb_rank.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(ra_f, text="缩放系数 (Alpha):", bg=self.card_bg, fg=self.text_main).pack(side=tk.LEFT)
        ent_alpha = tk.Entry(ra_f, textvariable=self.var_alpha, width=8, bg=self.input_bg, fg=self.text_main)
        ent_alpha.pack(side=tk.LEFT, padx=(6, 0))

        # 学习率 与 精度
        lr_f = tk.Frame(hp_card, bg=self.card_bg)
        lr_f.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=4)
        tk.Label(lr_f, text="基础学习率 (LR):", bg=self.card_bg, fg=self.text_main, width=18, anchor="w").pack(side=tk.LEFT)
        ent_lr = tk.Entry(lr_f, textvariable=self.var_lr, width=10, bg=self.input_bg, fg=self.text_main)
        ent_lr.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(lr_f, text="计算精度:", bg=self.card_bg, fg=self.text_main).pack(side=tk.LEFT)
        cb_prec = ttk.Combobox(lr_f, textvariable=self.var_precision, values=["bf16", "fp16"], width=8, state="readonly")
        cb_prec.pack(side=tk.LEFT, padx=(6, 0))

        # 离线设备路由 (显存防爆设置)
        dev_f = tk.Frame(hp_card, bg=self.card_bg)
        dev_f.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))
        tk.Label(dev_f, text="文本提取设备:", bg=self.card_bg, fg=self.text_main, width=18, anchor="w").pack(side=tk.LEFT)
        cb_txt_dev = ttk.Combobox(dev_f, textvariable=self.var_text_device, values=["cpu", "cuda"], width=8, state="readonly")
        cb_txt_dev.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(dev_f, text="(8GB 卡请锁定 CPU 避免 OOM)", bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8)).pack(side=tk.LEFT)

    def create_path_row(self, parent, label_text, var, browse_cmd, row_idx):
        tk.Label(parent, text=label_text, bg=self.card_bg, fg=self.text_main, width=15, anchor="w").grid(row=row_idx, column=0, sticky=tk.W, pady=3)
        ent = tk.Entry(parent, textvariable=var, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main)
        ent.grid(row=row_idx, column=1, sticky=tk.EW, padx=6, pady=3)
        parent.columnconfigure(1, weight=1)
        btn = tk.Button(parent, text="浏览...", bg="#45475a", fg=self.text_main, relief=tk.FLAT, font=(self.font_family, 8),
                        cursor="hand2", command=browse_cmd)
        btn.grid(row=row_idx, column=2, padx=2, pady=3)

    def create_input_row(self, parent, label_text, var, row_idx):
        tk.Label(parent, text=label_text, bg=self.card_bg, fg=self.text_main, width=18, anchor="w").grid(row=row_idx, column=0, sticky=tk.W, pady=4)
        ent = tk.Entry(parent, textvariable=var, bg=self.input_bg, fg=self.text_main, insertbackground=self.text_main)
        ent.grid(row=row_idx, column=1, sticky=tk.EW, pady=4)
        parent.columnconfigure(1, weight=1)

    def build_right_panel(self, parent):
        card = tk.Frame(parent, bg=self.card_bg, padx=12, pady=10)
        card.pack(fill=tk.BOTH, expand=True)

        header_box = tk.Frame(card, bg=self.card_bg)
        header_box.pack(fill=tk.X, pady=(0, 6))

        tk.Label(header_box, text="🖥️ 训练实时日志与终端控制台", bg=self.card_bg, fg=self.accent_blue,
                 font=(self.font_family, 11, "bold")).pack(side=tk.LEFT)

        btn_clear = tk.Button(header_box, text="清空控制台", bg="#313244", fg=self.text_sub, relief=tk.FLAT,
                              font=(self.font_family, 8), command=self.clear_console)
        btn_clear.pack(side=tk.RIGHT)

        # 终端文本框 (黑底绿字 Terminal 风格)
        self.console = tk.Text(card, bg="#11111b", fg="#a6e3a1", insertbackground="#a6e3a1",
                               font=("Consolas", 9), wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 底部大按钮操作区
        btn_bar = tk.Frame(card, bg=self.card_bg)
        btn_bar.pack(fill=tk.X)

        self.btn_start = tk.Button(btn_bar, text="🚀 启动本地极速训练", bg="#2e7d32", fg="white",
                                   activebackground="#388e3c", font=(self.font_family, 11, "bold"),
                                   relief=tk.FLAT, pady=8, cursor="hand2", command=self.start_training)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_stop = tk.Button(btn_bar, text="⏹ 终止训练", bg="#c62828", fg="white",
                                  activebackground="#d32f2f", font=(self.font_family, 11, "bold"),
                                  relief=tk.FLAT, pady=8, state=tk.DISABLED, cursor="hand2", command=self.stop_training)
        self.btn_stop.pack(side=tk.RIGHT, padx=(6, 0), ipadx=16)

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
        selected = filedialog.askdirectory(initialdir=self.var_output_dir.get(), title="选择权重产出保存目录")
        if selected:
            self.var_output_dir.set(selected)

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
        
        self.var_calc_stat.set(f"曝光换算：{img_count}图 × {rep}重复 × {ep}轮 = {total_exposures}次曝光 (等效推荐约 {recommended_steps} 步)")

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
        self.var_run_name.set("ness_style")
        self.align_steps_by_formula()
        messagebox.showinfo("模板切换", "已成功加载【艺术画风模板】：\nRepeats=10, Epochs=6, Rank=32, Alpha=32.0, LR=1e-4")

    def apply_character_preset(self):
        self.var_repeats.set(6)
        self.var_epochs.set(3)
        self.var_rank.set(32)
        self.var_alpha.set(16.0)
        self.var_lr.set("8e-5")
        self.var_run_name.set("my_character")
        self.align_steps_by_formula()
        messagebox.showinfo("模板切换", "已成功加载【人物角色模板】：\nRepeats=6, Epochs=3, Rank=32, Alpha=16.0, LR=8e-5")

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

        cmd = [
            sys.executable, "-u", "train_local.py",
            "--data_dir", self.var_data_dir.get(),
            "--comfy_models_dir", self.var_comfy_models.get(),
            "--output_dir", self.var_output_dir.get(),
            "--run_name", self.var_run_name.get(),
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
            self.log(f"\n{'='*75}\n>>> [成功] 本地训练已圆满完成！权重已归档至输出目录。<<<\n{'='*75}\n")
            messagebox.showinfo("训练成功", "LoRA 训练已圆满完成！\n权重文件已自动导出并同步。")
        else:
            self.log(f"\n>>> [提示] 训练进程已结束，退出码: {return_code} <<<\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernLocalTrainerGUI(root)
    root.mainloop()
