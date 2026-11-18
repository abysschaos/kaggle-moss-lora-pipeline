import os
import sys
import threading
import subprocess
import webbrowser
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
        
        # 窗口大小与屏幕居中适配 (适配各类笔记本与显示器)
        win_w, win_h = 1240, 850
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(1050, 720)

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
        self.var_text_device = tk.StringVar(value="cpu")
        self.var_vae_device = tk.StringVar(value="cuda")

        self.var_dataset_stat = tk.StringVar(value="未检测")
        self.var_calc_stat = tk.StringVar(value="曝光计算：就绪")
        self.var_output_preview = tk.StringVar(value="")

        self.build_ui()
        self.update_dataset_stats()
        self.recalculate_math()
        self.update_output_preview()
        self.check_trainer_status()

        # 动态分配初始左右分栏位置，确保左侧控制区充足不被挤压
        self.root.update_idletasks()
        try:
            self.main_paned.sash_place(0, 640, 0)
        except Exception:
            pass

        # 启动时执行一次静默自动底模探测并预填
        self.auto_scan_models(silent=True)

        # 监听名称变动实时刷新产出路径预览
        self.var_lora_name.trace_add("write", lambda *args: self.update_output_preview())

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
            cfg = {
                "data_dir": self.var_data_dir.get(),
                "comfy_models_dir": self.var_comfy_models.get(),
                "output_dir": self.var_output_dir.get(),
                "lora_name": self.var_lora_name.get(),
                "trigger_word": self.var_trigger_word.get()
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

        # 2. 主体左右分栏 (带清晰可拖拽分割条)
        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.bg_dark, sashwidth=6, sashrelief=tk.RAISED)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 左侧控制面板 (带平滑滚动条，确保小屏幕不被挤压)
        left_outer = tk.Frame(self.main_paned, bg=self.card_bg)
        self.main_paned.add(left_outer, minsize=480)

        # 右侧实时监控与日志控制台
        right_container = tk.Frame(self.main_paned, bg=self.card_bg)
        self.main_paned.add(right_container, minsize=380)

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

        # 核心关键：记录 window 实例并在 canvas 大小变动时动态同步绑定框架宽度
        self.canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def on_canvas_configure(event):
            # 将可滚动框架的宽度强行锁定为 Canvas 的可见宽度，彻底杜绝内容横向超宽被遮挡
            canvas.itemconfig(self.canvas_window, width=event.width)

        canvas.bind("<Configure>", on_canvas_configure)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.configure(yscrollcommand=scrollbar.set)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        scrollbar.pack(side="right", fill="y", pady=4)

        self.build_left_controls(scrollable_frame)

    def build_left_controls(self, parent):
        # 1. 快捷预设按钮栏
        preset_frame = tk.Frame(parent, bg=self.card_bg)
        preset_frame.pack(fill=tk.X, pady=(0, 6))
        
        tk.Label(preset_frame, text="快速模板预设：", bg=self.card_bg, fg=self.text_main, font=(self.font_family, 9, "bold")).pack(side=tk.LEFT, padx=(4, 6), pady=4)

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

        lbl_preview = tk.Label(name_card, textvariable=self.var_output_preview, bg=self.card_bg, fg=self.accent_green,
                               font=(self.font_family, 8), wraplength=480, justify=tk.LEFT)
        lbl_preview.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))

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
            browse_title="选择 MOSS-Image 底模文件 (moss_image_heal_bf16.safetensors 等)",
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

        tk.Label(trainer_card, text="说明: 推荐点击【⬇️ 一键自动克隆/下载】全自动安装；若网络受限，可点击【📂 打开训练器放置目录】手动解压源码入内，并点击【🩹 检查并注入补丁】自愈修复。",
                 bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8), wraplength=480, justify=tk.LEFT).pack(fill=tk.X, pady=(2, 0))

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
            path_card, "ComfyUI 模型库目录 (自动复用无需下载):", self.var_comfy_models,
            browse_cmd=self.browse_comfy_dir
        )

        chk_comfy = tk.Checkbutton(path_card, text="训练完成后自动同步复制权重至 ComfyUI models/loras 专属子目录",
                                   variable=self.var_copy_comfy, bg=self.card_bg, fg=self.text_main,
                                   selectcolor=self.input_bg, activebackground=self.card_bg,
                                   font=(self.font_family, 8), wraplength=480, justify=tk.LEFT)
        chk_comfy.pack(fill=tk.X, pady=(4, 2))

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

        dev_f = tk.Frame(hp_card, bg=self.card_bg)
        dev_f.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(2, 4))
        tk.Label(dev_f, text="文本提取设备:", bg=self.card_bg, fg=self.text_main, width=17, anchor="w").pack(side=tk.LEFT)
        cb_txt_dev = ttk.Combobox(dev_f, textvariable=self.var_text_device, values=["cpu", "cuda"], width=7, state="readonly")
        cb_txt_dev.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(dev_f, text="(8GB 显卡请锁定 CPU 避免 OOM)", bg=self.card_bg, fg=self.text_sub, font=(self.font_family, 8)).pack(side=tk.LEFT)

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

        tk.Label(top_bar, text=label_text, bg=self.card_bg, fg=self.text_main,
                 font=(self.font_family, 8, "bold"), anchor="w").pack(side=tk.LEFT)

        def browse_file():
            selected = filedialog.askopenfilename(
                title=browse_title,
                filetypes=filetypes
            )
            if selected:
                var.set(selected)
                self.var_models_status.set("已手动自选模型权重")
                self.update_output_preview()

        # 醒目的青蓝强调色大按钮，方便用户一眼识别并点击自选
        btn_browse = tk.Button(top_bar, text="📁 点此自选文件...", bg=self.accent_blue, fg="#11111b",
                               relief=tk.FLAT, font=(self.font_family, 8, "bold"), cursor="hand2", padx=8, pady=1,
                               command=browse_file)
        btn_browse.pack(side=tk.RIGHT, padx=(4, 0))

        btn_open = tk.Button(top_bar, text="📂 打开放置目录", bg="#313244", fg=self.accent_green,
                             relief=tk.FLAT, font=(self.font_family, 8), cursor="hand2", padx=6, pady=1,
                             command=lambda: self.open_in_explorer(folder_path))
        btn_open.pack(side=tk.RIGHT)

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

        tk.Label(top_bar, text=label_text, bg=self.card_bg, fg=self.text_main,
                 font=(self.font_family, 8, "bold"), anchor="w").pack(side=tk.LEFT)

        btn_browse = tk.Button(top_bar, text="📁 点此自选文件夹...", bg=self.accent_blue, fg="#11111b",
                               relief=tk.FLAT, font=(self.font_family, 8, "bold"), cursor="hand2", padx=8, pady=1,
                               command=browse_cmd)
        btn_browse.pack(side=tk.RIGHT, padx=(4, 0))

        target = open_target_cmd if open_target_cmd else lambda: self.open_in_explorer(var.get())
        btn_open = tk.Button(top_bar, text="📂 打开此目录", bg="#313244", fg=self.accent_green,
                             relief=tk.FLAT, font=(self.font_family, 8), cursor="hand2", padx=6, pady=1,
                             command=target)
        btn_open.pack(side=tk.RIGHT)

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
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
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
                self.var_models_status.set("✅ 全部自动就绪！")
                self.save_local_config()
                if not silent:
                    messagebox.showinfo("扫描成功", "已成功自动检索并匹配到全部三大核心模型权重！\n路径已自动填入输入框。")
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
        card = tk.Frame(parent, bg=self.card_bg, padx=10, pady=8)
        card.pack(fill=tk.BOTH, expand=True)

        header_box = tk.Frame(card, bg=self.card_bg)
        header_box.pack(fill=tk.X, pady=(0, 4))

        tk.Label(header_box, text="🖥️ 训练实时日志与终端控制台", bg=self.card_bg, fg=self.accent_blue,
                 font=(self.font_family, 10, "bold")).pack(side=tk.LEFT)

        btn_clear = tk.Button(header_box, text="清空控制台", bg="#313244", fg=self.text_sub, relief=tk.FLAT,
                              font=(self.font_family, 8), command=self.clear_console)
        btn_clear.pack(side=tk.RIGHT)

        self.console = tk.Text(card, bg="#11111b", fg="#a6e3a1", insertbackground="#a6e3a1",
                               font=("Consolas", 9), wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

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

        # 检查训练器框架就绪状态
        trainer_dir = Path("./KohakuMossTrainer").resolve()
        if not (trainer_dir / "pyproject.toml").exists():
            messagebox.showwarning("训练器未就绪", "检测到 KohakuMossTrainer 训练器尚未就绪！\n系统将在启动时自动尝试从 GitHub 克隆。若您网络无法访问 GitHub，请先点击【📂 打开训练器放置目录】手动解压源码。")

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

        if self.var_transformer_path.get().strip():
            cmd.extend(["--transformer_path", self.var_transformer_path.get().strip()])
        if self.var_text_encoder_path.get().strip():
            cmd.extend(["--text_encoder_path", self.var_text_encoder_path.get().strip()])
        if self.var_vae_path.get().strip():
            cmd.extend(["--vae_path", self.var_vae_path.get().strip()])

        if self.var_copy_comfy.get():
            cmd.append("--copy_to_comfy_loras")

        self.log(f"\n{'='*75}\n>>> 正在启动训练引擎...\n执行命令: {' '.join(cmd)}\n{'='*75}\n")
        self.is_training = True
        self.btn_start.config(state=tk.DISABLED, bg="#45475a")
        self.btn_stop.config(state=tk.NORMAL)

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
                pid = self.current_process.pid
                self.log(f"\n[用户操作] 正在强制级联终止训练进程树 (PID: {pid})...\n")
                try:
                    # Windows 下使用 taskkill /F /T 强杀目标进程及其派生的整个子孙进程树 (彻底清理 kogine 和主训练 python 进程)
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False
                    )
                except Exception as e:
                    self.log(f"taskkill 异常: {e}\n")

                try:
                    self.current_process.kill()
                except Exception:
                    pass

                self.log("[用户操作] 训练进程树已全部强制中断，GPU 与内存资源已安全释放。\n")
                self.root.after(0, self.training_finished, -9)

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
