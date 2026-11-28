#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
九大场景黄金超参数预设矩阵与高阶默认基线 (Single Source of Truth)
严格对齐上游 KohakuBlueleaf/KohakuMossTrainer 源码架构 (configs/moss/algo/*, quality_lokr16.py, muon.py, oft, tlora, EDM2 Power EMA)
与 MOSS-Image 5B 单流 DiT + Qwen3.5-9B 因果注意力训练动力学规范。
同时声明本机单卡基准与云端 Kaggle T4×2 DDP 等效 Batch 自适应校准映射。
"""

ADVANCED_DEFAULTS = {
    "lokr_factor": "8",
    "lokr_full_matrix": False,
    "lokr_decompose_both": False,
    "train_norms": False,
    "rank_dropout": "0.0",
    "module_dropout": "0.0",
    "input_dropout": "0.0",
    "optimizer": "adamw",
    "lr_scheduler": "cosine",
    "warmup_steps": "-1",
    "weight_decay": "0.01",
    "wd_rules": "moss",
    "ema": "0.0",
    "timestep": "moss",
    "text_drop": "0.10",
    "flip": False,
    "seed": "0",
    "save_format": "both",
    "caption_template": "auto",
    "caption_p_nl": "0.75",
    "caption_variants": "4",
    "int8_compute": False,
    "compile_mode": "off",
    "step_release": "none",
    "sample_every": "0",
    "sample_prompt": "",
    "teacher_mode": "none",
    "distill_weight": "0.05",
}

# 防欠拟合优化器更新步数安全下限（尤其针对 Dual-T4 DDP 4 图/步吞吐下的 10~25 张小图集）
MIN_RECOMMENDED_STEPS_SINGLE = 180
MIN_RECOMMENDED_STEPS_LOKR = 200
MIN_RECOMMENDED_STEPS_MULTI = 240

# Kaggle 双卡 T4 DDP（MAX_TOKENS=9216, 1024px 下 2图/卡 × 2卡 = 4图/步）相对本机单卡的参数自适应修正
CLOUD_T4X2_OVERRIDES = {
    "style": {"grad_acc": "1", "lr": "1.2e-4"},
    "character": {"grad_acc": "1", "lr": "1e-4"},
    "multi_char": {"grad_acc": "1", "lr": "1e-4"},
    "multi_style": {"grad_acc": "1", "lr": "1.2e-4"},
    "concept": {"grad_acc": "1", "lr": "1.2e-4"},
    "lokr_style": {"grad_acc": "1", "lr": "4e-4"},
    "pixel_style": {"grad_acc": "1", "lr": "4e-4"},
    "antifit_char": {"grad_acc": "1", "lr": "1e-4"},
    "muon_style": {"grad_acc": "1", "lr": "1.3e-4"},
}

PRESETS = [
    {
        "id": "style",
        "row": 1,
        "label": "🎨 艺术画风",
        "bg": "#45475a",
        "fg_key": "text_main",
        "active_bg_key": "accent_blue",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": "4",
            "epochs": "6",
            "rank": "32",
            "alpha": "32.0",
            "algo": "lora",
            "dora": False,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "1.0",
            "lr": "1e-4",
            "grad_acc": "1",
            "timestep_mu": "-0.45",
            "text_drop": "0.10",
            "ema": "0.0",
            "lora_name": "my_art_style",
            "trigger_word": "",
        },
        "dialog_title": "🎨 艺术画风预设已加载",
        "dialog_message": (
            "已配置标准【艺术画风 LoRA】稳健方案：\n"
            "• 算法结构：标准 LoRA (Rank=32 / Alpha=32.0, DoRA 关闭防锁死构图轮廓)\n"
            "• 注入范围：attn-mlp（注意力 + 前馈网络双覆盖，兼顾光影排布与色彩渲染）\n"
            "• 时间步重心：moss (mu=-0.45)，覆盖全局光影色温同时强化笔触质感与线稿收敛\n"
            "• 曝光与优化：Repeats=4 / Epochs=6（24 次/图），LR=1e-4（双卡 T4 1.2e-4），Text Drop=0.10，默认 EMA=0.0（长步数画风可在高级面板切换 EMA=power）\n"
            "• 提示词建议：触发词栏已保留您的原有输入（无固定风格触发词亦可留空），建议将画风本体词从 .txt 中剪枝以集中学习笔触。"
        ),
    },
    {
        "id": "character",
        "row": 1,
        "label": "👤 单人角色",
        "bg": "#45475a",
        "fg_key": "text_main",
        "active_bg_key": "accent_blue",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": "7",
            "epochs": "4",
            "rank": "24",
            "alpha": "24.0",
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "2.0",
            "lr": "8e-5",
            "grad_acc": "2",
            "timestep_mu": "0.35",
            "text_drop": "0.06",
            "ema": "0.0",
            "lora_name": "my_character",
            "trigger_word": "",
        },
        "dialog_title": "👤 单人角色预设已加载",
        "dialog_message": (
            "已配置高精度【单人角色 DoRA】方案：\n"
            "• 权重解耦：开启 DoRA + Rank=24 / Alpha=24.0（精准锁定五官瞳孔与发饰裁片，抑制 32+ 高秩带来的训练背景死记）\n"
            "• 加速收敛：开启 LoRA+ (2.0) + GradAcc=2 平滑单图梯度方差 + LR=8e-5（双卡 T4 自动设为 GradAcc=1 / LR=1e-4）\n"
            "• 时间步重心：moss (mu=+0.35)，将低噪高频细节区采样占比提升至约 27.5%，专攻瞳孔高光与复杂服饰纹理\n"
            "• 触发词与泛化：Text Drop=0.06 兼顾触发词绑定与无词泛化弹性；请在【核心触发词】栏填入角色英文名。"
        ),
    },
    {
        "id": "multi_char",
        "row": 1,
        "label": "👥 多人物配比",
        "bg": "#313244",
        "fg_key": "accent_purple",
        "active_bg_key": "accent_purple",
        "multi_subset": True,
        "align_steps": False,
        "values": {
            "repeats": "6",
            "epochs": "4",
            "rank": "48",
            "alpha": "48.0",
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp+text-refiner",
            "loraplus_ratio": "2.0",
            "lr": "8e-5",
            "grad_acc": "2",
            "timestep_mu": "0.35",
            "text_drop": "0.05",
            "ema": "0.0",
            "lora_name": "my_multi_char",
            "trigger_word": "",
        },
        "dialog_title": "👥 多人物混合预设已加载",
        "dialog_message": (
            "已为您自动开启【多子集配比】并配置防串脸参数：\n"
            "• 专属注入：attn-mlp+text-refiner（训练文本重整层，强化 Qwen3.5-9B 区分不同角色专属触发词的语义边界）\n"
            "• 充足容量：DoRA + Rank=48 / Alpha=48.0 + LoRA+ (2.0) + Text Drop=0.05 + mu=+0.35\n"
            "• 下一步操作：请切换到上方【📂 数据与续炼】标签页添加各角色文件夹与独立触发词，点击【⚖️ 自动反比配平 Repeats】后再对齐步数（保底 >=240 步）。"
        ),
    },
    {
        "id": "multi_style",
        "row": 1,
        "label": "🎭 多画风配比",
        "bg": "#313244",
        "fg_key": "accent_blue",
        "active_bg_key": "accent_blue",
        "multi_subset": True,
        "align_steps": False,
        "values": {
            "repeats": "4",
            "epochs": "5",
            "rank": "48",
            "alpha": "48.0",
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp+modulation",
            "loraplus_ratio": "2.0",
            "lr": "1e-4",
            "grad_acc": "2",
            "timestep_mu": "-0.45",
            "text_drop": "0.10",
            "ema": "0.0",
            "lora_name": "my_multi_style",
            "trigger_word": "",
        },
        "dialog_title": "🎭 多画风混合预设已加载",
        "dialog_message": (
            "已自动开启【多子集配比】并配置多画风融合参数：\n"
            "• 专属注入：attn-mlp+modulation（额外纳入 AdaLN 全局调制投影，解耦不同画风的色彩氛围与光影基底）\n"
            "• 容量与采样：DoRA + Rank=48 / Alpha=48.0 + LR=1e-4 + mu=-0.45 + Text Drop=0.10\n"
            "• 下一步操作：请在【📂 数据与续炼】中为各画风子集设置不同触发前缀（如 style_ink, style_cel），点击【⚖️ 自动反比配平 Repeats】后再对齐步数。"
        ),
    },
    {
        "id": "concept",
        "row": 1,
        "label": "💡 概念/动作",
        "bg": "#313244",
        "fg_key": "accent_green",
        "active_bg_key": "accent_green",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": "6",
            "epochs": "4",
            "rank": "32",
            "alpha": "32.0",
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp+text-refiner",
            "loraplus_ratio": "2.0",
            "lr": "1e-4",
            "grad_acc": "2",
            "timestep_mu": "0.0",
            "text_drop": "0.08",
            "ema": "0.0",
            "lora_name": "my_concept",
            "trigger_word": "",
        },
        "dialog_title": "💡 概念与动作预设已加载",
        "dialog_message": (
            "已配置【特定概念 / 复杂动作 / 特殊构图视角】方案：\n"
            "• 注入范围：attn-mlp+text-refiner + DoRA (Rank=32 / Alpha=32.0) + LoRA+ (2.0)\n"
            "• 对称时间步：moss (mu=0.0)，全频段均衡覆盖大肢体骨架排布与局部手部/道具交互细节（Text Drop=0.08）\n"
            "• 曝光建议：Repeats=6 / Epochs=4（24 次/图），请在 .txt 或 train.jsonl 中使用准确英文自然语言描述空间方位与交互关系。"
        ),
    },
    {
        "id": "lokr_style",
        "row": 2,
        "label": "💎 LoKr 满秩画风",
        "bg": "#313244",
        "fg_key": "accent_peach",
        "active_bg_key": "accent_peach",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": "5",
            "epochs": "5",
            "rank": "10000",
            "alpha": "1.0",
            "algo": "lokr",
            "dora": False,
            "lokr_factor": "8",
            "lokr_full_matrix": True,
            "lokr_decompose_both": False,
            "target_preset": "attn-mlp+modulation",
            "loraplus_ratio": "1.0",
            "lr": "3e-4",
            "grad_acc": "1",
            "timestep_mu": "-0.45",
            "text_drop": "0.10",
            "ema": "0.0",
            "lora_name": "my_lokr_style",
            "trigger_word": "",
        },
        "dialog_title": "💎 LoKr 满秩高还原画风预设已加载",
        "dialog_message": (
            "已校准 LyCORIS / KohakuMossTrainer 满秩 Kronecker 高还原画风方案：\n"
            "• 算法矩阵：LoKr + Full Matrix 开启 + Factor=8 + Rank=10000 / Alpha=1.0（强制大维度精确满秩 Kronecker 乘积，杜绝伪满秩低秩坍缩）\n"
            "• 注入与学习率：attn-mlp+modulation + LR=3e-4（双卡 T4 自动升至 4e-4，匹配 Factor=8 梯度范数）+ Text Drop=0.10\n"
            "• 纹理时间步：moss (mu=-0.45)，平衡全局色温氛围、笔触线条与高对比锐度\n"
            "• 产物加载：默认导出 -comfy 与 -lycoris 双格式（EMA=0.0 零冗余），ComfyUI 原生 Load LoRA 或 LycorisLoader 均可直读。"
        ),
    },
    {
        "id": "pixel_style",
        "row": 2,
        "label": "🕹️ 像素/硬边",
        "bg": "#313244",
        "fg_key": "accent_peach",
        "active_bg_key": "accent_yellow",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": "8",
            "epochs": "5",
            "rank": "10000",
            "alpha": "1.0",
            "algo": "lokr",
            "dora": False,
            "lokr_factor": "8",
            "lokr_full_matrix": True,
            "lokr_decompose_both": False,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "1.0",
            "lr": "3e-4",
            "grad_acc": "1",
            "timestep_mu": "-0.35",
            "text_drop": "0.10",
            "ema": "0.0",
            "lora_name": "my_pixel_style",
            "trigger_word": "",
        },
        "dialog_title": "🕹️ 像素画与极锐硬边 LoKr 预设已加载",
        "dialog_message": (
            "已配置专为 Pixel Art / Sprite / 极锐赛璐璐平涂设计的 LoKr 满秩高频方案：\n"
            "• 算法矩阵：LoKr + Full Matrix + Factor=8 + Rank=10000 / Alpha=1.0（无锯齿丢失的精确网格响应，专注 attn-mlp 防止 AdaLN 泛糊）\n"
            "• 曝光与学习率：Repeats=8 / Epochs=5 + LR=3e-4（双卡 T4 4e-4）+ mu=-0.35（强化高频像素网格与锐利边缘）+ Text Drop=0.10\n"
            "• 数据提醒：像素小图训练前务必采用 Nearest-Neighbor 最近邻整数倍放大至适配 MOSS 的基准分辨率，严禁双线性/三次插值糊边！"
        ),
    },
    {
        "id": "antifit_char",
        "row": 2,
        "label": "🛡️ 小图防过拟合",
        "bg": "#313244",
        "fg_key": "accent_yellow",
        "active_bg_key": "accent_yellow",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": "8",
            "epochs": "4",
            "rank": "16",
            "alpha": "8.0",
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "1.5",
            "lr": "8e-5",
            "grad_acc": "2",
            "timestep_mu": "0.35",
            "rank_dropout": "0.05",
            "module_dropout": "0.05",
            "input_dropout": "0.0",
            "text_drop": "0.06",
            "ema": "0.0",
            "lora_name": "my_antifit_lora",
            "trigger_word": "",
        },
        "dialog_title": "🛡️ 稀缺小图集防过拟合预设已加载",
        "dialog_message": (
            "已针对 8~18 张稀缺角色/概念图集配置【半秩物理阻尼 + 防欠拟合保底】方案：\n"
            "• 结构阻尼：DoRA + Rank=16 / Alpha=8.0（0.5 倍缩放比物理抑制背景死记与色块烤焦，替代旧版高秩+超低LR欠拟合组合）\n"
            "• 温和正则：Rank Dropout=0.05 + Module Dropout=0.05（DoRA 下自动锁死 Input Dropout=0.0）+ Text Drop=0.06 防止过拟合\n"
            "• 充足更新：Repeats=8 / Epochs=4 + LoRA+ (1.5) + LR=8e-5（双卡 T4 自动升至 1e-4 并执行 >=180 步防坍缩保底）。"
        ),
    },
    {
        "id": "muon_style",
        "row": 2,
        "label": "⚡ Muon 加速(>=12G)",
        "bg": "#313244",
        "fg_key": "accent_green",
        "active_bg_key": "accent_green",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": "4",
            "epochs": "5",
            "rank": "32",
            "alpha": "32.0",
            "algo": "lora",
            "dora": False,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "1.0",
            "lr": "1e-4",
            "grad_acc": "1",
            "timestep_mu": "-0.45",
            "optimizer": "muon",
            "weight_decay": "0.02",
            "text_drop": "0.10",
            "ema": "0.0",
            "lora_name": "my_muon_style",
            "trigger_word": "",
        },
        "dialog_title": "⚡ Muon 正交优化器加速预设已加载",
        "dialog_message": (
            "已启用上游 Newton-Schulz 正交矩阵更新【Muon 优化器】方案：\n"
            "• 正交收敛：引擎内置 0.2*sqrt(max(m,n)) RMS 尺度对齐，LR=1e-4（双卡 T4 自动升至 1.3e-4）+ Weight Decay=0.02\n"
            "• 步数精简：Repeats=4 / Epochs=5（20 次/图，同质量收敛比标准 AdamW 快约 25%~35%）+ mu=-0.45 + Text Drop=0.10\n"
            "• 硬件提示：Muon 要求优化器状态驻留 GPU（自动禁用 optcpu/gradoff），强烈推荐本地 >=12GB 显存或云端 T4 运行；6G/8G 显卡建议改用【🎨 艺术画风】或【💎 LoKr 满秩画风】。"
        ),
    },
]

_PRESET_MAP = {p["id"]: p for p in PRESETS}
PRESET_ORDER = [p["id"] for p in PRESETS]

# 仅当用户尚未自定义名称/触发词（仍处于系统占位状态）时才填入模板默认名，防止覆盖用户已输入的项目名与触发词
DEFAULT_PLACEHOLDER_LORA_NAMES = frozenset(
    {
        "",
        "my_lora",
        "my_moss_lora",
        "multi_char_pack",
        "multi_style_pack",
        "my_char_antifit",
    }
    | {str(p["values"]["lora_name"]) for p in PRESETS}
)

DEFAULT_LEGACY_TRIGGERS = frozenset(
    {
        "",
        "in the style of my_art_style",
        "in the style of my_lokr_style",
        "in the style of my_muon_style",
        "my_char",
        "my_character_char",
        "my_concept",
        "my_concept_v1",
        "my_char_v1",
    }
)


def get_preset_by_id(preset_id: str):
    """按 ID 检索预设定义字典。"""
    return _PRESET_MAP.get(preset_id)


def recommend_min_steps(algo: str = "lora", multi_subset: bool = False) -> int:
    """返回防止大等效 Batch 下小图集优化器更新次数过少导致欠拟合的安全最低步数。"""
    if multi_subset:
        return MIN_RECOMMENDED_STEPS_MULTI
    if str(algo or "lora").strip().lower() == "lokr":
        return MIN_RECOMMENDED_STEPS_LOKR
    return MIN_RECOMMENDED_STEPS_SINGLE


def resolve_preset_for_target(preset_id: str, target_compute: str = "local") -> tuple:
    """
    根据目标算力（local | kaggle_t4x2 | kaggle_t4x1）返回 (resolved_values_dict, notes_list)。
    双卡 T4 DDP 单步等效吞吐较高（1024px 下 4 图/步），自动将 GradAcc 归 1 并微幅补偿学习率。
    """
    preset = get_preset_by_id(preset_id)
    if not preset:
        return {}, []
    vals = dict(preset["values"])
    notes = []
    target_key = (target_compute or "local").strip().lower()
    if target_key == "kaggle_t4x2" and preset_id in CLOUD_T4X2_OVERRIDES:
        ov = CLOUD_T4X2_OVERRIDES[preset_id]
        old_ga = vals.get("grad_acc", "1")
        old_lr = vals.get("lr", "1e-4")
        vals.update(ov)
        if old_ga != vals.get("grad_acc"):
            notes.append(f"Kaggle 双卡 T4 DDP 单步已并发 4 张图，GradAcc 已自动由 {old_ga} 归 1。")
        if old_lr != vals.get("lr"):
            notes.append(f"匹配双卡大等效 Batch (√B 缩放)，学习率已由 {old_lr} 微调至 {vals['lr']}。")
        return vals, notes
    if target_key == "kaggle_t4x1":
        return vals, ["已按 Kaggle 单卡 T4（MAX_TOKENS=8192）核算单步吞吐与步数。"]
    return vals, notes


def recommend_save_every_and_ema(rec_steps: int, base_ema: str) -> tuple:
    """
    针对云端短总步数场景，自动推荐合理的 SaveEvery 间隔与短程 EMA 衰减率。
    返回 (save_every_str, adjusted_ema_str, note_or_empty)。
    """
    steps_int = max(1, int(rec_steps or 100))
    if steps_int <= 120:
        save_every = "30"
    elif steps_int <= 260:
        save_every = "50"
    elif steps_int <= 450:
        save_every = "80"
    else:
        save_every = "100"

    ema_str = str(base_ema or "0.0").strip()
    if ema_str.lower() == "power":
        return save_every, "power", "Power EMA 已启用（EDM2 双尺度 sigma_rel=[0.05, 0.10] 自适应平滑，不受短总步数惩罚）。"
    note = ""
    try:
        ema_val = float(ema_str)
    except Exception:
        ema_val = 0.0
    if ema_val >= 0.985 and steps_int < 250:
        ema_str = "0.97"
        note = f"总步数约 {steps_int} 步 (<250)，已将手动开启的 EMA 由 {base_ema} 下调至 0.97 以缩短平滑预热期。"
    return save_every, ema_str, note
