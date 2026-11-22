"""Declarative preset definitions for basic and v1.5+ specialized LoRA training templates."""

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
    "text_drop": "0.05",
    "flip": False,
    "seed": "0",
    "save_format": "both",
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
            "repeats": 3,
            "epochs": 5,
            "rank": 32,
            "alpha": 32.0,
            "algo": "lora",
            "dora": False,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "1.0",
            "lr": "1e-4",
            "grad_acc": "1",
            "timestep_mu": "-1.09",
            "lora_name": "my_art_style",
            "trigger_word": "in the style of my_art_style",
        },
        "dialog_title": "模板切换",
        "dialog_message": (
            "已成功加载【艺术画风模板】(极速画风黄金配比)：\n"
            "LoRA名称: my_art_style | 算法: LoRA (attn-mlp) | LoRA+=1.0\n"
            "Repeats=3, Epochs=5, Rank=32, Alpha=32.0, LR=1e-4\n"
            "GradAcc=1, Mu=-1.09 (按真实等效Batch计步，40图约600步)"
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
            "repeats": 6,
            "epochs": 4,
            "rank": 32,
            "alpha": 32.0,
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "2.0",
            "lr": "8e-5",
            "grad_acc": "2",
            "timestep_mu": "0.35",
            "lora_name": "my_character",
            "trigger_word": "my_character_char",
        },
        "dialog_title": "模板切换",
        "dialog_message": (
            "已成功加载【单人角色模板】(高精度五官服饰版)：\n"
            "LoRA名称: my_character | 算法: LoRA + DoRA (attn-mlp) | LoRA+=2.0\n"
            "Repeats=6, Epochs=4, Rank=32, Alpha=32.0, LR=8e-5\n"
            "GradAcc=2, Mu=0.35 (强化中低噪面部结构与细节一致)"
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
            "epochs": 4,
            "rank": 48,
            "alpha": 48.0,
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp+text-refiner",
            "loraplus_ratio": "2.0",
            "lr": "8e-5",
            "grad_acc": "2",
            "timestep_mu": "0.35",
            "lora_name": "multi_char_pack",
            "trigger_word": "",
        },
        "dialog_title": "多人物模板就绪",
        "dialog_message": (
            "已加载【多人物配比模板】并切换为多子集配比模式！\n"
            "• 架构配置：LoRA + DoRA (Rank=48, Alpha=48.0, attn-mlp+text-refiner, LoRA+=2.0)\n"
            "• 进阶调度：GradAcc=2 (稳梯度防串角), Mu=0.35 (精细五官服饰对齐)\n"
            "• 使用提示：请在【📂 数据与续炼】选项卡中添加各角色子集并点击【⚖️ 一键配平 Repeat】。"
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
            "epochs": 5,
            "rank": 48,
            "alpha": 48.0,
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp+modulation",
            "loraplus_ratio": "2.0",
            "lr": "1e-4",
            "grad_acc": "2",
            "timestep_mu": "-0.60",
            "lora_name": "multi_style_pack",
            "trigger_word": "",
        },
        "dialog_title": "多画风模板就绪",
        "dialog_message": (
            "已加载【多画风配比模板】并切换为多子集配比模式！\n"
            "• 架构配置：LoRA + DoRA (Rank=48, Alpha=48.0, attn-mlp+modulation, LoRA+=2.0)\n"
            "• 进阶调度：GradAcc=2 (降方差防风格糊化), Mu=-0.60 (兼顾宏观笔触与线条锐度)\n"
            "• 使用提示：请在【📂 数据与续炼】选项卡分别添加不同画风子目录并绑定唯一触发词！"
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
            "repeats": 5,
            "epochs": 4,
            "rank": 32,
            "alpha": 32.0,
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp+text-refiner",
            "loraplus_ratio": "2.0",
            "lr": "1e-4",
            "grad_acc": "2",
            "timestep_mu": "0.0",
            "lora_name": "my_concept",
            "trigger_word": "my_concept_v1",
        },
        "dialog_title": "概念/动作模板就绪",
        "dialog_message": (
            "已加载【概念/动作/材质模板】：\n"
            "• 架构配置：LoRA + DoRA (Rank=32, Alpha=32.0, attn-mlp+text-refiner, LoRA+=2.0)\n"
            "• 进阶调度：GradAcc=2, Mu=0.0 (全时间段对称采样，兼顾动态肢体与局部材质)\n"
            "• 建议图集保持主体与背景高度随机，仅目标动作/镜头/材质保持一致。"
        ),
    },
    {
        "id": "lokr_style",
        "row": 2,
        "label": "💎 LoKr 高还原画风",
        "bg": "#313244",
        "fg_key": "accent_peach",
        "active_bg_key": "accent_peach",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": 4,
            "epochs": 5,
            "rank": 32,
            "alpha": 32.0,
            "algo": "lokr",
            "lokr_factor": "8",
            "lokr_full_matrix": False,
            "lokr_decompose_both": False,
            "dora": False,
            "target_preset": "attn-mlp+modulation",
            "loraplus_ratio": "1.0",
            "lr": "1.5e-4",
            "grad_acc": "1",
            "timestep": "moss",
            "timestep_mu": "-1.09",
            "ema": "0.99",
            "lora_name": "my_lokr_style",
            "trigger_word": "in the style of my_lokr_style",
        },
        "dialog_title": "特色实战预设：💎 LoKr 高还原画风",
        "dialog_message": (
            "已加载【💎 LoKr 高还原画风预设】：\n"
            "• 算法：LoKr (Factor=8) + attn-mlp+modulation | Rank=32, Alpha=32.0\n"
            "• 调度：LR=1.5e-4, Mu=-1.09, EMA=0.99 (兼顾极小产出体积与高频笔触还原)"
        ),
    },
    {
        "id": "antifit_char",
        "row": 2,
        "label": "🛡️ 小图集防过拟合角色",
        "bg": "#313244",
        "fg_key": "accent_yellow",
        "active_bg_key": "accent_yellow",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": 6,
            "epochs": 4,
            "rank": 32,
            "alpha": 32.0,
            "algo": "lora",
            "dora": True,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "2.0",
            "lr": "6e-5",
            "grad_acc": "2",
            "timestep": "moss",
            "timestep_mu": "0.35",
            "rank_dropout": "0.1",
            "module_dropout": "0.05",
            "input_dropout": "0.0",
            "text_drop": "0.1",
            "ema": "0.99",
            "lora_name": "my_char_antifit",
            "trigger_word": "my_char_v1",
        },
        "dialog_title": "特色实战预设：🛡️ 小图集防过拟合角色",
        "dialog_message": (
            "已加载【🛡️ 小图集防过拟合角色预设】(专治 10~20 张小样本衣服/背景死锁)：\n"
            "• 架构：LoRA + DoRA (attn-mlp, LoRA+=2.0) | LR=6e-5, GradAcc=2, Mu=0.35\n"
            "• 防过拟合：RankDrop=0.1, ModDrop=0.05, InDrop=0.0(DoRA锁定), TextDrop=0.1, EMA=0.99"
        ),
    },
    {
        "id": "muon_style",
        "row": 2,
        "label": "⚡ Muon 画风加速",
        "bg": "#313244",
        "fg_key": "accent_green",
        "active_bg_key": "accent_green",
        "multi_subset": False,
        "align_steps": True,
        "values": {
            "repeats": 3,
            "epochs": 5,
            "rank": 32,
            "alpha": 32.0,
            "algo": "lora",
            "dora": False,
            "target_preset": "attn-mlp",
            "loraplus_ratio": "1.0",
            "optimizer": "muon",
            "lr": "1e-4",
            "grad_acc": "1",
            "timestep": "moss",
            "timestep_mu": "-1.09",
            "lora_name": "my_muon_style",
            "trigger_word": "in the style of my_muon_style",
        },
        "dialog_title": "特色实战预设：⚡ Muon 画风加速",
        "dialog_message": (
            "已加载【⚡ Muon 画风加速预设】：\n"
            "• 优化器：Muon (正交矩阵动量加速画风早期收敛)，后端自动切入 GPU 状态安全流式显存档位\n"
            "• 配置：LoRA (Rank=32, Alpha=32.0), LR=1e-4, Mu=-1.09"
        ),
    },
]

_PRESET_MAP = {p["id"]: p for p in PRESETS}

# 针对 Kaggle T4 x2 DDP（MAX_TOKENS=9216 x 2卡, 1024px=4图/步）的预设智能派生覆盖：
# Batch 放大后默认归 1 GradAcc（单步已 4 图足够稳定），并按平方根法则温和调整学习率。
CLOUD_T4X2_OVERRIDES = {
    "style": {"grad_acc": "1", "lr": "1.2e-4"},
    "character": {"grad_acc": "1", "lr": "1e-4"},
    "multi_char": {"grad_acc": "1", "lr": "1e-4"},
    "multi_style": {"grad_acc": "1", "lr": "1.2e-4"},
    "concept": {"grad_acc": "1", "lr": "1.2e-4"},
    "lokr_style": {"grad_acc": "1", "lr": "1.8e-4"},
    "antifit_char": {"grad_acc": "1", "lr": "8e-5"},
    "muon_style": {"grad_acc": "1", "lr": "1.3e-4"},
}


def get_preset_by_id(preset_id: str):
    """按 ID 检索预设定义字典。"""
    return _PRESET_MAP.get(preset_id)


def recommend_save_every_and_ema(rec_steps: int, base_ema: str = "0.0") -> tuple:
    """根据推荐总步数自适应建议存档间隔与短步数 EMA，避免总步数骤降后检查点过稀或 EMA 滞后。"""
    steps = max(0, int(rec_steps or 0))
    if steps <= 0:
        save_every = "100"
    elif steps <= 120:
        save_every = "30"
    elif steps <= 320:
        save_every = "50"
    else:
        save_every = "100"

    ema_raw = str(base_ema if base_ema is not None else "0.0").strip() or "0.0"
    try:
        ema_val = float(ema_raw)
    except Exception:
        ema_val = 0.0

    if ema_val >= 0.985 and 0 < steps < 250:
        return save_every, "0.97", f"短步数({steps} < 250)已将 EMA {ema_raw} 调平至 0.97 防收敛滞后"
    return save_every, ema_raw, ""


def resolve_preset_for_target(preset_id: str, target_compute: str = "local") -> tuple:
    """按目标算力环境返回 (派生后的参数字典, 云端自适应说明列表)。"""
    preset = get_preset_by_id(preset_id)
    if not preset:
        return {}, []

    merged = dict(preset.get("values", {}))
    tc = (target_compute or "local").strip().lower()
    if tc == "kaggle_t4x2":
        overrides = CLOUD_T4X2_OVERRIDES.get(preset_id, {})
        merged.update(overrides)
        notes = [
            f"Kaggle T4×2 DDP 单步并发 4 图（MAX_TOKENS=9216×2卡），已自动将 GradAcc 设为 {merged.get('grad_acc', '1')}，"
            f"并按 √B 法则温和校准 LR={merged.get('lr', '1e-4')}。"
        ]
        return merged, notes
    if tc == "kaggle_t4x1":
        return merged, ["已按 Kaggle 单卡 T4（MAX_TOKENS=8192）核算单步吞吐与步数。"]
    return merged, []
