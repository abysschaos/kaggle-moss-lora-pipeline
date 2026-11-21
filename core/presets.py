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


def get_preset_by_id(preset_id: str):
    """按 ID 检索预设定义字典。"""
    return _PRESET_MAP.get(preset_id)
