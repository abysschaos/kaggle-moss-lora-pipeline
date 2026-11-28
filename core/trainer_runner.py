"""Parameter validation, CLI command builder, graceful pause, process tree kill, and checkpoint rescue."""

import os
import subprocess
from pathlib import Path

from .config_store import resolve_cli_python

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def validate_training_params(raw: dict) -> tuple:
    """校验并清洗基础与进阶训练超参数，返回 (cleaned_dict, error_message_or_none)。"""
    try:
        dora = bool(raw.get("dora", False))
        repeats = int(raw["repeats"])
        epochs = int(raw["epochs"])
        steps = int(raw["steps"])
        rank = int(raw["rank"])
        alpha = float(raw["alpha"])
        lr = str(raw["lr"]).strip()
        lr_val = float(lr)
        save_every = int(raw["save_every"])
        grad_acc = int(float(str(raw["grad_acc"]).strip()))
        mu_raw = str(raw.get("timestep_mu", "")).strip()
        timestep_mu = float(mu_raw) if mu_raw else 0.0
        lokr_factor = int(str(raw["lokr_factor"]).strip())
        rank_drop = float(str(raw["rank_dropout"]).strip())
        mod_drop = float(str(raw["module_dropout"]).strip())
        in_drop = 0.0 if dora else float(str(raw["input_dropout"]).strip())
        warmup_raw = str(raw["warmup_steps"]).strip()
        warmup_steps = -1 if warmup_raw.lower() == "auto" else int(float(warmup_raw))
        weight_decay = float(str(raw["weight_decay"]).strip())
        ema_raw = str(raw["ema"]).strip()
        if ema_raw.lower() == "power":
            ema = "power"
        else:
            ema_num = float(ema_raw)
            if not (0.0 <= ema_num < 1.0):
                raise ValueError("EMA 需为 'power' 或 [0.0, 1.0) 内的衰减率")
            ema = ema_num
        text_drop = float(str(raw["text_drop"]).strip())
        seed = int(float(str(raw["seed"]).strip()))

        algo = str(raw.get("algo", "lora")).strip().lower()
        if algo not in ("lora", "loha", "lokr", "oft", "tlora"):
            raise ValueError(f"不支持的适配器算法：{algo}")
        optimizer = str(raw.get("optimizer", "adamw")).strip().lower()
        if optimizer not in ("adamw", "adamw16", "adamw8bit_kernel", "muon"):
            raise ValueError(f"不支持的优化器：{optimizer}")
        vram_preset = str(raw.get("vram_preset", "auto")).strip().lower()
        if vram_preset not in ("auto", "6g", "8g", "12g", "16g", "24g"):
            raise ValueError(f"不支持的显存档位：{vram_preset}")
        text_device = str(raw.get("text_device", "cpu")).strip().lower()
        if text_device not in ("cpu", "cuda", "cuda-ring", "auto"):
            raise ValueError(f"不支持的 Text Encoder 设备：{text_device}")

        caption_template = str(raw.get("caption_template", "auto")).strip().lower() or "auto"
        if caption_template not in ("auto", "nl", "mix"):
            caption_template = "auto"
        caption_p_nl = float(str(raw.get("caption_p_nl", "0.75")).strip() or "0.75")
        caption_variants = int(float(str(raw.get("caption_variants", "4")).strip() or "4"))
        if not (0.0 <= caption_p_nl <= 1.0) or caption_variants < 1:
            raise ValueError("caption_p_nl 需在 [0.0, 1.0]，caption_variants 需为 >=1 的整数")

        compile_mode = str(raw.get("compile_mode", "off")).strip().lower() or "off"
        if compile_mode not in ("off", "block"):
            compile_mode = "off"

        step_release = str(raw.get("step_release", "none")).strip().lower() or "none"
        if step_release not in ("none", "empty_cache"):
            step_rel_n = int(float(step_release))
            if step_rel_n <= 0:
                raise ValueError("step_release 必须为 none、empty_cache 或正整数间隔")
            step_release = str(step_rel_n)

        sample_every = int(float(str(raw.get("sample_every", "0")).strip() or "0"))
        if sample_every < 0:
            raise ValueError("sample_every 不能为负数")
        sample_prompt = str(raw.get("sample_prompt", "")).strip()

        teacher_mode = str(raw.get("teacher_mode", "none")).strip().lower() or "none"
        if teacher_mode not in ("none", "weights"):
            teacher_mode = "none"
        distill_weight = float(str(raw.get("distill_weight", "0.05")).strip() or "0.05")
        if distill_weight < 0.0:
            raise ValueError("distill_weight 不能为负数")

        if repeats < 1 or epochs < 1 or steps < 1 or rank < 1 or lr_val <= 0 or grad_acc < 1:
            raise ValueError("Repeats/Epochs/Steps/Rank/GradAcc 与 LR 必须为正数")
        if lokr_factor == 0 or lokr_factor < -1:
            raise ValueError("LoKr Factor 必须为 -1（自动）或 >= 1 的整数（如 4/8/16）")
        if not (0.0 <= rank_drop < 1.0 and 0.0 <= mod_drop < 1.0 and 0.0 <= in_drop < 1.0):
            raise ValueError("Dropout 概率必须在 [0.0, 1.0) 之间")
        if weight_decay < 0.0 or not (0.0 <= text_drop <= 1.0):
            raise ValueError("WeightDecay>=0、TextDrop 需在 [0.0, 1.0]")

        cleaned = {
            **raw,
            "lora_name": (str(raw.get("lora_name", "")).strip() or "my_lora"),
            "trigger_word": str(raw.get("trigger_word", "")).strip(),
            "algo": algo,
            "optimizer": optimizer,
            "vram_preset": vram_preset,
            "text_device": text_device,
            "repeats": repeats,
            "epochs": epochs,
            "steps": steps,
            "rank": rank,
            "alpha": alpha,
            "lr": lr,
            "save_every": save_every,
            "grad_acc": grad_acc,
            "timestep_mu": timestep_mu,
            "lokr_factor": lokr_factor,
            "rank_dropout": rank_drop,
            "module_dropout": mod_drop,
            "input_dropout": in_drop,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "ema": ema,
            "text_drop": text_drop,
            "seed": seed,
            "dora": dora,
            "caption_template": caption_template,
            "caption_p_nl": caption_p_nl,
            "caption_variants": caption_variants,
            "int8_compute": bool(raw.get("int8_compute", False)),
            "compile_mode": compile_mode,
            "step_release": step_release,
            "sample_every": sample_every,
            "sample_prompt": sample_prompt,
            "teacher_mode": teacher_mode,
            "distill_weight": distill_weight,
        }
        return cleaned, None
    except Exception as err:
        return {}, f"请检查核心或进阶超参数是否填写了合法数字：\n{err}"


def build_train_command(cleaned: dict, mode_code: str, subsets_manifest_path=None) -> list:
    """根据清洗后的参数组装完整的 train_local.py CLI 启动命令。"""
    cmd = [
        resolve_cli_python(), "-u", "train_local.py",
        "--data_dir", str(cleaned["data_dir"]),
        "--comfy_models_dir", str(cleaned["comfy_models_dir"]),
        "--output_dir", str(cleaned["output_dir"]),
        "--lora_name", cleaned["lora_name"],
        "--trigger_word", cleaned["trigger_word"],
        "--repeats", str(cleaned["repeats"]),
        "--epochs", str(cleaned["epochs"]),
        "--steps", str(cleaned["steps"]),
        "--rank", str(cleaned["rank"]),
        "--alpha", str(cleaned["alpha"]),
        "--algo", str(cleaned["algo"]),
        "--lokr_factor", str(cleaned["lokr_factor"]),
        "--input_dropout", str(cleaned["input_dropout"]),
        "--rank_dropout", str(cleaned["rank_dropout"]),
        "--module_dropout", str(cleaned["module_dropout"]),
        "--target_preset", str(cleaned["target_preset"]),
        "--loraplus_ratio", str(cleaned["loraplus_ratio"]),
        "--lr", str(cleaned["lr"]),
        "--optimizer", str(cleaned["optimizer"]),
        "--lr_scheduler", str(cleaned["lr_scheduler"]),
        "--warmup_steps", str(cleaned["warmup_steps"]),
        "--weight_decay", str(cleaned["weight_decay"]),
        "--wd_rules", str(cleaned["wd_rules"]),
        "--ema", str(cleaned["ema"]),
        "--precision", str(cleaned["precision"]),
        "--vram_preset", str(cleaned["vram_preset"]),
        "--resolution", str(cleaned["resolution"]),
        "--grad_acc", str(cleaned["grad_acc"]),
        "--timestep", str(cleaned["timestep"]),
        "--timestep_mu", str(cleaned["timestep_mu"]),
        "--text_drop", str(cleaned["text_drop"]),
        "--seed", str(cleaned["seed"]),
        "--save_format", str(cleaned["save_format"]),
        "--base_precision", str(cleaned["base_precision"]),
        "--tune", str(cleaned["tune"]),
        "--overlong", "drop_tokens",
        "--variant", "auto",
        "--train_mode", mode_code,
        "--save_every", str(cleaned["save_every"]),
        "--text_device", str(cleaned["text_device"]),
        "--vae_device", str(cleaned["vae_device"]),
        "--caption_template", str(cleaned.get("caption_template", "auto")),
        "--caption_p_nl", str(cleaned.get("caption_p_nl", 0.75)),
        "--caption_variants", str(cleaned.get("caption_variants", 4)),
        "--compile_mode", str(cleaned.get("compile_mode", "off")),
        "--step_release", str(cleaned.get("step_release", "none")),
        "--sample_every", str(cleaned.get("sample_every", 0)),
        "--teacher_mode", str(cleaned.get("teacher_mode", "none")),
        "--distill_weight", str(cleaned.get("distill_weight", 0.05)),
    ]

    if str(cleaned.get("sample_prompt", "")).strip():
        cmd.extend(["--sample_prompt", str(cleaned["sample_prompt"]).strip()])
    if cleaned.get("int8_compute"):
        cmd.append("--int8_compute")
    if cleaned.get("dora"):
        cmd.append("--dora")
    if cleaned.get("lokr_full_matrix"):
        cmd.append("--lokr_full_matrix")
    if cleaned.get("lokr_decompose_both"):
        cmd.append("--lokr_decompose_both")
    if cleaned.get("train_norms"):
        cmd.append("--train_norms")
    if cleaned.get("flip"):
        cmd.append("--flip")
    if subsets_manifest_path is not None:
        cmd.extend(["--subsets_json", str(subsets_manifest_path)])

    custom_resume_p = str(cleaned.get("resume_path", "")).strip()
    if mode_code == "resume" and custom_resume_p:
        if os.path.isfile(custom_resume_p) and custom_resume_p.lower().endswith(".safetensors"):
            idx = cmd.index("--train_mode")
            cmd[idx + 1] = "init_from"
            cmd.extend(["--init_from", custom_resume_p])
        else:
            cmd.extend(["--resume_from", custom_resume_p])
    elif mode_code == "init_from" and custom_resume_p:
        cmd.extend(["--init_from", custom_resume_p])

    for flag, key in (
        ("--transformer_path", "transformer_path"),
        ("--text_encoder_path", "text_encoder_path"),
        ("--vae_path", "vae_path"),
    ):
        val = str(cleaned.get(key, "")).strip()
        if val:
            cmd.extend([flag, val])

    if cleaned.get("copy_to_comfy_loras"):
        cmd.append("--copy_to_comfy_loras")

    return cmd


def request_graceful_pause(workspace_dir: Path) -> Path:
    """写入 .pause_request 触发当前训练步结束后的完整存盘与优雅暂停。"""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    pause_flag_file = workspace_dir / ".pause_request"
    pause_flag_file.write_text("pause", encoding="utf-8")
    return pause_flag_file


def force_kill_process_tree(pid: int) -> None:
    """Windows 下级联强制清理主训练进程及其全部子孙进程树（kogine/python）。"""
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
        check=False,
    )


def rescue_latest_weights(lora_name: str, output_dir: str, comfy_models_dir: str, copy_to_comfy: bool) -> int:
    """强制中断后自动抢救最新已存盘的阶段权重至专属输出文件夹。"""
    try:
        import train_local

        dest_dir = Path(output_dir) / (lora_name.strip() or "my_lora")
        _, latest_step = train_local.export_and_sync_weights(
            Path("./KohakuMossTrainer").resolve(),
            lora_name.strip() or "my_lora",
            dest_dir,
            comfy_models_dir,
            copy_to_comfy,
            quiet=True,
        )
        return int(latest_step or 0)
    except Exception:
        return 0
