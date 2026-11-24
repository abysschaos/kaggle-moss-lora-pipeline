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
        ema = float(str(raw["ema"]).strip())
        text_drop = float(str(raw["text_drop"]).strip())
        seed = int(float(str(raw["seed"]).strip()))

        if repeats < 1 or epochs < 1 or steps < 1 or rank < 1 or lr_val <= 0 or grad_acc < 1:
            raise ValueError("Repeats/Epochs/Steps/Rank/GradAcc 与 LR 必须为正数")
        if lokr_factor == 0 or lokr_factor < -1:
            raise ValueError("LoKr Factor 必须为 -1（自动）或 >= 1 的整数（如 4/8/16）")
        if not (0.0 <= rank_drop < 1.0 and 0.0 <= mod_drop < 1.0 and 0.0 <= in_drop < 1.0):
            raise ValueError("Dropout 概率必须在 [0.0, 1.0) 之间")
        if weight_decay < 0.0 or not (0.0 <= ema < 1.0) or not (0.0 <= text_drop <= 1.0):
            raise ValueError("WeightDecay>=0、EMA 需在 [0.0, 1.0)、TextDrop 需在 [0.0, 1.0]")

        cleaned = {
            **raw,
            "lora_name": (str(raw.get("lora_name", "")).strip() or "my_lora"),
            "trigger_word": str(raw.get("trigger_word", "")).strip(),
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
    ]

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
