"""Configuration store, hardware probing, path helpers, and Token-Packing math calculations."""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .presets import recommend_min_steps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VRAM_MAX_TOKENS = {"6g": 4352, "8g": 4352, "12g": 6144, "16g": 8192}
COMPUTE_TARGET_PROFILES = {
    "local": {"label": "💻 本机显卡", "active_gpus": 1, "max_tokens": 0, "kaggle_accelerator": None},
    "kaggle_t4x2": {
        "label": "☁️ Kaggle 双卡 T4 DDP",
        "active_gpus": 2,
        "max_tokens": 9216,
        "kaggle_accelerator": "NvidiaTeslaT4",
    },
    "kaggle_t4x1": {
        "label": "⚡ Kaggle 单卡 T4",
        "active_gpus": 1,
        "max_tokens": 8192,
        "kaggle_accelerator": "default_gpu",
    },
}


def load_app_version(fallback: str = "1.8.5") -> str:
    """读取项目根目录 VERSION 单一版本信源，缺失时安全回退。"""
    ver_file = PROJECT_ROOT / "VERSION"
    try:
        ver = ver_file.read_text(encoding="utf-8").strip()
        if ver:
            return ver
    except Exception:
        pass
    return fallback


def is_trainer_ready(trainer_dir: Path) -> bool:
    """核验 KohakuMossTrainer 引擎目录是否完整（必须同时具备 pyproject.toml 与 src/kohakumoss）。"""
    return (trainer_dir / "pyproject.toml").is_file() and (trainer_dir / "src" / "kohakumoss").is_dir()


def resolve_cli_python() -> str:
    """当 GUI 通过 pythonw.exe 运行时，映射回同目录 python.exe 以保留子进程 stdout 实时管道。"""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        cli_py = exe.with_name("python.exe")
        if cli_py.is_file():
            return str(cli_py)
    return sys.executable


def probe_gpu_quick() -> tuple:
    """毫秒级探测 CUDA 显卡名称、显存与算力架构：优先 nvidia-smi 避免冷启动导入 PyTorch 耗时，失败时回退 torch，兜底保守按 8.0 GB。"""
    no_win = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            creationflags=no_win,
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                vram = float(parts[1]) / 1024.0
                major = int(parts[2].split(".")[0])
                bf16 = "支持原生 BF16" if major >= 8 else "仅 FP16"
                return f"{name} ({vram:.1f} GB, {bf16})", float(vram)
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024**3)
            bf16 = "支持原生 BF16" if int(props.major) >= 8 else "仅 FP16"
            return f"{name} ({vram:.1f} GB, {bf16})", float(vram)
        return "仅 CPU (未检测到 CUDA)", 8.0
    except Exception:
        return "GPU 未识别 (保守按 8G)", 8.0


def discover_default_comfy_models(saved_dir: str = "") -> str:
    """优先使用上次记忆的 ComfyUI 模型路径，缺失时自动宽域探测。"""
    if saved_dir and os.path.exists(saved_dir):
        return saved_dir
    try:
        import train_local

        discovered_models, _ = train_local.find_broad_comfy_paths()
        if discovered_models:
            return str(discovered_models[0])
    except Exception:
        pass
    return ""


def load_local_config(config_file: Path) -> dict:
    """读取本地私有配置文件（.gitignore 已忽略），损坏或缺失时返回空字典。"""
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_local_config(config_file: Path, cfg: dict) -> None:
    """持久化用户本机设置到 local_config.json。"""
    try:
        config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def open_in_explorer(target_path: str, is_file_hint: bool = False) -> None:
    """打开指定文件夹或文件（若目录不存在则先自动创建）。"""
    p = Path(target_path).resolve()
    if is_file_hint or p.is_file():
        if not p.exists():
            raise FileNotFoundError(f"文件不存在：{p}")
        os.startfile(str(p))
        return
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    os.startfile(str(p))


def count_images_in_dir(dir_path_str: str) -> int:
    """统计目录下的有效图片数量。"""
    try:
        p = Path(dir_path_str)
        if not p.is_dir():
            return 0
        return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS)
    except Exception:
        return 0


def resolve_effective_vram_label(vram_preset: str, detected_vram_gb: float = 8.0) -> tuple:
    """将显存档位 (auto/6g/8g/12g/16g) 解析为真实生效档位与单步 MAX_TOKENS 上限。"""
    preset = (vram_preset or "auto").strip().lower()
    if preset == "auto":
        if detected_vram_gb <= 10.0:
            eff_preset = "8g"
        elif detected_vram_gb <= 14.0:
            eff_preset = "12g"
        else:
            eff_preset = "16g"
    else:
        eff_preset = preset
    return eff_preset, VRAM_MAX_TOKENS.get(eff_preset, 4352)


def resolve_max_tokens_budget(vram_preset: str, detected_vram_gb: float = 8.0) -> int:
    """根据显存档位或检测到的显存推导单包最大 Token 上限（未知环境保守回退 4352）。"""
    _, max_tokens = resolve_effective_vram_label(vram_preset, detected_vram_gb)
    return max_tokens


def resolve_compute_target_budget(target_compute: str, vram_preset: str, detected_vram_gb: float = 8.0) -> tuple:
    """按目标算力档案（local / kaggle_t4x2 / kaggle_t4x1）解析 (eff_key, max_tokens_per_gpu, active_gpus, display_tag)。"""
    tc = (target_compute or "local").strip().lower()
    if tc == "kaggle_t4x2":
        prof = COMPUTE_TARGET_PROFILES["kaggle_t4x2"]
        return "kaggle_t4x2", prof["max_tokens"], prof["active_gpus"], "☁️ Kaggle T4×2 DDP"
    if tc == "kaggle_t4x1":
        prof = COMPUTE_TARGET_PROFILES["kaggle_t4x1"]
        return "kaggle_t4x1", prof["max_tokens"], prof["active_gpus"], prof["label"]

    raw_vram = (vram_preset or "auto").strip().lower()
    eff_vram, tok_budget = resolve_effective_vram_label(raw_vram, detected_vram_gb)
    vram_tag = f"auto→{eff_vram}" if raw_vram == "auto" else eff_vram
    return eff_vram, tok_budget, 1, f"💻 本机 {vram_tag}"


def compute_exposure_summary(
    *,
    multi_subset: bool,
    subsets: list,
    data_dir: str,
    repeats: int,
    epochs: int,
    vram_preset: str,
    detected_vram_gb: float,
    resolution: int,
    grad_acc: int,
    timestep_mu: float,
    active_gpus: int = 1,
    max_tokens_override: int = 0,
    img_count_override: int = 0,
    algo: str = "lora",
    min_steps_floor: int = None,
) -> tuple:
    """
    计算当前单图集/多子集的真实总曝光次数，以及基于 Token 打包、并行 GPU 数与梯度累积等效 Batch 的推荐步数。
    当有效图片 > 0 时，自动启用防欠拟合最低优化器更新步数保底（单图集 180 / LoKr 200 / 多子集 240），
    避免 10~25 张小图集在 Dual-T4 DDP (4 图/步) 下跌落至百步以内导致严重欠拟合。
    返回 (is_multi, img_count, factor, ep, total_exposures, rec, eff_batch, imgs_per_pack, grad_acc, mu, divisor)
    """
    max_tokens = int(max_tokens_override) if int(max_tokens_override) > 0 else resolve_max_tokens_budget(vram_preset, detected_vram_gb)
    est_tokens_per_img = (max(256, int(resolution)) // 16) ** 2 + 160
    imgs_per_pack = max(1, max_tokens // est_tokens_per_img)
    gpus = max(1, int(active_gpus))
    acc = max(1, int(grad_acc))
    eff_batch = imgs_per_pack * gpus * acc
    divisor = max(1, eff_batch)
    ep = int(epochs)
    mu = float(timestep_mu)
    floor_steps = (
        recommend_min_steps(algo=algo, multi_subset=multi_subset)
        if min_steps_floor is None
        else max(1, int(min_steps_floor))
    )

    if multi_subset:
        total_imgs = 0
        epoch_exp = 0
        for sub in subsets:
            if not sub.get("enabled", True):
                continue
            c = count_images_in_dir(sub.get("path", ""))
            r = max(1, int(sub.get("repeats", 1)))
            total_imgs += c
            epoch_exp += c * r
        total_exposures = epoch_exp * ep
        raw_rec = math.ceil(total_exposures / divisor) if total_exposures > 0 else 1
        rec = max(floor_steps, raw_rec) if total_exposures > 0 else 1
        return True, total_imgs, epoch_exp, ep, total_exposures, rec, eff_batch, imgs_per_pack, acc, mu, divisor

    dpath = Path(data_dir)
    img_count = int(img_count_override) if int(img_count_override) > 0 else (count_images_in_dir(str(dpath)) if dpath.exists() else 0)
    rep = int(repeats)
    total_exposures = img_count * rep * ep
    raw_rec = math.ceil(total_exposures / divisor) if total_exposures > 0 else 1
    rec = max(floor_steps, raw_rec) if total_exposures > 0 else 1
    return False, img_count, rep, ep, total_exposures, rec, eff_batch, imgs_per_pack, acc, mu, divisor


def _dir_size_bytes(path_obj: Path) -> int:
    total = 0
    if path_obj.is_file():
        try:
            return path_obj.stat().st_size
        except OSError:
            return 0
    if not path_obj.is_dir():
        return 0
    for fp in path_obj.rglob("*"):
        if fp.is_file():
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def cleanup_workspace_staging(project_root: Path = None, output_dir: str = "") -> dict:
    """
    安全精简临时工作区 (workspace/) 与空输出子目录：
    1. 若 workspace/kaggle_outputs/<name> 中尚有未归档到 output/<name> 的权重，先无损抢救归档；
    2. 清理过期暂存区 (kaggle_dataset_stage / kaggle_kernel_stage / init_from_stage / _trainer_stage / _trainer_extract / trainer_main.zip)、
       *.kaggle-partial、0 字节损坏文件，并精简 kaggle_outputs/* 冗余副本；
    3. 删除 output/ 下 0 文件的空子目录，严格保护 workspace/models 硬链接、workspace/temp 与非空成品目录。
    """
    from .kaggle_bridge import (
        _clean_partial_and_empty_downloads,
        _prune_kaggle_download_cache,
        extract_kaggle_outputs_from_dir,
    )

    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    ws_dir = root / "workspace"
    default_out = (root / "output").resolve()
    configured_out = Path(output_dir).resolve() if str(output_dir or "").strip() else default_out

    removed_items = []
    rescued_loras = []
    freed_bytes = 0

    # 1. 优先扫描并自愈归档 workspace/kaggle_outputs/* 中的云端成品，随后精简冗余缓存
    ko_root = ws_dir / "kaggle_outputs"
    if ko_root.is_dir():
        for dl_dir in sorted(ko_root.iterdir()):
            if not dl_dir.is_dir():
                continue
            lora_name = dl_dir.name
            target_out_lora = configured_out / lora_name
            has_valid_out = target_out_lora.is_dir() and any(
                f.is_file() and f.stat().st_size > 0 for f in target_out_lora.rglob("*.safetensors")
            )
            has_cloud_artifacts = any(
                f.is_file() and f.stat().st_size > 0 and f.suffix.lower() in (".safetensors", ".zip")
                for f in dl_dir.rglob("*")
            )
            before_sz = _dir_size_bytes(dl_dir)
            if not has_valid_out and has_cloud_artifacts:
                res = extract_kaggle_outputs_from_dir(
                    dl_dir,
                    lora_name=lora_name,
                    output_dir=str(configured_out),
                    prune_download_dir=True,
                )
                if res.get("ok"):
                    rescued_loras.append(lora_name)
            else:
                _clean_partial_and_empty_downloads(dl_dir, target_out_lora)
                _prune_kaggle_download_cache(dl_dir, target_out_lora)
            after_sz = _dir_size_bytes(dl_dir)
            if before_sz > after_sz:
                diff = before_sz - after_sz
                freed_bytes += diff
                removed_items.append(f"kaggle_outputs/{lora_name} 冗余副本 (-{diff / (1024**2):.1f} MB)")

    # 2. 清理过期暂存区目录与临时包（严格避开 models/ 与 temp/）
    stale_targets = [
        "kaggle_dataset_stage",
        "kaggle_kernel_stage",
        "init_from_stage",
        "_trainer_stage",
        "_trainer_extract",
        "_trainer_preserve",
        "_trainer_clone_stage",
        "trainer_main.zip",
    ]
    if ws_dir.is_dir():
        for rel_name in stale_targets:
            cand = ws_dir / rel_name
            if not cand.exists() and not cand.is_symlink():
                continue
            sz = _dir_size_bytes(cand)
            try:
                if cand.is_dir():
                    shutil.rmtree(cand, ignore_errors=True)
                else:
                    cand.unlink(missing_ok=True)
                freed_bytes += sz
                removed_items.append(f"workspace/{rel_name} (-{sz / (1024**2):.2f} MB)")
            except OSError:
                pass

        for fp in list(ws_dir.rglob("*")):
            if not fp.is_file():
                continue
            try:
                rel_parts = fp.relative_to(ws_dir).parts
            except ValueError:
                continue
            if rel_parts and rel_parts[0].lower() in ("models", "temp"):
                continue
            is_partial = fp.name.endswith(".kaggle-partial")
            is_zero = (fp.stat().st_size == 0) and (not rel_parts or rel_parts[0].lower() != "dataset")
            if is_partial or is_zero:
                try:
                    fp.unlink(missing_ok=True)
                    removed_items.append(f"workspace/{fp.relative_to(ws_dir).as_posix()}")
                except OSError:
                    pass

    # 3. 清理 output/ 目录下 0 文件的空子目录，严格保留任何含文件成品目录
    out_roots_to_scan = [default_out]
    if configured_out != default_out:
        out_roots_to_scan.append(configured_out)

    for out_root in out_roots_to_scan:
        if not out_root.is_dir():
            continue
        for sub in sorted(out_root.iterdir()):
            if not sub.is_dir():
                continue
            has_any_file = any(item.is_file() for item in sub.rglob("*"))
            if not has_any_file:
                try:
                    shutil.rmtree(sub, ignore_errors=True)
                    removed_items.append(f"{out_root.name}/{sub.name} (空目录)")
                except OSError:
                    pass

    return {
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024**2), 2),
        "removed_items": removed_items,
        "rescued_loras": rescued_loras,
    }
