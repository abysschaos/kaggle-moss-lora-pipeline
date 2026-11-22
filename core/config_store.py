"""Configuration store, hardware probing, path helpers, and Token-Packing math calculations."""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VRAM_MAX_TOKENS = {"6g": 4352, "8g": 4352, "12g": 6144, "16g": 8192}


def load_app_version(fallback: str = "1.6.0") -> str:
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
) -> tuple:
    """
    计算当前单图集/多子集的真实总曝光次数，以及基于 Token 打包与梯度累积等效 Batch 的推荐步数。
    返回 (is_multi, img_count, factor, ep, total_exposures, rec, eff_batch, imgs_per_pack, grad_acc, mu, divisor)
    """
    max_tokens = resolve_max_tokens_budget(vram_preset, detected_vram_gb)
    est_tokens_per_img = (max(256, int(resolution)) // 16) ** 2 + 160
    imgs_per_pack = max(1, max_tokens // est_tokens_per_img)
    acc = max(1, int(grad_acc))
    eff_batch = imgs_per_pack * acc
    divisor = max(1, eff_batch)
    ep = int(epochs)
    mu = float(timestep_mu)

    if multi_subset:
        total_imgs = 0
        epoch_exp = 0
        for sub in subsets:
            c = count_images_in_dir(sub.get("path", ""))
            r = max(1, int(sub.get("repeats", 1)))
            total_imgs += c
            epoch_exp += c * r
        total_exposures = epoch_exp * ep
        rec = max(1, total_exposures // divisor)
        return True, total_imgs, epoch_exp, ep, total_exposures, rec, eff_batch, imgs_per_pack, acc, mu, divisor

    dpath = Path(data_dir)
    img_count = count_images_in_dir(str(dpath)) if dpath.exists() else 0
    rep = int(repeats)
    total_exposures = img_count * rep * ep
    rec = max(1, total_exposures // divisor)
    return False, img_count, rep, ep, total_exposures, rec, eff_batch, imgs_per_pack, acc, mu, divisor
