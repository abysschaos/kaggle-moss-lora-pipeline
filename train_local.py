import os
import sys
import re
import glob
import json
import shutil
import argparse
import subprocess
import time
from pathlib import Path

def print_banner(msg: str):
    width = 78
    print("=" * width)
    print(f">>> {msg}")
    print("=" * width)

def sanitize_path(p: str) -> str:
    """标准化路径为 POSIX 正斜杠格式，防止在生成的 Python 配置文件中发生反斜杠转义错误。"""
    return Path(p).resolve().as_posix()

def probe_local_models(comfy_models_dir: str = None) -> dict:
    """智能检索本地已有的 MOSS-Image 底模、文本编码器与 VAE，实现零下载秒级就绪。"""
    search_dirs = []
    if comfy_models_dir and os.path.exists(comfy_models_dir):
        search_dirs.append(Path(comfy_models_dir))
    
    # 默认预设探测路径
    default_comfy = Path("C:/dowmload idm/ComfyUI-aki-v3.2(1)/ComfyUI-aki-v3.2/ComfyUI/models")
    if default_comfy.exists() and default_comfy not in search_dirs:
        search_dirs.append(default_comfy)

    local_models = Path("./models").resolve()
    if local_models.exists() and local_models not in search_dirs:
        search_dirs.append(local_models)

    found = {
        "transformer": None,
        "text_encoder": None,
        "vae": None,
        "base_weights_dir": None
    }

    # 1. 查找 Transformer (优先选择 MOSS-Image-heal 修复版)
    heal_candidates = [
        "**/moss_image_heal_bf16.safetensors",
        "**/moss_image_heal_fp16.safetensors",
        "**/moss_image_heal_int8_convrot.safetensors",
        "**/moss_image_base_bf16.safetensors",
        "**/moss_image_base.safetensors"
    ]
    for sdir in search_dirs:
        for pat in heal_candidates:
            matches = list(sdir.glob(pat))
            if matches:
                found["transformer"] = matches[0]
                break
        if found["transformer"]:
            break

    # 2. 查找 Text Encoder (Qwen3.5-9B)
    text_candidates = [
        "**/qwen3.5_9b_bf16.safetensors",
        "**/qwen3.5_9b_int8_convrot.safetensors",
        "**/qwen3.5_9b.safetensors"
    ]
    for sdir in search_dirs:
        for pat in text_candidates:
            matches = list(sdir.glob(pat))
            if matches:
                found["text_encoder"] = matches[0]
                break
        if found["text_encoder"]:
            break

    # 3. 查找 VAE (FLUX.2 VAE)
    vae_candidates = [
        "**/flux2-vae_small.safetensors",
        "**/flux2-vae.safetensors"
    ]
    for sdir in search_dirs:
        for pat in vae_candidates:
            matches = list(sdir.glob(pat))
            if matches:
                found["vae"] = matches[0]
                break
        if found["vae"]:
            break

    return found

def apply_compatibility_patches(trainer_dir: Path):
    """为本地环境注入全套自愈补丁，解决 PyTorch 2.11 加速器接口缺失与调度器防御断言。"""
    print_banner("Applying Environment and Compatibility Patches")
    
    # 1. 放宽 pyproject.toml 中的 torch 版本限制
    pyproject_path = trainer_dir / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text(encoding="utf-8")
        patched = re.sub(r'\"torch<[^\"]+\"', '\"torch>=2.0\"', content)
        pyproject_path.write_text(patched, encoding="utf-8")
        print(" [OK] Relaxed pyproject.toml torch version constraint.")

    # 2. 遍历源码替换 empty_host_cache 调用
    patched_count = 0
    for py_file in trainer_dir.rglob("*.py"):
        try:
            code = py_file.read_text(encoding="utf-8", errors="ignore")
            if "empty_host_cache" in code:
                patched_code = re.sub(
                    r'torch\.accelerator\.empty_host_cache\(\)',
                    "getattr(torch.accelerator, 'empty_host_cache', getattr(getattr(torch, 'accelerator', None), 'empty_cache', getattr(torch.cuda, 'empty_cache', lambda: None)))()",
                    code
                )
                py_file.write_text(patched_code, encoding="utf-8")
                patched_count += 1
        except Exception as e:
            print(f" Warning patching {py_file}: {e}")
    print(f" [OK] Patched empty_host_cache calls across {patched_count} source files.")

    # 3. 修补 objective.py 调度器字典防护
    obj_check_file = trainer_dir / "src" / "kohakumoss" / "training" / "checks" / "objective.py"
    if obj_check_file.exists():
        try:
            content = obj_check_file.read_text(encoding="utf-8")
            patched = content.replace(
                'raw_end = sub.get("end", -1)',
                'raw_end = sub.get("end", -1) if isinstance(sub, dict) else -1'
            ).replace(
                'warmup = sub.get("warmup", 0)',
                'warmup = sub.get("warmup", 0) if isinstance(sub, dict) else 0'
            )
            obj_check_file.write_text(patched, encoding="utf-8")
            print(" [OK] Patched objective.py for schedule defensive checks.")
        except Exception as e:
            print(f" Warning patching objective.py: {e}")

def prepare_dataset_structure(source_data_dir: Path, workspace_data_dir: Path, repeats: int, lora_name: str, trigger_word: str = "") -> Path:
    """
    将散装图集自动标准化为 Kohya 规范的 <repeats>_<name> 结构。
    支持自定义触发词自动注入：若提供了 trigger_word 且原标注首部未包含，则自动前置注入。
    """
    target_folder = workspace_data_dir / f"{repeats}_{lora_name}"
    target_folder.mkdir(parents=True, exist_ok=True)
    
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    raw_images = [f for f in source_data_dir.iterdir() if f.suffix.lower() in img_exts]
    
    print(f"Source Dataset: Found {len(raw_images)} images in {source_data_dir}")
    if len(raw_images) == 0:
        raise ValueError(f"No image files found in {source_data_dir}")

    trigger_clean = trigger_word.strip() if trigger_word else ""
    if trigger_clean:
        print(f" [Trigger Injection] Active Trigger Word: '{trigger_clean}' (will be prepended to captions)")

    linked_count = 0
    for img in raw_images:
        dest_img = target_folder / img.name
        dest_txt = target_folder / f"{img.stem}.txt"
        src_txt = source_data_dir / f"{img.stem}.txt"

        # 1. 复制或硬链接图像文件
        if not dest_img.exists():
            try:
                os.link(str(img), str(dest_img))
            except Exception:
                shutil.copy2(str(img), str(dest_img))

        # 2. 处理自然语言标注文本与触发词自动注入
        caption_content = ""
        if src_txt.exists():
            caption_content = src_txt.read_text(encoding="utf-8", errors="ignore").strip()

        if trigger_clean:
            # 检查原标注首部是否已经包含该触发词，避免重复堆叠
            if not caption_content.lower().startswith(trigger_clean.lower()):
                caption_content = f"{trigger_clean}, {caption_content}" if caption_content else trigger_clean

        dest_txt.write_text(caption_content, encoding="utf-8")
        linked_count += 1

    print(f" [OK] Structured dataset directory ready: {target_folder} ({linked_count} image-text pairs)")
    return workspace_data_dir

def main():
    parser = argparse.ArgumentParser(description="OpenMOSS MOSS-Image Local Training Engine for Windows Consumer GPUs")
    parser.add_argument("--data_dir", type=str, default="C:/Users/白音华/Pictures/ness",
                        help="Path to folder containing source images and .txt captions")
    parser.add_argument("--comfy_models_dir", type=str, default="C:/dowmload idm/ComfyUI-aki-v3.2(1)/ComfyUI-aki-v3.2/ComfyUI/models",
                        help="Path to ComfyUI models directory containing diffusion_models, text_encoders, and vae")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Root directory to save final LoRA folders and checkpoints")
    parser.add_argument("--lora_name", "--run_name", type=str, default="ness_style", dest="lora_name",
                        help="Custom LoRA name and dedicated output folder name (e.g. makima_v1)")
    parser.add_argument("--trigger_word", type=str, default="",
                        help="Custom trigger word/phrase to prepend to captions (e.g. makima_char)")
    parser.add_argument("--copy_to_comfy_loras", action="store_true",
                        help="Automatically copy trained .safetensors to ComfyUI models/loras folder upon completion")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Kohya folder repeats per image (default: 10)")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Training epochs (total iterations = num_images * repeats * epochs)")
    parser.add_argument("--steps", type=int, default=600,
                        help="Total optimizer training steps (default: 600)")
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA Rank / Dim (default: 32)")
    parser.add_argument("--alpha", type=float, default=32.0,
                        help="LoRA Alpha (default: 32.0 for full strength 1.0)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16"],
                        help="Training precision (bf16 recommended for RTX 40/30 series)")
    parser.add_argument("--text_device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for offline text encoder embedding extraction (cpu recommended for 8GB cards to prevent OOM)")
    parser.add_argument("--vae_device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="Device for offline VAE latents extraction (default: cuda)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    workspace_dir = project_root / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # 建立以用户自定义 LoRA 名称命名的专属产出文件夹
    base_output_dir = Path(args.output_dir).resolve()
    lora_dedicated_folder = base_output_dir / args.lora_name
    lora_dedicated_folder.mkdir(parents=True, exist_ok=True)

    print_banner(f"OpenMOSS MOSS-Image Local LoRA Training Pipeline: {args.lora_name}")
    print(f"Target LoRA Name & Subfolder: {args.lora_name}")
    print(f"Custom Trigger Word:          {args.trigger_word if args.trigger_word else '(None, using original captions)'}")
    print(f"Dedicated Output Directory:   {lora_dedicated_folder}")
    print(f"OS: Windows | Target Precision: {args.precision.upper()} | Text Device: {args.text_device.upper()}")
    print(f"Hyperparameters: Steps={args.steps}, Dim={args.rank}, Alpha={args.alpha}, LR={args.lr}")

    # 1. 硬件与 PyTorch 环境预检
    try:
        import torch
        import torchvision
        print(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} | VRAM: {total_vram:.2f} GB")
            if total_vram < 10.0 and args.text_device == "cuda":
                print(">>> NOTICE: GPU VRAM is under 10GB. Switching text_device to CPU to prevent OOM!")
                args.text_device = "cpu"
        else:
            print("ERROR: CUDA device not detected! Exiting...")
            sys.exit(1)
    except Exception as e:
        print(f"Error checking PyTorch/CUDA environment: {e}")
        sys.exit(1)

    # 2. 智能检索本地已有的模型权重
    print_banner("Locating Base Models and Weights")
    weights_info = probe_local_models(args.comfy_models_dir)
    print(f"Found Transformer:  {weights_info['transformer']}")
    print(f"Found Text Encoder: {weights_info['text_encoder']}")
    print(f"Found VAE:          {weights_info['vae']}")

    if not weights_info["transformer"] or not weights_info["text_encoder"] or not weights_info["vae"]:
        raise FileNotFoundError(
            "Missing core weights! Ensure MOSS-Image-heal (or base), Qwen3.5-9B, and FLUX.2 VAE exist in your ComfyUI models folder."
        )

    # 构建统一模型映射目录 (便于 KohakuMossTrainer 结构化索引)
    models_stage_dir = workspace_dir / "models"
    diff_stage = models_stage_dir / "diffusion_models"
    text_stage = models_stage_dir / "text_encoders"
    vae_stage = models_stage_dir / "vae"
    for d in [diff_stage, text_stage, vae_stage]:
        d.mkdir(parents=True, exist_ok=True)

    target_transformer = diff_stage / weights_info["transformer"].name
    target_text = text_stage / weights_info["text_encoder"].name
    target_vae = vae_stage / weights_info["vae"].name

    for src, dst in [(weights_info["transformer"], target_transformer),
                     (weights_info["text_encoder"], target_text),
                     (weights_info["vae"], target_vae)]:
        if not dst.exists():
            try:
                os.link(str(src), str(dst))
            except Exception:
                shutil.copyfile(str(src), str(dst))

    # 3. 准备 Kohya 结构化数据集并注入自定义触发词
    print_banner("Preparing Dataset Structure with Custom Trigger Word")
    src_data_path = Path(args.data_dir).resolve()
    workspace_data_root = workspace_dir / "dataset"
    prepare_dataset_structure(src_data_path, workspace_data_root, args.repeats, args.lora_name, args.trigger_word)

    # 4. 检查并准备 KohakuMossTrainer 训练器
    trainer_dir = project_root / "KohakuMossTrainer"
    if not trainer_dir.exists():
        print_banner("Cloning KohakuMossTrainer Framework")
        subprocess.run(["git", "clone", "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git", str(trainer_dir)], check=True)
    
    apply_compatibility_patches(trainer_dir)

    # 确保本地依赖与可编辑模块就绪
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", str(trainer_dir)
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "kohaku-engine", "kohakuvault", "lycoris-lora", "anyschedule", "safetensors"
    ], check=True)

    # 5. 生成本地训练与离线缓存专属配置文件 (名称与 lora_name 强对齐)
    print_banner(f"Generating Local Configurations for '{args.lora_name}'")
    user_cfg_dir = trainer_dir / "configs" / "user"
    user_cfg_dir.mkdir(parents=True, exist_ok=True)

    sanitized_data_root = sanitize_path(workspace_data_root)
    sanitized_weights_dir = sanitize_path(models_stage_dir)
    transformer_filename = target_transformer.name

    data_cfg_code = f"""
data_root = "{sanitized_data_root}"
WEIGHTS = "{sanitized_weights_dir}"

DATASETS = [
    {{
        "source": "kohya",
        "path": data_root,
        "repeat_prefix": "first",
        "artist_from_folder": "name",
        "txt_as": "nl",
    }}
]
RESOLUTION_MODE = "resize"
TARGET_AREA = 1024 * 1024
UPSCALE = "never"
ASPECT_RANGE = (0.33, 3.0)
BATCHING = "tokens"
MAX_TOKENS = 16384
OVERLONG = "drop"
CAPTION = {{
    "template": "nl",
}}
CAPTION_VARIANTS = 1
TEXT_DROP = 0.05
LATENT_CACHE = "offline"
TEXT_CACHE = "offline"
FLIP = False
TEXT_ENCODER_KWARGS = {{"path": WEIGHTS, "device": "{args.text_device}"}}
IMAGE_ENCODER_KWARGS = {{"path": WEIGHTS, "device": "{args.vae_device}"}}
"""

    train_cfg_code = f"""
from kohakuengine import use_config

use_config("../moss/recipes/recipes_lora_16g_fp16.py")
use_config("{args.lora_name}_local_data.py")

RUN_NAME = "{args.lora_name}"
TRANSFORMER = "{transformer_filename}"
FP16_FIX = "none"
PRECISION = "{args.precision}"
DIM = {args.rank}
ALPHA = {args.alpha}
STEPS = {args.steps}
LR = {args.lr}
SAVE_MINUTES = 20
SAVE_FORMAT = "both"
SCHEDULE = {{
    "mode": "cosine",
    "warmup": 30,
    "end": -1,
    "min_value": 0.1
}}
"""

    data_cfg_file = user_cfg_dir / f"{args.lora_name}_local_data.py"
    train_cfg_file = user_cfg_dir / f"{args.lora_name}_local.py"

    data_cfg_file.write_text(data_cfg_code.strip(), encoding="utf-8")
    train_cfg_file.write_text(train_cfg_code.strip(), encoding="utf-8")
    print(f" [OK] Generated Data Config:  {data_cfg_file}")
    print(f" [OK] Generated Train Config: {train_cfg_file}")

    # 6. 执行离线双特征提取 (双离线缓存)
    print_banner(f"Step 1/2: Extracting Text Embeddings (Device: {args.text_device.upper()})")
    subprocess.run([
        "kogine", "run", "scripts/data/cache_text.py",
        "--config", f"configs/user/{args.lora_name}_local_data.py"
    ], cwd=str(trainer_dir), check=True)

    print_banner(f"Step 2/2: Extracting Image Latents (Device: {args.vae_device.upper()})")
    subprocess.run([
        "kogine", "run", "scripts/data/cache_latents.py",
        "--config", f"configs/user/{args.lora_name}_local_data.py"
    ], cwd=str(trainer_dir), check=True)

    # 7. 物理级清洗配置文件中的临时离线编码器旋钮
    print_banner("Sanitizing Configuration Knobs for moss.py Validation")
    raw_cfg = data_cfg_file.read_text(encoding="utf-8")
    sanitized_cfg = re.sub(r'TEXT_ENCODER_KWARGS\s*=\s*\{[^\}]+\}', '', raw_cfg)
    sanitized_cfg = re.sub(r'IMAGE_ENCODER_KWARGS\s*=\s*\{[^\}]+\}', '', sanitized_cfg)
    data_cfg_file.write_text(sanitized_cfg.strip() + "\n", encoding="utf-8")
    print(" [OK] Knobs TEXT_ENCODER_KWARGS and IMAGE_ENCODER_KWARGS cleanly sanitized.")

    # 8. 启动主训练循环
    print_banner(f"Starting {args.steps}-step Local LoRA Training: '{args.lora_name}' (Precision: {args.precision.upper()})")
    start_time = time.time()
    train_cmd = [
        "kogine", "run", "scripts/train/moss.py",
        "--config", f"configs/user/{args.lora_name}_local.py"
    ]
    print(f"Executing: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, cwd=str(trainer_dir), check=True)
    elapsed = time.time() - start_time
    print(f" [OK] Core training completed in {elapsed / 60:.2f} minutes ({elapsed / args.steps:.2f} s/step)!")

    # 9. 智能检索导出的 LoRA 权重，规范重命名并归档至专属子文件夹
    print_banner(f"Locating and Exporting LoRA Weights to '{lora_dedicated_folder}'")
    search_paths = list(trainer_dir.glob("runs/**/*.safetensors")) + list(trainer_dir.glob("checkpoints/**/*.safetensors"))
    exported_files = []
    
    for sf in set(search_paths):
        sf_str = sf.as_posix().lower()
        if "moss_image" in sf_str or "qwen" in sf_str or "flux2" in sf_str:
            continue
        
        # 匹配 ComfyUI 原生权重与 LyCORIS 标准权重并对齐用户自定义名称
        dest_filename = sf.name
        if "-comfy" in sf_str or sf.stem.endswith("_comfy"):
            dest_filename = f"{args.lora_name}-comfy.safetensors"
        elif "-lycoris" in sf_str or sf.stem.endswith("_lycoris"):
            dest_filename = f"{args.lora_name}-lycoris.safetensors"
        else:
            dest_filename = f"{args.lora_name}-{sf.name}"

        dst = lora_dedicated_folder / dest_filename
        shutil.copy2(str(sf), str(dst))
        exported_files.append(dst)
        print(f" Exported: {sf.name} -> {dst}")

    # 生成训练摘要与触发词元数据文件
    summary_file = lora_dedicated_folder / "training_summary.json"
    summary_data = {
        "lora_name": args.lora_name,
        "trigger_word": args.trigger_word if args.trigger_word else "(None)",
        "output_folder": str(lora_dedicated_folder),
        "steps": args.steps,
        "repeats": args.repeats,
        "epochs": args.epochs,
        "rank": args.rank,
        "alpha": args.alpha,
        "learning_rate": args.lr,
        "precision": args.precision,
        "text_device": args.text_device,
        "comfyui_prompt_example": (
            f"A high quality digital painting of {args.trigger_word}, in the style of {args.lora_name}, masterpiece, cinematic lighting"
            if args.trigger_word else
            f"A high quality digital painting in the style of {args.lora_name}, masterpiece, cinematic lighting"
        )
    }
    summary_file.write_text(json.dumps(summary_data, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f" [OK] Saved training summary & trigger metadata to: {summary_file}")

    # 可选：直接复制到 ComfyUI 的 loras 目录 (支持专属子文件夹或扁平文件)
    if args.copy_to_comfy_loras:
        comfy_loras_dir = Path(args.comfy_models_dir) / "loras"
        if comfy_loras_dir.exists():
            print_banner(f"Deploying Weights to ComfyUI Loras Folder: {comfy_loras_dir}")
            # 同时在 ComfyUI loras 下建立专属子目录
            target_comfy_sub = comfy_loras_dir / args.lora_name
            target_comfy_sub.mkdir(parents=True, exist_ok=True)
            for ef in exported_files:
                target_comfy_dst = target_comfy_sub / ef.name
                shutil.copy2(str(ef), str(target_comfy_dst))
                print(f" [ComfyUI Ready] Copied {ef.name} -> {target_comfy_dst}")

    print_banner("Training Pipeline Completed Successfully!")
    print(f"Trained weights are available in: {lora_dedicated_folder}")

if __name__ == "__main__":
    main()
