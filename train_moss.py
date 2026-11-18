import os
import sys
import re
import glob
import shutil
import zipfile
import subprocess
import time
import struct
import json

import os
import sys
import re
import glob
import shutil
import zipfile
import subprocess
import time
import struct
import json
import argparse
from pathlib import Path

# 0. 解析命令行与环境变量超参数 (全动态支持用户自定义)
parser = argparse.ArgumentParser(description="OpenMOSS MOSS-Image Kaggle Cloud LoRA Training Engine")
parser.add_argument("--lora_name", type=str, default=os.environ.get("LORA_NAME", "my_lora"),
                    help="Name of the trained LoRA (default: my_lora)")
parser.add_argument("--trigger_word", type=str, default=os.environ.get("TRIGGER_WORD", ""),
                    help="Custom trigger word to prepend to captions (default: none)")
parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "10")),
                    help="Folder repeats per image (default: 10)")
parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "6")),
                    help="Training epochs (default: 6)")
parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "600")),
                    help="Total optimizer training steps (default: 600)")
parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "32")),
                    help="LoRA Rank / Dim (default: 32)")
parser.add_argument("--alpha", type=float, default=float(os.environ.get("ALPHA", "32.0")),
                    help="LoRA Alpha scaling factor (default: 32.0)")
parser.add_argument("--lr", type=str, default=os.environ.get("LR", "1e-4"),
                    help="Learning rate (default: 1e-4)")
parser.add_argument("--dataset_dir", type=str, default=os.environ.get("DATASET_DIR", ""),
                    help="Path to training dataset folder or archive")
args, _ = parser.parse_known_args()

print("=" * 75)
print(">>> OpenMOSS MOSS-Image Cloud LoRA Training Engine (Production) <<<")
print(">>> Upgraded Base: KBlueLeaf/MOSS-Image-heal (Full Precision FP16) <<<")
print(f">>> Target LoRA: '{args.lora_name}' | Trigger Word: '{args.trigger_word if args.trigger_word else '(None)'}' <<<")
print(f">>> Hyperparameters: Repeats={args.repeats}, Epochs={args.epochs}, Steps={args.steps} <<<")
print(f">>> Architecture: Rank={args.rank}, Alpha={args.alpha}, LR={args.lr} <<<")
print("=" * 75)

# 1. 验证 GPU 与宿主机 PyTorch / TorchVision 原生驱动
try:
    import torch
    import torchvision
    print(f"Python Executable: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
        scores = torch.tensor([0.9], device="cuda")
        _ = torchvision.ops.nms(boxes, scores, 0.5)
        print("TorchVision C++ NMS operator check: PASSED!")
    else:
        print("CRITICAL: CUDA is not available. Exiting...")
        sys.exit(1)
except Exception as e:
    print("Error probing GPU / PyTorch / TorchVision:", e)
    sys.exit(1)

# 2. 鉴权与环境变量设置
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if not HF_TOKEN:
    HF_TOKEN = None  # 允许匿名访问公开模型，或通过环境变量注入自定义 Token
if HF_TOKEN:
    os.environ["HF_TOKEN"] = str(HF_TOKEN)
    os.environ["HUGGING_FACE_HUB_TOKEN"] = str(HF_TOKEN)
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_extensions"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

os.makedirs("/tmp/huggingface", exist_ok=True)
os.makedirs("/tmp/models/diffusion_models", exist_ok=True)
os.makedirs("/tmp/models/text_encoders", exist_ok=True)
os.makedirs("/tmp/models/vae", exist_ok=True)

workspace_data_root = "/tmp/workspace_dataset"
os.makedirs(workspace_data_root, exist_ok=True)
target_data_dir = os.path.join(workspace_data_root, f"{args.repeats}_{args.lora_name}")
os.makedirs(target_data_dir, exist_ok=True)

# 3. 智能发现与准备数据集 (自动识别 /kaggle/input/ 挂载或解压，并自动前置注入触发词)
print("\n>>> Preparing Kohya structured dataset in /tmp/workspace_dataset...")
source_dir = None
if args.dataset_dir and os.path.exists(args.dataset_dir):
    print(f"Using explicitly specified dataset path: {args.dataset_dir}")
    source_dir = args.dataset_dir
elif os.path.exists("/kaggle/input"):
    candidate_dirs = []
    for root, dirs, files in os.walk("/kaggle/input"):
        imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if imgs:
            candidate_dirs.append((root, len(imgs)))
    if candidate_dirs:
        candidate_dirs.sort(key=lambda x: x[1], reverse=True)
        source_dir = candidate_dirs[0][0]
        print(f"Auto-detected Kaggle mounted dataset directory ({candidate_dirs[0][1]} images): {source_dir}")

valid_img_exts = {".jpg", ".jpeg", ".png", ".webp"}
if source_dir and os.path.isdir(source_dir):
    print(f"Ingesting training data from {source_dir} -> {target_data_dir}...")
    for f in os.listdir(source_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_img_exts:
            src_img = os.path.join(source_dir, f)
            dest_img = os.path.join(target_data_dir, f)
            shutil.copy2(src_img, dest_img)
            
            # 配套 txt 标注
            base_name = os.path.splitext(f)[0]
            src_txt = os.path.join(source_dir, base_name + ".txt")
            dest_txt = os.path.join(target_data_dir, base_name + ".txt")
            caption = ""
            if os.path.exists(src_txt):
                with open(src_txt, "r", encoding="utf-8", errors="ignore") as tf:
                    caption = tf.read().strip()
            
            # 若配置了触发词且尚未包含，自动前置拼接
            if args.trigger_word and not caption.startswith(args.trigger_word):
                caption = f"{args.trigger_word}, {caption}" if caption else args.trigger_word
            
            with open(dest_txt, "w", encoding="utf-8") as tf:
                tf.write(caption)
else:
    # 备用：若未挂载自定义数据集，尝试本地 zip 或镜像拉取
    local_zip = "/tmp/dataset.zip"
    dataset_mirrors = [
        "https://trystocare.com/u6ueup1457",
        "https://filebin.net/ness_dataset_2026/dataset.zip"
    ]
    chosen_zip = None
    found_zips = glob.glob("/kaggle/**/dataset.zip", recursive=True) + glob.glob("**/dataset.zip", recursive=True)
    if found_zips and os.path.getsize(found_zips[0]) > 10 * 1024 * 1024:
        chosen_zip = found_zips[0]
        print(f"Found local dataset candidate: {chosen_zip}")
    else:
        for mirror_url in dataset_mirrors:
            print(f"Fetching fallback dataset via curl from: {mirror_url}...")
            try:
                curl_cmd = [
                    "curl", "-s", "-L",
                    "--retry", "3",
                    "--retry-delay", "2",
                    "--retry-all-errors",
                    mirror_url,
                    "-o", local_zip
                ]
                subprocess.run(curl_cmd, check=True)
                if os.path.exists(local_zip) and os.path.getsize(local_zip) > 10 * 1024 * 1024:
                    chosen_zip = local_zip
                    print(f"Downloaded dataset archive: {chosen_zip} ({os.path.getsize(chosen_zip) / (1024**2):.2f} MB)")
                    break
            except Exception as dl_err:
                print(f"Mirror failed ({dl_err}), trying next mirror...")

    if not chosen_zip or not os.path.exists(chosen_zip) or os.path.getsize(chosen_zip) < 10 * 1024 * 1024:
        raise RuntimeError("No dataset found in /kaggle/input and all fallback mirrors failed.")

    print(f"Unpacking {chosen_zip} -> {target_data_dir}...")
    with zipfile.ZipFile(chosen_zip, 'r') as zf:
        zf.extractall(target_data_dir)

img_count = len([f for f in os.listdir(target_data_dir) if os.path.splitext(f)[1].lower() in valid_img_exts])
txt_count = len([f for f in os.listdir(target_data_dir) if f.lower().endswith(".txt")])
print(f"Dataset verified: {img_count} images, {txt_count} txt captions in {target_data_dir}.")
if img_count < 1 or txt_count < 1:
    raise RuntimeError(f"Dataset verification failed: Expected at least 1 image and caption, found {img_count} images, {txt_count} txt files.")

# 4. 安装外围依赖
print("\n>>> Installing auxiliary utilities (safetensors, hf-transfer, bitsandbytes)...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub", "hf-transfer", "safetensors", "bitsandbytes", "ftfy"
], check=True)

# 5. 克隆并安装 KohakuMossTrainer
print("\n>>> Setting up KohakuMossTrainer framework...")
trainer_dir = "/tmp/KohakuMossTrainer"
if not os.path.exists(trainer_dir):
    subprocess.run(["git", "clone", "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git", trainer_dir], check=True)

pyproject_path = os.path.join(trainer_dir, "pyproject.toml")
if os.path.exists(pyproject_path):
    with open(pyproject_path, "r", encoding="utf-8") as f:
        pyproject_content = f.read()
    patched_content = re.sub(r'\"torch<[^\"]+\"', '\"torch>=2.0\"', pyproject_content)
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(patched_content)
    print("Patched KohakuMossTrainer pyproject.toml: PyTorch constraint relaxed.")

# 注入 PyTorch 2.11 / torch.accelerator.empty_host_cache 全局兼容性 Hook
import site
for sp in site.getsitepackages():
    sc_file = os.path.join(sp, "sitecustomize.py")
    try:
        with open(sc_file, "a", encoding="utf-8") as sc:
            sc.write("\nimport torch\n")
            sc.write("if hasattr(torch, 'accelerator'):\n")
            sc.write("    if not hasattr(torch.accelerator, 'empty_host_cache'):\n")
            sc.write("        torch.accelerator.empty_host_cache = getattr(torch.accelerator, 'empty_cache', getattr(torch.cuda, 'empty_cache', lambda: None))\n")
        print(f"Injected sitecustomize hook at: {sc_file}")
    except Exception as e:
        print(f"Note: Could not inject sitecustomize at {sc_file}: {e}")

# 源码级遍历修补 KohakuMossTrainer 中的 empty_host_cache 硬编码调用
patched_files_count = 0
for root, _, files in os.walk(trainer_dir):
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    code = fp.read()
                if "empty_host_cache" in code:
                    patched_code = re.sub(
                        r'torch\.accelerator\.empty_host_cache\(\)',
                        "getattr(torch.accelerator, 'empty_host_cache', getattr(getattr(torch, 'accelerator', None), 'empty_cache', getattr(torch.cuda, 'empty_cache', lambda: None)))()",
                        code
                    )
                    with open(fpath, "w", encoding="utf-8") as fp:
                        fp.write(patched_code)
                    print(f"Patched compatibility in source: {fpath}")
                    patched_files_count += 1
            except Exception as patch_err:
                print(f"Error patching {fpath}: {patch_err}")
print(f"KohakuMossTrainer source patches applied to {patched_files_count} files.")

obj_check_file = os.path.join(trainer_dir, "src", "kohakumoss", "training", "checks", "objective.py")
if os.path.exists(obj_check_file):
    try:
        with open(obj_check_file, "r", encoding="utf-8") as f:
            obj_content = f.read()
        obj_content = obj_content.replace(
            'raw_end = sub.get("end", -1)',
            'raw_end = sub.get("end", -1) if isinstance(sub, dict) else -1'
        ).replace(
            'warmup = sub.get("warmup", 0)',
            'warmup = sub.get("warmup", 0) if isinstance(sub, dict) else 0'
        )
        with open(obj_check_file, "w", encoding="utf-8") as f:
            f.write(obj_content)
        print("Patched objective.py for schedule_problems defensive check.")
    except Exception as obj_err:
        print("Note: Could not patch objective.py:", obj_err)

os.chdir(trainer_dir)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", "."], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "kohaku-engine", "kohakuvault", "lycoris-lora", "anyschedule"], check=True)

# 6. 全链路核心 AI 库预检断言
print("\n>>> Running pre-flight import checks for core AI libraries...")
import diffusers
import transformers
import peft
import accelerate
print(f" - diffusers: {diffusers.__version__}")
print(f" - transformers: {transformers.__version__}")
print(f" - peft: {peft.__version__}")
print(f" - accelerate: {accelerate.__version__}")
print("Pre-flight import checks: ALL PASSED!")

# 7. 下载官方修复版底模 KBlueLeaf/MOSS-Image-heal (FP16 全精度)
print("\n>>> Downloading KBlueLeaf/MOSS-Image-heal weights to /tmp/models...")
from huggingface_hub import hf_hub_download

print("1/3 Downloading Healed DiT Transformer (moss_image_heal_fp16.safetensors, ~10.2GB)...")
hf_hub_download(
    repo_id="KBlueLeaf/MOSS-Image-heal",
    filename="comfyui/moss_image_heal_fp16.safetensors",
    local_dir="/tmp/models",
    token=HF_TOKEN
)
heal_downloaded = "/tmp/models/comfyui/moss_image_heal_fp16.safetensors"
heal_final = "/tmp/models/diffusion_models/moss_image_heal_fp16.safetensors"
if os.path.exists(heal_downloaded) and not os.path.exists(heal_final):
    shutil.move(heal_downloaded, heal_final)
    print(f"Moved healed DiT to: {heal_final}")

print("2/3 Downloading Text Encoder (qwen3.5_9b_int8_convrot.safetensors, ~9.7GB)...")
hf_hub_download(
    repo_id="Comfy-Org/MOSS-Image",
    filename="text_encoders/qwen3.5_9b_int8_convrot.safetensors",
    local_dir="/tmp/models",
    token=HF_TOKEN
)

print("3/3 Downloading VAE (flux2-vae_small.safetensors, ~168MB)...")
hf_hub_download(
    repo_id="Comfy-Org/MOSS-Image",
    filename="vae/flux2-vae_small.safetensors",
    local_dir="/tmp/models",
    token=HF_TOKEN
)
print("All model weights successfully downloaded to /tmp/models!")

# 8. 生成标准训练与数据配置
user_cfg_dir = os.path.join(trainer_dir, "configs", "user")
os.makedirs(user_cfg_dir, exist_ok=True)

data_cfg_content = f"""
data_root = "/tmp/workspace_dataset"
WEIGHTS = "/tmp/models"

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
TEXT_ENCODER_KWARGS = {{"path": WEIGHTS, "device": "cuda"}}
IMAGE_ENCODER_KWARGS = {{"path": WEIGHTS, "device": "cuda"}}
"""

train_cfg_content = f"""
from kohakuengine import use_config

use_config("../moss/recipes/recipes_lora_16g_fp16.py")
use_config("{args.lora_name}_data.py")

RUN_NAME = "{args.lora_name}"
TRANSFORMER = "moss_image_heal_fp16.safetensors"
FP16_FIX = "none"
DIM = {args.rank}
ALPHA = {args.alpha}
STEPS = {args.steps}
LR = {args.lr}
SAVE_MINUTES = 30
SAVE_FORMAT = "both"
SCHEDULE = {{
    "mode": "cosine",
    "warmup": 30,
    "end": -1,
    "min_value": 0.1
}}
"""

data_cfg_file = os.path.join(user_cfg_dir, f"{args.lora_name}_data.py")
train_cfg_file = os.path.join(user_cfg_dir, f"{args.lora_name}.py")

with open(data_cfg_file, "w", encoding="utf-8") as f:
    f.write(data_cfg_content.strip())

with open(train_cfg_file, "w", encoding="utf-8") as f:
    f.write(train_cfg_content.strip())
print(f"Configuration written: {train_cfg_file}")

# 9. 执行前置离线双特征提取 (双离线缓存)
print(f"\n>>> Step 1/2: Caching offline Text Embeddings for '{args.lora_name}'...")
subprocess.run([
    "kogine", "run", "scripts/data/cache_text.py",
    "--config", f"configs/user/{args.lora_name}_data.py"
], check=True)

print(f"\n>>> Step 2/2: Caching offline Image Latents for '{args.lora_name}'...")
subprocess.run([
    "kogine", "run", "scripts/data/cache_latents.py",
    "--config", f"configs/user/{args.lora_name}_data.py"
], check=True)

# 9.5 物理级清洗临时离线编码器旋钮
print(f"\n>>> Sanitizing {args.lora_name}_data.py on disk for moss.py strict knob validation...")
with open(data_cfg_file, "r", encoding="utf-8") as f:
    raw_cfg = f.read()

sanitized_cfg = re.sub(r'TEXT_ENCODER_KWARGS\s*=\s*\{[^\}]+\}', '', raw_cfg)
sanitized_cfg = re.sub(r'IMAGE_ENCODER_KWARGS\s*=\s*\{[^\}]+\}', '', sanitized_cfg)

with open(data_cfg_file, "w", encoding="utf-8") as f:
    f.write(sanitized_cfg.strip() + "\n")
print(f"Sanitization complete! Knobs TEXT_ENCODER_KWARGS and IMAGE_ENCODER_KWARGS physically removed.")

# 9.8 实时重定向与软链接：确保所有生成的 checkpoints 实时写入 /kaggle/working/runs
working_runs = "/kaggle/working/runs"
os.makedirs(working_runs, exist_ok=True)
trainer_runs = os.path.join(trainer_dir, "runs")
if os.path.exists(trainer_runs) and not os.path.islink(trainer_runs):
    shutil.rmtree(trainer_runs)
if not os.path.exists(trainer_runs):
    os.symlink(working_runs, trainer_runs)
    print(f"Created real-time runs symlink: {trainer_runs} -> {working_runs}")

# 10. 启动核心 LoRA 深度训练
print(f"\n>>> Starting {args.steps}-step Deep LoRA Training Loop: '{args.lora_name}' (Repeats={args.repeats}, Epochs={args.epochs})...")
train_cmd = [
    "kogine", "run", "scripts/train/moss.py",
    "--config", f"configs/user/{args.lora_name}.py"
]
print("Executing command:", " ".join(train_cmd))
subprocess.run(train_cmd, check=True)

# 11. 智能检索所有生成的 .safetensors 权重并归档打包
print("\n>>> Locating and packaging all generated .safetensors weights...")
output_lora_dir = f"/kaggle/working/output/{args.lora_name}"
os.makedirs(output_lora_dir, exist_ok=True)

safetensors_files = glob.glob(f"{trainer_dir}/**/runs/**/*.safetensors", recursive=True) + \
                    glob.glob(f"{trainer_dir}/**/checkpoints/**/*.safetensors", recursive=True) + \
                    glob.glob("/kaggle/working/**/*.safetensors", recursive=True)

safetensors_files = [f for f in set(safetensors_files) if "/tmp/models" not in f and "moss_image_heal" not in f and "qwen" not in f and "flux2" not in f]

print(f"Found {len(safetensors_files)} LoRA weight files:")
for sf in safetensors_files:
    fname = os.path.basename(sf)
    dest = os.path.join(output_lora_dir, fname)
    print(f" - Archiving {sf} -> {dest}")
    shutil.copy2(sf, dest)

summary_data = {
    "lora_name": args.lora_name,
    "trigger_word": args.trigger_word if args.trigger_word else "(None)",
    "steps": args.steps,
    "repeats": args.repeats,
    "epochs": args.epochs,
    "rank": args.rank,
    "alpha": args.alpha,
    "learning_rate": args.lr,
    "base_model": "KBlueLeaf/MOSS-Image-heal (FP16)",
    "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
}
summary_file = os.path.join(output_lora_dir, "training_summary.json")
with open(summary_file, "w", encoding="utf-8") as sf:
    json.dump(summary_data, sf, indent=2, ensure_ascii=False)

archive_path = f"/kaggle/working/{args.lora_name}_lora_results"
shutil.make_archive(archive_path, 'zip', output_lora_dir)
print(f"\nFinal archive created: {archive_path}.zip (Size: {os.path.getsize(archive_path + '.zip') / (1024**2):.2f} MB)")
print("=" * 75)
print(f">>> MOSS-Image Deep LoRA Training '{args.lora_name}' Finished Successfully! <<<")
print("=" * 75)
