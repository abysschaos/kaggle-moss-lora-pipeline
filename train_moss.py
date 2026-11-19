import os
import sys
import re
import glob
import shutil
import zipfile
import subprocess
import time
import json
import argparse
from pathlib import Path

# 0. 解析命令行与环境变量超参数 (全面对齐本地引擎，支持显存档位、分辨率、精度与 Base/Heal/SFT 变体)
parser = argparse.ArgumentParser(description="OpenMOSS MOSS-Image Cloud LoRA Training Engine (Kaggle / Colab)")
parser.add_argument("--lora_name", "--run_name", type=str, default=os.environ.get("LORA_NAME", "my_lora"), dest="lora_name",
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
parser.add_argument("--vram_preset", type=str, default=os.environ.get("VRAM_PRESET", "auto"),
                    choices=["auto", "6g", "8g", "12g", "16g"],
                    help="VRAM optimization preset: auto, 6g, 8g, 12g, 16g (default: auto)")
parser.add_argument("--resolution", type=int, default=int(os.environ.get("RESOLUTION", "1024")),
                    help="Target training image resolution area, e.g. 1024, 768, 512 (default: 1024)")
parser.add_argument("--variant", type=str, default=os.environ.get("MODEL_VARIANT", os.environ.get("VARIANT", "heal")),
                    choices=["heal", "base", "sft"],
                    help="Backbone model variant: heal (MOSS-Image-heal), base (MOSS-Image-Base), sft (MOSS-Image-SFT)")
parser.add_argument("--base_quant", type=str, default=os.environ.get("BASE_QUANT", "auto"),
                    choices=["auto", "int8", "full"],
                    help="Backbone weight quantization: auto (int8 for 6g/8g, full for 12g/16g), int8, full")
parser.add_argument("--precision", type=str, default=os.environ.get("PRECISION", "auto"),
                    choices=["auto", "fp16", "bf16"],
                    help="Compute precision: auto (bf16 if GPU supports, else fp16), fp16, bf16")
parser.add_argument("--text_device", type=str, default=os.environ.get("TEXT_DEVICE", "auto"),
                    choices=["auto", "cuda", "cpu"],
                    help="Device for offline text embedding extraction: auto, cuda, cpu")
parser.add_argument("--vae_device", type=str, default=os.environ.get("VAE_DEVICE", "cuda"),
                    choices=["cuda", "cpu"],
                    help="Device for offline VAE latents extraction (default: cuda)")
parser.add_argument("--train_mode", type=str, default=os.environ.get("TRAIN_MODE", "fresh"),
                    choices=["fresh", "resume", "init_from"],
                    help="Training mode: 'fresh' (from scratch), 'resume' (continue from checkpoint), 'init_from' (continue training from an existing .safetensors LoRA)")
parser.add_argument("--resume_from", type=str, default=os.environ.get("RESUME_FROM", "latest"),
                    help="Checkpoint directory path to resume from, or 'latest'")
parser.add_argument("--init_from", type=str, default=os.environ.get("INIT_FROM", ""),
                    help="Path to an existing .safetensors LoRA file or checkpoint dir to continue training from")
parser.add_argument("--save_every", type=int, default=int(os.environ.get("SAVE_EVERY", "100")),
                    help="Save intermediate checkpoint every N steps (default: 100)")
parser.add_argument("--dataset_dir", type=str, default=os.environ.get("DATASET_DIR", ""),
                    help="Path to training dataset folder or archive")
args, _ = parser.parse_known_args()

print("=" * 75)
print(">>> OpenMOSS MOSS-Image Cloud LoRA Training Engine (Production v2) <<<")
print(f">>> Target LoRA: '{args.lora_name}' | Trigger Word: '{args.trigger_word if args.trigger_word else '(None)'}' <<<")
print(f">>> Hyperparameters: Repeats={args.repeats}, Epochs={args.epochs}, Steps={args.steps} <<<")
print(f">>> Architecture: Rank={args.rank}, Alpha={args.alpha}, LR={args.lr} <<<")
print(f">>> Cloud Options: Variant={args.variant}, VRAM={args.vram_preset}, Res={args.resolution}, Quant={args.base_quant}, Prec={args.precision} <<<")
print("=" * 75)

# 1. 验证 GPU 与宿主机 PyTorch / TorchVision 原生驱动
vram_gb = 16.0
bf16_supported = False
try:
    import torch
    import torchvision
    print(f"Python Executable: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"GPU Device: {gpu_name}")
        print(f"Total VRAM: {vram_gb:.2f} GB | Native BF16 Supported: {bf16_supported}")
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

# 1.5 动态解析显存档位、计算精度、量化格式与文本提取设备
selected_preset = args.vram_preset.lower()
if selected_preset == "auto":
    if vram_gb <= 10.0:
        effective_preset = "6g"
    elif vram_gb <= 14.0:
        effective_preset = "12g"
    else:
        effective_preset = "16g"
    vram_note = f"(Auto-detected {vram_gb:.1f} GB VRAM -> selected '{effective_preset}')"
else:
    effective_preset = selected_preset
    vram_note = f"(Manually specified: '{effective_preset}')"

if effective_preset == "6g":
    recipe_import = 'use_config("../moss/_base/base_lora.py")\nuse_config("../moss/vram/vram_6g.py")'
    max_tokens_val = 4352
    memory_profile_str = 'MEMORY_PROFILE = "ckpt-ring2-actoff-optcpu-gradoff"'
    print(f" [VRAM Preset] Applied 6G Profile (extreme ~4.4GB VRAM): {vram_note}")
elif effective_preset == "8g":
    recipe_import = 'use_config("../moss/_base/base_lora.py")\nuse_config("../moss/vram/vram_8g.py")'
    max_tokens_val = 4352
    memory_profile_str = 'MEMORY_PROFILE = "minimal"'
    print(f" [VRAM Preset] Applied 8G Profile (~5.5GB VRAM): {vram_note}")
elif effective_preset == "12g":
    recipe_import = 'use_config("../moss/recipes/recipes_lora_12g_stream.py")'
    max_tokens_val = 8192
    memory_profile_str = 'MEMORY_PROFILE = "low"'
    print(f" [VRAM Preset] Applied 12G Profile (~8.5GB VRAM): {vram_note}")
else:
    recipe_import = 'use_config("../moss/recipes/recipes_lora_16g_fp16.py")'
    max_tokens_val = 16384
    memory_profile_str = ""
    print(f" [VRAM Preset] Applied 16G Profile (full speed): {vram_note}")

if args.precision == "auto":
    effective_precision = "bf16" if bf16_supported else "fp16"
    prec_note = f"(Auto-detected BF16={bf16_supported} -> '{effective_precision}')"
else:
    effective_precision = args.precision.lower()
    if effective_precision == "bf16" and not bf16_supported:
        print(" [Warning] Requested BF16 on a GPU without native BF16 support (e.g. T4/P100). Falling back to FP16.")
        effective_precision = "fp16"
    prec_note = f"(User specified -> '{effective_precision}')"
print(f" [Precision] Compute Precision: {effective_precision.upper()} {prec_note}")

if args.base_quant == "auto":
    use_int8_backbone = effective_preset in ("6g", "8g")
else:
    use_int8_backbone = (args.base_quant == "int8")

if args.text_device == "auto":
    effective_text_device = "cpu" if (vram_gb <= 10.0 or effective_preset == "6g") else "cuda"
else:
    effective_text_device = args.text_device.lower()
print(f" [Devices] Text Encoder: {effective_text_device.upper()} | VAE: {args.vae_device.upper()}")
print(f" [Resolution] Target Area: {args.resolution}x{args.resolution} ({args.resolution * args.resolution} px)")

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

# 3. 智能发现与准备数据集 (自动识别 /kaggle/input/ 挂载或解压，严格孤儿清理并自动前置注入触发词)
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
trigger_clean = args.trigger_word.strip() if args.trigger_word else ""

# 清理 workspace_data_root 下的历史陈旧数据集子目录，防止多任务残留污染
for sub_name in os.listdir(workspace_data_root):
    sub_path = os.path.join(workspace_data_root, sub_name)
    if os.path.isdir(sub_path) and sub_name != "cache" and sub_name != os.path.basename(target_data_dir):
        shutil.rmtree(sub_path, ignore_errors=True)
        print(f" [Cleanup] Removed obsolete dataset folder: {sub_name}")

if source_dir and os.path.isdir(source_dir):
    print(f"Ingesting training data from {source_dir} -> {target_data_dir}...")
    src_imgs = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in valid_img_exts]
    valid_img_names = set(src_imgs)
    valid_txt_names = {os.path.splitext(f)[0] + ".txt" for f in src_imgs}

    # 清理 target_data_dir 中的孤儿文件，保证 1:1 对齐
    orphan_purged = 0
    for existing in os.listdir(target_data_dir):
        existing_path = os.path.join(target_data_dir, existing)
        if os.path.isfile(existing_path) and existing not in valid_img_names and existing not in valid_txt_names:
            try:
                os.remove(existing_path)
                orphan_purged += 1
            except Exception:
                pass
    if orphan_purged > 0:
        cache_dir = os.path.join(workspace_data_root, "cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
        print(f" [Cleanup] Purged {orphan_purged} orphan files and reset stale cache.")

    for f in src_imgs:
        src_img = os.path.join(source_dir, f)
        dest_img = os.path.join(target_data_dir, f)
        shutil.copy2(src_img, dest_img)

        base_name = os.path.splitext(f)[0]
        src_txt = os.path.join(source_dir, base_name + ".txt")
        dest_txt = os.path.join(target_data_dir, base_name + ".txt")
        caption = ""
        if os.path.exists(src_txt):
            with open(src_txt, "r", encoding="utf-8", errors="ignore") as tf:
                caption = tf.read().strip()

        if trigger_clean and not caption.lower().startswith(trigger_clean.lower()):
            caption = f"{trigger_clean}, {caption}" if caption else trigger_clean

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

native_driver_file = os.path.join(trainer_dir, "src", "kohakumoss", "training", "drivers", "native.py")
if os.path.exists(native_driver_file):
    try:
        with open(native_driver_file, "r", encoding="utf-8") as f:
            nd_content = f.read()
        if "KOHAKUMOSS_PAUSE_FILE" not in nd_content:
            target_line = "finish_step(state, group, bundle, ema, self.runner, out)"
            pause_hook = (
                "finish_step(state, group, bundle, ema, self.runner, out)\n"
                "            import os as _os\n"
                "            _pfile = _os.environ.get('KOHAKUMOSS_PAUSE_FILE', '')\n"
                "            if _pfile and _os.path.exists(_pfile):\n"
                "                try:\n"
                "                    _os.remove(_pfile)\n"
                "                except Exception:\n"
                "                    pass\n"
                "                print(f'\\n[Pause & Save] Detected pause request at step {state.step}! Saving checkpoint and exporting current weights...', flush=True)\n"
                "                self.checkpoints.save(state.step, self.parts(state))\n"
                "                break"
            )
            if target_line in nd_content:
                with open(native_driver_file, "w", encoding="utf-8") as f:
                    f.write(nd_content.replace(target_line, pause_hook, 1))
                print("Injected graceful Pause & Save hook into drivers/native.py.")
    except Exception as nd_err:
        print("Note: Could not patch native.py:", nd_err)

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

# 7. 根据所选变体与量化规格下载对应底模 (支持 MOSS-Image-heal / Base / SFT)
from huggingface_hub import hf_hub_download

selected_variant = args.variant.lower()
if selected_variant == "heal":
    arch_variant = "base"
    repo_id = "KBlueLeaf/MOSS-Image-heal"
    if use_int8_backbone:
        hf_filename = "comfyui/moss_image_heal_int8_convrot.safetensors"
    elif effective_precision == "bf16":
        hf_filename = "comfyui/moss_image_heal_bf16.safetensors"
    else:
        hf_filename = "comfyui/moss_image_heal_fp16.safetensors"
    fp16_fix_val = "none"
elif selected_variant == "base":
    arch_variant = "base"
    repo_id = "Comfy-Org/MOSS-Image"
    if use_int8_backbone:
        hf_filename = "diffusion_models/moss_image_base_int8_convrot.safetensors"
    else:
        hf_filename = "diffusion_models/moss_image_base_bf16.safetensors"
    fp16_fix_val = "online" if effective_precision == "fp16" else "none"
else:  # sft
    arch_variant = "sft"
    repo_id = "Comfy-Org/MOSS-Image"
    if use_int8_backbone:
        hf_filename = "diffusion_models/moss_image_sft_int8_convrot.safetensors"
    else:
        hf_filename = "diffusion_models/moss_image_sft_bf16.safetensors"
    fp16_fix_val = "online" if effective_precision == "fp16" else "none"

transformer_filename = os.path.basename(hf_filename)
is_int8_transformer = "int8" in transformer_filename.lower()
base_prec_str = 'BASE_PRECISION = "int8"' if is_int8_transformer else ""

print(f"\n>>> Downloading Model Weights to /tmp/models (Variant: {selected_variant.upper()} -> Arch: {arch_variant.upper()})...")
print(f"1/3 Downloading DiT Transformer ({repo_id} :: {hf_filename})...")
downloaded_dit = hf_hub_download(
    repo_id=repo_id,
    filename=hf_filename,
    local_dir="/tmp/models",
    token=HF_TOKEN
)
dit_final = os.path.join("/tmp/models/diffusion_models", transformer_filename)
if os.path.exists(downloaded_dit) and os.path.abspath(downloaded_dit) != os.path.abspath(dit_final):
    if not os.path.exists(dit_final):
        shutil.move(downloaded_dit, dit_final)
        print(f"Moved DiT backbone to: {dit_final}")

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

# 7.8 提前建立 /kaggle/working/runs 软链接，确保断点检测与实时落盘统一
working_runs = "/kaggle/working/runs"
os.makedirs(working_runs, exist_ok=True)
trainer_runs = os.path.join(trainer_dir, "runs")
if os.path.exists(trainer_runs) and not os.path.islink(trainer_runs):
    shutil.rmtree(trainer_runs)
if not os.path.exists(trainer_runs):
    os.symlink(working_runs, trainer_runs)
    print(f"Created real-time runs symlink: {trainer_runs} -> {working_runs}")


def _inspect_safetensors(sf_path: str) -> dict:
    info = {"rank": None, "alpha": None, "step": None}
    try:
        import struct
        with open(sf_path, "rb") as f:
            h_size = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(h_size).decode("utf-8"))
        meta = hdr.get("__metadata__", {}) or {}
        if "ss_network_dim" in meta:
            info["rank"] = int(float(meta["ss_network_dim"]))
        if "ss_network_alpha" in meta:
            info["alpha"] = float(meta["ss_network_alpha"])
        if "kohakumoss_step" in meta:
            info["step"] = int(meta["kohakumoss_step"])
        if info["rank"] is None:
            for k, v in hdr.items():
                if k != "__metadata__" and k.endswith(".lora_down.weight"):
                    shp = v.get("shape", [])
                    if len(shp) >= 2:
                        info["rank"] = int(shp[0])
                        break
        if info["alpha"] is None and info["rank"] is not None:
            info["alpha"] = float(info["rank"])
    except Exception:
        pass
    return info


# 7.9 解析训练模式 (fresh / resume / init_from)
train_mode = args.train_mode.lower()
resume_cfg_str = ""
init_from_cfg_str = ""
effective_steps = args.steps
resumed_step = 0

if train_mode == "resume":
    ckpt_candidates = []
    search_roots = []
    if args.resume_from and args.resume_from.lower() not in ("auto", "latest") and os.path.exists(args.resume_from):
        search_roots.append(args.resume_from)
    default_c_root = os.path.join(working_runs, "moss", args.lora_name, "checkpoints")
    if os.path.exists(default_c_root):
        search_roots.append(default_c_root)
    if os.path.exists("/kaggle/input"):
        for r, d, files in os.walk("/kaggle/input"):
            if "state.json" in files and "trainable.safetensors" in files:
                search_roots.append(r)

    for sr in search_roots:
        if os.path.isfile(os.path.join(sr, "state.json")) and os.path.isfile(os.path.join(sr, "trainable.safetensors")):
            try:
                st = json.loads(Path(os.path.join(sr, "state.json")).read_text(encoding="utf-8"))
                ckpt_candidates.append((int(st.get("step", 0)), sr))
            except Exception:
                ckpt_candidates.append((0, sr))
        elif os.path.isdir(sr):
            for sub in glob.glob(os.path.join(sr, "step_*")):
                if os.path.isfile(os.path.join(sub, "state.json")) and os.path.isfile(os.path.join(sub, "trainable.safetensors")):
                    try:
                        st = json.loads(Path(os.path.join(sub, "state.json")).read_text(encoding="utf-8"))
                        ckpt_candidates.append((int(st.get("step", 0)), sub))
                    except Exception:
                        ckpt_candidates.append((0, sub))

    if ckpt_candidates:
        ckpt_candidates.sort(key=lambda x: x[0])
        resumed_step, chosen_ckpt = ckpt_candidates[-1]
        l_info = _inspect_safetensors(os.path.join(chosen_ckpt, "trainable.safetensors"))
        if l_info.get("rank") and l_info["rank"] != args.rank:
            print(f" [Resume Auto-Align] Aligned Rank {args.rank} -> {l_info['rank']}")
            args.rank = l_info["rank"]
            if l_info.get("alpha"):
                args.alpha = l_info["alpha"]
        resume_cfg_str = f'RESUME = "{Path(chosen_ckpt).as_posix()}"'
        if args.steps <= resumed_step:
            effective_steps = resumed_step + args.steps
            print(f" [Resume Auto-Extend] Checkpoint at step {resumed_step} (>= {args.steps}). Extended target STEPS = {effective_steps}!")
        else:
            print(f" [Resume] Resuming from {chosen_ckpt} (step {resumed_step} -> {effective_steps})")
    else:
        print(" [Resume Fallback] No checkpoint dir found. Falling back to INIT_FROM mode...")
        train_mode = "init_from"

if train_mode == "init_from":
    chosen_sf = None
    if args.init_from and os.path.exists(args.init_from):
        if os.path.isfile(args.init_from):
            chosen_sf = args.init_from
        elif os.path.isdir(args.init_from):
            sfs = glob.glob(os.path.join(args.init_from, "*.safetensors"))
            if sfs:
                chosen_sf = sorted(sfs, key=os.path.getmtime, reverse=True)[0]
    if not chosen_sf:
        # 自动从 /kaggle/working/output 或 /kaggle/input 搜索已训练好的 LoRA .safetensors
        cand_sfs = glob.glob(f"/kaggle/working/output/{args.lora_name}/*.safetensors") + \
                   glob.glob("/kaggle/input/**/*.safetensors", recursive=True)
        cand_sfs = [
            f for f in cand_sfs
            if "moss_image" not in os.path.basename(f).lower()
            and "qwen" not in os.path.basename(f).lower()
            and "flux2" not in os.path.basename(f).lower()
        ]
        if cand_sfs:
            chosen_sf = sorted(cand_sfs, key=os.path.getmtime, reverse=True)[0]

    if not chosen_sf:
        raise FileNotFoundError("No existing .safetensors LoRA found for --train_mode=init_from!")

    stage_init_dir = f"/tmp/init_from_stage/{args.lora_name}"
    os.makedirs(stage_init_dir, exist_ok=True)
    shutil.copy2(chosen_sf, os.path.join(stage_init_dir, "trainable.safetensors"))
    l_info = _inspect_safetensors(chosen_sf)
    if l_info.get("rank") and l_info["rank"] != args.rank:
        print(f" [Init-From Auto-Align] Aligned Rank {args.rank} -> {l_info['rank']} from {chosen_sf}")
        args.rank = l_info["rank"]
        if l_info.get("alpha"):
            args.alpha = l_info["alpha"]
    init_from_cfg_str = f'INIT_FROM = "{stage_init_dir}"'
    print(f" [Init-From] Continuing training from existing LoRA: {chosen_sf} (Rank={args.rank}, Alpha={args.alpha}, Steps={effective_steps})")

# 8. 生成标准训练与数据配置 (与本地引擎完全对齐，注入分辨率、显存档位、架构变体、续炼与防 OOM 守卫)
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
TARGET_AREA = {args.resolution} * {args.resolution}
UPSCALE = "never"
ASPECT_RANGE = (0.33, 3.0)
BATCHING = "tokens"
MAX_TOKENS = {max_tokens_val}
OVERLONG = "drop"
CAPTION = {{
    "template": "nl",
}}
CAPTION_VARIANTS = 1
TEXT_DROP = 0.05
LATENT_CACHE = "offline"
TEXT_CACHE = "offline"
FLIP = False
TEXT_ENCODER_KWARGS = {{"path": WEIGHTS, "device": "{effective_text_device}"}}
IMAGE_ENCODER_KWARGS = {{"path": WEIGHTS, "device": "{args.vae_device}"}}
VAE_ENCODE = {{"compile": "off"}}
"""

warmup_steps = min(30, max(1, int(effective_steps * 0.05))) if effective_steps > 5 else 0
save_every_val = max(0, int(args.save_every))

train_cfg_content = f"""
from kohakuengine import use_config

{recipe_import}
use_config("{args.lora_name}_data.py")

VARIANT = "{arch_variant}"
RUN_NAME = "{args.lora_name}"
TRANSFORMER = "{transformer_filename}"
{base_prec_str}
{memory_profile_str}
{resume_cfg_str}
{init_from_cfg_str}
SAMPLE_EVERY = 0
VALIDATE_EVERY = 0
TUNE = "off"
FP16_FIX = "{fp16_fix_val}"
PRECISION = "{effective_precision}"
DIM = {args.rank}
ALPHA = {args.alpha}
STEPS = {effective_steps}
LR = {args.lr}
SAVE_EVERY = {save_every_val}
SAVE_MINUTES = 25
KEEP_LAST = 5
SAVE_FORMAT = "both"
SCHEDULE = {{
    "mode": "cosine",
    "warmup": {warmup_steps},
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
print(f"\n>>> Step 1/2: Caching offline Text Embeddings for '{args.lora_name}' (Device: {effective_text_device.upper()})...")
subprocess.run([
    "kogine", "run", "scripts/data/cache_text.py",
    "--config", f"configs/user/{args.lora_name}_data.py"
], check=True)

print(f"\n>>> Step 2/2: Caching offline Image Latents for '{args.lora_name}' (Device: {args.vae_device.upper()}, Res: {args.resolution}x{args.resolution})...")
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
print("Sanitization complete! Knobs TEXT_ENCODER_KWARGS and IMAGE_ENCODER_KWARGS physically removed.")

# 10. 启动核心 LoRA 深度训练
os.environ["KOHAKUMOSS_PAUSE_FILE"] = "/tmp/.pause_request"
print(f"\n>>> Starting {effective_steps}-step Deep LoRA Training Loop: '{args.lora_name}' (Mode={train_mode.upper()}, Variant={selected_variant.upper()}, VRAM={effective_preset}, Res={args.resolution})...")
start_time = time.time()
train_cmd = [
    "kogine", "run", "scripts/train/moss.py",
    "--config", f"configs/user/{args.lora_name}.py"
]
print("Executing command:", " ".join(train_cmd))
subprocess.run(train_cmd, check=True)
elapsed = time.time() - start_time
print(f" [OK] Core training completed in {elapsed / 60:.2f} minutes ({elapsed / max(1, effective_steps - resumed_step):.2f} s/step)!")

# 11. 智能检索所有生成的 .safetensors 权重并归档打包
print("\n>>> Locating and packaging all generated .safetensors weights...")
output_lora_dir = f"/kaggle/working/output/{args.lora_name}"
os.makedirs(output_lora_dir, exist_ok=True)

safetensors_files = glob.glob(f"{working_runs}/moss/{args.lora_name}/**/*.safetensors", recursive=True)
if not safetensors_files:
    safetensors_files = glob.glob(f"{trainer_dir}/**/runs/**/*.safetensors", recursive=True) + \
                        glob.glob("/kaggle/working/runs/**/*.safetensors", recursive=True)

safetensors_files = [
    f for f in set(safetensors_files)
    if "/tmp/models" not in f
    and "moss_image" not in os.path.basename(f).lower()
    and "qwen" not in os.path.basename(f).lower()
    and "flux2" not in os.path.basename(f).lower()
]

# 按修改时间排序，确保最终同名的 comfy / lycoris 归档版本始终为最新步数产物
safetensors_files.sort(key=lambda x: os.path.getmtime(x))

print(f"Found {len(safetensors_files)} LoRA weight files:")
for sf in safetensors_files:
    fname = os.path.basename(sf)
    sf_lower = sf.lower()
    if "-comfy" in sf_lower or fname.endswith("_comfy.safetensors"):
        dest_name = f"{args.lora_name}-comfy.safetensors"
    elif "-lycoris" in sf_lower or fname.endswith("_lycoris.safetensors"):
        dest_name = f"{args.lora_name}-lycoris.safetensors"
    elif fname == "trainable.safetensors":
        dest_name = f"{args.lora_name}-trainable.safetensors"
    else:
        dest_name = fname
    dest = os.path.join(output_lora_dir, dest_name)
    print(f" - Archiving {sf} -> {dest}")
    shutil.copy2(sf, dest)

summary_data = {
    "lora_name": args.lora_name,
    "train_mode": train_mode,
    "trigger_word": args.trigger_word if args.trigger_word else "(None)",
    "steps": effective_steps,
    "resumed_from_step": resumed_step,
    "repeats": args.repeats,
    "epochs": args.epochs,
    "rank": args.rank,
    "alpha": args.alpha,
    "learning_rate": args.lr,
    "model_variant": selected_variant,
    "arch_variant": arch_variant,
    "backbone_file": transformer_filename,
    "base_quant": "int8" if is_int8_transformer else "full",
    "vram_preset": effective_preset,
    "resolution": args.resolution,
    "precision": effective_precision,
    "fp16_fix": fp16_fix_val,
    "text_device": effective_text_device,
    "vae_device": args.vae_device,
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
