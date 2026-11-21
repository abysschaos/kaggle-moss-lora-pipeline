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

def _load_app_version() -> str:
    """读取项目根目录 VERSION 单一版本信源，缺失时安全回退默认版本号。"""
    ver_file = Path(__file__).resolve().parent / "VERSION"
    try:
        ver = ver_file.read_text(encoding="utf-8").strip()
        if ver:
            return ver
    except Exception:
        pass
    return "1.5.2"


def _env_bool(*keys: str, default: bool = False) -> bool:
    for k in keys:
        if k in os.environ and os.environ[k].strip():
            return os.environ[k].strip().lower() in ("1", "true", "yes", "on")
    return default


APP_VERSION = _load_app_version()

# 0. 解析命令行与环境变量超参数 (全面对齐本地引擎，支持显存档位、分辨率、精度与 Base/Heal/SFT 变体)
parser = argparse.ArgumentParser(description="OpenMOSS MOSS-Image Cloud LoRA Training Engine (Kaggle / Colab)")
parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
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
parser.add_argument("--algo", type=str, default=os.environ.get("ALGO", "lora"),
                    choices=["lora", "loha", "lokr"],
                    help="Adapter algorithm: lora (default), loha (Hadamard), lokr (Kronecker)")
parser.add_argument("--lokr_factor", type=int, default=int(os.environ.get("LOKR_FACTOR", "8")),
                    help="LoKr Kronecker factor (-1=auto, 4=high capacity, 8=default, 16=ultra-compact)")
parser.add_argument("--lokr_full_matrix", action="store_true",
                    default=_env_bool("LOKR_FULL_MATRIX"),
                    help="LoKr: use full second matrix factor")
parser.add_argument("--lokr_decompose_both", action="store_true",
                    default=_env_bool("LOKR_DECOMPOSE_BOTH"),
                    help="LoKr: decompose both Kronecker matrices when dimensions permit")
parser.add_argument("--train_norms", action="store_true",
                    default=_env_bool("TRAIN_NORMS"),
                    help="Also train AdaLN/RMS/LayerNorm scale weights alongside adapters")
parser.add_argument("--input_dropout", type=float, default=float(os.environ.get("INPUT_DROPOUT", "0.0")),
                    help="Adapter input dropout in [0.0, 1.0) (forced to 0.0 when --dora is enabled)")
parser.add_argument("--rank_dropout", type=float, default=float(os.environ.get("RANK_DROPOUT", "0.0")),
                    help="Adapter rank dropout in [0.0, 1.0) to prevent overfitting")
parser.add_argument("--module_dropout", type=float, default=float(os.environ.get("MODULE_DROPOUT", "0.0")),
                    help="Whole-module stochastic drop probability in [0.0, 1.0)")
parser.add_argument("--dora", action="store_true",
                    default=_env_bool("USE_DORA", "DORA"),
                    help="Enable DoRA (Weight-Decomposed Low-Rank Adaptation)")
parser.add_argument("--target_preset", type=str, default=os.environ.get("TARGET_PRESET", "attn-mlp"),
                    choices=["attn", "attn-mlp", "attn-mlp+modulation", "attn-mlp+text-refiner", "full-block", "all"],
                    help="Target transformer layers preset (default: attn-mlp)")
parser.add_argument("--loraplus_ratio", type=float, default=float(os.environ.get("LORAPLUS_RATIO", "1.0")),
                    help="LoRA+ up-projection learning rate ratio (1.0=off, 2.0~4.0 recommended for multi-concept)")
parser.add_argument("--subsets_json", type=str, default=os.environ.get("SUBSETS_JSON", ""),
                    help="Optional JSON file path or JSON array string for multi-subset training")
parser.add_argument("--lr", type=str, default=os.environ.get("LR", "1e-4"),
                    help="Learning rate (default: 1e-4)")
parser.add_argument("--optimizer", type=str, default=os.environ.get("OPTIMIZER", "adamw"),
                    choices=["adamw", "adamw8bit_kernel", "muon"],
                    help="Optimizer: adamw (default), adamw8bit_kernel, or muon")
parser.add_argument("--lr_scheduler", type=str, default=os.environ.get("LR_SCHEDULER", "cosine"),
                    choices=["cosine", "constant"],
                    help="LR scheduler mode (default: cosine)")
parser.add_argument("--warmup_steps", type=int, default=int(os.environ.get("WARMUP_STEPS", "-1")),
                    help="LR warmup steps (-1=auto 5%% clamped to [5, 100]; >=0 explicit)")
parser.add_argument("--weight_decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.01")),
                    help="Decoupled weight decay coefficient (default: 0.01)")
parser.add_argument("--wd_rules", type=str, default=os.environ.get("WD_RULES", "moss"),
                    choices=["moss", "none", "all"],
                    help="Weight decay parameter selection rule (default: moss)")
parser.add_argument("--ema", type=float, default=float(os.environ.get("EMA", "0.0")),
                    help="EMA decay rate (0.0=off, 0.99~0.995 recommended for long runs)")
parser.add_argument("--grad_acc", type=int, default=int(os.environ.get("GRAD_ACC", "1")),
                    help="Gradient accumulation steps (default: 1; 2~4 stabilizes complex outfits on <=12GB)")
parser.add_argument("--timestep", type=str, default=os.environ.get("TIMESTEP", "moss"),
                    choices=["moss", "uniform", "logit_normal"],
                    help="Timestep distribution: moss, uniform, or logit_normal")
parser.add_argument("--timestep_mu", type=float, default=float(os.environ.get("TIMESTEP_MU", "-1.09")),
                    help="Mean mu for moss/logit_normal timestep sampling")
parser.add_argument("--text_drop", type=float, default=float(os.environ.get("TEXT_DROP", "0.05")),
                    help="Empty-prompt unconditional dropout probability (default: 0.05)")
parser.add_argument("--flip", action="store_true",
                    default=_env_bool("FLIP"),
                    help="Enable random horizontal flip augmentation (caution on asymmetric outfits/text)")
parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")),
                    help="Random seed for reproducibility (default: 0)")
parser.add_argument("--save_format", type=str, default=os.environ.get("SAVE_FORMAT", "both"),
                    choices=["both", "comfy", "hf", "lycoris"],
                    help="Export adapter format: both, comfy, or hf/lycoris")
parser.add_argument("--base_precision", type=str, default=os.environ.get("BASE_PRECISION", "auto"),
                    choices=["auto", "int8", "fp8", "fp8_compute", "compute"],
                    help="Frozen backbone storage/compute precision: auto, int8, fp8, fp8_compute, compute")
parser.add_argument("--tune", type=str, default=os.environ.get("TUNE", "off"),
                    choices=["off", "table"],
                    help="Triton kernel autotuner mode: off (default) or table")
parser.add_argument("--overlong", type=str, default=os.environ.get("OVERLONG", "drop_tokens"),
                    choices=["drop_tokens", "drop"],
                    help="Overlong sample handling when tokens exceed MAX_TOKENS (default: drop_tokens)")
parser.add_argument("--vram_preset", type=str, default=os.environ.get("VRAM_PRESET", "auto"),
                    choices=["auto", "6g", "8g", "12g", "16g"],
                    help="VRAM optimization preset: auto, 6g, 8g, 12g, 16g (default: auto)")
parser.add_argument("--resolution", type=int, default=int(os.environ.get("RESOLUTION", "1024")),
                    help="Target training image resolution area, e.g. 1024, 896, 768, 512 (default: 1024)")
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
parser.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "auto"),
                    choices=["auto", "1", "2"],
                    help="Active GPU count: auto (2 on Linux with >=2 GPUs), 1, or 2 (default: auto)")
parser.add_argument("--parallel", type=str, default=os.environ.get("PARALLEL_MODE", "auto"),
                    choices=["auto", "none", "ddp", "pipeline"],
                    help="Multi-GPU parallelism mode on Linux: auto (DDP when 2 GPUs), none, ddp, pipeline")
parser.add_argument("--max_tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "0")),
                    help="Override MAX_TOKENS per GPU pack (0 = auto; Dual-T4 DDP uses 9216 for 2x1024px packing)")
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
parser.add_argument("--dataset_url", type=str, default=os.environ.get("DATASET_URL", ""),
                    help="Optional direct download URL for a dataset .zip archive if not mounted under /kaggle/input")
args, _ = parser.parse_known_args()

print("=" * 75)
print(f">>> OpenMOSS MOSS-Image Cloud LoRA Training Engine v{APP_VERSION} <<<")
print(f">>> Target LoRA: '{args.lora_name}' | Trigger Word: '{args.trigger_word if args.trigger_word else '(None)'}' <<<")
print(f">>> Hyperparameters: Repeats={args.repeats}, Epochs={args.epochs}, Steps={args.steps} <<<")
print(f">>> Architecture: Algo={args.algo.upper()}, DoRA={'ON' if args.dora else 'OFF'}, TrainNorms={'ON' if args.train_norms else 'OFF'}, Preset={args.target_preset}, LoRA+={args.loraplus_ratio} <<<")
print(f">>> Dimensions & Optim: Rank={args.rank}, Alpha={args.alpha}, LR={args.lr}, Opt={args.optimizer.upper()} ({args.lr_scheduler}), GradAcc={max(1, int(args.grad_acc))} <<<")
print(f">>> Dynamics & Augment: Timestep={args.timestep}(mu={args.timestep_mu}), TextDrop={args.text_drop}, Flip={bool(args.flip)}, Seed={args.seed}, EMA={args.ema}, SaveFmt={args.save_format} <<<")
print(f">>> Cloud Options: Variant={args.variant}, VRAM={args.vram_preset}, Res={args.resolution}, Quant={args.base_quant}, Prec={args.precision}, GPUs={args.gpus}, Parallel={args.parallel} <<<")
print("=" * 75)

# 1. 验证 GPU 与宿主机 PyTorch / TorchVision 原生驱动
vram_gb = 16.0
sm_major, sm_minor = 0, 0
bf16_supported = False
gpu_count = 1
active_gpus = 1
effective_parallel = "none"
try:
    import torch
    import torchvision
    print(f"Python Executable: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_count = max(1, int(torch.cuda.device_count()))
        for gi in range(gpu_count):
            g_name = torch.cuda.get_device_name(gi)
            g_props = torch.cuda.get_device_properties(gi)
            print(f"GPU[{gi}]: {g_name} (sm{int(g_props.major)}{int(g_props.minor)}, {g_props.total_memory / (1024**3):.2f} GB)")
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        sm_major, sm_minor = int(props.major), int(props.minor)
        bf16_supported = (sm_major >= 8)
        print(f"Primary GPU: {gpu_name} (sm{sm_major}{sm_minor}) | Native BF16 Supported (sm>=80): {bf16_supported}")
        if args.gpus == "auto":
            active_gpus = 2 if (sys.platform.startswith("linux") and gpu_count >= 2) else 1
        else:
            active_gpus = max(1, min(int(args.gpus), gpu_count))
        if active_gpus >= 2 and sys.platform.startswith("linux"):
            if args.parallel in ("auto", "ddp"):
                effective_parallel = "ddp"
            elif args.parallel == "pipeline":
                effective_parallel = "pipeline"
            else:
                effective_parallel = "none"
                active_gpus = 1
        else:
            effective_parallel = "none"
            active_gpus = 1
        # KohakuMossTrainer 强制要求全部参与训练的 GPU 原生具备 sm >= 80 (Ampere+) 才允许 BF16
        bf16_supported = all(int(torch.cuda.get_device_properties(i).major) >= 8 for i in range(active_gpus))
        print(f" [Multi-GPU] Detected={gpu_count} GPU(s) | Active={active_gpus} | Parallel={effective_parallel.upper()} | Cluster Native BF16={bf16_supported}")
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
    base_profile_name = "ckpt-ring2-actoff-optcpu-gradoff"
    print(f" [VRAM Preset] Applied 6G Profile (extreme ~4.4GB VRAM): {vram_note}")
elif effective_preset == "8g":
    recipe_import = 'use_config("../moss/_base/base_lora.py")\nuse_config("../moss/vram/vram_8g.py")'
    max_tokens_val = 4352
    base_profile_name = "minimal"
    print(f" [VRAM Preset] Applied 8G Profile (~5.5GB VRAM): {vram_note}")
elif effective_preset == "12g":
    recipe_import = 'use_config("../moss/_base/base_lora.py")\nuse_config("../moss/vram/vram_12g.py")'
    max_tokens_val = 6144
    base_profile_name = "ckpt-ring4-actoff-optstream-gradoff"
    print(f" [VRAM Preset] Applied 12G Profile (~8.5GB VRAM, MAX_TOKENS=6144): {vram_note}")
else:
    recipe_import = 'use_config("../moss/_base/base_lora.py")\nuse_config("../moss/vram/vram_16g.py")'
    max_tokens_val = 8192
    base_profile_name = "low"
    print(f" [VRAM Preset] Applied 16G Profile (MAX_TOKENS=8192, MEMORY_PROFILE=low): {vram_note}")

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
    use_int8_backbone = (effective_preset in ("6g", "8g")) or (effective_parallel == "ddp")
else:
    use_int8_backbone = (args.base_quant == "int8")

parallel_cfg_str = ""
if effective_parallel == "ddp":
    if effective_preset == "16g" and use_int8_backbone:
        base_profile_name = "ckpt-resident"
        max_tokens_val = 9216
        print(" [Dual-GPU DDP] 2x T4 16GB: Enabled int8 backbone resident mode (MEMORY_PROFILE='ckpt-resident', MAX_TOKENS=9216 -> 2x 1024px/GPU = 4 imgs/step).")
    parallel_cfg_str = (
        'PARALLEL = "ddp"\n'
        'DEVICES = (0, 1)\n'
        'DIST_BACKEND = "auto"\n'
        'DDP_REDUCE_DTYPE = "fp32"\n'
        'DDP_BUCKET_MB = 25'
    )
elif effective_parallel == "pipeline":
    use_int8_backbone = False
    args.base_precision = "compute"
    base_profile_name = "ckpt-resident"
    recipe_import += '\nuse_config("../moss/parallel/parallel_pipeline.py")'
    parallel_cfg_str = 'DEVICES = (0, 1)'
    print(" [Dual-GPU Pipeline] Enabled 1F1B stage split across GPU(0, 1) with full-precision resident weights.")

if args.optimizer != "adamw" and base_profile_name != "ckpt-resident":
    _opt_remap = {
        "6g": "ckpt-ring2-actoff",
        "8g": "ckpt-ring2-actoff",
        "12g": "ckpt-ring4-actoff",
        "16g": "ckpt-ring8-actoff",
    }
    resolved_profile_name = _opt_remap.get(effective_preset, "ckpt-ring4-actoff")
    print(f" [Optimizer Guard] OPTIMIZER='{args.optimizer}' requires optim_state='gpu' & fused_step=False; remapped MEMORY_PROFILE '{base_profile_name}' -> '{resolved_profile_name}'.")
else:
    resolved_profile_name = base_profile_name
memory_profile_str = f'MEMORY_PROFILE = "{resolved_profile_name}"'

if int(args.max_tokens) > 0:
    max_tokens_val = int(args.max_tokens)
    print(f" [Token Packing Override] MAX_TOKENS={max_tokens_val}")

if args.text_device == "auto":
    effective_text_device = "cpu" if (vram_gb <= 10.0 or effective_preset == "6g") else "cuda"
else:
    effective_text_device = args.text_device.lower()
print(f" [Devices] Text Encoder: {effective_text_device.upper()} | VAE: {args.vae_device.upper()}")
print(f" [Resolution] Target Area: {args.resolution}x{args.resolution} ({args.resolution * args.resolution} px) | MAX_TOKENS={max_tokens_val}")

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
valid_img_exts = {".jpg", ".jpeg", ".png", ".webp"}
trigger_clean = args.trigger_word.strip() if args.trigger_word else ""
cloud_subsets_summary = []

# 3. 智能发现与准备数据集 (支持多子集独立配比 --subsets_json，或单图集 /kaggle/input/ 挂载与解压)
parsed_cloud_subsets = []
if args.subsets_json and args.subsets_json.strip():
    raw_sub = args.subsets_json.strip()
    if os.path.isfile(raw_sub):
        parsed_cloud_subsets = json.loads(Path(raw_sub).read_text(encoding="utf-8"))
    else:
        parsed_cloud_subsets = json.loads(raw_sub)

if parsed_cloud_subsets:
    print(f"\n>>> Preparing Multi-Subset Kohya dataset ({len(parsed_cloud_subsets)} subsets) in /tmp/workspace_dataset...")
    active_folders = set()
    total_exp = 0
    for idx, item in enumerate(parsed_cloud_subsets, start=1):
        if not isinstance(item, dict) or item.get("enabled", True) is False:
            continue
        s_dir = str(item.get("path", "")).strip()
        if not s_dir or not os.path.isdir(s_dir):
            raise FileNotFoundError(f"Cloud subset #{idx} directory not found: {s_dir}")
        s_rep = max(1, int(item.get("repeats", 1)))
        s_name_raw = str(item.get("name") or os.path.basename(s_dir.rstrip("/\\")))
        s_name = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", s_name_raw).strip("_") or f"subset_{idx}"
        folder_name = f"{s_rep}_{s_name}"
        dup_n = 2
        while folder_name in active_folders:
            folder_name = f"{s_rep}_{s_name}_{dup_n}"
            dup_n += 1
        active_folders.add(folder_name)
        dst_sub = os.path.join(workspace_data_root, folder_name)
        os.makedirs(dst_sub, exist_ok=True)
        s_trig = str(item.get("trigger", "") or "").strip()

        src_imgs = [f for f in os.listdir(s_dir) if os.path.splitext(f)[1].lower() in valid_img_exts]
        if not src_imgs:
            raise RuntimeError(f"No images found in cloud subset '{s_name}' ({s_dir})")
        for f in src_imgs:
            shutil.copy2(os.path.join(s_dir, f), os.path.join(dst_sub, f))
            base_name = os.path.splitext(f)[0]
            src_txt = os.path.join(s_dir, base_name + ".txt")
            cap = ""
            if os.path.exists(src_txt):
                with open(src_txt, "r", encoding="utf-8", errors="ignore") as tf:
                    cap = tf.read().strip()
            if s_trig and s_trig.lower() not in cap.lower():
                cap = f"{s_trig}, {cap}" if cap else s_trig
            with open(os.path.join(dst_sub, base_name + ".txt"), "w", encoding="utf-8") as tf:
                tf.write(cap)
        exp = len(src_imgs) * s_rep
        total_exp += exp
        cloud_subsets_summary.append({
            "name": s_name_raw, "folder": folder_name, "images": len(src_imgs),
            "repeats": s_rep, "epoch_exposures": exp, "trigger": s_trig or "(None)"
        })

    for sub_name in os.listdir(workspace_data_root):
        sub_path = os.path.join(workspace_data_root, sub_name)
        if os.path.isdir(sub_path) and sub_name != "cache" and sub_name not in active_folders:
            shutil.rmtree(sub_path, ignore_errors=True)
    for s in cloud_subsets_summary:
        pct = (s["epoch_exposures"] / total_exp * 100.0) if total_exp > 0 else 0.0
        s["share_percent"] = round(pct, 1)
        print(f" - [{s['folder']}] {s['images']} imgs x {s['repeats']} repeats = {s['epoch_exposures']} exp/epoch ({pct:.1f}%) | Trigger: {s['trigger']}")
else:
    target_data_dir = os.path.join(workspace_data_root, f"{args.repeats}_{args.lora_name}")
    os.makedirs(target_data_dir, exist_ok=True)
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

            if trigger_clean and trigger_clean.lower() not in caption.lower():
                caption = f"{trigger_clean}, {caption}" if caption else trigger_clean

            with open(dest_txt, "w", encoding="utf-8") as tf:
                tf.write(caption)
    else:
        local_zip = "/tmp/dataset.zip"
        chosen_zip = None
        found_zips = glob.glob("/kaggle/**/dataset.zip", recursive=True) + glob.glob("**/dataset.zip", recursive=True)
        if found_zips and os.path.getsize(found_zips[0]) > 1024:
            chosen_zip = found_zips[0]
            print(f"Found local dataset archive candidate: {chosen_zip}")
        elif args.dataset_url:
            print(f"Fetching dataset archive via curl from --dataset_url: {args.dataset_url}...")
            curl_cmd = [
                "curl", "-s", "-L",
                "--retry", "3",
                "--retry-delay", "2",
                "--retry-all-errors",
                args.dataset_url,
                "-o", local_zip
            ]
            subprocess.run(curl_cmd, check=True)
            if os.path.exists(local_zip) and os.path.getsize(local_zip) > 1024:
                chosen_zip = local_zip
                print(f"Downloaded dataset archive: {chosen_zip} ({os.path.getsize(chosen_zip) / (1024**2):.2f} MB)")

        if not chosen_zip or not os.path.exists(chosen_zip):
            raise RuntimeError(
                "No dataset found! Please mount your Kaggle Private Dataset under /kaggle/input/, "
                "pass --dataset_dir <path>, or pass --dataset_url <zip_url>."
            )

        print(f"Unpacking {chosen_zip} -> {target_data_dir}...")
        with zipfile.ZipFile(chosen_zip, 'r') as zf:
            zf.extractall(target_data_dir)
        if trigger_clean:
            for txt_f in glob.glob(os.path.join(target_data_dir, "*.txt")):
                with open(txt_f, "r", encoding="utf-8", errors="ignore") as tf:
                    cap = tf.read().strip()
                if trigger_clean and trigger_clean.lower() not in cap.lower():
                    cap = f"{trigger_clean}, {cap}" if cap else trigger_clean
                    with open(txt_f, "w", encoding="utf-8") as tf:
                        tf.write(cap)

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
        patched_obj = re.sub(
            r'raw_end = sub\.get\("end", -1\)(?:\s+if isinstance\(sub, dict\) else -1)*',
            'raw_end = sub.get("end", -1) if isinstance(sub, dict) else -1',
            obj_content,
        )
        patched_obj = re.sub(
            r'warmup = sub\.get\("warmup", 0\)(?:\s+if isinstance\(sub, dict\) else 0)*',
            'warmup = sub.get("warmup", 0) if isinstance(sub, dict) else 0',
            patched_obj,
        )
        if patched_obj != obj_content:
            with open(obj_check_file, "w", encoding="utf-8") as f:
                f.write(patched_obj)
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

def _resolve_base_precision_cfg(req_base_prec: str, is_int8_file: bool, sm_major: int, sm_minor: int) -> tuple:
    req = (req_base_prec or "auto").lower()
    sm_ok_fp8 = (sm_major > 8) or (sm_major == 8 and sm_minor >= 9)
    if is_int8_file:
        if req in ("fp8", "fp8_compute"):
            print(f" [Base Precision Guard] Pre-quantized int8_convrot backbone cannot run as '{req}'. Falling back to 'int8'.")
            return "int8", 'BASE_PRECISION = "int8"'
        if req == "compute":
            return "compute", 'BASE_PRECISION = "compute"'
        return "int8", 'BASE_PRECISION = "int8"'
    if req == "int8":
        return "int8", 'BASE_PRECISION = "int8"'
    if req == "fp8":
        if sm_ok_fp8:
            return "fp8", 'BASE_PRECISION = "fp8"'
        print(f" [Base Precision Guard] GPU sm{sm_major}{sm_minor} < sm89 -> falling back to 'int8'.")
        return "int8", 'BASE_PRECISION = "int8"'
    if req == "fp8_compute":
        if sm_ok_fp8:
            return "fp8_compute", 'BASE_PRECISION = "fp8_compute"\nFP8_DGRAD = "16bit"'
        print(f" [Base Precision Guard] GPU sm{sm_major}{sm_minor} < sm89 -> falling back to 'int8'.")
        return "int8", 'BASE_PRECISION = "int8"'
    if req == "compute":
        return "compute", 'BASE_PRECISION = "compute"'
    return "auto", ""


transformer_filename = os.path.basename(hf_filename)
is_int8_transformer = "int8" in transformer_filename.lower()
effective_base_prec, base_prec_str = _resolve_base_precision_cfg(
    args.base_precision, is_int8_transformer, sm_major, sm_minor
)
print(f" [Base Precision] Effective Backbone Mode: {effective_base_prec}")

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
    info = {
        "rank": None,
        "alpha": None,
        "step": None,
        "algo": "lora",
        "dora": False,
        "preset": None,
        "train_norms": None,
        "lokr_factor": None,
        "lokr_full_matrix": None,
        "lokr_decompose_both": None,
    }
    try:
        import struct
        with open(sf_path, "rb") as f:
            h_size = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(h_size).decode("utf-8"))
        meta = hdr.get("__metadata__", {}) or {}
        tensor_keys = [k for k in hdr.keys() if k != "__metadata__"]

        if "ss_network_dim" in meta:
            try:
                info["rank"] = int(float(meta["ss_network_dim"]))
            except Exception:
                pass
        if "ss_network_alpha" in meta:
            try:
                info["alpha"] = float(meta["ss_network_alpha"])
            except Exception:
                pass
        if "kohakumoss_step" in meta:
            try:
                info["step"] = int(meta["kohakumoss_step"])
            except Exception:
                pass
        if "kohakumoss_algo" in meta and str(meta["kohakumoss_algo"]).strip():
            info["algo"] = str(meta["kohakumoss_algo"]).strip().lower()
        if "kohakumoss_spec" in meta:
            try:
                spec = json.loads(meta["kohakumoss_spec"])
                if info["rank"] is None and "dim" in spec:
                    info["rank"] = int(spec["dim"])
                if info["alpha"] is None and "alpha" in spec:
                    info["alpha"] = float(spec["alpha"])
                if spec.get("algo"):
                    info["algo"] = str(spec["algo"]).lower()
                if "dora" in spec:
                    info["dora"] = bool(spec["dora"])
                if "train_norms" in spec:
                    info["train_norms"] = bool(spec["train_norms"])
                if spec.get("preset") and isinstance(spec["preset"], str):
                    info["preset"] = spec["preset"]
                akw = spec.get("algo_kwargs")
                if isinstance(akw, dict):
                    if "factor" in akw:
                        info["lokr_factor"] = int(akw["factor"])
                    if "full_matrix" in akw:
                        info["lokr_full_matrix"] = bool(akw["full_matrix"])
                    if "decompose_both" in akw:
                        info["lokr_decompose_both"] = bool(akw["decompose_both"])
            except Exception:
                pass

        if any(k.endswith(".dora_scale") for k in tensor_keys):
            info["dora"] = True
        if info["algo"] == "lora":
            if any(".hada_w1_" in k for k in tensor_keys):
                info["algo"] = "loha"
            elif any(".lokr_w" in k for k in tensor_keys):
                info["algo"] = "lokr"

        if info["rank"] is None:
            for k in tensor_keys:
                if k.endswith(".lora_down.weight") or k.endswith(".hada_w1_b") or k.endswith(".lokr_w2_b"):
                    shp = hdr[k].get("shape", [])
                    if len(shp) >= 2:
                        info["rank"] = int(shp[0])
                        break
        if info["alpha"] is None and info["rank"] is not None:
            info["alpha"] = float(info["rank"])
    except Exception:
        pass
    return info


def _align_args_from_spec(l_info: dict, mode_tag: str):
    if l_info.get("rank") and l_info["rank"] != args.rank:
        print(f" [{mode_tag} Auto-Align] Aligned Rank {args.rank} -> {l_info['rank']}")
        args.rank = l_info["rank"]
        if l_info.get("alpha"):
            args.alpha = l_info["alpha"]
    if l_info.get("algo") and l_info["algo"] in ("lora", "loha", "lokr") and l_info["algo"] != args.algo:
        print(f" [{mode_tag} Auto-Align] Aligned Algo {args.algo} -> {l_info['algo']}")
        args.algo = l_info["algo"]
    if bool(l_info.get("dora", False)) != bool(args.dora):
        print(f" [{mode_tag} Auto-Align] Aligned DoRA {args.dora} -> {bool(l_info.get('dora', False))}")
        args.dora = bool(l_info.get("dora", False))
    if l_info.get("preset") and l_info["preset"] != args.target_preset:
        print(f" [{mode_tag} Auto-Align] Aligned Target Preset {args.target_preset} -> {l_info['preset']}")
        args.target_preset = l_info["preset"]
    if l_info.get("train_norms") is not None and bool(l_info["train_norms"]) != bool(args.train_norms):
        print(f" [{mode_tag} Auto-Align] Aligned Train Norms {args.train_norms} -> {bool(l_info['train_norms'])}")
        args.train_norms = bool(l_info["train_norms"])
    if args.algo == "lokr":
        if l_info.get("lokr_factor") is not None and int(l_info["lokr_factor"]) != int(args.lokr_factor):
            print(f" [{mode_tag} Auto-Align] Aligned LoKr Factor {args.lokr_factor} -> {int(l_info['lokr_factor'])}")
            args.lokr_factor = int(l_info["lokr_factor"])
        if l_info.get("lokr_full_matrix") is not None:
            args.lokr_full_matrix = bool(l_info["lokr_full_matrix"])
        if l_info.get("lokr_decompose_both") is not None:
            args.lokr_decompose_both = bool(l_info["lokr_decompose_both"])


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
        _align_args_from_spec(l_info, "Resume")
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
    _align_args_from_spec(l_info, "Init-From")
    init_from_cfg_str = f'INIT_FROM = "{stage_init_dir}"'
    print(f" [Init-From] Continuing training from existing LoRA: {chosen_sf} (Rank={args.rank}, Alpha={args.alpha}, Steps={effective_steps})")

# 8. 生成标准训练与数据配置 (与本地引擎完全对齐，注入分辨率、显存档位、架构变体、续炼与防 OOM 守卫)
user_cfg_dir = os.path.join(trainer_dir, "configs", "user")
os.makedirs(user_cfg_dir, exist_ok=True)

text_drop_val = max(0.0, min(1.0, float(args.text_drop)))
flip_val = bool(args.flip)
seed_val = int(args.seed)

data_cfg_content = f"""
data_root = "/tmp/workspace_dataset"
WEIGHTS = "/tmp/models"

DATASETS = [
    {{
        "source": "kohya",
        "path": data_root,
        "repeat_prefix": "first",
        "artist_from_folder": None,
        "txt_as": "nl",
    }}
]
RESOLUTION_MODE = "resize"
TARGET_AREA = {args.resolution} * {args.resolution}
UPSCALE = "never"
ASPECT_RANGE = (0.33, 3.0)
BATCHING = "tokens"
MAX_TOKENS = {max_tokens_val}
OVERLONG = "{args.overlong}"
CAPTION = {{
    "template": "nl",
}}
CAPTION_VARIANTS = 1
TEXT_DROP = {text_drop_val}
LATENT_CACHE = "offline"
TEXT_CACHE = "offline"
FLIP = {flip_val}
TEXT_ENCODER_KWARGS = {{"path": WEIGHTS, "device": "{effective_text_device}"}}
IMAGE_ENCODER_KWARGS = {{"path": WEIGHTS, "device": "{args.vae_device}"}}
VAE_ENCODE = {{"compile": "off"}}
"""

raw_warmup = min(100, max(5, int(effective_steps * 0.05))) if (args.warmup_steps < 0 and effective_steps > 10) else max(0, int(args.warmup_steps))
if args.warmup_steps < 0 and effective_steps <= 10:
    raw_warmup = 0
warmup_steps = max(0, min(raw_warmup, max(0, int(effective_steps) - 1)))
sched_mode = args.lr_scheduler.lower() if args.lr_scheduler.lower() in ("cosine", "constant") else "cosine"
if sched_mode == "constant":
    schedule_cfg_str = f'SCHEDULE = {{\n    "mode": "constant",\n    "warmup": {warmup_steps},\n    "end": -1,\n}}'
else:
    schedule_cfg_str = f'SCHEDULE = {{\n    "mode": "cosine",\n    "warmup": {warmup_steps},\n    "end": -1,\n    "min_value": 0.1,\n}}'

grad_acc_val = max(1, int(args.grad_acc))
timestep_mu_val = float(args.timestep_mu)
timestep_mode = args.timestep.lower()
if timestep_mode == "uniform":
    timestep_kwargs_str = 'TIMESTEP_KWARGS = {"shift": 1.0}'
elif timestep_mode == "logit_normal":
    timestep_kwargs_str = f'TIMESTEP_KWARGS = {{\n    "mu": {timestep_mu_val},\n    "s": 1.0,\n    "shift": 1.0,\n}}'
else:
    timestep_mode = "moss"
    timestep_kwargs_str = f'TIMESTEP_KWARGS = {{\n    "mu": {timestep_mu_val},\n    "s": 0.95,\n    "shift": 1.0,\n    "cutoff": 0.95,\n    "uniform_mix": 0.01,\n}}'

save_every_val = max(0, int(args.save_every))
resolved_save_format = "hf" if args.save_format.lower() == "lycoris" else args.save_format.lower()
weight_decay_val = max(0.0, float(args.weight_decay))
ema_val = max(0.0, min(0.99999, float(args.ema)))

in_drop = max(0.0, min(0.99, float(args.input_dropout)))
rank_drop = max(0.0, min(0.99, float(args.rank_dropout)))
mod_drop = max(0.0, min(0.99, float(args.module_dropout)))
if args.dora and in_drop > 0.0:
    print(f" [DoRA Guard] DoRA disables input dropout; forcing input_dropout {in_drop} -> 0.0.")
    in_drop = 0.0

fp16_fix_kwargs_str = 'FP16_FIX_KWARGS = {"c": 1 / 16}' if fp16_fix_val == "online" else "FP16_FIX_KWARGS = {}"

lokr_factor_val = int(args.lokr_factor)
if args.algo == "loha":
    algo_kwargs_str = 'ALGO_KWARGS = {"use_tucker": False, "use_scalar": False, "rank_dropout_scale": False}'
elif args.algo == "lokr":
    algo_kwargs_str = (
        f'ALGO_KWARGS = {{"factor": {lokr_factor_val}, "decompose_both": {bool(args.lokr_decompose_both)}, '
        f'"full_matrix": {bool(args.lokr_full_matrix)}, "use_tucker": False, "rank_dropout_scale": False}}'
    )
else:
    algo_kwargs_str = "ALGO_KWARGS = {}"
dora_detach_val = True if args.dora else False
loraplus_val = max(1.0, float(args.loraplus_ratio))

train_cfg_content = f"""
from kohakuengine import use_config

{recipe_import}
use_config("{args.lora_name}_data.py")

VARIANT = "{arch_variant}"
RUN_NAME = "{args.lora_name}"
SEED = {seed_val}
TRANSFORMER = "{transformer_filename}"
{parallel_cfg_str}
{base_prec_str}
{memory_profile_str}
{resume_cfg_str}
{init_from_cfg_str}
SAMPLE_EVERY = 0
VALIDATE_EVERY = 0
TUNE = "{args.tune}"
FP16_FIX = "{fp16_fix_val}"
{fp16_fix_kwargs_str}
PRECISION = "{effective_precision}"
ALGO = "{args.algo}"
{algo_kwargs_str}
DORA = {bool(args.dora)}
DORA_DETACH = {dora_detach_val}
TRAIN_NORMS = {bool(args.train_norms)}
DROPOUT = {{"input": {in_drop}, "rank": {rank_drop}, "module": {mod_drop}}}
PRESET = "{args.target_preset}"
LORAPLUS_RATIO = {loraplus_val}
DIM = {args.rank}
ALPHA = {args.alpha}
STEPS = {effective_steps}
GRAD_ACC = {grad_acc_val}
OPTIMIZER = "{args.optimizer}"
LR = {args.lr}
WEIGHT_DECAY = {weight_decay_val}
WD_RULES = "{args.wd_rules}"
EMA = {ema_val}
TIMESTEP = "{timestep_mode}"
{timestep_kwargs_str}
SAVE_EVERY = {save_every_val}
SAVE_MINUTES = 25
KEEP_LAST = 5
SAVE_FORMAT = "{resolved_save_format}"
SAVE_FUSED_QKV = False
{schedule_cfg_str}
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
    "algo": args.algo,
    "lokr_factor": lokr_factor_val,
    "lokr_full_matrix": bool(args.lokr_full_matrix),
    "lokr_decompose_both": bool(args.lokr_decompose_both),
    "train_norms": bool(args.train_norms),
    "dropout": {"input": in_drop, "rank": rank_drop, "module": mod_drop},
    "dora": bool(args.dora),
    "dora_detach": dora_detach_val,
    "target_preset": args.target_preset,
    "loraplus_ratio": loraplus_val,
    "dataset_mode": "multi_subset" if cloud_subsets_summary else "single",
    "subsets": cloud_subsets_summary,
    "optimizer": args.optimizer,
    "learning_rate": args.lr,
    "lr_scheduler": sched_mode,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay_val,
    "wd_rules": args.wd_rules,
    "ema": ema_val,
    "model_variant": selected_variant,
    "arch_variant": arch_variant,
    "backbone_file": transformer_filename,
    "base_quant": "int8" if is_int8_transformer else "full",
    "vram_preset": effective_preset,
    "memory_profile": resolved_profile_name,
    "resolution": args.resolution,
    "precision": effective_precision,
    "base_precision": effective_base_prec,
    "grad_acc": grad_acc_val,
    "timestep": timestep_mode,
    "timestep_mu": timestep_mu_val,
    "text_drop": text_drop_val,
    "flip": flip_val,
    "seed": seed_val,
    "save_format": resolved_save_format,
    "tune": args.tune,
    "overlong": args.overlong,
    "active_gpus": active_gpus,
    "parallel": effective_parallel,
    "max_tokens": max_tokens_val,
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
