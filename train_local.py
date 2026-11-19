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

# Windows 下彻底禁止内部子进程 (pip/git/kogine) 闪烁弹出 CMD 黑框
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# 确保当前 Python 解释器配套的 Scripts 目录进入系统 PATH，保障 kogine、accelerate 等 CLI 工具顺利执行
py_scripts_dir = Path(sys.executable).parent / "Scripts"
if py_scripts_dir.exists():
    os.environ["PATH"] = str(py_scripts_dir) + os.pathsep + os.environ.get("PATH", "")

# 防御 Windows 中文用户名路径导致 Triton / TorchInductor 发生 UnicodeDecodeError 崩溃
# 优先使用纯 ASCII 根路径（如 C:/tmp/moss_pipeline），彻底阻绝 Triton C++ getenv utf-8 致命解析异常
root_c = Path("C:/tmp/moss_pipeline")
try:
    root_c.mkdir(parents=True, exist_ok=True)
    safe_temp_dir = root_c
except Exception:
    safe_temp_dir = (Path(__file__).resolve().parent / "workspace" / "temp").resolve()
    safe_temp_dir.mkdir(parents=True, exist_ok=True)

for sub_name in ["triton", "dump", "override", "inductor"]:
    (safe_temp_dir / sub_name).mkdir(parents=True, exist_ok=True)

safe_temp_posix = safe_temp_dir.as_posix()
os.environ["TEMP"] = safe_temp_posix
os.environ["TMP"] = safe_temp_posix
os.environ["TRITON_CACHE_DIR"] = f"{safe_temp_posix}/triton"
os.environ["TRITON_DUMP_DIR"] = f"{safe_temp_posix}/dump"
os.environ["TRITON_OVERRIDE_DIR"] = f"{safe_temp_posix}/override"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{safe_temp_posix}/inductor"

def _load_app_version() -> str:
    """读取项目根目录 VERSION 单一版本信源，缺失时安全回退默认版本号。"""
    ver_file = Path(__file__).resolve().parent / "VERSION"
    try:
        ver = ver_file.read_text(encoding="utf-8").strip()
        if ver:
            return ver
    except Exception:
        pass
    return "1.2.2"


APP_VERSION = _load_app_version()


def print_banner(msg: str):
    width = 78
    print("=" * width)
    print(f">>> {msg}")
    print("=" * width)

def sanitize_path(p: str) -> str:
    """标准化路径为 POSIX 正斜杠格式，防止在生成的 Python 配置文件中发生反斜杠转义错误。"""
    return Path(p).resolve().as_posix()

def find_broad_comfy_paths():
    """全自动宽域智能检索本地 ComfyUI 模型库与 Python 解释器路径。
    自动覆盖：环境变量、相对路径、系统常见盘符根目录与常用下载目录下的 ComfyUI 便携版或各类独立环境。
    完全杜绝任何个人绝对路径的硬编码，确保开源泛用性与隐私安全。
    """
    candidates = []
    
    # 1. 检查环境变量
    for env_key in ["COMFYUI_PATH", "COMFY_PATH", "COMFYUI_MODELS"]:
        val = os.environ.get(env_key)
        if val and os.path.exists(val):
            p = Path(val)
            if p.is_dir() and p not in candidates:
                candidates.append(p)

    # 2. 检查项目同级与上级相对路径
    current = Path(".").resolve()
    for parent in [current, current.parent, current.parent.parent]:
        try:
            for p in parent.glob("*[Cc]omfy*"):
                if p.is_dir() and p not in candidates:
                    candidates.append(p)
        except Exception:
            pass

    # 3. 扫描 Windows 常见驱动器 (C:, D:, E:, F:, G:) 的一到两级目录
    for drive_letter in ["C", "D", "E", "F", "G"]:
        drive_root = Path(f"{drive_letter}:/")
        if not drive_root.exists():
            continue
        try:
            for p in drive_root.glob("*[Cc]omfy*"):
                if p.is_dir() and p not in candidates:
                    candidates.append(p)
        except Exception:
            pass
        try:
            for p in drive_root.glob("*/*[Cc]omfy*"):
                if p.is_dir() and p not in candidates:
                    candidates.append(p)
        except Exception:
            pass

    # 4. 扫描用户当前主目录 (Downloads, Desktop)
    home = Path.home()
    for sub in ["Downloads", "Desktop"]:
        sp = home / sub
        if sp.exists():
            try:
                for p in sp.glob("*[Cc]omfy*"):
                    if p.is_dir() and p not in candidates:
                        candidates.append(p)
                for p in sp.glob("*/*[Cc]omfy*"):
                    if p.is_dir() and p not in candidates:
                        candidates.append(p)
            except Exception:
                pass

    found_models = []
    found_pythons = []

    for base in candidates:
        # 探测模型库目录
        for m_pat in ["models", "ComfyUI/models", "*/ComfyUI/models"]:
            try:
                for m in base.glob(m_pat):
                    if m.is_dir() and m not in found_models:
                        found_models.append(m)
            except Exception:
                pass
        # 探测 Python 解释器
        for py_pat in ["python/python.exe", "python_embeded/python.exe", "*/python/python.exe"]:
            try:
                for py in base.glob(py_pat):
                    if py.is_file() and py not in found_pythons:
                        found_pythons.append(py)
            except Exception:
                pass

    return found_models, found_pythons

def probe_local_models(comfy_models_dir: str = None) -> dict:
    """智能检索本地已有的 MOSS-Image 底模、文本编码器与 VAE，实现零下载秒级就绪。"""
    search_dirs = []
    if comfy_models_dir and os.path.exists(comfy_models_dir):
        search_dirs.append(Path(comfy_models_dir))
    
    # 动态宽域扫描系统已安装的 ComfyUI 模型库 (完全移除任何个人固定路径)
    broad_models, _ = find_broad_comfy_paths()
    for bm in broad_models:
        if bm not in search_dirs:
            search_dirs.append(bm)

    local_models = Path("./models").resolve()
    if local_models.exists() and local_models not in search_dirs:
        search_dirs.append(local_models)

    found = {
        "transformer": None,
        "text_encoder": None,
        "vae": None,
        "base_weights_dir": None
    }

    # 1. 查找 Transformer (优先选择 int8 旋转量化版，防御 8G/12G 显存 OOM，全面兼容 Base/Heal/SFT)
    heal_candidates = [
        "**/moss_image_heal_int8_convrot.safetensors",
        "**/moss_image_base_int8_convrot.safetensors",
        "**/moss_image_sft_int8_convrot.safetensors",
        "**/moss_image_heal_bf16.safetensors",
        "**/moss_image_heal_fp16.safetensors",
        "**/moss_image_base_bf16.safetensors",
        "**/moss_image_sft_bf16.safetensors",
        "**/moss_image_base.safetensors",
        "**/moss_image_sft.safetensors"
    ]
    for sdir in search_dirs:
        for pat in heal_candidates:
            matches = list(sdir.glob(pat))
            if matches:
                found["transformer"] = matches[0]
                break
        if found["transformer"]:
            break

    # 2. 查找 Text Encoder (Qwen3.5-9B，优先匹配量化版 int8 避免 16GB 物理内存发生溢出换页)
    text_candidates = [
        "**/qwen3.5_9b_int8_convrot.safetensors",
        "**/qwen3.5_9b_bf16.safetensors",
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

    # 3. 幂等修补 objective.py 调度器字典防护（防止重复运行产生冗余嵌套）
    obj_check_file = trainer_dir / "src" / "kohakumoss" / "training" / "checks" / "objective.py"
    if obj_check_file.exists():
        try:
            content = obj_check_file.read_text(encoding="utf-8")
            patched = re.sub(
                r'raw_end = sub\.get\("end", -1\)(?:\s+if isinstance\(sub, dict\) else -1)*',
                'raw_end = sub.get("end", -1) if isinstance(sub, dict) else -1',
                content,
            )
            patched = re.sub(
                r'warmup = sub\.get\("warmup", 0\)(?:\s+if isinstance\(sub, dict\) else 0)*',
                'warmup = sub.get("warmup", 0) if isinstance(sub, dict) else 0',
                patched,
            )
            if patched != content:
                obj_check_file.write_text(patched, encoding="utf-8")
            print(" [OK] Patched objective.py for schedule defensive checks.")
        except Exception as e:
            print(f" Warning patching objective.py: {e}")

    # 4. 为 native.py 注入【随时保存当前进度并优雅暂停】钩子 (KOHAKUMOSS_PAUSE_FILE)
    native_driver_file = trainer_dir / "src" / "kohakumoss" / "training" / "drivers" / "native.py"
    if native_driver_file.exists():
        try:
            content = native_driver_file.read_text(encoding="utf-8")
            if "KOHAKUMOSS_PAUSE_FILE" not in content:
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
                    "                print(f'\\n[Pause & Save] Detected user pause request at step {state.step}! Saving checkpoint and exporting current weights...', flush=True)\n"
                    "                self.checkpoints.save(state.step, self.parts(state))\n"
                    "                break"
                )
                if target_line in content:
                    patched = content.replace(target_line, pause_hook, 1)
                    native_driver_file.write_text(patched, encoding="utf-8")
                    print(" [OK] Injected graceful Pause & Save hook into drivers/native.py.")
        except Exception as e:
            print(f" Warning patching native.py: {e}")


def inspect_safetensors_lora(sf_path: Path) -> dict:
    """秒级解析 .safetensors 文件头，提取真实 Rank(Dim)、Alpha、算法、DoRA、步数与键数量（零显存占用）。"""
    info = {
        "rank": None,
        "alpha": None,
        "step": None,
        "algo": "lora",
        "dora": False,
        "preset": None,
        "keys_count": 0,
        "path": str(sf_path),
    }
    try:
        import struct
        with open(sf_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))
        meta = header.get("__metadata__", {}) or {}
        tensor_keys = [k for k in header.keys() if k != "__metadata__"]
        info["keys_count"] = len(tensor_keys)

        # 1. 从 metadata 读取
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
                if spec.get("preset") and isinstance(spec["preset"], str):
                    info["preset"] = spec["preset"]
            except Exception:
                pass

        # 2. 从张量命名与形状兜底推断算法、DoRA 与真实 Rank
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
                    shape = header[k].get("shape", [])
                    if len(shape) >= 2:
                        info["rank"] = int(shape[0])
                        break
        if info["alpha"] is None and info["rank"] is not None:
            info["alpha"] = float(info["rank"])
    except Exception as e:
        print(f" Warning inspecting safetensors {sf_path}: {e}")
    return info


def list_lora_checkpoints(trainer_dir: Path, lora_name: str, custom_dir: str = "") -> list:
    """
    检索指定 LoRA 的所有完整断点目录 (同时具备 state.json 与 trainable.safetensors)。
    返回按步数升序排列的 [(step_num, ckpt_path), ...] 列表。
    """
    candidates = []
    search_roots = []
    if custom_dir and os.path.exists(custom_dir):
        cpath = Path(custom_dir).resolve()
        if (cpath / "state.json").exists() and (cpath / "trainable.safetensors").exists():
            try:
                st = json.loads((cpath / "state.json").read_text(encoding="utf-8"))
                return [(int(st.get("step", 0)), cpath)]
            except Exception:
                return [(0, cpath)]
        search_roots.append(cpath)

    default_ckpt_root = trainer_dir / "runs" / "moss" / lora_name / "checkpoints"
    if default_ckpt_root.exists() and default_ckpt_root not in search_roots:
        search_roots.append(default_ckpt_root)

    for root in search_roots:
        for p in root.glob("step_*"):
            if p.is_dir() and (p / "state.json").exists() and (p / "trainable.safetensors").exists():
                try:
                    st = json.loads((p / "state.json").read_text(encoding="utf-8"))
                    step_val = int(st.get("step", 0))
                except Exception:
                    m = re.search(r"step_(\d+)", p.name)
                    step_val = int(m.group(1)) if m else 0
                candidates.append((step_val, p))

    candidates.sort(key=lambda x: x[0])
    return candidates


def prepare_init_from_checkpoint(source_path_str: str, workspace_dir: Path, lora_name: str, base_output_dir: Path, trainer_dir: Path) -> tuple:
    """
    将已训练结束的 LoRA (.safetensors 单文件或目录) 包装为 KohakuMossTrainer INIT_FROM 所需的目录结构。
    返回 (staged_init_dir, lora_info_dict)。
    """
    chosen_sf = None
    if source_path_str and os.path.exists(source_path_str):
        sp = Path(source_path_str).resolve()
        if sp.is_file() and sp.suffix.lower() == ".safetensors":
            chosen_sf = sp
        elif sp.is_dir():
            if (sp / "trainable.safetensors").exists():
                chosen_sf = sp / "trainable.safetensors"
            else:
                sfs = sorted(sp.glob("*.safetensors"), key=lambda x: x.stat().st_mtime, reverse=True)
                if sfs:
                    chosen_sf = sfs[0]
    else:
        # 自动从 output/<lora_name> 或 runs/moss/<lora_name> 检索已训好的最新权重
        auto_candidates = []
        out_folder = base_output_dir / lora_name
        if out_folder.exists():
            auto_candidates.extend(out_folder.glob("*.safetensors"))
        run_folder = trainer_dir / "runs" / "moss" / lora_name
        if run_folder.exists():
            auto_candidates.extend(run_folder.rglob("*.safetensors"))
        auto_candidates = [
            f for f in auto_candidates
            if "moss_image" not in f.name.lower() and "qwen" not in f.name.lower() and "flux2" not in f.name.lower()
        ]
        if auto_candidates:
            auto_candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            chosen_sf = auto_candidates[0]

    if not chosen_sf or not chosen_sf.exists():
        raise FileNotFoundError(
            f"未找到可用于继续训练 (init_from) 的已有 LoRA 权重文件！请确认路径或先选择一个已训练的 .safetensors 文件。"
        )

    stage_dir = workspace_dir / "init_from_stage" / lora_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    target_trainable = stage_dir / "trainable.safetensors"
    if target_trainable.exists():
        target_trainable.unlink()
    shutil.copy2(str(chosen_sf), str(target_trainable))

    info = inspect_safetensors_lora(chosen_sf)
    info["source_file"] = str(chosen_sf)
    return stage_dir, info


def export_and_sync_weights(
    trainer_dir: Path,
    lora_name: str,
    lora_dedicated_folder: Path,
    comfy_models_dir: str = None,
    copy_to_comfy: bool = False,
    quiet: bool = False
) -> tuple:
    """
    智能扫描当前 LoRA 运行目录下的所有阶段性与最终权重：
    1. 将每个阶段 checkpoint (step_XXXXXXXX) 导出的权重归档至 output/<lora_name>/checkpoints/ 下留存；
    2. 严格按步数与修改时间顺序，将最新一步的权重同步至 output/<lora_name>/<lora_name>-comfy.safetensors；
    3. 若勾选了同步至 ComfyUI，实时更新 ComfyUI models/loras/<lora_name>/。
    返回 (exported_main_files, latest_step_num)。
    """
    run_dir = trainer_dir / "runs" / "moss" / lora_name
    if not run_dir.exists():
        return [], 0

    lora_dedicated_folder.mkdir(parents=True, exist_ok=True)
    ckpt_archive_dir = lora_dedicated_folder / "checkpoints"

    # 1. 收集所有阶段性 checkpoints 并按步数排序
    step_records = []  # (step_num, mtime, comfy_file, lycoris_file, trainable_file)
    ckpts_root = run_dir / "checkpoints"
    if ckpts_root.exists():
        for step_dir in ckpts_root.glob("step_*"):
            if not step_dir.is_dir() or step_dir.name.endswith(".partial"):
                continue
            step_num = 0
            st_file = step_dir / "state.json"
            if st_file.exists():
                try:
                    step_num = int(json.loads(st_file.read_text(encoding="utf-8")).get("step", 0))
                except Exception:
                    pass
            if step_num == 0:
                m = re.search(r"step_(\d+)", step_dir.name)
                step_num = int(m.group(1)) if m else 0

            exp_dir = step_dir / "export"
            comfy_f = None
            lyco_f = None
            if exp_dir.exists():
                for sf in exp_dir.glob("*.safetensors"):
                    if "-comfy" in sf.name.lower() or sf.stem.endswith("_comfy"):
                        comfy_f = sf
                    elif "-lycoris" in sf.name.lower() or sf.stem.endswith("_lycoris"):
                        lyco_f = sf
            train_f = step_dir / "trainable.safetensors"
            if not train_f.exists():
                train_f = None

            if comfy_f or lyco_f or train_f:
                ref_f = comfy_f or lyco_f or train_f
                step_records.append((step_num, ref_f.stat().st_mtime, comfy_f, lyco_f, train_f))

                # 同步阶段性权重到 output/<lora_name>/checkpoints/
                ckpt_archive_dir.mkdir(parents=True, exist_ok=True)
                if comfy_f:
                    step_dst = ckpt_archive_dir / f"{lora_name}_step{step_num:06d}-comfy.safetensors"
                    if not step_dst.exists() or comfy_f.stat().st_mtime > step_dst.stat().st_mtime + 1:
                        shutil.copy2(str(comfy_f), str(step_dst))
                if lyco_f:
                    step_dst_ly = ckpt_archive_dir / f"{lora_name}_step{step_num:06d}-lycoris.safetensors"
                    if not step_dst_ly.exists() or lyco_f.stat().st_mtime > step_dst_ly.stat().st_mtime + 1:
                        shutil.copy2(str(lyco_f), str(step_dst_ly))

    # 2. 检查顶层最终 export/ 目录
    final_exp_dir = run_dir / "export"
    if final_exp_dir.exists():
        comfy_f = None
        lyco_f = None
        step_num = 0
        for sf in final_exp_dir.glob("*.safetensors"):
            info = inspect_safetensors_lora(sf)
            if info.get("step"):
                step_num = max(step_num, int(info["step"]))
            if "-comfy" in sf.name.lower() or sf.stem.endswith("_comfy"):
                comfy_f = sf
            elif "-lycoris" in sf.name.lower() or sf.stem.endswith("_lycoris"):
                lyco_f = sf
        if comfy_f or lyco_f:
            ref_f = comfy_f or lyco_f
            step_records.append((step_num, ref_f.stat().st_mtime, comfy_f, lyco_f, None))

    if not step_records:
        return [], 0

    # 按 (step_num, mtime) 升序排列，取最后一个即为最新最完整的版本
    step_records.sort(key=lambda x: (x[0], x[1]))
    latest_step, _, latest_comfy, latest_lyco, latest_trainable = step_records[-1]

    # 如果顶层 export 没有 trainable_f，从最后一个含 trainable 的记录中取
    if latest_trainable is None:
        for rec in reversed(step_records):
            if rec[4] is not None:
                latest_trainable = rec[4]
                break

    exported_files = []
    for src_f, suffix in [
        (latest_comfy, f"{lora_name}-comfy.safetensors"),
        (latest_lyco, f"{lora_name}-lycoris.safetensors"),
        (latest_trainable, f"{lora_name}-trainable.safetensors")
    ]:
        if src_f and src_f.exists():
            dst = lora_dedicated_folder / suffix
            if not dst.exists() or src_f.stat().st_mtime > dst.stat().st_mtime + 0.5 or src_f.stat().st_size != dst.stat().st_size:
                shutil.copy2(str(src_f), str(dst))
                if not quiet:
                    print(f" [Exported Step {latest_step}] {src_f.name} -> {dst}")
            exported_files.append(dst)

    if copy_to_comfy and comfy_models_dir and os.path.exists(comfy_models_dir):
        comfy_loras_dir = Path(comfy_models_dir) / "loras"
        if comfy_loras_dir.exists():
            target_comfy_sub = comfy_loras_dir / lora_name
            target_comfy_sub.mkdir(parents=True, exist_ok=True)
            for ef in exported_files:
                if ef.name.endswith("-trainable.safetensors"):
                    continue
                target_comfy_dst = target_comfy_sub / ef.name
                if not target_comfy_dst.exists() or ef.stat().st_mtime > target_comfy_dst.stat().st_mtime + 0.5:
                    shutil.copy2(str(ef), str(target_comfy_dst))
                    if not quiet:
                        print(f" [ComfyUI Synced] {ef.name} -> {target_comfy_dst}")

    return exported_files, latest_step

def prepare_dataset_structure(source_data_dir: Path, workspace_data_dir: Path, repeats: int, lora_name: str, trigger_word: str = "") -> Path:
    """
    将散装图集自动标准化为 Kohya 规范的 <repeats>_<name> 结构。
    支持自定义触发词自动注入：若提供了 trigger_word 且原标注首部未包含，则自动前置注入。
    严格执行孤儿文件清理与缓存一致性维护，确保工作区图集与源目录 1:1 严格对齐。
    """
    target_folder = workspace_data_dir / f"{repeats}_{lora_name}"
    target_folder.mkdir(parents=True, exist_ok=True)
    
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    raw_images = [f for f in source_data_dir.iterdir() if f.suffix.lower() in img_exts]
    
    print(f"Source Dataset: Found {len(raw_images)} images in {source_data_dir}")
    if len(raw_images) == 0:
        raise ValueError(f"No image files found in {source_data_dir}")

    # 构建当前有效图片与标注文件名称集合
    valid_img_names = {img.name for img in raw_images}
    valid_txt_names = {f"{img.stem}.txt" for img in raw_images}

    # 1. 扫描并彻底清理 target_folder 内历史遗留的孤儿文件（旧图与旧标注）
    orphan_purged = 0
    for existing_file in list(target_folder.iterdir()):
        if existing_file.is_file():
            if existing_file.name not in valid_img_names and existing_file.name not in valid_txt_names:
                try:
                    existing_file.unlink()
                    orphan_purged += 1
                except Exception as e:
                    print(f" Warning deleting orphan file {existing_file.name}: {e}")

    # 2. 清理 workspace_data_dir 下其他非当前 target_folder 的陈旧数据集目录
    for sub_dir in list(workspace_data_dir.iterdir()):
        if sub_dir.is_dir() and sub_dir.name != "cache" and sub_dir.name != target_folder.name:
            try:
                shutil.rmtree(str(sub_dir))
                print(f" [Cleanup] Removed obsolete dataset directory from previous runs: {sub_dir.name}")
                orphan_purged += 1
            except Exception as e:
                print(f" Warning cleaning obsolete dir {sub_dir.name}: {e}")

    # 3. 若检测到旧数据残留被清理，同步重置 cache 目录以防脏缓存污染
    if orphan_purged > 0:
        cache_dir = workspace_data_dir / "cache"
        if cache_dir.exists():
            try:
                shutil.rmtree(str(cache_dir))
            except Exception as e:
                print(f" Warning resetting dataset cache: {e}")
        print(f" [Cleanup] Purged {orphan_purged} orphan files from previous runs to align 1:1 with active dataset")

    trigger_clean = trigger_word.strip() if trigger_word else ""
    if trigger_clean:
        print(f" [Trigger Injection] Active Trigger Word: '{trigger_clean}' (will be prepended to captions)")

    linked_count = 0
    replaced_imgs = 0
    for img in raw_images:
        dest_img = target_folder / img.name
        dest_txt = target_folder / f"{img.stem}.txt"
        src_txt = source_data_dir / f"{img.stem}.txt"

        # 4. 复制或硬链接图像文件（同时检测同名替换的新图片，避免旧图残留）
        need_update_img = not dest_img.exists()
        if not need_update_img:
            try:
                src_st = img.stat()
                dst_st = dest_img.stat()
                if src_st.st_size != dst_st.st_size or src_st.st_mtime > dst_st.st_mtime + 1.0:
                    need_update_img = True
                    dest_img.unlink()
                    replaced_imgs += 1
            except Exception:
                pass
        if need_update_img:
            try:
                os.link(str(img), str(dest_img))
            except Exception:
                shutil.copy2(str(img), str(dest_img))

        # 5. 处理自然语言标注文本与触发词自动注入
        caption_content = ""
        if src_txt.exists():
            caption_content = src_txt.read_text(encoding="utf-8", errors="ignore").strip()

        if trigger_clean:
            # 检查原标注首部是否已经包含该触发词，避免重复堆叠
            if not caption_content.lower().startswith(trigger_clean.lower()):
                caption_content = f"{trigger_clean}, {caption_content}" if caption_content else trigger_clean

        if not dest_txt.exists() or dest_txt.read_text(encoding="utf-8", errors="ignore") != caption_content:
            dest_txt.write_text(caption_content, encoding="utf-8")
        linked_count += 1

    print(f" [OK] Structured dataset directory ready: {target_folder} ({linked_count} image-text pairs)")
    return workspace_data_dir


def sanitize_subset_folder_name(raw_name: str, fallback_idx: int) -> str:
    """将子集名清洗为安全的文件系统目录名（保留字母/数字/下划线/短横线，避免特殊字符触发路径异常）。"""
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", (raw_name or "").strip()).strip("_")
    return cleaned if cleaned else f"subset_{fallback_idx}"


def load_subsets_spec(subsets_arg: str) -> list:
    """解析 --subsets_json 参数（支持 JSON 文件路径或直接内联 JSON 字符串）。"""
    if not subsets_arg or not subsets_arg.strip():
        return []
    raw = subsets_arg.strip()
    if os.path.isfile(raw):
        data = json.loads(Path(raw).read_text(encoding="utf-8"))
    else:
        data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("--subsets_json must resolve to a JSON array of subset objects.")
    return data


def prepare_multi_subset_dataset(subsets: list, workspace_data_dir: Path) -> tuple:
    """
    多子集配比数据集构建器（适用于多人物、多画风、多概念 LoRA）：
    1. 为每个启用的子集创建独立的 Kohya 规范目录 <repeats>_<subset_name>；
    2. 每个子集独立同步图片并前置注入专属触发词，互不污染；
    3. 严格清理陈旧目录与孤儿文件，必要时重置缓存，并打印多子集曝光配比总表。
    返回 (workspace_data_dir, subset_summaries_list)。
    """
    workspace_data_dir.mkdir(parents=True, exist_ok=True)
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}

    active_specs = []
    used_folder_names = set()
    for idx, item in enumerate(subsets, start=1):
        if not isinstance(item, dict):
            continue
        if item.get("enabled", True) is False:
            continue
        src_str = str(item.get("path", "")).strip()
        if not src_str or not os.path.isdir(src_str):
            raise FileNotFoundError(f"Subset #{idx} directory not found: {src_str}")
        rep = max(1, int(item.get("repeats", 1)))
        base_name = sanitize_subset_folder_name(item.get("name") or Path(src_str).name, idx)
        folder_name = f"{rep}_{base_name}"
        suffix_n = 2
        while folder_name.lower() in used_folder_names:
            folder_name = f"{rep}_{base_name}_{suffix_n}"
            suffix_n += 1
        used_folder_names.add(folder_name.lower())
        active_specs.append({
            "name": item.get("name") or Path(src_str).name,
            "folder_name": folder_name,
            "source_dir": Path(src_str).resolve(),
            "repeats": rep,
            "trigger": str(item.get("trigger", "") or "").strip(),
        })

    if not active_specs:
        raise ValueError("Multi-subset mode enabled, but no valid active subsets were provided!")

    active_folder_names = {s["folder_name"] for s in active_specs}
    orphan_purged = 0

    # 1. 清理不属于当前活跃子集的旧目录
    for sub_dir in list(workspace_data_dir.iterdir()):
        if sub_dir.is_dir() and sub_dir.name != "cache" and sub_dir.name not in active_folder_names:
            try:
                shutil.rmtree(str(sub_dir))
                print(f" [Cleanup] Removed obsolete subset directory: {sub_dir.name}")
                orphan_purged += 1
            except Exception as e:
                print(f" Warning cleaning obsolete subset dir {sub_dir.name}: {e}")

    subset_summaries = []
    total_epoch_exposures = 0

    for spec in active_specs:
        target_folder = workspace_data_dir / spec["folder_name"]
        target_folder.mkdir(parents=True, exist_ok=True)
        src_dir = spec["source_dir"]
        raw_images = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts]
        if not raw_images:
            raise ValueError(f"No image files found in subset '{spec['name']}' ({src_dir})")

        valid_img_names = {img.name for img in raw_images}
        valid_txt_names = {f"{img.stem}.txt" for img in raw_images}

        # 清理当前子集目录孤儿文件
        for existing_file in list(target_folder.iterdir()):
            if existing_file.is_file() and existing_file.name not in valid_img_names and existing_file.name not in valid_txt_names:
                try:
                    existing_file.unlink()
                    orphan_purged += 1
                except Exception:
                    pass

        trigger_clean = spec["trigger"]
        for img in raw_images:
            dest_img = target_folder / img.name
            dest_txt = target_folder / f"{img.stem}.txt"
            src_txt = src_dir / f"{img.stem}.txt"

            need_update_img = not dest_img.exists()
            if not need_update_img:
                try:
                    src_st = img.stat()
                    dst_st = dest_img.stat()
                    if src_st.st_size != dst_st.st_size or src_st.st_mtime > dst_st.st_mtime + 1.0:
                        need_update_img = True
                        dest_img.unlink()
                except Exception:
                    pass
            if need_update_img:
                try:
                    os.link(str(img), str(dest_img))
                except Exception:
                    shutil.copy2(str(img), str(dest_img))

            caption_content = ""
            if src_txt.exists():
                caption_content = src_txt.read_text(encoding="utf-8", errors="ignore").strip()
            if trigger_clean and not caption_content.lower().startswith(trigger_clean.lower()):
                caption_content = f"{trigger_clean}, {caption_content}" if caption_content else trigger_clean
            if not dest_txt.exists() or dest_txt.read_text(encoding="utf-8", errors="ignore") != caption_content:
                dest_txt.write_text(caption_content, encoding="utf-8")

        epoch_exp = len(raw_images) * spec["repeats"]
        total_epoch_exposures += epoch_exp
        subset_summaries.append({
            "name": spec["name"],
            "folder": spec["folder_name"],
            "source_dir": str(src_dir),
            "images": len(raw_images),
            "repeats": spec["repeats"],
            "epoch_exposures": epoch_exp,
            "trigger": trigger_clean if trigger_clean else "(None)",
        })

    if orphan_purged > 0:
        cache_dir = workspace_data_dir / "cache"
        if cache_dir.exists():
            try:
                shutil.rmtree(str(cache_dir))
            except Exception:
                pass
        print(f" [Cleanup] Purged {orphan_purged} stale entries and reset dataset cache for consistency.")

    print(f" [Multi-Subset Summary] Active Subsets: {len(subset_summaries)} | Total Exposures/Epoch: {total_epoch_exposures}")
    for s in subset_summaries:
        pct = (s["epoch_exposures"] / total_epoch_exposures * 100.0) if total_epoch_exposures > 0 else 0.0
        s["share_percent"] = round(pct, 1)
        print(
            f"   - [{s['folder']}] {s['images']} imgs x {s['repeats']} repeats = {s['epoch_exposures']} exp/epoch "
            f"({pct:.1f}%) | Trigger: {s['trigger']}"
        )

    return workspace_data_dir, subset_summaries

def main():
    parser = argparse.ArgumentParser(description="OpenMOSS MOSS-Image Local Training Engine for Windows Consumer GPUs")
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("--data_dir", type=str, default="./dataset",
                        help="Path to folder containing source images and .txt captions")
    parser.add_argument("--subsets_json", type=str, default="",
                        help="Optional JSON file path or JSON array string for multi-subset training (multi-character / multi-style / concept balancing)")
    parser.add_argument("--comfy_models_dir", type=str, default=None,
                        help="Path to ComfyUI models directory containing diffusion_models, text_encoders, and vae")
    parser.add_argument("--transformer_path", type=str, default="",
                        help="Explicit path to MOSS-Image DiT transformer weights (.safetensors)")
    parser.add_argument("--text_encoder_path", type=str, default="",
                        help="Explicit path to Qwen3.5-9B text encoder weights (.safetensors)")
    parser.add_argument("--vae_path", type=str, default="",
                        help="Explicit path to FLUX.2 VAE weights (.safetensors)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Root directory to save final LoRA folders and checkpoints")
    parser.add_argument("--lora_name", "--run_name", type=str, default="my_lora", dest="lora_name",
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
    parser.add_argument("--algo", type=str, default="lora", choices=["lora", "loha", "lokr"],
                        help="Adapter algorithm: lora (default), loha (Hadamard), lokr (Kronecker)")
    parser.add_argument("--dora", action="store_true",
                        help="Enable DoRA (Weight-Decomposed Low-Rank Adaptation, ICML 2024) for improved detail and style fidelity")
    parser.add_argument("--target_preset", type=str, default="attn-mlp",
                        choices=["attn", "attn-mlp", "attn-mlp+modulation", "attn-mlp+text-refiner", "full-block", "all"],
                        help="Target transformer layers preset for adapter injection (default: attn-mlp)")
    parser.add_argument("--loraplus_ratio", type=float, default=1.0,
                        help="LoRA+ learning rate multiplier for up-projection weights (1.0=off, recommended 2.0~4.0 for faster multi-concept convergence)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16"],
                        help="Training precision (bf16 recommended for RTX 40/30 series)")
    parser.add_argument("--vram_preset", type=str, default="auto", choices=["auto", "6g", "8g", "12g", "16g"],
                        help="VRAM optimization preset (auto, 6g, 8g, 12g, 16g)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Target image resolution area for training (e.g. 1024, 768, 512)")
    parser.add_argument("--variant", type=str, default="auto", choices=["auto", "base", "sft"],
                        help="Model architecture variant ('auto', 'base', 'sft'). Auto-detected from weights filename by default.")
    parser.add_argument("--train_mode", type=str, default="fresh", choices=["fresh", "resume", "init_from"],
                        help="Training mode: 'fresh' (from scratch), 'resume' (continue from checkpoint), 'init_from' (continue training from an existing .safetensors LoRA)")
    parser.add_argument("--resume_from", type=str, default="latest",
                        help="Checkpoint directory path to resume from, or 'latest' (used when --train_mode=resume)")
    parser.add_argument("--init_from", type=str, default="",
                        help="Path to an existing .safetensors LoRA file or folder to continue training from (used when --train_mode=init_from)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save an intermediate checkpoint every N steps (default: 100)")
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

    print_banner(f"OpenMOSS MOSS-Image Local LoRA Training Pipeline v{APP_VERSION}: {args.lora_name}")
    print(f"Target LoRA Name & Subfolder: {args.lora_name}")
    print(f"Custom Trigger Word:          {args.trigger_word if args.trigger_word else '(None, using original captions)'}")
    print(f"Dedicated Output Directory:   {lora_dedicated_folder}")
    print(f"OS: Windows | Target Precision: {args.precision.upper()} | Text Device: {args.text_device.upper()}")
    print(f"Adapter Architecture: Algo={args.algo.upper()}, DoRA={'ON' if args.dora else 'OFF'}, Preset={args.target_preset}")
    print(f"Hyperparameters: Steps={args.steps}, Dim={args.rank}, Alpha={args.alpha}, LR={args.lr}")

    # 1. 硬件与 PyTorch 环境预检
    try:
        import torch
        import torchvision
        print(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            bf16_ok = torch.cuda.is_bf16_supported()
            print(f"GPU: {gpu_name} | VRAM: {total_vram:.2f} GB | BF16 Supported: {bf16_ok}")
            if args.precision == "bf16" and not bf16_ok:
                print(">>> NOTICE: Current GPU does not support native BF16 (e.g. Turing/Pascal). Falling back to FP16 automatically!")
                args.precision = "fp16"
            if total_vram < 10.0 and args.text_device == "cuda":
                print(">>> NOTICE: GPU VRAM is under 10GB. Switching text_device to CPU to prevent OOM!")
                args.text_device = "cpu"
        else:
            print("ERROR: CUDA device not detected! Exiting...")
            sys.exit(1)
    except Exception as e:
        print(f"Error checking PyTorch/CUDA environment: {e}")
        sys.exit(1)

    # 2. 定位或自选模型权重 (支持用户点选单文件与自动探测混合模式)
    print_banner("Locating Base Models and Weights")
    weights_info = {"transformer": None, "text_encoder": None, "vae": None}
    auto_detected = None

    # (1) DiT Transformer 判定
    if args.transformer_path and os.path.exists(args.transformer_path):
        weights_info["transformer"] = Path(args.transformer_path).resolve()
        print(f" [User Specified] Transformer:  {weights_info['transformer']}")
    else:
        auto_detected = probe_local_models(args.comfy_models_dir)
        weights_info["transformer"] = auto_detected["transformer"]
        print(f" [Auto Probed]    Transformer:  {weights_info['transformer']}")

    # (2) Text Encoder 判定
    if args.text_encoder_path and os.path.exists(args.text_encoder_path):
        weights_info["text_encoder"] = Path(args.text_encoder_path).resolve()
        print(f" [User Specified] Text Encoder: {weights_info['text_encoder']}")
    else:
        if not auto_detected:
            auto_detected = probe_local_models(args.comfy_models_dir)
        weights_info["text_encoder"] = auto_detected["text_encoder"]
        print(f" [Auto Probed]    Text Encoder: {weights_info['text_encoder']}")

    # (3) VAE 判定
    if args.vae_path and os.path.exists(args.vae_path):
        weights_info["vae"] = Path(args.vae_path).resolve()
        print(f" [User Specified] VAE:          {weights_info['vae']}")
    else:
        if not auto_detected:
            auto_detected = probe_local_models(args.comfy_models_dir)
        weights_info["vae"] = auto_detected["vae"]
        print(f" [Auto Probed]    VAE:          {weights_info['vae']}")

    if not weights_info["transformer"] or not weights_info["text_encoder"] or not weights_info["vae"]:
        raise FileNotFoundError(
            "Missing core weights! Ensure MOSS-Image DiT, Qwen3.5-9B, and FLUX.2 VAE exist or specify their paths explicitly."
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

    # 3. 准备 Kohya 结构化数据集并注入自定义触发词（支持单图集与多子集独立配比）
    workspace_data_root = workspace_dir / "dataset"
    multi_subset_summaries = []
    parsed_subsets = load_subsets_spec(args.subsets_json) if args.subsets_json else []
    if parsed_subsets:
        print_banner(f"Preparing Multi-Subset Dataset ({len(parsed_subsets)} configured subsets)")
        _, multi_subset_summaries = prepare_multi_subset_dataset(parsed_subsets, workspace_data_root)
    else:
        print_banner("Preparing Dataset Structure with Custom Trigger Word")
        src_data_path = Path(args.data_dir).resolve()
        prepare_dataset_structure(src_data_path, workspace_data_root, args.repeats, args.lora_name, args.trigger_word)

    # 4. 检查并准备 KohakuMossTrainer 训练器
    trainer_dir = project_root / "KohakuMossTrainer"
    if not trainer_dir.exists():
        print_banner("Cloning KohakuMossTrainer Framework")
        subprocess.run(["git", "clone", "https://github.com/KohakuBlueleaf/KohakuMossTrainer.git", str(trainer_dir)], creationflags=_NO_WINDOW, check=True)
    
    apply_compatibility_patches(trainer_dir)

    # 确保本地依赖与可编辑模块就绪
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", str(trainer_dir)
    ], creationflags=_NO_WINDOW, check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "kohaku-engine", "kohakuvault", "lycoris-lora", "anyschedule", "safetensors"
    ], creationflags=_NO_WINDOW, check=True)

    # 5. 生成本地训练与离线缓存专属配置文件 (名称与 lora_name 强对齐)
    print_banner(f"Generating Local Configurations for '{args.lora_name}'")
    user_cfg_dir = trainer_dir / "configs" / "user"
    user_cfg_dir.mkdir(parents=True, exist_ok=True)

    sanitized_data_root = sanitize_path(workspace_data_root)
    sanitized_weights_dir = sanitize_path(models_stage_dir)
    transformer_filename = target_transformer.name

    # 动态检测当前硬件 VRAM，精准匹配显存优化层
    vram_gb = 16.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass

    is_int8_transformer = "int8" in transformer_filename.lower()
    
    # 确定目标显存优化档位 (VRAM Preset)
    selected_preset = args.vram_preset.lower()
    if selected_preset == "auto":
        if vram_gb <= 10.0:
            effective_preset = "6g"
        elif vram_gb <= 14.0:
            effective_preset = "12g"
        else:
            effective_preset = "16g"
        auto_note = f"(Auto-detected {vram_gb:.1f} GB VRAM -> selected '{effective_preset}')"
    else:
        effective_preset = selected_preset
        auto_note = f"(Manually specified: '{effective_preset}')"

    if effective_preset == "6g":
        # 针对 6G/8G 消费级显卡（如 RTX 3070 / 4060 等），应用官方极限 4.4G 压制配方：两层流式环形缓冲 + 激活与梯度全面卸载 + CPU 优化器
        recipe_import = 'use_config("../moss/_base/base_lora.py")\nuse_config("../moss/vram/vram_6g.py")'
        max_tokens_val = 4352
        memory_profile_str = 'MEMORY_PROFILE = "ckpt-ring2-actoff-optcpu-gradoff"'
        print(f" [VRAM Preset] Applied 6G Profile (extreme ~4.4GB VRAM): {auto_note}")
    elif effective_preset == "8g":
        # 针对 8G 显卡标准配置，minimal profile (两层流式环形缓冲 + 激活梯度卸载 + host AdamW)
        recipe_import = 'use_config("../moss/_base/base_lora.py")\nuse_config("../moss/vram/vram_8g.py")'
        max_tokens_val = 4352
        memory_profile_str = 'MEMORY_PROFILE = "minimal"'
        print(f" [VRAM Preset] Applied 8G Profile (~5.5GB VRAM): {auto_note}")
    elif effective_preset == "12g":
        # 针对 12G 显卡进阶配置，recipes_lora_12g_stream.py
        recipe_import = 'use_config("../moss/recipes/recipes_lora_12g_stream.py")'
        max_tokens_val = 8192
        memory_profile_str = 'MEMORY_PROFILE = "low"'
        print(f" [VRAM Preset] Applied 12G Profile (~8.5GB VRAM): {auto_note}")
    else:
        # 针对 16G/24G 显卡满血极速配置
        recipe_import = 'use_config("../moss/recipes/recipes_lora_16g_fp16.py")'
        max_tokens_val = 16384
        memory_profile_str = ""
        print(f" [VRAM Preset] Applied 16G Profile (full speed): {auto_note}")

    print(f" [Resolution] Target Area: {args.resolution}x{args.resolution} ({args.resolution * args.resolution} px)")

    # 确定模型架构变体 (Base 还是 SFT)
    if args.variant.lower() == "auto":
        is_sft_model = "sft" in transformer_filename.lower()
        model_variant = "sft" if is_sft_model else "base"
        variant_note = f"(Auto-detected from backbone: '{transformer_filename}')"
    else:
        model_variant = args.variant.lower()
        variant_note = "(Manually specified)"
    print(f" [Model Variant] Architecture: {model_variant.upper()} {variant_note}")

    base_prec_str = 'BASE_PRECISION = "int8"' if is_int8_transformer else ""

    # 5.5 处理训练模式: 全新训练 (fresh) / 断点续炼 (resume) / 基于已有 LoRA 继续训练 (init_from)
    train_mode = args.train_mode.lower()
    resume_cfg_str = ""
    init_from_cfg_str = ""
    effective_steps = args.steps
    resumed_step = 0
    resume_source_info = "(Fresh training from step 0)"

    run_ckpts_dir = trainer_dir / "runs" / "moss" / args.lora_name / "checkpoints"

    if train_mode == "resume":
        custom_resume = args.resume_from if (args.resume_from and args.resume_from.lower() not in ("auto", "latest")) else ""
        ckpts = list_lora_checkpoints(trainer_dir, args.lora_name, custom_resume)
        if ckpts:
            resumed_step, latest_ckpt_dir = ckpts[-1]
            # 自动探测断点中的真实 Rank、Alpha 与适配器架构，防止恢复时结构/维度冲突
            t_sf = latest_ckpt_dir / "trainable.safetensors"
            if t_sf.exists():
                l_info = inspect_safetensors_lora(t_sf)
                if l_info.get("rank") and l_info["rank"] != args.rank:
                    print(f" [Resume Auto-Align] Aligned Rank from {args.rank} -> {l_info['rank']} (matching checkpoint {latest_ckpt_dir.name})")
                    args.rank = l_info["rank"]
                    if l_info.get("alpha"):
                        args.alpha = l_info["alpha"]
                if l_info.get("algo") and l_info["algo"] in ("lora", "loha", "lokr") and l_info["algo"] != args.algo:
                    print(f" [Resume Auto-Align] Aligned Algo from {args.algo} -> {l_info['algo']}")
                    args.algo = l_info["algo"]
                if bool(l_info.get("dora", False)) != bool(args.dora):
                    print(f" [Resume Auto-Align] Aligned DoRA from {args.dora} -> {bool(l_info.get('dora', False))}")
                    args.dora = bool(l_info.get("dora", False))
                if l_info.get("preset") and l_info["preset"] != args.target_preset:
                    print(f" [Resume Auto-Align] Aligned Target Preset from {args.target_preset} -> {l_info['preset']}")
                    args.target_preset = l_info["preset"]
            resume_cfg_str = f'RESUME = "{sanitize_path(latest_ckpt_dir)}"'
            if args.steps <= resumed_step:
                effective_steps = resumed_step + args.steps
                print(
                    f" [Resume Auto-Extend] Checkpoint '{latest_ckpt_dir.name}' is already at step {resumed_step} (>= input steps {args.steps}). "
                    f"Treating {args.steps} as additional steps -> new target total STEPS = {effective_steps}!"
                )
            else:
                effective_steps = args.steps
                print(f" [Resume] Resuming from '{latest_ckpt_dir.name}' (step {resumed_step}) -> target step {effective_steps} (remaining: {effective_steps - resumed_step} steps).")
            resume_source_info = f"Resumed from {latest_ckpt_dir.name} (step {resumed_step} -> {effective_steps})"
        else:
            print(f" [Resume Fallback] No checkpoint directory found for '{args.lora_name}'. Checking for exported .safetensors to continue via INIT_FROM...")
            train_mode = "init_from"

    if train_mode == "init_from":
        # 清理或归档同名 run 下的旧 step_* 目录，确保 INIT_FROM 作为全新周期 (step 0 -> steps) 运行而不误触旧进度
        if run_ckpts_dir.exists():
            backup_root = trainer_dir / "runs" / "moss" / args.lora_name / f"checkpoints_bak_{int(time.time())}"
            try:
                shutil.move(str(run_ckpts_dir), str(backup_root))
                print(f" [Init-From] Archived previous checkpoint states to: {backup_root.name}")
            except Exception:
                pass
        staged_init_dir, l_info = prepare_init_from_checkpoint(
            args.init_from, workspace_dir, args.lora_name, base_output_dir, trainer_dir
        )
        if l_info.get("rank") and l_info["rank"] != args.rank:
            print(f" [Init-From Auto-Align] Aligned Rank from {args.rank} -> {l_info['rank']} (matching existing LoRA '{Path(l_info['source_file']).name}')")
            args.rank = l_info["rank"]
            if l_info.get("alpha"):
                args.alpha = l_info["alpha"]
        if l_info.get("algo") and l_info["algo"] in ("lora", "loha", "lokr") and l_info["algo"] != args.algo:
            print(f" [Init-From Auto-Align] Aligned Algo from {args.algo} -> {l_info['algo']}")
            args.algo = l_info["algo"]
        if bool(l_info.get("dora", False)) != bool(args.dora):
            print(f" [Init-From Auto-Align] Aligned DoRA from {args.dora} -> {bool(l_info.get('dora', False))}")
            args.dora = bool(l_info.get("dora", False))
        if l_info.get("preset") and l_info["preset"] != args.target_preset:
            print(f" [Init-From Auto-Align] Aligned Target Preset from {args.target_preset} -> {l_info['preset']}")
            args.target_preset = l_info["preset"]
        init_from_cfg_str = f'INIT_FROM = "{sanitize_path(staged_init_dir)}"'
        resume_source_info = f"Continued training (INIT_FROM) based on '{Path(l_info['source_file']).name}' (Rank={args.rank}, Alpha={args.alpha})"
        print(f" [Init-From] Loaded base LoRA weights from: {l_info['source_file']} (training {effective_steps} new steps)")

    elif train_mode == "fresh":
        # 全新训练：若同名 run_ckpts_dir 下有历史高步数断点，将其移入备份目录，防止新训练 latest() 指针混淆
        if run_ckpts_dir.exists() and any(run_ckpts_dir.glob("step_*")):
            backup_root = trainer_dir / "runs" / "moss" / args.lora_name / f"checkpoints_bak_{int(time.time())}"
            try:
                shutil.move(str(run_ckpts_dir), str(backup_root))
                print(f" [Fresh Start] Backed up previous run checkpoints to: {backup_root.name}")
            except Exception:
                pass

    data_cfg_code = f"""
data_root = "{sanitized_data_root}"
WEIGHTS = "{sanitized_weights_dir}"

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
VAE_ENCODE = {{"compile": "off"}}
"""

    warmup_steps = min(30, max(1, int(effective_steps * 0.05))) if effective_steps > 5 else 0
    save_every_val = max(0, int(args.save_every))
    is_healed_backbone = "heal" in transformer_filename.lower()
    fp16_fix_val = "online" if (args.precision == "fp16" and not is_healed_backbone) else "none"
    if fp16_fix_val == "online":
        print(f" [FP16 Guard] Non-heal backbone '{transformer_filename}' under FP16 detected -> enabled FP16_FIX='online' to prevent overflow!")
    fp16_fix_kwargs_str = 'FP16_FIX_KWARGS = {"c": 1 / 16}' if fp16_fix_val == "online" else "FP16_FIX_KWARGS = {}"

    effective_comfy_models = args.comfy_models_dir
    if not effective_comfy_models:
        broad_m, _ = find_broad_comfy_paths()
        if broad_m:
            effective_comfy_models = str(broad_m[0])

    # 按 KohakuMossTrainer 官方 algo_*.py 配方注入各算法防坑参数，确保 LyCORIS 融合内核与 ComfyUI 导出零冲突
    if args.algo == "loha":
        algo_kwargs_str = 'ALGO_KWARGS = {"use_tucker": False, "use_scalar": False, "rank_dropout_scale": False}'
    elif args.algo == "lokr":
        algo_kwargs_str = 'ALGO_KWARGS = {"factor": 8, "decompose_both": False, "full_matrix": False, "use_tucker": False, "rank_dropout_scale": False}'
    else:
        algo_kwargs_str = "ALGO_KWARGS = {}"
    dora_detach_val = True if args.dora else False
    loraplus_val = max(1.0, float(args.loraplus_ratio))

    train_cfg_code = f"""
from kohakuengine import use_config

{recipe_import}
use_config("{args.lora_name}_local_data.py")

VARIANT = "{model_variant}"
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
{fp16_fix_kwargs_str}
PRECISION = "{args.precision}"
ALGO = "{args.algo}"
{algo_kwargs_str}
DORA = {bool(args.dora)}
DORA_DETACH = {dora_detach_val}
PRESET = "{args.target_preset}"
LORAPLUS_RATIO = {loraplus_val}
DIM = {args.rank}
ALPHA = {args.alpha}
STEPS = {effective_steps}
LR = {args.lr}
SAVE_EVERY = {save_every_val}
SAVE_MINUTES = 20
KEEP_LAST = 5
SAVE_FORMAT = "both"
SAVE_FUSED_QKV = False
SCHEDULE = {{
    "mode": "cosine",
    "warmup": {warmup_steps},
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

    # 定位 kogine 可执行命令绝对路径，防御环境变量未刷新导致的 WinError 2
    kogine_cmd = "kogine"
    py_scripts_kogine = Path(sys.executable).parent / "Scripts" / "kogine.exe"
    if py_scripts_kogine.exists():
        kogine_cmd = str(py_scripts_kogine)

    # 6. 执行离线双特征提取 (双离线缓存)
    print_banner(f"Step 1/2: Extracting Text Embeddings (Device: {args.text_device.upper()})")
    subprocess.run([
        kogine_cmd, "run", "scripts/data/cache_text.py",
        "--config", f"configs/user/{args.lora_name}_local_data.py"
    ], cwd=str(trainer_dir), env=os.environ, creationflags=_NO_WINDOW, check=True)

    print_banner(f"Step 2/2: Extracting Image Latents (Device: {args.vae_device.upper()})")
    subprocess.run([
        kogine_cmd, "run", "scripts/data/cache_latents.py",
        "--config", f"configs/user/{args.lora_name}_local_data.py"
    ], cwd=str(trainer_dir), env=os.environ, creationflags=_NO_WINDOW, check=True)

    # 7. 物理级清洗配置文件中的临时离线编码器旋钮
    print_banner("Sanitizing Configuration Knobs for moss.py Validation")
    raw_cfg = data_cfg_file.read_text(encoding="utf-8")
    sanitized_cfg = re.sub(r'TEXT_ENCODER_KWARGS\s*=\s*\{[^\}]+\}', '', raw_cfg)
    sanitized_cfg = re.sub(r'IMAGE_ENCODER_KWARGS\s*=\s*\{[^\}]+\}', '', sanitized_cfg)
    data_cfg_file.write_text(sanitized_cfg.strip() + "\n", encoding="utf-8")
    print(" [OK] Knobs TEXT_ENCODER_KWARGS and IMAGE_ENCODER_KWARGS cleanly sanitized.")

    # 8. 配置优雅暂停信号文件 & 后台实时同步守护线程
    pause_flag_file = workspace_dir / ".pause_request"
    if pause_flag_file.exists():
        try:
            pause_flag_file.unlink()
        except Exception:
            pass
    os.environ["KOHAKUMOSS_PAUSE_FILE"] = str(pause_flag_file)

    import threading
    stop_sync_event = threading.Event()

    def _realtime_checkpoint_sync_loop():
        while not stop_sync_event.wait(10.0):
            try:
                export_and_sync_weights(
                    trainer_dir,
                    args.lora_name,
                    lora_dedicated_folder,
                    effective_comfy_models,
                    args.copy_to_comfy_loras,
                    quiet=True
                )
            except Exception:
                pass

    sync_thread = threading.Thread(target=_realtime_checkpoint_sync_loop, daemon=True)
    sync_thread.start()

    # 8.5 启动主训练循环
    print_banner(f"Starting Local LoRA Training: '{args.lora_name}' (Mode: {train_mode.upper()}, Target Steps: {effective_steps})")
    start_time = time.time()
    train_cmd = [
        kogine_cmd, "run", "scripts/train/moss.py",
        "--config", f"configs/user/{args.lora_name}_local.py"
    ]
    print(f"Executing: {' '.join(train_cmd)}")
    try:
        subprocess.run(train_cmd, cwd=str(trainer_dir), env=os.environ, creationflags=_NO_WINDOW, check=True)
    finally:
        stop_sync_event.set()

    elapsed = time.time() - start_time

    # 9. 智能检索并按最新步数同步导出的 LoRA 权重至专属子文件夹与 ComfyUI
    print_banner(f"Locating and Exporting Latest LoRA Weights to '{lora_dedicated_folder}'")
    exported_files, actual_final_step = export_and_sync_weights(
        trainer_dir,
        args.lora_name,
        lora_dedicated_folder,
        effective_comfy_models,
        args.copy_to_comfy_loras,
        quiet=False
    )

    actual_ran_steps = max(1, (actual_final_step - resumed_step) if actual_final_step > resumed_step else (effective_steps - resumed_step))
    print(f" [OK] Training session ran {actual_ran_steps} steps in {elapsed / 60:.2f} minutes (~{elapsed / actual_ran_steps:.2f} s/step)!")

    is_paused_early = (actual_final_step > 0 and actual_final_step < effective_steps)
    status_label = f"paused_at_step_{actual_final_step}" if is_paused_early else "completed"

    # 生成训练摘要与触发词元数据文件
    summary_file = lora_dedicated_folder / "training_summary.json"
    summary_data = {
        "lora_name": args.lora_name,
        "status": status_label,
        "train_mode": train_mode,
        "resume_info": resume_source_info,
        "completed_step": actual_final_step if actual_final_step > 0 else effective_steps,
        "target_steps": effective_steps,
        "trigger_word": args.trigger_word if args.trigger_word else "(None)",
        "output_folder": str(lora_dedicated_folder),
        "repeats": args.repeats,
        "epochs": args.epochs,
        "rank": args.rank,
        "alpha": args.alpha,
        "algo": args.algo,
        "dora": bool(args.dora),
        "dora_detach": dora_detach_val,
        "target_preset": args.target_preset,
        "loraplus_ratio": loraplus_val,
        "learning_rate": args.lr,
        "precision": args.precision,
        "vram_preset": effective_preset,
        "resolution": args.resolution,
        "model_variant": model_variant,
        "text_device": args.text_device,
        "dataset_mode": "multi_subset" if multi_subset_summaries else "single",
        "subsets": multi_subset_summaries,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comfyui_prompt_example": (
            f"A high quality digital painting of {args.trigger_word}, in the style of {args.lora_name}, masterpiece, cinematic lighting"
            if args.trigger_word else
            f"A high quality digital painting in the style of {args.lora_name}, masterpiece, cinematic lighting"
        )
    }
    summary_file.write_text(json.dumps(summary_data, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f" [OK] Saved training summary & checkpoint metadata to: {summary_file}")

    if is_paused_early:
        print_banner(f"Training Safely Paused & Saved at Step {actual_final_step} / {effective_steps}!")
        print(f"Current weights exported to: {lora_dedicated_folder}")
        print("You can test the current LoRA in ComfyUI now, or resume training anytime using '--train_mode resume'!")
    else:
        print_banner("Training Pipeline Completed Successfully!")
        print(f"Trained weights are available in: {lora_dedicated_folder}")

if __name__ == "__main__":
    main()
