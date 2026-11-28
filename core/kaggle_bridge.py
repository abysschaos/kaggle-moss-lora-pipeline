"""Zero-dependency Kaggle CLI & Cloud API bridge for headless dataset upload, kernel push, status polling, output sync, and offline bundle export."""

import datetime
import hashlib
import hmac
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import zipfile
from pathlib import Path

from .config_store import PROJECT_ROOT, SUPPORTED_IMAGE_EXTS, resolve_cli_python
from .trainer_runner import NO_WINDOW

KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"
KAGGLE_JSON_PATH = KAGGLE_CONFIG_DIR / "kaggle.json"
R2_CREDENTIALS_PATH = KAGGLE_CONFIG_DIR / "r2_credentials.json"
KAGGLE_ACCELERATORS = [
    ("NvidiaTeslaT4", "🔥 双卡 T4 DDP (推荐)"),
    ("default_gpu", "⚡ 单卡默认 GPU"),
    ("NvidiaTeslaP100", "🟦 兼容队列 (P100/替补)"),
]


def get_kaggle_json_path() -> Path:
    """返回当前生效的 kaggle.json 凭据路径，优先尊重 KAGGLE_CONFIG_DIR 环境变量。"""
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR", "").strip()
    if env_dir:
        return Path(env_dir) / "kaggle.json"
    return KAGGLE_JSON_PATH


def _build_kaggle_env(kaggle_json_path: Path = None) -> dict:
    """构造 UTF-8 安全且自动指向凭据目录的子进程环境变量。"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if kaggle_json_path is not None:
        env["KAGGLE_CONFIG_DIR"] = str(Path(kaggle_json_path).resolve().parent)
    elif KAGGLE_JSON_PATH.exists():
        env.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_CONFIG_DIR))
    return env


def _probe_cli_version(cmd: list, source: str) -> dict:
    """运行 <cmd> --version 核验 Kaggle CLI 是否真正可用。"""
    try:
        res = subprocess.run(
            [*cmd, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            env=_build_kaggle_env(),
            creationflags=NO_WINDOW,
        )
        out = (res.stdout or res.stderr or "").strip().splitlines()
        ver_text = out[0].strip() if out else ""
        if res.returncode == 0 and ver_text:
            return {
                "found": True,
                "cmd": list(cmd),
                "exe_path": cmd[0],
                "version": ver_text,
                "source": source,
            }
    except Exception:
        pass
    return {}


def find_kaggle_cli() -> dict:
    """
    动态检索本机可用 Kaggle CLI（零侵入 ComfyUI Python 环境）：
    按 PATH -> 当前 Python Scripts -> 项目 venv -> 系统 Python3xx Scripts -> python -m kaggle 顺序探测。
    """
    seen = set()
    candidates = []

    which_kaggle = shutil.which("kaggle")
    if which_kaggle:
        candidates.append(([which_kaggle], "系统 PATH"))

    scripts_name = "kaggle.exe" if os.name == "nt" else "kaggle"
    local_scripts = Path(sys.executable).parent / "Scripts" / scripts_name
    candidates.append(([str(local_scripts)], "当前环境 Scripts"))

    for vdir in ("venv", ".venv"):
        v_exe = PROJECT_ROOT / vdir / "Scripts" / scripts_name
        candidates.append(([str(v_exe)], f"项目 {vdir}"))

    if os.name == "nt":
        for root_env in ("LOCALAPPDATA", "APPDATA"):
            base = os.environ.get(root_env, "").strip()
            if not base:
                continue
            py_parent = Path(base) / ("Programs/Python" if root_env == "LOCALAPPDATA" else "Python")
            if py_parent.is_dir():
                for sc in sorted(py_parent.glob(f"Python*/Scripts/{scripts_name}"), reverse=True):
                    candidates.append(([str(sc)], f"用户 Python ({sc.parent.parent.name})"))

    for cmd, src in candidates:
        exe = Path(cmd[0])
        key = str(exe).lower()
        if key in seen or not exe.is_file():
            continue
        seen.add(key)
        info = _probe_cli_version(cmd, src)
        if info:
            return info

    if importlib.util.find_spec("kaggle") is not None:
        info = _probe_cli_version([resolve_cli_python(), "-m", "kaggle"], "当前 Python 模块")
        if info:
            return info

    return {"found": False, "cmd": [], "exe_path": "", "version": "", "source": ""}


def _resolve_pip_python_for_kaggle() -> list:
    """优先寻找系统独立 Python 安装 kaggle CLI，避免向 ComfyUI 便携环境写入额外依赖。"""
    comfy_dir = str(Path(sys.executable).resolve().parent).lower()
    if shutil.which("py"):
        return ["py", "-3"]
    sys_py = shutil.which("python")
    if sys_py and str(Path(sys_py).resolve().parent).lower() != comfy_dir:
        return [sys_py]
    return [resolve_cli_python()]


def _stream_subprocess(cmd: list, cwd: Path, on_log=None, env: dict = None) -> tuple:
    """运行子进程并逐行流式回调输出，返回 (returncode, combined_text)。"""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env or _build_kaggle_env(),
        creationflags=NO_WINDOW,
    )
    lines = []
    for line in iter(proc.stdout.readline, ""):
        lines.append(line)
        if on_log:
            on_log(line)
    proc.stdout.close()
    rc = proc.wait()
    return rc, "".join(lines)


def install_kaggle_cli(on_log=None) -> tuple:
    """自动通过系统 Python 用户目录（或兜底环境）安装官方 kaggle CLI。"""
    py_cmd = _resolve_pip_python_for_kaggle()
    pip_cmd = [*py_cmd, "-m", "pip", "install", "--user", "-U", "kaggle"]
    if on_log:
        on_log(f"[Kaggle CLI] 正在安装官方工具：{' '.join(pip_cmd)}\n")
    try:
        rc, out = _stream_subprocess(pip_cmd, PROJECT_ROOT, on_log=on_log)
        if rc != 0:
            return False, f"安装返回非零退出码 ({rc})，请检查网络或手动执行 pip install kaggle。"
        cli = find_kaggle_cli()
        if cli.get("found"):
            return True, f"安装成功！已探测到 {cli['version']} ({cli['source']})"
        return False, "pip 安装已完成但未能直接定位 kaggle.exe，请重启应用或将 Python Scripts 加入 PATH。"
    except Exception as err:
        return False, f"安装 Kaggle CLI 异常: {err}"


def mask_api_key(api_key: str) -> str:
    """脱敏显示 API Key，仅保留首尾少量字符用于状态核对。"""
    k = (api_key or "").strip()
    if not k:
        return ""
    if len(k) <= 8:
        return "*" * len(k)
    return f"{k[:4]}****{k[-4:]}"


def read_kaggle_credentials(kaggle_json_path: Path = None) -> dict:
    """安全读取 ~/.kaggle/kaggle.json 凭据状态（绝不返回明文 Key）。"""
    path = Path(kaggle_json_path) if kaggle_json_path else get_kaggle_json_path()
    result = {"exists": False, "username": "", "has_key": False, "key_masked": "", "path": str(path)}
    if not path.is_file():
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(data, dict):
            user = str(data.get("username", "") or "").strip()
            key = str(data.get("key", "") or "").strip()
            result["exists"] = True
            result["username"] = user
            result["has_key"] = bool(key)
            result["key_masked"] = mask_api_key(key)
    except Exception:
        result["exists"] = True
    return result


def save_kaggle_credentials(username: str, api_key: str, kaggle_json_path: Path = None) -> tuple:
    """将 Kaggle 用户名与 API Key 写入受保护的 kaggle.json（严禁存入 local_config.json）。"""
    user = (username or "").strip()
    key = (api_key or "").strip()
    if not user or not key:
        return False, "请同时填写有效的 Kaggle 用户名与 API Key。"

    path = Path(kaggle_json_path) if kaggle_json_path else get_kaggle_json_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"username": user, "key": key}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return True, f"凭证已安全保存至 {path}（账号: {user}）"
    except Exception as err:
        return False, f"写入凭证文件失败: {err}"


def import_kaggle_json(src_path: str, kaggle_json_path: Path = None) -> tuple:
    """从用户下载的 kaggle.json 导入并验证凭证字段，返回 (ok, username, message)。"""
    src = Path(src_path)
    if not src.is_file():
        return False, "", f"未找到所选文件：{src}"
    try:
        data = json.loads(src.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(data, dict):
            return False, "", "所选 JSON 文件格式无效（顶层应为对象）。"
        user = str(data.get("username", "") or "").strip()
        key = str(data.get("key", "") or "").strip()
        if not user or not key:
            return False, "", "kaggle.json 缺少 'username' 或 'key' 字段，请从 Kaggle Settings 重新生成。"
        ok, msg = save_kaggle_credentials(user, key, kaggle_json_path=kaggle_json_path)
        return ok, user if ok else "", msg
    except Exception as err:
        return False, "", f"解析 kaggle.json 失败: {err}"


def test_kaggle_connection(cli_info: dict = None, kaggle_json_path: Path = None) -> tuple:
    """校验本机 Kaggle CLI 与 API 凭证连通性，将认证/网络状态转为清晰的中文提示。"""
    cli = cli_info if (cli_info and cli_info.get("found")) else find_kaggle_cli()
    if not cli.get("found"):
        return False, "未检测到 Kaggle CLI，请点击【⬇️ 安装 CLI】完成部署。"

    creds = read_kaggle_credentials(kaggle_json_path=kaggle_json_path)
    if not creds.get("exists") or not creds.get("username") or not creds.get("has_key"):
        return False, f"尚未配置有效的 Kaggle 凭证（未发现 {creds['path']}），请先导入或保存 Token。"

    cmd = [*cli["cmd"], "kernels", "list", "--mine", "--page-size", "1"]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
            env=_build_kaggle_env(kaggle_json_path=kaggle_json_path),
            creationflags=NO_WINDOW,
        )
        combined = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
        lower = combined.lower()
        if res.returncode == 0:
            return True, f"✅ API 连接正常！已验证账号 [{creds['username']}]（{cli['version']}）"
        if "401" in lower or "403" in lower or "unauthorized" in lower:
            return False, f"🔴 Kaggle API 认证失败（401/403）：请检查用户名 [{creds['username']}] 与 API Key 是否匹配或已过期。"
        if "timeout" in lower or "connection" in lower or "ssl" in lower:
            return False, f"🟠 连接 kaggle.com 网络异常，请检查代理/网络后重试。\n详情: {combined[:240]}"
        return False, f"⚠️ Kaggle CLI 返回异常 (Exit={res.returncode}): {combined[:240]}"
    except subprocess.TimeoutExpired:
        return False, "🟠 检测 kaggle.com 超时（25 秒），请检查本机网络或代理连通性。"
    except Exception as err:
        return False, f"执行连通性检测异常: {err}"


def sanitize_kaggle_slug(raw: str, prefix: str = "moss") -> str:
    """规范化任意中英文名称为 Kaggle 合规 Slug（全小写字母/数字/连字符，长度 6~50）。"""
    clean_prefix = re.sub(r"[^a-z0-9]+", "-", (prefix or "moss").lower()).strip("-") or "moss"
    raw_str = (raw or "").strip()
    ascii_part = re.sub(r"[^a-z0-9]+", "-", raw_str.lower()).strip("-")
    if not ascii_part:
        short_h = hashlib.sha1(raw_str.encode("utf-8", errors="ignore")).hexdigest()[:6]
        ascii_part = f"{clean_prefix}-{short_h}"
    elif ascii_part != raw_str.lower() and len(ascii_part) < 6:
        short_h = hashlib.sha1(raw_str.encode("utf-8", errors="ignore")).hexdigest()[:4]
        ascii_part = f"{clean_prefix}-{ascii_part}-{short_h}"

    if len(ascii_part) < 6:
        slug = f"{clean_prefix}-{ascii_part}"
    else:
        slug = ascii_part

    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if len(slug) < 6:
        slug = f"{slug}-lora"[:50].strip("-")
    if len(slug) > 50:
        slug = slug[:50].rstrip("-")
        if len(slug) < 6:
            slug = f"{clean_prefix}-lora"
    return slug


def derive_interactive_kernel_slug(kernel_slug: str) -> str:
    """为 `--no-run` 交互式 Notebook 生成独立于 script 内核的合规 Slug（带 `-nb` 后缀，防止与既有 script 同名撞型）。"""
    base = sanitize_kaggle_slug(kernel_slug, prefix="moss-train")
    if base.endswith("-nb"):
        return base
    trimmed = base[:47].rstrip("-") or "moss-train"
    return sanitize_kaggle_slug(f"{trimmed}-nb", prefix="moss-train")


def normalize_hf_dataset_repo(raw: str) -> str:
    """将 Hugging Face Dataset 网址或 owner/repo 字符串规范化为标准 owner/repo 标识。"""
    text = (raw or "").strip()
    if not text:
        return ""
    text = re.sub(r"^https?://huggingface\.co/datasets/", "", text, flags=re.IGNORECASE).strip("/")
    for sep in ("/tree/", "/blob/", "/resolve/"):
        if sep in text:
            text = text.split(sep, 1)[0]
    parts = [p for p in text.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return text


def sanitize_external_dataset_url(raw_url: str) -> str:
    """剔除会触发 Kaggle 容器运行时网络风控（Canceled by backend.）的 x0.at 镜像并规范化 '|' 分隔直链。"""
    parts = [u.strip() for u in str(raw_url or "").split("|") if u.strip()]
    safe_parts = [u for u in parts if not re.match(r"^https?://(?:www\.)?x0\.at(?:/|$)", u, re.IGNORECASE)]
    return "|".join(safe_parts)


def build_cloud_config_payload(cleaned_params: dict, cloud_opts: dict = None) -> dict:
    """将本地 Tab 1~3 清洗参数与云端算力选项映射为与 train_moss.py 完全一致的配置载荷。"""
    opts = cloud_opts or {}
    acc = (opts.get("accelerator") or "NvidiaTeslaT4").strip()
    variant = (opts.get("variant") or "heal").strip().lower()
    if variant not in ("heal", "base", "sft"):
        variant = "heal"
    base_quant = (opts.get("base_quant") or "auto").strip().lower()
    if base_quant not in ("auto", "int8", "full"):
        base_quant = "auto"

    if acc == "NvidiaTeslaT4":
        gpus_val, parallel_val = "auto", "ddp"
    else:
        gpus_val, parallel_val = "1", "none"

    try:
        res_int = max(256, int(cleaned_params.get("resolution", 1024)))
    except Exception:
        res_int = 1024

    save_fmt = str(cleaned_params.get("save_format", "both")).strip().lower()
    if save_fmt not in ("both", "comfy", "hf", "lycoris"):
        save_fmt = "both"

    raw_max_min = opts.get("max_run_minutes", 540)
    max_run_minutes = (
        540
        if (raw_max_min is None or str(raw_max_min).strip() == "")
        else max(0, int(float(str(raw_max_min).strip())))
    )

    payload = {
        "lora_name": cleaned_params["lora_name"],
        "trigger_word": cleaned_params["trigger_word"],
        "repeats": int(cleaned_params["repeats"]),
        "epochs": int(cleaned_params["epochs"]),
        "steps": int(cleaned_params["steps"]),
        "rank": int(cleaned_params["rank"]),
        "alpha": float(cleaned_params["alpha"]),
        "algo": str(cleaned_params.get("algo", "lora")).lower(),
        "lokr_factor": int(cleaned_params["lokr_factor"]),
        "lokr_full_matrix": bool(cleaned_params.get("lokr_full_matrix", False)),
        "lokr_decompose_both": bool(cleaned_params.get("lokr_decompose_both", False)),
        "train_norms": bool(cleaned_params.get("train_norms", False)),
        "input_dropout": float(cleaned_params["input_dropout"]),
        "rank_dropout": float(cleaned_params["rank_dropout"]),
        "module_dropout": float(cleaned_params["module_dropout"]),
        "dora": bool(cleaned_params.get("dora", False)),
        "target_preset": str(cleaned_params.get("target_preset", "attn-mlp")),
        "loraplus_ratio": float(cleaned_params.get("loraplus_ratio", 1.0)),
        "lr": str(cleaned_params["lr"]),
        "optimizer": str(cleaned_params.get("optimizer", "adamw")),
        "lr_scheduler": str(cleaned_params.get("lr_scheduler", "cosine")),
        "warmup_steps": int(cleaned_params["warmup_steps"]),
        "weight_decay": float(cleaned_params["weight_decay"]),
        "wd_rules": str(cleaned_params.get("wd_rules", "moss")),
        "ema": "power" if str(cleaned_params.get("ema", "0.0")).strip().lower() == "power" else float(cleaned_params["ema"]),
        "grad_acc": int(cleaned_params["grad_acc"]),
        "timestep": str(cleaned_params.get("timestep", "moss")),
        "timestep_mu": float(cleaned_params["timestep_mu"]),
        "text_drop": float(cleaned_params["text_drop"]),
        "flip": bool(cleaned_params.get("flip", False)),
        "seed": int(cleaned_params["seed"]),
        "save_format": save_fmt,
        "resolution": res_int,
        "save_every": int(cleaned_params["save_every"]),
        "tune": str(cleaned_params.get("tune", "off")),
        "overlong": "drop_tokens",
        "caption_template": str(cleaned_params.get("caption_template", "auto")),
        "caption_p_nl": float(cleaned_params.get("caption_p_nl", 0.75)),
        "caption_variants": int(cleaned_params.get("caption_variants", 4)),
        "int8_compute": bool(cleaned_params.get("int8_compute", False)),
        "compile_mode": str(cleaned_params.get("compile_mode", "off")),
        "step_release": str(cleaned_params.get("step_release", "none")),
        "sample_every": int(cleaned_params.get("sample_every", 0)),
        "sample_prompt": str(cleaned_params.get("sample_prompt", "")),
        "teacher_mode": str(cleaned_params.get("teacher_mode", "none")),
        "distill_weight": float(cleaned_params.get("distill_weight", 0.05)),
        "variant": variant,
        "base_quant": base_quant,
        "vram_preset": "auto",
        "precision": "auto",
        "base_precision": "auto",
        "text_device": "auto",
        "vae_device": "cuda",
        "gpus": gpus_val,
        "parallel": parallel_val,
        "max_tokens": 0,
        "train_mode": "fresh",
        "max_run_minutes": max_run_minutes,
    }

    ds_source = str(opts.get("dataset_source", "local") or "local").strip().lower()
    if ds_source == "hf":
        hf_repo = normalize_hf_dataset_repo(str(opts.get("hf_dataset_repo", "")))
        if hf_repo:
            payload["hf_dataset_repo"] = hf_repo
            payload["hf_dataset_subdir"] = str(opts.get("hf_dataset_subdir", "") or "").strip().strip("/")
            payload["hf_dataset_revision"] = str(opts.get("hf_dataset_revision", "main") or "main").strip() or "main"
            hf_tok = str(opts.get("hf_token", "") or "").strip()
            if hf_tok:
                payload["hf_token"] = hf_tok
    elif ds_source == "url":
        ds_url = sanitize_external_dataset_url(opts.get("dataset_url", ""))
        if ds_url:
            payload["dataset_url"] = ds_url
    elif ds_source == "r2":
        ds_url = str(opts.get("dataset_url", "") or "").strip()
        if ds_url:
            payload["dataset_url"] = ds_url

    return payload


def _copy_image_txt_pairs(src_dir: Path, dst_dir: Path) -> tuple:
    """将单目录中的有效图片、同名 .txt 标注与 train.jsonl 以纯 ASCII 安全短名复制到暂存目录，返回 (图片数, 标注数, 总字节数)。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    img_cnt, txt_cnt, total_bytes = 0, 0, 0
    name_map = {}
    for item in sorted(src_dir.iterdir(), key=lambda p: p.name.lower()):
        if item.name.startswith(".") or item.name.startswith("._"):
            continue
        if not item.is_file() or item.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        try:
            if item.stat().st_size == 0:
                continue
        except OSError:
            continue

        img_cnt += 1
        stem = f"sample_{img_cnt:04d}"
        dst_name = f"{stem}{item.suffix.lower()}"
        dst_img = dst_dir / dst_name
        shutil.copy2(item, dst_img)
        name_map[item.name] = dst_name
        total_bytes += dst_img.stat().st_size
        src_txt = item.with_suffix(".txt")
        if src_txt.is_file():
            dst_txt = dst_dir / f"{stem}.txt"
            shutil.copy2(src_txt, dst_txt)
            txt_cnt += 1
            total_bytes += dst_txt.stat().st_size

    src_jsonl = src_dir / "train.jsonl"
    if src_jsonl.is_file() and name_map:
        rewritten_lines = []
        for raw_line in src_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            key_hit = None
            for k in ("image", "file", "path", "file_name"):
                if k in rec and str(rec[k] or "").strip():
                    key_hit = k
                    break
            if not key_hit:
                continue
            orig_base = Path(str(rec[key_hit]).replace("\\", "/")).name
            if orig_base not in name_map:
                continue
            rec[key_hit] = name_map[orig_base]
            rewritten_lines.append(json.dumps(rec, ensure_ascii=False))
        if rewritten_lines:
            dst_jsonl = dst_dir / "train.jsonl"
            dst_jsonl.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")
            total_bytes += dst_jsonl.stat().st_size
            if txt_cnt == 0:
                txt_cnt = len(rewritten_lines)
    return img_cnt, txt_cnt, total_bytes


def build_kaggle_dataset_title(slug: str, suffix: str = "") -> str:
    """将连字符 Slug 转为合法且不易撞车的 Title Case 标题，确保总长严格处于 Kaggle 的 [6, 50] 区间。"""
    tokens = re.findall(r"[A-Za-z0-9]+", str(slug or "").strip())
    if tokens:
        words = [tok.capitalize() for tok in tokens]
        base = " ".join(words)
        if words[0].lower() != "moss":
            base = f"MOSS {base}"
    else:
        base = "MOSS LoRA Dataset"

    sfx = re.sub(r"[^A-Za-z0-9]+", "", str(suffix or "").strip())
    sfx_part = f" {sfx}" if sfx else ""
    max_base_len = max(1, 50 - len(sfx_part))
    clipped_base = base[:max_base_len].rstrip() or "MOSS"
    title = f"{clipped_base}{sfx_part}".strip()
    if len(title) < 6:
        pad = " Dataset"
        avail = max(1, 50 - len(pad) - len(sfx_part))
        title = f"{clipped_base[:avail].rstrip()}{pad}{sfx_part}".strip()
    return title[:50].rstrip()


def _cleanup_stale_kaggle_upload_cache(cache_dir: Path = None) -> int:
    """清理本地 .kaggle-uploads 中指向已失效临时压缩包的孤立断点缓存，返回清理文件数。"""
    target_dir = Path(cache_dir) if cache_dir else (Path(tempfile.gettempdir()) / ".kaggle-uploads")
    if not target_dir.is_dir():
        return 0
    removed = 0
    for jf in target_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            cached_path = str(data.get("path", "") or "").strip() if isinstance(data, dict) else ""
            if not cached_path or not Path(cached_path).exists():
                jf.unlink(missing_ok=True)
                removed += 1
        except Exception:
            try:
                jf.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    return removed


def _extract_kaggle_cli_error(output: str) -> str:
    """识别 Kaggle CLI 在 rc=0 下的业务失败行及 HTTP 400 正文，命中则返回合并错误摘要。"""
    markers = (
        "dataset creation error:",
        "dataset version creation error:",
        "[kaggle api http ",
        '"status":"error"',
        '"status": "error"',
    )
    matched = []
    seen = set()
    for line in (output or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(m in low for m in markers) and stripped not in seen:
            seen.add(stripped)
            matched.append(stripped)
    return "\n".join(matched)[:500]


def _resolve_kaggle_python_exe(cli: dict) -> str:
    """若 CLI 为真实本机可执行的 Python Scripts\\kaggle.exe，返回同环境 python.exe 路径；否则返回空串回退 CLI。"""
    cmd = cli.get("cmd") or []
    if not cmd:
        return ""
    if len(cmd) >= 3 and cmd[1] == "-m" and cmd[2] == "kaggle":
        return cmd[0] if Path(cmd[0]).is_file() else ""
    exe = Path(cmd[0])
    if not exe.is_file():
        return ""
    for cand in (
        exe.parent.parent / "python.exe",
        exe.parent / "python.exe",
        exe.parent.parent / "bin" / "python",
        exe.parent / "python",
    ):
        if cand.is_file():
            return str(cand)
    return ""


_KAGGLE_DATASET_OP_RUNNER = """
import sys
import requests

mode, folder, notes = sys.argv[1], sys.argv[2], sys.argv[3]
_captured_http_errors = []
try:
    from kagglesdk.kaggle_http_client import KaggleHttpClient
    _orig_prep = KaggleHttpClient._prepare_response

    def _dbg_prep(self, response_type, http_response):
        if getattr(http_response, "status_code", 0) >= 400:
            body = (getattr(http_response, "text", "") or "").strip()
            if body:
                _captured_http_errors.append((http_response.status_code, body[:1000]))
        return _orig_prep(self, response_type, http_response)

    KaggleHttpClient._prepare_response = _dbg_prep
except Exception:
    pass


def _dump_http_errors():
    for code, body in _captured_http_errors:
        # KaggleApi.dataset_create_new 会先调用 dataset_status 检查是否存在，新数据集固定返回被捕获的 403 datasets.get
        if int(code) == 403 and "datasets.get" in body:
            continue
        print(f"[Kaggle API HTTP {code}] {body}", flush=True)


from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
try:
    if mode == "version":
        res = api.dataset_create_version(
            folder=folder,
            version_notes=notes,
            quiet=False,
            convert_to_csv=False,
            delete_old_versions=False,
            dir_mode="zip",
        )
    else:
        res = api.dataset_create_new(
            folder=folder,
            public=False,
            quiet=False,
            convert_to_csv=False,
            dir_mode="zip",
        )
except requests.exceptions.HTTPError as exc:
    _dump_http_errors()
    resp = getattr(exc, "response", None)
    body = (getattr(resp, "text", "") or "").strip() if resp is not None else ""
    if body and not _captured_http_errors:
        print(f"[Kaggle API HTTP {getattr(resp, 'status_code', '?')}] {body[:1000]}", flush=True)
    print(f"Dataset creation error: {exc}", flush=True)
    sys.exit(1)
except Exception as exc:
    _dump_http_errors()
    print(f"Dataset creation error: {exc}", flush=True)
    sys.exit(1)

status_str = str(getattr(res, "status", "") or "").lower()
err_str = str(getattr(res, "error", "") or "").strip()
if status_str != "ok":
    _dump_http_errors()
    print(f"Dataset creation error: {err_str or status_str or 'Unknown error'}", flush=True)
    sys.exit(1)
url_str = str(getattr(res, "url", "") or "").strip()
if url_str:
    print(f"Dataset operation succeeded: {url_str}", flush=True)
""".strip()


def _run_kaggle_dataset_op(mode: str, stage_dir: Path, notes: str, cli: dict, env: dict, on_log=None) -> tuple:
    """优先使用带 HTTP 响应体抓包与 convert_to_csv=False 的 Python API 子进程；不可用时回退官方 CLI (-t)。"""
    kaggle_py = _resolve_kaggle_python_exe(cli)
    if kaggle_py:
        cmd = [kaggle_py, "-u", "-c", _KAGGLE_DATASET_OP_RUNNER, mode, str(stage_dir), notes]
    elif mode == "version":
        cmd = [*cli["cmd"], "datasets", "version", "-p", str(stage_dir), "-m", notes, "-r", "zip", "-t"]
    else:
        cmd = [*cli["cmd"], "datasets", "create", "-p", str(stage_dir), "-r", "zip", "-t"]
    return _stream_subprocess(cmd, stage_dir, on_log=on_log, env=env)


def _format_dataset_error(prefix: str, rc: int, out: str, err_hint: str) -> str:
    tail = (out or "").strip()[-600:]
    if err_hint:
        return f"{prefix} (Exit={rc})：\n{err_hint}\n\n{tail}"
    return f"{prefix} (Exit={rc})：\n{tail}"


def _build_subset_plan(raw_params: dict) -> list:
    """统一规划多子集目录映射与相对清单，保证数据集打包与导出 Notebook 参数 100% 对齐。"""
    subsets = [s for s in (raw_params.get("subsets") or []) if isinstance(s, dict) and s.get("enabled", True)]
    plan = []
    used_keys = set()
    for idx, sub in enumerate(subsets, start=1):
        src_path = Path(str(sub.get("path", "")).strip())
        raw_name = str(sub.get("name") or src_path.name or f"subset_{idx}")
        safe_sub = re.sub(r"[^0-9A-Za-z_-]+", "_", raw_name).strip("_") or f"subset_{idx}"
        folder_key = f"{idx:02d}_{safe_sub}"
        while folder_key in used_keys:
            folder_key = f"{folder_key}_x"
        used_keys.add(folder_key)
        plan.append({
            "src_path": src_path,
            "name": raw_name,
            "folder_key": folder_key,
            "rel_path": f"subsets/{folder_key}",
            "repeats": max(1, int(sub.get("repeats", 1))),
            "trigger": str(sub.get("trigger", "") or "").strip(),
        })
    return plan


def _populate_dataset_stage_dir(stage_dir: Path, raw_params: dict) -> dict:
    """按单图集或 Tab 2 多子集结构整理训练样本到暂存目录，并返回明细统计。"""
    multi_subset = bool(raw_params.get("multi_subset", False))
    lora_name = (str(raw_params.get("lora_name", "")).strip() or "my_moss_lora")
    total_imgs, total_txts, total_bytes = 0, 0, 0

    if multi_subset:
        plan = _build_subset_plan(raw_params)
        if not plan:
            raise ValueError("多子集模式下没有已启用的子集，请先前往【📂 数据与续炼】检查子集列表。")
        cloud_manifest = []
        for idx, item in enumerate(plan, start=1):
            src_path = item["src_path"]
            if not src_path.is_dir():
                raise FileNotFoundError(f"子集 #{idx} 目录不存在：{src_path}")
            ic, tc, b = _copy_image_txt_pairs(src_path, stage_dir / item["rel_path"])
            if ic == 0:
                raise ValueError(f"子集 [{item['name']}] 目录中没有支持的图片文件：{src_path}")
            total_imgs += ic
            total_txts += tc
            total_bytes += b
            cloud_manifest.append({
                "name": item["name"],
                "path": item["rel_path"],
                "repeats": item["repeats"],
                "trigger": item["trigger"],
                "enabled": True,
            })

        (stage_dir / "subsets_cloud.json").write_text(
            json.dumps(cloud_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "mode": "multi_subset",
            "subset_count": len(cloud_manifest),
            "images": total_imgs,
            "captions": total_txts,
            "total_mb": round(total_bytes / (1024**2), 2),
        }

    data_dir = Path(str(raw_params.get("data_dir", "")).strip())
    if not data_dir.is_dir():
        raise FileNotFoundError(f"本地训练图集目录不存在：{data_dir}")
    total_imgs, total_txts, total_bytes = _copy_image_txt_pairs(data_dir, stage_dir / "images")
    if total_imgs == 0:
        raise ValueError(f"图集目录 {data_dir} 中未发现有效图片，无法打包上传。")
    (stage_dir / "cloud_dataset_manifest.json").write_text(
        json.dumps(
            {
                "mode": "single",
                "lora_name": lora_name,
                "images": total_imgs,
                "captions": total_txts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "mode": "single",
        "subset_count": 1,
        "images": total_imgs,
        "captions": total_txts,
        "total_mb": round(total_bytes / (1024**2), 2),
    }


def stage_dataset_bundle(raw_params: dict, username: str, dataset_slug: str, stage_root: Path = None) -> tuple:
    """准备待上传 Kaggle Private Dataset 的完整暂存目录及 dataset-metadata.json。"""
    user = (username or "").strip()
    slug = sanitize_kaggle_slug(dataset_slug)
    if not user:
        raise ValueError("Kaggle 用户名不能为空。")

    root = Path(stage_root) if stage_root else (PROJECT_ROOT / "workspace" / "kaggle_dataset_stage")
    root.mkdir(parents=True, exist_ok=True)
    for old_sub in list(root.iterdir()):
        if old_sub.is_dir() and old_sub.name != slug:
            shutil.rmtree(old_sub, ignore_errors=True)
    stage_dir = root / slug
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    stats = _populate_dataset_stage_dir(stage_dir, raw_params)
    title = build_kaggle_dataset_title(slug)
    metadata = {
        "title": title,
        "id": f"{user}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    stats["dataset_id"] = f"{user}/{slug}"
    stats["dataset_slug"] = slug
    stats["dataset_title"] = title
    return stage_dir, stats


def upload_kaggle_dataset(
    stage_dir: Path,
    dataset_id: str,
    version_notes: str = "Sync dataset from local GUI",
    cli_info: dict = None,
    kaggle_json_path: Path = None,
    on_log=None,
    verify_poll_attempts: int = 3,
    cleanup_stage: bool = True,
) -> tuple:
    """智能创建或更新 Kaggle 私有数据集：自动清洗缓存、防标题撞车重试、抓全报错正文并异步验活。"""
    try:
        cli = cli_info if (cli_info and cli_info.get("found")) else find_kaggle_cli()
        if not cli.get("found"):
            return False, "未找到 Kaggle CLI，请先安装并配置凭证。"

        _cleanup_stale_kaggle_upload_cache()
        env = _build_kaggle_env(kaggle_json_path=kaggle_json_path)

        def _query_status():
            return subprocess.run(
                [*cli["cmd"], "datasets", "status", dataset_id],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                env=env,
                creationflags=NO_WINDOW,
            )

        exists = False
        try:
            st_res = _query_status()
            exists = (st_res.returncode == 0)
        except Exception:
            exists = False

        notes = (version_notes or "Sync dataset from MOSS LoRA Trainer GUI").strip()

        def _is_op_success(exit_code: int, text: str, err_text: str) -> bool:
            if exit_code != 0:
                return False
            if not err_text:
                return True
            low_text = (text or "").lower()
            if "dataset operation succeeded:" in low_text and "dataset creation error:" not in low_text:
                remaining = [
                    ln for ln in err_text.splitlines()
                    if not ("403" in ln.lower() and "datasets.get" in ln.lower())
                ]
                return len(remaining) == 0
            return False

        if exists:
            if on_log:
                on_log(f"[Kaggle 图集] 检测到私有数据集 {dataset_id} 已存在，正在推送新版本...\n")
            rc, out = _run_kaggle_dataset_op("version", Path(stage_dir), notes, cli, env, on_log=on_log)
            err_hint = _extract_kaggle_cli_error(out)
            if _is_op_success(rc, out, err_hint):
                return True, f"已成功更新私有数据集版本：{dataset_id}"
            return False, _format_dataset_error("更新数据集失败", rc, out, err_hint)

        if on_log:
            on_log(f"[Kaggle 图集] 正在创建私有数据集 {dataset_id}...\n")
        rc, out = _run_kaggle_dataset_op("create", Path(stage_dir), notes, cli, env, on_log=on_log)
        err_hint = _extract_kaggle_cli_error(out)
        effective_ok = _is_op_success(rc, out, err_hint)

        if not effective_ok:
            combined_lower = f"{err_hint}\n{out}".lower()
            if "title" in combined_lower and "already in use" in combined_lower:
                meta_file = Path(stage_dir) / "dataset-metadata.json"
                slug_part = dataset_id.split("/", 1)[1] if "/" in dataset_id else dataset_id
                new_title = build_kaggle_dataset_title(slug_part, suffix=secrets.token_hex(2))
                if meta_file.is_file():
                    try:
                        meta_obj = json.loads(meta_file.read_text(encoding="utf-8"))
                        meta_obj["title"] = new_title
                        meta_file.write_text(json.dumps(meta_obj, indent=2, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass
                if on_log:
                    on_log(f"[Kaggle 图集] 检测到标题占用，已自动更换防撞车标题 [{new_title}] 并重试创建...\n")
                rc, out = _run_kaggle_dataset_op("create", Path(stage_dir), notes, cli, env, on_log=on_log)
                err_hint = _extract_kaggle_cli_error(out)
                effective_ok = _is_op_success(rc, out, err_hint)
                combined_lower = f"{err_hint}\n{out}".lower()

            if not effective_ok and (
                "already exists" in combined_lower
                or "409" in combined_lower
                or ("already in use" in combined_lower and "title" not in combined_lower)
            ):
                if on_log:
                    on_log(f"[Kaggle 图集] 数据集 {dataset_id} 已存在，自动切换为增量版本推送...\n")
                rc2, out2 = _run_kaggle_dataset_op("version", Path(stage_dir), notes, cli, env, on_log=on_log)
                err_hint2 = _extract_kaggle_cli_error(out2)
                if _is_op_success(rc2, out2, err_hint2):
                    return True, f"已自动更新现有私有数据集版本：{dataset_id}"
                return False, _format_dataset_error("增量更新数据集失败", rc2, out2, err_hint2)

        if not effective_ok:
            return False, _format_dataset_error("上传私有数据集失败", rc, out, err_hint)

        # 创建受理后做轻量异步验活，确认云端解包是否真正进入 ready 状态
        if int(verify_poll_attempts) > 0:
            poll_delays = (3, 5, 6)
            for idx in range(int(verify_poll_attempts)):
                time.sleep(poll_delays[min(idx, len(poll_delays) - 1)])
                try:
                    st_after = _query_status()
                    st_txt = f"{st_after.stdout or ''} {st_after.stderr or ''}".strip()
                    if st_after.returncode == 0:
                        low_st = st_txt.lower()
                        if "ready" in low_st:
                            if on_log:
                                on_log(f"[Kaggle 验活] 数据集 {dataset_id} 云端解包完成，状态：ready ✅\n")
                            return True, f"已成功创建并就绪私有数据集：{dataset_id}"
                        if "error" in low_st:
                            return False, f"Kaggle 后台处理数据集失败：{st_txt}"
                except Exception:
                    pass
            if on_log:
                on_log(
                    f"[Kaggle 验活] 数据集 {dataset_id} 已受理创建但云端仍在后台解包索引；"
                    f"如随即推送报错 403，请等待 15~30 秒后重试或查看 https://www.kaggle.com/datasets/{dataset_id}\n"
                )
            return (
                True,
                f"已提交创建私有数据集：{dataset_id}（云端正在后台索引，建议稍候 15~30 秒或访问 https://www.kaggle.com/datasets/{dataset_id} 确认就绪后再推送）",
            )

        return True, f"已成功创建并上传私有数据集：{dataset_id}"
    finally:
        st_path = Path(stage_dir)
        if cleanup_stage and st_path.parent.name == "kaggle_dataset_stage" and st_path.exists():
            shutil.rmtree(st_path, ignore_errors=True)


def stage_external_dataset_zip(
    raw_params: dict,
    dataset_slug: str = "",
    stage_root: Path = None,
    on_log=None,
) -> tuple:
    """将单图集或多子集图片与 .txt 标注整洁打包为单一 ZIP 归档供外链中转站上传。"""
    raw_slug = dataset_slug or str(raw_params.get("lora_name", "") or "moss-data")
    slug_base = sanitize_kaggle_slug(raw_slug, prefix="moss-data")
    root = Path(stage_root) if stage_root else (PROJECT_ROOT / "workspace" / "kaggle_dataset_stage")
    root.mkdir(parents=True, exist_ok=True)

    ext_stage_dir = root / f"{slug_base}_ext_stage"
    if ext_stage_dir.exists():
        shutil.rmtree(ext_stage_dir, ignore_errors=True)
    ext_stage_dir.mkdir(parents=True, exist_ok=True)

    stats = _populate_dataset_stage_dir(ext_stage_dir, raw_params)
    zip_path = root / f"{slug_base}_direct.zip"
    if zip_path.exists():
        zip_path.unlink(missing_ok=True)

    if on_log:
        on_log(f"[外链打包] 正在压缩 {stats['images']} 张图与 {stats['captions']} 份标注 -> {zip_path.name}...\n")

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            for fp in sorted(ext_stage_dir.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(ext_stage_dir).as_posix())
    finally:
        shutil.rmtree(ext_stage_dir, ignore_errors=True)

    stats["zip_path"] = str(zip_path)
    stats["zip_mb"] = round(zip_path.stat().st_size / (1024**2), 2)
    return zip_path, stats


def upload_zip_to_external_mirrors(
    zip_path: Path,
    on_log=None,
    max_mirrors: int = 1,
) -> tuple:
    """按序尝试 5 个 Kaggle 安全的免登录外链站（temp.sh / filebin.net / transfer.whalebone.io / ul.sto.care / tmpfiles.org）；默认任一站点上传成功即止，失败自动顺延下一站。返回 (ok, 直链或 '|' 拼接多镜像或报错, 明细列表)。"""
    zpath = Path(zip_path).resolve()
    if not zpath.is_file() or zpath.stat().st_size == 0:
        return False, f"待上传 ZIP 文件不存在或为空：{zpath}", []

    curl_bin = shutil.which("curl.exe") or shutil.which("curl") or "curl"
    posix_zip = zpath.as_posix()
    size_mb = zpath.stat().st_size / (1024**2)
    mirrors = []
    errors = []

    def _run_upload(label: str, cmd: list):
        if on_log:
            on_log(f"[外链上传] 正在向 {label} 上传 {zpath.name} ({size_mb:.2f} MB)...\n")
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=320,
            creationflags=NO_WINDOW,
        )

    def _record_fail(host: str, detail: str, has_next: bool = True):
        errors.append(f"{host}: {detail}")
        if on_log:
            if has_next:
                on_log(f"  ⚠️ {host} 上传失败，自动顺延尝试下一候选站...\n")
            else:
                on_log(f"  ⚠️ {host} 上传失败：{detail}\n")

    # 1. temp.sh（72 小时有效，单文件最高 4GB）
    if len(mirrors) < max_mirrors:
        try:
            res = _run_upload(
                "temp.sh (72h/4GB)",
                [curl_bin, "-sS", "--connect-timeout", "20", "--max-time", "300", "-F", f"file=@{posix_zip}", "https://temp.sh/upload"],
            )
            url_out = (res.stdout or "").strip().splitlines()[0].strip() if (res.stdout or "").strip() else ""
            if res.returncode == 0 and re.match(r"^https?://temp\.sh/[A-Za-z0-9]+/", url_out, re.IGNORECASE):
                mirrors.append({"host": "temp.sh", "url": url_out})
                if on_log:
                    on_log(f"  ✅ temp.sh 直链就绪: {url_out}\n")
            else:
                _record_fail("temp.sh", f"rc={res.returncode} out={url_out[:120]} err={(res.stderr or '')[:120]}")
        except Exception as exc:
            _record_fail("temp.sh", str(exc))

    # 2. filebin.net（6 天有效）
    if len(mirrors) < max_mirrors:
        try:
            bin_id = f"moss-{secrets.token_hex(4)}"
            target_url = f"https://filebin.net/{bin_id}/{zpath.name}"
            res = _run_upload(
                "filebin.net (6d)",
                [
                    curl_bin, "-sS", "--connect-timeout", "20", "--max-time", "300",
                    "-X", "PUT", "--data-binary", f"@{posix_zip}", "-H", "cid: moss",
                    target_url,
                ],
            )
            raw_out = (res.stdout or "").strip()
            parsed = json.loads(raw_out) if raw_out.startswith("{") else {}
            if res.returncode == 0 and isinstance(parsed, dict) and ("bin" in parsed or "file" in parsed):
                mirrors.append({"host": "filebin.net", "url": target_url})
                if on_log:
                    on_log(f"  ✅ filebin.net 直链就绪: {target_url}\n")
            else:
                _record_fail("filebin.net", f"rc={res.returncode} out={raw_out[:140]} err={(res.stderr or '')[:120]}")
        except Exception as exc:
            _record_fail("filebin.net", str(exc))

    # 3. transfer.whalebone.io（transfer.sh 开源兼容镜像）
    if len(mirrors) < max_mirrors:
        try:
            res = _run_upload(
                "transfer.whalebone.io",
                [curl_bin, "-sS", "--connect-timeout", "20", "--max-time", "300", "-T", posix_zip, f"https://transfer.whalebone.io/{zpath.name}"],
            )
            url_out = (res.stdout or "").strip().splitlines()[0].strip() if (res.stdout or "").strip() else ""
            if res.returncode == 0 and re.match(r"^https?://transfer\.whalebone\.io/[A-Za-z0-9]+/", url_out, re.IGNORECASE):
                mirrors.append({"host": "transfer.whalebone.io", "url": url_out})
                if on_log:
                    on_log(f"  ✅ transfer.whalebone.io 直链就绪: {url_out}\n")
            else:
                _record_fail(
                    "transfer.whalebone.io",
                    f"rc={res.returncode} out={url_out[:120]} err={(res.stderr or '')[:120]}",
                    has_next=(size_mb <= 98.0),
                )
        except Exception as exc:
            _record_fail("transfer.whalebone.io", str(exc), has_next=(size_mb <= 98.0))

    # 4. ul.sto.care / trystocare.com（API 上限 100MB，72 小时有效）
    if len(mirrors) < max_mirrors and size_mb <= 98.0:
        try:
            res = _run_upload(
                "ul.sto.care (72h/100MB)",
                [curl_bin, "-sS", "--connect-timeout", "20", "--max-time", "300", "-T", posix_zip, f"https://ul.sto.care/{zpath.name}"],
            )
            raw_out = (res.stdout or "").strip()
            parsed = json.loads(raw_out) if raw_out.startswith("{") else {}
            sto_url = str((parsed.get("url", "")) or "").strip()
            if res.returncode == 0 and re.match(r"^https?://trystocare\.com/[A-Za-z0-9_-]+", sto_url, re.IGNORECASE):
                mirrors.append({"host": "trystocare.com", "url": sto_url})
                if on_log:
                    on_log(f"  ✅ trystocare.com 直链就绪: {sto_url}\n")
            else:
                _record_fail("ul.sto.care", f"rc={res.returncode} out={raw_out[:140]} err={(res.stderr or '')[:120]}")
        except Exception as exc:
            _record_fail("ul.sto.care", str(exc))

    # 5. tmpfiles.org（60 分钟有效，上限 100MB）
    if len(mirrors) < max_mirrors and size_mb <= 98.0:
        try:
            res = _run_upload(
                "tmpfiles.org (1h/100MB)",
                [curl_bin, "-sS", "--connect-timeout", "20", "--max-time", "300", "-F", f"file=@{posix_zip}", "https://tmpfiles.org/api/v1/upload"],
            )
            raw_out = (res.stdout or "").strip()
            parsed = json.loads(raw_out) if raw_out.startswith("{") else {}
            page_url = str(((parsed.get("data") or {}).get("url", "")) or "").strip()
            if res.returncode == 0 and page_url.startswith("http"):
                mirrors.append({"host": "tmpfiles.org", "url": page_url})
                if on_log:
                    on_log(f"  ✅ tmpfiles.org 直链就绪: {page_url}\n")
            else:
                _record_fail("tmpfiles.org", f"rc={res.returncode} out={raw_out[:140]} err={(res.stderr or '')[:120]}", has_next=False)
        except Exception as exc:
            _record_fail("tmpfiles.org", str(exc), has_next=False)

    if mirrors:
        combined_url = "|".join(m["url"] for m in mirrors)
        return True, combined_url, mirrors

    err_msg = "所有外链中转站均上传失败，请检查网络连接或改填自定义 ZIP 直链：\n" + "\n".join(f"• {e}" for e in errors)
    return False, err_msg, []


def upload_external_dataset(
    raw_params: dict,
    dataset_slug: str = "",
    stage_root: Path = None,
    on_log=None,
    cleanup_zip: bool = True,
    max_mirrors: int = 1,
) -> tuple:
    """打包本地单/多子集图集并按序上传至外链中转站（默认成功 1 站即止），返回 (ok, 直链或报错, 统计详情)。"""
    zip_path, stats = stage_external_dataset_zip(
        raw_params,
        dataset_slug=dataset_slug,
        stage_root=stage_root,
        on_log=on_log,
    )
    if on_log:
        on_log(
            f"[外链打包完成] 模式={stats['mode']} | {stats['images']} 张图 / {stats['captions']} 份标注 "
            f"| 压缩包 {stats['zip_mb']} MB -> {zip_path}\n"
        )
    try:
        ok, url_or_err, mirrors = upload_zip_to_external_mirrors(
            zip_path,
            on_log=on_log,
            max_mirrors=max_mirrors,
        )
        stats["mirrors"] = mirrors
        return ok, url_or_err, stats
    finally:
        if cleanup_zip:
            try:
                Path(zip_path).unlink(missing_ok=True)
            except OSError:
                pass


def _resolve_hf_hub_python() -> str:
    """定位安装有 huggingface_hub 的 Python 解释器（用于当前精简宿主缺少该包时自动桥接上传）。"""
    for cand in (resolve_cli_python(), sys.executable):
        if cand and Path(cand).is_file():
            try:
                chk = subprocess.run(
                    [cand, "-c", "import huggingface_hub"],
                    capture_output=True,
                    timeout=5,
                    creationflags=NO_WINDOW,
                )
                if chk.returncode == 0:
                    return cand
            except Exception:
                pass
    detect_py = PROJECT_ROOT / "scripts" / "detect_env.py"
    if detect_py.is_file():
        try:
            res = subprocess.run(
                [resolve_cli_python(), str(detect_py)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                creationflags=NO_WINDOW,
            )
            found = (res.stdout or "").strip()
            if found and Path(found).is_file():
                return found
        except Exception:
            pass
    return ""


def upload_dataset_to_hf_repo(
    raw_params: dict,
    repo_id: str,
    hf_token: str,
    subdir: str = "",
    private: bool = True,
    stage_root: Path = None,
    on_log=None,
) -> tuple:
    """将本地单/多子集图集整理后一键创建并上传至 Hugging Face Private Dataset 永久私有仓库。"""
    repo_norm = normalize_hf_dataset_repo(repo_id)
    if not repo_norm or "/" not in repo_norm:
        return False, "请填写有效的 Hugging Face Dataset 仓库 ID（如 owner/my-dataset）。", {}

    tok = (hf_token or os.environ.get("HF_TOKEN", "") or "").strip()
    if not tok:
        return False, "上传至 Hugging Face 私有仓库必须提供具备 Write 权限的 Access Token。", {}

    clean_sub = str(subdir or "").strip().strip("/")
    safe_slug = sanitize_kaggle_slug(repo_norm.replace("/", "-"), prefix="hf-data")
    root = Path(stage_root) if stage_root else (PROJECT_ROOT / "workspace" / "kaggle_dataset_stage")
    hf_stage_dir = root / f"{safe_slug}_hf_stage"
    if hf_stage_dir.exists():
        shutil.rmtree(hf_stage_dir, ignore_errors=True)
    hf_stage_dir.mkdir(parents=True, exist_ok=True)

    try:
        stats = _populate_dataset_stage_dir(hf_stage_dir, raw_params)
        if on_log:
            on_log(
                f"[HF 私有仓] 已整理 {stats['images']} 张图 / {stats['captions']} 份标注 ({stats['total_mb']} MB)，"
                f"正在同步至私有仓库 `{repo_norm}`" + (f" (子目录: {clean_sub})" if clean_sub else "") + "...\n"
            )

        commit_msg = f"Sync {stats['images']} images from MOSS LoRA Trainer GUI"
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=tok)
            api.create_repo(repo_id=repo_norm, repo_type="dataset", private=bool(private), exist_ok=True)
            upload_kwargs = {
                "folder_path": str(hf_stage_dir),
                "repo_id": repo_norm,
                "repo_type": "dataset",
                "commit_message": commit_msg,
            }
            if clean_sub:
                upload_kwargs["path_in_repo"] = clean_sub
            api.upload_folder(**upload_kwargs)
        except ImportError:
            py_hf = _resolve_hf_hub_python()
            if not py_hf:
                return (
                    False,
                    "当前 Python 环境及已探测环境均未安装 huggingface_hub，请执行 `pip install huggingface_hub` 后重试。",
                    stats,
                )
            helper_code = (
                "import os, sys\n"
                "from huggingface_hub import HfApi\n"
                "repo_id, folder, sub, msg = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]\n"
                "priv = sys.argv[5] == '1'\n"
                "api = HfApi(token=os.environ['HF_TOKEN'])\n"
                "api.create_repo(repo_id=repo_id, repo_type='dataset', private=priv, exist_ok=True)\n"
                "kw = {'folder_path': folder, 'repo_id': repo_id, 'repo_type': 'dataset', 'commit_message': msg}\n"
                "if sub: kw['path_in_repo'] = sub\n"
                "api.upload_folder(**kw)\n"
                "print('HF_UPLOAD_OK')\n"
            )
            env = _build_kaggle_env()
            env["HF_TOKEN"] = tok
            res = subprocess.run(
                [py_hf, "-c", helper_code, repo_norm, str(hf_stage_dir), clean_sub, commit_msg, "1" if private else "0"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
                env=env,
                creationflags=NO_WINDOW,
            )
            if res.returncode != 0 or "HF_UPLOAD_OK" not in (res.stdout or ""):
                err_txt = (res.stderr or res.stdout or "").strip()[-400:]
                return False, f"Hugging Face 私有仓库上传失败：{err_txt}", stats

        if on_log:
            on_log(f"  ✅ 已完成私有仓库同步：https://huggingface.co/datasets/{repo_norm}\n")
        stats["repo_id"] = repo_norm
        stats["subdir"] = clean_sub
        stats["private"] = bool(private)
        return True, repo_norm, stats
    except Exception as exc:
        return False, f"上传至 Hugging Face 私有仓库失败：{exc}", {}
    finally:
        shutil.rmtree(hf_stage_dir, ignore_errors=True)


def get_r2_json_path() -> Path:
    """返回 Cloudflare R2 凭据保存路径，优先尊重 R2_CREDENTIALS_PATH 环境变量。"""
    env_path = os.environ.get("R2_CREDENTIALS_PATH", "").strip()
    if env_path:
        return Path(env_path)
    return R2_CREDENTIALS_PATH


def read_r2_credentials(r2_json_path: Path = None) -> dict:
    """安全读取 Cloudflare R2 S3 凭据文件（Secret 仅用于内存签名，展示侧提供脱敏掩码）。"""
    path = Path(r2_json_path) if r2_json_path else get_r2_json_path()
    result = {
        "exists": False,
        "account_id": "",
        "access_key_id": "",
        "secret_access_key": "",
        "has_secret": False,
        "secret_masked": "",
        "bucket": "",
        "public_base_url": "",
        "path": str(path),
    }
    if not path.is_file():
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(data, dict):
            sec = str(data.get("secret_access_key", "") or "").strip()
            result["exists"] = True
            result["account_id"] = str(data.get("account_id", "") or "").strip()
            result["access_key_id"] = str(data.get("access_key_id", "") or "").strip()
            result["secret_access_key"] = sec
            result["has_secret"] = bool(sec)
            result["secret_masked"] = mask_api_key(sec)
            result["bucket"] = str(data.get("bucket", "") or "").strip()
            result["public_base_url"] = str(data.get("public_base_url", "") or "").strip()
    except Exception:
        result["exists"] = True
    return result


def save_r2_credentials(
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str = "",
    public_base_url: str = "",
    r2_json_path: Path = None,
) -> tuple:
    """将 Cloudflare R2 S3 API 凭据保存至用户目录 ~/.kaggle/r2_credentials.json（严禁写入仓库目录）。"""
    acct = (account_id or "").strip()
    ak = (access_key_id or "").strip()
    sk = (secret_access_key or "").strip()
    bkt = (bucket or "").strip()
    pub = (public_base_url or "").strip().rstrip("/")
    if not acct or not ak or not sk or not bkt:
        return False, "请完整填写 R2 Account ID、Bucket 名称、Access Key ID 与 Secret Access Key。"

    path = Path(r2_json_path) if r2_json_path else get_r2_json_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "account_id": acct,
            "access_key_id": ak,
            "secret_access_key": sk,
            "bucket": bkt,
            "public_base_url": pub,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return True, f"R2 凭据已安全保存至 {path}（Bucket: {bkt}）"
    except Exception as err:
        return False, f"保存 R2 凭据失败: {err}"


def _aws_v4_sign(key_bytes: bytes, msg: str) -> bytes:
    return hmac.new(key_bytes, msg.encode("utf-8"), hashlib.sha256).digest()


def generate_r2_presigned_url(
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str,
    object_key: str,
    method: str = "GET",
    expires_seconds: int = 604800,
    endpoint_url: str = "",
    now_utc: datetime.datetime = None,
) -> str:
    """
    零第三方依赖生成 Cloudflare R2 S3 兼容 AWS4-HMAC-SHA256 预签名 URL（region=auto, service=s3, UNSIGNED-PAYLOAD）。
    """
    acct = (account_id or "").strip()
    ak = (access_key_id or "").strip()
    sk = (secret_access_key or "").strip()
    bkt = (bucket or "").strip("/")
    key = (object_key or "").lstrip("/")
    if not acct or not ak or not sk or not bkt or not key:
        raise ValueError("Cloudflare R2 预签名缺少必要参数（account_id / access_key_id / secret_access_key / bucket / object_key）。")

    if endpoint_url and endpoint_url.strip():
        parsed_ep = urllib.parse.urlparse(endpoint_url.strip())
        scheme = parsed_ep.scheme or "https"
        host = parsed_ep.netloc or parsed_ep.path.strip("/")
    else:
        scheme = "https"
        host = f"{acct}.r2.cloudflarestorage.com"

    http_method = (method or "GET").strip().upper()
    region = "auto"
    service = "s3"
    expires = max(60, min(604800, int(expires_seconds)))

    ts = now_utc or datetime.datetime.now(datetime.timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    amz_date = ts.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = ts.strftime("%Y%m%d")
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"

    canonical_uri = f"/{urllib.parse.quote(bkt, safe='')}/{urllib.parse.quote(key, safe='/~_.-')}"
    query_params = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{ak}/{credential_scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_qs = "&".join(
        f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
        for k, v in sorted(query_params.items())
    )
    canonical_request = f"{http_method}\n{canonical_uri}\n{canonical_qs}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
    req_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{req_hash}"

    k_date = _aws_v4_sign(f"AWS4{sk}".encode("utf-8"), date_stamp)
    k_region = _aws_v4_sign(k_date, region)
    k_service = _aws_v4_sign(k_region, service)
    k_signing = _aws_v4_sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    return f"{scheme}://{host}{canonical_uri}?{canonical_qs}&X-Amz-Signature={signature}"


def upload_dataset_to_cloudflare_r2(
    raw_params: dict,
    r2_config: dict,
    dataset_slug: str = "",
    stage_root: Path = None,
    on_log=None,
    cleanup_zip: bool = True,
) -> tuple:
    """将本地图集打包为 ZIP 并通过 SigV4 预签名 PUT 上传至私有 Cloudflare R2 桶，返回 7 天有效期预签名直拉链接。"""
    saved_r2 = read_r2_credentials()
    acct = str(r2_config.get("account_id") or saved_r2.get("account_id") or "").strip()
    ak = str(r2_config.get("access_key_id") or saved_r2.get("access_key_id") or "").strip()
    sk = str(r2_config.get("secret_access_key") or saved_r2.get("secret_access_key") or "").strip()
    bkt = str(r2_config.get("bucket") or saved_r2.get("bucket") or "").strip("/")
    prefix = str(r2_config.get("prefix", "moss-datasets") or "moss-datasets").strip("/")
    pub_base = str(r2_config.get("public_base_url") or saved_r2.get("public_base_url") or "").strip().rstrip("/")
    endpoint_url = str(r2_config.get("endpoint_url", "") or "").strip()

    if not acct or not ak or not sk or not bkt:
        return False, "Cloudflare R2 配置不完整，请填写 Account ID、Bucket、Access Key ID 与 Secret Access Key。", {}

    zip_path, stats = stage_external_dataset_zip(
        raw_params,
        dataset_slug=dataset_slug,
        stage_root=stage_root,
        on_log=on_log,
    )

    try:
        slug_part = sanitize_kaggle_slug(dataset_slug or str(raw_params.get("lora_name", "") or "moss-data"), prefix="moss-data")
        ts_tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        obj_name = f"{slug_part}_{ts_tag}.zip"
        object_key = f"{prefix}/{obj_name}" if prefix else obj_name

        put_url = generate_r2_presigned_url(
            account_id=acct,
            access_key_id=ak,
            secret_access_key=sk,
            bucket=bkt,
            object_key=object_key,
            method="PUT",
            expires_seconds=3600,
            endpoint_url=endpoint_url,
        )

        curl_bin = shutil.which("curl.exe") or shutil.which("curl") or "curl"
        posix_zip = Path(zip_path).resolve().as_posix()
        if on_log:
            on_log(f"[R2 上传] 正在上传 {stats['zip_mb']} MB -> r2://{bkt}/{object_key} ...\n")

        try:
            res = subprocess.run(
                [
                    curl_bin, "-sS", "-g",
                    "--connect-timeout", "20",
                    "--max-time", "600",
                    "-w", "\n__HTTP_CODE__:%{http_code}",
                    "-X", "PUT",
                    "--upload-file", posix_zip,
                    put_url,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=620,
                creationflags=NO_WINDOW,
            )
        except Exception as exc:
            return False, f"Cloudflare R2 PUT 上传异常：{exc}", stats

        raw_out = res.stdout or ""
        http_code = 0
        m_code = re.search(r"__HTTP_CODE__:(\d+)", raw_out)
        if m_code:
            http_code = int(m_code.group(1))
        body_clean = re.sub(r"\s*__HTTP_CODE__:\d+\s*", "", raw_out).strip()

        if res.returncode != 0 or http_code not in (200, 201, 204):
            err_detail = body_clean[:300] or (res.stderr or "")[:200]
            return False, f"Cloudflare R2 上传失败 (curl rc={res.returncode}, HTTP {http_code})：{err_detail}", stats

        get_url = generate_r2_presigned_url(
            account_id=acct,
            access_key_id=ak,
            secret_access_key=sk,
            bucket=bkt,
            object_key=object_key,
            method="GET",
            expires_seconds=604800,
            endpoint_url=endpoint_url,
        )
        final_url = f"{pub_base}/{urllib.parse.quote(object_key, safe='/~_.-')}|{get_url}" if pub_base else get_url
        stats["r2_bucket"] = bkt
        stats["r2_object_key"] = object_key
        if on_log:
            on_log(f"  ✅ Cloudflare R2 私有上传完成：r2://{bkt}/{object_key}（已签发 7 天免密预签名直链）\n")
        return True, final_url, stats
    finally:
        if cleanup_zip:
            try:
                Path(zip_path).unlink(missing_ok=True)
            except OSError:
                pass


def inject_cloud_config_into_train_moss(source_script: Path, config_payload: dict) -> str:
    """将 GUI 超参数载荷安全注入 train_moss.py 的 CLOUD_GUI_CONFIG_JSON 占位符。"""
    src_text = Path(source_script).read_text(encoding="utf-8")
    marker = "CLOUD_GUI_CONFIG_JSON = None"
    if marker not in src_text:
        raise RuntimeError(f"脚本 {source_script} 中缺少 '{marker}' 注入锚点，请先同步最新 train_moss.py。")
    serialized = json.dumps(config_payload, ensure_ascii=False)
    return src_text.replace(marker, f"CLOUD_GUI_CONFIG_JSON = {repr(serialized)}", 1)


def _prepare_kernel_stage_dir(
    cleaned_params: dict,
    cloud_opts: dict,
    username: str,
    dataset_id: str,
    kernel_slug: str,
    stage_root: Path = None,
) -> tuple:
    """共享组装内核暂存目录、注入版 train_moss.py 源码与公共元数据字段。"""
    user = (username or "").strip()
    if not user:
        raise ValueError("Kaggle 用户名不能为空。")
    slug = sanitize_kaggle_slug(kernel_slug, prefix="moss-train")

    root = Path(stage_root) if stage_root else (PROJECT_ROOT / "workspace" / "kaggle_kernel_stage")
    root.mkdir(parents=True, exist_ok=True)
    for old_sub in list(root.iterdir()):
        if old_sub.is_dir() and old_sub.name != slug:
            shutil.rmtree(old_sub, ignore_errors=True)
    stage_dir = root / slug
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    payload = build_cloud_config_payload(cleaned_params, cloud_opts)
    injected_code = inject_cloud_config_into_train_moss(PROJECT_ROOT / "train_moss.py", payload)

    acc = (cloud_opts.get("accelerator") or "NvidiaTeslaT4").strip()
    ds_source = str(cloud_opts.get("dataset_source", "local") or "local").strip().lower()
    ds_sources = [] if ds_source in ("hf", "url", "r2") else ([dataset_id.strip()] if (dataset_id and "/" in dataset_id) else [])
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "train_moss.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ds_sources,
        "competition_sources": [],
        "kernel_sources": [],
    }
    if acc in ("NvidiaTeslaT4", "NvidiaTeslaP100"):
        meta["machine_shape"] = acc
    return stage_dir, user, slug, payload, injected_code, meta


def stage_kernel_bundle(
    cleaned_params: dict,
    cloud_opts: dict,
    username: str,
    dataset_id: str,
    kernel_slug: str,
    stage_root: Path = None,
) -> tuple:
    """组装带有注入超参数与 kernel-metadata.json 的 Kaggle 云端脚本暂存包。"""
    stage_dir, user, slug, payload, injected_code, meta = _prepare_kernel_stage_dir(
        cleaned_params, cloud_opts, username, dataset_id, kernel_slug, stage_root=stage_root
    )
    (stage_dir / "train_moss.py").write_text(injected_code, encoding="utf-8")
    (stage_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stage_dir, {"kernel_id": f"{user}/{slug}", "kernel_slug": slug, "payload": payload, "metadata": meta}


def stage_interactive_notebook_bundle(
    cleaned_params: dict,
    cloud_opts: dict,
    username: str,
    dataset_id: str,
    kernel_slug: str,
    stage_root: Path = None,
) -> tuple:
    """组装面向 `--no-run` 快速保存的自包含交互式 `.ipynb` 暂存包，可在 Kaggle 编辑器内点击 Edit 后直接 Run All 绕过后台批处理队列。"""
    stage_dir, user, slug, payload, injected_code, meta = _prepare_kernel_stage_dir(
        cleaned_params, cloud_opts, username, dataset_id, kernel_slug, stage_root=stage_root
    )
    lora_name = payload["lora_name"]
    ds_source = str(cloud_opts.get("dataset_source", "local") or "local").strip().lower()
    payload_cell = f"%%writefile /kaggle/working/train_moss.py\n{injected_code}"
    summary_line = (
        f"lora={lora_name} | steps={payload['steps']} | lr={payload['lr']} "
        f"| res={payload['resolution']} | algo={payload['algo']} | src={ds_source}"
    )
    runner_cell = (
        "import pathlib, subprocess, sys\n\n"
        "script_path = pathlib.Path('/kaggle/working/train_moss.py')\n"
        "if not script_path.is_file():\n"
        "    raise FileNotFoundError('请点击顶部 Run All（或先运行上一个 %%writefile 隐藏单元）以生成 /kaggle/working/train_moss.py')\n\n"
        f"print('>>> 配置摘要: ' + {json.dumps(summary_line, ensure_ascii=False)}, flush=True)\n"
        "print(f'>>> Starting interactive MOSS LoRA training ({script_path})...', flush=True)\n"
        "proc = subprocess.Popen(\n"
        "    [sys.executable, '-u', str(script_path)],\n"
        "    stdout=subprocess.PIPE,\n"
        "    stderr=subprocess.STDOUT,\n"
        "    text=True,\n"
        "    encoding='utf-8',\n"
        "    errors='replace',\n"
        "    bufsize=1,\n"
        ")\n"
        "if proc.stdout is not None:\n"
        "    for line in proc.stdout:\n"
        "        print(line, end='', flush=True)\n"
        "rc = proc.wait()\n"
        "if rc != 0:\n"
        "    raise RuntimeError(f'train_moss.py exited with code {rc}')\n"
    )
    download_cell = (
        "from pathlib import Path\n"
        "from IPython.display import FileLink, display\n\n"
        f"zip_path = Path('/kaggle/working/{lora_name}_lora_results.zip')\n"
        "if zip_path.is_file():\n"
        "    print(f'Archive ready: {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)')\n"
        "    display(FileLink(zip_path.name))\n"
        "else:\n"
        "    print('ZIP archive not found yet; check training logs above.')\n"
    )
    nb_obj = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# MOSS-Image LoRA Interactive Runner (`{lora_name}`)\n\n"
                    "0. 如果页面右上角显示黑色 **`Edit`** 按钮且右侧显示 `Runtime: 9s` / `In [ ]:`，说明快速保存已成功（尚未开始训练，并非出错）——请先点右上角 **`Edit`** 进入编辑器！\n"
                    "1. 先确认左下角 **`View Active Events`** 中没有其它占用 GPU 的运行/排队会话；\n"
                    "2. 确认右侧 **Settings -> Accelerator = GPU T4 x2** 且 **Internet = ON**；\n"
                    "3. 点击顶部 **`Run All`** 直接在交互式 GPU 池开机开炼（无需进入后台批处理排队）。\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"_kg_hide-input": True, "jupyter": {"source_hidden": True}, "tags": ["hide-input"]},
                "outputs": [],
                "source": [payload_cell],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [runner_cell],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [download_cell],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    nb_file = stage_dir / "train_moss_interactive.ipynb"
    nb_file.write_text(json.dumps(nb_obj, indent=1, ensure_ascii=False), encoding="utf-8")

    meta["code_file"] = nb_file.name
    meta["kernel_type"] = "notebook"
    (stage_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stage_dir, {"kernel_id": f"{user}/{slug}", "kernel_slug": slug, "payload": payload, "metadata": meta}


_PUSH_ERROR_MARKERS = (
    "400 client error",
    "401 client error",
    "403 client error",
    "404 client error",
    "409 client error",
    "500 server error",
    "traceback (most recent call last)",
    "valueerror:",
    "connectionreseterror",
    "connection aborted",
)

_PUSH_TRANSIENT_MARKERS = (
    "connectionreseterror",
    "connection aborted",
    "10054",
    "remotedisconnected",
    "sslerror",
    "eof occurred in violation of protocol",
    "read timed out",
    "connecttimeout",
    "max retries exceeded",
    "connectionerror",
    "proxyerror",
    "broken pipe",
)


def _read_raw_kaggle_auth(kaggle_json_path: Path = None) -> tuple:
    """内部读取 Kaggle 认证信息 `(username, key, token, proxy)`，严禁直接日志输出或暴露给界面。"""
    path = Path(kaggle_json_path) if kaggle_json_path else get_kaggle_json_path()
    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    proxy = ""
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                user = str(data.get("username", "") or "").strip() or user
                key = str(data.get("key", "") or "").strip() or key
                token = str(data.get("token", "") or "").strip() or token
                proxy = str(data.get("proxy", "") or "").strip()
        except Exception:
            pass
    return user, key, token, proxy


def _meta_bool(val, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def _push_kaggle_kernel_via_curl(
    stage_dir: Path,
    accelerator: str = "NvidiaTeslaT4",
    no_run: bool = False,
    kaggle_json_path: Path = None,
    on_log=None,
) -> tuple:
    """当 Kaggle CLI 因代理/TLS 重置 (WinError 10054) 失败时，回退通过系统原生 curl 直连 SaveKernel API。"""
    stage = Path(stage_dir)
    meta_file = stage / "kernel-metadata.json"
    if not meta_file.is_file():
        return False, "https://www.kaggle.com/code", "curl 直推失败：缺少 kernel-metadata.json。"

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as err:
        return False, "https://www.kaggle.com/code", f"curl 直推失败：解析 kernel-metadata.json 异常 ({err})。"

    kernel_id = str(meta.get("id", "") or "").strip()
    kernel_url = (
        (f"https://www.kaggle.com/code/{kernel_id}/edit" if no_run else f"https://www.kaggle.com/code/{kernel_id}")
        if kernel_id
        else "https://www.kaggle.com/code"
    )
    code_name = str(meta.get("code_file", "") or "").strip()
    code_file = stage / code_name if code_name else None
    if not code_file or not code_file.is_file():
        return False, kernel_url, f"curl 直推失败：未找到代码文件 {code_name or '<empty>'}。"

    script_body = code_file.read_text(encoding="utf-8")
    kernel_type = str(meta.get("kernel_type", "script") or "script").strip().lower()
    if kernel_type == "notebook":
        nb_json = json.loads(script_body)
        for cell in nb_json.get("cells") or []:
            if cell.get("cell_type") == "code" and "outputs" in cell:
                cell["outputs"] = []
            if isinstance(cell.get("source"), list):
                cell["source"] = "".join(cell["source"])
        script_body = json.dumps(nb_json, ensure_ascii=False)

    user, key, token, proxy_cfg = _read_raw_kaggle_auth(kaggle_json_path=kaggle_json_path)
    if not token and not (user and key):
        return False, kernel_url, "无法回退 curl 直推：缺少 Kaggle API 凭证（请先导入或保存 kaggle.json）。"

    curl_exe = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_exe:
        return False, kernel_url, "无法回退 curl 直推：系统未找到 curl.exe。"

    req_payload = {
        "slug": kernel_id,
        "newTitle": str(meta.get("title") or kernel_id.split("/")[-1]),
        "text": script_body,
        "language": str(meta.get("language", "python") or "python"),
        "kernelType": kernel_type,
        "isPrivate": _meta_bool(meta.get("is_private"), True),
        "enableGpu": _meta_bool(meta.get("enable_gpu"), True),
        "enableTpu": _meta_bool(meta.get("enable_tpu"), False),
        "enableInternet": _meta_bool(meta.get("enable_internet"), True),
        "datasetDataSources": list(meta.get("dataset_sources") or []),
        "competitionDataSources": list(meta.get("competition_sources") or []),
        "kernelDataSources": list(meta.get("kernel_sources") or []),
    }
    shape = accelerator if accelerator in ("NvidiaTeslaT4", "NvidiaTeslaP100") else str(meta.get("machine_shape", "") or "").strip()
    if shape and shape.lower() != "none":
        req_payload["machineShape"] = shape
    if no_run:
        req_payload["kernelExecutionType"] = "QUICK_SAVE"

    cfg_lines = []
    if user and key:
        safe_u = user.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")
        safe_k = key.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")
        cfg_lines.append(f'user = "{safe_u}:{safe_k}"')
    elif token:
        safe_t = token.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")
        cfg_lines.append(f'header = "Authorization: Bearer {safe_t}"')
    if proxy_cfg:
        safe_p = proxy_cfg.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")
        cfg_lines.append(f'proxy = "{safe_p}"')

    tmp_json_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_save_kernel.json",
            delete=False,
            encoding="utf-8",
        ) as tf:
            json.dump(req_payload, tf, ensure_ascii=False)
            tmp_json_path = Path(tf.name)

        cmd = [
            curl_exe,
            "-sS",
            "--connect-timeout", "20",
            "--max-time", "90",
            "--retry", "2",
            "--retry-delay", "2",
            "-K", "-",
            "-X", "POST",
            "-H", "Content-Type: application/json",
            "-H", "User-Agent: kaggle-api/2.2.5",
            "--data-binary", f"@{tmp_json_path}",
            "-w", "\n__HTTP_CODE__:%{http_code}\n",
            "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel",
        ]
        res = subprocess.run(
            cmd,
            input="\n".join(cfg_lines) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            creationflags=NO_WINDOW,
        )
    except Exception as err:
        return False, kernel_url, f"curl 直推异常：{err}"
    finally:
        if tmp_json_path is not None:
            try:
                tmp_json_path.unlink(missing_ok=True)
            except Exception:
                pass

    stdout_txt = (res.stdout or "").strip()
    stderr_txt = (res.stderr or "").strip()
    body_text = stdout_txt
    http_code = 0
    marker = "__HTTP_CODE__:"
    if marker in stdout_txt:
        body_part, code_part = stdout_txt.rsplit(marker, 1)
        body_text = body_part.strip()
        try:
            http_code = int(code_part.strip().splitlines()[0])
        except Exception:
            http_code = 0

    data = {}
    try:
        parsed = json.loads(body_text) if body_text else {}
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        data = {}

    if res.returncode == 0 and 200 <= http_code < 300 and not data.get("error"):
        resp_url = str(data.get("url", "") or "").strip().rstrip(".,)")
        if resp_url:
            kernel_url = f"{resp_url}/edit" if (no_run and not resp_url.endswith("/edit")) else resp_url
        ver_num = data.get("versionNumber")
        ver_tag = f"v{ver_num}" if ver_num else "saved"
        if on_log:
            on_log(f"[Kaggle 推送] ✅ 已通过原生 curl 容灾通道提交成功 ({ver_tag}): {kernel_url}\n")
        ok_msg = f"已通过 curl 容灾保存交互式 Notebook ({ver_tag})：{kernel_id}" if no_run else f"已通过 curl 容灾提交云端任务 ({ver_tag})：{kernel_id}"
        return True, kernel_url, ok_msg

    detail = (
        str(data.get("error") or data.get("message") or "").strip()
        or body_text[-350:]
        or stderr_txt[-350:]
        or "未知错误"
    )
    return False, kernel_url, f"curl 直推失败 (HTTP {http_code}, Exit={res.returncode})：\n{detail}"


def push_kaggle_kernel(
    stage_dir: Path,
    accelerator: str = "NvidiaTeslaT4",
    cli_info: dict = None,
    kaggle_json_path: Path = None,
    on_log=None,
    no_run: bool = False,
    max_retries: int = 3,
) -> tuple:
    """将阶段化后的训练内核推送至 Kaggle 云端队列或 Quick-Save 为交互式 Notebook（含断链重试与 curl 容灾回退）。"""
    cli = cli_info if (cli_info and cli_info.get("found")) else find_kaggle_cli()
    if not cli.get("found"):
        return False, "", "未找到 Kaggle CLI，请先安装并配置凭证。"

    meta_file = Path(stage_dir) / "kernel-metadata.json"
    kernel_id = ""
    if meta_file.is_file():
        try:
            kernel_id = json.loads(meta_file.read_text(encoding="utf-8")).get("id", "")
        except Exception:
            pass
    kernel_url = (
        (f"https://www.kaggle.com/code/{kernel_id}/edit" if no_run else f"https://www.kaggle.com/code/{kernel_id}")
        if kernel_id
        else "https://www.kaggle.com/code"
    )

    cmd = [*cli["cmd"], "kernels", "push", "-p", str(stage_dir)]
    if accelerator in ("NvidiaTeslaT4", "NvidiaTeslaP100"):
        cmd.extend(["--accelerator", accelerator])
    if no_run:
        cmd.append("--no-run")

    attempts = max(1, int(max_retries or 1))
    last_rc = 1
    last_out = ""
    for attempt in range(1, attempts + 1):
        if on_log and attempt == 1:
            action_lbl = "正在 Quick-Save 为交互式 Notebook (免排队)" if no_run else "正在提交任务"
            on_log(f"[Kaggle 推送] {action_lbl} {kernel_id}...\n")
        rc, out = _stream_subprocess(cmd, Path(stage_dir), on_log=on_log, env=_build_kaggle_env(kaggle_json_path))
        last_rc = rc
        last_out = (out or "").strip()
        out_lower = last_out.lower()
        has_push_err = (
            rc != 0
            or "kernel push error" in out_lower
            or ("successfully" not in out_lower and any(marker in out_lower for marker in _PUSH_ERROR_MARKERS))
        )
        if not has_push_err:
            m_url = re.search(r"https://www\.kaggle\.com/code/[^\s\"'<>]+", last_out)
            if m_url:
                matched_url = m_url.group(0).rstrip(".,)")
                kernel_url = f"{matched_url}/edit" if (no_run and not matched_url.endswith("/edit")) else matched_url
            ok_msg = f"已免排队保存交互式 Notebook：{kernel_id}" if no_run else f"任务已提交云端：{kernel_id}"
            return True, kernel_url, ok_msg

        is_transient = any(marker in out_lower for marker in _PUSH_TRANSIENT_MARKERS)
        if not is_transient:
            m_err = re.search(r"(Kernel push error:[^\r\n]+)", last_out, re.IGNORECASE)
            detail = m_err.group(1).strip() if m_err else last_out[-400:]
            return False, kernel_url, f"推送任务失败 (Exit={rc})：\n{detail}"

        if attempt < attempts:
            delay_sec = attempt * 2
            if on_log:
                on_log(f"[Kaggle 推送] ⚠️ 检测到网络/代理连接重置，{delay_sec} 秒后自动重试 ({attempt + 1}/{attempts})...\n")
            time.sleep(delay_sec)

    if on_log:
        on_log("[Kaggle 推送] 🔄 Kaggle CLI 连续断链，自动切换 Windows 原生 curl 直推 SaveKernel API...\n")
    ok_curl, url_curl, msg_curl = _push_kaggle_kernel_via_curl(
        Path(stage_dir),
        accelerator=accelerator,
        no_run=no_run,
        kaggle_json_path=kaggle_json_path,
        on_log=on_log,
    )
    if ok_curl:
        return True, url_curl, msg_curl
    return False, kernel_url, f"推送任务失败 (CLI Exit={last_rc} 且 curl 容灾未通)：\n{msg_curl}\n\n[CLI 原始输出]\n{last_out[-240:]}"


def _extract_kernel_failure_message(raw: str) -> str:
    """从 Kaggle CLI 状态输出提取 Failure message / failureMessage。"""
    text = raw or ""
    for pat in (
        r'Failure message:\s*"([^"]+)"',
        r'Failure message:\s*([^\r\n]+)',
        r'"failureMessage"\s*:\s*"([^"]+)"',
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            msg = m.group(1).strip().strip('"').strip()
            if msg:
                return msg[:240]
    return ""


_KAGGLE_KERNEL_FAILURE_PROBE = """
import sys
kernel_id = sys.argv[1]
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

api = KaggleApi()
api.authenticate()
owner, slug, ver_arg = api.parse_kernel_string(kernel_id)
if not ver_arg:
    with api.build_kaggle_client() as client:
        req = ApiGetKernelRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.get_kernel(req)
        meta = (resp.to_dict() or {}).get("metadata") or {}
        ver_num = meta.get("currentVersionNumber")
        ver_arg = str(ver_num) if ver_num else None
target = f"{owner}/{slug}/{ver_arg}" if ver_arg else f"{owner}/{slug}"
st = api.kernels_status(target)
msg = getattr(st, "failure_message", "") or getattr(st, "failureMessage", "") or ""
if msg:
    print(str(msg).strip(), flush=True)
""".strip()


def _probe_kernel_failure_message(kernel_id: str, cli: dict, env: dict) -> str:
    """在状态为 CANCEL/ERROR 且 CLI 未输出 Failure message 时，经官方 SDK 查询当前版本号对应的 failure_message。"""
    kaggle_py = _resolve_kaggle_python_exe(cli)
    if not kaggle_py:
        return ""
    try:
        res = subprocess.run(
            [kaggle_py, "-u", "-c", _KAGGLE_KERNEL_FAILURE_PROBE, kernel_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
            env=env,
            creationflags=NO_WINDOW,
        )
        if res.returncode == 0:
            lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
            return lines[-1][:240] if lines else ""
    except Exception:
        pass
    return ""


_KAGGLE_QUEUE_DIAG_PROBE = """
import json, sys
target_id = (sys.argv[1] or "").strip().lower()
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

api = KaggleApi()
api.authenticate()
active_commits = []
interactive_gpu_drafts = []
with api.build_kaggle_client() as client:
    for item in (api.kernels_list(mine=True, page_size=8) or []):
        ref = str(getattr(item, "ref", "") or "").strip()
        if not ref or ref.lower() == target_id or "/" not in ref:
            continue
        owner, slug = ref.split("/", 1)
        try:
            st = api.kernels_status(ref)
            st_name = str(getattr(st, "status", "") or "").upper()
            if "RUNNING" in st_name or "QUEUED" in st_name:
                active_commits.append(f"{ref} ({st_name.split('.')[-1]})")
            continue
        except Exception:
            pass
        try:
            req = ApiGetKernelRequest()
            req.user_name = owner
            req.kernel_slug = slug
            meta = (client.kernels.kernels_api_client.get_kernel(req).to_dict() or {}).get("metadata") or {}
            if meta.get("enableGpu") and not meta.get("currentVersionNumber"):
                interactive_gpu_drafts.append(ref)
        except Exception:
            pass
print(json.dumps({"active_commits": active_commits, "interactive_gpu_drafts": interactive_gpu_drafts}), flush=True)
""".strip()


def diagnose_kaggle_queue_blockers(
    kernel_id: str,
    cli_info: dict = None,
    kaggle_json_path: Path = None,
) -> dict:
    """探测同账号下是否存在占用 GPU 槽位的其它运行/排队后台任务或交互式 GPU 草稿本。"""
    cli = cli_info if (cli_info and cli_info.get("found")) else find_kaggle_cli()
    if not cli.get("found"):
        return {"ok": False, "active_commits": [], "interactive_gpu_drafts": [], "hint": ""}

    kaggle_py = _resolve_kaggle_python_exe(cli)
    if not kaggle_py:
        return {"ok": False, "active_commits": [], "interactive_gpu_drafts": [], "hint": ""}

    env = _build_kaggle_env(kaggle_json_path=kaggle_json_path)
    try:
        res = subprocess.run(
            [kaggle_py, "-u", "-c", _KAGGLE_QUEUE_DIAG_PROBE, kernel_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12,
            env=env,
            creationflags=NO_WINDOW,
        )
        if res.returncode != 0:
            return {"ok": False, "active_commits": [], "interactive_gpu_drafts": [], "hint": ""}
        lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip().startswith("{")]
        data = json.loads(lines[-1]) if lines else {}
        active = list(data.get("active_commits") or [])
        drafts = list(data.get("interactive_gpu_drafts") or [])
        hints = []
        if active:
            hints.append(f"发现其它运行/排队中的后台任务: {', '.join(active[:2])}")
        if drafts:
            hints.append(f"发现启用 GPU 的交互式草稿本: {', '.join(drafts[:2])}（请先在 Kaggle Active Events 停掉旧会话或改用【⚡ 免排队交互式推送】）")
        return {"ok": True, "active_commits": active, "interactive_gpu_drafts": drafts, "hint": "；".join(hints)}
    except Exception:
        return {"ok": False, "active_commits": [], "interactive_gpu_drafts": [], "hint": ""}


def fetch_kaggle_kernel_status(kernel_id: str, cli_info: dict = None, kaggle_json_path: Path = None) -> dict:
    """查询 Kaggle 云端任务实时运行状态并映射为统一状态码、失败原因与中文文案。"""
    cli = cli_info if (cli_info and cli_info.get("found")) else find_kaggle_cli()
    if not cli.get("found"):
        return {"ok": False, "state": "unknown", "failure_message": "", "queue_hint": "", "label": "⚪ 未检测到 Kaggle CLI", "raw": ""}

    env = _build_kaggle_env(kaggle_json_path=kaggle_json_path)
    try:
        res = subprocess.run(
            [*cli["cmd"], "kernels", "status", kernel_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=env,
            creationflags=NO_WINDOW,
        )
        raw = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
        failure_msg = _extract_kernel_failure_message(raw)
        lower = raw.lower()
        if res.returncode != 0:
            if "404" in lower or "not found" in lower:
                return {"ok": False, "state": "not_found", "failure_message": failure_msg, "queue_hint": "", "label": f"⚪ 云端尚无任务记录: {kernel_id}", "raw": raw}
            return {"ok": False, "state": "error", "failure_message": failure_msg, "queue_hint": "", "label": f"🟠 状态查询异常: {raw[:160]}", "raw": raw}

        if "complete" in lower:
            return {"ok": True, "state": "complete", "failure_message": failure_msg, "queue_hint": "", "label": "🟢 云端训练已完成！请点击【一键拉回成品权重】", "raw": raw}
        if "running" in lower:
            return {"ok": True, "state": "running", "failure_message": failure_msg, "queue_hint": "", "label": "🔵 云端 GPU 正在全速训练中 (RUNNING)...", "raw": raw}
        if "queued" in lower:
            diag = diagnose_kaggle_queue_blockers(kernel_id, cli_info=cli, kaggle_json_path=kaggle_json_path)
            qhint = str(diag.get("hint", "") or "").strip()
            lbl = f"🟡 已进入云端算力排队队列 (QUEUED · {qhint})" if qhint else "🟡 已进入云端算力排队队列 (QUEUED)..."
            return {"ok": True, "state": "queued", "failure_message": failure_msg, "queue_hint": qhint, "label": lbl, "raw": raw}
        if "error" in lower:
            if not failure_msg:
                failure_msg = _probe_kernel_failure_message(kernel_id, cli, env)
            label = (
                f"🔴 云端执行出错 (ERROR · {failure_msg})，请点击【拉取云端日志】或【打开任务页】排查"
                if failure_msg
                else "🔴 云端执行出错 (ERROR)，请点击右侧【打开任务页】排查"
            )
            return {"ok": True, "state": "error", "failure_message": failure_msg, "queue_hint": "", "label": label, "raw": raw}
        if "cancel" in lower:
            if not failure_msg:
                failure_msg = _probe_kernel_failure_message(kernel_id, cli, env)
            if failure_msg:
                if "backend" in failure_msg.lower():
                    label = f"🔴 云端任务被 Kaggle 后端中止 (CANCELLED · {failure_msg}，非本地刷新所致，请换非 x0.at 源或换新 Slug 重推)"
                else:
                    label = f"⚪ 云端任务已取消 (CANCELLED · {failure_msg}，非本地刷新所致)"
            else:
                label = "⚪ 云端任务已取消 (CANCELLED，云端会话已停止，非本地刷新所致)"
            return {"ok": True, "state": "cancel", "failure_message": failure_msg, "queue_hint": "", "label": label, "raw": raw}
        return {"ok": True, "state": "unknown", "failure_message": failure_msg, "queue_hint": "", "label": f"❔ 当前状态: {raw}", "raw": raw}
    except Exception as err:
        return {"ok": False, "state": "error", "failure_message": "", "queue_hint": "", "label": f"🟠 查询状态超时或异常: {err}", "raw": str(err)}


def fetch_kaggle_kernel_logs(
    kernel_id: str,
    cli_info: dict = None,
    kaggle_json_path: Path = None,
    tail_lines: int = 200,
) -> dict:
    """拉取指定 Kaggle 云端任务的实时或最新执行日志尾部，返回结构化结果。"""
    cli = cli_info if (cli_info and cli_info.get("found")) else find_kaggle_cli()
    if not cli.get("found"):
        return {"ok": False, "text": "", "tail": "", "line_count": 0, "error": "未检测到 Kaggle CLI。"}

    try:
        res = subprocess.run(
            [*cli["cmd"], "kernels", "logs", kernel_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=_build_kaggle_env(kaggle_json_path=kaggle_json_path),
            creationflags=NO_WINDOW,
        )
        raw = f"{res.stdout or ''}\n{res.stderr or ''}".strip()
        lower = raw.lower()
        if res.returncode != 0:
            if "404" in lower or "not found" in lower:
                return {"ok": False, "text": raw, "tail": "", "line_count": 0, "error": f"云端尚无任务记录: {kernel_id}"}
            if "queued" in lower or "no logs" in lower:
                return {"ok": False, "text": raw, "tail": "", "line_count": 0, "error": "任务仍在排队或容器刚启动，暂未产出运行日志，请稍后再试。"}
            return {"ok": False, "text": raw, "tail": "", "line_count": 0, "error": f"拉取日志失败: {raw[:240]}"}

        lines = raw.splitlines()
        limit = max(20, int(tail_lines))
        tail = "\n".join(lines[-limit:]) if len(lines) > limit else raw
        return {"ok": True, "text": raw, "tail": tail, "line_count": len(lines), "error": ""}
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "", "tail": "", "line_count": 0, "error": "拉取云端日志超时（30 秒），请稍后重试。"}
    except Exception as err:
        return {"ok": False, "text": "", "tail": "", "line_count": 0, "error": f"拉取日志异常: {err}"}


def delete_kaggle_kernel(
    kernel_id: str,
    cli_info: dict = None,
    kaggle_json_path: Path = None,
    on_log=None,
) -> tuple:
    """强制删除指定 Kaggle 云端任务以中止死锁队列并释放配额（破坏性操作：将清空远端输出与历史版本）。"""
    cli = cli_info if (cli_info and cli_info.get("found")) else find_kaggle_cli()
    if not cli.get("found"):
        return False, "未找到 Kaggle CLI。"

    cmd = [*cli["cmd"], "kernels", "delete", kernel_id, "--yes"]
    if on_log:
        on_log(f"[Kaggle 强删] 正在彻底删除云端任务 {kernel_id}...\n")
    rc, out = _stream_subprocess(cmd, PROJECT_ROOT, on_log=on_log, env=_build_kaggle_env(kaggle_json_path))
    if rc == 0:
        return True, f"已从 Kaggle 彻底删除任务：{kernel_id}"
    lower = out.lower()
    if "404" in lower or "not found" in lower:
        return False, f"云端未找到任务：{kernel_id}（可能已被删除或尚未成功创建）。"
    return False, f"删除云端任务失败 (Exit={rc})：\n{out[-300:]}"


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> list:
    """防目录穿越（Zip-Slip）安全解压归档文件并返回已解压路径列表。"""
    extracted = []
    dest_root = target_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_dest = (dest_root / member.filename).resolve()
            try:
                if os.path.commonpath([str(dest_root), str(member_dest)]) != str(dest_root):
                    continue
            except ValueError:
                continue
            member_dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src_f, open(member_dest, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)
            extracted.append(member_dest)
    return extracted


def _extract_step_num_from_path(path_obj: Path) -> int:
    """从 checkpoints/step_XXXXXXXX/ 路径或同级 state.json 解析步数编号。"""
    for idx, part in enumerate(path_obj.parts):
        m = re.match(r"^step_(\d+)$", part, re.IGNORECASE)
        if m:
            step_dir = Path(*path_obj.parts[: idx + 1])
            st_file = step_dir / "state.json"
            if st_file.is_file():
                try:
                    return int(json.loads(st_file.read_text(encoding="utf-8")).get("step", m.group(1)))
                except Exception:
                    pass
            return int(m.group(1))
    return 0


def _is_under_dir(path_obj: Path, parent_dir: Path) -> bool:
    try:
        return os.path.commonpath([str(path_obj.resolve()), str(parent_dir.resolve())]) == str(parent_dir.resolve())
    except ValueError:
        return False


def _clean_partial_and_empty_downloads(dl_path: Path, protect_dir: Path) -> int:
    """清除 *.kaggle-partial 断点标记、对应的半成品目标文件及 0 字节损坏文件。"""
    cleaned = 0
    for partial in list(dl_path.rglob("*.kaggle-partial")):
        if _is_under_dir(partial, protect_dir):
            continue
        target_name = partial.name[: -len(".kaggle-partial")]
        target_file = partial.with_name(target_name)
        if target_file.is_file() and not _is_under_dir(target_file, protect_dir):
            try:
                target_file.unlink(missing_ok=True)
                cleaned += 1
            except OSError:
                pass
        try:
            partial.unlink(missing_ok=True)
            cleaned += 1
        except OSError:
            pass

    for fp in list(dl_path.rglob("*")):
        if _is_under_dir(fp, protect_dir) or not fp.is_file():
            continue
        try:
            if fp.stat().st_size <= 0:
                fp.unlink(missing_ok=True)
                cleaned += 1
        except OSError:
            pass
    return cleaned


def _prune_kaggle_download_cache(dl_path: Path, protect_dir: Path, on_log=None) -> int:
    """在成品完整归档至 output/<lora_name>/ 后，自动精简 dl_path 下的冗余 ZIP、重复 output 与旧检查点 optim.pt。"""
    freed_bytes = 0

    def _rm_file(p: Path):
        nonlocal freed_bytes
        if not p.is_file() or _is_under_dir(p, protect_dir):
            return
        try:
            freed_bytes += p.stat().st_size
            p.unlink(missing_ok=True)
        except OSError:
            pass

    def _rm_dir(d: Path):
        nonlocal freed_bytes
        if not d.is_dir() or _is_under_dir(protect_dir, d):
            return
        for f in d.rglob("*"):
            if f.is_file():
                try:
                    freed_bytes += f.stat().st_size
                except OSError:
                    pass
        shutil.rmtree(d, ignore_errors=True)

    for zf in list(dl_path.rglob("*.zip")):
        _rm_file(zf)

    out_dup = dl_path / "output"
    if out_dup.is_dir():
        _rm_dir(out_dup)

    runs_root = dl_path / "runs"
    if runs_root.is_dir():
        for exp_dir in list(runs_root.rglob("export")):
            if exp_dir.is_dir():
                _rm_dir(exp_dir)
        for ckpts_root in list(runs_root.rglob("checkpoints")):
            if not ckpts_root.is_dir():
                continue
            step_dirs = [p for p in ckpts_root.iterdir() if p.is_dir() and re.match(r"^step_\d+$", p.name, re.IGNORECASE)]
            step_dirs.sort(key=lambda d: (_extract_step_num_from_path(d / "x"), d.stat().st_mtime))
            if step_dirs:
                latest_step = step_dirs[-1]
                for old_step in step_dirs[:-1]:
                    _rm_dir(old_step)
                optim_f = latest_step / "optim.pt"
                if optim_f.is_file():
                    _rm_file(optim_f)

    if on_log and freed_bytes > 0:
        on_log(f"[自动精简] 已清理冗余云端过程副本，释放 {freed_bytes / (1024**2):.1f} MB（已保留最新步数断点 state.json + trainable.safetensors）。\n")
    return freed_bytes


def extract_kaggle_outputs_from_dir(
    download_dir: Path,
    lora_name: str,
    output_dir: str,
    comfy_models_dir: str = "",
    copy_to_comfy: bool = False,
    on_log=None,
    prune_download_dir: bool = True,
) -> dict:
    """将下载到本地的 Kaggle 输出目录抗断流解压、按最新步数整理至 output/<lora_name>/，按需同步 ComfyUI 并检索 metrics.jsonl。"""
    dl_path = Path(download_dir)
    out_root = Path(output_dir) if output_dir else (PROJECT_ROOT / "output")
    lora_out_dir = out_root / lora_name
    lora_out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_archive_dir = lora_out_dir / "checkpoints"

    if dl_path.exists():
        _clean_partial_and_empty_downloads(dl_path, lora_out_dir)

    extracted_files = []
    # 1. 先整理散落文件（严格过滤 0 字节，并按 (is_in_output_dir, step_num, mtime) 升序让最新步数覆盖旧步数）
    main_candidates = []
    for extra in sorted(dl_path.rglob("*")) if dl_path.exists() else []:
        if not extra.is_file() or _is_under_dir(extra, lora_out_dir):
            continue
        lname = extra.name.lower()
        if lname.endswith(".kaggle-partial") or lname.endswith(".zip"):
            continue
        if "moss_image" in lname or "qwen" in lname or "flux2" in lname:
            continue
        try:
            fsize = extra.stat().st_size
            fmtime = extra.stat().st_mtime
        except OSError:
            continue
        if fsize <= 0:
            continue

        rel_parts = extra.relative_to(dl_path).parts
        rel_lower = [p.lower() for p in rel_parts]
        in_checkpoints_tree = "checkpoints" in rel_lower

        if in_checkpoints_tree and extra.suffix.lower() == ".safetensors":
            if "output" in rel_lower and ("_step" in lname or "-step" in lname):
                ckpt_archive_dir.mkdir(parents=True, exist_ok=True)
                dst_ck = ckpt_archive_dir / extra.name
                shutil.copy2(extra, dst_ck)
                extracted_files.append(dst_ck)
                continue
            step_n = _extract_step_num_from_path(extra)
            if step_n > 0 and ("-comfy" in lname or extra.stem.lower().endswith("_comfy")):
                ckpt_archive_dir.mkdir(parents=True, exist_ok=True)
                dst_ck = ckpt_archive_dir / f"{lora_name}_step{step_n:06d}-comfy.safetensors"
                shutil.copy2(extra, dst_ck)
                extracted_files.append(dst_ck)
            elif step_n > 0 and ("-lycoris" in lname or extra.stem.lower().endswith("_lycoris")):
                ckpt_archive_dir.mkdir(parents=True, exist_ok=True)
                dst_ck = ckpt_archive_dir / f"{lora_name}_step{step_n:06d}-lycoris.safetensors"
                shutil.copy2(extra, dst_ck)
                extracted_files.append(dst_ck)

        if extra.suffix.lower() == ".safetensors" or lname in ("training_summary.json", "metrics.jsonl"):
            if lname == "trainable.safetensors":
                dst_name = f"{lora_name}-trainable.safetensors"
            elif "-comfy" in lname or extra.stem.lower().endswith("_comfy"):
                dst_name = f"{lora_name}-comfy.safetensors"
            elif "-lycoris" in lname or extra.stem.lower().endswith("_lycoris"):
                dst_name = f"{lora_name}-lycoris.safetensors"
            else:
                dst_name = extra.name
            is_in_output_dir = 1 if (rel_lower and rel_lower[0] == "output") else 0
            step_num = _extract_step_num_from_path(extra)
            main_candidates.append((is_in_output_dir, step_num, fmtime, extra, dst_name))

    main_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for _in_out, _step, _mt, src_file, dst_name in main_candidates:
        dst = lora_out_dir / dst_name
        shutil.copy2(src_file, dst)
        extracted_files.append(dst)

    # 2. 最后解压合法非空 ZIP 成品包，确保最新聚合包覆盖任何残余旧步数文件
    for zfile in sorted(dl_path.rglob("*.zip")) if dl_path.exists() else []:
        try:
            if zfile.stat().st_size <= 22 or not zipfile.is_zipfile(zfile):
                continue
        except OSError:
            continue
        if on_log:
            on_log(f"[解压产物] 正在解包 {zfile.name} -> {lora_out_dir}\n")
        extracted_files.extend(_safe_extract_zip(zfile, lora_out_dir))

    comfy_synced = []
    if copy_to_comfy and comfy_models_dir:
        comfy_base = Path(comfy_models_dir)
        if comfy_base.is_dir():
            loras_root = comfy_base / "models" / "loras" if (comfy_base / "models").is_dir() else comfy_base / "loras"
            target_comfy_dir = loras_root / lora_name
            target_comfy_dir.mkdir(parents=True, exist_ok=True)
            for sf in sorted(lora_out_dir.glob("*.safetensors")):
                sl = sf.name.lower()
                if sl.endswith("-trainable.safetensors") or sl == "ema.safetensors":
                    continue
                if "-comfy" in sl or "_comfy" in sl or "-lycoris" in sl or "_lycoris" in sl:
                    dst_c = target_comfy_dir / sf.name
                    shutil.copy2(sf, dst_c)
                    comfy_synced.append(str(dst_c))

    metrics_candidates = list(lora_out_dir.rglob("metrics.jsonl"))
    metrics_path = str(metrics_candidates[0]) if metrics_candidates else ""
    safetensors_list = [str(p) for p in sorted(lora_out_dir.rglob("*.safetensors")) if p.is_file() and p.stat().st_size > 0]

    ok_flag = len(safetensors_list) > 0
    if ok_flag and prune_download_dir and dl_path.exists():
        _prune_kaggle_download_cache(dl_path, lora_out_dir, on_log=on_log)

    return {
        "ok": ok_flag,
        "output_dir": str(lora_out_dir),
        "safetensors": safetensors_list,
        "comfy_synced": comfy_synced,
        "metrics_path": metrics_path,
        "file_count": len(extracted_files),
    }


def pull_and_extract_kaggle_outputs(
    kernel_id: str,
    lora_name: str,
    output_dir: str,
    comfy_models_dir: str = "",
    copy_to_comfy: bool = False,
    cli_info: dict = None,
    kaggle_json_path: Path = None,
    on_log=None,
) -> dict:
    """调用 Kaggle CLI 拉回云端内核产出并自动解压、同步 ComfyUI 与定位 Loss 指标文件（支持断流自愈）。"""
    cli = cli_info if (cli_info and cli_info.get("found")) else find_kaggle_cli()
    if not cli.get("found"):
        return {"ok": False, "error": "未找到 Kaggle CLI。"}

    dl_dir = PROJECT_ROOT / "workspace" / "kaggle_outputs" / lora_name
    dl_dir.mkdir(parents=True, exist_ok=True)
    cmd = [*cli["cmd"], "kernels", "output", kernel_id, "-p", str(dl_dir), "--force"]
    if on_log:
        on_log(f"[Kaggle 拉回] 正在下载 {kernel_id} 云端产物到 {dl_dir}...\n")
    rc, out = _stream_subprocess(cmd, PROJECT_ROOT, on_log=on_log, env=_build_kaggle_env(kaggle_json_path))

    res = extract_kaggle_outputs_from_dir(
        dl_dir,
        lora_name=lora_name,
        output_dir=output_dir,
        comfy_models_dir=comfy_models_dir,
        copy_to_comfy=copy_to_comfy,
        on_log=on_log,
    )
    if rc != 0:
        if res.get("ok"):
            warn_msg = (
                f"⚠️ Kaggle CLI 下载过程出现部分中断 (Exit={rc})，但已从本地已落盘的完整成品包中成功恢复 "
                f"{len(res['safetensors'])} 个权重至 {res['output_dir']}。"
            )
            res["partial_warning"] = warn_msg
            if on_log:
                on_log(f"{warn_msg}\n")
            return res
        return {"ok": False, "error": f"下载云端输出失败 (Exit={rc})：\n{out[-300:]}"}

    if not res.get("ok"):
        res["error"] = "下载成功但未在云端输出中找到 LoRA 权重文件，请确认训练是否完整结束。"
    return res


def _build_configured_notebook_json(template_ipynb: Path, payload: dict, raw_params: dict) -> dict:
    """读取标准 Kaggle Notebook 模板，将参数控制单元自动替换为用户当前 GUI 真实配置。"""
    nb = json.loads(template_ipynb.read_text(encoding="utf-8"))
    subsets_str = ""
    if bool(raw_params.get("multi_subset", False)):
        plan = _build_subset_plan(raw_params)
        if plan:
            subsets_manifest = [
                {
                    "name": item["name"],
                    "path": item["rel_path"],
                    "repeats": item["repeats"],
                    "trigger": item["trigger"],
                    "enabled": True,
                }
                for item in plan
            ]
            subsets_str = json.dumps(subsets_manifest, ensure_ascii=False)

    replacements = {
        "LORA_NAME": repr(payload["lora_name"]),
        "TRIGGER_WORD": repr(payload["trigger_word"]),
        "REPEATS": repr(payload["repeats"]),
        "EPOCHS": repr(payload["epochs"]),
        "STEPS": repr(payload["steps"]),
        "RANK": repr(payload["rank"]),
        "ALPHA": repr(payload["alpha"]),
        "LR": repr(payload["lr"]),
        "GRAD_ACC": repr(payload["grad_acc"]),
        "TIMESTEP": repr(payload["timestep"]),
        "TIMESTEP_MU": repr(payload["timestep_mu"]),
        "ALGO": repr(payload["algo"]),
        "LOKR_FACTOR": repr(payload["lokr_factor"]),
        "LOKR_FULL_MATRIX": repr(payload["lokr_full_matrix"]),
        "LOKR_DECOMPOSE_BOTH": repr(payload["lokr_decompose_both"]),
        "TRAIN_NORMS": repr(payload["train_norms"]),
        "USE_DORA": repr(payload["dora"]),
        "RANK_DROPOUT": repr(payload["rank_dropout"]),
        "MODULE_DROPOUT": repr(payload["module_dropout"]),
        "INPUT_DROPOUT": repr(payload["input_dropout"]),
        "TARGET_PRESET": repr(payload["target_preset"]),
        "LORAPLUS_RATIO": repr(payload["loraplus_ratio"]),
        "OPTIMIZER": repr(payload["optimizer"]),
        "LR_SCHEDULER": repr(payload["lr_scheduler"]),
        "WARMUP_STEPS": repr(payload["warmup_steps"]),
        "WEIGHT_DECAY": repr(payload["weight_decay"]),
        "WD_RULES": repr(payload["wd_rules"]),
        "EMA": repr(payload["ema"]),
        "TEXT_DROP": repr(payload["text_drop"]),
        "FLIP": repr(payload["flip"]),
        "SEED": repr(payload["seed"]),
        "SAVE_FORMAT": repr(payload["save_format"]),
        "SUBSETS_JSON": repr(subsets_str),
        "MODEL_VARIANT": repr(payload["variant"]),
        "RESOLUTION": repr(payload["resolution"]),
        "BASE_QUANT": repr(payload["base_quant"]),
        "GPUS": repr(payload["gpus"]),
        "PARALLEL_MODE": repr(payload["parallel"]),
        "SAVE_EVERY": repr(payload["save_every"]),
        "HF_DATASET_REPO": repr(payload.get("hf_dataset_repo", "")),
        "HF_DATASET_SUBDIR": repr(payload.get("hf_dataset_subdir", "")),
        "HF_DATASET_REVISION": repr(payload.get("hf_dataset_revision", "main")),
        "DATASET_URL": repr(payload.get("dataset_url", "")),
        "MAX_RUN_MINUTES": repr(payload.get("max_run_minutes", 540)),
    }

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "LORA_NAME =" not in src:
            continue
        new_lines = []
        for line in src.splitlines(keepends=True):
            replaced = False
            for var_name, py_val in replacements.items():
                m = re.match(rf"^({var_name}\s*=\s*)([^#\n]*)(#.*)?(\r?\n?)$", line)
                if m:
                    comment = f"  {m.group(3)}" if m.group(3) else ""
                    new_lines.append(f"{m.group(1)}{py_val}{comment}{m.group(4)}")
                    replaced = True
                    break
            if not replaced:
                new_lines.append(line)
        cell["source"] = new_lines
        break
    return nb


def export_manual_upload_bundle(
    cleaned_params: dict,
    raw_params: dict,
    cloud_opts: dict,
    export_root: Path = None,
) -> dict:
    """
    零 API 备选方案：导出一键免配置网页手动上传包（支持本地图集 ZIP+预填 Notebook，或免本地传图的 HF 直拉 Notebook）。
    """
    lora_name = cleaned_params["lora_name"]
    safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", lora_name).strip("_") or "moss_lora"
    base_out = Path(export_root) if export_root else (PROJECT_ROOT / "workspace" / "kaggle_export" / safe_name)
    if base_out.exists():
        shutil.rmtree(base_out, ignore_errors=True)
    base_out.mkdir(parents=True, exist_ok=True)

    ds_source = str(cloud_opts.get("dataset_source", "local") or "local").strip().lower()
    hf_repo = normalize_hf_dataset_repo(str(cloud_opts.get("hf_dataset_repo", "")))
    ext_url = str(cloud_opts.get("dataset_url", "") or "").strip()
    use_hf_direct = (ds_source == "hf" and bool(hf_repo))
    use_url_direct = (ds_source in ("url", "r2") and bool(ext_url))

    payload = build_cloud_config_payload(cleaned_params, cloud_opts)
    nb_obj = _build_configured_notebook_json(
        PROJECT_ROOT / "notebooks" / "kaggle_moss_lora_training.ipynb",
        payload,
        raw_params,
    )
    nb_path = base_out / f"kaggle_moss_lora_{safe_name}.ipynb"
    nb_path.write_text(json.dumps(nb_obj, indent=1, ensure_ascii=False), encoding="utf-8")

    if use_hf_direct or use_url_direct:
        mode_tag = "hf_direct" if use_hf_direct else "url_direct"
        stats = {
            "mode": mode_tag,
            "hf_dataset_repo": hf_repo if use_hf_direct else "",
            "dataset_url": ext_url if use_url_direct else "",
            "images": 0,
            "captions": 0,
            "total_mb": 0.0,
        }
        zip_path_str = ""
        src_line = f" • 目标 HF 数据集：{hf_repo}\n" if use_hf_direct else f" • 外链 ZIP 直拉地址：{ext_url}\n"
        src_desc = "Hugging Face" if use_hf_direct else "外部 ZIP 直链"
        extra_note = (
            "   （若该 HF 数据集为 Private 私有仓库，请在 Kaggle 顶部 Add-ons -> Secrets 中添加只读 HF_TOKEN）\n"
            if use_hf_direct
            else ""
        )
        guide_text = (
            f"================================================================\n"
            f"OpenMOSS MOSS-Image Kaggle 免本地传图上传包 ({lora_name})\n"
            f"================================================================\n\n"
            f"本包已配置为【云端直接从{src_desc}拉取图集】，无需在 Kaggle 上传数据集压缩包：\n"
            f"{src_line}"
            f" • 超时自动存盘保护：{payload['max_run_minutes']} 分钟\n\n"
            f"【三步完成云端训练】：\n"
            f"1. 导入训练本：登录 https://www.kaggle.com -> 左侧【+ Create】->【New Notebook】->\n"
            f"   顶部 File -> Import Notebook，导入 `{nb_path.name}`（参数已全部自动填好）。\n"
            f"{extra_note}"
            f"2. 开启算力与联网：\n"
            f"   - 右侧/顶部 Settings -> Accelerator 选【GPU T4 x2】（双卡加速）；\n"
            f"   - Settings -> Internet 选【Internet on】。\n"
            f"3. 点击顶部【Run All】开始训练。完成后在右侧 Output 下载 `{lora_name}_lora_results.zip`，\n"
            f"   解压出 `{lora_name}-comfy.safetensors` 放入 ComfyUI/models/loras/ 即可出图！\n"
        )
    else:
        stage_tmp = base_out / "_dataset_tmp"
        stats = _populate_dataset_stage_dir(stage_tmp, raw_params)

        zip_path = base_out / f"{safe_name}_dataset.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(stage_tmp.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(stage_tmp).as_posix())
        shutil.rmtree(stage_tmp, ignore_errors=True)
        zip_path_str = str(zip_path)

        guide_text = (
            f"================================================================\n"
            f"OpenMOSS MOSS-Image Kaggle 免 API 手动上传操作包 ({lora_name})\n"
            f"================================================================\n\n"
            f"本目录已由本地控制台为您自动打包当前全部参数与图集（共 {stats['images']} 张图 / {stats['captions']} 份标注）。\n\n"
            f"【四步零代码完成云端训练】：\n"
            f"1. 上传图集：登录 https://www.kaggle.com -> 左侧【+ Create】->【New Dataset】，\n"
            f"   把 `{zip_path.name}` 拖进去，勾选 Private 创建私有数据集。\n"
            f"2. 导入训练本：左侧【+ Create】->【New Notebook】-> 顶部 File -> Import Notebook，\n"
            f"   导入同目录下的 `{nb_path.name}`（您的所有参数均已自动填好，无需改代码！）。\n"
            f"3. 开启免费算力与挂载数据：\n"
            f"   - 右侧/顶部 Settings -> Accelerator 选【GPU T4 x2】（双卡加速）；\n"
            f"   - Settings -> Internet 选【Internet on】；\n"
            f"   - File -> Add Data 挂载刚才第 1 步上传的数据集。\n"
            f"4. 点击顶部【Run All】开始训练。完成后在右侧 Output 下载 `{lora_name}_lora_results.zip`，\n"
            f"   解压出 `{lora_name}-comfy.safetensors` 放入 ComfyUI/models/loras/ 即可出图！\n"
        )

    guide_path = base_out / "README_操作步骤.txt"
    guide_path.write_text(guide_text, encoding="utf-8")

    return {
        "ok": True,
        "export_dir": str(base_out),
        "zip_path": zip_path_str,
        "notebook_path": str(nb_path),
        "guide_path": str(guide_path),
        "stats": stats,
    }
