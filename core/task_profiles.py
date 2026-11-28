"""Per-LoRA task profile storage, sanitization, local/cloud preset templating, diffing, and portable import/export."""

import copy
import datetime
import json
import os
import re
import secrets
from pathlib import Path
from typing import Optional

from .config_store import PROJECT_ROOT
from .kaggle_bridge import sanitize_external_dataset_url, sanitize_kaggle_slug
from .presets import ADVANCED_DEFAULTS, get_preset_by_id, resolve_preset_for_target

PROFILE_SCHEMA_VERSION = 1
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "task_profiles"

PROFILE_CATEGORIES = [
    ("all", "全部类型"),
    ("style", "🎨 画风"),
    ("character", "👤 角色"),
    ("multi_char", "👥 多角色"),
    ("multi_style", "🎭 多画风"),
    ("concept", "💡 概念/动作"),
    ("lokr", "💎 LoKr"),
    ("continue", "🔁 续炼/二训"),
    ("custom", "🧩 自定义"),
]
VALID_PROFILE_CATEGORIES = frozenset(k for k, _ in PROFILE_CATEGORIES if k != "all")

FORBIDDEN_SECRET_KEYS = frozenset(
    {
        "kaggle_api_key",
        "hf_token",
        "r2_secret_access_key",
        "r2_dataset_url",
        "api_key",
        "secret_access_key",
        "token",
        "password",
    }
)

MACHINE_ENV_KEYS = (
    "comfy_models_dir",
    "transformer_path",
    "text_encoder_path",
    "vae_path",
    "output_dir",
    "kaggle_username",
    "r2_account_id",
    "r2_bucket",
    "r2_access_key_id",
)

PORTABLE_STRIP_KEYS = frozenset(MACHINE_ENV_KEYS) | {
    "data_dir",
    "resume_path",
    "external_dataset_url",
}

_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

_BOOL_KEYS = frozenset(
    {
        "multi_subset",
        "copy_to_comfy_loras",
        "dora",
        "lokr_full_matrix",
        "lokr_decompose_both",
        "train_norms",
        "flip",
        "int8_compute",
        "hf_upload_local",
    }
)

_BASE_CONFIG_DEFAULTS = {
    "data_dir": "",
    "multi_subset": False,
    "subsets": [],
    "comfy_models_dir": "",
    "output_dir": "",
    "lora_name": "my_moss_lora",
    "trigger_word": "",
    "transformer_path": "",
    "text_encoder_path": "",
    "vae_path": "",
    "copy_to_comfy_loras": False,
    "precision": "bf16",
    "base_precision": "auto",
    "vram_preset": "auto",
    "resolution": "1024",
    "text_device": "cpu",
    "vae_device": "cuda",
    "algo": "lora",
    "dora": False,
    "target_preset": "attn-mlp",
    "loraplus_ratio": "1.0",
    "rank": "32",
    "alpha": "32.0",
    "lr": "1e-4",
    "repeats": "6",
    "epochs": "4",
    "steps": "600",
    "save_every": "100",
    "grad_acc": "1",
    "timestep_mu": "-0.85",
    "tune": "off",
    "train_mode": "fresh",
    "resume_path": "",
    "kaggle_username": "",
    "kaggle_dataset_slug": "",
    "kaggle_kernel_slug": "",
    "kaggle_accelerator": "NvidiaTeslaT4",
    "kaggle_variant": "heal",
    "kaggle_base_quant": "auto",
    "kaggle_dataset_source": "local",
    "external_dataset_url": "",
    "hf_dataset_repo": "",
    "hf_dataset_subdir": "",
    "hf_upload_local": True,
    "r2_account_id": "",
    "r2_bucket": "",
    "r2_prefix": "moss-datasets",
    "r2_access_key_id": "",
    "kaggle_max_run_min": "540",
    "target_compute": "local",
    "manual_img_count": "",
    **ADVANCED_DEFAULTS,
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_dir(profiles_dir: Optional[Path]) -> Path:
    return Path(profiles_dir).resolve() if profiles_dir is not None else DEFAULT_PROFILES_DIR.resolve()


def sanitize_profile_id(raw_name: str, fallback: str = "task") -> str:
    """将任意任务名转为安全的单文件 ID（保留字母/数字/中文/下划线/连字符，拦截 Windows 保留字）。"""
    text = str(raw_name or "").strip()
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._- ")
    if not cleaned:
        cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", str(fallback or "task")).strip("._- ") or "task"
    if cleaned.lower() in _WINDOWS_RESERVED_NAMES:
        cleaned = f"task_{cleaned}"
    cleaned = cleaned[:64].rstrip("._- ")
    if not cleaned or cleaned.lower() in _WINDOWS_RESERVED_NAMES:
        return "task_profile"
    return cleaned


def _resolve_profile_path(profiles_dir: Path, profile_id: str) -> Path:
    """严格校验 profile_id 并解析目标 JSON 路径，阻断任何路径穿越。"""
    raw = str(profile_id or "").strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw or "\x00" in raw:
        raise ValueError(f"非法的任务档案 ID：{profile_id!r}")
    safe_id = sanitize_profile_id(raw, fallback="")
    if safe_id != raw:
        raise ValueError(f"任务档案 ID 包含非法字符：{profile_id!r}")
    base = profiles_dir.resolve()
    candidate = (base / f"{safe_id}.json").resolve()
    if candidate.parent != base:
        raise ValueError(f"检测到非法档案路径：{candidate}")
    return candidate


def _atomic_write_json(path: Path, payload: dict) -> None:
    """原子写入 JSON（临时文件 + os.replace），防止断电或异常中断导致配置半截损坏。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.stem}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _normalize_subsets(raw_subsets, *, portable: bool = False) -> list:
    if not isinstance(raw_subsets, list):
        return []
    out = []
    for idx, item in enumerate(raw_subsets):
        if not isinstance(item, dict):
            continue
        try:
            repeats = max(1, int(float(item.get("repeats", 1) or 1)))
        except Exception:
            repeats = 1
        out.append(
            {
                "name": str(item.get("name", f"subset_{idx + 1}") or f"subset_{idx + 1}").strip(),
                "path": "" if portable else str(item.get("path", "") or "").strip(),
                "repeats": repeats,
                "trigger": str(item.get("trigger", "") or "").strip(),
                "enabled": bool(item.get("enabled", True)),
            }
        )
    return out


def sanitize_profile_config(raw_cfg: dict, *, portable: bool = False) -> dict:
    """
    深度清洗并补全任务配置字典（覆盖本地与云端全量字段）：
    1. 强制剥离全部密钥与临时签名字段；
    2. 过滤 x0.at 及含 AWS SigV4 签名的预签名外链；
    3. 补齐默认高阶参数并统一字段类型，避免类型抖动引发脏状态误报；
    4. portable=True 时额外剥离本机路径与个人云端账号。
    """
    src = copy.deepcopy(raw_cfg) if isinstance(raw_cfg, dict) else {}
    cleaned_in = {}
    for k, v in src.items():
        if not isinstance(k, str):
            continue
        kl = k.strip().lower()
        if kl in FORBIDDEN_SECRET_KEYS or kl == "active_profile_id":
            continue
        cleaned_in[k.strip()] = v

    merged = copy.deepcopy(_BASE_CONFIG_DEFAULTS)
    for k, v in cleaned_in.items():
        merged[k] = v

    for k in list(merged.keys()):
        if k in _BOOL_KEYS:
            merged[k] = bool(merged[k])
        elif k == "subsets":
            merged[k] = _normalize_subsets(merged[k], portable=portable)
        elif merged[k] is None:
            merged[k] = ""
        elif isinstance(merged[k], (int, float)) and not isinstance(merged[k], bool):
            merged[k] = str(merged[k])
        elif isinstance(merged[k], str):
            merged[k] = merged[k].strip()

    tc = str(merged.get("target_compute", "local") or "local").strip().lower()
    merged["target_compute"] = tc if tc in ("local", "kaggle_t4x2", "kaggle_t4x1") else "local"

    ds_src = str(merged.get("kaggle_dataset_source", "local") or "local").strip().lower()
    merged["kaggle_dataset_source"] = ds_src if ds_src in ("local", "hf", "url", "r2") else "local"

    acc = str(merged.get("kaggle_accelerator", "NvidiaTeslaT4") or "NvidiaTeslaT4").strip()
    merged["kaggle_accelerator"] = acc if acc in ("NvidiaTeslaT4", "default_gpu", "NvidiaTeslaP100") else "NvidiaTeslaT4"

    var = str(merged.get("kaggle_variant", "heal") or "heal").strip().lower()
    merged["kaggle_variant"] = var if var in ("heal", "base", "sft") else "heal"

    quant = str(merged.get("kaggle_base_quant", "auto") or "auto").strip().lower()
    merged["kaggle_base_quant"] = quant if quant in ("auto", "int8", "full") else "auto"

    tmode = str(merged.get("train_mode", "fresh") or "fresh").strip().lower()
    merged["train_mode"] = tmode if tmode in ("fresh", "resume", "init_from") else "fresh"

    vram_p = str(merged.get("vram_preset", "auto") or "auto").strip().lower()
    merged["vram_preset"] = vram_p if vram_p in ("auto", "6g", "8g", "12g", "16g") else "auto"

    ext_url = str(merged.get("external_dataset_url", "") or "").strip()
    if "x-amz-signature=" in ext_url.lower():
        ext_url = ""
    else:
        ext_url = sanitize_external_dataset_url(ext_url)
    merged["external_dataset_url"] = ext_url

    if not merged.get("kaggle_max_run_min"):
        merged["kaggle_max_run_min"] = "540"
    if not merged.get("lora_name"):
        merged["lora_name"] = "my_moss_lora"

    if portable:
        for pk in PORTABLE_STRIP_KEYS:
            merged[pk] = ""

    return merged


def _normalize_tags(raw_tags) -> list:
    if isinstance(raw_tags, str):
        items = [t.strip() for t in re.split(r"[,，]+", raw_tags) if t.strip()]
    elif isinstance(raw_tags, (list, tuple)):
        items = [str(t).strip() for t in raw_tags if str(t).strip()]
    else:
        items = []
    seen = set()
    out = []
    for tag in items:
        low = tag.lower()
        if low not in seen:
            seen.add(low)
            out.append(tag)
    return out


def normalize_profile_record(data: dict, fallback_id: str = "task", *, portable: bool = False) -> dict:
    """将原生任务档案或扁平 local_config.json 规整为标准 schema_version=1 档案字典。"""
    if not isinstance(data, dict):
        raise ValueError("任务档案内容必须是 JSON 对象")

    if isinstance(data.get("config"), dict):
        raw_cfg = data["config"]
        raw_name = str(data.get("name") or raw_cfg.get("lora_name") or fallback_id).strip()
        raw_id = str(data.get("id") or fallback_id or raw_name).strip()
        category = str(data.get("category", "custom") or "custom").strip().lower()
        tags = _normalize_tags(data.get("tags", []))
        notes = str(data.get("notes", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        pinned = bool(data.get("pinned", False))
        created_at = str(data.get("created_at") or _utc_now_iso())
        updated_at = str(data.get("updated_at") or created_at)
        last_loaded_at = str(data.get("last_loaded_at") or "")
    else:
        raw_cfg = data
        raw_name = str(data.get("lora_name") or fallback_id or "imported_task").strip()
        raw_id = str(fallback_id or raw_name).strip()
        category = "custom"
        tags = []
        notes = ""
        preset_id = ""
        pinned = False
        created_at = _utc_now_iso()
        updated_at = created_at
        last_loaded_at = ""

    if category not in VALID_PROFILE_CATEGORIES:
        category = "custom"

    safe_id = sanitize_profile_id(raw_id, fallback=sanitize_profile_id(raw_name, "task"))
    clean_cfg = sanitize_profile_config(raw_cfg, portable=portable)
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "id": safe_id,
        "name": raw_name or safe_id,
        "category": category,
        "tags": tags,
        "notes": notes,
        "preset_id": preset_id,
        "pinned": pinned,
        "created_at": created_at,
        "updated_at": updated_at,
        "last_loaded_at": last_loaded_at,
        "config": clean_cfg,
    }


def _generate_unique_id(profiles_dir: Path, raw_name: str, exclude_id: str = "") -> str:
    base_id = sanitize_profile_id(raw_name, fallback="task")
    cand = base_id
    idx = 2
    while True:
        if cand != exclude_id:
            target = profiles_dir / f"{cand}.json"
            if not target.exists():
                return cand
        else:
            return cand
        suffix = f"_{idx}"
        cand = f"{base_id[: max(1, 64 - len(suffix))]}{suffix}"
        idx += 1


def list_task_profiles(
    profiles_dir: Optional[Path] = None,
    *,
    query: str = "",
    compute_filter: str = "all",
    category_filter: str = "all",
    warnings: Optional[list] = None,
) -> list:
    """列出任务档案（支持关键字、算力目标、分类筛选；置顶优先、按更新时间倒序；损坏文件跳过并记录告警）。"""
    base = _resolve_dir(profiles_dir)
    if not base.is_dir():
        return []

    q = str(query or "").strip().lower()
    cf = str(compute_filter or "all").strip().lower()
    catf = str(category_filter or "all").strip().lower()

    items = []
    for fp in sorted(base.iterdir()):
        if not fp.is_file() or fp.suffix.lower() != ".json":
            continue
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
            rec = normalize_profile_record(raw, fallback_id=fp.stem)
            rec["id"] = fp.stem
        except Exception as err:
            if warnings is not None:
                warnings.append(f"{fp.name}: {err}")
            continue

        cfg = rec["config"]
        if cf != "all" and cfg.get("target_compute", "local") != cf:
            continue
        if catf != "all" and rec.get("category", "custom") != catf:
            continue
        if q:
            haystack = " ".join(
                [
                    rec.get("name", ""),
                    rec.get("id", ""),
                    rec.get("notes", ""),
                    " ".join(rec.get("tags", [])),
                    cfg.get("lora_name", ""),
                    cfg.get("trigger_word", ""),
                    cfg.get("kaggle_kernel_slug", ""),
                    cfg.get("hf_dataset_repo", ""),
                ]
            ).lower()
            if q not in haystack:
                continue
        items.append(rec)

    pinned_items = [r for r in items if r.get("pinned")]
    normal_items = [r for r in items if not r.get("pinned")]
    pinned_items.sort(key=lambda r: (str(r.get("updated_at") or ""), str(r.get("name") or "").lower()), reverse=True)
    normal_items.sort(key=lambda r: (str(r.get("updated_at") or ""), str(r.get("name") or "").lower()), reverse=True)
    return pinned_items + normal_items


def load_task_profile(profile_id: str, profiles_dir: Optional[Path] = None) -> dict:
    """读取并规范化指定 ID 的任务档案。"""
    base = _resolve_dir(profiles_dir)
    fp = _resolve_profile_path(base, profile_id)
    if not fp.is_file():
        raise FileNotFoundError(f"任务档案不存在：{profile_id}")
    raw = json.loads(fp.read_text(encoding="utf-8"))
    rec = normalize_profile_record(raw, fallback_id=fp.stem)
    rec["id"] = fp.stem
    return rec


def save_task_profile(
    *,
    name: str,
    config: dict,
    profiles_dir: Optional[Path] = None,
    profile_id: Optional[str] = None,
    category: str = "custom",
    tags=None,
    notes: str = "",
    preset_id: str = "",
    pinned: bool = False,
    overwrite: bool = True,
) -> dict:
    """创建或更新独立任务档案并原子写盘。"""
    base = _resolve_dir(profiles_dir)
    base.mkdir(parents=True, exist_ok=True)
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("任务档案名称不能为空")

    now = _utc_now_iso()
    existing = None
    if profile_id:
        fp = _resolve_profile_path(base, profile_id)
        if fp.is_file():
            if not overwrite:
                raise FileExistsError(f"任务档案已存在：{profile_id}")
            existing = load_task_profile(profile_id, base)
        target_id = profile_id
    else:
        target_id = _generate_unique_id(base, clean_name)
        fp = _resolve_profile_path(base, target_id)

    cat = str(category or (existing or {}).get("category", "custom") or "custom").strip().lower()
    if cat not in VALID_PROFILE_CATEGORIES:
        cat = "custom"

    record = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "id": target_id,
        "name": clean_name,
        "category": cat,
        "tags": _normalize_tags(tags if tags is not None else (existing or {}).get("tags", [])),
        "notes": str(notes if notes is not None else (existing or {}).get("notes", "")).strip(),
        "preset_id": str(preset_id if preset_id is not None else (existing or {}).get("preset_id", "")).strip(),
        "pinned": bool(pinned if pinned is not None else (existing or {}).get("pinned", False)),
        "created_at": (existing or {}).get("created_at") or now,
        "updated_at": now,
        "last_loaded_at": (existing or {}).get("last_loaded_at") or "",
        "config": sanitize_profile_config(config),
    }
    _atomic_write_json(fp, record)
    return record


def touch_profile_loaded(profile_id: str, profiles_dir: Optional[Path] = None) -> dict:
    """记录任务档案最近载入时间。"""
    base = _resolve_dir(profiles_dir)
    rec = load_task_profile(profile_id, base)
    rec["last_loaded_at"] = _utc_now_iso()
    fp = _resolve_profile_path(base, rec["id"])
    _atomic_write_json(fp, rec)
    return rec


def rename_task_profile(
    profile_id: str,
    new_name: str,
    profiles_dir: Optional[Path] = None,
    *,
    update_lora_name: bool = False,
    update_cloud_slugs: bool = False,
    category: Optional[str] = None,
    tags=None,
    notes: Optional[str] = None,
) -> dict:
    """重命名任务档案或更新其元信息（先写入新文件再安全移除旧文件）。"""
    base = _resolve_dir(profiles_dir)
    old_rec = load_task_profile(profile_id, base)
    old_fp = _resolve_profile_path(base, old_rec["id"])

    clean_name = str(new_name or "").strip()
    if not clean_name:
        raise ValueError("新任务名称不能为空")

    new_id = old_rec["id"] if clean_name == old_rec["name"] else _generate_unique_id(base, clean_name, exclude_id=old_rec["id"])
    cfg = copy.deepcopy(old_rec["config"])
    if update_lora_name:
        safe_lora = sanitize_profile_id(clean_name, fallback=cfg.get("lora_name", "my_moss_lora"))
        cfg["lora_name"] = safe_lora
    if update_cloud_slugs:
        slug_base = cfg.get("lora_name") or clean_name
        cfg["kaggle_dataset_slug"] = sanitize_kaggle_slug(f"{slug_base}-data", prefix="moss-data")
        cfg["kaggle_kernel_slug"] = sanitize_kaggle_slug(f"{slug_base}-train", prefix="moss-train")

    cat = str(category if category is not None else old_rec.get("category", "custom")).strip().lower()
    if cat not in VALID_PROFILE_CATEGORIES:
        cat = "custom"

    new_rec = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "id": new_id,
        "name": clean_name,
        "category": cat,
        "tags": _normalize_tags(tags if tags is not None else old_rec.get("tags", [])),
        "notes": str(notes if notes is not None else old_rec.get("notes", "")).strip(),
        "preset_id": old_rec.get("preset_id", ""),
        "pinned": bool(old_rec.get("pinned", False)),
        "created_at": old_rec.get("created_at") or _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "last_loaded_at": old_rec.get("last_loaded_at", ""),
        "config": sanitize_profile_config(cfg),
    }

    new_fp = _resolve_profile_path(base, new_id)
    _atomic_write_json(new_fp, new_rec)
    if new_fp != old_fp and old_fp.exists():
        old_fp.unlink()
    return new_rec


def duplicate_task_profile(
    source_id: str,
    new_name: str = "",
    profiles_dir: Optional[Path] = None,
    *,
    sync_lora_and_slugs: bool = True,
) -> dict:
    """复制已有任务档案为新副本。"""
    base = _resolve_dir(profiles_dir)
    src = load_task_profile(source_id, base)
    target_name = str(new_name or "").strip() or f"{src['name']} 副本"
    cfg = copy.deepcopy(src["config"])
    if sync_lora_and_slugs:
        safe_lora = sanitize_profile_id(target_name, fallback=f"{cfg.get('lora_name', 'lora')}_copy")
        cfg["lora_name"] = safe_lora
        cfg["kaggle_dataset_slug"] = sanitize_kaggle_slug(f"{safe_lora}-data", prefix="moss-data")
        cfg["kaggle_kernel_slug"] = sanitize_kaggle_slug(f"{safe_lora}-train", prefix="moss-train")
    return save_task_profile(
        name=target_name,
        config=cfg,
        profiles_dir=base,
        category=src.get("category", "custom"),
        tags=src.get("tags", []),
        notes=src.get("notes", ""),
        preset_id=src.get("preset_id", ""),
        pinned=False,
    )


def create_continue_task_profile(
    source_id: str,
    *,
    new_name: str,
    train_mode: str = "init_from",
    resume_path: str = "",
    steps: str = "",
    lr: str = "",
    notes: str = "",
    profiles_dir: Optional[Path] = None,
) -> dict:
    """
    基于已有任务档案派生独立的断点续炼 / 权重继承二训任务：
    严格继承源任务的 algo/dora/rank/alpha/target_preset/lokr_*/train_norms 与云端配置，
    同时分配独立的 lora_name 与云端 Slug，避免覆盖原任务产物。
    """
    base = _resolve_dir(profiles_dir)
    src = load_task_profile(source_id, base)
    clean_name = str(new_name or "").strip()
    if not clean_name:
        raise ValueError("派生续炼任务名称不能为空")

    mode = str(train_mode or "init_from").strip().lower()
    if mode not in ("resume", "init_from"):
        mode = "init_from"

    cfg = copy.deepcopy(src["config"])
    cfg["train_mode"] = mode
    cfg["resume_path"] = str(resume_path or "").strip()
    if str(steps or "").strip():
        cfg["steps"] = str(steps).strip()
    if str(lr or "").strip():
        cfg["lr"] = str(lr).strip()

    safe_lora = sanitize_profile_id(clean_name, fallback=f"{cfg.get('lora_name', 'lora')}_v2")
    cfg["lora_name"] = safe_lora
    cfg["kaggle_dataset_slug"] = sanitize_kaggle_slug(f"{safe_lora}-data", prefix="moss-data")
    cfg["kaggle_kernel_slug"] = sanitize_kaggle_slug(f"{safe_lora}-train", prefix="moss-train")

    merged_notes = str(notes or "").strip()
    if not merged_notes:
        merged_notes = f"派生自任务「{src['name']}」({mode})"

    return save_task_profile(
        name=clean_name,
        config=cfg,
        profiles_dir=base,
        category="continue",
        tags=_normalize_tags(list(src.get("tags", [])) + ["续炼"]),
        notes=merged_notes,
        preset_id=src.get("preset_id", ""),
        pinned=False,
    )


def delete_task_profile(profile_id: str, profiles_dir: Optional[Path] = None) -> bool:
    """仅删除指定的任务档案 JSON 文件，绝不触碰数据集或 output/ 下任何训练产物。"""
    base = _resolve_dir(profiles_dir)
    fp = _resolve_profile_path(base, profile_id)
    if not fp.is_file():
        return False
    fp.unlink()
    return True


def toggle_pin_task_profile(profile_id: str, profiles_dir: Optional[Path] = None) -> dict:
    """切换任务档案的置顶状态。"""
    base = _resolve_dir(profiles_dir)
    rec = load_task_profile(profile_id, base)
    rec["pinned"] = not bool(rec.get("pinned", False))
    rec["updated_at"] = _utc_now_iso()
    fp = _resolve_profile_path(base, rec["id"])
    _atomic_write_json(fp, rec)
    return rec


def export_task_profile_to_file(profile: dict, dest_path: Path, *, portable: bool = False) -> Path:
    """导出任务档案为 JSON 文件（portable=True 时自动剥离本机路径与云端个人账号）。"""
    out_path = Path(dest_path)
    normalized = normalize_profile_record(profile, fallback_id=profile.get("id", "task"), portable=portable)
    if portable:
        normalized["last_loaded_at"] = ""
    _atomic_write_json(out_path, normalized)
    return out_path


def import_task_profile_from_file(
    src_path: Path,
    profiles_dir: Optional[Path] = None,
    *,
    name_override: str = "",
) -> dict:
    """导入外部任务档案或 local_config.json，清洗密钥与危险链接后写入档案库。"""
    fp = Path(src_path)
    if not fp.is_file():
        raise FileNotFoundError(f"导入文件不存在：{fp}")
    raw = json.loads(fp.read_text(encoding="utf-8"))
    rec = normalize_profile_record(raw, fallback_id=fp.stem)
    target_name = str(name_override or "").strip() or rec["name"]
    return save_task_profile(
        name=target_name,
        config=rec["config"],
        profiles_dir=profiles_dir,
        category=rec["category"],
        tags=rec["tags"],
        notes=rec["notes"],
        preset_id=rec["preset_id"],
        pinned=False,
    )


def compute_profile_diff(saved_config: dict, current_config: dict) -> list:
    """对比已保存配置与当前配置，返回变更三元组列表 [(key, saved_val, current_val), ...]。"""
    a = sanitize_profile_config(saved_config or {})
    b = sanitize_profile_config(current_config or {})
    diffs = []
    for k in sorted(set(a.keys()) | set(b.keys())):
        va = a.get(k)
        vb = b.get(k)
        if va != vb:
            diffs.append((k, va, vb))
    return diffs


def build_config_from_preset(
    preset_id: str,
    target_compute: str = "local",
    base_config: Optional[dict] = None,
    *,
    lora_name: str = "",
    trigger_word: str = "",
) -> dict:
    """基于 9 套黄金预设与目标算力档案生成初始化配置字典（保留机器环境路径）。"""
    cfg = sanitize_profile_config(base_config or {})
    preset = get_preset_by_id(preset_id)
    tc = str(target_compute or "local").strip().lower()
    if tc not in ("local", "kaggle_t4x2", "kaggle_t4x1"):
        tc = "local"
    cfg["target_compute"] = tc
    if tc == "kaggle_t4x2":
        cfg["kaggle_accelerator"] = "NvidiaTeslaT4"
    elif tc == "kaggle_t4x1":
        cfg["kaggle_accelerator"] = "default_gpu"

    for k, v in ADVANCED_DEFAULTS.items():
        cfg[k] = v

    if preset:
        cfg["multi_subset"] = bool(preset.get("multi_subset", False))
        resolved_vals, _ = resolve_preset_for_target(preset_id, tc)
        for k, v in resolved_vals.items():
            cfg[k] = v

    if str(lora_name or "").strip():
        cfg["lora_name"] = str(lora_name).strip()
    if trigger_word is not None and str(trigger_word).strip():
        cfg["trigger_word"] = str(trigger_word).strip()

    slug_base = cfg.get("lora_name") or "my_moss_lora"
    cfg["kaggle_dataset_slug"] = sanitize_kaggle_slug(f"{slug_base}-data", prefix="moss-data")
    cfg["kaggle_kernel_slug"] = sanitize_kaggle_slug(f"{slug_base}-train", prefix="moss-train")
    return sanitize_profile_config(cfg)


def format_profile_summary(profile: dict) -> dict:
    """生成用于列表展示和详情面板的格式化摘要字典。"""
    rec = normalize_profile_record(profile, fallback_id=profile.get("id", "task"))
    cfg = rec["config"]

    compute_map = {
        "local": "💻 本机",
        "kaggle_t4x2": "☁️ 双卡T4",
        "kaggle_t4x1": "⚡ 单卡T4",
    }
    tc = cfg.get("target_compute", "local")
    compute_label = compute_map.get(tc, tc)

    algo = str(cfg.get("algo", "lora") or "lora").upper()
    if cfg.get("dora"):
        algo_spec = f"{algo}+DoRA R{cfg.get('rank', '32')}"
    elif algo == "LOKR":
        fm = "Full" if cfg.get("lokr_full_matrix") else f"F{cfg.get('lokr_factor', '8')}"
        algo_spec = f"LoKr({fm}) R{cfg.get('rank', '32')}"
    else:
        algo_spec = f"{algo} R{cfg.get('rank', '32')}"

    steps_lr = f"{cfg.get('steps', '?')}步 / {cfg.get('lr', '?')}"

    src_map = {
        "url": "🔗 外链5站",
        "hf": "🤗 HF私有",
        "r2": "☁️ R2私有",
        "local": "📦 Kaggle集",
    }
    cloud_src = src_map.get(cfg.get("kaggle_dataset_source", "local"), "📦 Kaggle集")

    updated_raw = str(rec.get("updated_at") or "")
    updated_short = updated_raw.replace("T", " ").rstrip("Z")[:16] if updated_raw else "-"

    cat_label = dict(PROFILE_CATEGORIES).get(rec.get("category", "custom"), "🧩 自定义")
    tags_txt = ", ".join(rec.get("tags", [])) or "无"
    mode_map = {"fresh": "全新从头训练", "resume": "断点续炼 (Resume)", "init_from": "继承权重再微调 (Init-From)"}

    if cfg.get("multi_subset"):
        subs = cfg.get("subsets", [])
        en_cnt = sum(1 for s in subs if s.get("enabled", True))
        ds_desc = f"多子集混训（共 {len(subs)} 组，启用 {en_cnt} 组）"
    else:
        ds_desc = f"单目录：{cfg.get('data_dir') or '(未指定)'}"

    cloud_extra = ""
    src_code = cfg.get("kaggle_dataset_source", "local")
    if src_code == "hf":
        cloud_extra = f"HF 仓库：{cfg.get('hf_dataset_repo') or '(未填)'}  |  开炼自动上传：{'是' if cfg.get('hf_upload_local') else '否'}"
    elif src_code == "r2":
        cloud_extra = f"R2 桶：{cfg.get('r2_bucket') or '(未填)'}  |  前缀：{cfg.get('r2_prefix') or 'moss-datasets'}"
    elif src_code == "url":
        cloud_extra = f"外链地址：{cfg.get('external_dataset_url') or '(推包时自动传5站)'}"
    else:
        cloud_extra = f"Kaggle Dataset Slug：{cfg.get('kaggle_dataset_slug') or '(自动)'}"

    detail_lines = [
        f"【档案元信息】名称：{rec['name']}  |  ID：{rec['id']}  |  分类：{cat_label}  |  标签：{tags_txt}",
        f"  更新时间：{updated_short}  |  置顶：{'⭐ 是' if rec.get('pinned') else '否'}  |  预设源：{rec.get('preset_id') or '手动/导入'}",
        f"  备注：{rec.get('notes') or '(无备注)'}",
        "",
        f"【任务与图集】LoRA 名：{cfg.get('lora_name')}  |  触发词：{cfg.get('trigger_word') or '(无)'}",
        f"  训练模式：{mode_map.get(cfg.get('train_mode', 'fresh'), 'fresh')}  |  续炼路径：{cfg.get('resume_path') or '(自动/无)'}",
        f"  图集来源：{ds_desc}",
        f"  输出目录：{cfg.get('output_dir') or '(沿用本机默认 output/)'}",
        "",
        f"【核心超参数】目标算力：{compute_label}  |  结构：{algo_spec}  |  Alpha：{cfg.get('alpha')}  |  Target：{cfg.get('target_preset')}",
        f"  分辨率：{cfg.get('resolution')}px  |  Repeats：{cfg.get('repeats')}  |  Epochs：{cfg.get('epochs')}  |  Steps：{cfg.get('steps')}  |  SaveEvery：{cfg.get('save_every')}",
        f"  LR：{cfg.get('lr')}  |  LoRA+：{cfg.get('loraplus_ratio')}  |  GradAcc：{cfg.get('grad_acc')}  |  优化器：{cfg.get('optimizer')} ({cfg.get('lr_scheduler')})",
        f"  时间步：{cfg.get('timestep')} (mu={cfg.get('timestep_mu')})  |  EMA：{cfg.get('ema')}  |  TextDrop：{cfg.get('text_drop')}  |  Flip：{'开' if cfg.get('flip') else '关'}",
        f"  高级引擎：标注模板={cfg.get('caption_template', 'auto')} (p_nl={cfg.get('caption_p_nl', '0.75')}, var={cfg.get('caption_variants', '4')})  |  Compile={cfg.get('compile_mode', 'off')}  |  INT8计算={'开' if cfg.get('int8_compute') else '关'}  |  预览间隔={cfg.get('sample_every', '0')}  |  Teacher={cfg.get('teacher_mode', 'none')}",
        "",
        f"【本地硬件档位】Precision：{cfg.get('precision')}  |  BasePrec：{cfg.get('base_precision')}  |  VRAM：{cfg.get('vram_preset')}  |  TextEnc：{cfg.get('text_device')}  |  VAE：{cfg.get('vae_device')}",
        f"  同步 ComfyUI/loras：{'开启' if cfg.get('copy_to_comfy_loras') else '关闭'}",
        "",
        f"【云端 Kaggle 配置】队列：{cfg.get('kaggle_accelerator')}  |  分支：{cfg.get('kaggle_variant')}  |  量化：{cfg.get('kaggle_base_quant')}  |  超时存盘：{cfg.get('kaggle_max_run_min')} 分",
        f"  Kernel Slug：{cfg.get('kaggle_kernel_slug') or '(自动)'}  |  图集通道：{cloud_src}",
        f"  {cloud_extra}",
    ]

    return {
        "pinned_icon": "⭐" if rec.get("pinned") else "",
        "compute_label": compute_label,
        "algo_spec": algo_spec,
        "steps_lr": steps_lr,
        "cloud_src": cloud_src,
        "updated_short": updated_short,
        "category_label": cat_label,
        "detail_text": "\n".join(detail_lines),
    }
