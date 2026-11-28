#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对每 LoRA 独立任务档案管理模板（local + cloud）的自动化测试套件：
涵盖路径穿越与 Windows 保留名防护、凭据强剥离与便携导出脱敏、CRUD / 续炼派生 / 差异比对、
以及真实 Tkinter 6-Tab GUI 跨任务切换、未保存修改拦截与训练期切档阻断。
所有测试均在本地临时目录运行，零外部网络请求、零真实配置文件污染。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.presets import ADVANCED_DEFAULTS
from core.task_profiles import (
    FORBIDDEN_SECRET_KEYS,
    PORTABLE_STRIP_KEYS,
    PROFILE_SCHEMA_VERSION,
    build_config_from_preset,
    compute_profile_diff,
    create_continue_task_profile,
    delete_task_profile,
    duplicate_task_profile,
    export_task_profile_to_file,
    format_profile_summary,
    import_task_profile_from_file,
    list_task_profiles,
    load_task_profile,
    rename_task_profile,
    sanitize_profile_config,
    sanitize_profile_id,
    save_task_profile,
    toggle_pin_task_profile,
)


class TaskProfilesTests(unittest.TestCase):
    def test_sanitize_profile_id_and_path_traversal_protection(self):
        self.assertEqual(sanitize_profile_id("CON"), "task_CON")
        self.assertEqual(sanitize_profile_id("nul"), "task_nul")
        self.assertEqual(sanitize_profile_id("COM1"), "task_COM1")
        self.assertEqual(sanitize_profile_id("角色_Rani v2 / 测试"), "角色_Rani_v2_测试")

        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "profiles"
            for bad_id in ("../evil", "..\\evil", "dir/sub", "a\\b", "", "   "):
                with self.assertRaises(ValueError):
                    load_task_profile(bad_id, pdir)
                with self.assertRaises(ValueError):
                    delete_task_profile(bad_id, pdir)

            r1 = save_task_profile(name="same_name", config={"lora_name": "a"}, profiles_dir=pdir)
            r2 = save_task_profile(name="same_name", config={"lora_name": "b"}, profiles_dir=pdir)
            r3 = save_task_profile(name="same_name", config={"lora_name": "c"}, profiles_dir=pdir)
            self.assertEqual(r1["id"], "same_name")
            self.assertEqual(r2["id"], "same_name_2")
            self.assertEqual(r3["id"], "same_name_3")

    def test_secret_stripping_and_portable_export(self):
        dirty_cfg = {
            "lora_name": "secret_guard_style",
            "target_compute": "kaggle_t4x2",
            "algo": "lokr",
            "lokr_factor": "16",
            "lokr_full_matrix": True,
            "data_dir": "D:/my_datasets/art1",
            "output_dir": "C:/gravityanti/output",
            "comfy_models_dir": "C:/ComfyUI/models",
            "transformer_path": "C:/ComfyUI/models/moss.safetensors",
            "kaggle_username": "my_user",
            "r2_account_id": "acct123",
            "r2_bucket": "private-bkt",
            "r2_access_key_id": "AKIA_PUBLIC",
            "kaggle_api_key": "KAGGLE_SECRET_SHOULD_VANISH",
            "hf_token": "hf_secret_token_should_vanish",
            "r2_secret_access_key": "R2_SECRET_SHOULD_VANISH",
            "r2_dataset_url": "https://r2.example.com/signed?X-Amz-Signature=deadbeef",
            "api_key": "generic_secret",
            "external_dataset_url": "https://x0.at/abc.zip|https://temp.sh/good.zip",
            "multi_subset": True,
            "subsets": [
                {"name": "s1", "path": "D:/ds/s1", "repeats": 4, "trigger": "trig1", "enabled": True}
            ],
            "active_profile_id": "should_not_nest",
        }
        cleaned = sanitize_profile_config(dirty_cfg)
        for sec_k in FORBIDDEN_SECRET_KEYS | {"active_profile_id"}:
            self.assertNotIn(sec_k, cleaned)
        self.assertEqual(cleaned["external_dataset_url"], "https://temp.sh/good.zip")

        # 含 AWS SigV4 签名的 external_dataset_url 也必须清零
        sig_cleaned = sanitize_profile_config(
            {"external_dataset_url": "https://bucket.r2.cloudflarestorage.com/a.zip?X-Amz-Signature=12345"}
        )
        self.assertEqual(sig_cleaned["external_dataset_url"], "")

        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "profiles"
            rec = save_task_profile(
                name="隐私核验档案",
                config=dirty_cfg,
                profiles_dir=pdir,
                category="lokr",
                tags=["画风", "双卡"],
            )
            disk_raw = (pdir / f"{rec['id']}.json").read_text(encoding="utf-8")
            for secret_val in ("KAGGLE_SECRET_SHOULD_VANISH", "hf_secret_token_should_vanish", "R2_SECRET_SHOULD_VANISH", "x0.at", "deadbeef"):
                self.assertNotIn(secret_val, disk_raw)

            portable_file = Path(tmp) / "share_template.json"
            export_task_profile_to_file(rec, portable_file, portable=True)
            exported = json.loads(portable_file.read_text(encoding="utf-8"))
            self.assertEqual(exported["schema_version"], PROFILE_SCHEMA_VERSION)
            for k in PORTABLE_STRIP_KEYS:
                self.assertEqual(exported["config"][k], "", f"portable export leaked {k}")
            self.assertEqual(exported["config"]["subsets"][0]["path"], "")
            self.assertEqual(exported["config"]["subsets"][0]["trigger"], "trig1")
            self.assertEqual(exported["config"]["algo"], "lokr")
            self.assertEqual(exported["config"]["lokr_factor"], "16")
            self.assertTrue(exported["config"]["lokr_full_matrix"])
            self.assertEqual(exported["config"]["target_compute"], "kaggle_t4x2")

    def test_crud_rename_duplicate_continue_pin_filter_and_delete_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdir = root / "task_profiles"
            out_dir = root / "output" / "char_a"
            ds_dir = root / "dataset" / "char_a"
            out_dir.mkdir(parents=True)
            ds_dir.mkdir(parents=True)
            weight_file = out_dir / "char_a-comfy.safetensors"
            img_file = ds_dir / "01.png"
            weight_file.write_bytes(b"weights")
            img_file.write_bytes(b"image")

            p1 = save_task_profile(
                name="角色A_本机",
                config={
                    "lora_name": "char_a",
                    "trigger_word": "char_a",
                    "target_compute": "local",
                    "algo": "lora",
                    "dora": True,
                    "rank": "24",
                    "alpha": "24.0",
                    "target_preset": "attn-mlp",
                    "data_dir": str(ds_dir),
                    "output_dir": str(root / "output"),
                    "kaggle_accelerator": "NvidiaTeslaT4",
                    "kaggle_variant": "heal",
                },
                profiles_dir=pdir,
                category="character",
                tags=["character", "dora"],
            )
            p2 = save_task_profile(
                name="画风B_双卡HF",
                config={
                    "lora_name": "style_b",
                    "target_compute": "kaggle_t4x2",
                    "algo": "lokr",
                    "lokr_factor": "16",
                    "lokr_full_matrix": True,
                    "kaggle_dataset_source": "hf",
                    "hf_dataset_repo": "owner/style-b",
                },
                profiles_dir=pdir,
                category="lokr",
                tags="style, lokr",
            )

            # 写入一个损坏 JSON 文件，验证 list_task_profiles 容错
            (pdir / "broken.json").write_text("{not valid json", encoding="utf-8")
            warns = []
            all_list = list_task_profiles(pdir, warnings=warns)
            self.assertEqual(len(all_list), 2)
            self.assertEqual(len(warns), 1)
            self.assertIn("broken.json", warns[0])

            # 置顶与过滤
            toggle_pin_task_profile(p1["id"], pdir)
            pinned_first = list_task_profiles(pdir)
            self.assertEqual(pinned_first[0]["id"], p1["id"])
            self.assertEqual(len(list_task_profiles(pdir, compute_filter="kaggle_t4x2")), 1)
            self.assertEqual(len(list_task_profiles(pdir, category_filter="character")), 1)
            self.assertEqual(len(list_task_profiles(pdir, query="owner/style-b")), 1)

            # 重命名与复制
            renamed = rename_task_profile(
                p2["id"],
                "画风B_正式版",
                pdir,
                update_lora_name=True,
                update_cloud_slugs=True,
            )
            self.assertFalse((pdir / f"{p2['id']}.json").exists())
            self.assertTrue((pdir / f"{renamed['id']}.json").exists())
            self.assertEqual(renamed["config"]["lora_name"], "画风B_正式版")

            dup = duplicate_task_profile(p1["id"], "角色A_副本", profiles_dir=pdir)
            self.assertNotEqual(dup["id"], p1["id"])
            self.assertEqual(dup["config"]["rank"], "24")

            # 派生续炼任务：继承全部架构与云端参数，生成独立 lora_name 与 Slug
            cont = create_continue_task_profile(
                p1["id"],
                new_name="char_a_round2",
                train_mode="init_from",
                resume_path=str(weight_file),
                steps="220",
                lr="5e-5",
                profiles_dir=pdir,
            )
            self.assertEqual(cont["category"], "continue")
            self.assertEqual(cont["config"]["train_mode"], "init_from")
            self.assertEqual(cont["config"]["resume_path"], str(weight_file))
            self.assertEqual(cont["config"]["steps"], "220")
            self.assertEqual(cont["config"]["lr"], "5e-5")
            self.assertEqual(cont["config"]["algo"], "lora")
            self.assertTrue(cont["config"]["dora"])
            self.assertEqual(cont["config"]["rank"], "24")
            self.assertEqual(cont["config"]["lora_name"], "char_a_round2")
            self.assertIn("char-a-round2", cont["config"]["kaggle_kernel_slug"])

            # 验证摘要格式与删除安全（仅删配置 JSON，不删数据集和产出权重）
            summary = format_profile_summary(cont)
            self.assertIn("char_a_round2", summary["detail_text"])
            self.assertTrue(delete_task_profile(p1["id"], pdir))
            self.assertTrue(weight_file.is_file())
            self.assertTrue(img_file.is_file())

    def test_import_legacy_local_config_and_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy_file = Path(tmp) / "local_config.json"
            legacy_file.write_text(
                json.dumps(
                    {
                        "lora_name": "legacy_lora",
                        "target_compute": "kaggle_t4x1",
                        "rank": 48,
                        "alpha": 48.0,
                        "kaggle_api_key": "leak_attempt",
                        "active_profile_id": "old_id",
                    }
                ),
                encoding="utf-8",
            )
            pdir = Path(tmp) / "profiles"
            imported = import_task_profile_from_file(legacy_file, pdir)
            self.assertEqual(imported["schema_version"], PROFILE_SCHEMA_VERSION)
            self.assertEqual(imported["name"], "legacy_lora")
            for k, v in ADVANCED_DEFAULTS.items():
                self.assertIn(k, imported["config"])
                self.assertEqual(imported["config"][k], v)
            self.assertNotIn("kaggle_api_key", imported["config"])

            # 等价配置（int/str 混合）不触发误报差异
            same_with_int = dict(imported["config"])
            same_with_int["rank"] = 48
            self.assertEqual(compute_profile_diff(imported["config"], same_with_int), [])

            changed = dict(imported["config"])
            changed["lr"] = "5e-5"
            diffs = compute_profile_diff(imported["config"], changed)
            self.assertEqual(len(diffs), 1)
            self.assertEqual(diffs[0][0], "lr")

    def test_build_config_from_preset_local_and_dual_t4(self):
        cfg = build_config_from_preset(
            "lokr_style",
            "kaggle_t4x2",
            {"output_dir": "./output"},
            lora_name="ink_master",
            trigger_word="in the style of ink",
        )
        self.assertEqual(cfg["target_compute"], "kaggle_t4x2")
        self.assertEqual(cfg["kaggle_accelerator"], "NvidiaTeslaT4")
        self.assertEqual(cfg["algo"], "lokr")
        self.assertEqual(cfg["lokr_factor"], "8")
        self.assertTrue(cfg["lokr_full_matrix"])
        self.assertEqual(cfg["grad_acc"], "1")
        self.assertEqual(cfg["lr"], "4e-4")
        self.assertEqual(cfg["lora_name"], "ink_master")
        self.assertEqual(cfg["trigger_word"], "in the style of ink")

    def test_real_gui_profile_switching_dirty_check_and_busy_guard(self):
        import tkinter as tk
        from ui.app import ModernLocalTrainerGUI

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"No GUI display available: {exc}")
            return

        root.withdraw()
        app = None
        with tempfile.TemporaryDirectory() as tmp:
            tmp_pdir = Path(tmp) / "task_profiles"
            try:
                with mock.patch("ui.app.save_local_config"), mock.patch("ui.app.messagebox.showinfo"):
                    app = ModernLocalTrainerGUI(root)
                    app.profiles_dir = tmp_pdir
                    app.active_profile_id = ""
                    app._active_profile_record = None
                    app._active_profile_snapshot = None

                    # 1. 验证 6 个 Tab 与任务档案页挂载正常
                    self.assertEqual(len(app.tab_frames), 6)
                    self.assertIn("profiles", app.tab_index_map)
                    self.assertEqual(app.tab_index_map["profiles"], 4)
                    self.assertEqual(app.tab_index_map["help"], 5)
                    self.assertTrue(hasattr(app, "profile_tree"))
                    self.assertTrue(hasattr(app, "cb_quick_profile"))

                    # 2. 构造两份迥异任务档案（本地 LoKr 多子集 vs 云端双卡 T4 HF + init_from）
                    local_cfg = build_config_from_preset(
                        "lokr_style",
                        "local",
                        app._collect_raw_params(),
                        lora_name="local_ink_lokr",
                        trigger_word="ink_brush",
                    )
                    local_cfg["multi_subset"] = True
                    local_cfg["subsets"] = [
                        {"name": "sub_a", "path": str(PROJECT_ROOT), "repeats": 5, "trigger": "ink_a", "enabled": True}
                    ]
                    prof_local = save_task_profile(
                        name="本地水墨LoKr",
                        config=local_cfg,
                        profiles_dir=tmp_pdir,
                        category="lokr",
                        preset_id="lokr_style",
                    )

                    cloud_cfg = build_config_from_preset(
                        "character",
                        "kaggle_t4x2",
                        app._collect_raw_params(),
                        lora_name="cloud_char_t4",
                        trigger_word="hero_girl",
                    )
                    cloud_cfg["train_mode"] = "init_from"
                    cloud_cfg["rank"] = "16"
                    cloud_cfg["alpha"] = "8.0"
                    cloud_cfg["kaggle_dataset_source"] = "hf"
                    cloud_cfg["hf_dataset_repo"] = "user/hero-char-ds"
                    cloud_cfg["kaggle_variant"] = "base"
                    cloud_cfg["comfy_models_dir"] = "Z:/nonexistent_other_machine/ComfyUI/models"
                    prof_cloud = save_task_profile(
                        name="云端双卡角色",
                        config=cloud_cfg,
                        profiles_dir=tmp_pdir,
                        category="character",
                        preset_id="character",
                    )

                    # 3. 载入本地档案并注入一条临时 R2 签名链接，再切换至云端档案，校验全量参数切换与临时 R2 清理
                    self.assertTrue(app.load_profile_by_id(prof_local["id"], skip_dirty_check=True))
                    app.var_r2_dataset_url.set("https://r2.example.com/temp?X-Amz-Signature=abc")
                    self.assertFalse(app.has_unsaved_profile_changes())

                    self.assertTrue(app.load_profile_by_id(prof_cloud["id"], switch_to_tab="cloud"))
                    self.assertEqual(app.active_profile_id, prof_cloud["id"])
                    self.assertEqual(app.var_lora_name.get(), "cloud_char_t4")
                    self.assertEqual(app.var_target_compute.get(), "kaggle_t4x2")
                    self.assertEqual(app.var_kaggle_dataset_source.get(), "hf")
                    self.assertEqual(app.var_hf_dataset_repo.get(), "user/hero-char-ds")
                    self.assertEqual(app.var_kaggle_variant.get(), "base")
                    self.assertEqual(app.var_train_mode.get(), "init_from")
                    self.assertEqual(app.var_rank.get(), "16")
                    self.assertEqual(app.var_alpha.get(), "8.0")
                    self.assertEqual(app.var_r2_dataset_url.get(), "")
                    self.assertFalse(app.has_unsaved_profile_changes())
                    self.assertEqual(app.notebook.index(app.notebook.select()), app.tab_index_map["cloud"])

                    # 4. 未保存修改拦截测试（选取消留原地、选是自动保存再切换）
                    app.var_lr.set("5e-5")
                    self.assertTrue(app.has_unsaved_profile_changes())
                    with mock.patch("ui.app.messagebox.askyesnocancel", return_value=None):
                        self.assertFalse(app.load_profile_by_id(prof_local["id"]))
                        self.assertEqual(app.active_profile_id, prof_cloud["id"])
                    with mock.patch("ui.app.messagebox.askyesnocancel", return_value=True):
                        self.assertTrue(app.load_profile_by_id(prof_local["id"]))
                        self.assertEqual(app.active_profile_id, prof_local["id"])
                    reloaded_cloud = load_task_profile(prof_cloud["id"], tmp_pdir)
                    self.assertEqual(reloaded_cloud["config"]["lr"], "5e-5")
                    self.assertTrue(app.var_multi_subset.get())
                    self.assertEqual(app.var_algo.get(), "lokr")
                    self.assertEqual(app.var_lokr_factor.get(), "8")

                    # 5. 训练进行中严格禁止切换档案
                    app.is_training = True
                    with mock.patch("ui.app.messagebox.showwarning") as warn_mock:
                        self.assertFalse(app.load_profile_by_id(prof_cloud["id"], skip_dirty_check=True))
                        self.assertTrue(warn_mock.called)
                    app.is_training = False

                    root.update_idletasks()
            finally:
                if app is not None:
                    app._closing = True
                    for job_attr in ("_ui_queue_job", "_dirty_check_job", "_cloud_poll_job"):
                        job = getattr(app, job_attr, None)
                        if job is not None:
                            try:
                                root.after_cancel(job)
                            except Exception:
                                pass
                    try:
                        app.monitor.stop_metrics_polling()
                    except Exception:
                        pass
                try:
                    root.destroy()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
