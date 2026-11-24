#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对 v1.8.5 黄金预设矩阵、双卡 T4 步数保底公式、自定义名称/触发词防覆盖保护、
真实 EMA 平滑权重导出补丁及满秩 LoKr 文件头嗅探的自动化单元测试套件。
所有测试均在本地临时目录运行，零外部网络请求、零真实账号写操作。
"""

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_local
from core.config_store import compute_exposure_summary, load_app_version
from core.kaggle_bridge import build_cloud_config_payload
from core.presets import (
    ADVANCED_DEFAULTS,
    CLOUD_T4X2_OVERRIDES,
    DEFAULT_LEGACY_TRIGGERS,
    DEFAULT_PLACEHOLDER_LORA_NAMES,
    MIN_RECOMMENDED_STEPS_LOKR,
    MIN_RECOMMENDED_STEPS_MULTI,
    MIN_RECOMMENDED_STEPS_SINGLE,
    PRESET_ORDER,
    PRESETS,
    get_preset_by_id,
    recommend_min_steps,
    recommend_save_every_and_ema,
    resolve_preset_for_target,
)
from core.trainer_runner import validate_training_params


class _VarStub:
    """无窗口轻量 Tk Variable 替身，用于直接验证 ModernLocalTrainerGUI.apply_preset 逻辑契约。"""

    def __init__(self, value=None):
        self._val = value

    def get(self):
        return self._val

    def set(self, val):
        self._val = val


class PresetsAndMathTest(unittest.TestCase):
    def test_version_alignment(self):
        ver = load_app_version()
        self.assertEqual(ver, "1.8.5")
        self.assertEqual(train_local._load_app_version(), "1.8.5")

    def test_all_eight_presets_validate_and_align_with_upstream_specs(self):
        self.assertIsInstance(PRESETS, list)
        self.assertEqual(len(PRESETS), 8)
        self.assertEqual(len(PRESET_ORDER), 8)
        required_ui_keys = {
            "id",
            "row",
            "label",
            "bg",
            "fg_key",
            "active_bg_key",
            "multi_subset",
            "align_steps",
            "values",
            "dialog_title",
            "dialog_message",
        }
        for preset in PRESETS:
            self.assertTrue(required_ui_keys.issubset(preset.keys()), f"preset missing UI keys: {preset.get('id')}")
            self.assertIn(preset["row"], (1, 2))
        for pid in PRESET_ORDER:
            preset_obj = get_preset_by_id(pid)
            self.assertIsNotNone(preset_obj)
            self.assertIn(pid, CLOUD_T4X2_OVERRIDES)

            for target in ("local", "kaggle_t4x2"):
                vals, notes = resolve_preset_for_target(pid, target)
                self.assertTrue(vals, f"preset {pid} resolved empty")
                self.assertEqual(vals.get("ema"), "0.0", f"preset {pid} should default EMA to 0.0")
                self.assertEqual(vals.get("trigger_word"), "", f"preset {pid} should not hardcode fake trigger")

                raw = {
                    "data_dir": str(PROJECT_ROOT),
                    "multi_subset": bool(preset_obj.get("multi_subset", False)),
                    "subsets": [
                        {"name": "s1", "path": str(PROJECT_ROOT), "repeats": 4, "trigger": "t1", "enabled": True},
                        {"name": "s2", "path": str(PROJECT_ROOT), "repeats": 4, "trigger": "t2", "enabled": True},
                    ],
                    "comfy_models_dir": "",
                    "output_dir": "./output",
                    "precision": "bf16",
                    "base_precision": "auto",
                    "vram_preset": "auto",
                    "resolution": "1024",
                    "text_device": "cpu",
                    "vae_device": "cuda",
                    "steps": "240",
                    "save_every": "100",
                    "tune": "off",
                    "copy_to_comfy_loras": False,
                    **ADVANCED_DEFAULTS,
                    **vals,
                }
                cleaned, err = validate_training_params(raw)
                self.assertIsNone(err, f"validate_training_params failed for {pid}/{target}: {err}")
                self.assertIsNotNone(cleaned)
                if target == "kaggle_t4x2":
                    self.assertEqual(cleaned["grad_acc"], 1)

        # 关键预设架构断言：lokr_style 满秩 factor=16 + mu=-0.30；antifit_char 半秩 16/8.0 + text_drop=0.02；muon_style WD=0.02
        lokr_local, _ = resolve_preset_for_target("lokr_style", "local")
        lokr_t4, _ = resolve_preset_for_target("lokr_style", "kaggle_t4x2")
        self.assertEqual(lokr_local["algo"], "lokr")
        self.assertEqual(lokr_local["lokr_factor"], "16")
        self.assertTrue(lokr_local["lokr_full_matrix"])
        self.assertEqual(lokr_local["lr"], "2e-4")
        self.assertEqual(lokr_t4["lr"], "2.5e-4")
        self.assertEqual(lokr_local["timestep_mu"], "-0.30")

        anti_local, _ = resolve_preset_for_target("antifit_char", "local")
        anti_t4, _ = resolve_preset_for_target("antifit_char", "kaggle_t4x2")
        self.assertEqual(anti_local["rank"], "16")
        self.assertEqual(anti_local["alpha"], "8.0")
        self.assertTrue(anti_local["dora"])
        self.assertEqual(anti_local["rank_dropout"], "0.05")
        self.assertEqual(anti_local["module_dropout"], "0.05")
        self.assertEqual(anti_local["text_drop"], "0.02")
        self.assertEqual(anti_local["lr"], "8e-5")
        self.assertEqual(anti_t4["lr"], "1e-4")

        muon_local, _ = resolve_preset_for_target("muon_style", "local")
        muon_t4, _ = resolve_preset_for_target("muon_style", "kaggle_t4x2")
        self.assertEqual(muon_local["optimizer"], "muon")
        self.assertEqual(muon_local["weight_decay"], "0.02")
        self.assertEqual(muon_t4["lr"], "1.3e-4")

    def test_cloud_dry_config_check_for_upgraded_lokr_and_antifit_presets(self):
        lokr_t4, _ = resolve_preset_for_target("lokr_style", "kaggle_t4x2")
        raw = {
            "data_dir": str(PROJECT_ROOT),
            "multi_subset": False,
            "subsets": [],
            "comfy_models_dir": "",
            "output_dir": "./output",
            "precision": "bf16",
            "base_precision": "auto",
            "vram_preset": "auto",
            "resolution": "1024",
            "text_device": "cpu",
            "vae_device": "cuda",
            "steps": "200",
            "save_every": "50",
            "tune": "off",
            "copy_to_comfy_loras": False,
            **ADVANCED_DEFAULTS,
            **lokr_t4,
        }
        cleaned, err = validate_training_params(raw)
        self.assertIsNone(err)
        payload = build_cloud_config_payload(cleaned, {"accelerator": "NvidiaTeslaT4", "variant": "heal"})
        res = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "train_moss.py"), "--config_json", json.dumps(payload), "--dry_config_check"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        parsed = json.loads(res.stdout)
        self.assertEqual(parsed["algo"], "lokr")
        self.assertEqual(int(parsed["lokr_factor"]), 16)
        self.assertTrue(bool(parsed["lokr_full_matrix"]))
        self.assertAlmostEqual(float(parsed["lr"]), 2.5e-4, places=8)
        self.assertAlmostEqual(float(parsed["timestep_mu"]), -0.30, places=5)

    def test_step_floor_prevents_small_dataset_underfitting_on_dual_t4(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "small_char"
            ds.mkdir()
            for idx in range(15):
                (ds / f"img_{idx:02d}.png").write_bytes(b"fake")

            # 15 图 * 7 repeats * 4 epochs = 420 次曝光；Dual-T4 下 4 图/步，原始公式仅 105 步 -> 必须触发 180 步保底
            res_single = compute_exposure_summary(
                multi_subset=False,
                subsets=[],
                data_dir=str(ds),
                repeats=7,
                epochs=4,
                vram_preset="auto",
                detected_vram_gb=16.0,
                resolution=1024,
                grad_acc=1,
                timestep_mu=0.35,
                active_gpus=2,
                max_tokens_override=9216,
                algo="lora",
            )
            self.assertEqual(res_single[1], 15)
            self.assertEqual(res_single[4], 420)
            self.assertEqual(res_single[6], 4)
            self.assertEqual(res_single[5], MIN_RECOMMENDED_STEPS_SINGLE)

            # LoKr 单图集触发 200 步保底
            res_lokr = compute_exposure_summary(
                multi_subset=False,
                subsets=[],
                data_dir=str(ds),
                repeats=5,
                epochs=5,
                vram_preset="auto",
                detected_vram_gb=16.0,
                resolution=1024,
                grad_acc=1,
                timestep_mu=-0.30,
                active_gpus=2,
                max_tokens_override=9216,
                algo="lokr",
            )
            self.assertEqual(res_lokr[5], MIN_RECOMMENDED_STEPS_LOKR)

            # 多子集过滤 enabled=False 子集，并触发 240 步保底
            disabled_dir = Path(tmp) / "disabled_big"
            disabled_dir.mkdir()
            for idx in range(40):
                (disabled_dir / f"d_{idx:02d}.jpg").write_bytes(b"fake")

            res_multi = compute_exposure_summary(
                multi_subset=True,
                subsets=[
                    {"name": "a", "path": str(ds), "repeats": 4, "enabled": True},
                    {"name": "off", "path": str(disabled_dir), "repeats": 10, "enabled": False},
                ],
                data_dir="",
                repeats=1,
                epochs=4,
                vram_preset="auto",
                detected_vram_gb=16.0,
                resolution=1024,
                grad_acc=1,
                timestep_mu=0.35,
                active_gpus=2,
                max_tokens_override=9216,
                algo="lora",
            )
            self.assertEqual(res_multi[1], 15)
            self.assertEqual(res_multi[4], 240)
            self.assertEqual(res_multi[5], MIN_RECOMMENDED_STEPS_MULTI)

            # 大图集超出下限时严格走 math.ceil(total_exposures / eff_batch)
            res_large = compute_exposure_summary(
                multi_subset=False,
                subsets=[],
                data_dir=str(ds),
                repeats=5,
                epochs=5,
                vram_preset="auto",
                detected_vram_gb=16.0,
                resolution=1024,
                grad_acc=1,
                timestep_mu=-0.85,
                active_gpus=2,
                max_tokens_override=9216,
                img_count_override=61,
                algo="lora",
            )
            # 61 * 5 * 5 = 1525; ceil(1525 / 4) = 382
            self.assertEqual(res_large[5], 382)

            # 空图集保持向后兼容返回 1
            empty_dir = Path(tmp) / "empty"
            empty_dir.mkdir()
            res_empty = compute_exposure_summary(
                multi_subset=False,
                subsets=[],
                data_dir=str(empty_dir),
                repeats=4,
                epochs=6,
                vram_preset="auto",
                detected_vram_gb=8.0,
                resolution=1024,
                grad_acc=1,
                timestep_mu=-0.85,
            )
            self.assertEqual(res_empty[5], 1)

    def test_apply_preset_preserves_user_custom_name_and_trigger(self):
        from ui.app import ModernLocalTrainerGUI

        dummy = ModernLocalTrainerGUI.__new__(ModernLocalTrainerGUI)
        for attr, default_v in [
            ("var_multi_subset", False),
            ("var_target_compute", "local"),
            ("var_repeats", "4"),
            ("var_epochs", "6"),
            ("var_rank", "32"),
            ("var_alpha", "32.0"),
            ("var_algo", "lora"),
            ("var_dora", False),
            ("var_target_preset", "attn-mlp"),
            ("var_loraplus_ratio", "1.0"),
            ("var_lr", "1e-4"),
            ("var_grad_acc", "1"),
            ("var_timestep_mu", "-0.85"),
            ("var_lora_name", "rani3_style_v2"),
            ("var_trigger", "by rani3"),
            ("var_lokr_factor", "8"),
            ("var_lokr_full_matrix", False),
            ("var_lokr_decompose_both", False),
            ("var_train_norms", False),
            ("var_rank_dropout", "0.0"),
            ("var_module_dropout", "0.0"),
            ("var_input_dropout", "0.0"),
            ("var_text_drop", "0.05"),
            ("var_ema", "0.0"),
            ("var_optimizer", "adamw"),
            ("var_lr_scheduler", "cosine"),
            ("var_warmup_steps", "-1"),
            ("var_weight_decay", "0.01"),
            ("var_wd_rules", "moss"),
            ("var_timestep", "moss"),
            ("var_flip", False),
            ("var_seed", "0"),
            ("var_save_format", "both"),
            ("var_save_every", "100"),
        ]:
            setattr(dummy, attr, _VarStub(default_v))

        dummy._apply_subset_panel_visibility = lambda: None
        dummy.align_steps_by_formula = lambda silent=False: None
        dummy.on_dynamic_controls_changed = lambda: None
        dummy.recalculate_math = lambda: None
        dummy.update_quick_status_summary = lambda: None
        dummy.highlight_active_preset = lambda pid: None

        with mock.patch("ui.app.messagebox.showinfo"):
            dummy.apply_preset("lokr_style")
            self.assertEqual(dummy.var_lora_name.get(), "rani3_style_v2")
            self.assertEqual(dummy.var_trigger.get(), "by rani3")
            self.assertEqual(dummy.var_lokr_factor.get(), "16")
            self.assertTrue(dummy.var_lokr_full_matrix.get())
            self.assertEqual(dummy.var_timestep_mu.get(), "-0.30")

            dummy.apply_preset("muon_style")
            self.assertEqual(dummy.var_optimizer.get(), "muon")
            self.assertEqual(dummy.var_weight_decay.get(), "0.02")
            self.assertEqual(dummy.var_lora_name.get(), "rani3_style_v2")
            self.assertEqual(dummy.var_trigger.get(), "by rani3")

            # 当处于系统默认占位符时，正常替换为新预设默认名并清空旧版假触发词
            dummy.var_lora_name.set("my_moss_lora")
            dummy.var_trigger.set("in the style of my_art_style")
            dummy.apply_preset("character")
            self.assertEqual(dummy.var_lora_name.get(), "my_character")
            self.assertEqual(dummy.var_trigger.get(), "")

    def test_ema_export_patch_idempotency_and_separate_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            t_dir = Path(tmp) / "KohakuMossTrainer"
            entry_py = t_dir / "src" / "kohakumoss" / "training" / "entry.py"
            entry_py.parent.mkdir(parents=True)
            entry_py.write_text(
                "def export_into(k, layout, built, parts, step, root, is_main):\n"
                "    with layout.full_weights(built.backbone):\n"
                "        if is_main:\n"
                '            meta = {"kohakumoss_step": str(step), "kohakumoss_algo": str(k["ALGO"])}\n'
                "            try:\n"
                "                for path in export(k, built, root, meta):\n"
                '                    log.info(f"exported {path}")\n'
                "            except OSError as err:\n"
                '                log.error(f"export of step {step} into {root} failed: {err}")\n',
                encoding="utf-8",
            )
            train_local.apply_compatibility_patches(t_dir)
            first_patched = entry_py.read_text(encoding="utf-8")
            self.assertIn("MOSS_PIPELINE_PATCH_EMA_EXPORT", first_patched)
            compile(first_patched, str(entry_py), "exec")

            # 二次调用严格幂等
            train_local.apply_compatibility_patches(t_dir)
            self.assertEqual(entry_py.read_text(encoding="utf-8"), first_patched)

            # 验证 export_and_sync_weights 同时保留 raw 与 -ema 权重且互不覆盖
            exp_dir = t_dir / "runs" / "moss" / "demo_lora" / "export"
            exp_dir.mkdir(parents=True)
            raw_bytes = self._make_minimal_safetensors({"a": [2, 2]}, {"kohakumoss_step": "200"})
            ema_bytes = self._make_minimal_safetensors({"a": [2, 2]}, {"kohakumoss_step": "200", "kohakumoss_ema": "ema0.97"}) + b"EMA"
            (exp_dir / "demo_lora-comfy.safetensors").write_bytes(raw_bytes)
            (exp_dir / "demo_lora-ema0.97-comfy.safetensors").write_bytes(ema_bytes)

            out_folder = Path(tmp) / "output" / "demo_lora"
            exported, step_num = train_local.export_and_sync_weights(
                t_dir, "demo_lora", out_folder, quiet=True
            )
            self.assertEqual(step_num, 200)
            self.assertEqual((out_folder / "demo_lora-comfy.safetensors").read_bytes(), raw_bytes)
            self.assertEqual((out_folder / "demo_lora-ema0.97-comfy.safetensors").read_bytes(), ema_bytes)
            self.assertEqual(len(exported), 2)

    def test_inspect_safetensors_full_matrix_lokr_without_spec_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            sf_p = Path(tmp) / "ext_lokr_full.safetensors"
            sf_p.write_bytes(
                self._make_minimal_safetensors(
                    {
                        "lora_unet_blocks_0_mlp_fc1.lokr_w1": [16, 16],
                        "lora_unet_blocks_0_mlp_fc1.lokr_w2": [240, 240],
                    },
                    {},
                )
            )
            info = train_local.inspect_safetensors_lora(sf_p)
            self.assertEqual(info["algo"], "lokr")
            self.assertTrue(info["lokr_full_matrix"])
            self.assertEqual(info["lokr_factor"], 16)
            self.assertEqual(info["rank"], 32)
            self.assertEqual(info["alpha"], 1.0)

            # 同时验证 KohakuMossTrainer 导出的 -comfy.safetensors 头（含 ss_network_dim="32" 与 ss_network_args，不含 kohakumoss_spec）
            sf_comfy = Path(tmp) / "exported-comfy.safetensors"
            sf_comfy.write_bytes(
                self._make_minimal_safetensors(
                    {
                        "lora_unet_layers_0_proj_k.lokr_w1": [16, 16],
                        "lora_unet_layers_0_proj_k.lokr_w2": [192, 192],
                    },
                    {
                        "ss_network_dim": "32",
                        "ss_network_alpha": "32.0",
                        "ss_network_args": json.dumps({"algo": "lokr", "factor": 16, "full_matrix": True}),
                    },
                )
            )
            info_c = train_local.inspect_safetensors_lora(sf_comfy)
            self.assertEqual(info_c["algo"], "lokr")
            self.assertEqual(info_c["rank"], 32)
            self.assertEqual(info_c["alpha"], 32.0)
            self.assertEqual(info_c["lokr_factor"], 16)
            self.assertTrue(info_c["lokr_full_matrix"])

            # 验证 prepare_init_from_checkpoint 点选 -comfy.safetensors 时自动优先选用同目录 -trainable.safetensors
            sf_train = Path(tmp) / "exported-trainable.safetensors"
            raw_train_bytes = self._make_minimal_safetensors({"lora_unet_layers_0_proj_k.lokr_w1": [16, 16]}, {"kohakumoss_raw_state": "1"}) + b"RAW"
            sf_train.write_bytes(raw_train_bytes)
            staged_dir, _ = train_local.prepare_init_from_checkpoint(
                str(sf_comfy), Path(tmp) / "ws", "exported", Path(tmp), Path(tmp)
            )
            self.assertEqual((staged_dir / "trainable.safetensors").read_bytes(), raw_train_bytes)

    def test_real_gui_startup_and_preset_buttons(self):
        import tkinter as tk
        from ui.app import ModernLocalTrainerGUI

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"No GUI display available: {exc}")
            return

        root.withdraw()
        app = None
        try:
            # 阻止后台初始化自动覆写本地 local_config.json 与弹出窗口
            with mock.patch("ui.app.save_local_config"), mock.patch("ui.app.messagebox.showinfo"):
                app = ModernLocalTrainerGUI(root)
                self.assertEqual(len(app.preset_buttons), 8)
                for pid in PRESET_ORDER:
                    self.assertIn(pid, app.preset_buttons)
                    app.apply_preset(pid)
                root.update_idletasks()
        finally:
            if app is not None:
                app._closing = True
                if getattr(app, "_ui_queue_job", None) is not None:
                    try:
                        root.after_cancel(app._ui_queue_job)
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

    @staticmethod
    def _make_minimal_safetensors(shapes: dict, metadata: dict) -> bytes:
        header = {"__metadata__": metadata}
        offset = 0
        for name, shp in shapes.items():
            numel = 1
            for d in shp:
                numel *= d
            nbytes = numel * 2
            header[name] = {"dtype": "BF16", "shape": shp, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        hdr_bytes = json.dumps(header).encode("utf-8")
        return struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + (b"\x00" * offset)


if __name__ == "__main__":
    unittest.main()
