"""Unit tests for external-URL dataset staging, Kaggle metadata/payload injection, fallback download, and 403 precheck handling."""

import ast
import datetime
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import (
    PROJECT_ROOT as CORE_ROOT,
    build_cloud_config_payload,
    generate_r2_presigned_url,
    read_r2_credentials,
    save_r2_credentials,
    stage_external_dataset_zip,
    stage_kernel_bundle,
    upload_dataset_to_cloudflare_r2,
    upload_dataset_to_hf_repo,
    upload_kaggle_dataset,
    upload_zip_to_external_mirrors,
    validate_training_params,
)


def _sample_cleaned_and_raw(data_dir: Path, multi_subset: bool = False, subsets: list = None):
    raw = {
        "data_dir": str(data_dir),
        "multi_subset": multi_subset,
        "subsets": subsets or [],
        "comfy_models_dir": "",
        "output_dir": str(data_dir / "out"),
        "lora_name": "test_ext_style",
        "trigger_word": "test_trig",
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
        "algo": "lokr",
        "dora": False,
        "target_preset": "attn-mlp+modulation",
        "loraplus_ratio": "1.0",
        "rank": "32",
        "alpha": "32.0",
        "lr": "1.3e-4",
        "repeats": "6",
        "epochs": "8",
        "steps": "540",
        "save_every": "50",
        "grad_acc": "1",
        "timestep_mu": "-1.09",
        "tune": "off",
        "lokr_factor": "8",
        "lokr_full_matrix": False,
        "lokr_decompose_both": False,
        "train_norms": False,
        "rank_dropout": "0.0",
        "module_dropout": "0.0",
        "input_dropout": "0.0",
        "optimizer": "adamw",
        "lr_scheduler": "cosine",
        "warmup_steps": "-1",
        "weight_decay": "0.01",
        "wd_rules": "moss",
        "ema": "0.97",
        "timestep": "moss",
        "text_drop": "0.05",
        "flip": False,
        "seed": "0",
        "save_format": "both",
        "train_mode": "fresh",
        "resume_path": "",
    }
    cleaned, err = validate_training_params(raw)
    assert not err, err
    return cleaned, raw


class ExternalUrlFlowTests(unittest.TestCase):
    def test_stage_external_dataset_zip_single_and_multi(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds1 = root / "ds1"
            ds1.mkdir()
            (ds1 / "img_one.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
            (ds1 / "img_one.txt").write_text("style caption one", encoding="utf-8")

            stage_out = root / "stage"
            _, raw_single = _sample_cleaned_and_raw(ds1)
            zpath, stats = stage_external_dataset_zip(raw_single, "test-ext-data", stage_root=stage_out)
            self.assertTrue(zpath.is_file())
            self.assertEqual(stats["mode"], "single")
            self.assertEqual(stats["images"], 1)
            self.assertEqual(stats["captions"], 1)
            with zipfile.ZipFile(zpath, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("images/sample_0001.png", names)
                self.assertIn("images/sample_0001.txt", names)
                self.assertIn("cloud_dataset_manifest.json", names)

            ds2 = root / "ds2"
            ds2.mkdir()
            (ds2 / "two.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"1" * 64)
            (ds2 / "two.txt").write_text("second subset", encoding="utf-8")
            _, raw_multi = _sample_cleaned_and_raw(
                ds1,
                multi_subset=True,
                subsets=[
                    {"name": "sub_a", "path": str(ds1), "repeats": 4, "trigger": "trig_a", "enabled": True},
                    {"name": "sub_b", "path": str(ds2), "repeats": 8, "trigger": "trig_b", "enabled": True},
                ],
            )
            zpath_m, stats_m = stage_external_dataset_zip(raw_multi, "test-ext-multi", stage_root=stage_out)
            self.assertEqual(stats_m["mode"], "multi_subset")
            self.assertEqual(stats_m["images"], 2)
            with zipfile.ZipFile(zpath_m, "r") as zf:
                names_m = set(zf.namelist())
                self.assertIn("subsets_cloud.json", names_m)
                manifest = json.loads(zf.read("subsets_cloud.json").decode("utf-8"))
                self.assertEqual(len(manifest), 2)

    def test_payload_and_kernel_bundle_external_url_and_dry_check(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "ds"
            ds.mkdir()
            (ds / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
            (ds / "a.txt").write_text("cap", encoding="utf-8")

            cleaned, _raw = _sample_cleaned_and_raw(ds)
            ext_url = "https://temp.sh/abc/test.zip|https://filebin.net/moss-1234/test.zip"
            opts_url = {
                "accelerator": "NvidiaTeslaT4",
                "variant": "heal",
                "base_quant": "auto",
                "max_run_minutes": 540,
                "dataset_source": "url",
                "dataset_url": ext_url,
            }
            payload = build_cloud_config_payload(cleaned, opts_url)
            self.assertEqual(payload.get("dataset_url"), ext_url)

            payload_local = build_cloud_config_payload(
                cleaned,
                {"accelerator": "NvidiaTeslaT4", "dataset_source": "local", "dataset_url": ext_url},
            )
            self.assertNotIn("dataset_url", payload_local)

            kdir, kinfo = stage_kernel_bundle(
                cleaned,
                opts_url,
                username="testuser",
                dataset_id="testuser/unused-ds",
                kernel_slug="test-ext-kernel",
                stage_root=root / "kstage",
            )
            self.assertEqual(kinfo["metadata"]["dataset_sources"], [])
            self.assertEqual(kinfo["metadata"]["enable_internet"], "true")

            dry = subprocess.run(
                [sys.executable, str(kdir / "train_moss.py"), "--dry_config_check"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            parsed = json.loads(dry.stdout)
            self.assertEqual(parsed["dataset_url"], ext_url)
            self.assertEqual(parsed["algo"], "lokr")
            self.assertEqual(parsed["steps"], 540)

    def test_403_precheck_ignored_when_dataset_op_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            stage_dir = Path(td)
            (stage_dir / "dataset-metadata.json").write_text("{}", encoding="utf-8")
            fake_cli = {"found": True, "cmd": ["kaggle"], "exe_path": "kaggle", "version": "2.2.5", "source": "test"}

            precheck_403_output = (
                '[Kaggle API HTTP 403] {"error":{"code":403,"message":"Permission \\u0027datasets.get\\u0027 was denied","status":"PERMISSION_DENIED"}}\n'
                "Upload successful: images.zip (36MB)\n"
                "Dataset operation succeeded: https://www.kaggle.com/datasets/user/slug\n"
            )
            fake_status_404 = mock.Mock(returncode=1, stdout="", stderr="403 Forbidden")
            with (
                mock.patch("core.kaggle_bridge.subprocess.run", return_value=fake_status_404),
                mock.patch("core.kaggle_bridge._run_kaggle_dataset_op", return_value=(0, precheck_403_output)),
            ):
                ok, msg = upload_kaggle_dataset(
                    stage_dir,
                    "user/slug",
                    cli_info=fake_cli,
                    verify_poll_attempts=0,
                )
            self.assertTrue(ok, msg)
            self.assertIn("user/slug", msg)

    def test_train_moss_download_dataset_archive_mirror_fallback(self):
        src = (CORE_ROOT / "train_moss.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        ns = {
            "os": __import__("os"),
            "re": __import__("re"),
            "subprocess": subprocess,
            "zipfile": zipfile,
            "Path": Path,
        }
        want = {"_run_curl_download", "_download_dataset_archive"}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in want:
                mod = ast.Module(body=[node], type_ignores=[])
                exec(compile(mod, str(CORE_ROOT / "train_moss.py"), "exec"), ns)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            zip_payload_file = root / "valid.zip"
            with zipfile.ZipFile(zip_payload_file, "w") as zf:
                zf.writestr("images/sample_0001.jpg", b"\xff\xd8\xff\xe0" + b"x" * 512)
                zf.writestr("images/sample_0001.txt", "my_trig, crisp artwork")
            zip_bytes = zip_payload_file.read_bytes()

            class _Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/bad":
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.end_headers()
                        self.wfile.write(b"<html>not a zip</html>")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.end_headers()
                    self.wfile.write(zip_bytes)

                def log_message(self, format, *args):
                    pass

            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            port = srv.server_address[1]
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                dest_zip = str(root / "out.zip")
                spec = f"http://127.0.0.1:{port}/bad|http://127.0.0.1:{port}/good.zip"
                used = ns["_download_dataset_archive"](spec, dest_zip)
                self.assertTrue(used.endswith("/good.zip"))
                self.assertTrue(zipfile.is_zipfile(dest_zip))
            finally:
                srv.shutdown()
                srv.server_close()

    def test_r2_presigned_url_sigv4_format_and_dry_check_masking(self):
        fixed_ts = datetime.datetime(2026, 11, 23, 4, 0, 0, tzinfo=datetime.timezone.utc)
        url1 = generate_r2_presigned_url(
            account_id="acct123456",
            access_key_id="ak_test_key",
            secret_access_key="sk_test_secret_value",
            bucket="moss-private",
            object_key="moss-datasets/my_data.zip",
            method="GET",
            expires_seconds=604800,
            now_utc=fixed_ts,
        )
        url2 = generate_r2_presigned_url(
            account_id="acct123456",
            access_key_id="ak_test_key",
            secret_access_key="sk_test_secret_value",
            bucket="moss-private",
            object_key="moss-datasets/my_data.zip",
            method="GET",
            expires_seconds=604800,
            now_utc=fixed_ts,
        )
        self.assertEqual(url1, url2)
        self.assertIn("https://acct123456.r2.cloudflarestorage.com/moss-private/moss-datasets/my_data.zip?", url1)
        self.assertIn("X-Amz-Algorithm=AWS4-HMAC-SHA256", url1)
        self.assertIn("ak_test_key%2F20261123%2Fauto%2Fs3%2Faws4_request", url1)
        self.assertIn("X-Amz-Expires=604800", url1)
        self.assertRegex(url1, r"X-Amz-Signature=[0-9a-f]{64}$")

        dry = subprocess.run(
            [sys.executable, str(CORE_ROOT / "train_moss.py"), "--dry_config_check", "--dataset_url", url1],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        parsed = json.loads(dry.stdout)
        self.assertIn("X-Amz-Signature=***", parsed["dataset_url"])
        self.assertNotIn(url1.split("X-Amz-Signature=")[1], parsed["dataset_url"])

    def test_upload_dataset_to_cloudflare_r2_mocked_put(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "ds"
            ds.mkdir()
            (ds / "art.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"a" * 64)
            (ds / "art.txt").write_text("clean brushwork", encoding="utf-8")

            r2_cred_file = root / "r2_credentials.json"
            ok_save, msg_save = save_r2_credentials(
                "acct999", "ak999", "sk999", bucket="my-bucket", r2_json_path=r2_cred_file
            )
            self.assertTrue(ok_save, msg_save)
            snap = read_r2_credentials(r2_json_path=r2_cred_file)
            self.assertTrue(snap["has_secret"])
            self.assertEqual(snap["bucket"], "my-bucket")

            cleaned, raw = _sample_cleaned_and_raw(ds)
            fake_put = mock.Mock(returncode=0, stdout="\n__HTTP_CODE__:200\n", stderr="")
            with mock.patch("core.kaggle_bridge.subprocess.run", return_value=fake_put) as m_run:
                ok, signed_url, stats = upload_dataset_to_cloudflare_r2(
                    raw,
                    {
                        "account_id": "acct999",
                        "access_key_id": "ak999",
                        "secret_access_key": "sk999",
                        "bucket": "my-bucket",
                        "prefix": "style-zips",
                    },
                    dataset_slug="r2-style",
                    stage_root=root / "stage",
                )
            self.assertTrue(ok, signed_url)
            self.assertIn("X-Amz-Signature=", signed_url)
            self.assertEqual(stats["r2_bucket"], "my-bucket")
            put_cmd = m_run.call_args[0][0]
            self.assertIn("-X", put_cmd)
            self.assertIn("PUT", put_cmd)

            kdir, kinfo = stage_kernel_bundle(
                cleaned,
                {"accelerator": "NvidiaTeslaT4", "dataset_source": "r2", "dataset_url": signed_url},
                username="testuser",
                dataset_id="testuser/unused",
                kernel_slug="r2-kernel-test",
                stage_root=root / "kstage",
            )
            self.assertEqual(kinfo["metadata"]["dataset_sources"], [])
            self.assertTrue((kdir / "train_moss.py").is_file())

    def test_upload_dataset_to_hf_repo_mocked_hf_api(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "ds"
            ds.mkdir()
            (ds / "sample.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"b" * 64)
            (ds / "sample.txt").write_text("painted style", encoding="utf-8")
            _, raw = _sample_cleaned_and_raw(ds)

            captured = {}

            class _FakeHfApi:
                def __init__(self, token=None):
                    captured["token"] = token

                def create_repo(self, **kwargs):
                    captured["create_repo"] = kwargs

                def upload_folder(self, **kwargs):
                    captured["upload_folder"] = kwargs
                    staged = Path(kwargs["folder_path"])
                    captured["files"] = sorted(p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file())

            fake_mod = types.SimpleNamespace(HfApi=_FakeHfApi)
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_mod}):
                ok, repo_out, stats = upload_dataset_to_hf_repo(
                    raw,
                    repo_id="https://huggingface.co/datasets/alice/my-private-style",
                    hf_token="hf_test_write_tok",
                    subdir="v1_subset",
                    private=True,
                    stage_root=root / "hf_stage",
                )
            self.assertTrue(ok, repo_out)
            self.assertEqual(repo_out, "alice/my-private-style")
            self.assertEqual(stats["images"], 1)
            self.assertEqual(captured["token"], "hf_test_write_tok")
            self.assertEqual(captured["create_repo"]["repo_type"], "dataset")
            self.assertTrue(captured["create_repo"]["private"])
            self.assertTrue(captured["create_repo"]["exist_ok"])
            self.assertEqual(captured["upload_folder"]["path_in_repo"], "v1_subset")
            self.assertIn("images/sample_0001.jpg", captured["files"])
            self.assertIn("images/sample_0001.txt", captured["files"])

    def test_6_mirror_rotation_parsers(self):
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / "pack.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("hello.txt", "ok")

            def _fake_subproc(cmd, **kwargs):
                joined = " ".join(str(x) for x in cmd)
                if "temp.sh/upload" in joined:
                    return mock.Mock(returncode=0, stdout="https://temp.sh/AbC12/pack.zip\n", stderr="")
                if "x0.at" in joined:
                    return mock.Mock(returncode=0, stdout="https://x0.at/xYz9.zip\n", stderr="")
                if "filebin.net" in joined:
                    return mock.Mock(returncode=0, stdout='{"bin":{"id":"moss-12345678"}}', stderr="")
                if "transfer.whalebone.io" in joined:
                    return mock.Mock(returncode=0, stdout="https://transfer.whalebone.io/Wk99z/pack.zip\n", stderr="")
                if "ul.sto.care" in joined:
                    return mock.Mock(returncode=0, stdout='{"url":"https://trystocare.com/abc123xyz"}', stderr="")
                if "tmpfiles.org" in joined:
                    return mock.Mock(returncode=0, stdout='{"status":"success","data":{"url":"https://tmpfiles.org/999/pack.zip"}}', stderr="")
                return mock.Mock(returncode=1, stdout="", stderr="unknown")

            with mock.patch("core.kaggle_bridge.subprocess.run", side_effect=_fake_subproc):
                ok, combined, mirrors = upload_zip_to_external_mirrors(zpath, max_mirrors=6)
            self.assertTrue(ok, combined)
            self.assertEqual(len(mirrors), 6)
            self.assertEqual(
                [m["host"] for m in mirrors],
                ["temp.sh", "x0.at", "filebin.net", "transfer.whalebone.io", "trystocare.com", "tmpfiles.org"],
            )


if __name__ == "__main__":
    unittest.main()
