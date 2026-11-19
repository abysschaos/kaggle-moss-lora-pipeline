# 贡献指南 (Contributing Guide)

感谢关注并为 **OpenMOSS MOSS-Image Dual-Track LoRA Training Pipeline** 贡献代码或文档。为确保仓库结构整洁、双轨兼容性稳定以及开源合规，请在开发与提交前遵循以下工程规范。

---

## 1. 开发环境准备

1. **克隆仓库**
   ```bash
   git clone https://github.com/abysschaos/kaggle-moss-lora-pipeline.git
   cd kaggle-moss-lora-pipeline
   ```
2. **准备 Python 环境**
   - 建议使用 Python 3.10 ~ 3.12（64-bit）与 CUDA PyTorch 环境，或复用本地 ComfyUI 的高性能 `python.exe`。
   - 遵循本仓 **核心架构解耦原则**：`KohakuMossTrainer` 通过运行时本地克隆/安装拉取，**禁止将 `KohakuMossTrainer/`、`.safetensors` 模型权重、个人数据集 `dataset/` 或输出 `output/` 提交进 Git 仓库**。
3. **构建本地原生启动器**
   ```powershell
   .\scripts\build_exe.ps1 -EmitChecksum
   ```
   确认 `MOSS_LoRA_Trainer.exe` 在本机 `.NET Framework 4.x csc.exe` 下可 1 秒内无警告编译，且产物与 `SHA256SUMS.txt` 均被 `.gitignore` 正确忽略。

---

## 2. 提交与分支规范

### 分支约定
- `main`：稳定主分支，始终保持可编译、可执行、文档对齐的发布就绪状态。
- `feat/<topic>` / `fix/<topic>` / `docs/<topic>`：功能开发、缺陷修复或文档修订分支。

### 约定式提交 (Conventional Commits)
所有 Commit Message 遵循 `<type>(<scope>): <subject>` 格式：
- `feat(...)`：新增面向用户的新功能（例如适配器选项、GUI 控件、云端同步特性）
- `fix(...)`：缺陷修复（例如引擎补丁、显存 OOM 修复、Windows 启动器或路径兼容修复）
- `docs(...)`：文档新增或与源码的同步修正
- `refactor(...)`：不改变对外行为的代码梳理、去重与死代码清理
- `chore(release)`：版本发版提交（需同步更新 `VERSION`、`CHANGELOG.md`、`README.md` 与 `launcher/MossTrainerLauncher.cs`）

---

## 3. 代码、文档与隐私底线

1. **反重复造轮子与引擎对齐**：
   - 新增适配器或训练参数前，务必先核对上游 `KohakuBlueleaf/KohakuMossTrainer` 的源码与 `knobs.validate()` 白名单，避免传递未定义字段。
   - 若同时影响本地（`train_local.py` / `gui_local.py`）与云端（`train_moss.py` / `notebooks/kaggle_moss_lora_training.ipynb`），必须在同一次 PR 中完成双轨同步。
2. **PowerShell 5.1 兼容性**：
   - `.ps1` 脚本必须兼容 Windows 默认自带的 **PowerShell 5.1**，禁止使用 PowerShell 7+ 专属语法（如 `?:`、`??`、`-PipelineVariable` 新扩展等）。
   - 向外部 REST API 发送含中文的 JSON 时，必须显式编码为 UTF-8 字节流（`[System.Text.Encoding]::UTF8.GetBytes(...)`）并设置 `application/json; charset=utf-8`。
3. **文档与代码强同步**：
   - 增删 CLI 参数、环境变量、目录结构或版本号时，须同步更新 `README.md`、`.env.example` 与 `docs/parameter_configuration_reference.md`。
4. **隐私与凭据零泄露**：
   - 提交前确认不包含任何本地系统用户名、个人盘符路径、`local_config.json`、Kaggle 私人 Slug、HuggingFace Token 或 GitHub PAT。

---

## 4. 提交前本地自检清单

在提交 PR 前，请至少运行以下本地检查并确保全部通过：

```powershell
# 1. Python 核心脚本语法检查
python -m py_compile gui_local.py train_local.py train_moss.py scripts/detect_env.py scripts/generate_icon.py scripts/silent_launcher.pyw

# 2. 单一版本信源与 CLI 参数一致性自检
python train_local.py --version
python train_moss.py --version

# 3. Notebook JSON 合法性检查
python -c "import json; json.load(open('notebooks/kaggle_moss_lora_training.ipynb', encoding='utf-8')); print('Notebook JSON OK')"

# 4. 原生 Windows 启动器干净编译验证
.\scripts\build_exe.ps1 -EmitChecksum

# 5. 工作区洁净度检查（确保没有误加入私有配置、模型权重或 exe）
git status
```
