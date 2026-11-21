# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned (Roadmap)
- **体验增强候选 (v1.5.x)**：
  - 训练期定时样图预览（`SAMPLE_EVERY` / `SAMPLE_PROMPTS` + 显存守卫）；
  - 更多冷门/专项适配器算法接入（`tlora` / `oft`）；
  - 可选 `tags` 词表打标模式（暴露 `shuffle` 与 `keep_first` 控制，兼容常规标签图集）。
- **云端平台远程控制专项 (v1.6.0 Cloud Remote Control)**：
  - 在本地 GUI 内直接远程驱动云端训练，大幅降低个人用户使用云卡的门槛：
    - **Kaggle 无头一键流水线**：本地图集自动打包 ZIP 并调用 Kaggle CLI 推送私有 Dataset、`kernels push` 投递作业、定时轮询状态并自动拉回最终与阶段权重；
    - **通用 SSH/SFTP 云服务器遥控器**：基于 `paramiko` 直连任意 Linux GPU 实例，支持一键传图、远程拉起 `train_moss.py` 与日志流式回传、本地看板实时渲染远端 Loss 曲线；
    - **按量算力平台轻量 API 适配**：预留 AutoDL / RunPod / Vast.ai / 腾讯云 HAI 等平台的一键启停机与费用告警适配器接口。

## [1.5.1] - 2026-11-22

### Added
- **GUI 多标签页（5-Tab Notebook）清爽工作区重构与模块化解耦（`core/` + `ui/`）**：
  - 将原先单栏长达三千余行的 `gui_local.py` 拆分为清晰的逻辑内核包 [core/](core/__init__.py)（`config_store.py` 路径/GPU/配置持久化与曝光计算、`presets.py` 8 套声明式预设数据、`trainer_runner.py` 参数校验/CLI 构建/优雅暂停/进程树清理/权重抢救）与表现层包 [ui/](ui/__init__.py)（`theme.py`、`widgets.py`、`monitor_panel.py`、`app.py` 及 5 个独立选项卡模块）；
  - 左侧配置区升级为 `ttk.Notebook` 暗色无边框 5 大选项卡：**`🚀 快速训练`**（8 套预设 + 命名状态摘要条 + 核心超参 & 默认折叠的 18 项高阶抽屉 + 一键公式对齐）、**`📂 数据与续炼`**（单目录/多子集配比 Treeview 与一键配平 + 图集体检 + `fresh / resume / init_from` 规格自动对齐）、**`📦 底模与框架`**（ComfyUI/本地三核心底模探测 + `KohakuMossTrainer` 下载同步与补丁修复）、**`☁️ 云端遥控`**（`v1.6.0` 云端遥控架构规划展示及现有 Kaggle Notebook / `kernel-metadata.json` / `.env.example` / 部署指南快捷入口）、**`📖 指南与工具`**（五大离线指南弹窗直读器、本机运行环境一键自检、常用目录直达）；
  - 右侧常驻 [MonitorPanel](ui/monitor_panel.py)（4 枚 KPI 卡片、实时进度条、Loss/EMA 原生画布折线图、终端控制台与启动/暂停存盘/强制终止控制条），在任意选项卡下均可实时监控收敛情况。

### Changed
- **全入口零破坏兼容与主题细节优化**：[gui_local.py](gui_local.py) 保留为轻量兼容入口，无缝对接 `MOSS_LoRA_Trainer.exe`、`launch_ui.vbs`、`launch_ui.bat` 与 `scripts/silent_launcher.pyw`，完整向后兼容既有的 `local_config.json` 与 `train_local.py` 全部 54 个 CLI 参数；切换选项卡时自动清理滚轮监听并刷新滚动区避免漂移。
- **版本与文档同步**：更新 [VERSION](VERSION) 至 `1.5.1`、[launcher/MossTrainerLauncher.cs](launcher/MossTrainerLauncher.cs) 程序集版本至 `1.5.1.0`，同步更新 [README.md](README.md)、[docs/beginner_guide.md](docs/beginner_guide.md) 及全局技能 `kohaku`，重新编译 `MOSS_LoRA_Trainer.exe` 与 `SHA256SUMS.txt`。

## [1.5.0] - 2026-11-22

### Added
- **Phase 1 — 高阶适配器与数据复现旋钮双轨透出（`train_local.py` / `train_moss.py` / `.env.example` / Kaggle Notebook）**：
  - 开放 LoKr 精细结构控制：`--lokr_factor`（`-1 / 4 / 8 / 16`，默认 `8`）、`--lokr_full_matrix`、`--lokr_decompose_both`，并在 GUI 中绑定随 `algo == "lokr"` 自动显隐；
  - 新增 `--train_norms`（`TRAIN_NORMS`，允许协同微调 AdaLN / RMS / LayerNorm 权重）；
  - 新增三层级防过拟合 Dropout（`--input_dropout`、`--rank_dropout`、`--module_dropout` -> `DROPOUT = {...}`），内置与 `adapters/build/config.py` 对齐的 **DoRA 冲突强制守卫**：开启 DoRA 时界面锁定禁用并强制后端将 `input_dropout` 置 `0.0`（保留 `rank_dropout` 与 `module_dropout` 正常生效）；
  - 新增 `--seed`（随机种子复现）、`--save_format`（`both / comfy / hf / lycoris`，自动将 `lycoris` 别名合规映射至上游 `hf` 导出键名）、`--text_drop`（空提示词概率，默认 `0.05`）与 `--flip`（水平镜像增强，默认 `False` 并附非对称服饰与文字严禁提示）。
- **Phase 2 — 多优化器、学习率调度、EMA 与多时间步分布控制**：
  - 新增 `--optimizer {adamw, adamw8bit_kernel, muon}`：针对非 `adamw` 优化器注入 **GPU 优化器安全流式显存档位自动重映射**（`6g/8g -> ckpt-ring2-actoff`、`12g -> ckpt-ring4-actoff`、`16g -> ckpt-ring8-actoff`），严格满足 `optim_state="gpu"` 与 `fused_step=False` 验证门禁；
  - 新增 `--lr_scheduler {cosine, constant}` 与 `--warmup_steps`（`-1` 为自动 5% 裁剪 `[5, 100]`，并对显式数值实施 `< STEPS` 边界保护防止 `schedule_problems` 报错）；
  - 新增 `--weight_decay`、`--wd_rules {moss, none, all}` 与 `--ema`（如 `0.99 ~ 0.995`）；
  - 新增 `--timestep {moss, uniform, logit_normal}`，严格按不同分布生成合规 `TIMESTEP_KWARGS`（训练 `shift` 恒定锁定 `1.0`），GUI 醒目提示 `logit_normal` 与 `moss` 在 `mu` 正负号上的反转差异并在 `uniform` 下禁用 `mu`。
- **训练前【🩺 一键体检图集】与 3 套特色实战预设 (`gui_local.py` / `train_local.py`)**：
  - 在 `train_local.py` 实现纯 Python 零像素解码消耗的 `audit_dataset_directories()` 检查器（支持 PIL 与纯 `struct` PNG/JPEG/WEBP/BMP 头读取兜底），在 GUI 数据集卡片顶部新增 **【🩺 一键体检图集】** 按钮（支持单图集与多子集）：精准检出缺/空 `.txt`、低分辨率 `<768px`、极端长宽比 `>3:1`、按当前 `MAX_TOKENS` + 分辨率预算估算超限的超长 `.txt` 以及疑似 Danbooru 纯逗号标签堆砌并输出诊断报告窗；
  - 超参数卡片采用**默认折叠的【🧪 进阶微调、防过拟合与优化器设置】抽屉**保持主界面简洁；预设栏扩展为两行 8 套，新增 **【💎 LoKr 高还原画风】**、**【🛡️ 小图集防过拟合角色】** 与 **【⚡ Muon 画风加速】**，并统一通过 `_reset_advanced_defaults()` 防预设切换串味。
- **小白傻瓜式使用手册与全文档同步**：
  - 新增完全面向零基础用户的大白话完全指引 [docs/beginner_guide.md](docs/beginner_guide.md)，涵盖 30 秒上手三步走、8 大预设挑选表、全量 v1.4.0 + v1.5.0 界面按钮字典、翻车速查表与 ComfyUI 推理对齐规范；
  - 同步更新 [README.md](README.md)、[docs/parameter_configuration_reference.md](docs/parameter_configuration_reference.md)、[docs/hyperparameter_tuning_guide.md](docs/hyperparameter_tuning_guide.md) 及全局技能 `kohaku`。

### Changed
- **断点与继续训练自动结构对齐增强 (`inspect_safetensors_lora` / `_inspect_safetensors`)**：续炼与 `init_from` 时支持额外探测并自动对齐 `train_norms` 与 LoKr 参数（`factor`、`full_matrix`、`decompose_both`）。
- **单版本信源同步更新**：更新 [VERSION](VERSION) 至 `1.5.0`、[launcher/MossTrainerLauncher.cs](launcher/MossTrainerLauncher.cs) 程序集版本至 `1.5.0.0`，重新编译 `MOSS_LoRA_Trainer.exe` 并更新 `SHA256SUMS.txt`。

## [1.4.0] - 2026-11-20

### Added
- **Kaggle `GPU T4 x2` 双卡自动并行加速管线 (`--gpus` / `--parallel` / `--max_tokens`)**：
  - 在 [train_moss.py](train_moss.py)、[.env.example](.env.example)、[notebooks/kaggle_moss_lora_training.ipynb](notebooks/kaggle_moss_lora_training.ipynb)、[kernel-metadata.json](kernel-metadata.json) 与 [scripts/push_to_kaggle.ps1](scripts/push_to_kaggle.ps1) 中新增多卡控制旋钮（`--gpus {auto,1,2}`、`--parallel {auto,none,ddp,pipeline}`、`--max_tokens` 与 `machine_shape = "NvidiaTeslaT4"`）。
  - 在 Kaggle Linux 双 T4 环境下默认自动启用 `PARALLEL = "ddp"`、`DEVICES = (0, 1)`、`DIST_BACKEND = "auto"`、`DDP_REDUCE_DTYPE = "fp32"`、`DDP_BUCKET_MB = 25`，并配套锁定 **`int8` 底模 + `MEMORY_PROFILE = "ckpt-resident"` + `MAX_TOKENS = 9216`**（消除 CPU-GPU 频繁卸载同时保护 30GB 主机内存，每卡打包 2 × 1024px、双卡全局并发 4 × 1024px / 步，实测稳态达 `30.16 s/step`，较单卡 7 小时方案提速约 2.2 倍）。
  - 同步保留 `--parallel pipeline`（1F1B 跨卡层切分）供需要全精度 FP16 常驻的场景手动切换。

### Changed
- **五位一体同步升级与文档补全**：更新 [VERSION](VERSION) 至 `1.4.0`、[launcher/MossTrainerLauncher.cs](launcher/MossTrainerLauncher.cs) 程序集版本至 `1.4.0.0`，同步更新 [README.md](README.md)、[docs/kaggle_deployment_guide.md](docs/kaggle_deployment_guide.md)、[docs/parameter_configuration_reference.md](docs/parameter_configuration_reference.md) 与 [docs/troubleshooting_matrix.md](docs/troubleshooting_matrix.md)。

### Fixed
- **修复 PyTorch 2.11+ `is_bf16_supported()` 在 T4 (`sm75`) 误报导致训练验证崩溃的缺陷**：由于新版 PyTorch 将软件模拟检测纳入 `torch.cuda.is_bf16_supported()` 返回值，T4 被误判为支持 BF16 并被 `KohakuMossTrainer` 拦截报错 `PRECISION='bf16' on cuda:0 (sm75, no bf16)`；现已于 [train_moss.py](train_moss.py)、[train_local.py](train_local.py) 与 [gui_local.py](gui_local.py) 全面切换为严谨的物理计算能力检查 `sm_major >= 8`（多卡模式下联检全部参与设备），确保 T4 自动锁定原生 `fp16`。
- **修复自然语言触发词重复拼接问题**：将 [train_moss.py](train_moss.py) 注入触发词的前置判断由仅比对句首（`startswith`）升级为全文不区分大小写检索（`trigger_clean.lower() not in caption.lower()`），避免已含 `in the style of <trigger>` 的高质量自然语言标注被二次重复加前缀。

## [1.3.0] - 2026-11-20

### Added
- **5 大双轨核心超参数与量化精度控制 (`train_local.py` / `train_moss.py` / `gui_local.py` / `.env.example` / Kaggle Notebook)**：
  - 新增 **`--grad_acc` / `GRAD_ACC`**（梯度累积步数，默认 `1`），在不改变默认极速画风训练吞吐的前提下，为人物五官与多子集混合训练提供低方差平滑梯度更新。
  - 新增 **`--timestep_mu` / `TIMESTEP_MU`**（显式注入 `TIMESTEP = "moss"` 与 `TIMESTEP_KWARGS = {"mu": ...}`，默认 `-1.09` 保持约 89% 高噪画风笔触覆盖；`0.35` 可增强人物面部服饰细节，`-0.60` / `0.0` 适配多画风与动作概念）。
  - 新增 **`--base_precision` / `BASE_PRECISION`**（`auto / int8 / fp8 / fp8_compute / compute`）：内置 `sm89+`（RTX 40 系 / L4 / A100）算力检查与 `int8_convrot` 底模防误配守卫，启用 `fp8_compute` 时强制注入 `FP8_DGRAD = "16bit"` 避免双份转置副本造成显存零节省。
  - 新增 **`--tune` / `TUNE`**（`off / table`）与 **`--overlong` / `OVERLONG`**（`drop_tokens / drop`）。
- **新增 `896` 黄金甜点分辨率与 GUI 动态步数计算引擎**：本地 GUI、CLI 与云端 Notebook 全线支持 `1024 / 896 / 768 / 512` 四档分辨率（`896` 对应 `56×56 = 3136` 视觉 Token，可在 `8192` Budget 下稳定打包 2 图/步）；GUI 新增 `_estimate_effective_batch()` 与 Mu 感知动态除数，实时估算等效 Batch 并推荐对齐步数。
- **五大实战预设模板全面升级**：保留【🎨 艺术画风】(`GradAcc=1`, `Mu=-1.09`) 极速收敛优势不变；同步为【👤 单人角色】、`【👥 多人物配比】`、`【🎭 多画风配比】`与`【💡 概念/动作】`补全精准配比的 `Alpha`、`DoRA`、`LoRA+`、`TargetPreset`、`GradAcc` 与 `Timestep Mu`。

### Changed
- **修正 `12g` / `16g` 显存档位与文本编码器内存阈值**：将 `12g` 档位校准为 `vram_12g.py` + `MAX_TOKENS = 6144`（约 8.5GB 峰值），`16g` 档位校准为 `vram_16g.py` + `MAX_TOKENS = 8192`（避免 T4/16G OOM 溢出）；本地 `probe_local_models` 在系统内存 ≥22GB 或 GPU ≥15GB 时自动优先选用高精度 `qwen3.5_9b_bf16`。
- **扩宽 Warmup 上限与同步云端续炼元信息解析**：将 `SCHEDULE["warmup"]` 上限由 `30` 步放宽至 `100` 步以适配长步数梯度累积任务；同步升级云端 `train_moss._inspect_safetensors` 解析历史权重的 `algo`、`dora` 与 `preset`。
- **版本与启动器同步**：更新 [VERSION](VERSION) 至 `1.3.0`，同步 [launcher/MossTrainerLauncher.cs](launcher/MossTrainerLauncher.cs) 程序集版本至 `1.3.0.0` 并更新全套技术文档。

### Fixed
- **默认切换 `OVERLONG = "drop_tokens"` 防长文本静默丢图**：修复原默认 `OVERLONG = "drop"` 在窄 `MAX_TOKENS` 档位下将超长高质量自然语言 `.txt` 样本整条丢弃的问题（仅截断超限尾部 Token，完整保留图片样本）；同时修复 `apply_concept_preset()` 未复位多子集开关的 UI 状态残留。

## [1.2.2] - 2026-11-20

### Added
- **单一版本信源 (`VERSION`) 与命令行版本校验**：新增根目录 [VERSION](VERSION)（`1.2.2`），为 [gui_local.py](gui_local.py)、[train_local.py](train_local.py) 与 [train_moss.py](train_moss.py) 统一注入版本号读取、`--version` 参数及启动横幅显示。
- **GitHub 社区标准与开源合规基建**：补齐完整 [Apache License 2.0](LICENSE) 协议、脱敏环境变量模板 [.env.example](.env.example)、贡献指南 [CONTRIBUTING.md](CONTRIBUTING.md)、安全政策 [SECURITY.md](SECURITY.md) 以及 `.github/ISSUE_TEMPLATE/` 与 `.github/PULL_REQUEST_TEMPLATE.md` 标准交互模板。
- **GitHub 主页与 Releases 自动化同步工具**：新增原生 PowerShell 5.1 兼容脚本 [scripts/sync_github_release.ps1](scripts/sync_github_release.ps1)（支持严格 UTF-8 编码同步仓库简介、主页、Topics，并幂等创建/更新 Release 及挂载二进制资产）；升级 [scripts/build_exe.ps1](scripts/build_exe.ps1) 支持版本一致性校验与 `-EmitChecksum` 生成 `SHA256SUMS.txt`。

### Changed
- **工业级重构 [README.md](README.md) 与全量对齐 `docs/*.md` 文档体系**：重写主文档架构、Mermaid 系统流程图、Quickstart 启动入口对照表与目录树（与仓库实际文件 100% 同步）；全面修复 [parameter_configuration_reference.md](docs/parameter_configuration_reference.md)、[local_installation_guide.md](docs/local_installation_guide.md)、[kaggle_deployment_guide.md](docs/kaggle_deployment_guide.md)、[hyperparameter_tuning_guide.md](docs/hyperparameter_tuning_guide.md)、[local_deployment_guide.md](docs/local_deployment_guide.md) 与 [comfyui_inference_guide.md](docs/comfyui_inference_guide.md) 中的过时行号、旧版示例文件名及 `v1.1`/`v1.2` 新特性说明漂移；更新 [install_windows.bat](install_windows.bat)、[scripts/install.ps1](scripts/install.ps1) 与 Notebook 文档指向最新 `.exe` / `.vbs` 无黑框启动入口。
- **遵循 Keep a Changelog 1.1.0 与统一发行体系**：按标准归档历史版本 `v1.1.0`、`v1.2.0`、`v1.2.1` 及本次 `v1.2.2` 的 GitHub Releases 发行页，同步更新 [launcher/MossTrainerLauncher.cs](launcher/MossTrainerLauncher.cs) 程序集版本至 `1.2.2.0` 与 [docs/github_release_workflow_template.yml](docs/github_release_workflow_template.yml) 构建校验步骤。

### Fixed
- **修复 `scripts/run_local.ps1` 自定义分辨率空值崩溃**：修正定制模式下用户按回车采纳默认分辨率时传递未定义 `$customRes` 变量导致 `--resolution` 报错的问题（正确传递 `$defaultResolution`），并对齐展示 `VERSION` 版本号。

### Removed
- **清理孤立死脚本与废弃 IGNORE 规则**：移除已被原生 `.exe` 与 `.vbs` 完全取代的冗余启动器 `scripts/launch_ui.ps1`，并从 [.gitignore](.gitignore) 清理废弃条目 `local_env.bat`、新增 `SHA256SUMS.txt` 忽略规则。

## [1.2.1] - 2026-11-20

### Fixed
- **补齐 `FP16_FIX_KWARGS = {"c": 1 / 16}` 防溢出系数**：修复本地 [train_local.py](train_local.py) 与云端 [train_moss.py](train_moss.py) 在启用 `FP16_FIX="online"`（如 `6g/8g/12g + FP16 + Base/SFT` 组合）时因缺少缩放系数参数触发 `KohakuMossTrainer` `knobs.validate()` 报错的问题。
- **清除 `artist_from_folder` 空标注图目录名污染**：将本地与云端 `DATASETS` 配置中的 `"artist_from_folder"` 统一由 `"name"` 调整为 `None`，避免纯多子集目录下的无标注图片回退触发 `TagTemplate` 时把文件夹名误当画师标签拼接进训练文本。
- **保留 `.txt` 原始 `mtime` 保护增量扫描缓存**：`prepare_dataset_structure` 与 `prepare_multi_subset_dataset` 仅在目标 `.txt` 不存在或内容确实变化时写盘，防止每次重跑无故刷新时间戳导致 `kohakumoss_data` catalog 缓存失效。
- **幂等化 `objective.py` 兼容补丁**：改用正则归一化替换逻辑，彻底消除多次运行引发的 `if isinstance(sub, dict)` 嵌套叠加并自动清理既有冗余。

### Changed
- 同步更新原生启动器程序集版本至 `1.2.1.0`。

## [1.2.0] - 2026-11-20

### Added
- **多子集配比管理器（本地 GUI / CLI & 云端全同步）**：支持通过 `--subsets_json` 或 GUI 表格在单个训练任务中配置多个独立子目录（多人物 / 多画风 / 概念 LoRA），每个子集独立绑定专属触发词与独立 `Repeat` 重复权重，自动生成标准 Kohya `<repeat>_<subset_name>` 目录结构并独立隔离清理孤儿文件；GUI 新增深色 `Treeview` 子集管理表格、添加/编辑弹窗与【⚖️ 一键配平 Repeat】功能（按子集图数反比自动均衡单轮曝光占比），并与公式步数换算实时联动。
- **高阶适配器架构与目标注入层预设**：严格参照 `KohakuMossTrainer` 官方 `algo_*.py` 与 `export-to-comfy.md` 规范，双轨开放 **LoRA / LoHa / LoKr** 算法切换（自动注入对应安全内核参数 `ALGO_KWARGS`）、**DoRA** 权重分解（自动配对 `DORA_DETACH = True` 兼顾稳定性与省显存）、**LoRA+ 学习率倍率**（`LORAPLUS_RATIO`）以及 6 档 Transformer 注入层预设（`attn` / `attn-mlp` / `attn-mlp+modulation` / `attn-mlp+text-refiner` / `full-block` / `all`），显式锁定 `SAVE_FUSED_QKV = False` 保障 ComfyUI 紧凑原生加载。
- **五大实战预设模板与续炼结构自动对齐**：GUI 顶部预设栏扩充为【🎨 艺术画风】、【👤 单人角色】、【👥 多人物配比】、【🎭 多画风配比】与【💡 概念/动作】；增强 `inspect_safetensors_lora`（及云端 `_inspect_safetensors`）自动识别已有权重中的 `LoRA / LoHa / LoKr`、`DoRA` 与 `PRESET` 并在续炼时自动对齐。

## [1.1.0] - 2026-11-20

### Added
- **Windows 原生零黑框 `.exe` 启动器与统一图标**：新增开源可审计的 C# 原生 WinExe 启动器源码 [launcher/MossTrainerLauncher.cs](launcher/MossTrainerLauncher.cs)、1 秒透明本地编译器（[build_exe.bat](build_exe.bat) / [scripts/build_exe.ps1](scripts/build_exe.ps1)）、1024px 超采样专属多分辨率图标（[assets/moss_trainer.ico](assets/moss_trainer.ico)）与任务栏独立 `AppUserModelID` 分组。
- **全链路无黑框静默启动**：新增 [launch_ui.vbs](launch_ui.vbs) 与 [scripts/silent_launcher.pyw](scripts/silent_launcher.pyw)，全链路注入 `CREATE_NO_WINDOW` 彻底消除 CMD/PowerShell 窗口弹出，并在 `pythonw.exe` 运行时安全将子进程解释器映射回 `python.exe` 以保留实时日志管道。
- **宽屏防挤压 UI 与实时 Loss 折线图看板**：重构 [gui_local.py](gui_local.py) 左右分栏布局（双侧 `stretch="always"` + 左侧 `minsize=620` + 按钮优先排布防截断）；右侧新增纯原生 `tk.Canvas` 零依赖实时 Loss / EMA 折线图、4 枚实时 KPI 卡片与纤细进度条，支持 `metrics.jsonl` 增量读取与历史曲线自动载入。
- **全场景续炼与中途随时暂停存盘**：向 `KohakuMossTrainer` 注入 `KOHAKUMOSS_PAUSE_FILE` 即时存盘退出钩子，支持 `--train_mode {fresh,resume,init_from}`、断点步数智能顺延、`safetensors` 头部秒级嗅探自动对齐 Rank/Alpha，以及后台实时权重同步归档。
- **MOSS-Image-SFT 全谱系底模支持与显存档位**：本地与云端双轨全面支持 `Base` / `Heal` / `SFT` 三大变体自动探测与架构感知、自动开启非 Heal 底模 FP16 防溢出守卫、支持 `--vram_preset {auto,6g,8g,12g,16g}` 与 `--resolution {1024,768,512}` 切换。

### Fixed
- **全盘 9 项深度审计修复与 Windows 底层稳定性加固**：修复 Windows 进程树残留（`taskkill /F /T` + `WM_DELETE_WINDOW`）、修复 PowerShell 5.1 语法兼容性、修复空输入框 `TclError` 崩溃、修复 `clam` 主题下拉框对比度异常、修复多目录导出旧步数覆盖最新权重的问题；禁用 VAE `torch.compile` 规避 Triton UTF-8 崩溃，绑定纯 ASCII 路径 `C:/tmp/moss_pipeline` 缓存与 `TUNE="off"`，显式禁用训练期采样防止二次加载 Qwen 导致 OOM。

### Security
- **全面隐私脱敏与宽域自动探测**：全面清除历史硬编码私人路径、私人 Kaggle ID 与测试外链，实现全盘宽域自动探测 ComfyUI 环境与模型库。

## [1.0.0] - 2026-11-18

### Added
- 发布 OpenMOSS MOSS-Image 双轨（本地消费级显卡 + Kaggle 云端免费 T4）自动化 LoRA 训练管线首个正式稳定版。
- 提供 [gui_local.py](gui_local.py) 图形控制台、[train_local.py](train_local.py) 本地执行引擎、[train_moss.py](train_moss.py) 云端执行脚本及开箱即用 Kaggle Notebook 模板。
- 提供完整的中文部署手册、参数速查、超参数调优与排错矩阵文档体系。

[Unreleased]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.5.1...HEAD
[1.5.1]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.2.2...v1.3.0
[1.2.2]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/abysschaos/kaggle-moss-lora-pipeline/releases/tag/v1.0.0
