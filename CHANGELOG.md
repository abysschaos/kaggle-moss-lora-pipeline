# Changelog

本项目遵循 [Semantic Versioning 2.0.0 (语义化版本规范)](https://semver.org/lang/zh-CN/) 进行版本管理：
- **MAJOR（主版本号）**：不向后兼容的架构或配置格式变更
- **MINOR（次版本号）**：向后兼容的重大新功能模块上线
- **PATCH（修订号）**：向后兼容的缺陷修复、性能优化与文档完善

---

## [v1.2.1] - 2026-11-20

### 🛠️ `KohakuMossTrainer` 引擎深度适配加固
- **补齐 `FP16_FIX_KWARGS = {"c": 1 / 16}` 防溢出系数**：修复本地 `train_local.py` 与云端 `train_moss.py` 在启用 `FP16_FIX="online"`（如 `6g/8g/12g + FP16 + Base/SFT` 组合）时因缺少缩放系数参数触发引擎 `knobs.validate()` 报错的问题。
- **清除 `artist_from_folder` 空标注图目录名污染**：将本地与云端 `DATASETS` 配置中的 `"artist_from_folder"` 统一由 `"name"` 调整为 `None`，避免纯多子集目录下的无标注图片回退触发 `TagTemplate` 时把文件夹名（如 `char_bob`）误当画师标签拼接进训练文本。
- **保留 `.txt` 原始 `mtime` 保护增量扫描缓存**：`prepare_dataset_structure` 与 `prepare_multi_subset_dataset` 仅在目标 `.txt` 不存在或内容确实变化时执行写盘，防止每次重跑无故刷新时间戳导致 `kohakumoss_data` catalog 缓存失效。
- **幂等化 `objective.py` 兼容补丁**：改用正则归一化替换逻辑，彻底消除多次运行引发的 `if isinstance(sub, dict)` 嵌套叠加并自动清理既有冗余。

---

## [v1.2.0] - 2026-11-20

### ✨ 核心新增功能（多人物 / 多画风 / 概念 LoRA 与高阶适配器引擎）
- **多子集配比管理器（本地 GUI / CLI & 云端全同步）**：支持在单个训练任务中配置多个独立子目录，每个子集独立绑定专属触发词与独立 `Repeat` 重复权重，自动生成标准 Kohya `<repeat>_<subset_name>` 目录结构并独立隔离清理孤儿文件；GUI 新增深色 `Treeview` 子集管理表格、独立添加/编辑弹窗与【⚖️ 一键配平 Repeat】功能（按子集图片数反比自动均衡曝光占比），并与公式步数换算实时联动。
- **高阶适配器架构与目标注入层预设**：严格参照 `KohakuMossTrainer` 官方 `algo_*.py` 与 `export-to-comfy.md` 规范，双轨开放 **LoRA / LoHa / LoKr** 算法切换（自动注入对应安全内核参数 `ALGO_KWARGS`）、**DoRA** 权重分解（自动配对 `DORA_DETACH = True` 兼顾稳定性与省显存）、**LoRA+ 学习率倍率**（`LORAPLUS_RATIO`）以及 6 档 Transformer 注入层预设（`attn` / `attn-mlp` / `attn-mlp+modulation` / `attn-mlp+text-refiner` / `full-block` / `all`），并显式锁定 `SAVE_FUSED_QKV = False` 保障 ComfyUI 紧凑原生加载。
- **五大实战预设模板与续炼结构自动对齐**：GUI 顶部预设栏扩充为【🎨 艺术画风】、【👤 单人角色】、【👥 多人物配比】、【🎭 多画风配比】与【💡 概念/动作】；增强 `inspect_safetensors_lora` 支持自动识别已有权重中的 `LoRA / LoHa / LoKr`、`DoRA` 与 `PRESET` 并在续炼时自动对齐。

---

## [v1.1.0] - 2026-11-20

### ✨ 核心新增功能
- **Windows 原生零黑框 `.exe` 启动器与统一图标**：新增开源可审计的 C# 原生 WinExe 启动器源码（`launcher/MossTrainerLauncher.cs`）、一键本地编译器（`build_exe.bat` / `scripts/build_exe.ps1`，零第三方依赖 1 秒出包）、1024px 超采样专属多分辨率图标（`assets/moss_trainer.ico`）与任务栏独立 `AppUserModelID` 分组，支持一步直达本地 ComfyUI `pythonw.exe`。
- **全链路无黑框静默启动**：新增 `launch_ui.vbs` 与 `scripts/silent_launcher.pyw`，全链路注入 `CREATE_NO_WINDOW` 彻底消除 CMD/PowerShell 窗口弹出，并支持 `pythonw.exe` 运行时子进程安全回退至 `python.exe` 以保持实时日志管道完整。
- **宽屏防挤压 UI 与实时 Loss 折线图看板**：重构 `gui_local.py` 左右分栏布局（双侧 `stretch="always"` + 左侧 `minsize=620` + 按钮优先排布防截断）；右侧新增纯原生 `tk.Canvas` 零依赖实时 Loss / EMA 折线图、4 枚实时 KPI 卡片与纤细进度条，支持 `metrics.jsonl` 增量读取与历史曲线自动载入。
- **全场景续炼与中途随时暂停存盘**：向 `KohakuMossTrainer` 注入 `KOHAKUMOSS_PAUSE_FILE` 即时存盘退出钩子，支持 `--train_mode {fresh,resume,init_from}`、断点步数智能顺延、`safetensors` 头部秒级嗅探自动对齐 Rank/Alpha，以及后台实时权重同步归档。
- **MOSS-Image-SFT 全谱系底模支持**：本地与云端双轨全面支持 `Base` / `Heal` / `SFT` 三大变体自动探测与架构感知，自动开启非 Heal 底模在 FP16 下的防溢出在线修正守卫。
- **显存优化档位与多分辨率自由切换**：本地 GUI/CLI 与云端 Kaggle Notebook / `train_moss.py` 同步支持 `--vram_preset {auto,6g,8g,12g,16g}` 与 `--resolution {1024,768,512}`，引入 <=10GB 消费级显卡极限省显存方案（`ring2 + actoff + optcpu + gradoff`）。

### 🛠️ 修复与安全加固
- **全盘 9 项深度审计修复**：修复 Windows 进程树残留（`taskkill /F /T` + `WM_DELETE_WINDOW`）、修复 PowerShell 5.1 语法兼容性、修复空输入框 `TclError` 崩溃、修复 `clam` 主题下拉框对比度异常、修复多目录导出旧步数覆盖最新权重的问题。
- **Windows 底层稳定性修复**：禁用 VAE `torch.compile` 规避 Windows Triton UTF-8 解码崩溃，绑定纯 ASCII 路径 `C:/tmp/moss_pipeline` 缓存与 `TUNE=off`，显式禁用训练期采样防止二次加载 Qwen 文本编码器导致 OOM。
- **数据同步与彻底隐私脱敏**：自动清理缓存孤儿文件并支持原图集同名替换实时同步；全面清除历史硬编码私人路径、私人 Kaggle ID 与测试外链，实现全盘宽域自动探测 ComfyUI 环境与模型库。

---

## [v1.0.0] - 2026-11-18

### 🎉 首个正式稳定版
- 发布 OpenMOSS MOSS-Image 双轨（本地消费级显卡 + Kaggle 云端免费 T4）自动化 LoRA 训练管线。
- 提供 `gui_local.py` 图形控制台、`train_local.py` 本地执行引擎、`train_moss.py` 云端执行脚本及开箱即用 Kaggle Notebook 模板。
- 提供完整的中文部署手册、参数速查、超参数调优与排错矩阵文档体系。
