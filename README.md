# OpenMOSS MOSS-Image Dual-Track LoRA Training Pipeline

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base%20Model-MOSS--Image--heal-green.svg)](https://huggingface.co/KBlueLeaf/MOSS-Image-heal)
[![Framework](https://img.shields.io/badge/Framework-KohakuMossTrainer-orange.svg)](https://github.com/KohakuBlueleaf/KohakuMossTrainer)
[![Dual Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Windows%20Local-20BEFF.svg)](https://github.com/abysschaos/kaggle-moss-lora-pipeline)

一套专为 **OpenMOSS MOSS-Image (5B Single-Stream DiT)** 打造的工业级 LoRA 自动化炼制工程方案。依托 [KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) 架构，支持 **云端 Kaggle 免费算力** 与 **本地 Windows 消费级显卡 (RTX 4070/4060/3060 8GB~12GB)** 双轨运行。

---

## 🌟 双轨架构对比矩阵

| 评估维度 | 模式 A：云端 Kaggle 自动化模式 | 模式 B：本地消费级显卡极速模式 |
| :--- | :--- | :--- |
| **目标硬件** | Tesla T4 GPU (16GB VRAM, Turing 架构) | RTX 4070 Laptop / 4060 / 3060 (8GB~12GB VRAM, Ada / Ampere) |
| **计算精度** | FP16 全精度 (依托 MOSS-Image-heal 防溢出) | **原生硬件级 BF16** (Ada Lovelace Tensor Core, 零 NaN 风险) |
| **单步运算耗时** | ~42 秒 / step | **~8 - 10 秒 / step (性能跃升 4.5 倍)** |
| **600 步训练周期**| 约 7.5 小时 | **约 1.0 - 1.5 小时** |
| **8GB 显存控制** | 默认全载入离线缓存 | **CPU 路由提取文本 Embeddings (显存恒等于 0 GiB)** |
| **反向传播显存峰值** | 3.5 GiB | **3.5 GiB (在 8GB 显卡下保留 4.5GB 巨大冗余)** |
| **模型资产依赖** | 每次启动需从 HuggingFace 重新下载 | **直连本地 ComfyUI 现有模型库，零网络下载秒级启动** |
| **启动命令** | `.\scripts\push_to_kaggle.ps1` | `.\scripts\run_local.ps1` 或双击 `launch_ui.bat` |
| **自定义命名与触发词** | 配置文件中修改 `RUN_NAME` | **GUI / CLI 全自定义：自动创建专属产出子文件夹、同步权重名称、前置注入触发词** |

---

## 📂 项目结构概览

```
kaggle_moss_lora_pipeline/
├── README.md                      # 项目核心说明与双轨使用指南
├── install_windows.bat            # Windows 专属一键自动部署虚拟环境与 CUDA 依赖
├── launch_ui.bat                  # Windows 专属一键双击启动 GUI 控制面板
├── gui_local.py                   # 现代极简本地训练 GUI (图形化参数调节/模板切换/实时日志)
├── train_local.py                 # 生产级本地消费级显卡执行引擎 (Windows 专属)
├── train_moss.py                  # 生产级云端一键执行脚本 (Kaggle 专用)
├── requirements.txt               # 云端辅助工具与 CLI 依赖清单
├── requirements-local.txt         # 本地 Windows 平台依赖清单
├── kernel-metadata.json           # Kaggle CLI 专用元数据配置
├── docs/
│   ├── local_installation_guide.md # Windows 本地部署与环境安装完全手册 (零基础一键上手)
│   ├── parameter_configuration_reference.md # 参数调节位置与实操修改手册 (行号/CLI/配置速查)
│   ├── hyperparameter_tuning_guide.md # 画风与人物超参数调优指南 (Repeats/Epochs/打标解耦)
│   ├── local_deployment_guide.md  # 消费级显卡 8GB 显存控制与 BF16 本地部署深度指南
│   ├── troubleshooting_matrix.md  # 25 个迭代版本的 Exhaustive 排错矩阵与踩坑实录
│   ├── dit_flow_matching_guide.md # 5B DiT 与 Flow Matching 连续流训练超参数理论
│   └── comfyui_inference_guide.md # ComfyUI 官方原生推理规范 (CFG 3.0 / Shift 5.0)
└── scripts/
    ├── install.ps1                # 本地环境一键安装脚本 (PowerShell 版)
    ├── launch_ui.ps1              # 本地 GUI 启动脚本 (PowerShell 版)
    ├── run_local.ps1              # 本地一键启动与硬件自检交互式脚本 (CLI 版)
    ├── push_to_kaggle.ps1         # 本地一键推送任务至 Kaggle 云端
    └── pull_weights.ps1           # 本地一键从 Kaggle 拉取训练产出物
```

---

## 💻 模式 B：本地消费级显卡极速部署 (Windows 11)

### 1. 适用场景
适用于拥有个人电脑或游戏笔记本（例如配备 RTX 4070 Laptop 8GB / RTX 4060 8GB / RTX 3060 12GB）的用户，追求本地数据隐私、免排队即时调试与 4.5 倍以上的极速迭代效率。

### 2. 环境一键安装 (极简部署)
全新设备直接双击运行根目录下的 **`install_windows.bat`**（或执行 `.\scripts\install.ps1`）：
- 自动检测 Python 64-bit 与 Git 环境；
- 自动建立 `venv` 纯净隔离虚拟环境；
- 自动安装带 **CUDA 12.4** 的 PyTorch 与 TorchVision；
- 自动安装 `requirements-local.txt` 辅助依赖；
- 自动初始化预创标准资产目录（`models/`、`dataset/`、`output/`）。

详细图文指导与底模获取，请参考：
👉 [docs/local_installation_guide.md](docs/local_installation_guide.md)

### 3. 一键启动方式

#### 方式 1：双击图形化控制面板 (GUI 模式，强烈推荐)
直接双击项目根目录下的 **`launch_ui.bat`**（或在终端运行 `python gui_local.py`）：
- **训练器框架一键管理**：新增【🛠️ 训练器框架管理】卡片，自带就绪状态检测指示灯、一键打开放置目录、一键自动克隆/下载、离线防套娃放置指引与一键自愈补丁注入；
- **底模自选与目录直达**：支持自选单模型文件（DiT/文本编码器/VAE）或一键自动探测；每个组件均配备【📂 打开放置目录】按钮，点击即可直接在 Windows 窗口中直达目标文件夹；
- **自定义命名与专属子目录**：支持用户自定义 LoRA 名称并自动创建专属产出子文件夹（如 `output/makima_v1/`），自动同步权重命名与生成训练元数据摘要；
- **触发词智能前置注入**：支持用户自定义触发词，系统自动在训练缓存区将触发词前置拼接至标注文本最前，原始图集标注保持纯净无破坏；
- **全可视化操作**：自带 Windows 文件夹“浏览”选择器，选定图集后自动统计当前图片与 txt 数量；
- **模板一键切换**：内置【🎨 艺术画风模板】与【👤 人物角色模板】，单选自动填入最佳参数；
- **智能公式联动**：动态计算总样本曝光量，提供一键“按公式对齐”核心步数按钮；
- **实时流式控制台**：多线程终端监控面板，实时回传日志与进度，支持中途安全终止与完成提示；
- **无缝部署**：勾选“自动复制到 ComfyUI”，训练完毕后自动将权重直推 `models/loras/`。

#### 方式 2：终端交互式脚本 (CLI 模式)
在 PowerShell 终端中执行：
```powershell
.\scripts\run_local.ps1
```
脚本将自动探测您的 NVIDIA 显卡信息与显存总量、检索本地 ComfyUI 模型库路径及训练集，敲击回车即可全自动完成：
1. 建立 Kohya 规范目录结构；
2. 调度 CPU 执行 Qwen3.5-9B 文本特征离线提取（40 张图耗时 15 秒，显存占用 0 GiB）；
3. 调度 GPU 执行 FLUX.2 VAE 图像潜在向量提取并释放；
4. 启动 5B DiT 本地 BF16 硬件加速主训练循环；
5. 自动检索并在 `./output` 与 `ComfyUI/models/loras/` 中落盘标准的 `ness_style-comfy.safetensors` 与 `ness_style-lycoris.safetensors`。

### 4. 详细技术白皮书
深入理解 8GB 显存解耦机制与参数调优，请参考：
👉 [docs/local_deployment_guide.md](docs/local_deployment_guide.md)

---

## ☁️ 模式 A：云端 Kaggle 自动化部署

### 1. 适用场景
适用于本地显卡算力不足或显存过小（< 6GB），希望利用 Kaggle 每周免费 30 小时 GPU 算力的用户。

### 2. 配置与推送
1. 配置 Kaggle 凭据 `~/.kaggle/kaggle.json`；
2. 在 PowerShell 中运行：
```powershell
.\scripts\push_to_kaggle.ps1
```
3. 训练完成后，拉取成果包：
```powershell
.\scripts\pull_weights.ps1
```

---

## 🎨 ComfyUI 官方原生推理配置

将产出的 `ness_style-comfy.safetensors` 放置在 `ComfyUI/models/loras/` 目录下：

* **LoRA 强度 (Strength)**：建议设为 **0.6 ~ 0.8**。
* **CFG 引导值**：必须设为 **3.0 ~ 3.5**（严禁使用常规 SD 的 7.0+，否则会导致过饱和烧焦死黑）。
* **采样器与调度器**：采用 **Flow-matching Euler**，配合 **Shift = 5.0** 偏移量。
* **采样步数**：**>= 50 步**。
* **负向提示词**：保持 **留空 (Empty)**。
* **触发词 (Trigger)**：配合提示词自然语言起手，例如：`A digital anime painting of ..., in the style of ness_style`。

---

## 📖 核心技术文档体系 (Documentation)

1. **Windows 本地部署与环境安装完全手册 (New)**：
   包含前置硬件环境自检、Python/Git 准备、`install_windows.bat` 一键部署、底模下载源与放置目录详解，以及 ComfyUI 现有资产零拷贝直连指南：
   👉 [docs/local_installation_guide.md](docs/local_installation_guide.md)
2. **参数调节位置与实操修改手册 (Where & How to Configure)**：
   精准罗列图片重复次数（Repeats）、训练轮数（Epochs）、优化步数（Steps）、Rank/Alpha、学习率等在本地与云端各文件中的确切行号与命令行传参方法：
   👉 [docs/parameter_configuration_reference.md](docs/parameter_configuration_reference.md)
2. **超参数调优指南 (画风 vs 人物)**：
   深入推演 Token Packing 批次换算机制，系统对比画风与人物在图片重复次数、轮数、步数、Rank/Alpha 以及打标剪枝（Token Pruning）上的本质区别：
   👉 [docs/hyperparameter_tuning_guide.md](docs/hyperparameter_tuning_guide.md)
2. **消费级显卡本地部署指南**：
   详解 8GB 显存三阶解耦、CPU 文本特征路由与 BF16 硬件加速原理：
   👉 [docs/local_deployment_guide.md](docs/local_deployment_guide.md)
3. **排错矩阵与踩坑实录 (Troubleshooting)**：
   25 个迭代版本沉淀的 17 项致命报错排查实录（包括 PyTorch 动态库脱节、Flow Matching 假性收敛、ComfyUI 键名桥接失配等）：
   👉 [docs/troubleshooting_matrix.md](docs/troubleshooting_matrix.md)
4. **DiT 与 Flow Matching 理论**：
   👉 [docs/dit_flow_matching_guide.md](docs/dit_flow_matching_guide.md)
5. **ComfyUI 官方原生推理规范**：
   👉 [docs/comfyui_inference_guide.md](docs/comfyui_inference_guide.md)
