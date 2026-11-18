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
| **启动命令** | `.\scripts\push_to_kaggle.ps1` | `.\scripts\run_local.ps1` |

---

## 📂 项目结构概览

```
kaggle_moss_lora_pipeline/
├── README.md                      # 项目核心说明与双轨使用指南
├── requirements.txt               # 云端辅助工具与 CLI 依赖清单
├── requirements-local.txt         # 本地 Windows 平台依赖清单
├── kernel-metadata.json           # Kaggle CLI 专用元数据配置
├── train_moss.py                  # 生产级云端一键执行脚本 (Kaggle 专用)
├── train_local.py                 # 生产级本地消费级显卡执行引擎 (Windows 专属)
├── docs/
│   ├── local_deployment_guide.md  # 消费级显卡 8GB 显存控制与 BF16 本地部署深度指南
│   ├── troubleshooting_matrix.md  # 25 个迭代版本的 Exhaustive 排错矩阵与踩坑实录
│   ├── dit_flow_matching_guide.md # 5B DiT 与 Flow Matching 连续流训练超参数理论
│   └── comfyui_inference_guide.md # ComfyUI 官方原生推理规范 (CFG 3.0 / Shift 5.0)
└── scripts/
    ├── run_local.ps1              # 本地一键启动与硬件自检交互式脚本
    ├── push_to_kaggle.ps1         # 本地一键推送任务至 Kaggle 云端
    └── pull_weights.ps1           # 本地一键从 Kaggle 拉取训练产出物
```

---

## 💻 模式 B：本地消费级显卡极速部署 (Windows 11)

### 1. 适用场景
适用于拥有个人电脑或游戏笔记本（例如配备 RTX 4070 Laptop 8GB / RTX 4060 8GB / RTX 3060 12GB）的用户，追求本地数据隐私、免排队即时调试与 4.5 倍以上的极速迭代效率。

### 2. 环境安装
确保本地安装了支持 CUDA 的 PyTorch（建议 CUDA 12.1 或 12.4）：
```powershell
pip install -r requirements-local.txt
```

### 3. 一键启动
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

## 🛠️ 排错矩阵与技术实录 (Troubleshooting)

本项目在演进过程中完整跨越了从 Version 1 到 Version 25 的 25 个版本。关于完整的 17 项致命报错排查实录（包括 PyTorch 动态库脱节、Flow Matching 假性收敛、ComfyUI 键名桥接失配等），请详细查阅文档：
👉 [docs/troubleshooting_matrix.md](docs/troubleshooting_matrix.md)
