# OpenMOSS MOSS-Image Kaggle LoRA Training Pipeline

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base%20Model-MOSS--Image--heal-green.svg)](https://huggingface.co/KBlueLeaf/MOSS-Image-heal)
[![Trainer Framework](https://img.shields.io/badge/Framework-KohakuMossTrainer-orange.svg)](https://github.com/KohakuBlueleaf/KohakuMossTrainer)
[![Target Platform](https://img.shields.io/badge/Target-Kaggle%20Tesla%20T4-20BEFF.svg)](https://www.kaggle.com)

一套专为 **OpenMOSS MOSS-Image (5B Single-Stream DiT)** 打造的工业级云端 LoRA 自动化炼制工程方案。依托 [KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) 架构，在免费的 **Kaggle Tesla T4 (16GB VRAM)** 实例上实现了 **零爆显存、全精度浮点、双离线缓存、大轮次深度收敛** 的完整闭环。

---

## 🌟 核心特性与工程亮点

1. **底模升级为官方修复版 ([KBlueLeaf/MOSS-Image-heal](https://huggingface.co/KBlueLeaf/MOSS-Image-heal))**：
   彻底消除了官方原版 `OpenMOSS-Team/MOSS-Image-Base` 中高达 58,411 的深层残差流激活尖峰，使模型在 FP16 全精度运算下具备极致的数值稳定性，彻底杜绝了半精度溢出与 NaN 崩溃。
2. **前置双离线特征提取与极限显存解耦**：
   在主训练开始前，一次性调用 9.7GB 的 Qwen3.5-9B 文本编码器与 FLUX.2 VAE 完成全量数据集的 Text Embeddings 与 Image Latents 离线特征落盘，随后将其彻底从显存中物理卸载，将主训练反向传播显存峰值严格压低至 **3.5 GiB**，在 16GB 显存环境下留出巨大的安全冗余。
3. **历经 25 个版本沉淀的全链路自愈与环境补丁**：
   内置自动化热补丁，解决 PyTorch 2.11 加速器接口缺失 (`empty_host_cache`)、TorchVision C++ NMS 算子兼容、调度器字典类型防御、配置文件临时参数磁盘物理清洗等 17 项已知环境断言。
4. **Token Packing 批次机制与大轮次深度收敛**：
   针对 40 张小样本画风数据集，启用 `MAX_TOKENS = 16384`（单步打包约 4 张 1024 完整图文），总训练步数扩展至 **600 步**（累计 2400 次单图图文迭代，等价于 **60 个完整 Epoch**），配合 **LoRA Alpha = 32.0**（满额 1.0 注入强度），彻底击穿 5B 底模的原生风格惯性。
5. **ComfyUI 官方原生加载器完美对齐**：
   严格遵循 ComfyUI 底层 `lora_unet_` 自动桥接机制，导出即可直接连接官方标准 `Load LoRA` 与 `LycorisLoader` 节点，无需任何魔改或插件。

---

## 📂 项目结构

```
kaggle_moss_lora_pipeline/
├── README.md                      # 项目核心说明与使用指南
├── kernel-metadata.json           # Kaggle CLI 专用元数据配置
├── train_moss.py                  # 生产级云端一键执行脚本 (包含全套自愈 Hook)
├── requirements.txt               # 辅助工具依赖清单
├── .gitignore                     # Git 忽略规则
├── docs/
│   ├── troubleshooting_matrix.md  # 25 个迭代版本的 Exhaustive 排错矩阵与踩坑实录
│   ├── dit_flow_matching_guide.md # 5B DiT 与 Flow Matching 连续流训练超参数理论
│   └── comfyui_inference_guide.md # ComfyUI 官方原生推理规范 (CFG 3.0 / Shift 5.0)
└── scripts/
    ├── push_to_kaggle.ps1         # 本地一键推送任务至 Kaggle 云端
    └── pull_weights.ps1           # 本地一键从 Kaggle 拉取训练产出物
```

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备
确保本地安装了 Kaggle 官方 CLI，并将您的 `kaggle.json` 凭据放置于 `~/.kaggle/kaggle.json`：
```bash
pip install -r requirements.txt
```

### 2. 配置数据集
准备符合 Kohya 规范的图集文件夹（例如包含 40 张 `.jpg` 与配套同名英文自然语言散文 `.txt`），打包为 `dataset.zip` 并上传至支持直链的云存储服务，在 `train_moss.py` 中更新下载链接。

### 3. 一键推送到 Kaggle 启动训练
在 PowerShell 中执行一键推送脚本：
```powershell
.\scripts\push_to_kaggle.ps1
```
或者使用原生命令：
```bash
kaggle kernels push -p .
```

### 4. 训练监控与成果拉取
云端将全自动执行：
1. 双镜像容灾数据集拉取与校验；
2. 宿主机 PyTorch/TorchVision 算子防护与 KohakuMossTrainer 兼容补丁；
3. 下载 `KBlueLeaf/MOSS-Image-heal` 修复版底模与文本/VAE 权重；
4. 离线缓存文本与图像特征，并卸载编码器释放显存；
5. 执行 600 步（60 个完整 Epoch）主训练循环（耗时约 7 小时）；
6. 自动扫描并打包各阶段产出的 `.safetensors` 权重生成 `ness_style_lora_results.zip`。

待状态变更为 `KernelWorkerStatus.COMPLETE` 后，执行一键拉取：
```powershell
.\scripts\pull_weights.ps1
```

---

## 🎨 ComfyUI 推理配置要点

将产出的 `ness_style-comfy.safetensors` 或 `ness_style-lycoris.safetensors` 放置在 `ComfyUI/models/loras/` 目录下：

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

---

## 📄 开源许可证 (License)

本项目采用 [Apache-2.0 License](LICENSE) 开源许可证。
底模衍生自 [OpenMOSS MOSS-Image](https://github.com/OpenMOSS-Team/MOSS-Image) 与 [KBlueLeaf/MOSS-Image-heal](https://huggingface.co/KBlueLeaf/MOSS-Image-heal)。训练器底层依托 [KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer)。
