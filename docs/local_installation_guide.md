# OpenMOSS MOSS-Image Windows 本地部署与环境安装完全手册

本手册专门为希望在个人电脑（Windows 10/11 64位 + NVIDIA 消费级显卡）上部署并运行本训练项目的用户而编写。项目提供了一键自动化安装脚本，能够自动隔离环境并完成全套 CUDA 深度学习依赖的部署。

---

## 1. 硬件配置与系统要求

在部署前，请确保您的计算机满足以下基本要求：

| 评估维度 | 最低配置要求 | 推荐配置要求 |
| :--- | :--- | :--- |
| **操作系统** | Windows 10 64-bit (21H2+) | Windows 11 64-bit |
| **独立显卡 (GPU)**| NVIDIA GPU 具有 **6GB** 以上显存 (如 RTX 2060 6GB) | NVIDIA RTX 4070 / 4060 / 3060 (8GB ~ 12GB VRAM) |
| **显卡硬件架构** | Turing (RTX 20系列) 或 Ampere (RTX 30系列) | **Ada Lovelace (RTX 40系列)**，具备原生硬件级 BF16 加速 |
| **NVIDIA 驱动版本**| >= 528.33 | **>= 531.00 (推荐安装最新 Game Ready 或 Studio 驱动)** |
| **系统内存 (RAM)** | 16 GB | 32 GB (用于 Qwen3.5-9B 文本特征离线路由提取) |
| **可用磁盘空间** | 至少保留 30 GB 剩余空间 (用于模型底模与离线缓存) | 高速 NVMe SSD 固态硬盘 |

---

## 2. 前置软件准备 (只需两项)

若您的电脑为全新纯净系统，请先安装以下两个通用开发底座：

### 2.1 安装 Python (必须为 64 位版本)
1. 访问 Python 官方下载站：[https://www.python.org/downloads/](https://www.python.org/downloads/)；
2. 下载 **Python 3.10、3.11 或 3.12 (Windows 64-bit installer)**；
3. **重要安装细节**：双击运行安装程序时，**务必勾选界面底部的 “Add python.exe to PATH”**（将 Python 添加到系统环境变量），随后点击 “Install Now” 完成安装。

### 2.2 安装 Git for Windows
1. 访问 Git 官方下载页面：[https://git-scm.com/download/win](https://git-scm.com/download/win)；
2. 下载并按默认配置安装即可（用于后续组件拉取与更新）。

---

## 3. 一键全自动化安装流程 (小白首选)

本项目提供了一个全自动环境构建脚本 `install_windows.bat`，它会在项目根目录下建立一个纯净隔离的虚拟环境（`venv`），绝不污染或破坏您电脑上原本的其他 Python 软件。

### 步骤说明：
1. **下载项目源码**：
   - 若熟悉 Git，在终端执行：
     ```bash
     git clone https://github.com/abysschaos/kaggle-moss-lora-pipeline.git
     cd kaggle-moss-lora-pipeline
     ```
   - 若不熟悉 Git，直接在 GitHub 页面点击绿色的 **Code -> Download ZIP**，下载后解压到没有中文特殊乱码的磁盘路径。
2. **执行一键安装**：
   - 鼠标双击运行项目根目录下的 **`install_windows.bat`**。
3. **自动化安装内容**：
   - 脚本会自动自检 Python 64 位与 Git 环境；
   - 自动在本地创建独立的 `venv` 虚拟运行沙箱；
   - 自动安装搭载 **CUDA 12.4** 的 PyTorch 与 TorchVision 官方预编译核心包；
   - 自动安装 `requirements-local.txt` 涵盖的 Transformers、Accelerate、PEFT、Safetensors 等辅助工具链；
   - 自动初始化预创标准资产目录（`models/`、`dataset/`、`output/`）；
   - 执行 GPU 算力自检并打印当前显卡型号、VRAM 容量与 BF16 硬件支持状态；
   - 当终端打印出“`恭喜！本地部署环境已全部就绪！`”时，按任意键即可退出安装程序。

---

## 4. 底模与组件获取及放置指南

为了保持 Git 代码仓库的轻量化（仅包含纯代码与文档，不到 1MB），**模型底模不随代码仓库分发**。用户需要自行准备以下三个核心权重文件。

### 4.1 需要准备的核心模型文件清单

| 组件类别 | 推荐文件名 | 文件尺寸 | 放置的目标文件夹路径 | 官方下载地址 |
| :--- | :--- | :--- | :--- | :--- |
| **DiT 底模** | `moss_image_heal_bf16.safetensors` / `fp16` / `int8`<br>*(亦支持官方 `Base` 与 `MOSS-Image-SFT` 权重)* | ~5.4~10.2 GB | `models/diffusion_models/` | [HuggingFace: MOSS-Image-heal](https://huggingface.co/KBlueLeaf/MOSS-Image-heal/tree/main/comfyui) |
| **文本编码器** | `qwen3.5_9b_bf16.safetensors`<br>*(或 `qwen3.5_9b_int8_convrot.safetensors`)* | ~5~9.7 GB | `models/text_encoders/` | [HuggingFace: Comfy-Org MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/text_encoders) |
| **图像 VAE** | `flux2-vae_small.safetensors` | ~168 MB | `models/vae/` | [HuggingFace: Comfy-Org MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/vae) |

> **提示（国内镜像高速下载）**：
> 若您在直连 HuggingFace 时遭遇网络缓慢或限流，可将上述下载链接中的 `huggingface.co` 替换为国内镜像站 `hf-mirror.com`，例如：
> `https://hf-mirror.com/KBlueLeaf/MOSS-Image-heal/tree/main/comfyui`

---

## 5. 极速方案：零拷贝直接复用现有 ComfyUI 资产

如果您的电脑中**已经安装了 ComfyUI 或各类 WebUI 整合包**，并且已经在其中运行过 MOSS-Image，那么上述底模与文本编码器通常已经在您的硬盘中存在。

**您完全不需要重新下载这 20 多 GB 的庞大文件！**

### 复用操作方法：
1. 找到您电脑上 ComfyUI 的模型根目录（通常为 `.../ComfyUI/models`）；
2. 确认该目录下包含以下文件：
   - `diffusion_models/moss/moss_image_heal_bf16.safetensors`
   - `text_encoders/qwen3.5_9b_bf16.safetensors`
   - `vae/flux2-vae_small.safetensors`
3. 启动本项目的 GUI 控制面板，在“**ComfyUI模型库**”一栏中点击“浏览”，直接选中该 `models` 文件夹；
4. 项目引擎会自动检索并建立符号化映射，**全程零网络流量消耗，不额外占用磁盘空间，秒级就绪**。

---

## 6. 训练器框架准备 (KohakuMossTrainer)

本项目底层基于开源训练框架 [KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) 构建。为了让训练项目保持轻量纯净，训练器框架与核心底模一样采用模块化解耦设计。

针对用户关心的“是否需要自主下载”以及“如何正确放置”，系统提供了全方位的自动化与可视化一键指导：

### 6.1 用户是否需要自主下载？
**通常情况下不需要用户手动前往网页下载！**
- **方案一（GUI 一键自动拉取，推荐）**：只要您的网络能顺畅访问 GitHub，运行 `MOSS_LoRA_Trainer.exe`（或 `launch_ui.vbs`）进入控制面板后，点击左侧【🛠️ 训练器框架管理】卡片中的【⬇️ 一键自动克隆/下载】按钮，系统将在后台自动完成源码克隆并全自动注入所有自愈补丁。
- **方案二（启动训练自动补齐）**：若您未手动下载直接点击【🚀 启动本地极速训练】，系统在校验环境时若发现缺少训练器，也会自动在终端中尝试拉取并修补。
- **方案三（手动离线下载，适合网络受限场景）**：若您的电脑无法直连 GitHub，您可以在有网络的环境下自主下载源码压缩包进行手动离线部署。

---

### 6.2 手动离线下载与放置步骤 (防“套娃目录”避坑指南)

若您选择自主下载或通过外部拷贝放置训练器，请严格按照以下规范执行：

1. **获取源码压缩包**：
   - 访问官方仓库：[https://github.com/KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer)；
   - 点击绿色按钮 **Code -> Download ZIP**，下载最新源码包（或通过 Git 克隆）。
2. **目标放置路径**：
   - 必须放置在项目根目录下的 **`KohakuMossTrainer/`** 文件夹中。
   - **一键直达**：在 GUI 界面上点击【📂 打开训练器放置目录】按钮，系统会自动创建并直接弹出该文件夹窗口。
3. **重要避坑点：严禁双层嵌套（套娃目录）**：
   解压软件解压 ZIP 时通常会默认生成一级同名父文件夹（例如 `KohakuMossTrainer-main`）。如果您直接将该文件夹复制进去，路径会变成：
   - ❌ 错误结构：`项目根目录/KohakuMossTrainer/KohakuMossTrainer-main/pyproject.toml`（此结构下训练引擎将无法识别并报错）。
   - ✅ 正确结构：必须把压缩包内部的内容直接解压或移动到 `KohakuMossTrainer/` 根层级下，确保打开 `KohakuMossTrainer/` 第一层就能直接看到核心项目文件：
     ```
     项目根目录/KohakuMossTrainer/
     ├── pyproject.toml      <--- 核心标记文件 (必须位于第一层)
     ├── configs/            <--- 训练与网络配置
     ├── scripts/            <--- 启动与转换脚本
     ├── src/                <--- 训练核心源码
     └── README.md
     ```
4. **一键注入自愈兼容补丁**：
   完成解压放置后，在 GUI 界面点击【🩹 检查并注入补丁】按钮。系统会自动对源码注入 17 项已知环境自愈修复（放宽 PyTorch 限制、防御 PyTorch 2.11 的 empty_host_cache 接口缺失、以及优化器调度器字典解析等问题），提示“自愈兼容补丁注入成功”即代表完全就绪。

---

### 6.3 GUI 控制面板的一键指导功能概览

在图形化控制台（`MOSS_LoRA_Trainer.exe` / `launch_ui.vbs`）中，专门设计了【🛠️ 训练器框架管理 (KohakuMossTrainer)】专属卡片：
- **实时状态灯**：绿色显示“✅ 框架已就绪”或黄色显示“❌ 未发现框架”；
- **【📂 打开训练器放置目录】**：一键在 Windows 资源管理器打开目录，同时自动生成《训练器放置说明.txt》；
- **【⬇️ 一键自动克隆/下载】**：多线程后台执行 `git clone`，并在下载成功后自动打补丁，全程在右侧控制台展示日志；
- **【🩹 检查并注入补丁】**：为手动放置源码的用户提供一键自愈注入；
- **【🌐 访问官方 GitHub】**：点击自动调用系统默认浏览器直达官方源码主页。

---

## 7. 启动控制面板与开始训练

完成上述准备后，即可开启本地极速训练流程：

1. **构建并启动零黑框图形化面板**：
   - **最推荐**：首次使用双击项目根目录 **`build_exe.bat`**，1 秒无第三方依赖编译生成 **`MOSS_LoRA_Trainer.exe`**，随后直接双击即可零黑框秒开界面（还可右键固定到任务栏）。
   - **免编译备选**：也可直接双击根目录 **`launch_ui.vbs`**（或兼容入口 `launch_ui.bat`）静默拉起控制面板。
2. **选择图集或开启多子集配比**：
   - 单图集模式：在“训练图片目录”点击“浏览”，选择包含成对 `.jpg/.png` 与同名 `.txt` 标注的文件夹，界面自动统计图文数量；
   - 多子集模式：勾选【启用多子集配比模式】，添加多个人物/画风/概念子目录并各自填写专属触发词与 `Repeat` 权重，支持点击【⚖️ 一键配平 Repeat】自动按图数反比均衡曝光占比。
3. **加载预设与配置适配器**：
   - 在顶部一键套用五大实战模板：【🎨 艺术画风】、【👤 单人角色】、【👥 多人物配比】、【🎭 多画风配比】或【💡 概念/动作】；
   - 按需选择显存档位（`auto / 6g / 8g / 12g / 16g`）、目标分辨率（`1024 / 768 / 512`）、适配器算法（`LoRA / LoHa / LoKr`）、**DoRA**、**LoRA+ 倍率**与注入层预设，并点击【⚡ 按公式对齐】计算核心步数。
4. **启动训练与实时监控**：
   - 点击【🚀 启动本地极速训练】。右侧面板会实时渲染 Loss / EMA 折线图、4 枚核心 KPI 指标卡片与控制台流式日志；训练期间可随时点击【⏸️ 暂停并存盘】安全保存权重退出，后续使用“断点续炼”无缝接着跑。
5. **获取成果**：
   - 训练完成后，产出的 `*-comfy.safetensors` 与 `*-lycoris.safetensors` 会自动归档至 `./output/<lora_name>/` 目录，并按勾选自动同步到 `ComfyUI/models/loras/` 中。
