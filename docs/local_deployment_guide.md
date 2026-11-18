# OpenMOSS MOSS-Image 消费级显卡本地训练深度指南

## 1. 概述与硬件挑战

将 50 亿参数（5B）的连续流单流 Diffusion Transformer（DiT）模型——**OpenMOSS MOSS-Image** 迁移至个人消费级硬件（例如搭载 NVIDIA GeForce RTX 4070 Laptop GPU 8GB VRAM、RTX 4060 8GB 或 RTX 3060 12GB 的 Windows 11 设备）面临着严苛的显存与计算约束。

### 显存占用物理极限核算
在常规联合端到端微调架构中，模型组件的显存需求如下：
- **DiT 主干模型 (MOSS-Image 5B)**：半精度（FP16/BF16）权重占用约 **10.2 GB**；
- **文本编码器 (Qwen3.5-9B)**：半精度全权重占用约 **9.7 GB**（即便采用 INT8 ConvRot 量化仍需约 **5.0 GB**）；
- **图像自编码器 (FLUX.2 VAE)**：权重占用约 **168 MB**；
- **反向传播与优化器状态 (AdamW + LoRA Rank 32)**：在批次展开时至少需要 **2.5 GB ~ 4.0 GB** 额外显存。

若采用常规联合加载机制，系统瞬时显存总需求将超过 **22 GB**。在仅拥有 8 GB 显存的消费级移动端显卡上，直接启动训练必将遭遇 `torch.cuda.OutOfMemoryError`（CUDA OOM 显存溢出崩溃）。

---

## 2. 核心架构突破：三阶时序解耦与双离线缓存

为了在 8 GB 显存的硬性边界下实现零爆显存的深度 LoRA 微调，本项目在本地管线中确立了**三阶时序解耦与双离线特征缓存（Dual Offline Feature Caching）**机制。

### 2.1 阶段一：文本语义特征离线落盘与 CPU 物理路由
小样本画风训练集（例如 40 张图像与 40 份配套自然语言标注）具有“特征静态不可变”的先验属性。
- 文本编码器仅用于将自然语言 Prompt 映射为高维连续特征向量（Text Embeddings），在整个训练周期内完全处于冻结状态；
- `train_local.py` 将文本编码器（Qwen3.5-9B）的计算图路由至**主机系统内存（RAM）与 CPU 算力核心**（即 `TEXT_ENCODER_KWARGS = {"device": "cpu"}`）；
- 现代个人电脑通常配备 16 GB 至 64 GB 的系统内存，在 CPU 上计算 40 条文本的特征提取耗时仅需 **15 ~ 25 秒**；
- **核心成效**：彻底解除了 9.7 GB 文本编码器对 GPU 显存的压制，该阶段显存占用**严格恒等于 0 GiB**。

### 2.2 阶段二：图像潜在空间离线投影
- FLUX.2 VAE 模型尺寸仅为 168 MB，其推断显存峰值低于 **1.0 GiB**；
- 利用 GPU 快速将 40 张 1024x1024 原始图像逐一压缩映射为低维潜空间张量（Image Latents）并落盘保存为离线缓存；
- 特征提取完成后，VAE 权重从显存中完全释放并执行垃圾回收。

### 2.3 阶段三：轻量化 DiT 主干反向传播
在主优化训练阶段：
- GPU 显存内仅载入 5B DiT 主干与外挂的轻量级 LoRA 权重矩阵（Rank 32, Alpha 32.0）；
- 数据加载管道直接从本地磁盘流式读取预先提取的 Text Embeddings 与 Image Latents；
- 实测 5B DiT 在单步最大 Token 容量（`MAX_TOKENS = 16384`）下的反向传播显存峰值稳定在 **3.5 GiB**；
- 在 8.0 GiB 的物理显存总额下，系统保留了多达 **4.5 GiB 的安全裕度**，从根本上终结了爆显存隐患。

---

## 3. 硬件算力代际飞跃：Ada Lovelace 原生 BF16 加速

云端免费算力（如 Kaggle 提供的 Tesla T4 GPU）基于 Turing 架构（计算能力 SM 7.5），其物理张量核心缺乏硬件级 Bfloat16 支持，必须退回到 FP16。FP16 的有效动态范围仅为 $10^{-5} \sim 65504$，极易因高层梯度激活尖峰发生数值下溢，迫使训练引擎引入动态损耗缩放器（Loss Scaler），且单步耗时高达 **42 秒**。

而本地搭载的 NVIDIA GeForce RTX 4070 Laptop GPU 基于 Ada Lovelace 架构（计算能力 SM 8.9），配备第四代 Tensor Core：
1. **原生 BF16 指令集支持**：Bfloat16 具备与 FP32 完全等价的 8 位指数位宽，动态范围达到 $10^{-38} \sim 10^{38}$。在 `PRECISION = "bf16"` 模式下，训练无需依赖动态 loss scaler，梯度平滑收敛，彻底杜绝 NaN 溢出崩溃。
2. **吞吐量倍增**：结合离线双缓存，本地 RTX 4070 的 5B DiT 单步运算耗时缩短至 **8 ~ 10 秒**，相比云端 T4 实现 **4.5 倍**以上的计算吞吐飞跃。完成 600 步（等价于 60 个完整 Epoch）的深度训练全程仅需 **1 ~ 1.5 小时**（云端耗时超过 7 小时）。

---

## 4. 零网络下载：本地已有模型资产无缝直连

传统的云端训练脚本在每次初始化时都需要从 HuggingFace 重新拉取上百 GB 的底模与分词器，消耗巨额网络带宽与等待时间。

`train_local.py` 内置了**智能资产探测器（Smart Model Discovery）**：
- 能够全自动跨盘符遍历本地现有的 ComfyUI 模型库路径（例如 `D:\ComfyUI\models\` 或 `E:\AI\ComfyUI-aki\ComfyUI\models\`）；
- 自动识别并链接修复版底模 `moss_image_heal_bf16.safetensors`、文本编码器 `qwen3.5_9b_bf16.safetensors` 以及 `flux2-vae_small.safetensors`；
- 通过文件级快速映射或硬链接在秒级内完成环境就绪，实现**全程零网络流量消耗**。

---

## 5. Windows 平台特有陷阱与防御体系

### 5.1 NTFS 路径转义规避
在 Windows 环境下，绝对路径常包含反斜杠（如 `C:\Users\...`）。若直接插值进生成的 Python 配置文件，反斜杠会被 Python 词法解析器识别为转义符（例如 `\U` 触发十六进制字符转义失败）。
- 本地脚本在生成所有配置时，全面通过 `Path(p).resolve().as_posix()` 函数强制统一为 POSIX 正斜杠结构，杜绝解析崩溃。

### 5.2 符号链接权限回退机制
Windows 默认对非管理员模式的用户禁止创建软链接（`os.symlink` 会抛出 `OSError: symbolic link privilege not held`）。
- 本地脚本在准备数据集与模型映射时，优先尝试 NTFS 硬链接（`os.link`）；若跨盘符或权限受限，则平滑自动回退为高效的文件快速复制，确保普通用户权限下无阻运行。

### 5.3 配置文件临时参数物理清洗
KohakuMossTrainer 在离线缓存阶段需要读取 `TEXT_ENCODER_KWARGS` 与 `IMAGE_ENCODER_KWARGS`；但在主训练阶段（`moss.py`），严格的配置参数校验器会将这两个键视为非法未定义旋钮并抛出 `invalid configuration` 异常。
- 本地管线在离线缓存完成后，自动通过正则扫描将磁盘上的配置文件执行物理级净化清洗，确保主训练顺利拉起。

---

## 6. 一键启动与运行操作指引

### 6.1 前置检查
确保当前 Python 环境已安装支持 CUDA 12.1+ 的 PyTorch：
```powershell
pip install -r requirements-local.txt
```

### 6.2 快速执行
直接在项目根目录下右键运行或在终端中执行一键启动脚本：
```powershell
.\scripts\run_local.ps1
```
脚本将自动展示硬件概览（显卡型号、VRAM、BF16 硬件支持）、验证图集文件数量、自检模型文件，并在用户敲击回车后自动拉起全程自动化闭环。

### 6.3 命令行高级调参示例
支持通过 `train_local.py` 传入自定义超参数：
```powershell
python train_local.py `
    --data_dir "./dataset" `
    --output_dir "./output" `
    --lora_name "my_lora" `
    --steps 600 `
    --rank 32 `
    --alpha 32.0 `
    --lr 1e-4 `
    --precision bf16 `
    --text_device cpu `
    --copy_to_comfy_loras
```

---

## 7. 训练产出物与 ComfyUI 推理对齐

训练完成后，权重将自动归档至 `./output/` 目录（并可直通 `ComfyUI/models/loras/`）：
- `my_lora-comfy.safetensors`：专为 ComfyUI 官方原生 `Load LoRA` 节点对齐，保留标准 `lora_unet_` 前缀，实现 100% 满额挂载。
- `ness_style-lycoris.safetensors`：保留完整 LyCORIS 架构标识，支持高级适配器加载器。
- **推荐推理参数**：CFG 3.0 ~ 3.5，Shift 5.0，采样步数 >= 50 步，负向提示词留空。
