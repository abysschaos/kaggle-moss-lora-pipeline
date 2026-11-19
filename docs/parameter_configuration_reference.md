# MOSS-Image 项目参数调节位置与实操修改手册

本手册专门罗列项目中所有超参数（图片重复次数、轮数、步数、Rank/Alpha、学习率、路径等）在**各个文件中的具体行号、命令行开关以及修改方法**，供日常调整参数时快速查阅。

---

## 1. 核心参数速查总表

| 参数名称 | 物理含义 | 推荐值 | 本地命令行开关 | 本地源码定义行 (`train_local.py`) | 云端源码修改行 (`train_moss.py`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA名称与文件夹** | 产出权重文件前缀与专属保存子文件夹名 | 字符串 | `--lora_name` (或 `--run_name`) | 第 182 行 (`default="ness_style"`) | 第 283 行 (`RUN_NAME = ...`) |
| **自定义触发词** | 自动注入每张图片标注首部的触发词 | 字符串 | `--trigger_word` | 第 184 行 (`default=""`) | 写入标注文本首部 |
| **单图重复次数** | 每张图片在单轮中的采样倍数 | 4 ~ 15 | `--repeats <int>` | 第 188 行 (`default=10`) | 第 65 行 (目录名 `10_...`) |
| **训练轮数 (Epochs)** | 全量数据集完整遍历轮次 | 3 ~ 8 | `--epochs <int>` | 第 190 行 (`default=6`) | 影响总迭代次数换算 |
| **优化器步数 (Steps)** | 梯度反向传播总更新步数 | 200 ~ 800 | `--steps <int>` | 第 192 行 (`default=600`) | 第 288 行 (`STEPS = 600`) |
| **LoRA 矩阵秩 (Rank)** | 权重低秩适应矩阵维度 | 16 ~ 64 | `--rank <int>` | 第 194 行 (`default=32`) | 第 286 行 (`DIM = 32`) |
| **LoRA 缩放倍率 (Alpha)**| 权重注入强度系数 | 16.0 ~ 64.0 | `--alpha <float>` | 第 196 行 (`default=32.0`)| 第 287 行 (`ALPHA = 32.0`) |
| **学习率 (LR)** | 优化器学习速率 | 8e-5 ~ 1e-4 | `--lr <float>` | 第 198 行 (`default=1e-4`) | 第 289 行 (`LR = 1e-4`) |
| **训练计算精度** | 浮点运算精度模式 | `auto` / `bf16` | `--precision` | `default="bf16"` | `--precision` (`auto`/`fp16`/`bf16`) |
| **显存优化档位** | 流式缓冲、卸载与最大 Token 预算 | `auto` (`6g`/`8g`/`12g`/`16g`) | `--vram_preset` | `default="auto"` | `--vram_preset` (`VRAM_PRESET`) |
| **训练目标分辨率** | 动态分桶像素面积池 (`TARGET_AREA`) | `1024` / `768` / `512` | `--resolution` | `default=1024` | `--resolution` (`RESOLUTION`) |
| **模型架构变体** | 切换 Base / Heal / SFT 底模分支 | `auto` / `heal` / `sft` | `--variant` | `default="auto"` | `--variant` (`heal`/`base`/`sft`) |
| **云端底模量化** | 云端拉取 int8 量化版或全精度底模 | `auto` (`int8`/`full`) | 自动识别本地文件 | 本地自动匹配 | `--base_quant` (`BASE_QUANT`) |
| **文本提取设备** | Qwen3.5-9B 特征提取宿主 | `cpu` / `auto` | `--text_device` | `default="cpu"` | `--text_device` (`auto`/`cuda`/`cpu`) |
| **训练/续炼模式** | 全新训练 / 断点续炼 / 基于已有 LoRA 继续训练 | `fresh` / `resume` / `init_from` | `--train_mode` | `default="fresh"` | `--train_mode` (`TRAIN_MODE`) |
| **断点续炼目录** | 指定中断或已完结的 `step_XXXXXXXX` 目录 | `latest` 或目录路径 | `--resume_from` | `default="latest"` | `--resume_from` (`RESUME_FROM`) |
| **基底 LoRA 路径** | 指定已训练结束的 `.safetensors` 权重继续微调 | `.safetensors` 路径 | `--init_from` | `default=""` | `--init_from` (`INIT_FROM`) |
| **自动存盘间隔** | 每隔 N 步自动保存一次中间 checkpoint 与权重 | `100` | `--save_every` | `default=100` | `--save_every` (`SAVE_EVERY`) |
| **数据集路径** | 输入图片与 txt 所在文件夹 | 绝对路径 | `--data_dir` | 第 176 行 | 第 65 行 (`target_data_dir`) |
| **ComfyUI模型库路径** | ComfyUI 模型存放根目录 | 绝对路径 | `--comfy_models_dir`| 第 178 行 | 第 61 ~ 63 行 (`/tmp/models`) |
| **DiT 底模单文件** | 自选 MOSS-Image 底模文件路径 | 绝对路径 | `--transformer_path` | 第 180 行 (`default=""`) | 自动下载 |
| **文本编码器单文件** | 自选 Qwen3.5-9B 编码器文件路径 | 绝对路径 | `--text_encoder_path` | 第 182 行 (`default=""`) | 自动下载 |
| **图像 VAE 单文件** | 自选 FLUX.2 VAE 权重文件路径 | 绝对路径 | `--vae_path` | 第 184 行 (`default=""`) | 自动下载 |

---

## 2. 模式 A：本地训练参数修改位置 (`train_local.py`)

本地运行提供两种调节方式：**第一种无需改代码（通过命令行传参，推荐）**；**第二种直接修改代码里的默认预设**。

### 方式一：命令行参数覆盖（最快捷）
直接在终端或 PowerShell 中附加参数运行，优先级高于源码中的默认值：
```powershell
python train_local.py `
    --data_dir "./dataset" `
    --lora_name "my_character_v1" `
    --trigger_word "my_character_char" `
    --repeats 10 `
    --epochs 6 `
    --steps 600 `
    --rank 32 `
    --alpha 32.0 `
    --lr 1e-4 `
    --precision bf16 `
    --text_device cpu `
    --copy_to_comfy_loras
```

### 方式二：直接在源码中修改默认值
若希望每次双击或不带参数运行时默认生效，直接打开 [train_local.py](../train_local.py)：

#### 1. 命令行默认值定义段落
找到 `def main():` 下方的参数解析代码：
```python
# 修改默认数据集路径 (默认通用相对路径 ./dataset)
parser.add_argument("--data_dir", type=str, default="./dataset", ...)

# 修改默认产出权重的前缀名
parser.add_argument("--lora_name", type=str, default="my_lora", ...)

# 第 185 行：修改默认单图重复次数 (Repeats)
parser.add_argument("--repeats", type=int, default=10, ...)

# 第 187 行：修改默认 Epochs
parser.add_argument("--epochs", type=int, default=6, ...)

# 第 189 行：修改默认核心训练步数 (Steps)
parser.add_argument("--steps", type=int, default=600, ...)

# 第 191 行：修改默认 LoRA Rank (Dim)
parser.add_argument("--rank", type=int, default=32, ...)

# 第 193 行：修改默认 LoRA Alpha
parser.add_argument("--alpha", type=float, default=32.0, ...)

# 第 195 行：修改默认学习率 (LR)
parser.add_argument("--lr", type=float, default=1e-4, ...)
```

#### 2. Kohya 数据集生成逻辑（第 137 ~ 140 行）
重复次数本质上是由数据集子文件夹名开头的数字决定的：
```python
# 第 139 行：
target_folder = workspace_data_dir / f"{repeats}_{run_name}"
```
KohakuMossTrainer 的底层数据读取器会自动识别 `f"{repeats}_..."` 并在数据迭代时按该倍率采样。

#### 3. 动态配置生成模板与学习率调度器（第 331 ~ 353 行）
`train_local.py` 会将参数动态拼装写入训练配置文件：
```python
# 第 331 ~ 353 行：
train_cfg_code = f"""
from kohakuengine import use_config

use_config("../moss/recipes/recipes_lora_16g_fp16.py")
use_config("{args.run_name}_local_data.py")

RUN_NAME = "{args.run_name}"
TRANSFORMER = "{transformer_filename}"
FP16_FIX = "none"
PRECISION = "{args.precision}"
DIM = {args.rank}
ALPHA = {args.alpha}
STEPS = {args.steps}
LR = {args.lr}
SAVE_MINUTES = 20
SAVE_FORMAT = "both"
SCHEDULE = {{
    "mode": "cosine",
    "warmup": 30,          # <- 可以在此处修改预热步数 (Warmup Steps)
    "end": -1,
    "min_value": 0.1       # <- 可以在此处修改最终学习率衰减下限比例
}}
"""
```

---

## 3. 模式 B：PowerShell 交互脚本中的预设位置 (`scripts/run_local.ps1`)

如果您使用一键启动脚本 [scripts/run_local.ps1](../scripts/run_local.ps1)：

### 1. 默认路径修改（第 41 ~ 43 行）
打开该脚本，在开头可直接指定您的常用路径：
```powershell
# 默认图片与标注所在路径 (默认通用相对路径)
$defaultDataDir = ".\dataset"

# 本地 ComfyUI 模型库路径 (默认通用相对路径或由系统动态宽域探测)
$defaultComfyModels = ".\models"

# 产出保存目录
$outputDir = ".\output"
```

### 2. 交互式调参调用位置（第 70 ~ 87 行）
在终端运行脚本时，直接输入 `custom` 敲回车，即可在终端界面动态输入新的步数或数据集路径。

---

## 4. 模式 C：云端 Kaggle 自动化脚本修改位置 (`train_moss.py`)

在推送到 Kaggle 之前，若要更改云端运行的超参数，直接在 [train_moss.py](../train_moss.py) 中修改以下两处代码：

### 1. 修改单图重复次数 (Repeats)
在 `train_moss.py` 第 65 行：
```python
# 第 65 行：
target_data_dir = "/tmp/dataset/10_ness_style"
```
- 若要将单图重复次数从 10 改为 5，直接将该路径重命名为：`/tmp/dataset/5_ness_style`；
- 若要改为 15 次，重命名为：`/tmp/dataset/15_ness_style`。

### 2. 修改核心训练超参数（第 283 ~ 298 行）
在 `train_moss.py` 中找到配置拼装段落：
```python
# 第 283 ~ 298 行：
RUN_NAME = "ness_style"                            # 输出权重命名前缀
TRANSFORMER = "moss_image_heal_fp16.safetensors"
FP16_FIX = "none"
DIM = 32                                           # <- 修改 LoRA Rank (16 ~ 64)
ALPHA = 32.0                                       # <- 修改 LoRA Alpha (16.0 ~ 64.0)
STEPS = 600                                        # <- 修改总训练步数 (200 ~ 800)
LR = 1e-4                                          # <- 修改学习率 (8e-5 ~ 1e-4)
SAVE_MINUTES = 30                                  # 中途保存权重的间隔分钟数
SAVE_FORMAT = "both"                               # 同时导出 comfy 与 lycoris 权重
SCHEDULE = {
    "mode": "cosine",                              # 学习率衰减曲线类型
    "warmup": 30,                                  # 预热步数
    "end": -1,
    "min_value": 0.1                               # 衰减到底时的最低学习率比例 (0.1 即 1e-5)
}
```

修改完成后，在本地执行 `.\scripts\push_to_kaggle.ps1` 即可将新参数推送到 Kaggle 运行。
