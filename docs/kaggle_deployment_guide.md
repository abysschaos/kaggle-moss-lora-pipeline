# OpenMOSS MOSS-Image Kaggle 云端部署与自定义训练完全手册 (无需任何 AI Agent)

本手册专为希望在 **Kaggle 免费云端 GPU (NVIDIA Tesla T4 16GB)** 上训练自定义 OpenMOSS MOSS-Image LoRA 的个人用户编写。

无需安装复杂的本地 AI Agent，也无需对电脑配置有任何要求（手机、轻薄本均可操作）。只要拥有一个 Kaggle 账号，即可按照自己的图集与参数设置完成全流程训练。

---

## 🌟 为什么选择 Kaggle 云端训练？

| 评估维度 | 本地个人电脑部署 | Kaggle 云端免费部署 |
| :--- | :--- | :--- |
| **显卡硬件** | 依赖本地显卡 (如 RTX 4070/4060 8GB) | **NVIDIA Tesla T4 (16GB 大显存)** |
| **硬件成本** | 需购置昂贵的游戏本或台式机 | **完全免费** (Kaggle 每周赠送 30 小时 GPU 算力) |
| **设备负担** | 训练时风扇狂转、无法流畅玩游戏或办公 | **云端静默运行**，关机或关闭网页依然在后台继续炼 |
| **计算精度** | 8GB 显卡需锁定 BF16 + CPU 提取文本 | **16GB 大显存**，支持全精度 FP16 高速并发提取 |
| **模型下载** | 需消耗本地 20GB 磁盘空间并经历漫长下载 | **Kaggle 直连 HuggingFace**，千兆带宽 1~2 分钟极速下完底模 |

---

## 方案一：Kaggle 纯网页端交互式部署 (零基础小白首选，强烈推荐)

本方案全程在浏览器中通过鼠标点击与极简配置完成，**本地不需要配置任何 Python 或 Git 环境**。

### 第一步：准备并上传您的训练图集 (1 分钟)

1. **整理图集**：
   在本地电脑上准备好您的训练图片（`.jpg`、`.png` 或 `.webp`）以及同名的自然语言标注文本（`.txt`）。
   例如文件夹中有：
   ```
   my_dataset/
   ├── 001.jpg
   ├── 001.txt
   ├── 002.jpg
   ├── 002.txt
   └── ...
   ```
2. **打包为 ZIP**：
   将这批图片与文本全选，压缩打包为一个 `dataset.zip`（或者直接将文件夹压缩为 `my_character_data.zip`）。
3. **上传至 Kaggle**：
   - 访问 [Kaggle 官网](https://www.kaggle.com/) 并登录；
   - 点击左侧导航栏的 **+ Create** -> **New Dataset**；
   - 将刚才打包好的 ZIP 文件拖入网页上传区域；
   - 在顶部输入数据集标题（例如：`my-lora-dataset`）；
   - 在右下角选择 **Private**（私有，保护您的数据隐私），然后点击 **Create**。
   - 等待约 10 秒钟，数据集上传创建完成。

---

### 第二步：创建 Notebook 并配置 GPU 算力

1. 点击 Kaggle 页面左侧的 **+ Create** -> **New Notebook**，进入在线编程界面。
2. **配置运行环境 (点击顶部菜单栏中的 Settings)**：
   - **Accelerator (加速硬件)**：点击顶部菜单 **Settings** -> **Accelerator**，选择 **GPU T4 x1** (16GB VRAM)；
   - **Internet (网络开关)**：点击顶部菜单 **Settings** -> **Internet**，切换为 **Internet on** (必须开启网络权限以拉取底模与依赖)。
3. **挂载您的图集**：
   - 点击顶部菜单 **File** -> **Add Data** (或展开右侧面板中的 **+ Add Input**)；
   - 在 **Your Work** 中找到刚才上传的私有数据集，点击右侧的 **+** 号挂载进当前 Notebook；
   - 挂载成功后，您的数据集路径通常位于：`/kaggle/input/` 目录下。

---

### 第三步：填入参数并一键启动训练

在 Notebook 的第一个代码单元格（Cell）中，直接粘贴以下整合好的**傻瓜式一键执行脚本**。

您只需要在最上方的 **【用户自定义超参数控制区】** 填入您自己的名称、触发词与轮数，随后点击单元格左侧的 **▶ 运行按钮 (Run)** 即可：

```python
# ==============================================================================
# 🎯 用户自定义超参数控制区 (请根据您的图集在此处直接修改)
# ==============================================================================
LORA_NAME = "my_character"          # 您的 LoRA 英文名称 (如 makima, genshin_style)
TRIGGER_WORD = "my_character_char"  # 触发词 (例如: makima_char；留空 "" 则使用原标注)

REPEATS = 6                         # 单图重复次数 (人物推荐 4~6，画风推荐 8~12)
EPOCHS = 4                          # 训练轮数 (人物推荐 3~5，画风推荐 6~10)
STEPS = 400                         # 核心步数 (公式: 图片数 * REPEATS * EPOCHS // 4)

RANK = 32                           # LoRA 矩阵秩 (单概念推荐 32，多人物/多画风推荐 48~64)
ALPHA = 32.0                        # 缩放系数 (建议与 Rank 对齐 = 32.0 避免 5B 先验稀释)
LR = "1e-4"                         # 学习率 (Base/Heal 推荐 1e-4，SFT 画风推荐 1.2e-4~1.5e-4)
GRAD_ACC = 1                        # 梯度累积步数 (单画风=1 极速；角色/多子集混合=2)
TIMESTEP_MU = -1.09                 # 时间步采样重心 (-1.09=画风高噪；-0.60=多画风；0.0=动作；0.35=人物细节)

# --- 进阶适配器架构与多子集配比控制 ---
ALGO = "lora"                       # 算法: "lora" | "loha"(高容量哈达玛) | "lokr"(紧凑克罗内克)
USE_DORA = False                    # 是否启用 DoRA 权重分解 (人物/多画风/细节还原建议 True)
TARGET_PRESET = "attn-mlp"          # 注入层: "attn" | "attn-mlp" | "attn-mlp+modulation" | "attn-mlp+text-refiner" | "full-block" | "all"
LORAPLUS_RATIO = 1.0                # LoRA+ 上投影学习率倍率 (1.0=关闭，人物/多概念建议 2.0~4.0)
SUBSETS_JSON = ""                   # 留空 "" 为单图集；填写 JSON 数组可开启多子集配比

# --- 云端显存优化、分辨率与底模变体控制 ---
MODEL_VARIANT = "heal"              # 底模分支: "heal"(推荐防溢出) | "base"(官方原版) | "sft"(官方SFT对齐版)
VRAM_PRESET = "auto"                # 显存档位: "auto" | "6g"(4352) | "8g"(4352) | "12g"(6144) | "16g"(8192)
RESOLUTION = 1024                   # 目标分辨率: 1024(标准高清) | 896(黄金甜点) | 768(T4加速) | 512(快速测试)
BASE_QUANT = "auto"                 # 底模权重来源: "auto" | "int8"(下载体积减半且省显存) | "full"
BASE_PRECISION = "auto"             # 底模计算精度: "auto" | "int8" | "fp8" | "fp8_compute"(L4/A100 sm89+) | "compute"
PRECISION = "auto"                  # 计算精度: "auto"(T4自动FP16，L4/A100自动BF16) | "fp16" | "bf16"
TUNE = "off"                        # Triton 调优: "off" | "table"
OVERLONG = "drop_tokens"            # 超长图文策略: "drop_tokens"(保留全图仅截断超限文本) | "drop"
TEXT_DEVICE = "auto"                # 文本编码器设备: "auto" | "cuda" | "cpu"

# --- 续炼与存档间隔控制 ---
TRAIN_MODE = "fresh"                # "fresh" | "resume"(断点续炼) | "init_from"(基于已有LoRA微调)
INIT_LORA_PATH = ""                 # 指定已有 .safetensors 权重路径 (续炼/二阶段微调时使用)
SAVE_EVERY = 100                    # 每 N 步自动保存一次中间 checkpoint 权重
# ==============================================================================

# 1. 克隆训练工程
import os, subprocess, sys
if not os.path.exists("kaggle-moss-lora-pipeline"):
    !git clone https://github.com/abysschaos/kaggle-moss-lora-pipeline.git

%cd kaggle-moss-lora-pipeline

# 2. 调用云端训练引擎执行全自动训练
cmd = [
    sys.executable, "train_moss.py",
    "--lora_name", str(LORA_NAME),
    "--trigger_word", str(TRIGGER_WORD),
    "--repeats", str(REPEATS),
    "--epochs", str(EPOCHS),
    "--steps", str(STEPS),
    "--rank", str(RANK),
    "--alpha", str(ALPHA),
    "--algo", str(ALGO),
    "--target_preset", str(TARGET_PRESET),
    "--loraplus_ratio", str(LORAPLUS_RATIO),
    "--lr", str(LR),
    "--grad_acc", str(GRAD_ACC),
    "--timestep_mu", str(TIMESTEP_MU),
    "--variant", str(MODEL_VARIANT),
    "--vram_preset", str(VRAM_PRESET),
    "--resolution", str(RESOLUTION),
    "--base_quant", str(BASE_QUANT),
    "--base_precision", str(BASE_PRECISION),
    "--precision", str(PRECISION),
    "--tune", str(TUNE),
    "--overlong", str(OVERLONG),
    "--text_device", str(TEXT_DEVICE),
    "--train_mode", str(TRAIN_MODE),
    "--save_every", str(SAVE_EVERY),
]
if USE_DORA:
    cmd.append("--dora")
if SUBSETS_JSON.strip():
    cmd.extend(["--subsets_json", SUBSETS_JSON.strip()])
if INIT_LORA_PATH.strip():
    cmd.extend(["--init_from", INIT_LORA_PATH.strip(), "--resume_from", INIT_LORA_PATH.strip()])
subprocess.run(cmd, check=True)
```

#### 系统在后台执行的自动化过程：
1. 自动扫描挂载在 `/kaggle/input/` 的任意图集，自动清理旧图孤儿文件并建立 1:1 对齐的 Kohya 标准目录；
2. 若填写了 `TRIGGER_WORD`，自动将触发词前置拼接至每一张图片的标注头部；
3. 自动嗅探云端 GPU 型号（T4 / P100 / L4 / A100）、显存容量与原生 BF16 能力，按所选 `MODEL_VARIANT`（`heal` / `base` / `sft`）与量化模式从 HuggingFace 极速下载对应底模，并对未治愈的 Base/SFT 权重在 FP16 模式下自动开启 `FP16_FIX="online"` 防溢出守卫；
4. 自动打入 PyTorch 2.11 驱动兼容性自愈补丁，同时关闭训练中途采样与盲测防 OOM；
5. 按照设定的 `VRAM_PRESET` 与 `RESOLUTION` 执行离线双特征提取与 Flow Matching 反向传播训练（在 T4 上切换 `RESOLUTION = 768` 可显著加速，400~600 步约耗时 30~50 分钟）。

---

### 第四步：获取与下载训练产出的 LoRA 权重

训练执行完成后，脚本会自动将权重整理打包并输出提示：
- 成果文件存放在 `/kaggle/working/output/{LORA_NAME}/` 目录下；
- 包含符合官方标准的权重：
  - `{LORA_NAME}-comfy.safetensors`（ComfyUI 原生即插即用）
  - `{LORA_NAME}-lycoris.safetensors`（LyCORIS 标准格式）
  - `training_summary.json`（训练超参数与日期留档元数据）
- 同时在根目录下自动生成了 `{LORA_NAME}_lora_results.zip` 压缩包。

#### 如何下载到自己的电脑：
1. 查看 Kaggle Notebook 界面右侧的 **Output**（输出文件）列表；
2. 找到生成的 `{LORA_NAME}_lora_results.zip` 或 `.safetensors` 文件；
3. 鼠标悬停在文件名右侧，点击出现的 **三点图标 (⋮)** -> **Download** 即可秒级下载到本地电脑。
4. 将解压出来的 `.safetensors` 文件放入本地 ComfyUI 的 `models/loras/` 文件夹中即可开始出图！

---

## 方案二：本地 CLI 一键推送到 Kaggle 后台 (进阶玩家首选)

如果您熟悉本地终端，并且希望在本地电脑上像运行脚本一样，直接把代码和参数推送到 Kaggle 云端后台静默运行（训练完自动拉回，连网页都不用看）：

### 1. 配置 Kaggle 命令行密钥 (只需配一次)
1. 登录 Kaggle，点击右上角头像 -> **Settings**；
2. 往下拉找到 **API** 板块，点击 **Create New Token**，浏览器会自动下载一个 `kaggle.json` 凭据文件；
3. 将该文件放置在您电脑的用户目录下：
   `C:\Users\<您的Windows用户名>\.kaggle\kaggle.json`
4. 在终端中验证：执行 `kaggle -v`，若显示版本号即代表凭证生效。

### 2. 推送任务至云端
在项目根目录打开 PowerShell 终端，执行：
```powershell
.\scripts\push_to_kaggle.ps1
```
脚本会自动将本地训练配置与任务推送到 Kaggle 创建无头计算实例，云端 T4 GPU 实例将自动唤醒并开始后台训练。您可以随时关机，任务在云端服务器独立推进。

### 3. 查看状态与拉回产出权重
- 查看云端当前运行进度：
  ```powershell
  kaggle kernels status
  ```
- 训练状态显示 `COMPLETE` 后，执行以下命令一键把云端生成的权重拉回本地：
  ```powershell
  .\scripts\pull_weights.ps1
  ```
  导出的 `.safetensors` 文件会自动归档到本地的 `output/` 文件夹中。

---

## 💡 云端调参建议速查表 (单概念 vs 多子集配比)

| 场景目标 | Repeats / Epochs / Steps | Rank / Alpha | 推荐适配器配置 | 触发词与多子集建议 |
| :--- | :--- | :--- | :--- | :--- |
| **二次元 / 艺术画风** | `10` / `6` / `600` | `32` / `32.0` | `lora` + `attn-mlp` + `GradAcc=1` + `Mu=-1.09` | 填风格词，如 `in the style of my_style` |
| **特定动漫 / 游戏人物** | `6` / `4` / `360~400` | `32` / `32.0` | `lora` + `dora` + `loraplus=2.0` + `GradAcc=2` + `Mu=0.35` | 必须填独特专有词，如 `makima_char` |
| **多人物 / 多画风融合** | 按各子集图数反比配平 / `4~5` | `48` / `48.0` | `lora` + `dora` + `loraplus=2.0` + `GradAcc=2` + (`Mu=0.35`人物/`-0.60`画风) | 启用 `SUBSETS_JSON` 为每角色/每画风绑定独立子目录与触发词 |
| **特定动作 / 服饰概念** | `8` / `4` / `400~500` | `32` / `32.0` | `lora` + `dora` + `attn-mlp+text-refiner` + `GradAcc=2` + `Mu=0.0` | 填具体动作/物体名词，确保主体人物环境分散 |

> **动态步数对齐公式**：
> 当您拥有 $N$ 张图片，有效 Token 打包与梯度累积等效批大小为 $B_{\text{eff}} = \lfloor \frac{\text{MAX\_TOKENS}}{(\text{res}/16)^2+160} \rfloor \times \text{GRAD\_ACC}$ 时，GUI 会按时间步重心划分推荐除数（画风 $\mu \le -0.3$ 时基础除数 `4`，人物/概念 $\mu > -0.3$ 时基础除数 `2`）：
> $$\text{Steps} = \frac{N \times \text{Repeats} \times \text{Epochs}}{\max(\text{基础除数},\ B_{\text{eff}})}$$
> 在保持 `GradAcc=1, Mu=-1.09` 的标准画风训练下即自动沿用 $\div 4$ 极速节奏。
