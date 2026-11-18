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

RANK = 32                           # LoRA 矩阵秩 (推荐 32)
ALPHA = 16.0                        # 缩放系数 (人物推荐 16.0，画风推荐 32.0)
LR = "1e-4"                         # 学习率 (默认 1e-4 最佳)
# ==============================================================================

# 1. 克隆训练工程
import os, subprocess, sys
if not os.path.exists("kaggle-moss-lora-pipeline"):
    !git clone https://github.com/abysschaos/kaggle-moss-lora-pipeline.git

%cd kaggle-moss-lora-pipeline

# 2. 调用云端训练引擎执行全自动训练
!python train_moss.py \
    --lora_name "{LORA_NAME}" \
    --trigger_word "{TRIGGER_WORD}" \
    --repeats {REPEATS} \
    --epochs {EPOCHS} \
    --steps {STEPS} \
    --rank {RANK} \
    --alpha {ALPHA} \
    --lr "{LR}"
```

#### 系统在后台执行的自动化过程：
1. 自动扫描挂载在 `/kaggle/input/` 的任意图集，自动建立 Kohya 标准目录；
2. 若填写了 `TRIGGER_WORD`，自动将触发词前置拼接至每一张图片的标注头部；
3. 依托 Kaggle 云端千兆带宽，自动从 HuggingFace 极速下载 MOSS-Image-heal (FP16 全精度) 底模、Qwen3.5-9B 与 FLUX.2 VAE；
4. 自动打入 17 项 PyTorch 2.11 驱动兼容性自愈补丁；
5. 在 16GB T4 显卡上执行离线特征提取与 Flow Matching 反向传播训练（400~600 步约耗时 40~60 分钟）。

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

## 💡 云端调参建议速查表 (画风 vs 人物)

| 场景目标 | 推荐 Repeats | 推荐 Epochs | 推荐核心步数 (Steps) | 推荐 Rank | 推荐 Alpha | 推荐学习率 (LR) | 触发词建议 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **二次元 / 艺术画风** | 10 | 6 | 600 | 32 | 32.0 | 1e-4 | 建议填风格词，如 `in the style of my_style` |
| **特定动漫 / 游戏人物** | 6 | 4 | 400 | 32 | 16.0 | 8e-5 ~ 1e-4 | 必须填独特专有词，如 `makima_char` |
| **现实摄影 / 景物微调** | 8 | 5 | 500 | 32 | 24.0 | 1e-4 | 填主题名词，如 `cyberpunk_cityscape` |

> **步数对齐小公式**：
> 当您拥有 $N$ 张图片时，建议核心步数设为：
> $$\text{Steps} = \frac{N \times \text{Repeats} \times \text{Epochs}}{4}$$
> 确保模型既不欠拟合（学得太浅），也不过拟合（画面崩坏溢出）。
