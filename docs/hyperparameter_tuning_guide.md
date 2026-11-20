# MOSS-Image LoRA 超参数调优与实战指南 (画风 vs 人物)

## 1. 核心换算公式与参数数学模型

在针对 50 亿参数（5B）单流 Diffusion Transformer（DiT）模型——**OpenMOSS MOSS-Image** 进行 LoRA 训练时，必须理解训练引擎底层的样本曝光与步数传导机制。

### 1.1 真实 Token 打包批处理机制 (Token Packing) 与 `MAX_TOKENS` 档位
传统的卷积神经网络（UNet）通常采用固定的图像张数批次（如 `batch_size = 1`）。而本项目依托的 KohakuMossTrainer 采用基于连续流序列长度的自适应 Token 打包机制，并按显存档位校准 `MAX_TOKENS`：
```python
BATCHING = "tokens"
OVERLONG = "drop_tokens"  # 仅截断超限文本尾部，绝不静默丢图
# 6g/8g -> MAX_TOKENS = 4352 | 12g -> MAX_TOKENS = 6144 | 16g -> MAX_TOKENS = 8192
```
在 DiT 架构中，图像潜空间视觉 Token 数约等于 `(resolution / 16)²`（例如 `1024` 为 `4096`，`896` 为 `3136`，`768` 为 `2304`），加上自然语言文本（约 `120~200` Token）即构成单样本序列长度：
- **`1024` 分辨率**：单图约 `4256` Token，因此在 `4352 / 6144 / 8192` 下通常每包 **1 张/步**（高分辨率细节拉满）。
- **`896` 黄金甜点分辨率**：单图仅约 `3296` Token，在 `16g (8192)` 档位下可稳定打包 **2 张/步**，兼顾精细度与双倍吞吐！
- **`768` 分辨率**：单图约 `2464` Token，在 `12g (6144)` 下打包 **2 张/步**，在 `16g (8192)` 下打包 **3 张/步**。

### 1.2 真实有效批大小联动公式（严格消除虚假 `/ 4` 除数）
$$\text{总图文曝光次数 (Total Exposures)} = \text{原始图片数 (Images)} \times \text{单图重复 (Repeats)} \times \text{训练轮数 (Epochs)}$$
$$\text{等效 Batch (Effective Batch)} = \lfloor \frac{\text{MAX\_TOKENS}}{(\text{res}/16)^2 + 160} \rfloor \times \text{GRAD\_ACC}$$
$$\text{GUI 推荐核心步数 (Steps)} = \frac{\text{总图文曝光次数}}{\max(1,\ \text{Effective Batch})}$$

#### 实例推演（以默认画风工作流为例）
- **场景**：现有 40 张画风图像，`1024px` 真实打包 1 图/包，`GradAcc=1`（真实等效 `Effective Batch = 1`），`TIMESTEP_MU=-1.09`；
- **设置**：`Repeats = 3`，`Epochs = 5`（共 15 个真实遍历轮次），总曝光 $= 40 \times 3 \times 5 = 600$ 次；
- **优化器步数**：严格按真实吞吐计算为 **$600 \div 1 = 600$ 步**，既彻底去掉了不真实的固定 `/ 4` 虚高重复，又完整保留了约 1.3 小时 600 步的黄金节奏；当你将分辨率切至 `896px`（`16g` 档位下打包 2 图/步）或设置 `GradAcc=2`、`GradAcc=4` 时，推荐步数会自动按真实等效 Batch 倍率实时缩放。

在调整参数时，必须保持 Repeats、Epochs 与 Steps 三者在数值上的逻辑闭环。若手动指定了 `--steps`，优化器会在达到该步数时强制截断训练。

---

## 2. 画风 (Style) 与 人物 (Character) 调优对照矩阵

画风与人物属于两种截然相反的优化任务：画风要求模型跨越所有具象实体，抽象出全局笔触与色调；人物则要求模型在变换所有外部服装与场景的前提下，死锁特定的面容与身体特征。

| 调优维度 | 画风 LoRA (Style) | 人物 / 角色 LoRA (Character) |
| :--- | :--- | :--- |
| **优化目标** | 捕捉全图高频笔触、线条肌理、全局光影分布、固有色调与上色风格 | 精准固定人物面容比例、标志性发型发色、瞳色、固有体态与面部微特征 |
| **泛化要求** | 风格必须与画面主体（人物、动物、机械、风景、构图）彻底解耦 | 人物面容必须与服装、饰品、动作姿态、情绪表情、背景环境彻底解耦 |
| **推荐图片量** | **30 ~ 80+ 张**（画面主体高度分散，忌全是同一个人） | **20 ~ 50 张**（严格同一角色，但角度、景别、光线需高度分散） |
| **单图重复 (Repeats)**| **8 ~ 15 次**（通常推荐 10 次） | **2 ~ 6 次**（避免单张特定姿势被过度拟合死锁） |
| **训练轮数 (Epochs)** | **5 ~ 8 轮**（总等效轮次约 50 ~ 80 次，需深层击穿底模风格惯性） | **3 ~ 5 轮**（总等效轮次约 15 ~ 25 次，对过拟合极度敏感） |
| **总优化步数 (Steps)** | **500 ~ 800 步**（对应总曝光 2000 ~ 3200 次） | **200 ~ 400 步**（对应总曝光 800 ~ 1600 次） |
| **LoRA 矩阵秩 (Rank)**| **32 ~ 64**（画风涉及全频段残差流扰动，低秩无法承载复杂笔触）| **16 ~ 32**（Rank 过高极易导致姿态死锁、背景串色与衣服换不掉） |
| **缩放强度 (Alpha)** | **等于 Rank**（如 Rank=32, Alpha=32.0，满额 1.0 注入） | **半额至满额**（如 Rank=32 时，Alpha 取 16.0 ~ 32.0） |
| **学习率 (Learning Rate)**| **1e-4**（配合 Cosine 余弦退火调度器） | **8e-5 ~ 1e-4**（适度微调降速，确保面部五官平滑收敛） |
| **打标标注规范** | **穷尽式自然语言散文描述 (Full Captioning)** | **生物特征剪枝与触发词隔离 (Token Pruning)** |

---

## 3. 打标标注工程规范与避坑法则

在基于 Qwen3.5-9B 作为文本编码器的 MOSS-Image 体系中，标注文本直接决定了损失函数将图像特征归因到哪个语义槽。

### 3.1 画风训练标注：穷尽式详细描述
- **核心原则**：把画面中所有可见的具体实体与动作全部写出来。
- **底层机制**：当文本标注完整覆盖了“短发女孩”、“白色衬衫”、“城市街道”、“雨天倒影”等概念后，DiT 模型的交叉注意力机制会将这些几何形状归因于对应的名词。画面中剩下的、无法被名词解释的“粗糙油画笔触”、“赛博朋克霓虹色散”、“低饱和水彩晕染”等全局画风，就会被干净地剥离并沉淀进 LoRA 矩阵中。
- **标注示例**：
  > `A digital anime concept art of a young woman with silver hair tied in a ponytail, wearing a dark green military jacket, leaning against a rusty metal wall in an abandoned industrial warehouse, dramatic rim lighting, dust particles in the air, cinematic composition.`

### 3.2 人物训练标注：生物固有特征剪枝 (Token Pruning)
- **核心原则**：**先天固有特征坚决不写，后天动态变量必须巨细靡遗写全**。
- **生物特征剪枝**：角色的天生特征（如红发、双马尾、黄色瞳孔、泪痣）绝对不要在 `.txt` 中描述。如果在每张图的标注中都写了 `red hair, twintails`，模型就会认为“红发双马尾”是一个公共通用词汇，而不会将其绑定给角色触发词。在文本中彻底删去这些词汇，只留下唯一的触发词（如 `lucy_char`），模型为了解释画面中稳定存在的红发与双马尾，就会被迫将其全部固化到 `lucy_char` 的权重中。
- **动态变量显式标注**：角色当前穿着的服装、所处的背景、当下的姿态与表情，必须详细写入。只有在标注中明确写了 `wearing a maid outfit`，模型才知道女仆装属于外部变量，从而在后续生成时允许用户随意更换服装。
- **对比范例**：
  - **错误写法（直接导致服装死锁与特征稀释）**：
    `lucy_char, a girl with red hair, twin braids, yellow eyes, cute.`（丢失了服装与背景描述，且重复标注了固有特征）
  - **正确写法（干净解耦，支持自由换装与场景迁移）**：
    `lucy_char, looking at viewer with a gentle smile, sitting on a wooden bench in a sunny park, wearing a white oversized hoodie and denim shorts, full body shot, blurred natural greenery background.`

---

## 4. 常见异常表征与诊断处方

### 4.1 症状 A：生成的图片烧焦、色斑死黑或严重过饱和
- **故障诊断**：学习率（LR）过高或总训练步数严重超标，导致权重矩阵参数爆炸；或者推理时 CFG 设置错误。
- **排查处方**：
  1. 检查 ComfyUI 推理配置：MOSS-Image 为连续流模型，CFG 必须设为 **3.0 ~ 3.5**（严禁使用 SD1.5 的 7.0+）。
  2. 若 CFG 正确依然烧焦，将学习率从 `1e-4` 下调至 `7e-5 ~ 8e-5`。
  3. 将总步数截断缩减 30%（例如从 600 步降低至 400 步）。

### 4.2 症状 B：人物换装失败，总是强制穿训练集里的同一套衣服 (Clothes Bleeding)
- **故障诊断**：严重过拟合。原因通常是打标时遗漏了服装描述，或者 Rank 设得过高（如 Rank 64），亦或是 Epochs 轮数跑得过多。
- **排查处方**：
  1. 检查训练集标注：确保每张图片的衣服都被详细写出（如 `black leather jacket`、`swimsuit`）。
  2. 将单图重复次数从 10 降为 3 ~ 4，总步数控制在 250 ~ 350 步之间。
  3. 将 LoRA Rank 降低为 16 或 32，Alpha 设定为 Rank 的一半（如 Rank=32, Alpha=16.0）。

### 4.3 症状 C：LoRA 强度开到 1.0 依然几乎看不出画风效果 (欠拟合)
- **故障诊断**：网络容量不足或总曝光量过低，未能打破 5B 底模的原生风格惯性。
- **排查处方**：
  1. 提高 Rank 与 Alpha：从 16 提升至 32 或 64，且 Alpha 与 Rank 保持 1:1（如 Rank=64, Alpha=64.0）。
  2. 增加训练轮次：将 Epochs 增加至 6 ~ 8 轮，总优化步数拉升至 600 ~ 800 步。
  3. 确认底模版本：务必使用修复版 `KBlueLeaf/MOSS-Image-heal`，避免原生底模数值不稳定。

---

## 5. 多子集配比与高阶适配器（DoRA / LoHa / LoKr / LoRA+）实战策略

除基础单图集 LoRA 外，本管线原生接入了 `KohakuMossTrainer` 全套高阶低秩结构与配比能力：

| 高阶特性 | CLI / GUI 开关 | 核心适用场景与调优建议 |
| :--- | :--- | :--- |
| **多子集配比 (`--subsets_json`)** | `SUBSETS_JSON` / GUI 多子集表格 | 同时炼制多位角色、多种艺术画风或互补概念时使用。强烈建议使用 GUI【⚖️ 一键配平 Repeat】按各子集图像数量反比设定 `repeats`，保证单轮中弱势少图子集的曝光占比与强势多图子集接近 1:1，并各自绑定独立触发词避免特征串味。 |
| **DoRA 权重分解 (`--dora`)** | 勾选【启用 DoRA】 | 将权重的方向向量与幅值标量分离优化。极度适合**复杂画风细节与多人物面部五官高保真还原**。训练引擎已默认注入 `DORA_DETACH=True` 控制显存与数值稳定，在 6G~8G `int8` 底模上亦可借助引擎旁路安全运行。 |
| **适配器算法 (`--algo`)** | `lora` / `loha` / `lokr` | `lora` 通用最稳；`loha`（Hadamard 乘积）具备更强的非线性表征容量，适合**跨风格大融合**；`lokr`（Kronecker 乘积）体积最小、压缩比极高，适合轻量分发。 |
| **目标注入层 (`--target_preset`)** | 6 档注入预设 | `attn-mlp` 通用均衡；**多人物/复杂动作推荐 `attn-mlp+text-refiner`**（微调文本细化器加强语义绑定）；**画风迁移推荐 `attn-mlp+modulation`**（控制自适应层归一化时间调制特征，全局色调笔触收敛更强）；**重度风格重写可选 `full-block`**。 |
| **LoRA+ (`--loraplus_ratio`)** | `1.0` / `2.0 ~ 4.0` | 按 ICML 论文设定放大 B（上投影）矩阵学习率倍率。单角色/多人物/多画风设为 `2.0 ~ 4.0` 可明显加快早期特征成形；简单单画风保持 `1.0` 避免过冲。 |
| **梯度累积 (`--grad_acc`)** | `1` / `2 ~ 4` | 累计 `N` 个微批次后执行一次参数更新。**零额外显存开销**（仅线性按倍率延长单步优化周期）。画风默认 `1` 极速收敛；人物五官及多子集混合设为 `2` 可平滑单图随机姿态/噪声梯度方差。 |
| **时间步采样重心 (`--timestep_mu`)** | `-1.09` / `-0.60` / `0.0` / `0.35` | 调控 `MossTimestep` 的 Logit-Normal 偏移均值（底层高分辨率 shift=5）。`-1.09` 为默认极致高噪焦点（约 89% 采样在 $\sigma>0.5$，最适合全局色彩构图画风）；`-0.60` 兼顾多画风笔触与线条；`0.0` 对称覆盖动态概念；`0.35` 显著拉高 $\sigma<0.5$ 低噪区间比重，精细雕刻瞳孔、发丝与服饰纹理。不增加任何单步显存与时长。 |
| **底模计算精度 (`--base_precision`)** | `auto` / `int8` / `fp8` / `fp8_compute` / `compute` | `int8` 底模加载 + `bf16` 训练占用极低显存（~5.4 GB 权重）且可跑满 RTX 30/40 全系；而 `bf16` 底模 + `fp8_compute` 需 Ada（RTX 40 系）/ L4 / A100 显卡（`sm89+`）并强制搭配 `FP8_DGRAD="16bit"`（否则反向保存两份转置副本将吃满 10.24 GB 毫无省显存效果），且**禁止对 `int8_convrot` 底模强开 `fp8`**。 |

### 5.1 五大内置预设模板核心参数矩阵（v1.3.0）

| 预设模板 | Repeats / Epochs | Rank / Alpha | 算法与注入层 | LoRA+ | GradAcc | Timestep Mu | 学习率 (LR) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **🎨 艺术画风 (`style`)** | `3` / `5` | `32` / `32.0` | `lora` + `attn-mlp` | `1.0` | **`1`** | **`-1.09`** | `1e-4` |
| **👤 单人角色 (`character`)** | `6` / `4` | `32` / `32.0` | `lora` + `DoRA` + `attn-mlp` | `2.0` | **`2`** | **`0.35`** | `8e-5` |
| **👥 多人物配比 (`multi_char`)** | 按图数反比配平 / `4` | `48` / `48.0` | `lora` + `DoRA` + `attn-mlp+text-refiner` | `2.0` | **`2`** | **`0.35`** | `8e-5` |
| **🎭 多画风配比 (`multi_style`)** | 按图数反比配平 / `5` | `48` / `48.0` | `lora` + `DoRA` + `attn-mlp+modulation` | `2.0` | **`2`** | **`-0.60`** | `1e-4` |
| **💡 概念与动作 (`concept`)** | `5` / `4` | `32` / `32.0` | `lora` + `DoRA` + `attn-mlp+text-refiner` | `2.0` | **`2`** | **`0.0`** | `1e-4` |

---

## 6. 实战命令行调用模板

在本地 Windows 环境下，直接通过 [train_local.py](../train_local.py) 传入针对性超参数（推荐优先使用规范参数 `--lora_name` 与 `--trigger_word`）。

### 6.1 人物角色训练模板 (以 30 张图为例 + DoRA + 精细细节 Mu=0.35)
```powershell
python train_local.py `
    --data_dir "./dataset/my_character" `
    --output_dir "./output" `
    --lora_name "my_character" `
    --trigger_word "my_character_char" `
    --repeats 6 `
    --epochs 4 `
    --steps 360 `
    --rank 32 `
    --alpha 32.0 `
    --algo lora `
    --dora `
    --target_preset attn-mlp `
    --loraplus_ratio 2.0 `
    --grad_acc 2 `
    --timestep_mu 0.35 `
    --resolution 896 `
    --lr 8e-5 `
    --precision bf16 `
    --text_device cpu `
    --copy_to_comfy_loras
```

### 6.2 艺术画风训练模板 (以 40 张图为例 + 极速 Mu=-1.09)
```powershell
python train_local.py `
    --data_dir "./dataset/my_art_style" `
    --output_dir "./output" `
    --lora_name "my_style" `
    --trigger_word "in the style of my_style" `
    --repeats 3 `
    --epochs 5 `
    --steps 600 `
    --rank 32 `
    --alpha 32.0 `
    --algo lora `
    --target_preset attn-mlp+modulation `
    --grad_acc 1 `
    --timestep_mu -1.09 `
    --lr 1e-4 `
    --precision bf16 `
    --text_device cpu `
    --copy_to_comfy_loras
```
