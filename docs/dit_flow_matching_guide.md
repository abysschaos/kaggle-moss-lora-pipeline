# OpenMOSS MOSS-Image (5B DiT) 与 Flow Matching 训练工程理论

## 1. 架构概览与核心特性
OpenMOSS MOSS-Image 采用了基于 50 亿参数（5B）的单流扩散变换器（Single-Stream Diffusion Transformer，简称 DiT）架构，文本编码器为深度定制的 Qwen3.5-9B，潜空间编解码器为 FLUX.2 VAE。与早期基于 UNet 和 DDPM 离散高斯扩散模型相比，MOSS-Image 具有以下关键技术差异：
1. **连续流匹配（Flow Matching / Rectified Flow）**：生成过程通过拟合两点之间的速度向量场（$\mathbf{v} = x_1 - x_0$）实现，概率演化轨迹更平直，步进效率更高。
2. **原生支持中英双语精准排版与绝对空间布局**：模型支持原生的 JSON 边界框（Bounding Box）坐标输入，能精准控制文字排布与局部物体空间方位。
3. **强大的底模先验惯性**：5B 的庞大参数量赋予了底模极强的通用材质与空间认知，微调时需要足够的轮次与注入强度才能克服其原生风格惯性。

---

## 2. 步数（Steps）、时间步采样与 Token Packing 数学映射
在传统训练器中，通常默认以单张图像微步推进（`batch_size = 1`），步数直接等于样本遍历次数。而在 KohakuMossTrainer 框架中，启用了面向连续序列长度的 Token 打包机制（`BATCHING = "tokens"`、`OVERLONG = "drop_tokens"`，并按 `6g/8g -> 4352`、`12g -> 6144`、`16g -> 8192` 校准显存预算）：

- 在 `1024×1024`（`64×64 = 4096` 视觉 Token）下单包可稳装 **1 张** 图像（加文本约 `4256` Token），保留全部高清纹理；
- 在 `896×896`（`56×56 = 3136` 视觉 Token）与 `16g (8192)` 档位下可稳装 **2 张/包**，实现吞吐翻倍；
- 再结合 `--grad_acc` 梯度累积，实际有效批大小为 $B_{\text{eff}} = \text{包内图数} \times \text{GRAD\_ACC}$。

此外，MOSS-Image 的 Flow Matching 训练采用 `MossTimestep` 对称/偏置采样分布（由 `--timestep_mu` 调控，底层带 `shift=5` 分辨率偏移）：
- **$\mu = -1.09$（默认画风配方）**：约 **89%** 梯度更新落在高噪构图与色域笔触阶段（$\sigma > 0.5$），这也是画风任务仅用 600 步即显著超越传统局部低噪配方的理论根因；
- **$\mu = 0.0$ / $0.35$（动作概念与单人角色配方）**：显著拉升中低噪精细雕刻权重，确保面部五官、服饰纽扣与轮廓边缘收敛充分。

GUI 推荐步数公式严格按照真实有效批大小计算，不做虚假除数下限截断：
$$\text{推荐训练步数 (Steps)} = \frac{\text{图集总数} \times \text{Repeats} \times \text{Epochs}}{\max(1,\ B_{\text{eff}})}$$

以 40 张画风图集为例（`1024px`, `GradAcc=1`, `Mu=-1.09`, 真实 `B_eff=1`）：`Repeats = 3`、`Epochs = 5` 共 15 个真实数据遍历轮次，即 $40 \times 3 \times 5 \div 1 = \mathbf{600 \text{ 步}}$，完整保留约 1.3 小时极速深度收敛节奏；若切为 `896px` + `16g`（包内 2 图、`B_eff=2`）或开启 `GradAcc=2`，推荐步数会自动同步按真实等效 Batch 缩放。若以 `int8_convrot + bf16` 对比 `bf16 + fp8_compute`（`sm89+` 专属且必须配对 `FP8_DGRAD="16bit"`），前者显存占用约 5.4 GB 且兼容全系 RTX 显卡，后者需 RTX 40 系+ Tensor Core 才可获得计算增益。

---

## 3. LoRA 超参数配比建议
* **网络维度（Rank / Dim）**：推荐设为 **32** 或 **64**。5B 参数的 DiT 模型注意力层宽阔，过小的 Rank（如 8 或 16）无法承载复杂的多尺度笔触。
* **缩放因子（Alpha）**：推荐设为与 Rank 相同（如 **32.0**），使有效学习步长缩放比达到 $\alpha / r = 1.0$。若 Alpha 设为 16.0（缩放比 0.5），在大模型面前容易被先验稀释。
* **学习率（Learning Rate）**：推荐 **1e-4**，配合余弦退火调度器（Cosine Annealing）以及 5% 总步数的预热阶段（Warmup）。
* **底模选择**：强烈推荐使用经过平滑重训的官方修复底模 [KBlueLeaf/MOSS-Image-heal](https://huggingface.co/KBlueLeaf/MOSS-Image-heal)，彻底消除深层残差流中高达 58,411 的数值尖峰，杜绝半精度溢出与 NaN 崩溃。
