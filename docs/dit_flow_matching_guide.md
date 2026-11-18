# OpenMOSS MOSS-Image (5B DiT) 与 Flow Matching 训练工程理论

## 1. 架构概览与核心特性
OpenMOSS MOSS-Image 采用了基于 50 亿参数（5B）的单流扩散变换器（Single-Stream Diffusion Transformer，简称 DiT）架构，文本编码器为深度定制的 Qwen3.5-9B，潜空间编解码器为 FLUX.2 VAE。与早期基于 UNet 和 DDPM 离散高斯扩散模型相比，MOSS-Image 具有以下关键技术差异：
1. **连续流匹配（Flow Matching / Rectified Flow）**：生成过程通过拟合两点之间的速度向量场（$\mathbf{v} = x_1 - x_0$）实现，概率演化轨迹更平直，步进效率更高。
2. **原生支持中英双语精准排版与绝对空间布局**：模型支持原生的 JSON 边界框（Bounding Box）坐标输入，能精准控制文字排布与局部物体空间方位。
3. **强大的底模先验惯性**：5B 的庞大参数量赋予了底模极强的通用材质与空间认知，微调时需要足够的轮次与注入强度才能克服其原生风格惯性。

---

## 2. 步数（Steps）与轮次（Epoch）的数学映射
在传统训练器中，通常默认以单张图像微步推进（`batch_size = 1`），步数直接等于样本遍历次数。而在 KohakuMossTrainer 框架中，启用了面向大语言模型的 Token 打包机制（`BATCHING = "tokens"`, `MAX_TOKENS = 16384`）：

$$\text{单步吞吐量} \approx 4 \text{ 张 } 1024 \times 1024 \text{ 高分辨率图文特征}$$

换算公式如下：
$$\text{总训练步数 (Steps)} = \frac{\text{图集总数} \times \text{图片循环次数 (Repeats)} \times \text{最大轮次 (Epoch)}}{\text{单步批次大小 (Batch Size)}}$$

以 40 张画风图集为例：
* **100 步（浅层测试）**：$100 \times 4 = 400$ 次遍历 $\approx 10$ 个 Epoch（容易陷入严重欠拟合，画风难以凸显）。
* **320 步（标准收敛）**：$320 \times 4 = 1280$ 次遍历 $\approx 32$ 个 Epoch（适合中轻度画风迁移）。
* **600 步（黄金深度收敛）**：$600 \times 4 = 2400$ 次遍历 $\approx 60$ 个 Epoch（在 Tesla T4 上耗时约 7 小时，彻底击穿底模先验，画风浓郁纯正）。

---

## 3. LoRA 超参数配比建议
* **网络维度（Rank / Dim）**：推荐设为 **32** 或 **64**。5B 参数的 DiT 模型注意力层宽阔，过小的 Rank（如 8 或 16）无法承载复杂的多尺度笔触。
* **缩放因子（Alpha）**：推荐设为与 Rank 相同（如 **32.0**），使有效学习步长缩放比达到 $\alpha / r = 1.0$。若 Alpha 设为 16.0（缩放比 0.5），在大模型面前容易被先验稀释。
* **学习率（Learning Rate）**：推荐 **1e-4**，配合余弦退火调度器（Cosine Annealing）以及 5% 总步数的预热阶段（Warmup）。
* **底模选择**：强烈推荐使用经过平滑重训的官方修复底模 [KBlueLeaf/MOSS-Image-heal](https://huggingface.co/KBlueLeaf/MOSS-Image-heal)，彻底消除深层残差流中高达 58,411 的数值尖峰，杜绝半精度溢出与 NaN 崩溃。
