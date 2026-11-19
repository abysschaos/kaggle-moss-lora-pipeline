# MOSS-Image LoRA 在 ComfyUI 中的最佳推理规范与避坑指南

## 1. 为什么 MOSS-Image 推理参数极其敏感？
MOSS-Image-Base 属于未经分类器自由引导（CFG）蒸馏的原生 Base 模型，其内部完整保留了流匹配动力学分布。如果套用传统 Stable Diffusion 1.5 或 SDXL 的推理习惯（例如 CFG 7.0、步数 20 步、加一堆负向词），会引发严重的画面过饱和发黑、噪点爆炸与几何失真。

根据 [OpenMOSS-Team/MOSS-Image](https://github.com/OpenMOSS-Team/MOSS-Image) 官方 `USERGUIDE.md` 的规范，必须严格遵循以下参数矩阵：

| 推理旋钮 | 官方推荐设定 | 常见错误设定及后果 |
| :--- | :--- | :--- |
| **CFG 引导值** | **3.0 ~ 3.5** | 设为 7.0 ~ 8.0（导致画面瞬间严重过饱和烧焦死黑） |
| **采样器 (Sampler)** | **Flow-matching Euler** | 使用 DPM++ 2M 等（导致去噪轨迹错乱） |
| **调度器 (Scheduler)** | 带有 **Shift = 5.0** 偏移量 | 使用普通 Normal / Karras（导致画面全局严重涂抹失焦） |
| **采样步数 (Steps)** | **>= 50 步** | 设为 20 ~ 30 步（步数过低无法完成速度场积分，画面模糊） |
| **负向提示词** | **留空 (Empty)** | 填入常规负面套话（破坏流匹配的概率轨迹） |
| **分枝跳层 (SLG)** | 跳过第 **22** 层 (`uncond_skip_layers=[22]`) | 未跳层导致负分支过度干涉主干注意力 |

---

## 2. LoRA 节点挂载与选择
将导出的 LoRA 文件放置于 `ComfyUI/models/loras/` 目录下（本管线在导出 DoRA / LoHa / LoKr 与常规 LoRA 时均已显式锁定 `SAVE_FUSED_QKV = False`，避免 QKV 合并膨胀为全量差分张量）。

* **使用标准 `Load LoRA` 节点**：
  直接加载 `<lora_name>-comfy.safetensors`。ComfyUI 底层的 `comfy/lora.py` 会自动通过内置的 `lora_unet_` 桥接字典将权重张量满额映射并加载到底模中。
* **使用 `LycorisLoader` 节点**：
  对于 **LoHa / LoKr / DoRA** 或需要完整保留 LyCORIS 架构信息的权重，推荐加载 `<lora_name>-lycoris.safetensors`。
* **LoRA 强度调节**：
  在 LoRA 加载节点中，推荐将模型强度（Strength Model）设在 **0.6 ~ 0.8** 区间内（多人物或多画风混合融合 LoRA 可从 **0.7** 起步测试），既能完整呈现目标风格或角色特征，又能保留极佳的自然语言提示词泛化自由度。
