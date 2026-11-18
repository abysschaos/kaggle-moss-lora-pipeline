# OpenMOSS MOSS-Image Kaggle 训练全流程排障实录与错误矩阵 (Exhaustive Troubleshooting Matrix)

在将 [KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer) 移植部署至 Kaggle 免费 GPU 实例（Tesla T4 16GB）的过程中，针对硬件限制、依赖冲突、驱动脱节、数据枚举、数值溢出及界面加载机制等问题，先后经历了 25 个迭代版本的调试与攻坚。本矩阵完整记录所有历史报错、根本诱因与最终工程修复方案。

---

## 错误分类与快速索引

| 序号 | 错误现象 / 关键字 | 根因分类 | 涉及版本 | 最终修复方案 |
| :--- | :--- | :--- | :--- | :--- |
| **01** | PowerShell `Set-Content` 导致 UTF-8 BOM 乱码 | CLI 文本编码 | Version 1 | 改用 Python 原生 `open(..., 'w', encoding='utf-8')` 生成纯净文本 |
| **02** | `ModuleNotFoundError: No module named 'six'` | CLI 依赖缺失 | Version 2 | 宿主机与容器预先安装 `pip install six` |
| **03** | `ValueError: kohya source: no images in the subfolders` | 数据目录规范 | Version 3 | 解压至 Kohya 严格规范的 `<repeats>_<name>` 目录结构 |
| **04** | `unknown knob LORA_ALPHA / LORA_RANK / SAVE_STEPS` | 配置白名单 | Version 4 | 严格规范为框架白名单键名 `DIM=32`、`ALPHA=32.0`、`SAVE_MINUTES=30` |
| **05** | HuggingFace 429 Client Error (Too Many Requests) | 鉴权与限流 | Version 5 | 注入专用 `HF_TOKEN` 并启用 `hf-transfer` 极速流式下载 |
| **06** | Kaggle Dataset 机器视觉合规误杀返回 403 Forbidden | 云端挂载风控 | Version 6 | 解绑公共数据集，改为基于高可用直链多源动态拉取 |
| **07** | `RuntimeError: operator torchvision::nms does not exist` | C++ 算子动态库脱节 | Version 7 | 正则放宽 `pyproject.toml` 中的 PyTorch 限制，使用 `--no-deps` 保护原生驱动 |
| **08** | `ValueError: txt_as must be 'tags' or 'nl', got caption` | 数据层枚举限制 | Version 8 | 在数据配置文件中明确定义 `"txt_as": "nl"` |
| **09** | `KeyError: unknown caption template 'raw'` | 模板注册表限制 | Version 9 | 将 `CAPTION` 显式设置为 `{"template": "nl"}` |
| **10** | `invalid configuration: unknown knob IMAGE_ENCODER_KWARGS` | 参数残留拦截 | Version 10 | 离线双缓存完成后对磁盘配置文件执行物理级正则清洗 |
| **11** | `AttributeError: module 'torch.accelerator' has no attribute 'empty_host_cache'` | PyTorch 接口兼容性 | Version 11~17 | 注入 `sitecustomize.py` 运行时 Hook 并在源码全局递归修补为安全调用 |
| **12** | `invalid configuration: SCHEDULE warmup=100 >= its end 80` | 学习率调度器断言 | Version 19 | 显式重构完整的 `SCHEDULE` 调度字典，设置合理的 warmup 步数 |
| **13** | `AttributeError: 'int' object has no attribute 'get'` | 调度器类型误判 | Version 20 | 补齐完整调度器字典并在 `objective.py` 打入 `isinstance(sub, dict)` 防御补丁 |
| **14** | `RuntimeError: Downloaded dataset archive is invalid (80 bytes)` | 第三方直链超限 | Version 21~22 | 部署“主镜像直链 + 备用直链”双重自动容灾拉取管线 |
| **15** | `400 Client Error: Bad Request for url: .../SaveKernel` | Kaggle 脚本体积超限 | Version 23 | 放弃 Base64 嵌入，保持 Python 脚本体积在 16KB 轻量级别 |
| **16** | 画面完全无变化、基本没有收敛 (欠拟合与坐标系失真) | 底模量化与轮次不足 | Version 24 | 底模升级为无损修复版 `MOSS-Image-heal`，步数拉升至 600 步 (60 Epoch)，Alpha 提升至 32 |
| **17** | ComfyUI 中 native 版本无效，而 comfy/lycoris 有效 | 键名自动桥接失配 | Version 25 | 深入 ComfyUI 源码查明 `lora_unet_` 自动桥接机制，彻底移除错误的前缀替换逻辑 |

---

## 核心致命报错深度剖析

### 1. TorchVision C++ 算子丢失问题 (`torchvision::nms does not exist`)
* **报错复现**：当执行 `pip install -e .` 安装 KohakuMossTrainer 时，其 `pyproject.toml` 硬编码了 `torch<2.15,>=2.13`，导致 pip 自动从 PyPI 下载了预编译的新版 PyTorch 轮子替换了宿主机原生的 PyTorch 2.11。然而宿主机环境中的 TorchVision 0.26 是依赖于原生 PyTorch 编译的动态链接库，PyTorch 核心版本变动导致符号表不一致，报出 C++ 算子缺失异常。
* **修复举措**：在安装前使用 Python 正则表达式将 `pyproject.toml` 中的限制动态替换为 `torch>=2.0`，并在安装时附带 `--no-build-isolation --no-deps` 参数，坚决不触碰宿主机原生的 CUDA 驱动与 PyTorch 环境。

### 2. PyTorch 2.11 加速器接口缺失 (`empty_host_cache`)
* **报错复现**：KohakuMossTrainer 针对最新的 PyTorch 2.3+ 开发，在清理显存时频繁硬编码调用 `torch.accelerator.empty_host_cache()`。而在 Kaggle 预装的 PyTorch 2.11 中，加速器抽象层尚不包含该方法，程序直接中断。
* **修复举措**：采取“双保险防御”：首先在 Python 的 `site-packages/sitecustomize.py` 中注入全局 Shim，若检测到缺失则动态回退至 `torch.cuda.empty_cache()`；其次对训练器源代码目录进行全局递归文本扫描，将所有硬编码调用替换为带 `getattr` 保护的安全回退执行。

### 3. Flow Matching 假性收敛与深度欠拟合 (100 步 vs 600 步)
* **认知陷阱**：在初期测试中，Loss 在第 10 步至第 20 步迅速跌入 0.29 平缓期，模型在数学均方误差上表现出“收敛平台”。然而 MOSS-Image 属于 50 亿参数量的连续流匹配架构，底模原生的低频预测先验极强，掩盖了 LoRA 欠拟合的假象。40 张图片跑 100 步实际仅经历了 10 个数据轮次（Epoch），无法动摇大底模的生成惯性。
* **修复举措**：将训练策略扩展至 60 个完整 Epoch（对应 600 步 Token 打包，2400 次单图图文迭代），并将 LoRA Alpha 由 16.0 提升至 32.0（注入强度达 1.0），配合无损修复底模，成功实现强力且鲜明的画风吸收。

### 4. ComfyUI 底层键名桥接机制真相
* **认知陷阱**：误以为 DiT 架构会排斥 UNet 前缀，在收尾打包时人工将 `lora_unet_` 替换为 `diffusion_model.`。
* **源码真相**：ComfyUI 官方原生加载器（`comfy/lora.py`）在遍历底模时，主动构建了 `key_map["lora_unet_{}".format(key_lora)] = k` 的索引字典，显式寻找 `lora_unet_` 前缀的 LoRA 键名。人为修改的 native 键名破坏了下划线规则，导致 756 个张量全部落空；而原汁原味的 `lora_unet_` 权重直接百分之百命中，完美起效。
