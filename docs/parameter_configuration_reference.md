# MOSS-Image 项目参数调节位置与 CLI / 环境变量速查手册

本手册完整罗列本项目在 **Windows 本地模式（[train_local.py](../train_local.py) / [gui_local.py](../gui_local.py)）** 与 **云端模式（[train_moss.py](../train_moss.py) / Notebook）** 中的全量命令行开关、环境变量名称、默认值及底层配置映射位置，方便日常调参、自动化调度与排障查阅。

---

## 1. 全量核心参数速查总表

### 1.1 任务命名、数据源与触发词

| 参数名称 | 物理含义 | 本地 CLI ([train_local.py](../train_local.py) `main()`) | 云端 CLI & 环境变量 ([train_moss.py](../train_moss.py)) |
| :--- | :--- | :--- | :--- |
| **LoRA 名称** | 产出权重前缀与 `output/<lora_name>` 子目录名 | `--lora_name`（兼容别名 `--run_name`，默认 `"my_lora"`） | `--lora_name` / `LORA_NAME`（默认 `"my_lora"`） |
| **全局触发词** | 单图集模式下前置注入每张图片 `.txt` 标注最前的触发词 | `--trigger_word`（默认 `""`） | `--trigger_word` / `TRIGGER_WORD`（默认 `""`） |
| **本地图集目录** | 包含源图片与同名 `.txt` 标注的单图集文件夹 | `--data_dir`（默认 `"./dataset"`） | `--dataset_dir` / `DATASET_DIR`（默认自动扫描 `/kaggle/input/`） |
| **云端归档直链** | 当未通过 Kaggle Dataset 挂载时从直链自动拉取 `.zip` | — | `--dataset_url` / `DATASET_URL`（默认 `""`） |
| **多子集配比 JSON** | 多人物/多画风/概念独立子目录、专属触发词与独立 Repeat | `--subsets_json`（JSON 文件路径或数组串，默认 `""`） | `--subsets_json` / `SUBSETS_JSON`（默认 `""`） |

### 1.2 步数配方与样本暴露

| 参数名称 | 物理含义 | 推荐范围 | 本地 CLI | 云端 CLI & 环境变量 |
| :--- | :--- | :--- | :--- | :--- |
| **单图重复次数** | 单图在每轮中被采样的次数（写入 `<repeats>_<name>` 目录前缀） | 人物 `2~6`，画风 `8~15` | `--repeats`（默认 `10`） | `--repeats` / `REPEATS`（默认 `10`） |
| **训练轮数** | 结合图数、Repeats 与动态等效 Batch 推算推荐总步数 | 人物 `3~5`，画风 `5~8` | `--epochs`（默认 `6`） | `--epochs` / `EPOCHS`（默认 `6`） |
| **总优化步数** | 优化器反向传播更新步数（同时决定 Cosine 退火周期与 5% Warmup） | `200 ~ 800` | `--steps`（默认 `600`） | `--steps` / `STEPS`（默认 `600`） |
| **梯度累积 (`GRAD_ACC`)** | 微批次累计 N 步再执行一次权重更新，降低单人物/多子集混合梯度方差 | 画风 `1`，人物/多子集 `2~4` | `--grad_acc`（默认 `1`） | `--grad_acc` / `GRAD_ACC`（默认 `1`） |
| **时间步重心 (`TIMESTEP_MU`)** | 控制 Logit-Normal Flow Matching 采样重心（`TIMESTEP="moss"`） | `-1.09` 画风 / `-0.60` 多画风 / `0.0` 动作 / `0.35` 角色细节 | `--timestep_mu`（默认 `-1.09`） | `--timestep_mu` / `TIMESTEP_MU`（默认 `-1.09`） |
| **基础学习率** | 优化器学习速率（配合 Cosine 衰减至 `min_value=0.1`） | `8e-5 ~ 1.2e-4` | `--lr`（默认 `1e-4`） | `--lr` / `LR`（默认 `"1e-4"`） |

### 1.3 低秩适配器架构与高阶微调

| 参数名称 | 物理含义 | 取值选项 | 本地 CLI | 云端 CLI & 环境变量 |
| :--- | :--- | :--- | :--- | :--- |
| **适配器秩 (`DIM`)** | 低秩矩阵内部维度 | 人物 `16~32`，画风/多子集 `32~64` | `--rank`（默认 `32`） | `--rank` / `RANK`（默认 `32`） |
| **缩放系数 (`ALPHA`)** | 注入缩放幅度（`ALPHA / DIM` 决定基础放大比） | 人物 `16.0~32.0`，画风 `=Rank` | `--alpha`（默认 `32.0`） | `--alpha` / `ALPHA`（默认 `32.0`） |
| **适配器算法 (`ALGO`)** | 权重适应结构（传统 LoRA / 哈达玛 LoHa / 克罗内克 LoKr） | `lora` / `loha` / `lokr` | `--algo`（默认 `"lora"`） | `--algo` / `ALGO`（默认 `"lora"`） |
| **DoRA 权重分解** | 将权重分解为幅值与方向独立学习（开启自动注入 `DORA_DETACH=True`） | 开关标志 | `--dora`（`store_true`） | `--dora` / `USE_DORA`（默认 `false`） |
| **目标注入层 (`PRESET`)** | Transformer 适配器注入范围预设 | `attn` / `attn-mlp` / `attn-mlp+modulation` / `attn-mlp+text-refiner` / `full-block` / `all` | `--target_preset`（默认 `"attn-mlp"`） | `--target_preset` / `TARGET_PRESET`（默认 `"attn-mlp"`） |
| **LoRA+ 学习率倍率** | 零初始化上投影分支相对于下投影分支的学习率乘数 | `1.0`（关闭）或 `2.0 ~ 4.0` | `--loraplus_ratio`（默认 `1.0`） | `--loraplus_ratio` / `LORAPLUS_RATIO`（默认 `1.0`） |

### 1.4 硬件、底模变体、显存与缓存路由

| 参数名称 | 物理含义 | 取值选项 | 本地 CLI | 云端 CLI & 环境变量 |
| :--- | :--- | :--- | :--- | :--- |
| **底模变体 (`VARIANT`)** | 切换适配 Base / Heal / SFT 架构差异与 FP16 防溢出策略 | 本地 `auto`/`base`/`sft`；云端 `heal`/`base`/`sft` | `--variant`（默认 `"auto"`） | `--variant` / `MODEL_VARIANT`（默认 `"heal"`） |
| **云端底模量化** | 控制从 HuggingFace 拉取 `int8` 量化版或全精度底模 | `auto` / `int8` / `full` | 本地按文件名自动识别 | `--base_quant` / `BASE_QUANT`（默认 `"auto"`） |
| **底模计算精度 (`BASE_PRECISION`)** | 传递至引擎的量化/精度内核策略；`fp8_compute` 需 `sm89+` 且非 `int8_convrot` 底模，引擎自动配对 `FP8_DGRAD="16bit"` | `auto` / `int8` / `fp8` / `fp8_compute` / `compute` | `--base_precision`（默认 `"auto"`） | `--base_precision` / `BASE_PRECISION`（默认 `"auto"`） |
| **显存优化档位** | 映射至引擎流式缓冲、激活检查点与实测安全 `MAX_TOKENS`（`6g/8g:4352`、`12g:6144`、`16g:8192`） | `auto` / `6g` / `8g` / `12g` / `16g` | `--vram_preset`（默认 `"auto"`） | `--vram_preset` / `VRAM_PRESET`（默认 `"auto"`） |
| **训练目标分辨率** | 动态分桶像素面积基准 (`TARGET_AREA = resolution^2`)；`896` 对应 `3136` 视觉 Token，可在 `8192` 下打包 2 图/步 | `1024` / `896` / `768` / `512` | `--resolution`（默认 `1024`） | `--resolution` / `RESOLUTION`（默认 `1024`） |
| **计算精度** | 主训练张量精度（无原生 BF16 显卡自动回退 FP16，非 Heal 自动开启 `FP16_FIX="online"` 与 `FP16_FIX_KWARGS={"c": 1/16}`） | 本地 `bf16`/`fp16`；云端增设 `auto` | `--precision`（默认 `"bf16"`） | `--precision` / `PRECISION`（默认 `"auto"`） |
| **Triton 内核调优 (`TUNE`)** | 控制 Triton 矩阵乘法内核自动调优缓存模式（`table` 首次慢但长训更快） | `off` / `table` | `--tune`（默认 `"off"`） | `--tune` / `TUNE`（默认 `"off"`） |
| **超长样本策略 (`OVERLONG`)** | 当单图视觉 + `.txt` 总 Token 超出 `MAX_TOKENS` 时的处理策略 | `drop_tokens`（仅截断超限文本保留图像）/ `drop` | `--overlong`（默认 `"drop_tokens"`） | `--overlong` / `OVERLONG`（默认 `"drop_tokens"`） |
| **文本提取设备** | Qwen3.5-9B 离线文本特征提取宿主设备 | 本地 `cpu`/`cuda`；云端 `auto`/`cuda`/`cpu` | `--text_device`（默认 `"cpu"`） | `--text_device` / `TEXT_DEVICE`（默认 `"auto"`） |
| **VAE 提取设备** | FLUX.2 VAE 潜空间压缩设备 | `cuda` / `cpu` | `--vae_device`（默认 `"cuda"`） | `--vae_device` / `VAE_DEVICE`（默认 `"cuda"`） |

### 1.5 续炼恢复、存档与本地资产覆盖

| 参数名称 | 物理含义 | 默认值 / 可选值 | 本地 CLI | 云端 CLI & 环境变量 |
| :--- | :--- | :--- | :--- | :--- |
| **训练模式** | 全新训练 / 从检查点断点续炼 / 从已有 `.safetensors` 延续微调 | `fresh` / `resume` / `init_from` | `--train_mode`（默认 `"fresh"`） | `--train_mode` / `TRAIN_MODE`（默认 `"fresh"`） |
| **续炼检查点路径** | 搭配 `resume` 模式使用的检查点目录 | `"latest"` 或具体路径 | `--resume_from` | `--resume_from` / `RESUME_FROM` |
| **基底 LoRA 路径** | 搭配 `init_from` 模式继续训练的 `.safetensors` 文件路径 | `""` | `--init_from` | `--init_from` / `INIT_FROM` |
| **定期存盘间隔** | 每隔 N 步保存一次中间 checkpoint 与输出权重 | `100` | `--save_every` | `--save_every` / `SAVE_EVERY` |
| **输出根目录** | 本地专属权重与 `training_summary.json` 保存根目录 | `"./output"` | `--output_dir` | 固定 `/kaggle/working/output` |
| **ComfyUI 模型库** | 本地 `diffusion_models / text_encoders / vae / loras` 根目录 | 自动全盘探测 | `--comfy_models_dir` | — |
| **DiT 单文件覆盖** | 显式指定本地 `moss_image_*.safetensors` 绝对路径 | 自动匹配 | `--transformer_path` | — |
| **编码器单文件覆盖** | 显式指定本地 `qwen3.5_9b_*.safetensors` 绝对路径 | 自动匹配 | `--text_encoder_path` | — |
| **VAE 单文件覆盖** | 显式指定本地 `flux2-vae*.safetensors` 绝对路径 | 自动匹配 | `--vae_path` | — |
| **自动同步 ComfyUI** | 训练结束与后台轮询同步时自动推送至 `ComfyUI/models/loras/` | 开关标志 | `--copy_to_comfy_loras` | — |
| **版本查询** | 打印当前遵循 [VERSION](../VERSION) 单一信源的版本号并退出 | 动作标志 | `--version` | `--version` |

---

## 2. 本地与云端实战命令行范例

### 2.1 Windows 本地 CLI 单图集微调
```powershell
python train_local.py `
    --data_dir "./dataset/my_character" `
    --lora_name "my_character_v1" `
    --trigger_word "my_character_char" `
    --repeats 6 `
    --epochs 4 `
    --steps 400 `
    --rank 32 `
    --alpha 32.0 `
    --algo lora `
    --dora `
    --target_preset attn-mlp `
    --loraplus_ratio 2.0 `
    --grad_acc 2 `
    --timestep_mu 0.35 `
    --vram_preset 16g `
    --resolution 896 `
    --base_precision auto `
    --tune off `
    --overlong drop_tokens `
    --precision bf16 `
    --text_device cpu `
    --copy_to_comfy_loras
```

### 2.2 多子集配比 JSON 规范（`--subsets_json` / `SUBSETS_JSON`）
`--subsets_json` 同时支持 **JSON 文件路径** 或 **单行 JSON 数组字符串**。[train_local.py](../train_local.py) 中的 `load_subsets_spec()` 与 `prepare_multi_subset_dataset()` 会校验每个子目录并为启用的子集创建独立 Kohya `<repeats>_<subset_name>` 目录，同时保持 `"artist_from_folder": None` 防止无标注图被注入文件夹名称。

```json
[
  {
    "name": "style_ink",
    "path": "D:/datasets/ink_paintings",
    "trigger": "ink_wash_style",
    "repeats": 8,
    "enabled": true
  },
  {
    "name": "style_watercolor",
    "path": "D:/datasets/watercolor",
    "trigger": "soft_watercolor_style",
    "repeats": 12,
    "enabled": true
  }
]
```

> 使用 GUI 控制台时，只需在【训练集与多子集配比】卡片勾选【启用多子集配比模式】，点击【⚖️ 一键配平 Repeat】即可按各子集样本数量反比自动算好 `repeats`，无需手算 JSON。

---

## 3. 底层引擎生成的配置位置与自定义修改点

两个核心引擎脚本在运行时均会将有效参数组装并落盘到 `KohakuMossTrainer/configs/user/` 下（供 `kogine run` 调用），方便进阶用户核验实际传给引擎的 `knobs`：

1. **数据集配置文件**：
   - 本地：`KohakuMossTrainer/configs/user/<lora_name>_local_data.py`（由 [train_local.py](../train_local.py) `main()` 生成，并在离线缓存完成后清理 `TEXT_ENCODER_KWARGS` / `IMAGE_ENCODER_KWARGS`）
   - 云端：`KohakuMossTrainer/configs/user/<lora_name>_data.py`（由 [train_moss.py](../train_moss.py) 生成）
2. **主训练配置文件**：
   - 本地：`KohakuMossTrainer/configs/user/<lora_name>_local.py`
   - 云端：`KohakuMossTrainer/configs/user/<lora_name>.py`
3. **关键引擎内置策略（已在源码自动生成并经 `knobs.validate()` 校验）**：
   - 启用 `--dora` 时自动注入 `DORA_DETACH = True`；
   - 选择 `--algo loha` 或 `lokr` 时自动注入上游标准安全 `ALGO_KWARGS`；
   - 显式锁定 `SAVE_FUSED_QKV = False` 与 `SAVE_FORMAT = "both"` 保障原生导出与 ComfyUI 兼容性；
   - 使用 `SCHEDULE = {"mode": "cosine", "warmup": max(5, min(100, int(effective_steps * 0.05))), "end": -1, "min_value": 0.1}` 动态适配总步数；
   - 显式写入 `OVERLONG = "drop_tokens"`（默认保留图像并截断超限长文本）、`GRAD_ACC`、`TIMESTEP = "moss"` + `TIMESTEP_KWARGS = {"mu": ...}`、`BASE_PRECISION` 与可选 `FP8_DGRAD = "16bit"`。
   - 本地运行时通过 `KOHAKUMOSS_PAUSE_FILE` 环境变量与注入到 `drivers/native.py` 的钩子实现随时安全暂停存盘。

4. **全量环境变量参考**：所有云端与本地路径环境变量的完整清单与默认值模板，请直接查阅根目录 [.env.example](../.env.example)。
