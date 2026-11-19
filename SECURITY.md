# 安全政策 (Security Policy)

## 支持的版本矩阵

| 版本系列 | 维护状态 | 说明 |
| :--- | :--- | :--- |
| `1.2.x` | 维护中 | 当前活跃稳定分支（包含安全与引擎兼容性加固） |
| `< 1.2.0` | 已停止支持 | 请升级至最新 Release 以获取最新补丁 |

---

## 隐私与安全边界约定

1. **禁止提交敏感配置**：`local_config.json`、`SHA256SUMS.txt`、模型权重（`*.safetensors` / `*.bin` / `*.pt`）以及个人训练图集已全部配置于 `.gitignore` 中，切勿通过 `git add -f` 强制纳入版本控制。
2. **原生启动器可审计性**：仓库内不预置或提交编译好的二进制 `.exe` 源码文件，[launcher/MossTrainerLauncher.cs](launcher/MossTrainerLauncher.cs) 保持完全明文开源，用户可在本地 1 秒透明编译，亦可对照官方 Release 中的 `SHA256SUMS.txt` 校验散列值。
3. **Token 与凭据处理**：使用 `scripts/sync_github_release.ps1` 或 Kaggle CLI 时，请确保凭证仅存放在环境变量或本机系统凭据管理器中，切勿写入仓库文件或公开日志。

---

## 报告安全漏洞

如果您在本项目脚本、路径处理、启动器或辅助工具中发现潜在安全问题（例如命令注入、凭证泄露风险、非预期文件写入等）：

1. **请勿在公开 Issue 中直接披露可被利用的细节。**
2. 请通过 [GitHub Security Advisories](https://github.com/abysschaos/kaggle-moss-lora-pipeline/security/advisories/new) 发起私密安全报告，说明受影响的文件、复现步骤与潜在影响。
3. 维护团队将在确认后尽快修复并发布安全更新版本及 `CHANGELOG.md` 说明。
