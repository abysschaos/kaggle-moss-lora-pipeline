## 变更摘要 (Summary)
<!-- 简要说明本次 PR 解决的问题、新增的功能或重构内容 -->

## 变更类型 (Type of Change)
- [ ] 新功能 (`feat`)
- [ ] 缺陷修复 (`fix`)
- [ ] 文档与仓库工程对齐 (`docs`)
- [ ] 重构或死代码清理 (`refactor`)
- [ ] 发布与版本同步 (`chore(release)`)

## 核验清单 (Checklist)
- [ ] 已检查并确认若涉及参数/配置变更，本地 (`train_local.py` / `gui_local.py`) 与云端 (`train_moss.py` / Notebook) 已同步
- [ ] 已同步更新 `README.md`、`.env.example` 及 `docs/` 下对应参考文档
- [ ] PowerShell 脚本已在思维/语法层面确认兼容 Windows PowerShell 5.1
- [ ] 已本地运行 `python -m py_compile` 且若修改了启动器则通过 `.\scripts\build_exe.ps1` 编译验证
- [ ] 已自检 Diff 中不包含 `local_config.json`、私钥/Token、本地私人路径或大型权重二进制文件
