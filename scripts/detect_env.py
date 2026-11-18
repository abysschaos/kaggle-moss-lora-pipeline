import sys
from pathlib import Path

# 将项目根目录加入 sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

try:
    import train_local
    _, pythons = train_local.find_broad_comfy_paths()
    if pythons:
        print(str(pythons[0]))
except Exception:
    pass
