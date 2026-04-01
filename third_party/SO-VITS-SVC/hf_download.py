import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

# 启用 hf_transfer（需先安装: pip install -U huggingface_hub hf_transfer）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

REPO_ID = "ms903/sovits4.0-768vec-layer12"
REPO_TYPE = "dataset"
SUBFOLDER = "sovits_768l12_pre_large_320k"

# 目标目录：third_party/so-vits-svc/logs/44k
TARGET_DIR = Path(__file__).resolve().parent / "logs" / "44k"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
files_to_download = [f for f in all_files if f.startswith(f"{SUBFOLDER}/")]

if not files_to_download:
    raise RuntimeError(f"未找到目录: {SUBFOLDER}")

for f in files_to_download:
    rel_path = f[len(SUBFOLDER) + 1 :]  # 去掉前缀，仅保留子目录内相对路径
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        subfolder=SUBFOLDER,
        filename=rel_path,
        local_dir=str(TARGET_DIR),
        local_dir_use_symlinks=False,
    )

print(f"下载完成，共 {len(files_to_download)} 个文件 -> {TARGET_DIR}")