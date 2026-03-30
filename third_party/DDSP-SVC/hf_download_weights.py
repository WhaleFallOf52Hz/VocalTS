import os
from huggingface_hub import hf_hub_download

# 开启 hf_transfer 加速
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 可选：多线程（默认已经不错）
os.environ["HF_TRANSFER_MAX_CONCURRENCY"] = "8"

repo_id = "Sucial/Dereverb-Echo_Mel_Band_Roformer"
filename = "pytorch_model.bin"
local_dir = "pretrain/contentvec"

os.makedirs(local_dir, exist_ok=True)

file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Downloaded to:", file_path)