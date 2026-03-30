import os
from huggingface_hub import hf_hub_download

# 开启 hf_transfer 加速
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "Sucial/Dereverb-Echo_Mel_Band_Roformer"
filename = "dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt"
local_dir = "pretrain/single_stem_models"

os.makedirs(local_dir, exist_ok=True)

file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print("Downloaded to:", file_path)