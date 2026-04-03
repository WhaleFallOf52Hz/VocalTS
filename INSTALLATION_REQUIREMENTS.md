# VocalTS 安装指南（基于当前 `VocalTS` conda 环境 `pip list`）

本文档用于**复现当前工作区已跑通的环境与流程**，不是对各上游仓库“最新依赖”的机械跟随。

适用目录：`/home/pangjichen/workspace/VocalTS`

---

## 1. 目标与原则

本指南的目标：

- 按当前机器 `VocalTS` conda 环境的 `pip list` 作为默认基线安装
- 严格对齐 `third_party` 三个核心组件的上游安装流程
- 减少“版本追新导致工作流失效”的风险

默认策略（已固定）：

- 主线：**保守复现当前环境**
- Python：**3.11**
- `librosa`：作为**故障排查步骤**，默认不强制替换

---

## 2. 仓库结构与组件职责

```text
/home/pangjichen/workspace/VocalTS
├─ third_party
│  ├─ DDSP-SVC
│  ├─ MSST-WebUI
│  ├─ ncmdump
│  └─ taglib-2.1.1
├─ Slice_wav_for_DDSP_SVC.py
├─ Analyse_wav_SampleRate.py
└─ linked_data -> /data2/pangjichen/VocalTS/trainning_data
```

组件职责：

- `third_party/DDSP-SVC`：SVC 训练/推理主链路
- `third_party/MSST-WebUI`：伴奏分离、人声提取、去混响等前处理
- `third_party/ncmdump`：`ncm -> mp3/flac` 转换工具（C++ 二进制）
- `third_party/taglib-2.1.1`：`ncmdump` 依赖的 TagLib 2.x 源码

---

## 3. 当前环境基线（来自现有 `pip list` 记录）

当前 `VocalTS` conda 环境关键版本：

- Python `3.11.15`
- `torch 2.5.1+cu121`
- `torchaudio 2.5.1+cu121`
- `torchvision 0.20.1+cu121`
- `transformers 4.35.2`
- `accelerate 1.13.0`
- `asteroid 0.7.0`
- `audiomentations 0.24.0`
- `demucs 4.0.0`
- `fastapi 0.111.0`
- `gradio 4.38.1`
- `lightning 2.6.1`
- `pedalboard 0.8.9`
- `protobuf 4.25.1`
- `pydantic 2.10.6`
- `pyworld 0.3.5`
- `praat-parselmouth 0.4.7`
- `samplerate 0.2.3`
- `soundfile 0.11.0`
- `torchcrepe 0.0.20`
- `torchfcpe 0.0.4`

说明：

- 本基线用于“复现当前工程可运行状态”。
- 这不等于 `MSST-WebUI` 的最新 `pyproject.toml` 依赖上限。

---

## 4. 上游安装口径（已核对）

### 4.1 DDSP-SVC

上游口径：先安装 PyTorch，再执行：

```bash
pip install -r requirements.txt
```

并手动准备预训练资源：

- `ContentVec` 或 `HubertSoft`
- `NSF-HiFiGAN`
- `RMVPE`

### 4.2 MSST-WebUI

上游 README 推荐：

- Python 3.10（但当前策略固定 3.11 复现）
- 手动安装 torch 三件套
- 执行：

```bash
pip install -r requirements.txt --only-binary=samplerate
```

并提示 `librosa` 兼容性修复（见第 10 节故障排查）。

### 4.3 ncmdump

`ncmdump` 是 C++/CMake 项目，不是 Python 包。关键依赖：

- C++17 编译器
- CMake
- ZLIB
- TagLib 2.x

优先使用仓库已有二进制；不可用再本地编译。

---

## 5. 推荐安装流程（保守复现主线）

### 步骤 1：优先用 conda 安装系统级依赖（无 sudo）

```bash
conda create -n VocalTS python=3.11 -y
conda activate VocalTS

conda install -y -c conda-forge \
  cmake \
  make \
  pkg-config \
  cxx-compiler \
  zlib \
  ffmpeg \
  libsndfile \
  alsa-lib
```

说明：

- 该方案把大多数系统级依赖放到 conda 环境内管理，尽量避免 `sudo`。
- `git` 建议在系统层提前安装；若当前机器已可用可直接跳过。
- `cxx-compiler` 是 conda-forge 的编译器元包，用于替代 `build-essential`。

### 步骤 2：确认环境已激活

```bash
conda activate VocalTS
```

### 步骤 3：安装 PyTorch（按当前基线）

```bash
pip install \
  torch==2.5.1+cu121 \
  torchaudio==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

> 若你的 CUDA 与 `cu121` 不匹配，请改用对应 wheel 源。

### 步骤 4：安装 DDSP-SVC 依赖

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC
pip install -r requirements.txt
```

### 步骤 5：安装 MSST-WebUI 依赖（保守轨）

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI
pip install -r requirements.txt --only-binary=samplerate
```

### 步骤 6：按当前环境补齐常用包（避免漏依赖）

```bash
pip install \
  accelerate==1.13.0 \
  asteroid==0.7.0 \
  audiomentations==0.24.0 \
  colorama \
  demucs==4.0.0 \
  fastapi==0.111.0 \
  gradio==4.38.1 \
  lightning==2.6.1 \
  pedalboard==0.8.9 \
  protobuf==4.25.1 \
  pydantic==2.10.6 \
  samplerate==0.2.3 \
  soundfile==0.11.0 \
  spafe==0.3.2 \
  tensorboard \
  torchcrepe==0.0.20 \
  torchfcpe==0.0.4 \
  transformers==4.35.2
```

可选：

```bash
pip install bitsandbytes
```

CPU-only 可先跳过 `bitsandbytes`。

### 可选：仅当 conda 方案不可用时再用 apt（需要 sudo）

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  zlib1g-dev \
  ffmpeg \
  libsndfile1 \
  libasound2-dev
```

---

## 6. DDSP-SVC 预训练模型准备

目标目录：

```text
/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/pretrain
```

### 6.1 特征编码器（二选一）

1) ContentVec

- 下载：
  `https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin?download=true`
- 放置：`pretrain/contentvec`

2) HubertSoft

- 下载：
  `https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt`
- 放置：`pretrain/hubert`
- 并同步修改 DDSP-SVC 配置文件对应编码器项

### 6.2 声码器 NSF-HiFiGAN

- 下载：
  `https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip`
- 解压到：`pretrain/nsf_hifigan`

保证以下文件就位：

- `pretrain/nsf_hifigan/model`（与配置中的 `vocoder.ckpt` 一致）
- `pretrain/nsf_hifigan/config.json`

### 6.3 音高提取器 RMVPE

- 下载：
  `https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip`
- 解压到：`pretrain/`

### 6.4 DDSP-SVC 最低可运行资源检查

至少应有：

- `contentvec` 或 `hubert`
- `nsf_hifigan/model`
- `nsf_hifigan/config.json`
- `rmvpe`

---

## 7. MSST-WebUI 预训练模型准备

模型目录通常为：

```text
/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/pretrain
```

可通过以下方式准备模型：

- 在 WebUI 的 Install Models 页面安装
- 或从 Hugging Face 手动下载并放入 `pretrain/`

索引文件：

- `third_party/MSST-WebUI/data/models_info.json`

---

## 8. ncmdump 使用与编译

### 8.1 优先复用现成二进制

先测试：

```bash
/home/pangjichen/workspace/VocalTS/third_party/ncmdump/build/ncmdump -h
```

能正常输出帮助即无需重编译。

### 8.2 失败时再编译（conda 环境内，无 sudo 优先）

```bash
conda activate VocalTS

conda install -y -c conda-forge cmake make pkg-config cxx-compiler zlib

cd /home/pangjichen/workspace/VocalTS/third_party/taglib-2.1.1
cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release .
make -j$(nproc)
make install

cd /home/pangjichen/workspace/VocalTS/third_party/ncmdump
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:${CMAKE_PREFIX_PATH}"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH}"
cmake -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build -j$(nproc)
```

说明：

- `CMAKE_INSTALL_PREFIX="$CONDA_PREFIX"`：把 TagLib 安装到当前 conda 环境
- `CMAKE_PREFIX_PATH` / `PKG_CONFIG_PATH`：让 `ncmdump` 优先发现 conda 内 TagLib

---

## 9. 兼容性风险与边界

### 9.1 版本冲突来源

- `MSST-WebUI` README 推荐 Python 3.10
- `MSST-WebUI/pyproject.toml` 依赖更激进（如更高 torch/transformers）
- 当前工作流实际依赖的是 `Python 3.11 + torch 2.5.1+cu121 + transformers 4.35.2`

### 9.2 建议

- 默认只走本指南“保守复现主线”
- 不要混合使用 `pyproject` 的最新依赖上限与当前基线
- 若要追新版本，请单独新建 conda 环境进行 A/B 验证

---

## 10. 常见故障排查

### 10.1 `librosa` / `numba` 相关报错

若出现类似：

```text
RuntimeError: cannot cache function '__shear_dense': no locator available for file '.../librosa/util/utils.py'
```

可按 MSST-WebUI 的建议修复：

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI
pip uninstall librosa -y
pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
```

若仍异常，再排查：

- `numba` 与 `llvmlite` 版本组合
- 是否混装了多个来源的 `librosa`

### 10.2 GPU 不可见

若 `torch.cuda.is_available()` 为 `False`，常见原因：

- 当前会话未暴露 GPU
- CUDA 驱动与已装 torch wheel 不匹配
- 非 GPU 节点运行

---

## 11. 建议的最终执行顺序

1. 安装系统依赖
2. 创建并激活 conda 环境（Python 3.11）
3. 使用 conda 安装系统级构建/音频依赖
4. 安装 PyTorch（按 CUDA 匹配）
5. 安装 DDSP-SVC requirements
6. 安装 MSST-WebUI requirements
7. 补齐当前基线常用包
8. 准备 DDSP-SVC 预训练模型
9. 准备 MSST-WebUI 预训练模型
10. 检查 `ncmdump` 现成二进制
11. 必要时编译 `taglib + ncmdump`
12. 出现 `librosa` 报错时再执行修复

---

## 12. 参考文档

- `TRAINING_INFERENCE_WORKFLOW.md`
- `third_party/DDSP-SVC/README.md`
- `third_party/DDSP-SVC/cn_README.md`
- `third_party/DDSP-SVC/requirements.txt`
- `third_party/MSST-WebUI/README.md`
- `third_party/MSST-WebUI/requirements.txt`
- `third_party/MSST-WebUI/requirements-uv.txt`
- `third_party/MSST-WebUI/pyproject.toml`
- `third_party/ncmdump/README.md`
- `third_party/ncmdump/CMakeLists.txt`
