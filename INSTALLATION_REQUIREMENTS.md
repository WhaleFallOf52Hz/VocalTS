# VocalTS 安装与依赖说明

本文档基于当前工作区 `/home/pangjichen/workspace/VocalTS` 的实际仓库结构、各子仓库自带说明文件，以及当前本机 `conda` 环境 `VocalTS` 的已安装包状态整理而成。

目标：

- 说明这个工作区由哪些子仓库组成
- 给出一套相对完整、可落地的安装顺序
- 总结当前环境中关键依赖版本
- 标出需要特别注意的系统依赖、模型文件和已知兼容性问题

## 仓库组成

当前 `VocalTS` 根目录结构核心部分如下：

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

其中：

- [DDSP-SVC](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC) 负责预处理、训练和推理
- [MSST-WebUI](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI) 负责分离、卡拉 OK、人声提取、去混响等前处理
- [ncmdump](/home/pangjichen/workspace/VocalTS/third_party/ncmdump) 负责将网易云 `ncm` 音频转为 `mp3/flac`
- [taglib-2.1.1](/home/pangjichen/workspace/VocalTS/third_party/taglib-2.1.1) 是固定依赖库，适合作为子模块管理
- [Slice_wav_for_DDSP_SVC.py](/home/pangjichen/workspace/VocalTS/Slice_wav_for_DDSP_SVC.py) 用于把 `wav/flac` 切片成 DDSP-SVC 训练用 `wav`

## 当前环境现状

本机当前存在 `conda` 环境：

```text
/home/pangjichen/miniconda3/envs/VocalTS
```

其基础版本信息如下：

- Python: `3.11.15`
- torch: `2.5.1+cu121`
- torchaudio: `2.5.1+cu121`
- torchvision: `0.20.1+cu121`
- CUDA runtime reported by torch: `12.1`

说明：

- 这是“当前机器可参考环境”，不是各上游仓库共同维护的一套严格锁定环境。
- `torch.cuda.is_available()` 在我检查时返回 `False`，这通常说明当前会话没有可用 GPU，或当前节点没有把 GPU 暴露给这个环境。

## 上游依赖来源

### DDSP-SVC

主要依赖来源：

- [README.md](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/README.md)
- [cn_README.md](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/cn_README.md)
- [requirements.txt](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/requirements.txt)

`requirements.txt` 中核心依赖包括：

- `FreeSimpleGUI`
- `gin`
- `gin_config`
- `librosa`
- `matplotlib`
- `praat-parselmouth`
- `pyworld`
- `PyYAML`
- `resampy`
- `scikit_learn`
- `scipy`
- `sounddevice`
- `soundfile`
- `tensorboard`
- `torchcrepe`
- `torchfcpe`
- `tqdm`
- `transformers`

此外根据 README，还需要手动准备预训练资源：

- ContentVec 或 HubertSoft 编码器
- NSF-HiFiGAN 声码器
- RMVPE 音高提取器

### MSST-WebUI

主要依赖来源：

- [README.md](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/README.md)
- [requirements.txt](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/requirements.txt)
- [requirements-uv.txt](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/requirements-uv.txt)
- [pyproject.toml](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/pyproject.toml)

关键信息：

- 上游 README 推荐 `Python 3.10`
- `pyproject.toml` 声明 `requires-python = ">=3.10"`
- `requirements.txt` 推荐手动安装 `torch/torchaudio/torchvision`
- `pyproject.toml` 中依赖要求整体更“新”，例如：
  - `torch>=2.7.1`
  - `torchaudio>=2.7.1`
  - `torchvision>=0.22.1`
  - `transformers>=4.53.0`

但当前本机 `VocalTS` 环境中实际安装的是较早组合：

- `torch 2.5.1+cu121`
- `torchaudio 2.5.1+cu121`
- `torchvision 0.20.1+cu121`
- `transformers 4.35.2`

这说明：

- 当前仓库中的脚本工作流并不完全依赖 `pyproject.toml` 里最新声明的最高版本组合
- 如果目标是“复现当前项目工作流”，更适合优先参考当前环境和仓库内脚本，而不是盲目追最新依赖

### ncmdump

主要依赖来源：

- [README.md](/home/pangjichen/workspace/VocalTS/third_party/ncmdump/README.md)
- [CMakeLists.txt](/home/pangjichen/workspace/VocalTS/third_party/ncmdump/CMakeLists.txt)

Linux/macOS 下关键系统依赖：

- `cmake`
- C++17 编译器
- `zlib`
- `TagLib 2.x`

当前仓库里已经包含：

- [taglib-2.1.1](/home/pangjichen/workspace/VocalTS/third_party/taglib-2.1.1)
- 已编译好的本地二进制：
  [ncmdump](/home/pangjichen/workspace/VocalTS/third_party/ncmdump/build/ncmdump)

因此在这台机器上，如果该二进制可运行，可以不必重新编译 `ncmdump`。

## 顶层仓库组织建议

当前最适合的维护方式，是把 [VocalTS](/home/pangjichen/workspace/VocalTS) 作为顶层主仓库统一提交，而不是继续依赖 `git submodule`。

原因：

- 这个目录的价值已经不只是“上游仓库集合”，而是一个实际可运行的完整工作流
- 你已经在 `MSST-WebUI`、`DDSP-SVC`、根目录脚本和 Markdown 文档之间做了联动改造
- `MSST-WebUI/configs` 这类本地配置需要长期纳入版本控制，`submodule` 对这种场景维护成本较高

当前建议：

- 顶层仓库统一管理 `third_party/DDSP-SVC`、`third_party/MSST-WebUI`、`third_party/ncmdump`、脚本和文档
- `third_party/taglib-2.1.1` 作为固定第三方库，以 git submodule 形式管理
- `MSST-WebUI/configs` 和 `MSST-WebUI/config_unofficial` 完全放开，直接跟踪需要的配置文件
- 保留 `third_party/DDSP-SVC/data` 和 `third_party/DDSP-SVC/exp` 的软链接形式用于本地运行，但在顶层仓库中忽略，不提交实验数据和 checkpoint
- `third_party/DDSP-SVC/pretrain` 和 `third_party/MSST-WebUI/pretrain` 用于本地模型权重存放，也应忽略，不上传 GitHub

关于软链接：

- Git 可以跟踪软链接本身
- 不会跟踪软链接目标目录里的全部数据
- 这正适合当前 `data` 指向数据集、`exp` 指向 checkpoint 目录的用法

如果从 GitHub 拉取顶层仓库，建议使用：

```bash
git clone --recurse-submodules <your-repo-url>
```

如果已经 clone 过，再执行：

```bash
git submodule update --init --recursive
```

## 推荐安装流程

下面这套流程目标是尽量贴近当前仓库实际使用方式，而不是追求最“官方标准”的独立安装。

### 1. 准备系统依赖

在 Ubuntu 或 Debian 系 Linux 上，建议先安装：

```bash
sudo apt update
sudo apt install -y \
  git \
  build-essential \
  cmake \
  pkg-config \
  zlib1g-dev \
  ffmpeg \
  libsndfile1 \
  libasound2-dev
```

说明：

- `ffmpeg` 对音频转码和部分后处理很有帮助
- `libsndfile1` 对 `soundfile` 读取音频是常见底层依赖
- `libasound2-dev` 常用于 `sounddevice`

### 2. 创建 conda 环境

推荐新建环境，而不是直接污染 `base`：

```bash
conda create -n VocalTS python=3.11 -y
conda activate VocalTS
```

为什么这里推荐 `Python 3.11`：

- 当前机器上的 `VocalTS` 环境就是 `Python 3.11.15`
- `DDSP-SVC` README 明确提到 `Python 3.11` 可运行
- 这比单纯跟随 `MSST-WebUI` README 的 `Python 3.10` 更接近当前仓库的实际落地环境

如果你更倾向于贴近 `MSST-WebUI` 上游 README，也可以选择 `Python 3.10`，但这会和当前工作区的现有环境有所偏离。

### 3. 安装 PyTorch

如果你使用 CUDA 12.1，并希望贴近当前机器环境，可先安装：

```bash
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

如果你的机器 CUDA 版本不同，需要改成对应的 PyTorch 源和 wheel 组合。

### 4. 安装 DDSP-SVC 依赖

切换到目录：

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC
```

安装：

```bash
pip install -r requirements.txt
```

### 5. 安装 MSST-WebUI 依赖

切换到目录：

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI
```

建议优先使用较保守的 requirements 文件：

```bash
pip install -r requirements.txt --only-binary=samplerate
```

补充说明：

- `requirements-uv.txt` 更像一份较完整的锁定依赖导出文件
- 该文件当前看起来是 UTF-16 编码导出的文本，不适合作为首选手工维护入口
- `pyproject.toml` 中的版本整体偏新，若直接完全按其安装，可能与当前仓库里已经跑通的脚本环境不完全一致

### 6. 安装通用补充依赖

当前环境里实际还用到了不少不一定会被两边 requirements 完全覆盖的包。为了尽量贴近当前环境，可补装：

```bash
pip install \
  accelerate \
  aiohttp \
  colorama \
  demucs==4.0.0 \
  fastapi==0.111.0 \
  gradio==4.38.1 \
  lightning \
  pedalboard \
  protobuf==3.20.3 \
  pydantic==2.10.6 \
  safetensors \
  samplerate \
  soxr \
  spafe==0.3.2 \
  tensorboard \
  transformers \
  bitsandbytes
```

如果你只在 CPU 上运行，可以先不装 `bitsandbytes`。

### 7. 处理 `librosa` 兼容性

`MSST-WebUI` README 明确提到过 `librosa` 兼容问题，并建议：

```bash
pip uninstall librosa -y
pip install tools/webUI_for_clouds/librosa-0.9.2-py3-none-any.whl
```

我在当前 `VocalTS` 环境中实际测试导入 `librosa` 时，遇到了运行时错误：

```text
RuntimeError: cannot cache function '__shear_dense': no locator available for file '.../librosa/util/utils.py'
```

这说明当前环境的 `librosa/numba` 组合并不完全健康。

建议做法：

- 优先使用 `MSST-WebUI` README 提供的 `librosa 0.9.2` 修补方案
- 如果仍有问题，再考虑重新安装 `numba`、`llvmlite` 与 `librosa`
- 在文档或脚本中尽量避免混用多个来源安装的 `librosa`

### 8. 配置 DDSP-SVC 预训练模型

按照 [README.md](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/README.md) 的说明，需要在 [third_party/DDSP-SVC/pretrain](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/pretrain) 下手动准备以下预训练资源。

说明：

- 这些文件用于本地训练和推理
- 它们已经被顶层 `.gitignore` 忽略，不应上传到 GitHub

#### 特征编码器

二选一即可：

1. `ContentVec`

- 下载地址：
  `https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin?download=true`
- 放置目录：
  [third_party/DDSP-SVC/pretrain/contentvec](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/pretrain/contentvec)

2. `HubertSoft`

- 下载地址：
  `https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt`
- 放置目录：
  [third_party/DDSP-SVC/pretrain/hubert](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/pretrain/hubert)
- 如果选择 `HubertSoft`，还需要同步修改 `DDSP-SVC` 的配置文件

#### 声码器

推荐下载并解压预训练 `NSF-HiFiGAN`：

- 下载地址：
  `https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip`
- 解压后目录：
  [third_party/DDSP-SVC/pretrain/nsf_hifigan](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/pretrain/nsf_hifigan)

要求：

- 声码器权重文件需要放在配置文件 `vocoder.ckpt` 指定的位置
- 默认位置是：
  [third_party/DDSP-SVC/pretrain/nsf_hifigan/model](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/pretrain/nsf_hifigan/model)
- `config.json` 需要和模型文件放在同目录，例如：
  [third_party/DDSP-SVC/pretrain/nsf_hifigan/config.json](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/pretrain/nsf_hifigan/config.json)

如果你后续想进一步提升音质，也可以使用 `SingingVocoders` 另行微调声码器。

#### 音高提取器

推荐下载并解压 `RMVPE`：

- 下载地址：
  `https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip`
- 解压目标目录：
  [third_party/DDSP-SVC/pretrain](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/pretrain)

最终至少应满足下面这几类资源已经就位：

- `contentvec` 或 `hubert`
- `nsf_hifigan/model`
- `nsf_hifigan/config.json`
- `rmvpe`

这是 DDSP-SVC 正常训练和推理的必要条件。

### 9. 配置 MSST-WebUI 预训练模型

`MSST-WebUI` 的模型一般放在：

```text
/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/pretrain
```

当前仓库也提供了模型配置索引：

- [models_info.json](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/data/models_info.json)

你可以通过仓库脚本或 Hugging Face 下载对应模型，再放到 `pretrain/` 下的目标位置。

### 10. 处理 ncmdump

优先检查是否可以直接使用已有二进制：

```bash
/home/pangjichen/workspace/VocalTS/third_party/ncmdump/build/ncmdump -h
```

如果可用，可以直接把它加入 `PATH`，或写 alias。

如果不可用，再重新编译。为了规避 `sudo`，更推荐把 `taglib` 安装到当前 `conda` 环境前缀下。Linux 下一种可行方案是：

1. 先编译并安装 `taglib-2.1.1`
2. 再编译 `ncmdump`

示例：

```bash
conda activate VocalTS

cd /home/pangjichen/workspace/VocalTS/third_party/taglib-2.1.1
cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release .
make -j$(nproc)
make install

cd /home/pangjichen/workspace/VocalTS/third_party/ncmdump
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:${CMAKE_PREFIX_PATH}"
cmake -DCMAKE_BUILD_TYPE=Release -B build
cmake --build build -j$(nproc)
```

说明：

- `CMAKE_INSTALL_PREFIX="$CONDA_PREFIX"` 会把 `taglib` 安装到当前 conda 环境，而不是系统目录
- 这样通常不需要 `sudo`
- `CMAKE_PREFIX_PATH="$CONDA_PREFIX"` 可以帮助 `cmake` 在编译 `ncmdump` 时优先找到 conda 环境里的 `TagLib`
- 如果你希望 `pkg-config` 也优先读取 conda 环境里的 `.pc` 文件，可以额外执行：

```bash
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH}"
```

## 当前环境中值得关注的关键包

从当前 `conda` 环境 `VocalTS` 的 `pip list` 来看，比较关键的版本包括：

- `torch 2.5.1+cu121`
- `torchaudio 2.5.1+cu121`
- `torchvision 0.20.1+cu121`
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
- `transformers 4.35.2`

这些版本说明当前环境更像是一个“实际跑通后逐步补齐”的工作环境，而不是严格遵循某一份单独 requirements 文件的全新环境。

## 推荐的最小 requirements 思路

如果你想单独整理一份“偏当前环境”的 requirements，可以以以下思路为基础：

### 核心运行时

```text
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
numpy
scipy
soundfile
librosa
tqdm
PyYAML
```

### DDSP-SVC 相关

```text
FreeSimpleGUI
gin
gin_config
matplotlib
praat-parselmouth
pyworld
resampy
scikit_learn
sounddevice
tensorboard
torchcrepe
torchfcpe
transformers
```

### MSST-WebUI 相关

```text
accelerate
asteroid==0.7.0
audiomentations==0.24.0
auraloss
beartype==0.14.1
colorama
demucs==4.0.0
fastapi==0.111.0
gradio==4.38.1
lightning
ml_collections
mido
omegaconf==2.2.3
pedalboard
prodigyopt
protobuf==3.20.3
rotary_embedding_torch==0.3.5
samplerate
segmentation_models_pytorch==0.3.3
spafe==0.3.2
torch_audiomentations
torch_log_wmse
torchmetrics==0.11.4
torchseg
```

## 已知注意事项

- `MSST-WebUI` 上游 README 推荐 `Python 3.10`，而当前本机环境是 `Python 3.11.15`
- `MSST-WebUI/pyproject.toml` 对 torch 等依赖版本要求比当前环境更高
- 当前环境中 `librosa` 存在导入时触发 `numba` 缓存错误的问题
- `ncmdump` 在 Linux 下通常需要 `TagLib 2.x`，系统自带老版本不一定够用
- `DDSP-SVC` 和 `MSST-WebUI` 都依赖相对路径，因此安装完成后实际运行时仍需注意切换目录

## 建议的最终安装顺序

如果从零开始，建议按这个顺序做：

1. 安装系统依赖
2. 创建并激活 `conda` 环境 `VocalTS`
3. 安装与本机 CUDA 匹配的 `torch/torchaudio/torchvision`
4. 安装 `DDSP-SVC/requirements.txt`
5. 安装 `MSST-WebUI/requirements.txt`
6. 处理 `librosa` 兼容性
7. 准备 DDSP-SVC 预训练模型
8. 准备 MSST-WebUI 预训练模型
9. 检查 `ncmdump` 二进制能否直接使用
10. 如有需要，再编译 `taglib` 和 `ncmdump`

## 相关文档

- [TRAINING_INFERENCE_WORKFLOW.md](/home/pangjichen/workspace/VocalTS/TRAINING_INFERENCE_WORKFLOW.md)
- [DDSP-SVC/README.md](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/README.md)
- [DDSP-SVC/cn_README.md](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/cn_README.md)
- [MSST-WebUI/README.md](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/README.md)
- [MSST-WebUI/pyproject.toml](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/pyproject.toml)
- [ncmdump/README.md](/home/pangjichen/workspace/VocalTS/third_party/ncmdump/README.md)
