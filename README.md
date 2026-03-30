# VocalTS

`VocalTS` 是一个面向歌声处理与 DDSP-SVC 训练/推理的整合工作区。

这个仓库把若干上游项目和本地工作流脚本统一组织到一个顶层仓库中，方便集中维护：

- 音频解包：`third_party/ncmdump`
- 前处理与去混响：`third_party/MSST-WebUI`
- 训练与推理：`third_party/DDSP-SVC`
- 固定第三方库子模块：`third_party/taglib-2.1.1`
- 本地工作流脚本与说明文档：仓库根目录

## 目录结构

```text
VocalTS/
├─ third_party/
│  ├─ DDSP-SVC/
│  ├─ MSST-WebUI/
│  ├─ ncmdump/
│  └─ taglib-2.1.1/   # git submodule
├─ Analyse_wav_SampleRate.py
├─ Slice_wav_for_DDSP_SVC.py
├─ INSTALLATION_REQUIREMENTS.md
├─ TRAINING_INFERENCE_WORKFLOW.md
└─ linked_data -> /data2/pangjichen/VocalTS/trainning_data
```

## 文档

- 安装与依赖说明：
  [INSTALLATION_REQUIREMENTS.md](/home/pangjichen/workspace/VocalTS/INSTALLATION_REQUIREMENTS.md)
- 训练与推理流程：
  [TRAINING_INFERENCE_WORKFLOW.md](/home/pangjichen/workspace/VocalTS/TRAINING_INFERENCE_WORKFLOW.md)

## 快速开始

### 1. 克隆仓库

由于 `taglib-2.1.1` 使用了 submodule，建议使用：

```bash
git clone --recurse-submodules <your-repo-url>
cd VocalTS
```

如果已经 clone 过，再执行：

```bash
git submodule update --init --recursive
```

### 2. 安装环境

详细步骤见：

```text
INSTALLATION_REQUIREMENTS.md
```

### 3. 运行工作流

完整训练/推理流程见：

```text
TRAINING_INFERENCE_WORKFLOW.md
```

## 版本管理约定

这个顶层仓库用于统一管理：

- 本地工作流脚本
- 文档
- 对 `MSST-WebUI`、`DDSP-SVC`、`ncmdump` 的本地定制修改

以下内容默认不提交：

- `linked_data`
- `third_party/DDSP-SVC/data`
- `third_party/DDSP-SVC/exp`
- `third_party/DDSP-SVC/pretrain`
- `third_party/MSST-WebUI/pretrain`
- 缓存、输出结果、构建产物

说明：

- `third_party/DDSP-SVC/data` 和 `third_party/DDSP-SVC/exp` 当前用于本地实验，通常是软链接
- `third_party/MSST-WebUI/configs` 与 `third_party/MSST-WebUI/config_unofficial` 已放开，可直接纳入版本控制

## 当前整合目标

当前仓库的目标不是保持各上游仓库“纯净镜像”，而是把一套可运行的本地工作流稳定保存下来，包括：

1. 从 `ncm` 到 `mp3/flac`
2. 用 `MSST-WebUI` 执行三步 pipeline 分离与去混响
3. 用 `Slice_wav_for_DDSP_SVC.py` 切片训练数据
4. 用 `DDSP-SVC` 做预处理、训练和推理

如果后续要同步上游更新，建议以 `third_party/` 目录为单位谨慎合并，并优先保留当前工作流所需的本地改动。
