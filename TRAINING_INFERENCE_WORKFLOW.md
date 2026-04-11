# VocalTS 训练与推理流程

本文档描述当前项目中基于 `ncmdump`、`MSST-WebUI`、`DDSP-SVC` 和 `SO-VITS-SVC` 的流程。

为便于维护，流程拆分为三部分：

1. 通用预处理（`ncmdump` + `MSST-WebUI`）
2. `DDSP-SVC` 训练与推理（保持现有流程）
3. `SO-VITS-SVC` 训练与推理（预留章节）

## 总体说明

- 除了 `ncmdump` 是安装后的 `pip` 命令，其余步骤基本都依赖仓库内脚本。
- 因此执行命令时，需要特别注意当前所在目录。
- 本文中的 `{avatar}` 表示角色或歌手对应的数据目录名。
- 训练数据目录默认使用：
  `/home/pangjichen/workspace/VocalTS/linked_data/{avatar}`
- 推理数据目录默认使用：
  `/home/pangjichen/workspace/VocalTS/linked_data/inference_data`

## 通用预处理（ncmdump + MSST-WebUI + 切片）

这一部分是 `DDSP-SVC` 和 `SO-VITS-SVC` 共用的前置流程。

### 训练数据预处理

首先设置文件夹名称，例如：
```bash
avatar=CuSummer
```
创建系列文件夹：
```bash
python func_scripts/0_create_avatar.py ${avatar}
```
#### 1. 准备原始 `ncm` 文件

将训练用的 `ncm` 文件放到：

```text
/home/pangjichen/workspace/VocalTS/linked_data/avatars/${avatar}/ncm
```

建议一个角色单独一个目录，便于后续批量处理。

#### 2. 使用 `ncmdump` 批量转换为 `mp3`

`ncmdump` 不依赖当前仓库相对路径，可以直接在任意目录执行。

示例：

```bash
ncmdump -d ./linked_data/avatars/${avatar}/meta_data/ncm \
  -o ./linked_data/avatars/${avatar}/meta_data/audio
```

参考命令可见 [use.sh](/home/pangjichen/workspace/VocalTS/third_party/ncmdump/use.sh)。

#### 3. 进入 `MSST-WebUI`，对 `audio` 批量进行三步处理

切换到目录：

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI
```

使用 [msst_pipeline.py](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/msst_pipeline.py) 批量处理音频。当前 pipeline 包含三步：

1. 人声与伴奏初分离
2. 人声进一步细分
3. 去混响，输出 `noreverb` 和 `reverb`

示例：

```bash
python msst_pipeline.py \
  --input_dir ./../../linked_data/avatars/${avatar}/meta_data/audio \
  --output_dir ./../../linked_data/avatars/${avatar}/meta_data/MSST \
  --output_format flac \
  --dereverb_mode auto \
  --device_ids 0 1 \
  --jobs 8 
```

说明：

- `--dereverb_mode auto` 会根据输入音频声道数自动选择单声道或双声道去混响模型。
- step 3 的最终干声输出位于：
  `/home/pangjichen/workspace/VocalTS/linked_data/{avatar}/cleaned/step_3/noreverb`

#### 4. 回到 `VocalTS`，将 step 3 的 `noreverb` 进行切片

切换到目录：

```bash
cd /home/pangjichen/workspace/VocalTS
```

先不落盘测试：
```bash
python Slice_wav_for_DDSP_SVC.py \
  ./linked_data/avatars/${avatar}/meta_data/MSST/step_3/noreverb \
  ./linked_data/avatars/${avatar}/meta_data/vocal_sliced \
  --discard-short \
  --dry-run 

```

示例：

```bash
python Slice_wav_for_DDSP_SVC.py \
  ./linked_data/avatars/${avatar}/meta_data/MSST/step_3/noreverb \
  ./linked_data/avatars/${avatar}/meta_data/vocal_sliced \
  --discard-short
```

说明：

- 该脚本当前支持输入 `wav` 或 `flac`。
- 输出统一为切片后的 `wav`。

### 推理数据预处理

推理流程的前处理整体类似训练，只是不需要切片与训练。

先定义批次编号：
```bash
data_index=0001
```

#### 1. 准备推理用 `ncm` 文件

将 `ncm` 文件放到：

```text
/home/pangjichen/workspace/VocalTS/linked_data/inference_data
```

建议按批次或编号建立子目录，例如：

```text
/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}
```

#### 2. 使用 `ncmdump` 转换为 `mp3`

示例：

```bash
ncmdump -d /home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index} \
  -o /home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}
```

#### 3. 进入 `MSST-WebUI`，进行三步提取

切换到目录：

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI
```

示例：

```bash
python msst_pipeline.py \
  --input_dir /home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index} \
  --output_dir /home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/cleaned \
  --output_format flac \
  --dereverb_mode auto
```

通常会得到：

- `step_3/noreverb`：更适合做主干声推理
- `step_2/Instrumental` 或其他保留部分：可根据任务用于和声或伴随部分推理

## DDSP-SVC 流程

### 训练流程

#### 1. 建立数据与实验目录链接

切换到目录：

```bash
cd /home/pangjichen/workspace/VocalTS/
```

将训练数据目录链接到 `./data`，将 checkpoint 目录链接到 `./exp`。


对已有人声，链接：
```bash
python func_scripts/1_switch_avatar.py ${avatar} "DDSP-SVC"
```

实际使用时，也可以按你的实验命名习惯创建更具体的软链接名称。
#### 2. 进入 `DDSP-SVC`，切分验证集
```bash
cd /home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC
```
```bash
python ./draw.py
```
#### 3. 运行预处理

在 `DDSP-SVC` 目录下执行：

```bash
python preprocess.py -c configs/reflow.yaml -j 4
```

#### 4. 开始训练

在 `DDSP-SVC` 目录下执行：

```bash
python train_reflow.py -c configs/reflow.yaml
```

参考命令可见 [use.sh](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/use.sh)。

### 推理流程

以下内容维持原有流程不变。

#### 1. 进入 `DDSP-SVC`，分别对干声部分和和声部分进行推理

切换到目录：

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC
```

一个推理示例：
```bash
music_name="Guiano,花譜 - 花 feat. 花譜"
```
```bash
python main_reflow.py \
  -i "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/cleaned/step_3/noreverb/${music_name}_noreverb.flac" \
  -m "exp/reflow-test/model_best_mel_val_mse_step28000.pt" \
  -o "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/transfered/${music_name}_DryVocal_28000.flac" \
  -k 0 \
  -id 1 \
  -method "auto" \
  -ts 0.0
```
```bash
python main_reflow.py \
  -i "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/cleaned/step_2/Instrumental/${music_name}_Instrumental.flac" \
  -m "exp/reflow-test/model_best_mel_val_mse_step28000.pt" \
  -o "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/transfered/${music_name}_Harmonic_28000.flac" \
  -k 0 \
  -id 1 \
  -method "auto" \
  -ts 0.0
```

实际推理时，可以分别对：

- 干声部分
- 和声部分

执行两次推理，再按需要混合最终结果。

## SO-VITS-SVC 流程

### 训练流程

切换到目录：

```bash
cd /home/pangjichen/workspace/VocalTS/
```

对已有人声，链接：
```bash
python func_scripts/1_switch_avatar.py ${avatar} "SO-VITS-SVC"
```

切换到SOVITS文件夹
```bash
cd /home/pangjichen/workspace/VocalTS/third_party/SO-VITS-SVC
```
切分验证集并生成配置文件
```bash
python preprocess_flist_config.py --speech_encoder vec768l12
```
预处理生成特征
```bash
python preprocess_hubert_f0.py --f0_predictor rmvpe --use_diff --num_processes 8
```
#### 正式训练
主模型训练
```bash
python train.py -c configs/config.json -m 44k
```
如果需要浅扩散模型(高质量数据集适用)
```bash
python train_diff.py -c configs/diffusion.yaml
```

### 推理流程

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/SO-VITS-SVC
data_index=0019
music_name="蔡依林 - 日不落"
```

```bash
# CuSummer
python inference_main.py \
  -m "logs/44k/G_92000.pth" \
  -c "configs/config.json" \
  -n "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/cleaned/step_3/noreverb/${music_name}_noreverb.flac" \
  -t 0 \
  --enhance \
  -o "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/transfered_SOVITS_CuSummer/${music_name}_DryVocal_92000.flac"
```
```bash
python inference_main.py \
  -m "logs/44k/G_92000.pth" \
  -c "configs/config.json" \
  -n "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/cleaned/step_2/Instrumental/${music_name}_Instrumental.flac" \
  -t 0 \
  --enhance \
  -o "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/transfered_SOVITS_CuSummer/${music_name}_Harmonic_92000.flac"
```
---
```bash
# ikura
python inference_main.py \
  -m "logs/44k/G_best_146400_34.197796.pth" \
  -c "configs/config.json" \
  -n "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/cleaned/step_3/noreverb/${music_name}_noreverb.flac" \
  -t 0 \
  --enhance \
  -o "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/transfered_SOVITS_ikura/${music_name}_DryVocal_146400.flac"
```
```bash
python inference_main.py \
  -m "logs/44k/G_best_146400_34.197796.pth" \
  -c "configs/config.json" \
  -n "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/cleaned/step_2/Instrumental/${music_name}_Instrumental.flac" \
  -t 0 \
  --enhance \
  -o "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/${data_index}/transfered_SOVITS_ikura/${music_name}_Harmonic_146400.flac"
```

## 目录切换注意事项

- `ncmdump` 是外部命令，通常不受当前工作目录影响。
- `msst_pipeline.py` 需要在 `MSST-WebUI` 目录下运行，因为它依赖项目内相对路径和配置。
- `Slice_wav_for_DDSP_SVC.py` 建议在 `VocalTS` 根目录下运行，便于管理输入输出路径。
- `preprocess.py`、`train_reflow.py`、`main_reflow.py` 需要在 `DDSP-SVC` 目录下运行，因为它们依赖 `configs`、`data`、`exp` 等相对路径。

## 常用目录汇总

```text
VocalTS 根目录
/home/pangjichen/workspace/VocalTS

MSST-WebUI
/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI

DDSP-SVC
/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC

训练数据根目录
/home/pangjichen/workspace/VocalTS/linked_data/{avatar}

推理数据根目录
/home/pangjichen/workspace/VocalTS/linked_data/inference_data
```

## 当前三步 Pipeline 模型简介

当前 [msst_pipeline.py](/home/pangjichen/workspace/VocalTS/third_party/MSST-WebUI/msst_pipeline.py) 中的 pipeline 固定包含三步：

### Step 1: 初步人声分离

- 模型：`big_beta5e.ckpt`
- 输出：`vocals` 和 `other`
- 作用：先从完整混音中做第一轮分离，把后续需要进一步处理的人声部分提取出来

### Step 2: 人声进一步细分

- 模型：`mel_band_roformer_karaoke_becruily.ckpt`
- 输入：step 1 的 `vocals`
- 输出：`Vocals` 和 `Instrumental`
- 作用：对第一步提取出的人声结果继续细分，得到更干净的主人声与伴随部分，便于后续推理时分别处理

### Step 3: 去混响

- 单声道模型：`dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt`
- 双声道模型：`dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt`
- 输入：step 2 的 `Vocals`
- 最终保存输出：`noreverb` 和 `reverb`

说明：

- 当 `--dereverb_mode mono` 时，直接使用单声道去混响模型，输出 `noreverb/reverb`
- 当 `--dereverb_mode stereo` 时，使用双声道去混响模型，模型原始输出是 `dry/other`，在 pipeline 中会分别保存为 `noreverb/reverb`
- 当 `--dereverb_mode auto` 时，会根据输入音频声道数自动选择单声道或双声道模型

整体来说，这三步的设计思路是：

1. 先把人声从混音里分出来
2. 再把人声进一步整理成更适合使用的部分
3. 最后对目标人声做去混响，得到更适合 DDSP-SVC 训练和推理的干声

## Git 与目录维护说明

当前工程建议以 [VocalTS](/home/pangjichen/workspace/VocalTS) 作为顶层主仓库统一维护，而不是继续把 `DDSP-SVC`、`MSST-WebUI`、`ncmdump` 当作彼此独立的子仓库来管理。

建议：

- 在顶层仓库统一提交文档、脚本和对子项目源码的本地修改
- `MSST-WebUI/configs` 与 `MSST-WebUI/config_unofficial` 已改为可直接纳入版本控制
- `linked_data`、模型权重、缓存、构建产物等大文件或本地输出保持忽略，不直接提交

另外需要特别注意 [DDSP-SVC](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC) 下的两个软链接：

- [data](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/data)
- [exp](/home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC/exp)

这两个路径当前是软链接，而不是普通目录。维护时建议：

- 保留软链接结构用于本地训练和推理
- 不要把它们替换成真实目录
- 在顶层仓库中忽略这两个路径，不提交实验数据和 checkpoint
- 变更数据集目标或 checkpoint 目录时，使用重新创建软链接的方式更新

示例：

```bash
cd /home/pangjichen/workspace/VocalTS/third_party/DDSP-SVC
rm data exp
ln -s /home/pangjichen/workspace/VocalTS/linked_data/{avatar}/DDSP-SVC data
ln -s /home/pangjichen/workspace/VocalTS/linked_data/{avatar}/DDSP-SVC-ckpt exp
```
