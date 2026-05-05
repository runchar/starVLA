# VSI-590K 接入 StarVLA VL Co-Training 调研实践记录

日期：2026-05-05  
仓库：`/media/brl4090/data/Tamakoko/Mtimes/retrieval/docs/starVLA`

## 目标

调研 `nyu-visionx/VSI-590K` 是否适合接入 StarVLA 现有 VL co-training 流程。调研必须包含实际数据拉取、格式验证和最小实验，而不是只做理论判断。

## 当前结论快照

- StarVLA 已有 VL co-training 入口：`starVLA/training/train_starvla_cotrain.py`。
- 训练时会分别构建 `datasets.vla_data` 和 `datasets.vlm_data` 两个 dataloader，然后同一步内分别反传 `action_loss` 和 `vlm_loss`。
- VLM 数据入口是 `starVLA/dataloader/vlm_datasets.py`，注册表是 `starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py`。
- 当前 VLM loader 识别媒体字段为 `images` / `videos`，不是 VSI JSONL 实际使用的 `image` / `video` 单数字段；原始 VSI JSONL 需要做字段转换或扩展 loader。
- VSI 视频样本的 prompt 仍使用 `<image>` token 描述“frames of a video”；如果转成 StarVLA 的 `videos` 字段，还需要同步把视觉标记改成 `<video>`，否则 `vlm_datasets.py` 不会把视频 token 替换进 Qwen 输入。
- 本地可用训练环境是 conda env `dfs`，包含 `torch`、`transformers`、`decord`、`omegaconf`、`accelerate`、`huggingface_hub`，但不含 `datasets`。
- HuggingFace 元数据可访问；若后续下载失败，按用户要求切换 `HF_ENDPOINT=https://hf-mirror.com`。

## 数据集远端信息

来源：https://huggingface.co/datasets/nyu-visionx/VSI-590K

已通过 `huggingface_hub.HfApi().repo_info(..., files_metadata=True)` 确认文件：

| 文件 | 大小 |
| --- | ---: |
| `vsi_590k.jsonl` | 276,557,539 bytes |
| `robotics.tar.gz` | 1,104,873,829 bytes |
| `s3dis.tar.gz` | 2,865,916,732 bytes |
| `procthor.tar.gz` | 6,523,652,708 bytes |
| `hypersim.tar.gz` | 6,775,393,985 bytes |
| `adt.tar.gz` | 23,895,961,889 bytes |
| `scannet.tar.gz` | 27,632,592,849 bytes |
| `scannetppv2.tar.gz` | 21,438,045,428 bytes |
| `arkitscenes.tar.gz` | 90,154,741,781 bytes |
| `ytb_roomtour.tar.gz` | 55,261,730,714 bytes |

本机 `playground` 所在盘剩余约 113GB，不适合一次性全量拉取 236GB 数据集；本次先做 JSONL + 小源归档的实证验证。

## StarVLA 入口核对

### Co-training 训练脚本

`train_starvla_cotrain.py` 的关键路径：

1. `prepare_data()` 分别构建 VLA 和 VLM dataloader。
2. `_get_next_batch()` 每一步各取一个 `batch_vla` 和 `batch_vlm`。
3. `_train_step()` 先对 VLA 分支算 `action_loss`，再直接调用 `unwrapped.qwen_vl_interface(**batch_vlm)` 计算 VLM loss。

这说明 VSI-590K 若能被 VLM dataloader 正确产出 `input_ids`、`labels`、视觉 tensor 和 grid 信息，就能进入现有 co-training 反传路径。

### VLM loader 约束

`vlm_datasets.py` 的关键约束：

- 注册入口：`qwen_data_config.py::data_dict`。
- annotation 支持 `.jsonl` 或普通 JSON。
- 图片样本要求键名 `images`，视频样本要求键名 `videos`。
- 图片问题里的视觉标记应是 `<image>`；视频问题里的视觉标记应是 `<video>`。
- 视频读取依赖 `decord.VideoReader`。

## 实验日志

### 2026-05-05 16:14

- 已读取仓库要求的 Superpowers 安装说明。
- 本机已存在 `~/.agents/skills/superpowers -> ~/.codex/superpowers/skills`，不需要重新安装。
- 已确认 `dfs` 是可用 starVLA 环境。
- 已确认 HuggingFace 元数据直连可访问。

### 2026-05-05 16:19

已下载 `vsi_590k.jsonl` 到：

```bash
playground/Datasets/VSI-590K/raw/vsi_590k.jsonl
```

完整扫描结果：

- 总行数：590,667。
- 字段：所有样本都有 `conversations` 和 `question_type`；374,148 条有 `video`；216,519 条有 `image`；20,092 条有 `id`。
- 没有样本使用 StarVLA 当前 loader 直接识别的 `images` / `videos` 字段。
- 数据源前缀分布：
  - `hypersim`: 176,774
  - `scannetppv2`: 138,701
  - `scannet`: 92,145
  - `adt`: 60,207
  - `arkitscenes`: 57,816
  - `ytb_roomtour`: 20,100
  - `procthor`: 20,092
  - `robotics`: 19,645
  - `s3dis`: 5,187
- 问题类型前五：
  - `relative_direction_object`: 176,991
  - `relative_distance_object`: 119,263
  - `absolute_distance_object`: 59,154
  - `relative_size_object`: 52,158
  - `absolute_direction_object`: 43,248

代表样本：

```json
{"video": "scannet/scene0191_00.mp4", "question_type": "relative_direction_object"}
{"image": "robotics/agibot_685462_head_right_fisheye_color_0010.jpg", "question_type": "relative_direction_camera"}
```

实测风险：

1. 原始 JSONL 直接注册到 `qwen_data_config.py` 后，StarVLA VLM loader 不会进入图片/视频读取分支。
2. 视频样本即使把 `video` 改成 `videos`，还需要把首轮 human prompt 中的 `<image>` 改成 `<video>`。
3. 图片样本只需要把 `image` 改成 `images`，保留 `<image>`。

### 2026-05-05 16:29

直连下载 `robotics.tar.gz` 时停在约 192MB 后不再增长，随后按要求切换镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

镜像断点续传成功，已下载：

```bash
playground/Datasets/VSI-590K/raw/robotics.tar.gz
playground/Datasets/VSI-590K/raw/s3dis.tar.gz
```

只抽取 smoke 所需媒体：

```bash
playground/Datasets/VSI-590K/robotics/agibot_685462_head_right_fisheye_color_0010.jpg
playground/Datasets/VSI-590K/robotics/agibot_ea949482_videos_head_right_fisheye_color_0a1dbf_0100.jpg
playground/Datasets/VSI-590K/s3dis/area_1/Area_1_conferenceRoom_1.mp4
```

生成两份 4 条样本的小标注：

```bash
playground/Datasets/VSI-590K/annotations/vsi590k_raw_smoke.jsonl
playground/Datasets/VSI-590K/annotations/vsi590k_starvla_smoke.jsonl
```

已在 `starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py` 注册：

- `vsi590k_raw_smoke`
- `vsi590k_starvla_smoke`

并新增最小配置：

```bash
examples/CoTrainVLM/train_files/starvla_cotrain_vsi590k_smoke.yaml
```

### 2026-05-05 16:35

VLM loader 对照实验，环境固定：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
```

原始格式 `vsi590k_raw_smoke` 实测失败：

```text
ERROR TypeError 'NoneType' object is not subscriptable
```

原因是样本只有 `image` / `video` 单数字段，loader 没有进入媒体读取分支，但 prompt 中仍含 `<image>`，导致 `preprocess_qwen_2_visual(..., grid_thw=None)` 尝试索引空 grid。

转换格式 `vsi590k_starvla_smoke` 实测成功，首个 batch 同时包含图片和视频：

```text
input_ids           (2, 260) torch.int64
labels              (2, 260) torch.int64
attention_mask      (2, 260) torch.bool
pixel_values        (720, 1536) torch.float32
image_grid_thw      (1, 3) torch.int64
pixel_values_videos (648, 1536) torch.float32
video_grid_thw      (1, 3) torch.int64
position_ids        (3, 2, 260) torch.int64
```

这说明 VSI-590K 经过轻量字段转换后，可以进入 StarVLA 当前 VLM dataloader，并产出 co-training 所需的 `qwen_vl_interface(**batch_vlm)` 输入结构。

### 2026-05-05 16:36

新增可复用转换脚本：

```bash
examples/CoTrainVLM/train_files/prepare_vsi590k_for_starvla.py
```

脚本验证命令：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
python examples/CoTrainVLM/train_files/prepare_vsi590k_for_starvla.py \
  --input-jsonl playground/Datasets/VSI-590K/annotations/vsi590k_raw_smoke.jsonl \
  --output-jsonl playground/Datasets/VSI-590K/annotations/vsi590k_starvla_smoke_from_script.jsonl \
  --media-root playground/Datasets/VSI-590K \
  --require-media-exists
```

输出统计：

```json
{
  "image": 2,
  "read": 4,
  "video": 2,
  "written": 4,
  "written_image": 2,
  "written_video": 2
}
```

全量转换建议命令形状：

```bash
python examples/CoTrainVLM/train_files/prepare_vsi590k_for_starvla.py \
  --input-jsonl playground/Datasets/VSI-590K/raw/vsi_590k.jsonl \
  --output-jsonl playground/Datasets/VSI-590K/annotations/vsi590k_starvla_full.jsonl \
  --media-root playground/Datasets/VSI-590K \
  --require-media-exists
```

如果只想先混入 robotics 图片数据：

```bash
python examples/CoTrainVLM/train_files/prepare_vsi590k_for_starvla.py \
  --input-jsonl playground/Datasets/VSI-590K/raw/vsi_590k.jsonl \
  --output-jsonl playground/Datasets/VSI-590K/annotations/vsi590k_robotics_starvla.jsonl \
  --media-root playground/Datasets/VSI-590K \
  --prefix robotics \
  --require-media-exists
```

### 2026-05-05 16:37

Qwen3-VL VLM forward no-grad 验证通过。命令使用转换后的首个 batch，里面同时有 1 条图片样本和 1 条视频样本。

关键输出：

```text
batch_shapes {'input_ids': (2, 260), 'labels': (2, 260), 'attention_mask': (2, 260), 'pixel_values': (720, 1536), 'image_grid_thw': (1, 3), 'pixel_values_videos': (648, 1536), 'video_grid_thw': (1, 3), 'position_ids': (3, 2, 260)}
loss 2.8829784393310547
logits_shape (2, 260, 151936)
seconds 0.523
```

这个结果证明：转换后的 VSI-590K batch 不只是能被 dataloader collate，也能被当前 `Qwen3-VL-2B-Instruct` 的 `qwen_vl_interface(**batch)` 前向吃进去并计算 VLM loss。

收尾复验也通过：

```text
py_compile_ok
batch_shapes {'input_ids': (2, 260), 'labels': (2, 260), 'attention_mask': (2, 260), 'pixel_values': (720, 1536), 'image_grid_thw': (1, 3), 'pixel_values_videos': (648, 1536), 'video_grid_thw': (1, 3), 'position_ids': (3, 2, 260)}
forward_loss 4.006426
forward_logits_shape (2, 260, 151936)
```

注：`LazySupervisedDataset.__init__` 会 `random.shuffle(list_data_dict)`，所以不同复验的首个 batch 样本顺序可能不同，loss 数值不要求一致。

### 2026-05-05 16:38

尝试 1-step `train_starvla_cotrain.py`。当前 GPU 状态：

```text
RTX 4090 24GB，总显存 24564MiB，已有两个 .venv/bin/python 进程分别占用约 7964MiB 和 7888MiB。
```

第一次使用默认 ZeRO-2 bucket：

```bash
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
  starVLA/training/train_starvla_cotrain.py \
  --config_yaml examples/CoTrainVLM/train_files/starvla_cotrain_vsi590k_smoke.yaml \
  --run_id vsi590k_cotrain_smoke_20260505_1638 \
  --framework.qwenvl.attn_implementation sdpa \
  --trainer.freeze_modules qwen_vl_interface.model.model \
  --trainer.max_train_steps 1
```

失败点：

```text
torch.OutOfMemoryError: Tried to allocate 954.00 MiB
```

原因：DeepSpeed ZeRO-2 默认 `reduce_bucket_size=5e8`，在已有进程占用下无法再申请约 954MiB bucket。

第二次新增 smoke 专用 DeepSpeed 配置：

```bash
starVLA/config/deepseeds/ds_config_smoke.yaml
starVLA/config/deepseeds/deepspeed_zero2_smoke.yaml
```

核心变化：

```json
"allgather_bucket_size": 5e7,
"reduce_bucket_size": 5e7,
"overlap_comm": false
```

结果：越过 bucket 初始化和 action backward，但在 optimizer step 仍因当前显存余量不足失败：

```text
torch.OutOfMemoryError: Tried to allocate 592.00 MiB
```

这两个训练级失败都发生在 action 分支/优化器阶段，不是 VSI-590K loader 或 VLM batch 格式失败。当前工作站如果释放那两个各约 8GB 的 GPU 进程，或者进一步减小 action head / offload optimizer，1-step co-training 应该可以继续往后验证。

训练日志：

```bash
playground/Datasets/VSI-590K/experiments/vsi590k_cotrain_smoke_20260505_1638.log
playground/Datasets/VSI-590K/experiments/vsi590k_cotrain_smoke_smallbucket_20260505_1640.log
```

## 实验结论

1. **VSI-590K 不能原样接入 StarVLA VLM loader。** 原因不是语义格式，而是字段约定不一致：VSI 用 `image`/`video`，StarVLA 当前 loader 用 `images`/`videos`。
2. **视频样本必须把 prompt 的 `<image>` 改成 `<video>`。** 否则 StarVLA 会走视频媒体分支，但文本侧没有视频 token，占位替换不成立。
3. **轻量预转换后，VSI-590K 可以进入 StarVLA 的 VLM dataloader。** 已实测图片和视频混合 batch 能产出 token、label、视觉 tensor 和 grid。
4. **转换后的 batch 可以被 Qwen3-VL 前向计算 VLM loss。** no-grad forward 已返回有效 loss 和 logits。
5. **GPU 释放后，已跑通 1-step co-training optimizer step。** 同一步内产出了 `action_dit_loss` 和 `vlm_loss`，并保存了 `final_model`。

### 2026-05-05 19:21 GPU 释放后补充实验

GPU 释放后状态：

```text
NVIDIA GeForce RTX 4090, total 24564 MiB, used 12 MiB, free 24199 MiB
```

先重跑上一轮“冻结 `qwen_vl_interface.model.model`”的命令，结果不再 OOM，但暴露出一个真实配置问题：

```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

根因：`Qwen3VLForConditionalGeneration` 的 `lm_head.weight` 与 `language_model.embed_tokens.weight` 是 tied weight。冻结 `qwen_vl_interface.model.model` 会把 tied embedding 一起冻结，导致 VLM loss 没有任何可训练参数可反传。

验证结构：

```text
children of Qwen3VLModel:
visual Qwen3VLVisionModel 406957056
language_model Qwen3VLTextModel 1720574976
lm_head tied to embed? True
```

随后尝试不冻结 Qwen、打开 CPU optimizer offload，但当前环境无法编译 DeepSpeed CPUAdam：

```text
DeepSpeed Op Builder: Installed CUDA version 11.8 does not match the version torch was compiled with 12.4
```

因此实际跑通的 1-step 配置是：冻结视觉模块和 language layers，保留 tied `embed_tokens/lm_head` 与 action head 可训练。这样 VLM 分支有梯度，且单卡 4090 能完成 optimizer step。

命令：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
  starVLA/training/train_starvla_cotrain.py \
  --config_yaml examples/CoTrainVLM/train_files/starvla_cotrain_vsi590k_smoke.yaml \
  --run_id vsi590k_cotrain_lmhead_train_20260505_192119 \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.qwenvl.enable_gradient_checkpointing true \
  --trainer.freeze_modules qwen_vl_interface.model.model.visual,qwen_vl_interface.model.model.language_model.layers \
  --datasets.vlm_data.per_device_batch_size 1 \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.max_train_steps 1 \
  --trainer.eval_interval 9999 \
  --trainer.save_interval 9999 \
  --trainer.logging_frequency 1 \
  --wandb_project starVLA_Cotrain_VSI590K_smoke \
  --wandb_entity offline
```

结果：

```text
# Parameters (in millions): 2282.713 Total, 466.348 Trainable
Step 1, Loss: {
  'action_dit_loss': 1.2688288688659668,
  'vlm_loss': 0.15185903012752533,
  'data_time': 0.7801914310548455,
  'model_time': 0.7211939750704914,
  'learning_rate': 4e-07,
  'epoch': 0.0
}
Training complete. Final model saved at playground/Checkpoints/vsi590k_cotrain_lmhead_train_20260505_192119/final_model
```

产物：

```bash
playground/Datasets/VSI-590K/experiments/vsi590k_cotrain_lmhead_train_20260505_192119.log
playground/Checkpoints/vsi590k_cotrain_lmhead_train_20260505_192119/final_model/pytorch_model.pt
```

### 2026-05-05 19:30

训练目标口径修正：这里的“全量”指全模型参与训练，不是全量下载 VSI-590K 数据。当前训练脚本默认仍使用本机已经下载并验证过的 partial VSI 数据：

```bash
playground/Datasets/VSI-590K/annotations/vsi590k_starvla_smoke.jsonl
```

新增全模型 co-training 配置和启动脚本：

```bash
examples/CoTrainVLM/train_files/starvla_cotrain_vsi590k_partial_full_model.yaml
examples/CoTrainVLM/train_files/run_vsi590k_partial_full_model_cotrain.sh
```

脚本默认：

- `freeze_module_list=''`，即 Qwen-VL、action model 等全模型参数都保持可训练。
- 保留可切换的部分冻结行：

```bash
#freeze_module_list='qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual'
```

- 数据仍为当前已经落盘并实测通过的 partial 数据，正式 alias 为 `vsi590k_starvla_partial_downloaded`，不会自动切到 236GB 全量 VSI 数据。
- 默认 DeepSpeed 配置不启用 optimizer offload，避免本机出现过的 CUDA 11.8 与 torch CUDA 12.4 不匹配导致 `DeepSpeedCPUAdam` 编译失败。
- 启动日志会写入 `${run_root_dir}/${run_id}/train.log`，脚本本身也会复制到同一输出目录，方便之后复现。

同时修正 `starVLA/training/train_starvla_cotrain.py` 的 accelerator 创建时机：现在会在 YAML + CLI merge 之后读取 `trainer.gradient_accumulation_steps`，否则启动脚本中设置的 grad accumulation 在 co-training 入口里不会真实生效。

推荐本机/单卡 smoke：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
WANDB_MODE=offline \
num_processes=1 \
attn_implementation=sdpa \
max_train_steps=1 \
save_interval=9999 \
eval_interval=9999 \
logging_frequency=1 \
bash examples/CoTrainVLM/train_files/run_vsi590k_partial_full_model_cotrain.sh
```

大服务器多卡训练时重点改这些变量：

```bash
run_id=vsi590k_partial_full_model_server \
bash examples/CoTrainVLM/train_files/run_vsi590k_partial_full_model_cotrain.sh
```

`run_vsi590k_partial_full_model_cotrain.sh` 当前已经按单机 8 卡 A100 正式训练设置默认值：

```bash
num_processes=8
vla_batch_size=1
vlm_batch_size=1
grad_accum=8
max_train_steps=100000
save_interval=5000
eval_interval=1000
attn_implementation=flash_attention_2
```

本机单卡全模型 1-step 真启动结果：

```text
# Parameters (in millions): 2282.713 Total, 2282.713 Trainable
Gradient accumulation steps = 8
Total batch size = 8
```

训练已经进入 VLA forward 和 VLM backward，但 24GB 4090 在 VLM backward 的 ZeRO2 gradient bucket 处 OOM：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.25 GiB.
GPU 0 has a total capacity of 23.64 GiB of which 2.12 GiB is free.
```

结论：全模型路径和配置接线能启动；本机单卡不能完成全模型 optimizer step。大服务器上建议先用 `num_processes=8`、`vla_batch_size=1`、`vlm_batch_size=1`、`grad_accum=8`，如果仍 OOM，再启用脚本里保留的部分冻结行或切 ZeRO3。

### 2026-05-06 A100 正式默认配置

已将启动脚本默认值改成单机 8 卡 A100 正式训练口径：

- 默认 `num_processes=8`。
- 默认 `attn_implementation=flash_attention_2`。
- 默认 `run_id=vsi590k_partial_full_model_a100_时间戳`。
- 默认 `freeze_module_list=''`，保持全模型训练。
- 默认 VLM dataset key 改为 `vsi590k_starvla_partial_downloaded`，仍指向当前已下载并验证过的 partial JSONL，不扩大数据范围。

正式启动命令：

```bash
source /path/to/conda.sh
conda activate dfs
bash examples/CoTrainVLM/train_files/run_vsi590k_partial_full_model_cotrain.sh
```

如果调度器已经设置 GPU 可见性，不需要额外传 `num_processes`；如果只分配了部分 A100，需要显式覆盖：

```bash
num_processes=4 \
run_id=vsi590k_partial_full_model_a100_4gpu \
bash examples/CoTrainVLM/train_files/run_vsi590k_partial_full_model_cotrain.sh
```

注意：这不是“全量 VSI-590K 媒体下载”。当前 `playground` 所在盘只剩约 109GB，而 VSI-590K 压缩归档总量超过 230GB；完整下载/解压需要新的大容量目录或分盘布局。

## 推荐接入方式

短期建议采用“离线 JSONL 预转换 + registry 注册”的方式，而不是先改 loader：

- 改动小，和现有 `sharegpt4v_coco` 接入方式一致。
- 可以按 prefix 分批接入，先从 `robotics`、`s3dis` 这类小源做 sanity check，再考虑大源。
- 可以用 `--require-media-exists` 避免 JSONL 指向未下载媒体，适合分批下载场景。

正式训练前建议：

1. 先释放完整 24GB GPU 或换多卡环境。
2. 使用 smoke DeepSpeed 小 bucket 配置，或在正式多卡 ZeRO 下按实际通信效率调回更大 bucket。
3. 先混入 `robotics` image-only 或 `s3dis` video-only 子集，确认 `vlm_loss` 正常下降，再扩到全量 VSI-590K。
4. 若只做空间推理增强，优先抽 `relative_direction_*`、`relative_distance_*`、`absolute_distance_*`，这些问题类型占比最高，也最贴近 VLA 的空间泛化需求。
5. 单卡 4090 上建议先用 `qwen_vl_interface.model.model.visual,qwen_vl_interface.model.model.language_model.layers` 冻结方案做 VSI co-training sanity；若要全 Qwen 可训练，需要修复 CUDA 11.8 vs torch cu124 的 DeepSpeed CPUAdam 编译问题，或换匹配 CUDA toolkit / 多卡 ZeRO 环境。

## 已验证与仍待验证

已验证：

- `vsi_590k.jsonl` 实际使用单数 `image`/`video` 字段。
- 抽样媒体路径能映射到对应 tar 内文件。
- `robotics.tar.gz` 可以支持 image smoke，`s3dis.tar.gz` 可以支持 video smoke。
- 原始视频样本经 `video -> videos` 和 `<image> -> <video>` 转换后，`decord` 可以被 `vlm_datasets.py` 正常调用。
- 已完成至少一个 VLM dataloader batch smoke。
- 已完成 Qwen3-VL VLM forward no-grad smoke。
- GPU 释放后已完成 1-step co-training optimizer step。
- 已确认全模型训练脚本能进入全模型 trainable 路径，并且 co-training 入口的 `gradient_accumulation_steps=8` 已真实生效。

仍待验证：

- 多卡大服务器上完成全模型 1-step optimizer step；本机单卡 24GB 已在 VLM backward 处 OOM。
- 分 prefix 扩大样本后，检查长视频/坏视频比例和 dataloader retry 频率。
- 多卡正式训练时重新评估 DeepSpeed bucket，smoke 小 bucket 只是当前单卡低余量环境的保守配置。
