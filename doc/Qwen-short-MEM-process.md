# QwenPIShortMEM 实现与测试过程记录

## 信息来源

本文档根据以下两部分恢复和整理：

- 历史会话记录：`rollout-2026-05-01T00-25-02-019ddf35-4020-7281-9b95-56c638265c11.jsonl`
- 当前工作区未提交改动

上一次会话在准备写文档时中断。当时代码实现、单元测试和两次训练 smoke 已完成，最后停在 `ruff` 静态检查失败之后。

## 需求确认

用户最终锁定的需求是：

- 在 QwenPI 路线中实现 `QwenPIShortMEM`。
- dataloader 给 VLM 提供每个 view 的 K 帧历史图像。
- 在 Qwen3-VL visual encoder 内部扩展为 ShortMEM video encoder。
- 复用原 spatial ViT blocks。
- 每隔 4 层对相同 spatial patch index 的跨 timestep token 做 causal temporal attention。
- 上层丢弃历史 timestep tokens，只保留 current timestep visual tokens 给 Qwen language backbone 和原 action head。
- 不做 pseudo-image compression。
- 不把 K 帧直接当成多张普通 image 导致 token 数变成 K 倍。
- 使用 `conda activate dfs` 环境测试训练能跑通；显存不够时允许冻结部分参数。
- 将修改、测试和 debug 全流程写成中文 Markdown 文档。

## 方案形成过程

最初评估过直接改 PI framework 和改 VLM 内部两种方向。最终选择“新增 VLM wrapper + 本地模型目录名触发分支”的方案：

- 新建 `starVLA/model/modules/vlm/QWen3ShortMem.py`。
- 在 `get_vlm_model()` 中优先匹配 `"Qwen3-VL-ShortMem"`。
- wrapper 内部仍加载原 Qwen3-VL 权重。
- 将 `model.model.visual` 替换成 ShortMEM visual wrapper。
- 在 `playground/Pretrained_models/` 下创建 `Qwen3-VL-ShortMem-2B-Instruct`，通过目录名在配置中切换普通 Qwen3 和 ShortMEM Qwen3。

这个方案的好处是 StarVLA 外层基本不需要知道 visual encoder 内部发生了变化。QwenPI 只多传一个 `image_history` 旁路字段，action head 不改。

## 实现步骤

### 1. 先写小测试

新增测试文件：`tests/test_qwen3_shortmem.py`

覆盖点：

- `Qwen3-VL-ShortMem` 路由优先于普通 `Qwen3-VL`。
- `pack_shortmem_history()` 能将 current image placeholder 和 history frame 旁路输入分离。
- `select_current_timestep_tokens()` 只保留 current timestep tokens。
- temporal causal mask 会屏蔽未来 timestep。
- fake visual wrapper 路径能输出 current tokens 和 deepstack current tokens。

最初 RED 结果确认：

- ShortMEM 分支会被普通 Qwen3 分支吞掉。
- 新的 ShortMEM 模块尚不存在。

### 2. 新增 ShortMEM 模块

新增：

- `starVLA/model/modules/vlm/QWen3ShortMem.py`
- `starVLA/model/modules/vlm/shortmem_qwen3_visual.py`

关键 debug 点：

- Qwen3-VL 的 visual 模块真实挂载点是 `model.model.visual`，不是顶层 `model.visual`。
- history frames 不能进入 `apply_chat_template()` 的普通 image 列表，否则 language backbone 看到的 image tokens 会变成 K 倍。
- 正确做法是 current frame 继续走普通 Qwen processor，history frames 只通过 `processor.image_processor()` 生成 `shortmem_pixel_values` 和 `shortmem_grid_thw`。

### 3. 修改 QwenPI 与 dataloader

修改 `QwenPI.forward()` 和 `QwenPI.predict_action()`，将 batch 中的 `image_history` 传给 VLM wrapper。

修改 dataloader：

- 根据 `datasets.vla_data.image_history_frames` 设置 video `delta_indices`。
- `sample["image"]` 保持 current frame。
- `sample["image_history"]` 输出每个 view 的 K 帧历史。

额外修正：

- QwenPI 原先固定把 action DiT 层数设为 36。为了 4090 上能做最小训练 smoke，改为优先使用 `framework.action_model.diffusion_model_cfg.num_layers`，默认仍为 36。

### 4. 增加 framework alias

新增 `starVLA/model/framework/VLM4A/QwenPIShortMEM.py`，注册 `QwenPIShortMEM`，继承现有 `Qwen_PI`。

该 alias 主要用于配置和实验命名；核心行为仍由 VLM wrapper 和 `base_vlm` 名称触发。

### 5. 建立本地模型别名

创建了本地软链接：

```text
playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct -> Qwen3-VL-2B-Instruct
```

这样可以在 YAML 中通过模型目录名切换 wrapper，不需要复制 2B 权重。

## 验证记录

### 单元测试

命令：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
python -m pytest tests/test_qwen3_shortmem.py -q
```

结果：

```text
5 passed
```

### 静态编译

对本次新增和触及的 Python 文件执行过 `py_compile`，结果通过。覆盖范围包括：

- `starVLA/model/modules/vlm/QWen3ShortMem.py`
- `starVLA/model/modules/vlm/shortmem_qwen3_visual.py`
- `starVLA/model/framework/VLM4A/QwenPIShortMEM.py`
- `starVLA/model/framework/VLM4A/QwenPI.py`
- dataloader 相关文件
- `tests/test_qwen3_shortmem.py`

### processor 旁路检查

使用本地 Qwen3 processor 验证 history frame 可以直接通过 `image_processor` 生成旁路张量。

记录到的关键 shape：

```text
current pixel_values: (512, 1536)
history shortmem_pixel_values: (1536, 1536)
shortmem_history_frames: 3
```

这说明 current placeholder 和 history 旁路张量已经分离，history 没有让 language backbone 的 placeholder token 数膨胀为 K 倍。

### 训练 smoke 1：冻结整个 VLM

目的：先证明训练闭环能跑通，避免 4090 显存被 2B VLM + ShortMEM history 卡住。

命令：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_cotrain_libero.yaml \
  --framework.name QwenPIShortMEM \
  --framework.qwenvl.base_vlm ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.qwenvl.enable_gradient_checkpointing true \
  --framework.qwenvl.shortmem.history_frames 2 \
  --framework.qwenvl.shortmem.temporal_interval 4 \
  --framework.qwenvl.shortmem.prune_after_layer 4 \
  --framework.action_model.diffusion_model_cfg.num_layers 2 \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.image_history_frames 2 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps 1 \
  --trainer.num_warmup_steps 0 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency 1 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir playground/Checkpoints \
  --run_id smoke_qwenpi_shortmem_freeze_vlm
```

结果：

- 1 step 训练完成。
- 产物存在：`playground/Checkpoints/smoke_qwenpi_shortmem_freeze_vlm/final_model/pytorch_model.pt`
- 文件大小约 5.7G。

### 训练 smoke 2：训练 ShortMEM temporal attention + action head

目的：更贴近目标训练方式，确认新增 temporal attention 参数能进入训练参数组。

冻结项：

```text
qwen_vl_interface.model.model.language_model,
qwen_vl_interface.model.lm_head,
qwen_vl_interface.model.model.visual.base_visual
```

命令：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_cotrain_libero.yaml \
  --framework.name QwenPIShortMEM \
  --framework.qwenvl.base_vlm ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.qwenvl.enable_gradient_checkpointing true \
  --framework.qwenvl.shortmem.history_frames 2 \
  --framework.qwenvl.shortmem.temporal_interval 4 \
  --framework.qwenvl.shortmem.prune_after_layer 4 \
  --framework.action_model.diffusion_model_cfg.num_layers 2 \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.image_history_frames 2 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual \
  --trainer.max_train_steps 1 \
  --trainer.num_warmup_steps 0 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency 1 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir playground/Checkpoints \
  --run_id smoke_qwenpi_shortmem_train_temporal_attn
```

结果：

- 1 step 训练完成。
- 产物存在：`playground/Checkpoints/smoke_qwenpi_shortmem_train_temporal_attn/final_model/pytorch_model.pt`
- 文件大小约 5.7G。
- 该 smoke 验证了 ShortMEM temporal attention 参数没有被误冻结，能参与训练。

### 训练 smoke 3：中断恢复后的 fresh retest

2026-05-01 14:30 重新跑了一次 1 step 训练 smoke，避免只引用上一次会话的历史结果。

命令与 smoke 2 等价，但输出目录改到 `/tmp`，避免继续在仓库工作区生成 5.7G checkpoint：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_cotrain_libero.yaml \
  --framework.name QwenPIShortMEM \
  --framework.qwenvl.base_vlm ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.qwenvl.enable_gradient_checkpointing true \
  --framework.qwenvl.shortmem.history_frames 2 \
  --framework.qwenvl.shortmem.temporal_interval 4 \
  --framework.qwenvl.shortmem.prune_after_layer 4 \
  --framework.action_model.diffusion_model_cfg.num_layers 2 \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.image_history_frames 2 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual \
  --trainer.max_train_steps 1 \
  --trainer.num_warmup_steps 0 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency 1 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir /tmp/starvla_shortmem_smoke \
  --run_id retest_qwenpi_shortmem_train_temporal_attn
```

结果：

- 训练进度达到 `1/1`。
- 日志记录 `Step 1, Loss: {'action_dit_loss': 479.28790283203125, ...}`。
- 参数统计为 `2304.413M Total, 176.881M Trainable`。
- `qwen_vl_interface` 参数组数量为 36，`action_model` 参数组数量为 52，说明不是冻结整个 VLM 的最弱 smoke，而是保留了新增 ShortMEM temporal attention 的训练参数。
- 产物存在：`/tmp/starvla_shortmem_smoke/retest_qwenpi_shortmem_train_temporal_attn/final_model/pytorch_model.pt`，大小约 5.7G。

## Debug 记录

### CUDA 可见性

历史会话中默认执行环境曾看不到 CUDA，但在允许的命令环境中可以看到 GPU。因此训练 smoke 使用显式环境：

```bash
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 训练配置选择

`examples/LIBERO` 的某些默认配置走 GR00T 简化 action head，不适合直接验证 QwenPIShortMEM。因此 smoke 最终使用：

```text
starVLA/config/training/starvla_cotrain_libero.yaml
```

并通过命令行覆盖 framework、模型、数据路径、batch size、history frames 和训练步数。

### 显存控制

为了在 4090 上跑通：

- smoke 使用 `per_device_batch_size=1`。
- history frames 降到 2。
- action DiT 层数降到 2。
- 启用 gradient checkpointing。
- 使用 Deepspeed zero2。
- 第一轮冻结整个 VLM。
- 第二轮冻结原 Qwen language model、lm head 和原 visual base，只训练新增 temporal attention 与 action head。

### ruff 检查

`dfs` 环境最初没有 `ruff`：

```text
No module named ruff
```

随后安装：

```bash
python -m pip install "ruff>=0.2.2"
```

安装后执行：

```bash
python -m ruff check \
  starVLA/model/modules/vlm/QWen3ShortMem.py \
  starVLA/model/modules/vlm/shortmem_qwen3_visual.py \
  starVLA/model/modules/vlm/__init__.py \
  starVLA/model/framework/VLM4A/QwenPI.py \
  starVLA/model/framework/VLM4A/QwenPIShortMEM.py \
  starVLA/dataloader/lerobot_datasets.py \
  starVLA/dataloader/gr00t_lerobot/datasets.py \
  tests/test_qwen3_shortmem.py
```

结果失败，主要包括两类：

- 仓库既有文件中的 import 顺序、行宽、空白、中文标点、隐式 Optional 等风格问题。
- 本次新增文件中少量 lint，例如 `zip()` 未显式设置 `strict=`。

本次没有大范围运行 `ruff --fix`，因为会改动大量历史代码，容易引入和 ShortMEM 无关的 diff。

## 当前工作区状态

当前与 ShortMEM 相关的未提交文件包括：

```text
M  starVLA/dataloader/gr00t_lerobot/datasets.py
M  starVLA/dataloader/lerobot_datasets.py
M  starVLA/model/framework/VLM4A/QwenPI.py
M  starVLA/model/modules/vlm/__init__.py
?? starVLA/model/framework/VLM4A/QwenPIShortMEM.py
?? starVLA/model/modules/vlm/QWen3ShortMem.py
?? starVLA/model/modules/vlm/shortmem_qwen3_visual.py
?? tests/test_qwen3_shortmem.py
```

另外存在历史会话文件和 `.codex`：

```text
?? rollout-2026-05-01T00-25-02-019ddf35-4020-7281-9b95-56c638265c11.jsonl
?? .codex
```

它们不是 ShortMEM 实现代码。

## 后续建议

- 若要提交代码，建议先只修本次新增文件中的 lint，不要直接格式化整个仓库。
- 对 `QWen3ShortMem.py` 中的 `zip()` 增加 `strict=True` 可消除新增文件的 B905。
- 后续可以补一个真实 `QwenPI.forward()` 小 batch 的自动化 smoke，但需考虑 2B 权重加载成本。
- 长训练前建议确认 `freeze_modules` 是否符合目标：全量训练 ShortMEM temporal attention、只训练 action head，或两者都训练。
- 如果切换到 Qwen3-VL-4B，建议先冻结 `language_model`、`lm_head` 和 `visual.base_visual`，并保持 `history_frames=2`、batch size 1 做 1 step smoke，再逐步放开参数。
