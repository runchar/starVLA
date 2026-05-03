# QwenPI / ShortMEM 训练 Loss 不明显下降的风险点记录

## 背景

当前观察到：使用 `examples/LIBERO/train_files/run_short_mem.sh` 训练约 7500 step 后，`action_dit_loss` 没有明显下降。

这不一定说明模型完全没有学习。PI / flow-matching 类 action loss 本身会受 batch 内容、噪声 timestep、动作尺度影响，单点日志不一定单调。但当前脚本和训练配置里确实存在几个会显著影响收敛判断和复现实验可比性的风险点。

## 风险点 1：实际训练量可能远小于主 recipe

当前脚本类似：

```bash
--num_processes 1
--datasets.vla_data.per_device_batch_size 2
--trainer.gradient_accumulation_steps 8
```

如果梯度积累真的生效，则 global batch 为：

```text
1 GPU * batch 2 * grad_acc 8 = 16
```

LIBERO 主 recipe 的常见量级是：

```text
8 GPUs * batch 16 * grad_acc 1 = 128
```

因此 7500 optimizer steps 的等效样本量是：

```text
7500 * 16 = 120000 samples
```

折算到 global batch 128 的主 recipe：

```text
120000 / 128 = 937.5 steps
```

也就是说，从样本量角度看，7500 step 只相当于主 recipe 约 938 step。这个阶段 raw loss 不明显下降并不意外。

如果那台服务器上的 `trainer.gradient_accumulation_steps` 没有真正进入 Accelerate/DeepSpeed，实际 global batch 可能只有：

```text
1 GPU * batch 2 * grad_acc 1 = 2
```

此时 7500 step 只相当于主 recipe：

```text
7500 * 2 / 128 = 117.2 steps
```

这个训练量基本不足以判断收敛。

### 检查方式

启动日志必须看到：

```text
Gradient accumulation steps = 8
Total batch size = 16
```

如果日志仍显示：

```text
Gradient accumulation steps = 1
```

说明命令行或 YAML 中的 `trainer.gradient_accumulation_steps` 没有真正生效。

## 风险点 2：gradient accumulation 可能是 dead config

旧版 `train_starvla.py` 中 `Accelerator()` 在读取 YAML 和 CLI 参数之前创建：

```python
deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
```

因此即使 YAML 写了：

```yaml
trainer:
  gradient_accumulation_steps: 8
```

或命令行传入：

```bash
--trainer.gradient_accumulation_steps 8
```

也可能无法影响已经创建好的 `accelerator`。

### 最小修复

将 `Accelerator()` 创建挪到配置合并之后：

```python
def create_accelerator(cfg) -> Accelerator:
    grad_accum = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=grad_accum)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        deepspeed_plugin=deepspeed_plugin,
    )
    accelerator.print(accelerator.state)
    return accelerator
```

然后在 `main(cfg)` 开头：

```python
def main(cfg) -> None:
    accelerator = create_accelerator(cfg)
```

修复后重新确认日志中的 `Gradient accumulation steps` 和 `Total batch size`。

## 风险点 3：当前脚本可能不是 ShortMEM，而是普通 Qwen3 + PI

ShortMEM wrapper 是通过 `base_vlm` 字符串触发的：

```python
if "Qwen3-VL-ShortMem" in vlm_name:
    return _QWen3ShortMem_VL_Interface(config)
elif "Qwen3-VL" in vlm_name:
    return _QWen3_VL_Interface(config)
```

如果脚本中使用：

```bash
model='./playground/Pretrained_models/Qwen3-VL-2B-Instruct'
```

则会走普通 Qwen3 wrapper，不会启用 ShortMEM。此时即使写了：

```bash
--framework.name QwenPIShortMEM
--framework.qwenvl.shortmem.history_frames 6
--datasets.vla_data.image_history_frames 6
```

也不会真的替换成 `ShortMemQwen3VisionModel`。

### 检查方式

训练日志或 config 中应出现：

```text
base_vlm: ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct
```

并且如果开启 gradient checkpointing，应看到：

```text
[QWen3ShortMem] gradient_checkpointing ENABLED
```

否则大概率不是 ShortMEM 路线。

## 风险点 4：全量训练 VLM 对单卡低 global batch 过于激进

如果脚本中：

```bash
freeze_module_list=''
```

则会训练整个 Qwen3-VL-2B + PI action head。

在单卡、低 global batch、短 warmup 下，全量 VLM 训练可能表现为：

- loss 很 noisy
- 下降慢
- 训练初期不稳定
- action head 学习被 VLM 大量参数更新干扰

如果目标是先确认 PI head 能学习，建议先冻结 VLM：

```bash
--trainer.freeze_modules qwen_vl_interface
```

如果是 ShortMEM 实验，建议先冻结原 Qwen language model、lm head 和原 visual base，只训练新增 temporal attention 与 PI action head：

```bash
--trainer.freeze_modules qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual
```

## 风险点 5：warmup 太短

当前脚本可能使用：

```bash
--trainer.num_warmup_steps 100
```

对于全量 VLM + PI 训练，这个 warmup 很短。原 YAML 默认是：

```yaml
num_warmup_steps: 5000
```

如果按 30000 step 训练，建议至少：

```bash
--trainer.num_warmup_steps 1500
```

更稳可以：

```bash
--trainer.num_warmup_steps 3000
```

## 风险点 6：raw action_dit_loss 不适合只看单点

PI / flow-matching loss 会随以下因素变化：

- 当前 batch 的 action 分布
- 随机采样的 diffusion / flow timestep
- 数据集任务难度
- 动作归一化统计

因此单个 logging step 的 `action_dit_loss` 不应当被当作严格单调指标。

建议看：

- 500 step 或 1000 step moving average
- 固定 batch overfit 曲线
- `mse_score` 或实际 rollout/eval 指标
- 同一配置下不同 seed 的趋势

## 推荐 sanity 实验

### 实验 A：冻结 VLM，只训练 PI action head

目的：验证数据、action target、PI head 和训练循环能不能学习。

建议配置：

```bash
--trainer.freeze_modules qwen_vl_interface
--datasets.vla_data.per_device_batch_size 2
--trainer.gradient_accumulation_steps 64
--trainer.max_train_steps 3000
--trainer.num_warmup_steps 300
```

预期：

- 如果该实验 loss 仍完全不下降，需要优先检查数据、action normalization、训练循环、scheduler、optimizer step 是否正确。
- 如果该实验能下降，而全量训练不下降，则主要问题是全量 VLM 训练配置太激进。

### 实验 B：ShortMEM 参数高效训练

目的：验证 ShortMEM temporal attention + PI action head 能不能学习。

建议配置：

```bash
model='./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct'
--framework.name QwenPIShortMEM
--framework.qwenvl.base_vlm ${model}
--framework.qwenvl.shortmem.history_frames 4
--datasets.vla_data.image_history_frames 4
--trainer.freeze_modules qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual
--datasets.vla_data.per_device_batch_size 2
--trainer.gradient_accumulation_steps 64
--trainer.max_train_steps 30000
--trainer.num_warmup_steps 1500
```

预期：

- 启动日志中应确认 `Gradient accumulation steps = 64`、`Total batch size = 128`。
- 日志中应出现 `[QWen3ShortMem] gradient_checkpointing ENABLED`。
- 可训练参数应包含 `qwen_vl_interface.model.model.visual.temporal_attn.*` 和 `action_model.*`。

## 推荐正式训练量级

如果想尽量对齐 LIBERO 主 recipe：

```text
global batch: 128
max_train_steps: 30000 起步
warmup: 1500 到 3000
```

单卡 batch 2 时：

```bash
--datasets.vla_data.per_device_batch_size 2
--trainer.gradient_accumulation_steps 64
--trainer.max_train_steps 30000
--trainer.num_warmup_steps 1500
```

如果继续使用 global batch 16，则不能直接把 7500 step 与主 recipe 的 7500 step 对齐。需要按样本量折算：

```text
等效主 recipe step = 当前 step * 当前 global batch / 128
```

例如：

```text
7500 * 16 / 128 = 937.5
```

## 结论

当前“7500 step loss 不明显下降”最可能的解释不是单一 bug，而是训练配置与复现 recipe 不等价：

- 等效 global batch 可能太小。
- gradient accumulation 可能未真正生效。
- 当前脚本可能跑的是普通 Qwen3 + PI，而不是 ShortMEM。
- 全量训练 VLM 对单卡低 batch 太激进。
- warmup 太短。
- raw flow-matching loss 本身很 noisy。

建议先跑冻结 VLM 的 sanity 实验。如果冻结 VLM 后 loss 能下降，再逐步打开 ShortMEM temporal attention、visual base 或更大的 VLM 训练范围。
