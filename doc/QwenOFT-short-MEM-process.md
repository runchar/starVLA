# QwenOFTShortMEM 实现与测试过程记录

## 目标

本次任务是在已有 QwenPIShortMEM 的基础上，把 ShortMEM history 旁路推广到 OFT 架构，使 `QwenOFT` 路线也能使用 Qwen3-VL ShortMEM visual encoder。

核心约束沿用 `doc/Qwen-short-MEM.md`：

- current image 仍作为普通 Qwen image placeholder 输入。
- K 帧历史图像不进入 chat template，不让 language backbone 的 image token 数膨胀为 K 倍。
- 历史帧只通过 `shortmem_pixel_values`、`shortmem_grid_thw`、`shortmem_history_frames` 旁路传入 ShortMEM visual wrapper。
- action head 保持 OFT 的 MLP regression 逻辑，不改 ShortMEM visual wrapper 和 dataloader 主结构。

## 方案

OFT 与 PI 的差异在 action head，不在 VLM 输入构建。因此本次采用最小改动：

1. `QwenOFT.forward()` 读取 batch 中的 `image_history`。
2. `QwenOFT.predict_action()` 读取并保持 `image_history` 的嵌套结构，推理 resize 时同步 resize history。
3. 调用 `self.qwen_vl_interface.build_qwenvl_inputs(...)` 时传入 `image_history=batch_image_history`。
4. 新增 `QwenOFTShortMEM` framework alias，继承 `Qwenvl_OFT`，让 YAML 可以显式写：

```yaml
framework:
  name: QwenOFTShortMEM
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct
    attn_implementation: sdpa
    enable_gradient_checkpointing: true
    shortmem:
      history_frames: 2
      temporal_interval: 4
      prune_after_layer: 4

datasets:
  vla_data:
    image_history_frames: 2
```

实际 ShortMEM 分支仍由 `base_vlm` 名称中的 `Qwen3-VL-ShortMem` 触发。

## 修改文件

- `starVLA/model/framework/VLM4A/QwenOFT.py`
  - `forward()` 增加 `batch_image_history`，并传入 VLM input builder。
  - `predict_action()` 增加 `batch_image_history`，并在 `obs_image_size` 推理 resize 时同步 resize history。
- `starVLA/model/framework/VLM4A/QwenOFTShortMEM.py`
  - 注册 `QwenOFTShortMEM`，继承 `Qwenvl_OFT`。
- `starVLA/model/modules/vlm/QWen3.py`
  - 补齐普通 Qwen3 的 `enable_gradient_checkpointing` 行为：关闭 `use_cache`，启用 gradient checkpointing 和 input grads。
  - 这个修正用于让现有 ShortMEM 单元测试完整通过，也和 ShortMEM wrapper 的行为保持一致。
- `tests/test_qwen3_shortmem.py`
  - 增加 OFT forward / predict_action 的 history 传递测试。
  - 增加 `QwenOFTShortMEM` registry 测试。

## TDD 记录

先写 OFT 相关测试并确认失败：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
python -m pytest \
  tests/test_qwen3_shortmem.py::test_qwenoft_forward_passes_image_history_to_shortmem_vlm \
  tests/test_qwen3_shortmem.py::test_qwenoft_predict_action_passes_image_history_to_shortmem_vlm \
  tests/test_qwen3_shortmem.py::test_qwenoft_shortmem_framework_alias_is_registered \
  -q
```

RED 结果：

```text
3 failed
```

失败原因：

- `QwenOFT.forward()` 和 `QwenOFT.predict_action()` 传给 VLM 的 `image_history` 是 `None`。
- `QwenOFTShortMEM` 尚未注册。

实现后重新运行同一组测试：

```text
3 passed in 3.48s
```

随后运行完整 ShortMEM 测试文件：

```bash
python -m pytest tests/test_qwen3_shortmem.py -q
```

结果：

```text
9 passed in 3.47s
```

## 静态编译检查

命令：

```bash
python -m py_compile \
  starVLA/model/modules/vlm/QWen3.py \
  starVLA/model/framework/VLM4A/QwenOFT.py \
  starVLA/model/framework/VLM4A/QwenOFTShortMEM.py \
  tests/test_qwen3_shortmem.py
```

结果：通过，无语法错误。

## 1-step 训练 smoke

本机环境：

- GPU：NVIDIA GeForce RTX 4090，24GB。
- 本地模型别名存在：`playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct -> Qwen3-VL-2B-Instruct`。
- 本地数据存在：`playground/Datasets/LEROBOT_LIBERO_DATA`。

为了先验证闭环，smoke 冻结整个 VLM，只训练 OFT MLP action head：

```bash
source /home/brl4090/miniconda3/etc/profile.d/conda.sh
conda activate dfs
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_cotrain_libero.yaml \
  --framework.name QwenOFTShortMEM \
  --framework.qwenvl.base_vlm ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.qwenvl.enable_gradient_checkpointing true \
  --framework.qwenvl.shortmem.history_frames 2 \
  --framework.qwenvl.shortmem.temporal_interval 4 \
  --framework.qwenvl.shortmem.prune_after_layer 4 \
  --framework.action_model.action_model_type MLP \
  --framework.action_model.action_dim 7 \
  --framework.action_model.action_horizon 8 \
  --framework.action_model.future_action_window_size 7 \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.image_history_frames 2 \
  --datasets.vla_data.obs_image_size '[224,224]' \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps 1 \
  --trainer.num_warmup_steps 0 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency 1 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir /tmp/starvla_oft_shortmem_smoke \
  --run_id smoke_qwenoft_shortmem_freeze_vlm
```

关键日志：

```text
[QWen3ShortMem] gradient_checkpointing ENABLED (use_reentrant=False)
Step 1, Loss: {'action_dit_loss': 0.5096408128738403, ...}
Training complete. Final model saved at /tmp/starvla_oft_shortmem_smoke/smoke_qwenoft_shortmem_freeze_vlm/final_model
```

产物：

```text
/tmp/starvla_oft_shortmem_smoke/smoke_qwenoft_shortmem_freeze_vlm/final_model/pytorch_model.pt
size: 5.5G
```

保存的 accessed config 确认本次 smoke 使用了：

```yaml
framework:
  name: QwenOFTShortMEM
  action_model:
    action_hidden_dim: 2048
    action_horizon: 8
    action_model_type: MLP
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct
    shortmem:
      history_frames: 2
      prune_after_layer: 4
      temporal_interval: 4
datasets:
  vla_data:
    image_history_frames: 2
```

## 结论

`QwenOFTShortMEM` 已经可以沿用现有 ShortMEM Qwen3-VL wrapper 和 dataloader history 输出。当前验证覆盖：

- OFT forward 训练路径会传入 `image_history`。
- OFT predict_action 推理路径会传入 `image_history`。
- `QwenOFTShortMEM` framework alias 可被 registry 自动导入。
- 完整 ShortMEM 单元测试通过。
- 1-step LIBERO 训练 smoke 可跑通并保存 final model。
