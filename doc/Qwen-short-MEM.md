# QwenPIShortMEM 修改说明

## 背景

本次任务是在 StarVLA 的 QwenPI 路线中增加一个 `QwenPIShortMEM` framework，使 Qwen3-VL 的 visual encoder 能接收每个 view 的 K 帧历史图像，并在 visual encoder 内部做 MEM-style ShortMEM 时序建模。

核心约束如下：

- 不把 K 帧历史图像伪装成 pseudo-image compression。
- 不把 K 帧直接作为 K 张普通 image 喂给 Qwen processor，避免 language backbone 侧的 image token 数变成 K 倍。
- StarVLA 外层训练、action head 和 QwenPI 主流程尽量保持透明，只在 dataloader、VLM wrapper 和 Qwen visual encoder 包装层引入 ShortMEM。
- checkpoint 保存逻辑暂不修改，ShortMEM 新参数作为普通模型参数进入现有保存流程。

## 使用方式

配置中可以显式选择 `QwenPIShortMEM`，并通过本地模型目录名触发 ShortMEM VLM wrapper：

```yaml
framework:
  name: QwenPIShortMEM
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct
    attn_implementation: sdpa
    enable_gradient_checkpointing: true
    shortmem:
      history_frames: 4
      temporal_interval: 4
      prune_after_layer: 20

datasets:
  vla_data:
    image_history_frames: 4
```

当前本地模型别名为：

```text
playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct -> Qwen3-VL-2B-Instruct
```

这个目录仍复用原 Qwen3-VL-2B HF 权重文件。是否启用 ShortMEM 由 StarVLA 的 wrapper 分支决定，不需要修改 HF config 的 `model_type`。

## 修改组件

### 1. VLM 路由

修改文件：`starVLA/model/modules/vlm/__init__.py`

新增 `Qwen3-VL-ShortMem` 优先分支，并放在普通 `Qwen3-VL` 分支之前：

```python
if "Qwen3-VL-ShortMem" in vlm_name:
    from .QWen3ShortMem import _QWen3ShortMem_VL_Interface

    return _QWen3ShortMem_VL_Interface(config)
```

这样 `base_vlm` 中包含 `Qwen3-VL-ShortMem` 时会进入 ShortMEM wrapper，普通 `Qwen3-VL` 配置仍走原 `_QWen3_VL_Interface`。

### 2. Qwen3 ShortMEM VLM wrapper

新增文件：`starVLA/model/modules/vlm/QWen3ShortMem.py`

主要职责：

- 使用 `Qwen3VLForConditionalGeneration.from_pretrained()` 加载原 Qwen3-VL 权重。
- 将真实 visual 挂载点 `model.model.visual` 替换为 `ShortMemQwen3VisionModel`。
- 保留原 Qwen chat template 和 processor 输入格式。
- 普通 current image 仍通过 `apply_chat_template()` 生成 `<image>` placeholder。
- 历史帧通过旁路字段 `shortmem_pixel_values`、`shortmem_grid_thw`、`shortmem_history_frames` 传入 visual wrapper。
- `forward()` 调用前把旁路 history context 设置到 visual wrapper，调用结束后清理 context。

`pack_shortmem_history()` 会检查 batch 维度、view 维度和每个 view 的历史帧数一致性，然后把 history frame 展平成 image processor 可接收的列表。

### 3. ShortMEM Qwen3 visual wrapper

新增文件：`starVLA/model/modules/vlm/shortmem_qwen3_visual.py`

主要结构：

- `ShortMemQwen3VisionModel` 包装原 `Qwen3VLVisionModel`，复用原 `patch_embed`、spatial ViT blocks、merger、deepstack merger 等模块。
- `PatchTemporalCausalAttention` 在同一 spatial patch index 的跨 timestep 序列上做 causal temporal attention。
- `build_temporal_causal_mask()` 使用上三角 `-inf` mask，保证 timestep `t` 不能 attend 到未来帧。
- `select_current_timestep_tokens()` 在 prune 点或最终输出前只保留 current timestep tokens。
- `prune_after_layer` 之后会重建 `grid_thw`、rotary position embedding 和 `cu_seqlens`，使后续 Qwen language backbone 看到的视觉 token 数与单帧 current image 对齐。

当前实现要求同一个 batch 内 history frame 的 visual grid 固定一致，否则会抛出错误。这与 smoke 配置中的固定 224 输入匹配。

### 4. QwenPI 接入 history 旁路

修改文件：`starVLA/model/framework/VLM4A/QwenPI.py`

`forward()` 和 `predict_action()` 都增加了 `image_history` 读取，并传入 `build_qwenvl_inputs()`：

```python
batch_image_history = (
    [example.get("image_history") for example in examples] if "image_history" in examples[0] else None
)

qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
    images=batch_images,
    instructions=instructions,
    image_history=batch_image_history,
)
```

同时将 action head 使用的 VLM hidden layer 数改为优先读取 YAML 中的 `framework.action_model.diffusion_model_cfg.num_layers`，默认仍为 36。这个改动让训练 smoke 可以用更小的 DiT 层数验证闭环。

### 5. QwenPIShortMEM framework alias

新增文件：`starVLA/model/framework/VLM4A/QwenPIShortMEM.py`

该类继承 `Qwen_PI`，只注册新的 framework 名称：

```python
@FRAMEWORK_REGISTRY.register("QwenPIShortMEM")
class QwenPIShortMEM(Qwen_PI):
    pass
```

实际 ShortMEM 行为仍由 `framework.qwenvl.base_vlm` 触发。这让实验配置可以明确标识为 `QwenPIShortMEM`。

### 6. dataloader 历史帧输出

修改文件：

- `starVLA/dataloader/lerobot_datasets.py`
- `starVLA/dataloader/gr00t_lerobot/datasets.py`

当 `datasets.vla_data.image_history_frames = K` 且 K 不为 1 时：

- `make_LeRobotSingleDataset()` 会把 video modality 的 `delta_indices` 改为 `[-K+1, ..., 0]`。
- `_pack_sample()` 对每个 view 保留 K 帧 resize 后的 PIL 图像。
- `sample["image"]` 只放 current frame，继续服务 Qwen placeholder。
- `sample["image_history"]` 放每个 view 的 K 帧历史，作为 ShortMEM visual 的旁路输入。

样本结构：

```python
sample["image"] = [view0_current, view1_current, ...]
sample["image_history"] = [
    [view0_t0, view0_t1, ..., view0_current],
    [view1_t0, view1_t1, ..., view1_current],
]
```

## 训练参数建议

4090 smoke 已验证的保守配置：

```yaml
framework:
  name: QwenPIShortMEM
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct
    attn_implementation: sdpa
    enable_gradient_checkpointing: true
    shortmem:
      history_frames: 2
      temporal_interval: 4
      prune_after_layer: 4
  action_model:
    diffusion_model_cfg:
      num_layers: 2

datasets:
  vla_data:
    per_device_batch_size: 1
    image_history_frames: 2
    video_backend: torchvision_av

trainer:
  max_train_steps: 1
  gradient_accumulation_steps: 1
```

如果显存紧张，可以先冻结整个 VLM：

```yaml
trainer:
  freeze_modules: qwen_vl_interface
```

更贴近 ShortMEM 的 smoke 配置是冻结原 language model、lm head 和原 visual base，只训练新增 temporal attention 与 action head：

```yaml
trainer:
  freeze_modules: qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual
```

## 当前限制

- 第一版只实现 Qwen3-VL 路线。
- 当前 ShortMEM history 输入要求每个 batch 内所有 history frame 的 grid 尺寸一致。
- 还没有实现 adapter-only 或 ShortMEM-only checkpoint 保存，沿用项目现有全量 `state_dict` 保存。
- 尚未跑长训练，只验证了 1 step smoke。
- `ruff` 静态检查暴露了大量仓库既有风格问题，本次没有大范围格式化旧代码，避免引入无关 diff。

## Qwen3-VL-4B 兼容性判断

理论上支持 `Qwen/Qwen3-VL-4B-Instruct`，但当前工作区没有本地 4B 权重目录，因此还没有做 4B 实机加载和训练验证。

判断依据：

- 4B 官方 config 的 `architectures` 仍是 `Qwen3VLForConditionalGeneration`，`model_type` 仍是 `qwen3_vl`。
- 4B 的 `vision_config` 与本实现依赖的字段一致，包括 `deepstack_visual_indexes`、`depth`、`hidden_size`、`num_heads`、`patch_size`、`spatial_merge_size`、`temporal_patch_size`。
- 当前 wrapper 不写死 LLM hidden size，而是从 `self.model.config.text_config.hidden_size` 和 `self.qwen_vl_interface.model.config.hidden_size` 动态读取。4B 的 text hidden size 为 2560，2B 为 2048，action head 的 hidden dim 会跟随配置更新。
- ShortMEM temporal attention 的维度来自 `base_visual.config.hidden_size` 和 `base_visual.config.num_heads`。4B 的 visual hidden size 仍为 1024、num heads 仍为 16，因此新增 temporal attention 结构与 2B 相同。

需要注意：

- 4B 的 language backbone 更大，显存压力会明显高于 2B。建议先使用 `history_frames=2`、`per_device_batch_size=1`、`diffusion_model_cfg.num_layers=2`，并冻结 `language_model`、`lm_head` 和 `visual.base_visual`，只训练 ShortMEM temporal attention 与 action head。
- 如果要通过本地目录名触发 ShortMEM，需要建立类似软链接：
  ```bash
  ln -s Qwen3-VL-4B-Instruct playground/Pretrained_models/Qwen3-VL-ShortMem-4B-Instruct
  ```
  然后把 `framework.qwenvl.base_vlm` 指到 `./playground/Pretrained_models/Qwen3-VL-ShortMem-4B-Instruct`。
