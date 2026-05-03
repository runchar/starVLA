# QwenPIShortMEM Temporal Attention 初始化方案

## 背景

`QwenPIShortMEM` 在 Qwen3-VL visual encoder 内部新增了跨时间的 causal temporal attention。该模块每隔若干层插入一次，只在同一 view、同一 spatial patch index 的 K 帧历史 token 之间做时序注意力，然后在上层丢弃历史 timestep tokens，只保留 current timestep visual tokens 给 Qwen language backbone 和 PI action head。

当前实现中，原 Qwen3-VL visual encoder 的空间模块会复用预训练权重：

- `patch_embed`
- `blocks`
- `merger`
- `deepstack_merger_list`
- positional / rotary 相关模块

但新增的 temporal attention 是新模块，没有对应的 Qwen3-VL 预训练权重。

## 当前初始化

当前代码位置：

```text
starVLA/model/modules/vlm/shortmem_qwen3_visual.py
```

新增 temporal attention 的构造逻辑：

```python
self.temporal_attn = nn.ModuleDict()
for layer_idx in range(len(self.blocks)):
    if (layer_idx + 1) % self.temporal_interval == 0:
        self.temporal_attn[str(layer_idx)] = PatchTemporalCausalAttention(
            config.hidden_size,
            config.num_heads,
        )
```

`PatchTemporalCausalAttention` 当前结构：

```python
class PatchTemporalCausalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
```

因此初始化方式是：

- `LayerNorm` 使用 PyTorch 默认初始化：`weight=1`，`bias=0`
- `nn.MultiheadAttention` 使用 PyTorch 默认随机初始化
- temporal attention 输出通过残差直接加回主干特征

当前 forward 形式：

```python
normed = self.norm(temporal_sequences)
attended, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
temporal_sequences = temporal_sequences + attended
```

这意味着训练初始时，ShortMEM temporal branch 会给原 Qwen3-VL visual feature 加上一个随机残差。

## 风险

随机残差初始化可以训练，但不够保守，主要风险是：

- 初始行为不再等价于原单帧 Qwen3-VL。
- 新增 temporal attention 在训练早期可能扰动已预训练好的 visual feature。
- 如果同时冻结原 `base_visual`，新增 temporal attention 的随机输出会直接影响后续 language backbone 和 PI action head，训练初期 loss 可能更不稳定。
- 对复现实验不友好，因为性能变化同时包含“引入历史帧”和“随机残差扰动”两个因素。

## 推荐方案：Zero-Gate No-op 初始化

推荐给 temporal attention 增加一个可学习 gate，并将 gate 初始化为 0。这样 ShortMEM temporal branch 初始时是严格 no-op，模型行为尽量接近原 Qwen3-VL 单帧路径。

修改 `PatchTemporalCausalAttention`：

```python
class PatchTemporalCausalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden_states: torch.Tensor, num_items: int, history_frames: int, tokens_per_frame: int):
        hidden_size = hidden_states.shape[-1]
        grouped = hidden_states.reshape(num_items, history_frames, tokens_per_frame, hidden_size)
        temporal_sequences = grouped.permute(0, 2, 1, 3).reshape(
            num_items * tokens_per_frame,
            history_frames,
            hidden_size,
        )
        normed = self.norm(temporal_sequences)
        mask = build_temporal_causal_mask(history_frames, hidden_states.device)
        attended, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
        temporal_sequences = temporal_sequences + self.gate * attended
        return temporal_sequences.reshape(num_items, tokens_per_frame, history_frames, hidden_size).permute(
            0, 2, 1, 3
        ).reshape(num_items * history_frames * tokens_per_frame, hidden_size)
```

效果：

- `gate=0` 时，temporal attention 初始输出不改变主干特征。
- 原 Qwen3-VL spatial visual path 初始行为被最大程度保留。
- 训练过程中 `gate` 会逐步学习 temporal branch 的贡献强度。
- 如果冻结 `base_visual`，新增 temporal attention 仍然可训练，但初始阶段不会用随机残差破坏原特征。

## 可选改进：Layer-wise Gate

如果希望每个插入层单独控制 temporal branch 强度，上述 `gate` 已经满足，因为每个 `PatchTemporalCausalAttention` 实例都有自己的 gate。

如果希望每个 channel 单独控制，可以改成向量 gate：

```python
self.gate = nn.Parameter(torch.zeros(hidden_size))
```

forward 中广播：

```python
temporal_sequences = temporal_sequences + self.gate.view(1, 1, -1) * attended
```

更推荐第一版使用 scalar gate，因为参数更少，也更容易解释和 debug。

## 训练建议

采用 zero-gate no-op 初始化后，推荐训练策略：

- 第一阶段冻结 `language_model`、`lm_head` 和 `visual.base_visual`，只训练 `temporal_attn` 与 PI action head。
- 确认 loss 稳定后，再考虑放开 `visual.base_visual`。
- 不建议一开始同时放开 language backbone，除非显存和数据规模足够。

推荐 freeze 配置：

```bash
--trainer.freeze_modules qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual
```

## 验证方式

实现 zero-gate 后应增加以下检查：

1. 初始化检查：

```python
for module in model.qwen_vl_interface.model.model.visual.temporal_attn.values():
    assert module.gate.item() == 0.0
```

2. no-op 行为检查：

- 在相同 current frame 输入下，`gate=0` 时 ShortMEM visual 输出应尽量接近普通 Qwen3-VL current-frame visual 输出。
- 如果存在 dropout 或 dtype 差异，应使用 eval 模式和固定随机种子。

3. 训练检查：

- 训练若干 step 后，`gate` 应从 0 开始产生非零更新。
- `temporal_attn.*.gate` 和 `temporal_attn.*.attn.*` 参数应在未冻结参数组中。

## 实现状态

2026-05-03 已在 OFTShortMEM 路线上实现 zero-gate 初始化：

- `PatchTemporalCausalAttention` 增加 `gate` 参数。
- `QwenOFTShortMEM` 默认使用 `temporal_gate_init=0.0`。
- 其他 ShortMEM framework 默认仍使用 `temporal_gate_init=1.0`，尽量保持历史行为。
- 可通过 `--framework.qwenvl.shortmem.temporal_gate_init 0.0` 显式覆盖。

OFTShortMEM 已通过单元测试和 2-step temporal train smoke：

```text
Step 1, Loss: {'action_dit_loss': 0.3961758315563202, ...}
Step 2, Loss: {'action_dit_loss': 0.24638696014881134, ...}
```

## 结论

随机初始化可以用于 smoke test，但不推荐作为正式复现实验的默认方案。OFTShortMEM 当前已改成 zero-gate no-op 初始化，使 ShortMEM 在训练开始时更接近原单帧 Qwen3-VL，再通过训练逐步学习历史帧信息。
