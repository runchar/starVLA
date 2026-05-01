from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def build_temporal_causal_mask(history_frames: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((history_frames, history_frames), device=device)
    future_positions = torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(future_positions, float("-inf"))
    return mask


def select_current_timestep_tokens(
    hidden_states: torch.Tensor,
    num_items: int,
    history_frames: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    if hidden_states.shape[0] != num_items * history_frames * tokens_per_frame:
        raise ValueError(
            "ShortMEM token shape mismatch: "
            f"got {hidden_states.shape[0]} tokens, expected {num_items * history_frames * tokens_per_frame} "
            f"from num_items={num_items}, history_frames={history_frames}, tokens_per_frame={tokens_per_frame}."
        )
    hidden_size = hidden_states.shape[-1]
    return hidden_states.reshape(num_items, history_frames, tokens_per_frame, hidden_size)[:, -1].reshape(
        num_items * tokens_per_frame, hidden_size
    )


@dataclass
class ShortMemContext:
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    history_frames: int
    num_items: int


class PatchTemporalCausalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

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
        temporal_sequences = temporal_sequences + attended
        return temporal_sequences.reshape(num_items, tokens_per_frame, history_frames, hidden_size).permute(
            0, 2, 1, 3
        ).reshape(num_items * history_frames * tokens_per_frame, hidden_size)


class ShortMemQwen3VisionModel(nn.Module):
    def __init__(
        self,
        base_visual: nn.Module,
        history_frames: int = 4,
        temporal_interval: int = 4,
        prune_after_layer: Optional[int] = None,
    ):
        super().__init__()
        self.base_visual = base_visual
        self.history_frames = history_frames
        self.temporal_interval = temporal_interval
        self.prune_after_layer = prune_after_layer
        self._shortmem_context: ShortMemContext | None = None

        self.spatial_merge_size = base_visual.spatial_merge_size
        self.patch_size = base_visual.patch_size
        self.spatial_merge_unit = base_visual.spatial_merge_unit
        self.patch_embed = base_visual.patch_embed
        self.pos_embed = base_visual.pos_embed
        self.num_grid_per_side = base_visual.num_grid_per_side
        self.rotary_pos_emb = base_visual.rotary_pos_emb
        self.blocks = base_visual.blocks
        self.merger = base_visual.merger
        self.deepstack_visual_indexes = base_visual.deepstack_visual_indexes
        self.deepstack_merger_list = base_visual.deepstack_merger_list
        self.gradient_checkpointing = getattr(base_visual, "gradient_checkpointing", False)

        config = base_visual.config
        self.temporal_attn = nn.ModuleDict()
        for layer_idx in range(len(self.blocks)):
            if (layer_idx + 1) % self.temporal_interval == 0:
                self.temporal_attn[str(layer_idx)] = PatchTemporalCausalAttention(
                    config.hidden_size,
                    config.num_heads,
                )

    @property
    def dtype(self):
        dtype = getattr(self.base_visual, "dtype", None)
        if dtype is not None:
            return dtype
        param = next(self.base_visual.parameters(), None)
        return param.dtype if param is not None else torch.float32

    @property
    def device(self):
        device = getattr(self.base_visual, "device", None)
        if device is not None:
            return device
        param = next(self.base_visual.parameters(), None)
        return param.device if param is not None else torch.device("cpu")

    @property
    def config(self):
        return self.base_visual.config

    def set_shortmem_context(
        self,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        history_frames: int | None = None,
    ) -> None:
        if pixel_values is None:
            self._shortmem_context = None
            return
        if grid_thw is None:
            raise ValueError("ShortMEM requires history grid_thw when history pixel values are provided.")
        frames = int(history_frames or self.history_frames)
        if grid_thw.shape[0] % frames != 0:
            raise ValueError(f"ShortMEM history grid count {grid_thw.shape[0]} is not divisible by K={frames}.")
        self._shortmem_context = ShortMemContext(
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            history_frames=frames,
            num_items=grid_thw.shape[0] // frames,
        )

    def clear_shortmem_context(self) -> None:
        self._shortmem_context = None

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        return self.base_visual.rot_pos_emb(grid_thw)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        if self._shortmem_context is None:
            return self.base_visual(hidden_states, grid_thw=grid_thw, **kwargs)
        try:
            return self._forward_shortmem(self._shortmem_context, **kwargs)
        finally:
            self.clear_shortmem_context()

    def _forward_shortmem(self, context: ShortMemContext, **kwargs):
        hidden_states = self.patch_embed(context.pixel_values.type(self.dtype))
        grid_thw = context.grid_thw.to(device=hidden_states.device)
        tokens_per_frame_per_item = (grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        if len(set(tokens_per_frame_per_item)) != 1:
            raise ValueError("ShortMEM currently requires fixed visual grid across all history frames and views.")
        tokens_per_frame = int(tokens_per_frame_per_item[0])

        pos_embeds = self.base_visual.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        pruned = False
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if str(layer_num) in self.temporal_attn and not pruned:
                hidden_states = self.temporal_attn[str(layer_num)](
                    hidden_states,
                    num_items=context.num_items,
                    history_frames=context.history_frames,
                    tokens_per_frame=tokens_per_frame,
                )

            if self.prune_after_layer is not None and layer_num >= self.prune_after_layer and not pruned:
                hidden_states = select_current_timestep_tokens(
                    hidden_states, context.num_items, context.history_frames, tokens_per_frame
                )
                grid_thw = grid_thw.reshape(context.num_items, context.history_frames, 3)[:, -1]
                rotary_pos_emb = self.rot_pos_emb(grid_thw).reshape(hidden_states.shape[0], -1)
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                position_embeddings = (emb.cos(), emb.sin())
                cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                    dim=0,
                    dtype=torch.int32,
                )
                cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
                pruned = True

            if layer_num in self.deepstack_visual_indexes:
                deepstack_idx = self.deepstack_visual_indexes.index(layer_num)
                feature = hidden_states
                if not pruned:
                    feature = select_current_timestep_tokens(
                        feature, context.num_items, context.history_frames, tokens_per_frame
                    )
                deepstack_feature_lists.append(self.deepstack_merger_list[deepstack_idx](feature))

        if not pruned:
            hidden_states = select_current_timestep_tokens(
                hidden_states, context.num_items, context.history_frames, tokens_per_frame
            )
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists
