from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.model.modules.vlm.shortmem_qwen3_visual import ShortMemQwen3VisionModel
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
_ACTION_TOKEN_MIN = 151669
_ACTION_TOKEN_MAX = 153716


@dataclass
class PackedShortMemHistory:
    current_images: list
    flat_history_images: list
    num_samples: int
    num_views_per_sample: list[int]
    history_frames: int


def _cfg_get(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def pack_shortmem_history(current_images: list, image_history: Optional[list]) -> PackedShortMemHistory | None:
    if image_history is None:
        return None
    if len(current_images) != len(image_history):
        raise ValueError(
            f"ShortMEM batch mismatch: current image batch has {len(current_images)} samples, "
            f"history has {len(image_history)} samples."
        )

    flat_history_images = []
    num_views_per_sample = []
    history_frames = None
    for sample_idx, (sample_current, sample_history) in enumerate(zip(current_images, image_history)):
        if len(sample_current) != len(sample_history):
            raise ValueError(
                f"ShortMEM view mismatch at sample {sample_idx}: current has {len(sample_current)} views, "
                f"history has {len(sample_history)} views."
            )
        num_views_per_sample.append(len(sample_current))
        for view_history in sample_history:
            if history_frames is None:
                history_frames = len(view_history)
            elif history_frames != len(view_history):
                raise ValueError("ShortMEM requires the same K history frames for every view in a batch.")
            flat_history_images.extend(view_history)

    return PackedShortMemHistory(
        current_images=current_images,
        flat_history_images=flat_history_images,
        num_samples=len(current_images),
        num_views_per_sample=num_views_per_sample,
        history_frames=history_frames or 0,
    )


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class _QWen3ShortMem_VL_Interface(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        shortmem_config = _cfg_get(qwenvl_config, "shortmem", {})
        model_id = _cfg_get(qwenvl_config, "base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = _cfg_get(qwenvl_config, "attn_implementation", "sdpa")
        enable_grad_ckpt = _as_bool(
            _cfg_get(
                qwenvl_config,
                "enable_gradient_checkpointing",
                config.trainer.get("enable_gradient_checkpointing", False) if hasattr(config, "trainer") else False,
            )
        )

        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        )
        model.config.use_cache = False
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False
        if enable_grad_ckpt:
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                print("[QWen3ShortMem] gradient_checkpointing ENABLED (use_reentrant=False)", flush=True)
            except TypeError:
                model.gradient_checkpointing_enable()
                print("[QWen3ShortMem] gradient_checkpointing ENABLED", flush=True)

        model.model.visual = ShortMemQwen3VisionModel(
            model.model.visual,
            history_frames=int(_cfg_get(shortmem_config, "history_frames", 4)),
            temporal_interval=int(_cfg_get(shortmem_config, "temporal_interval", 4)),
            prune_after_layer=_cfg_get(shortmem_config, "prune_after_layer", 20),
        )

        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        shortmem_pixel_values = kwargs.pop("shortmem_pixel_values", None)
        shortmem_grid_thw = kwargs.pop("shortmem_grid_thw", None)
        shortmem_history_frames = kwargs.pop("shortmem_history_frames", None)
        visual = self.model.model.visual
        if shortmem_pixel_values is not None:
            visual.set_shortmem_context(shortmem_pixel_values, shortmem_grid_thw, shortmem_history_frames)

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(**kwargs)
        finally:
            if hasattr(visual, "clear_shortmem_context"):
                visual.clear_shortmem_context()
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, image_history=None, **kwargs):
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[len(messages)]}]})
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        packed_history = pack_shortmem_history(images, image_history)
        if packed_history is not None:
            history_inputs = self.processor.image_processor(
                images=packed_history.flat_history_images,
                return_tensors="pt",
            )
            batch_inputs["shortmem_pixel_values"] = history_inputs["pixel_values"]
            batch_inputs["shortmem_grid_thw"] = history_inputs["image_grid_thw"]
            batch_inputs["shortmem_history_frames"] = packed_history.history_frames

        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                mask_seq = (seq >= _ACTION_TOKEN_MIN) & (seq <= _ACTION_TOKEN_MAX)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)
