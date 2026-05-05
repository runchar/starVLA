#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Multi-node clusters can override these from the scheduler environment.
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name="${Framework_name:-QwenGR00T}"
freeze_module_list=''
#freeze_module_list='qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual'
base_vlm="${base_vlm:-./playground/Pretrained_models/Qwen3-VL-2B-Instruct}"
config_yaml="${config_yaml:-./examples/CoTrainVLM/train_files/starvla_cotrain_vsi590k_partial_full_model.yaml}"
vla_data_root="${vla_data_root:-playground/Datasets/LEROBOT_LIBERO_DATA}"
vla_data_mix="${vla_data_mix:-libero_goal}"
run_root_dir="${run_root_dir:-./playground/Checkpoints}"
run_id="${run_id:-vsi590k_partial_full_model_a100_$(date +%Y%m%d_%H%M%S)}"
wandb_project="${wandb_project:-starVLA_Cotrain_VSI590K}"
wandb_entity="${wandb_entity:-your_wandb_entity}"

num_processes="${num_processes:-8}"
vla_batch_size="${vla_batch_size:-1}"
vlm_batch_size="${vlm_batch_size:-1}"
grad_accum="${grad_accum:-8}"
max_train_steps="${max_train_steps:-100000}"
save_interval="${save_interval:-5000}"
eval_interval="${eval_interval:-1000}"
logging_frequency="${logging_frequency:-10}"
attn_implementation="${attn_implementation:-flash_attention_2}"
deepspeed_config="${deepspeed_config:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
# === End of environment variable configuration ===
###########################################################################################

vsi_root="${vsi_root:-playground/Datasets/VSI-590K}"
raw_jsonl="${raw_jsonl:-${vsi_root}/raw/vsi_590k.jsonl}"
partial_jsonl="${partial_jsonl:-${vsi_root}/annotations/vsi590k_starvla_smoke.jsonl}"

if [[ ! -f "${partial_jsonl}" ]]; then
  python examples/CoTrainVLM/train_files/prepare_vsi590k_for_starvla.py \
    --input-jsonl "${raw_jsonl}" \
    --output-jsonl "${partial_jsonl}" \
    --media-root "${vsi_root}" \
    --prefix robotics \
    --prefix s3dis \
    --require-media-exists
fi

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file "${deepspeed_config}" \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla_cotrain.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.attn_implementation "${attn_implementation}" \
  --framework.qwenvl.enable_gradient_checkpointing true \
  --datasets.vla_data.data_root_dir "${vla_data_root}" \
  --datasets.vla_data.data_mix "${vla_data_mix}" \
  --datasets.vla_data.per_device_batch_size "${vla_batch_size}" \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vlm_data.dataset_use vsi590k_starvla_partial_downloaded \
  --datasets.vlm_data.per_device_batch_size "${vlm_batch_size}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.eval_interval "${eval_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.gradient_accumulation_steps "${grad_accum}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  2>&1 | tee "${output_dir}/train.log"

exit "${PIPESTATUS[0]}"
