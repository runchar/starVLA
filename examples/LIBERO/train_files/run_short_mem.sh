CUDA_VISIBLE_DEVICES=0
#WANDB_MODE=disabled
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

freeze_module_list=''
#freeze_module_list='qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual'
#model='./playground/Pretrained_models/Qwen3-VL-2B-Instruct'
model='./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct'

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_cotrain_libero.yaml \
  --framework.name QwenPIShortMEM \
  --framework.qwenvl.base_vlm  ${model}\
  --framework.qwenvl.attn_implementation sdpa \
  --framework.qwenvl.enable_gradient_checkpointing true \
  --framework.qwenvl.shortmem.history_frames 6 \
  --framework.qwenvl.shortmem.temporal_interval 4 \
  --framework.qwenvl.shortmem.prune_after_layer 4 \
  --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vla_data.image_history_frames 6 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 12000 \
  --trainer.num_warmup_steps 100 \
  --trainer.save_interval 500 \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency 50 \
  --trainer.gradient_accumulation_steps 8 \
  --run_root_dir playground/Checkpoints \
  --run_id qwenpi_shortmem_qwen3_2b_libero_goal \
  --wandb_project starVLA_Libero_shortmem \
  --wandb_entity runchar