CUDA_VISIBLE_DEVICES=0
#WANDB_MODE=disabled
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

freeze_module_list=''
#freeze_module_list='qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head,qwen_vl_interface.model.model.visual.base_visual'
#model='./playground/Pretrained_models/Qwen3-VL-2B-Instruct'
model='./playground/Pretrained_models/Qwen3-VL-ShortMem-2B-Instruct'


CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml starVLA/config/training/starvla_cotrain_libero.yaml \
    --framework.name QwenOFTShortMEM \
    --framework.qwenvl.base_vlm ${model} \
    --framework.qwenvl.attn_implementation sdpa \
    --framework.qwenvl.enable_gradient_checkpointing true \
    --framework.qwenvl.shortmem.history_frames 6 \
    --framework.qwenvl.shortmem.temporal_interval 4 \
    --framework.qwenvl.shortmem.prune_after_layer 21 \
    --framework.action_model.action_model_type MLP \
    --framework.action_model.action_dim 7 \
    --framework.action_model.action_horizon 8 \
    --framework.action_model.future_action_window_size 7 \
    --framework.action_model.past_action_window_size 0 \
    --datasets.vla_data.data_root_dir playground/Datasets/LEROBOT_LIBERO_DATA \
    --datasets.vla_data.data_mix libero_all \
    --datasets.vla_data.per_device_batch_size 8 \
    --datasets.vla_data.image_history_frames 6 \
    --datasets.vla_data.obs_image_size '[224,224]' \
    --datasets.vla_data.video_backend torchvision_av \
    --trainer.freeze_modules ${freeze_module_list} \
    --trainer.max_train_steps 13000 \
    --trainer.save_interval 500 \
    --trainer.eval_interval 100 \
    --trainer.logging_frequency 10 \
    --trainer.gradient_accumulation_steps 4 \
    --run_root_dir playground/Checkpoints \
    --run_id qwenoft_shortmem_libero_all \
    --wandb_project starVLA_Libero_oft_shortmem