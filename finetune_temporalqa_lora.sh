#!/bin/bash
# finetune_temporalqa_FULL_LORA.sh (16-bit, NOT QLoRA)

CUDA_VISIBLE_DEVICES=0 python -u -m llava.train.train_mem \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --bits 16 \                        # ← Changed from 4
    --fp16 True \                      # ← Added FP16
    # REMOVED: --quant_type nf4
    # REMOVED: --double_quant True
    --model_name_or_path checkpoints/VisCoT-7b-336 \
    --model_arc llama \
    --version vicuna_v1 \
    --data_path temporal_train.json \
    --image_folder ./ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --output_dir ./temporalqa_lora_fp16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \              # ← Lower LR for FP16
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none