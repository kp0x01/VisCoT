#!/bin/bash
# finetune_recipeqa_qlora.sh
set -e

OUTPUT_DIR="./recipeqa_lora_output"
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python -u -m llava.train.train_mem \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --bits 4 \
    --quant_type nf4 \
    --double_quant True \
    --model_name_or_path checkpoints/VisCoT-7b-336 \
    --model_arc llama \
    --version vicuna_v1 \
    --data_path recipeqa_train_balanced.json \
    --image_folder ./ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --fp16 False \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 30 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo "Training complete!"