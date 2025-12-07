#!/bin/bash
set -e
set -x

export PYTHONDONTWRITEBYTECODE=1

MODEL_PATH="checkpoints/VisCoT-7b-336"
DATA_PATH="temporal_train.json"
OUTPUT_DIR="./temporal_lora_output"

mkdir -p ${OUTPUT_DIR}

echo "Training with 4-bit quantization..."

python -c "import torch; torch.cuda.empty_cache()"

CUDA_VISIBLE_DEVICES=0 python -u -m llava.train.train_mem \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bits 4 \
    --quant_type nf4 \
    --double_quant True \
    --model_name_or_path ${MODEL_PATH} \
    --model_arc llama \
    --version vicuna_v1 \
    --data_path ${DATA_PATH} \
    --image_folder ./ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --fp16 False \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo "Training complete!"
