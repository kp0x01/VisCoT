#!/bin/bash
set -euo pipefail

DATA_DIR=${1:-./temporal}
OUTPUT_DIR=${2:-./checkpoints/viscot-temporal-prefix}
MODEL_BASE=${MODEL_BASE:-lmsys/vicuna-7b-v1.5}
VISION_TOWER=${VISION_TOWER:-openai/clip-vit-large-patch14-336}
PROJECTOR=${PROJECTOR:-./checkpoints/llava_7b_mm_projector.bin}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
ENABLE_LORA=${ENABLE_LORA:-0}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_BIAS=${LORA_BIAS:-none}
TUNE_MM_MLP_ADAPTER=${TUNE_MM_MLP_ADAPTER:-0}

TRAIN_JSON=$DATA_DIR/temporal_train.json
VAL_JSON=$DATA_DIR/temporal_val.json
IMAGE_FOLDER=$DATA_DIR/workspace/data

if [[ ! -f "$TRAIN_JSON" ]]; then
  echo "Missing $TRAIN_JSON" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_FOLDER" ]]; then
  echo "Missing $IMAGE_FOLDER" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

declare -a CMD=(
  deepspeed llava/train/train_mem.py
    --deepspeed ./scripts/zero3.json
    --model_name_or_path "$MODEL_BASE"
    --version v1
    --data_path "$TRAIN_JSON"
    --image_folder "$IMAGE_FOLDER"
    --vision_tower "$VISION_TOWER"
    --pretrain_mm_mlp_adapter "$PROJECTOR"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --freeze_backbone True
    --tune_mm_mlp_adapter $([[ "$TUNE_MM_MLP_ADAPTER" == "1" ]] && echo True || echo False)
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs $EPOCHS
    --per_device_train_batch_size $BATCH_SIZE
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps $GRAD_ACCUM
    --learning_rate $LEARNING_RATE
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 10
    --evaluation_strategy "no"
    --save_strategy epoch
    --save_total_limit 3
    --bf16 False
    --fp16 True
    --tf32 False
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to tensorboard
)

if [[ "$ENABLE_LORA" == "1" ]]; then
  CMD+=(
    --lora_enable True
    --lora_r $LORA_R
    --lora_alpha $LORA_ALPHA
    --lora_dropout $LORA_DROPOUT
    --lora_bias $LORA_BIAS
  )
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
