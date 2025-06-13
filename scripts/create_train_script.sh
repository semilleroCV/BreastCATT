#!/bin/bash
set -e  # exit on error

########################################
# 1. Basic Configuration
########################################

VIT_VERSION="small"               # Options: small, base, large, huge
DATASET_NAME="SemilleroCV/DMR-IR"

NUM_TRAIN_EPOCHS=30
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1

LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
CHECKPOINTING_STEPS=1000

WITH_TRACKING="--with_tracking"
REPORT_TO="wandb"
WANDB_PROJECT="colcaci"
SEED=3407

########################################
# 2. Derived Variables Calculation
########################################

# Total number of training samples
TOTAL_TRAIN_SAMPLES=5663

# Effective batch size considering gradient accumulation
EFFECTIVE_BATCH_SIZE=$(( TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))

# Total training steps
MAX_TRAIN_STEPS=$(( TOTAL_TRAIN_SAMPLES * NUM_TRAIN_EPOCHS / EFFECTIVE_BATCH_SIZE ))

# Warm‑up steps = 5% of total training steps
NUM_WARMUP_STEPS=$(( MAX_TRAIN_STEPS * 5 / 100 ))

########################################
# 3. Dynamic Flags (Segmentation / Cross-Attention)
########################################

USE_SEGMENTATION=true # or false
USE_CROSS_ATTN=true # or false

SEG_FLAG=""
CROSS_ATTN_FLAG=""
SCRIPT_TAGS=""

if [ "$USE_SEGMENTATION" = true ]; then
  SEG_FLAG="--use_segmentation"
  SCRIPT_TAGS="${SCRIPT_TAGS}_seg"
fi

if [ "$USE_CROSS_ATTN" = true ]; then
  CROSS_ATTN_FLAG="--use_cross_attn"
  SCRIPT_TAGS="${SCRIPT_TAGS}_text"
fi

########################################
# 4. Directories and Script Naming
########################################

OUTPUT_DIR="fvit-${VIT_VERSION}${SCRIPT_TAGS}"
NEW_SCRIPT_NAME="${OUTPUT_DIR}_1x${EFFECTIVE_BATCH_SIZE}-${MAX_TRAIN_STEPS}.sh"

PUSH_TO_HUB="--push_to_hub"       # Use "--push_to_hub" if you want to push the model
HUB_MODEL_ID=""                   # e.g. "SemilleroCV/${OUTPUT_DIR}"
HUB_TOKEN=""                      # your Hugging Face token

########################################
# 5. Training Command
########################################

accelerate launch train.py \
    --vit_version "$VIT_VERSION" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --checkpointing_steps $CHECKPOINTING_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_warmup_steps $NUM_WARMUP_STEPS \
    $WITH_TRACKING \
    --report_to $REPORT_TO \
    --wandb_project "$WANDB_PROJECT" \
    --seed $SEED \
    $SEG_FLAG \
    $CROSS_ATTN_FLAG \
    $PUSH_TO_HUB \
    ${HUB_MODEL_ID:+--hub_model_id "$HUB_MODEL_ID"} \
    ${HUB_TOKEN:+--hub_token "$HUB_TOKEN"}

########################################
# 6. Rename Script
########################################

mkdir -p scripts
mv "$0" "scripts/$NEW_SCRIPT_NAME"