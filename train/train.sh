#!/bin/bash
# FSDP SFT launcher.
#   bash train/train.sh <set_prefix> <output_dir>
#     reads <set_prefix>_text.jsonl (from degmatch/select.py) and saves the model to <output_dir>
#
# Paper settings
#   subsets (1k-20k) : lr 3e-5, global batch 32 (8 GPUs x 1 x grad-accum 4), 5 epochs, block 32768, cosine, warmup 5%
#   full pools       : LR=5e-5 GA=16 EPOCHS=3 (global batch 128)
# Override with environment variables: BASE_MODEL, NPROC, PER_DEV, GA, EPOCHS, LR, BLOCK, MASTER_PORT, PYTHON, TORCHRUN
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: bash train/train.sh <set_prefix> <output_dir>"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$1_text.jsonl; OUT=$2
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-Math-7B}
NPROC=${NPROC:-8}; PER_DEV=${PER_DEV:-1}; GA=${GA:-4}; EPOCHS=${EPOCHS:-5}; LR=${LR:-3e-5}; BLOCK=${BLOCK:-32768}
PYTHON=${PYTHON:-python}; TORCHRUN=${TORCHRUN:-torchrun}
[ -f "$DATA" ] || { echo "missing $DATA"; exit 1; }
mkdir -p "$OUT"
echo "train $DATA -> $OUT (base=$BASE_MODEL, gbs=$((NPROC*PER_DEV*GA)), lr=$LR, epochs=$EPOCHS)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false WANDB_MODE=${WANDB_MODE:-disabled} \
$TORCHRUN --nproc-per-node "$NPROC" --master_port "${MASTER_PORT:-12457}" "$HERE/sft.py" \
  --model_name="$BASE_MODEL" --train_file_path="$DATA" \
  --block_size="$BLOCK" --per_device_train_batch_size="$PER_DEV" --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps="$GA" --num_train_epochs="$EPOCHS" --warmup_ratio=0.05 \
  --fsdp "full_shard auto_wrap" --fsdp_config "$HERE/fsdp_config_qwen.json" \
  --bf16=True --gradient_checkpointing=True --gradient_checkpointing_kwargs='{"use_reentrant":false}' \
  --eval_strategy=no --logging_steps=1 --save_strategy=no --lr_scheduler_type=cosine \
  --learning_rate="$LR" --weight_decay=1e-4 --adam_beta1=0.9 --adam_beta2=0.95 \
  --seed="${SEED:-42}" --ddp_timeout=14400 --report_to=none --output_dir="$OUT" --save_only_model=True

# make generation stop at <|im_end|>; otherwise evaluation keeps generating up to max_model_len
$PYTHON "$HERE/fix_eos.py" "$OUT"
