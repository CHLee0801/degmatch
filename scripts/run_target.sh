#!/bin/bash
# End-to-end DeGMatch run for one target, 8 GPUs.
#   bash scripts/run_target.sh <target_name> <reference_traces.jsonl> [K=10000]
# Paper settings
#   math    (AIME, AMC, HMMT, MATH500, Gaokao): POOL=data/pool_math    GATE_N=30000 K=10000
#   science (GPQA, JEE)                       : POOL=data/pool_science GATE_N=15000 K=5000 BENCHES="GPQADiamond JEEBench"
#   code    (LiveCodeBench, CodeForces)       : POOL=data/pool_code    GATE_N=9000  K=3000 BENCHES="LiveCodeBench CodeForces"
# Prerequisites: the pool built with degmatch/build_pool.py; reference traces as JSONL with problem + trace.
# Student: Qwen2.5-Math-7B with RoPE base 300k (set STUDENT to its local path).
set -euo pipefail
T=$1; REFS=$2; K=${3:-10000}; NG=${NGPU:-8}; PY=${PYTHON:-python}
STUDENT=${STUDENT:-Qwen/Qwen2.5-Math-7B}
P=${POOL:-data/pool_math}; GN=${GATE_N:-30000}; E=$P/emb; G=data/grads; PFX=$(basename "$P")_$T
BENCHES=${BENCHES:-AIME24 AIME25 AIME26 MATH500A MATH500B}; NAME=degmatch_${T}_$((K / 1000))k

# 1. relevance gate from Qwen3 problem embeddings (pool embeddings are computed once and shared by all targets)
if [ ! -f "$E/pool_problem.part0.npy" ]; then
  for s in $(seq 0 $((NG-1))); do CUDA_VISIBLE_DEVICES=$s $PY degmatch/embed.py --input $P/pool_default.jsonl --out $E/pool_problem --shard $s --nshard $NG & done; wait
fi
$PY degmatch/embed.py --input "$REFS" --out $E/ref_$T
$PY degmatch/gate.py --pool_emb "$E/pool_problem.part*.npy" --ref_emb $E/ref_$T.npy --n "$GN" --out data/gates/${T}_gate_idx.npy

# 2. student gradients (CountSketch, D=16384) for the gated candidates and the references
for s in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$s $PY degmatch/grads.py candidates --model "$STUDENT" --pool $P/pool_default.jsonl \
    --idx data/gates/${T}_gate_idx.npy --out_dir $G --prefix "$PFX" --shard $s --nshard $NG &
done; wait
CUDA_VISIBLE_DEVICES=0 $PY degmatch/grads.py references --model "$STUDENT" --refs "$REFS" --out $G/ref_$T.npy

# 3. deconfounded matching score and the top-K training set
$PY degmatch/score.py --gate data/gates/${T}_gate_idx.npy --grad_dir $G --prefix "$PFX" --refs $G/ref_$T.npy --out data/scores/$T.npz
$PY degmatch/select.py --scores data/scores/$T.npz --k "$K" --pool_dir $P --out data/sets/$NAME

# 4. train and evaluate
BASE_MODEL="$STUDENT" bash train/train.sh data/sets/$NAME checkpoints/$NAME
EVAL_BENCHES="$BENCHES" bash eval/eval.sh checkpoints/$NAME results
$PY eval/merge_results.py --results results --model $NAME --benches $(echo $BENCHES | sed 's/MATH500A MATH500B/MATH500/')
