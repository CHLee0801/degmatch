#!/bin/bash
# Data-parallel evaluation with Evalchemy + vLLM (tensor_parallel_size=1, one subjob per GPU).
#   EVAL_BENCHES="AIME24 AIME25 AIME26 MATH500A MATH500B" bash eval/eval.sh <model_dir> [results_root=results]
#
# Each benchmark is split into subjobs x repeats-per-subjob; the merged score is the mean over all samples (avg@N):
#   AIME24/25/26, AMC23, HMMT, GAOKAO24/25 : 8 x 2 = avg@16
#   MATH500A + MATH500B (250 + 250)        : 4 x 1 = avg@4
#   GPQADiamond 8 x 1, JEEBench 4 x 1, LiveCodeBench 4 x 1, CodeForces 4 x 1
# Sampling: temperature 0.6, top-p 0.95, max_model_len 32768.
# Environment: EVALCHEMY (default third_party/evalchemy), EVAL_GPUS (default "0 1 2 3 4 5 6 7"), PYTHON.
# Benchmark options used in the paper: JEE_SUBJECTS=phy,chem ; LCB_VERSION_TAG=release_v6 LCB_START=2025-01-01 LCB_END=2025-04-30.
# GPQA-Diamond is gated on the Hugging Face Hub: export HF_TOKEN beforehand.
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: [EVAL_BENCHES=...] bash eval/eval.sh <model_dir> [results_root]"; exit 1; }
MODEL=$(cd "$1" && pwd); RES=$(mkdir -p "${2:-results}" && cd "${2:-results}" && pwd)
NAME=$(basename "$MODEL"); EV=$(cd "${EVALCHEMY:-third_party/evalchemy}" && pwd); PY=${PYTHON:-python}
GPUS=(${EVAL_GPUS:-0 1 2 3 4 5 6 7}); BENCHES=(${EVAL_BENCHES:-AIME24 AIME25 MATH500A MATH500B})
mkdir -p "$RES/logs"

cfg(){ case "$1" in   # -> "<env prefix> <num subjobs> <repeats per subjob>"
  AIME24|AIME25|AMC23) echo "AIME 8 2";;  AIME26) echo "AIME26 8 2";;  HMMT) echo "HMMT 8 2";;
  GAOKAO24|GAOKAO25)   echo "$1 8 2";;     MATH500A|MATH500B) echo "MATH 4 1";;
  GPQADiamond) echo "GPQA 8 1";;  JEEBench) echo "JEE 4 1";;  LiveCodeBench) echo "LCB 4 1";;  CodeForces) echo "CF 4 1";;
  *) echo "AIME 8 1";; esac; }

JOBS=()
for B in "${BENCHES[@]}"; do read -r PFX NSUB NREP <<<"$(cfg "$B")"
  for (( s=0; s<NSUB; s++ )); do JOBS+=("$B|$((s*NREP))|$NREP|$PFX"); done; done

run_job(){ IFS='|' read -r B OFF NREP PFX <<<"$1"; local g=$2 out="$RES/${NAME}_${B}_off${OFF}"
  find "$out" -name 'results_*.json' 2>/dev/null | grep -q . && return 0
  ( cd "$EV" && env CUDA_VISIBLE_DEVICES="$g" HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "${PFX}_NREPEAT=$NREP" "${PFX}_REP_OFFSET=$OFF" EVAL_TEMPERATURE=0.6 EVAL_TOP_P=0.95 \
      $PY -m eval.eval --model vllm --tasks "$B" \
      --model_args "pretrained=$MODEL,tensor_parallel_size=1,max_model_len=32768,dtype=bfloat16,gpu_memory_utilization=0.9" \
      --batch_size auto --log_samples --output_path "$out" ) > "$RES/logs/${NAME}_${B}_off${OFF}.log" 2>&1; }

# simple GPU pool: launch up to one subjob per GPU, refill as jobs finish
declare -A PID; i=0
while [ $i -lt ${#JOBS[@]} ] || [ ${#PID[@]} -gt 0 ]; do
  for g in "${GPUS[@]}"; do
    if [ -n "${PID[$g]:-}" ] && ! kill -0 "${PID[$g]}" 2>/dev/null; then unset "PID[$g]"; fi
    if [ -z "${PID[$g]:-}" ] && [ $i -lt ${#JOBS[@]} ]; then run_job "${JOBS[$i]}" "$g" & PID[$g]=$!; i=$((i+1)); fi
  done; sleep 10
done
echo "done: $NAME (${#JOBS[@]} subjobs) -> $RES"
