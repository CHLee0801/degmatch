#!/bin/bash
# Clone Evalchemy at the commit we used, add our benchmark tasks, and apply our small eval.py patch.
#   bash eval/setup_evalchemy.sh [target_dir=third_party/evalchemy]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=${1:-third_party/evalchemy}
COMMIT=cc756119b7ad9bf8414a4e83413c1d3c9146927e
[ -d "$DST" ] || git clone https://github.com/mlfoundations/evalchemy.git "$DST"
cd "$DST" && git checkout -q $COMMIT
# Tasks added or modified for this paper (AIME26, HMMT, GAOKAO24/25, MATH500 halves for data-parallel runs,
# repeat/offset support, LiveCodeBench loading without a loading script, GPQA option shuffling fix, ...)
for t in "$HERE"/tasks/*/; do rsync -a --exclude __pycache__ "$t" "eval/chat_benchmarks/$(basename "$t")/"; done
cp "$HERE/tasks/aime_equiv.py" eval/chat_benchmarks/
git apply "$HERE/evalchemy_eval_py.patch"
echo "evalchemy ready at $DST"
