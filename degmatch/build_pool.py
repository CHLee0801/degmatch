#!/usr/bin/env python3
"""Build a candidate pool jsonl from the raw parquet files, plus per-example token counts.

Pools used in the paper
  math     : open-r1/OpenR1-Math-220k, `default` split (93,733 problems, one verified R1 trace each)
  science  : open-r1/Mixture-of-Thoughts, science split, one random trace per problem (seed 0) -> 53,195
  code     : open-r1/Mixture-of-Thoughts, code split,    one random trace per problem (seed 0) -> 26,061

Output schema (row i has idx == i): idx, uuid, problem, answer, problem_type, source, trace
plus <out_dir>/ntok_text.npy = token count of the full training text (chat format) under the student tokenizer.

Usage
  python degmatch/build_pool.py --kind math    --raw "raw/openr1/data/train-*.parquet" --out_dir data/pool_math
  python degmatch/build_pool.py --kind science --raw "raw/mot/science_train-*.parquet"  --out_dir data/pool_science
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degmatch.common import chat_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["math", "science", "code"], required=True)
ap.add_argument("--raw", required=True, help="glob of raw parquet files")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Math-7B", help="student tokenizer for token counts")
ap.add_argument("--seed", type=int, default=0, help="seed for picking one trace per problem (Mixture-of-Thoughts)")
args = ap.parse_args()

files = sorted(glob.glob(args.raw))
assert files, f"no parquet files match {args.raw}"
os.makedirs(args.out_dir, exist_ok=True)
rows = []
if args.kind == "math":
    cols = ["problem", "answer", "problem_type", "source", "uuid", "messages"]
    for f in files:
        t = pq.read_table(f, columns=cols).to_pydict()
        for i in range(len(t["problem"])):
            trace = next((m["content"] for m in (t["messages"][i] or []) if m.get("role") == "assistant"), None)
            if t["problem"][i] and trace:
                rows.append(dict(uuid=t["uuid"][i], problem=t["problem"][i], answer=t["answer"][i],
                                 problem_type=t["problem_type"][i], source=t["source"][i], trace=trace))
else:
    # Mixture-of-Thoughts has several traces per problem; keep one random trace per problem.
    groups = {}
    for f in files:
        t = pq.read_table(f, columns=["messages", "source"]).to_pydict()
        for msgs, src in zip(t["messages"], t["source"]):
            user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
            if user and asst:
                groups.setdefault(user, []).append((asst, src))
    rng = np.random.default_rng(args.seed)
    for user, items in groups.items():  # insertion order = first appearance
        asst, src = items[int(rng.integers(len(items)))]
        rows.append(dict(uuid=hashlib.md5(user.encode()).hexdigest(), problem=user, answer=None,
                         problem_type=None, source=src, trace=asst))

with open(os.path.join(args.out_dir, "pool_default.jsonl"), "w") as fo:
    for i, r in enumerate(rows):
        fo.write(json.dumps({"idx": i, **r}, ensure_ascii=False) + "\n")
print(f"{args.out_dir}/pool_default.jsonl: {len(rows)} rows", flush=True)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.tokenizer)
ntok = np.array([len(tok(chat_text(r["problem"], r["trace"]), add_special_tokens=False)["input_ids"]) for r in rows], np.int64)
np.save(os.path.join(args.out_dir, "ntok_text.npy"), ntok)
print(f"token counts saved (median {np.median(ntok):.0f})", flush=True)
