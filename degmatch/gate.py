#!/usr/bin/env python3
"""Relevance gate: per-reference round-robin over Qwen3 problem embeddings.

Each target reference contributes its nearest pool problems in turn (cosine similarity) until N
unique candidates are collected. The paper uses N = 30,000 for math targets, 15,000 for GPQA/JEE
and 9,000 for LiveCodeBench/CodeForces. Only gated candidates need gradients.

Usage
  python degmatch/gate.py --pool_emb "data/pool_math/emb/pool_problem.part*.npy" \
      --ref_emb data/pool_math/emb/anchor_aime.npy --n 30000 --out data/gates/aime_gate30k_idx.npy
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degmatch.common import load_npy_parts, normalize, round_robin  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool_emb", required=True, help="pool embedding .npy or glob of .part*.npy shards")
ap.add_argument("--ref_emb", required=True, help="target-reference embeddings .npy")
ap.add_argument("--n", type=int, default=30000)
ap.add_argument("--out", required=True)
args = ap.parse_args()

pool = normalize(load_npy_parts(args.pool_emb))
refs = normalize(np.load(args.ref_emb).astype(np.float32))
picked = round_robin(pool @ refs.T, args.n)
gate = np.array(sorted(picked.tolist()))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.save(args.out, gate)
print(f"gate: {len(gate)} candidates from {len(pool)} (references: {len(refs)})", flush=True)
