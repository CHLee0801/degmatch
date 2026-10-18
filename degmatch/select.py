#!/usr/bin/env python3
"""Write the top-K candidates by a score as an SFT training set.

Usage
  python degmatch/select.py --scores data/scores/aime.npz --k 10000 --pool_dir data/pool_math \
      --out data/sets/degmatch_aime_10k
Writes <out>_text.jsonl ({"text": chat-formatted problem + trace}) and <out>.meta.json.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degmatch.common import load_pool, write_subset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True, help="npz with 'cand' and a score array")
ap.add_argument("--key", default="s", help="score array to rank by (higher is better)")
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--pool_dir", required=True, help="directory with pool_default.jsonl and ntok_text.npy")
ap.add_argument("--out", required=True)
args = ap.parse_args()

S = np.load(args.scores)
sel = S["cand"].astype(int)[np.argsort(-S[args.key])[:args.k]]
pool = load_pool(os.path.join(args.pool_dir, "pool_default.jsonl"))
ntok = np.load(os.path.join(args.pool_dir, "ntok_text.npy"))
write_subset(pool, ntok, sel, args.out, f"DeGMatch top-{args.k} by '{args.key}' from {args.scores}")
