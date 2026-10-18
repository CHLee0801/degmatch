#!/usr/bin/env python3
"""Problem-text embeddings with Qwen3-Embedding-8B (used only for the relevance gate).

Recipe: last-token (EOS) pooling with left padding, max length 2048, no instruction prefix.
Shard over GPUs with --shard/--nshard; each shard writes <out>.part<k>.npy.

Usage
  # candidate pool, 8 shards (one per GPU)
  CUDA_VISIBLE_DEVICES=0 python degmatch/embed.py --input data/pool_math/pool_default.jsonl \
      --out data/pool_math/emb/pool_problem --shard 0 --nshard 8
  # target references (small; one process)
  python degmatch/embed.py --input data/refs/aime_traces.jsonl --out data/pool_math/emb/anchor_aime
"""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="jsonl with a 'problem' field")
ap.add_argument("--out", required=True, help="output path without .npy")
ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
ap.add_argument("--max_len", type=int, default=2048)
ap.add_argument("--max_batch_tokens", type=int, default=24576)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
args = ap.parse_args()

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")  # left padding -> last position is EOS
model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16 if dev != "cpu" else torch.float32).to(dev).eval()

texts = [json.loads(line)["problem"] for line in open(args.input) if line.strip()]
lo, hi = len(texts) * args.shard // args.nshard, len(texts) * (args.shard + 1) // args.nshard
texts = texts[lo:hi]
lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
order = np.argsort(lens)
out = np.zeros((len(texts), model.config.hidden_size), np.float32)


@torch.no_grad()
def encode(batch):
    b = tok(batch, return_tensors="pt", truncation=True, max_length=args.max_len, padding=True).to(dev)
    return model(**b).last_hidden_state[:, -1].float().cpu().numpy()


i = 0
while i < len(texts):
    L = min(max(lens[order[i]], 1), args.max_len)
    bs = max(1, min(32, args.max_batch_tokens // L))
    sel = order[i:i + bs]
    out[sel] = encode([texts[j] for j in sel])
    i += bs
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
dst = f"{args.out}.npy" if args.nshard == 1 else f"{args.out}.part{args.shard}.npy"
np.save(dst, out)
print(f"saved {dst} {out.shape}", flush=True)
