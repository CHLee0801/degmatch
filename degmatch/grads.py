#!/usr/bin/env python3
"""Per-example full-parameter gradients of the student, compressed with CountSketch.

For each (problem, trace) we take the token-mean cross-entropy over the trace tokens (prompt masked),
back-propagate through all parameters, and project the gradient to D dimensions with a CountSketch whose
hash/sign for parameter tensor k is regenerated from seed SEED*1000+k. Candidates and references must use
the same model, D and seed so that their sketches are comparable.

Usage
  # gated candidates, 8 shards (one per GPU)
  CUDA_VISIBLE_DEVICES=0 python degmatch/grads.py candidates --pool data/pool_math/pool_default.jsonl \
      --idx data/gates/aime_gate30k_idx.npy --out_dir data/grads --prefix math --shard 0 --nshard 8
  # target references (jsonl with problem + trace)
  python degmatch/grads.py references --refs data/refs/aime_traces.jsonl --out data/grads/ref_aime.npy

Outputs
  candidates: <out_dir>/<prefix>_<shard>.npy (rows), <prefix>_idx_<shard>.npy (pool indices), <prefix>_loss_<shard>.npy
  references: <out>.npy (rows in file order; all-zero rows mark skipped examples)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degmatch.common import PROMPT_PREFIX, PROMPT_SUFFIX, TURN_END  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["candidates", "references"])
ap.add_argument("--model", default="Qwen/Qwen2.5-Math-7B", help="student model (paper: Qwen2.5-Math-7B with RoPE base 300k)")
ap.add_argument("--dim", type=int, default=16384)
ap.add_argument("--max_len", type=int, default=32768)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--pool", help="candidates: pool jsonl")
ap.add_argument("--idx", help="candidates: .npy of pool indices to process (e.g. the gate)")
ap.add_argument("--out_dir", help="candidates: output directory")
ap.add_argument("--prefix", default="pool", help="candidates: output file prefix")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--refs", help="references: jsonl with problem and trace")
ap.add_argument("--out", help="references: output .npy")
args = ap.parse_args()

DEV = "cuda:0"
CHUNK = 2048  # logits are computed in chunks so a 32k x 152k logit matrix is never materialized

tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=DEV)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.train()
params = [p for _, p in model.named_parameters() if p.requires_grad]
gen = torch.Generator(device=DEV)


def chunk_ce(hs, labels):
    return torch.nn.functional.cross_entropy(model.lm_head(hs).float(), labels, ignore_index=-100, reduction="sum")


def countsketch():
    out = torch.zeros(args.dim, device=DEV, dtype=torch.float32)
    for k, p in enumerate(params):
        if p.grad is None:
            continue
        g = p.grad.reshape(-1).float()
        gen.manual_seed(args.seed * 1000 + k)
        idx = torch.randint(0, args.dim, (g.numel(),), generator=gen, device=DEV)
        sign = torch.randint(0, 2, (g.numel(),), generator=gen, device=DEV, dtype=torch.int8).float() * 2 - 1
        out.index_add_(0, idx, g * sign)
        del idx, sign, g
    return out


def sketch_gradient(problem, trace):
    p_ids = tok(PROMPT_PREFIX + problem + PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]
    t_ids = tok(trace + TURN_END, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[:args.max_len]
    if len(ids) <= len(p_ids) + 1:
        return None, None, None
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    x = torch.tensor([ids], device=DEV)
    lab = torch.tensor(labels[1:] + [-100], device=DEV)  # next-token targets
    model.zero_grad(set_to_none=True)
    h = model.model(input_ids=x).last_hidden_state[0]
    ntok = int((lab != -100).sum())
    loss = 0.0
    for s in range(0, h.shape[0], CHUNK):
        hs, ls = h[s:s + CHUNK], lab[s:s + CHUNK]
        if int((ls != -100).sum()):
            loss = loss + torch.utils.checkpoint.checkpoint(chunk_ce, hs, ls, use_reentrant=False)
    loss = loss / ntok  # token-mean loss over the trace
    loss.backward()
    v = countsketch()
    model.zero_grad(set_to_none=True)
    return v.cpu().numpy(), float(loss), ntok


t0 = time.time()
if args.mode == "references":
    rows = [json.loads(line) for line in open(args.refs) if line.strip()]
    out = np.zeros((len(rows), args.dim), np.float32)
    for i, r in enumerate(rows):
        v, _, _ = sketch_gradient(r["problem"], r["trace"])
        if v is not None:
            out[i] = v
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, out)
    print(f"saved {args.out} ({len(rows)} references, {time.time() - t0:.0f}s)", flush=True)
else:
    pool = [json.loads(line) for line in open(args.pool)]
    idx = np.load(args.idx)[args.shard::args.nshard]
    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, f"{args.prefix}_{args.shard}.npy")
    loss_dst = os.path.join(args.out_dir, f"{args.prefix}_loss_{args.shard}.npy")
    np.save(os.path.join(args.out_dir, f"{args.prefix}_idx_{args.shard}.npy"), idx)
    res = np.zeros((len(idx), args.dim), np.float32)
    loss = np.zeros((len(idx), 2), np.float32)
    start = 0
    if os.path.exists(dst):  # resume
        prev = np.load(dst)
        if prev.shape == res.shape:
            res, loss = prev, np.load(loss_dst)
            done = np.nonzero(np.linalg.norm(res, axis=1))[0]
            start = int(done[-1]) + 1 if len(done) else 0
    for k in range(start, len(idx)):
        r = pool[int(idx[k])]
        try:
            v, l, nt = sketch_gradient(r["problem"], r.get("trace") or "")
            if v is not None:
                res[k], loss[k] = v, (l, nt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            print(f"  OOM at pool index {idx[k]} (left as zero)", flush=True)
        if k % 50 == 0:
            np.save(dst, res)
            np.save(loss_dst, loss)
            print(f"  shard {args.shard}: {k}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    np.save(dst, res)
    np.save(loss_dst, loss)
    print(f"saved {dst} ({len(idx)} candidates, {time.time() - t0:.0f}s)", flush=True)
