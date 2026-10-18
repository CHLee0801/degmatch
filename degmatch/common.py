"""Shared helpers: chat formatting, pool I/O, training-set writing, round-robin selection."""
import glob
import json
import os

import numpy as np

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PROMPT_PREFIX = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n"
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TURN_END = "<|im_end|>"


def chat_text(problem, trace):
    """Training text in the Qwen chat format used for every selected subset."""
    return PROMPT_PREFIX + problem + PROMPT_SUFFIX + trace + TURN_END


def load_pool(path):
    """Pool jsonl rows ({idx, problem, trace, source, ...}); row i must have idx == i."""
    return [json.loads(line) for line in open(path)]


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_npy_parts(pattern):
    """Load a single .npy or concatenate shards named like *.part0.npy, *.part1.npy, ..."""
    files = sorted(glob.glob(pattern), key=lambda f: int(f.rsplit("part", 1)[-1][:-4]) if "part" in f else 0)
    assert files, f"no files match {pattern}"
    return np.concatenate([np.load(f) for f in files]).astype(np.float32)


def round_robin(sim, n):
    """Per-reference round-robin: in each round, every reference adds its highest-ranked
    candidate not yet selected. sim: (num_candidates, num_references). Returns indices in pick order."""
    order = np.argsort(-sim, axis=0)
    picked, seen, depth = [], set(), 0
    while len(picked) < n and depth < sim.shape[0]:
        for a in range(sim.shape[1]):
            j = int(order[depth, a])
            if j not in seen:
                seen.add(j)
                picked.append(j)
            if len(picked) >= n:
                break
        depth += 1
    return np.array(picked)


def load_candidate_grads(grad_dir, prefixes, want_idx, dim):
    """Gather per-example gradients written by grads.py (<prefix>_<shard>.npy + <prefix>_idx_<shard>.npy)
    for the pool indices in want_idx. prefixes: one prefix or a list. Rows without a gradient stay zero."""
    pos = {int(i): k for k, i in enumerate(want_idx)}
    out = np.zeros((len(want_idx), dim), np.float32)
    prefixes = [prefixes] if isinstance(prefixes, str) else list(prefixes)
    files = sorted(f for p in prefixes for f in glob.glob(os.path.join(grad_dir, f"{p}_idx_*.npy")))
    assert files, f"no gradient shards for {prefixes} in {grad_dir}"
    for f in files:
        idx = np.load(f)
        g = np.load(f.replace("_idx_", "_"), mmap_mode="r")
        for k, i in enumerate(idx):
            q = pos.get(int(i))
            if q is not None and np.any(g[k]):
                out[q] = g[k]
    return out


def write_subset(pool, ntok, idx, out_prefix, method):
    """Write <out_prefix>_text.jsonl (one {"text": ...} per example) and <out_prefix>.meta.json."""
    idx = [int(i) for i in idx]
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    with open(f"{out_prefix}_text.jsonl", "w") as fo:
        for i in idx:
            r = pool[i]
            fo.write(json.dumps({"text": chat_text(r["problem"], r["trace"])}, ensure_ascii=False) + "\n")
    meta = dict(name=os.path.basename(out_prefix), n=len(idx), method=method, idx=idx)
    if ntok is not None:
        meta.update(total_tokens=int(ntok[idx].sum()), median_tokens=float(np.median(ntok[idx])))
    json.dump(meta, open(f"{out_prefix}.meta.json", "w"))
    tokens = f", {meta['total_tokens'] / 1e6:.1f}M tokens" if ntok is not None else ""
    print(f"{out_prefix}: n={len(idx)}{tokens}", flush=True)
