#!/usr/bin/env python3
"""DeGMatch scoring: remove the shared gradient structure, then match to the target references.

  1. Center candidate and reference gradients at the gate mean mu.
  2. Remove the top-r principal components of the gated candidates (exact eigendecomposition of the
     D x D covariance; r = 50 in the paper):  g' = (I - U U^T)(g - mu).
  3. Direction d(x)   = mean of the top-5 cosine similarities between g'(x) and the references.
     Magnitude m(x)   = norm of the projection of g'(x) onto the span of the references.
     Score   s(x)     = d(x) * m(x).

Usage
  python degmatch/score.py --gate data/gates/aime_gate30k_idx.npy --grad_dir data/grads --prefix math \
      --refs data/grads/ref_aime.npy --out data/scores/aime.npz
Output npz: cand (pool indices), s, d, m, mu, pcs
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from degmatch.common import load_candidate_grads, normalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gate", required=True, help=".npy of gated pool indices")
ap.add_argument("--grad_dir", required=True)
ap.add_argument("--prefix", required=True, nargs="+", help="prefix(es) used when computing candidate gradients")
ap.add_argument("--refs", required=True, help="reference gradients .npy")
ap.add_argument("--n_pc", type=int, default=50)
ap.add_argument("--top", type=int, default=5, help="number of nearest references averaged in d")
ap.add_argument("--out", required=True)
args = ap.parse_args()

gate = np.load(args.gate).astype(int)
R = np.load(args.refs)
R = R[np.abs(R).sum(1) > 0]
dim = R.shape[1]
X = load_candidate_grads(args.grad_dir, args.prefix, gate, dim)
print(f"candidates {len(gate)} (with gradient {int((np.abs(X).sum(1) > 0).sum())}), references {len(R)}", flush=True)

mu = X.mean(0, dtype=np.float64).astype(np.float32)
C = np.zeros((dim, dim), np.float64)
for s0 in range(0, len(X), 6000):
    Z = X[s0:s0 + 6000] - mu
    C += (Z.T @ Z).astype(np.float64)
_, V = eigh(C / len(X), subset_by_index=[dim - args.n_pc, dim - 1], driver="evr")
U = V[:, ::-1].T.astype(np.float32)  # (n_pc, dim), largest first
del C


def deconfound(Y):
    Yc = Y - mu
    return Yc - (Yc @ U.T) @ U


Xp, Rp = deconfound(X), deconfound(R)
d = np.sort(normalize(Xp) @ normalize(Rp).T, axis=1)[:, -args.top:].mean(1)
Q, _ = np.linalg.qr(Rp.T)
m = np.linalg.norm(Xp @ Q, axis=1)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, cand=gate, s=d * m, d=d, m=m, mu=mu, pcs=U)
print(f"saved {args.out}", flush=True)
