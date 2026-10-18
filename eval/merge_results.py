#!/usr/bin/env python3
"""Merge data-parallel subjob results into one score per benchmark (mean over all samples, in %).

  AIME24/25/26, AMC23 : answers are re-graded with aime_equiv ('016' == '16')
  MATH500             : MATH500A + MATH500B re-graded with math_verify, weighted by problem count
  GAOKAO24/25         : official score; --gaokao_fix also applies five answer-format rules (e.g. 45 degrees == pi/4)
  others              : the task's accuracy_avg

Usage: python eval/merge_results.py --results results --model <model_name> --benches AIME24 AIME25 MATH500 GPQADiamond
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.set_int_max_str_digits(0)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks"))
from aime_equiv import aime_equiv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--model", required=True)
ap.add_argument("--benches", nargs="+", required=True)
ap.add_argument("--gaokao_fix", action="store_true")
args = ap.parse_args()


def latest(d):
    js = sorted(glob.glob(f"{d}/**/results_*.json", recursive=True), key=os.path.getmtime)
    return json.load(open(js[-1])) if js else None


def task_block(res, bench):
    r = res.get("results", {})
    for k, v in r.items():
        if bench[:5].upper() in k.upper() and isinstance(v, dict) and "accuracy_avg" in v:
            return v
    return next((v for v in r.values() if isinstance(v, dict) and "accuracy_avg" in v), None)


def math_eq(g, p):
    from math_verify import parse, verify
    if g is None or p is None:
        return False
    g, p = str(g).strip(), str(p).strip()
    if g and g == p:
        return True
    for a, b in ((g, p), (f"${g}$", f"${p}$")):
        try:
            if verify(parse(a), parse(b)):
                return True
        except Exception:
            pass
    return False


_norm = lambda s: re.sub(r"\\(text|mathrm)\{([^}]*)\}", r"\2", s or "").replace(" ", "").replace("\\left", "").replace("\\right", "").lower()
GAOKAO_RULES = {  # gold answer -> extra accepted forms
    "\\{x | -1 < x < 3\\}": lambda p: _norm(p) in ("(-1,3)", "\\{x|-1<x<3\\}", "\\{x\\mid-1<x<3\\}"),
    "\\{x | 1 < x < 2\\}": lambda p: _norm(p) in ("(1,2)", "\\{x|1<x<2\\}", "\\{x\\mid1<x<2\\}"),
    "Strictly decreasing": lambda p: _norm(p) in ("strictlydecreasing", "decreasing", "monotonicallydecreasing", "strictlydecreasing."),
    "Range: 10.15, Median: 210.015": lambda p: "10.15" in _norm(p) and ("210.015" in _norm(p) or "210.02" in _norm(p)),
    "\\frac{\\pi}{4}": lambda p: _norm(p).replace("°", "^\\circ") in ("45^\\circ", "\\frac{\\pi}{4}", "\\dfrac{\\pi}{4}", "\\pi/4"),
}


def score(bench):
    if bench == "MATH500":
        tot, n = 0.0, 0
        for half in ("MATH500A", "MATH500B"):
            for d in glob.glob(f"{args.results}/{args.model}_{half}_off*"):
                res = latest(d)
                ex = (task_block(res, half) or {}).get("examples", []) if res else []
                for e in ex:
                    preds = e.get("model_answers") or [e.get("model_answer")]
                    tot += sum(math_eq(e.get("expected_answer") or e.get("answer"), p) for p in preds) / len(preds)
                    n += 1
        return 100 * tot / n if n else None
    vals = []
    for d in sorted(glob.glob(f"{args.results}/{args.model}_{bench}_off*")):
        res = latest(d)
        v = task_block(res, bench) if res else None
        if v is None:
            continue
        if bench in ("AIME24", "AIME25", "AIME26", "AMC23") and "examples" in v:
            vals += [aime_equiv(str(e.get("expected_answer", e.get("answer"))), str(a)) for e in v["examples"] for a in e["model_answers"]]
        elif bench.startswith("GAOKAO") and args.gaokao_fix and "examples" in v:
            for e in v["examples"]:
                gold, rule = str(e.get("answer", e.get("expected_answer"))), None
                rule = GAOKAO_RULES.get(gold)
                labels = e.get("label", e.get("correct"))
                vals += [bool(c) or (rule is not None and rule(a)) for a, c in zip(e["model_answers"], labels)]
        else:
            vals += [v["accuracy_avg"]] * int(v.get("num_repeat") or 1)
    return 100 * float(np.mean(vals)) if vals else None


for b in args.benches:
    s = score(b)
    print(f"{args.model} {b}: " + (f"{s:.1f}" if s is not None else "no results"))
