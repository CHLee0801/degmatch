#!/usr/bin/env python3
"""Set the EOS of a fine-tuned model to <|im_end|> (idempotent).

The base Qwen2.5-Math-7B has eos_token_id 151643 (<|endoftext|>) only. Our training data ends every turn with
<|im_end|> (151645); without this patch, generation does not stop and runs to max_model_len during evaluation.
Usage: python train/fix_eos.py <model_dir> [...]
"""
import json
import os
import sys

EOS_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>
EOS_TOK = "<|im_end|>"


def fix(path):
    if not os.path.exists(os.path.join(path, "config.json")):
        print(f"{path}: no config.json (training not finished?)")
        return False
    for fn in ("config.json", "generation_config.json"):
        p = os.path.join(path, fn)
        if os.path.exists(p):
            d = json.load(open(p))
            if d.get("eos_token_id") != EOS_IDS:
                d["eos_token_id"] = EOS_IDS
                json.dump(d, open(p, "w"), indent=2)
    p = os.path.join(path, "tokenizer_config.json")
    if os.path.exists(p):
        d = json.load(open(p))
        if d.get("eos_token") != EOS_TOK:
            d["eos_token"] = EOS_TOK
            json.dump(d, open(p, "w"), indent=2)
    print(f"{path}: EOS set to {EOS_TOK}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(0 if all([fix(a) for a in sys.argv[1:]]) else 1)
