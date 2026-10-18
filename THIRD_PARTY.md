# Third-party code and data

| Component | Source | License | How it is used |
|---|---|---|---|
| `train/sft.py`, `train/fsdp_config_qwen.json` | s1, github.com/simplescaling/s1 (`train/sft.py`) | Apache-2.0 | adapted (local jsonl, Liger kernel, resume) |
| `eval/tasks/*` | Evalchemy, github.com/mlfoundations/evalchemy at commit cc756119 | see upstream | benchmark tasks copied and modified (repeats/offsets for data-parallel runs, new tasks AIME26/HMMT/GAOKAO24/25, grading and loading fixes) |
| `eval/evalchemy_eval_py.patch` | Evalchemy `eval/eval.py` | see upstream | small robustness patch |
| `eval/tasks/GAOKAO24/gaokao24_eval.json`, `GAOKAO25/gaokao25_eval.json` | Gaokao 2024/2025 math problems | public exam problems | evaluation data |
| pools | open-r1/OpenR1-Math-220k, open-r1/Mixture-of-Thoughts (Hugging Face) | Apache-2.0 | downloaded by the user, processed by `degmatch/build_pool.py` |
