# DeGMatch: Target-Aware Reasoning Data Selection via Deconfounded Gradient Matching

This is the official github repository for 'DeGMatch: Target-Aware Reasoning Data Selection via Deconfounded Gradient Matching' [[arXiv](https://arxiv.org/abs/2610.24963)].

Citation:
```
@misc{lee2026degmatchtargetawarereasoningdata,
      title={DeGMatch: Target-Aware Reasoning Data Selection via Deconfounded Gradient Matching}, 
      author={Changho Lee and Janghoon Han and Rajdeep Mukherjee and Yeda Song and Lening Nick Cui and Xinya Du and Honglak Lee},
      year={2026},
      eprint={2610.24963},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.24963}, 
}
```


## 0. Install Dependencies
We use two environments: one for selection and training, one for evaluation.
```
conda create -n degmatch python=3.12
conda activate degmatch
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.46.1" "trl==0.12.0" "accelerate==1.0.1" "datasets==3.1.0" "liger-kernel==0.5.10"
pip install numpy scipy pyarrow
```

```
conda create -n degmatch-eval python=3.12
conda activate degmatch-eval
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "vllm==0.10.2" "transformers==4.56.2" "lm_eval==0.4.7" "math-verify==0.9.0"
```

```
git clone https://github.com/CHLee0801/degmatch.git
cd degmatch
bash eval/setup_evalchemy.sh   # Evalchemy at the commit we used + our benchmark tasks
```

## 1. Prepare Data
First, download the candidate pools and build them (one trace per problem).
```
huggingface-cli download open-r1/OpenR1-Math-220k --repo-type dataset --include "data/*" --local-dir raw/openr1
huggingface-cli download open-r1/Mixture-of-Thoughts --repo-type dataset --local-dir raw/mot

python degmatch/build_pool.py --kind math    --raw "raw/openr1/data/train-*.parquet"      --out_dir data/pool_math
python degmatch/build_pool.py --kind science --raw "raw/mot/science/train-*.parquet"      --out_dir data/pool_science
python degmatch/build_pool.py --kind code    --raw "raw/mot/code/train-*.parquet"         --out_dir data/pool_code
```
Second, prepare the target references: a JSONL file per target with `problem` and `trace` (a reasoning trace, about 50 problems per target), e.g. `data/refs/aime_traces.jsonl`.

## 2. Select Data with DeGMatch
Run all steps (relevance gate → student gradients → deconfounded matching → top-K subset) for one target.
```
STUDENT=/path/to/student bash scripts/run_target.sh aime data/refs/aime_traces.jsonl 10000
```
The script header lists the settings for math, science and code targets. Each step can also be run on its own (`degmatch/embed.py`, `gate.py`, `grads.py`, `score.py`, `select.py`).

## 3. Train any LMs in Huggingface
```
BASE_MODEL=/path/to/student bash train/train.sh data/sets/degmatch_aime_10k checkpoints/degmatch_aime_10k
```

## 4. Evaluate any LMs in Huggingface
Choose the benchmarks to evaluate (AIME24, AIME25, AIME26, AMC23, HMMT, MATH500A/B, GAOKAO24/25, GPQADiamond, JEEBench, LiveCodeBench, CodeForces).
```
conda activate degmatch-eval
EVAL_BENCHES="AIME24 AIME25 AIME26 MATH500A MATH500B" bash eval/eval.sh checkpoints/degmatch_aime_10k results
python eval/merge_results.py --results results --model degmatch_aime_10k --benches AIME24 AIME25 AIME26 MATH500
```
