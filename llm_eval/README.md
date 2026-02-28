# LLM evaluation for Muzero-Hanoi

This directory contains everything needed to evaluate an instruction-tuned LLM
on the same Close / Moderate / Far Tower of Hanoi problem sets used in the
MuZero ablation experiments, and to plot the results on the same axes.

---

## Files

| File | Purpose |
|------|---------|
| `prompts.py` | State-to-text rendering; zero-shot and CoT prompt builders |
| `llm_hanoi_eval.py` | Main evaluation script (HuggingFace pipeline) |
| `plot_llm_comparison.py` | Bar + line overlay plots alongside MuZero baselines |
| `run_llm_eval.sh` | SLURM GPU job for a single condition |
| `run_all_llm_eval.sh` | Submit all 6 conditions (3 difficulties × 2 prompting strategies) at once |
| `llm_requirements.txt` | Extra Python dependencies (`transformers`, `accelerate`, etc.) |

---

## Setup (one-time)

```bash
source .venv/bin/activate
pip install -r llm_eval/llm_requirements.txt
```

If using a gated model (e.g. Llama-3), authenticate once on a login node:
```bash
huggingface-cli login
```

---

## Running evaluations

### Submit all 6 SLURM jobs at once

```bash
cd /well/costa/users/zqa082/Muzero-Hanoi

bash llm_eval/run_all_llm_eval.sh \
    --timestamp 1748875208 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model_label llama3_8b \
    --episodes 50 \
    --temperature 0.7
```

This submits one GPU job per condition:

| Job | Difficulty | Prompting |
|-----|-----------|-----------|
| `llm_llama3_8b_s0_zero_shot` | Far (ES, 7 moves) | zero-shot |
| `llm_llama3_8b_s0_cot` | Far (ES, 7 moves) | CoT |
| `llm_llama3_8b_s1_zero_shot` | Moderate (MS, 3 moves) | zero-shot |
| `llm_llama3_8b_s1_cot` | Moderate (MS, 3 moves) | CoT |
| `llm_llama3_8b_s2_zero_shot` | Close (LS, 1 move) | zero-shot |
| `llm_llama3_8b_s2_cot` | Close (LS, 1 move) | CoT |

Monitor with: `squeue -u $(whoami)`

### Or run a single condition manually

```bash
sbatch llm_eval/run_llm_eval.sh \
    --timestamp 1748875208 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --model_label llama3_8b \
    --prompting zero_shot \
    --start 0 \
    --episodes 50
```

`--start` values: `0` = Far (ES), `1` = Moderate (MS), `2` = Close (LS)

---

## Generating plots

Once all jobs have finished:

```bash
python3 llm_eval/plot_llm_comparison.py \
    --timestamp 1748875208 \
    --model_label llama3_8b \
    --llm_display_names "Llama-3 8B (zero-shot)" "Llama-3 8B (CoT)"
```

Three figures are saved under `stats/Hanoi/1748875208/`:

| File | Content |
|------|---------|
| `LLM_MuZero_ErrorComparison_*.png` | Mean excess moves per difficulty — same metric as MuZero bar charts |
| `LLM_MuZero_IllegalRate_*.png` | Illegal move rate (%) per difficulty |
| `LLM_MuZero_LinePlot_*.png` | MuZero error curves + LLM as horizontal dashed bands |

---

## Output data format

Results are saved alongside the existing MuZero files in
`stats/Hanoi/<timestamp>/<difficulty>/`:

| File | Shape | Contents |
|------|-------|---------|
| `LLM_{label}_{prompting}_actingAccuracy.pt` | `(1, 2)` | `[[0, mean_error]]` — compatible with `load_accuracy()` in the existing plotting code |
| `LLM_{label}_{prompting}_actingAccuracy_error.pt` | `(1, 3)` | `[[0, mean_error, se_error]]` |
| `LLM_{label}_{prompting}_results.json` | — | Full data: `mean_error`, `solve_rate`, `mean_illegal_rate`, per-episode records |

The `.pt` files use a dummy n_simulations value of 0 so that the existing
`load_accuracy()` function and bar-chart code can read them without modification.

---

## Key CLI arguments for `llm_hanoi_eval.py`

| Argument | Default | Notes |
|----------|---------|-------|
| `--timestamp` | required | Must match the MuZero model directory |
| `--model` | required | HuggingFace model ID or local path |
| `--model_label` | required | Short label used in output filenames (e.g. `llama3_8b`) |
| `--prompting` | `zero_shot` | `zero_shot` or `cot` |
| `--start` | `None` | `0`=Far, `1`=Moderate, `2`=Close; omit for random states |
| `--episodes` | `100` | Number of episodes per condition |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--max_new_tokens` | `300` | Tokens per step (300 gives room for CoT reasoning) |
| `--dtype` | `bfloat16` | `bfloat16`, `float16`, or `float32` |
| `--model_cache_dir` | `None` | HuggingFace cache dir (e.g. `/well/costa/users/zqa082/hf_cache`) |
| `--history_length` | `0` | Number of previous moves to include in prompt |
| `--seed` | `42` | Random seed for fallback actions |

---

## Practical notes

- **Model recommendation**: `meta-llama/Llama-3.1-8B-Instruct` (~16 GB bfloat16,
  fits on 1× A100). Alternatives: `Qwen/Qwen2.5-7B-Instruct` (no gated access
  needed), `mistralai/Mistral-7B-Instruct-v0.3`.
- **CUDA module**: `run_llm_eval.sh` loads `CUDA/12.1.1` — check with
  `module avail CUDA` and edit the script if your cluster version differs.
- **Compute estimate**: ~50 episodes × ~20 steps × ~0.5 s/step ≈ 8–10 min
  per condition; all 6 conditions run in parallel on the cluster.
- **If VRAM is tight**: add `bitsandbytes` 8-bit quantisation by passing
  `load_in_8bit=True` to `from_pretrained` in `llm_hanoi_eval.py:load_pipeline`.
