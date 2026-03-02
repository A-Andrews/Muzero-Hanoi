# CCN Paper — Analysis TODO
**Last updated: 2026-03-02**
**Timestamp in use: `1748875208`**

---

## Status overview

| Category | Done | Remaining |
|---|---|---|
| MuZero ablations (all 9 conditions) | ✅ | — |
| MuZero per-difficulty illegal rates | ✅ | — |
| LLM qwen25_7b — LS, MS, ES | ✅ zero_shot + cot | — |
| LLM llama3_8b — LS, MS, ES | ⏳ submitted | jobs 9384915–9384920 |
| Plot: ablation matrix heatmap | ✅ | — |
| Plot: LLM vs MuZero error comparison | ⚠️ Qwen-only | re-run after Llama done |
| Plot: Solve rate comparison | ⚠️ Qwen-only | re-run after Llama done |
| Plot: Illegal rate comparison | ⚠️ Qwen-only | re-run after Llama done |
| Plotting scripts: multi-model support | ✅ | — |

**Total remaining:** wait for 6 Llama jobs, then run 3 plot commands.

---

## Next step — Wait for Llama jobs, then plot

Monitor: `squeue -u $(whoami)`

Once all 6 jobs finish, run:

```bash
cd /well/costa/users/zqa082/Muzero-Hanoi
source .venv/bin/activate

# Figure 1: Error comparison
python3 llm_eval/plot_llm_comparison.py \
    --timestamp 1748875208 \
    --model_labels qwen25_7b llama3_8b \
    --llm_display_names "Qwen 7B (ZS)" "Qwen 7B (CoT)" "Llama 8B (ZS)" "Llama 8B (CoT)" \
    --no_latex

# Figure 2: Solve rate (mirrors Goel & Grafman 1995: controls 83.9% vs PFC patients 51.1%)
python3 llm_eval/plot_solve_rate.py \
    --timestamp 1748875208 \
    --model_labels qwen25_7b llama3_8b \
    --llm_display_names "Qwen 7B (ZS)" "Qwen 7B (CoT)" "Llama 8B (ZS)" "Llama 8B (CoT)" \
    --no_latex

# Figure 3: Illegal rate (mirrors Grafman et al. 1992: cerebellar patients F(1,19)=4.23, p<0.05)
python3 llm_eval/plot_illegal_comparison.py \
    --timestamp 1748875208 \
    --model_labels qwen25_7b llama3_8b \
    --llm_display_names "Qwen 7B (ZS)" "Qwen 7B (CoT)" "Llama 8B (ZS)" "Llama 8B (CoT)" \
    --no_latex
```

Outputs → `stats/Hanoi/1748875208/`:
- `LLM_MuZero_ErrorComparison_1748875208.png`
- `LLM_MuZero_IllegalRate_1748875208.png`
- `SolveRate_Comparison_1748875208.png`
- `IllegalRate_Comparison_1748875208.png`

> You can test with Qwen-only now (`--model_labels qwen25_7b`) while jobs run.

---

## What each figure shows

Each figure has 3 panels (Close / Moderate / Far) with 7 bars:
```
MuZero | Value abl. (PFC) | Policy abl. (Cerebellar) | Qwen(ZS) | Qwen(CoT) | Llama(ZS) | Llama(CoT)
```

**Key expected findings:**
- Error/solve rate: MuZero solves all difficulties; LLMs fail catastrophically on Far (7-move)
- LLM difficulty-dependent failure profile resembles PFC lesion more than cerebellar
- Illegal rate: Policy ablation spikes (matching Grafman 1992 cerebellar finding); LLMs high on MS

---

## Dropped from scope (keep for future work)

- Layer ablation sweep — not directly comparable to human data; save for supplementary
- Noise injection sweep — same reasoning
- LLM eval for layer/noise sweeps — blocked on layer sweep decision

---

## Notes

- Always run plotting scripts with `--no_latex` on the login node.
- All 3 plotting scripts now use `--model_labels` (plural, nargs="+").
- `plot_illegal_comparison.py` uses per-difficulty `muzero_illegal_rates.json` automatically (already exists for all 3 difficulties).
