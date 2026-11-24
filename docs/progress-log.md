# Progress log

One entry per update. Each entry names the commit or tag that holds that state.

## 2026-11-25 — Day 1–2 baselines (tag `v0.1-day2`)
- Built a standalone pipeline (separate from the 499B work): Qwen2-VL-7B-Instruct, 4-bit, T4 GPU.
- Ran 500 fixed-seed ChartQA test questions: greedy answers with token log-probabilities.
- Added sampling agreement (5 samples) and verbalized confidence as baselines.
- Metrics: accuracy, ECE, AUROC, risk–coverage. Outputs in `outputs/day1-2_chartqa500_qwen2vl7b/`.
- Next: label errors (perception / reasoning / format) and test confidence per category.

<!-- Template for the next entry:
## YYYY-MM-DD — <title> (tag `vX.Y-name`)
- What was done
- Key result (one line, with the file that holds it)
- Next
-->
