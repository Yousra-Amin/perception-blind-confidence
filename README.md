# Perception-blind confidence in chart VQA

**Status: preliminary (Day 1–2 baselines).** CSE498R research project, North South University.

## Research question
Do vision-language models know when they are wrong about a chart? The working hypothesis is that a VLM's
confidence reflects how hard the *reasoning* was, but stays high when the model has *misread* the chart
(a perception error), so it is confidently wrong in exactly the way charts make most likely.

## What this repo contains so far
Baseline confidence scores for **Qwen2-VL-7B-Instruct (4-bit)** on a fixed random subset of
**500 ChartQA test questions** (seed 0):

| Score | How it is computed |
|---|---|
| `seq_prob`, `min_prob`, `mean_prob` | answer-token probabilities from greedy decoding |
| `agreement` | fraction of 5 sampled answers (T=0.7) that match the greedy answer |
| `verbal` | the model's own 0–100 confidence for its answer |

Metrics: accuracy (ChartQA relaxed accuracy, 5% numeric tolerance), ECE, AUROC (correct vs wrong), and
risk–coverage curves (AURC).

## Layout
```
src/pbc_day1_day2.py                    # all stages: greedy / sample / verbal / analyze
notebooks/                              # Kaggle notebooks that ran the experiments
outputs/day1-2_chartqa500_qwen2vl7b/    # raw .jsonl outputs, merged.csv, baseline_table.csv, figures
configs/config.yaml                     # run settings (model, n, seed, resolution)
data/                                   # local data only (not committed)
reports/                                # one short report per supervisor update
docs/progress-log.md                    # dated log: what was done, which commit/tag
requirements.txt
```

## Reproduce (Kaggle, GPU T4, Internet on)
```bash
pip install -U transformers accelerate bitsandbytes datasets
python src/pbc_day1_day2.py greedy --n 500     # answers + token log-probs
python src/pbc_day1_day2.py sample --n 500     # sampling agreement
python src/pbc_day1_day2.py verbal --n 500     # verbalized confidence
python src/pbc_day1_day2.py analyze            # ECE, AUROC, risk-coverage, figures
```
Set `PBC_RESULTS=outputs/<run-name>` to choose where outputs go (one folder per run; never overwrite old runs). Every stage resumes from its saved file.
Use the same `--n` in every stage.

## Limitations
500 questions, ChartQA only, one model, and the relaxed-accuracy metric forgives numeric errors under 5%,
which can hide small misreadings. These are first baselines, not conclusions.

## Next
Label errors as perception / reasoning / format and test whether confidence separates perception errors
less well than reasoning errors; then a perception-aware confidence signal.
