# Anonymous Supplementary Materials

This package contains the data, code, and LaTeX source for the paper
"More Agents, Less Accuracy: When Multi-Agent LLM Voting Hurts on Knowledge-Bound Tasks"
(submitted to NeurIPS 2026 Evaluations and Datasets Track, double-blind).

## Layout

```
analysis/                 — Per-cell stats, K-rubric scores, baseline comparison
scripts/                  — Reproduction scripts (Python 3.11+)
results/merged/           — Raw multi-agent run outputs (8 trustable cells)
paper/                    — LaTeX source for the submission PDF
README.md                 — This file
LICENSE                   — MIT (code) / CC-BY-4.0 (analysis CSVs)
```

## Reproduction quickstart

Requires Python 3.11+, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

1. Install: `pip install pandas numpy scipy scikit-learn matplotlib`
2. Re-derive Section 4 numbers from raw JSONs:
   `python scripts/analyze_curves.py`
3. Re-derive Section 5 (decomposition):
   `python scripts/mechanism_test_correlation.py`
4. Re-derive Section 6 (predictor):
   `python scripts/phase3_full_model.py`
5. Re-derive figures (PDF in figures/):
   `python scripts/plot_curves.py`

## Datasets used

All from public benchmarks:
- MedQA-USMLE (Jin et al., 2021)
- LegalBench privacy-policy + contract-NLI (Guha et al., 2023)
- MMLU formal-logic + philosophy + logical-fallacies (Hendrycks et al., 2021)
- QuALITY (Pang et al., 2022)

We do not redistribute these; the JSON outputs in `results/merged/` contain
only model predictions and our `is_correct` judgments, not the original
benchmark questions.

## Data quality notes

A separate audit found systematic answer-extraction bugs in a Tier-2
extension dataset (math + reasoning + Gemini); those data are NOT included
here and the affected runs are flagged for re-collection in the paper's
Limitations section. The 8 cells in `results/merged/` were sanity-checked
and are the basis of the paper's claims.

## License

- Code (`scripts/`): MIT
- Analysis outputs (`analysis/`): CC-BY-4.0
- LaTeX source: CC-BY-4.0

See LICENSE.
