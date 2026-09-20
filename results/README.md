# Results

The findings live in [`REPORT.md`](../REPORT.md) at the repo root. This
directory holds the artifacts behind it:

| | |
| --- | --- |
| [`decisions.csv`](decisions.csv) | every decision from E1–E3 as one spreadsheet: utterance, gold, prediction, confidence, the model's top-3 options, and before/after columns for the description experiment |
| [`clinc150-inscope/`](clinc150-inscope/) | E1, 150-way classification: full report, charts, `results.json`, held-out gates |
| [`clinc150-oos/`](clinc150-oos/) | E2, 151-way with the `oos` rejection option |
| [`clinc150-gate/`](clinc150-gate/) | E3, the boolean scope gate |
| [`confident-errors.md`](confident-errors.md) | what the model gets wrong while claiming certainty, and whether the dataset is to blame |
| [`confident-errors.json`](confident-errors.json) | the same, machine-readable |
| [`rejector-comparison.json`](rejector-comparison.json) | E2-vs-E3 head-to-head on identical rows |

Everything regenerates from the committed task specs: see
[Reproducing the Jev audit](../README.md#reproducing-the-jev-audit). The raw
per-decision run files (`runs/*.jsonl`, ~155MB with full 150-way distributions
per row) are not committed; `decisions.csv` is the committed record, and the
runs rebuild from the specs for about $5.
