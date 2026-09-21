# Results

The findings live in [`REPORT.md`](../REPORT.md) at the repo root. This
directory holds the artifacts behind it:

| | |
| --- | --- |
| [`decisions.csv`](decisions.csv) | every decision from E1–E3 as one spreadsheet: utterance, gold, prediction, confidence, the model's top-3 options, and before/after columns for the description experiment |
| [`clinc150-inscope/`](clinc150-inscope/) | E1, 150-way classification: full report, charts, `results.json`, held-out gates |
| [`clinc150-oos/`](clinc150-oos/) | E2, 151-way with the `oos` rejection option |
| [`clinc150-gate/`](clinc150-gate/) | E3, the boolean scope gate, one-sentence scope |
| [`clinc150-gate-rich/`](clinc150-gate-rich/) | E3b, the same gate with the scope enumerated |
| [`clinc150-described-full/`](clinc150-described-full/) | E1c, all 150 classes described by the error-blind rule |
| [`confident-errors.md`](confident-errors.md) | what the model gets wrong while claiming certainty, and whether the dataset is to blame |
| [`label-errata.md`](label-errata.md) | the 12 CLINC150 rows whose gold label fails against the dataset's own usage, with evidence ([jsonl](label-errata.jsonl)) |
| [`confident-errors.json`](confident-errors.json) | the same, machine-readable |
| [`rejector-comparison.json`](rejector-comparison.json) | out-of-scope detectors head-to-head on identical rows |
| [`description-effect.json`](description-effect.json) | bare names vs described classes, targeted and clean |
| [`determinism.json`](determinism.json) | same request repeated: stable when confident, not when torn |

Everything regenerates from the committed task specs: see
[Reproducing the Jev audit](../README.md#reproducing-the-jev-audit). The raw
per-decision run files (`runs/*.jsonl`, ~155MB with full 150-way distributions
per row) are not committed; `decisions.csv` is the committed record, and the
runs rebuild from the specs for about $5.
