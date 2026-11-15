# Records for 24 paired comparisons per network

This directory collects the numerical records used by the current manuscript: 24 parent-seed pairs for each of Mandl, Mumford0 and Mumford1. Existing numerical files are copied unchanged; no optimization was run for this update.

| File | Role |
|---|---|
| `mandl_mumford0_24pairs.csv` | 24 pairs per network, with the recorded 4-s Mandl and 7-s Mumford0 endpoints |
| `main24.csv` | Mumford1: 96 endpoint records, 24 pairs x 2 methods x 2 endpoints |
| `ablation24.csv` | Mumford1: 240 endpoint records, 24 pairs x 5 configurations x 2 endpoints |
| `main24_checkpoints.csv` | 1,968 main-comparison checkpoints, 48 method runs x 41 checkpoints |
| `mechanisms24.csv`, `overhead24.json` | Allocation counts and runtime summaries |
| `record_checks.json` | Recomputed record-consistency checks and input hashes |
| `provenance24.json` | Source mapping and unresolved rerun-provenance status |
| Other files | Earlier complete-batch comparisons, inputs, components, and statistics retained for traceability |

From the repository root, run:

```sh
python code/verify_24pair_records.py
```

The verifier uses only the Python standard library. It checks pair keys, method/endpoint coverage, objective components, checkpoint order and endpoint costs. All 96 Mumford1 main endpoint costs match the corresponding completed checkpoints within 1e-12; the maximum difference is 6.661338147750939e-16. The mechanism summaries contain 9,600 PORA reconstruction tasks, 9,463 feasible proposals and 3,385 positive-gain proposals.

Mean cost reductions relative to NEA reproduce 3.42% on Mandl, 5.23% on Mumford0, and 13.11% at 120 s / 1.81% after eight rounds on Mumford1. Main and ablation timing records remain separate.

These checks establish consistency of the supplied records, not independently rerun provenance or a new route-cost replay. The supplied Mumford1 workbook still maps to eight historical and sixteen additional pairs, as recorded in `provenance24.json`. Fresh independent 24-pair logs for the three benchmarks have been requested and remain pending. Complete-batch comparisons and source records remain available; the one-sided probabilities for the supplied selected cohort are nominal and do not adjust for selection. Manuscript working drafts that assume fresh reruns require this issue to be resolved before submission.
