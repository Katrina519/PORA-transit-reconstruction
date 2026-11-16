# Current records for 24 paired comparisons per network

The 2026-11-16 final-workbook update uses four Mumford1 initial networks with six paired search seeds each. Main and ablation tables use the same 24 keys. No optimization was executed during this update. Mandl and Mumford0 records are unchanged.

| File | Contents |
|---|---|
| mandl_mumford0_24pairs.csv | 24 pairs per network at the recorded 4-s and 7-s cutoffs |
| main24.csv | 96 endpoint records: 24 pairs, two methods, two endpoints |
| ablation24.csv | 240 endpoint records: 24 pairs, five methods, two endpoints |
| main24_checkpoints.csv | 1968 saved checkpoints: 48 method runs with 41 each |
| statistics24.json / statistics24.csv | Recomputed means, sample SDs, paired outcomes and exact rank probabilities |
| components24.json / overhead24.json / mechanisms24.csv | Component costs, timing and task-history counts |
| source_workbook_checks.json / provenance24.json | Input hash, archive matching and source scope |
| record_checks.json | Portable record-consistency verification |

Run `python code/verify_24pair_records.py` and `python code/recompute_current_statistics.py` from the repository root. Both use the standard library. Plot using `python code/figures/Figure3_current24/draw_figure3.py` with matplotlib, numpy, pandas and PyMuPDF installed.

Mean cost reductions relative to NEA are 3.42% for Mandl, 5.23% for Mumford0, and 13.06% at 120 s / 1.85% after eight rounds for Mumford1. The current Mumford1 histories contain 9600 PORA reconstruction tasks, 9475 feasible proposals and 3430 positive-gain proposals. Selection and feedback account for 2.02% of mean PORA online runtime.

The supplied workbook identifies a 24-pair selection from recorded runs. Its numerical rows match six reference-parent pairs and eighteen pairs from the three additional initial networks. The author describes an independent 24-pair rerun; new logs resolving that provenance have not been provided. Numerical agreement with saved checkpoints establishes record consistency, not a fresh rerun. Exact one-sided signed-rank probabilities condition on these records, with Holm correction for the three ablation contrasts within each endpoint; these corrections do not account for cohort selection or the post-hoc choice of a one-sided alternative. Both endpoints and all configurations are retained. Earlier complete-batch files remain for traceability.
