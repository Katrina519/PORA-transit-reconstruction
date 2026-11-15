# PORA: prior-guided online reconstruction allocation

Version v1.0.0 accompanies **Prior-guided online allocation of route-reconstruction tasks for transit network improvement** by Ziming Shu, Xiaoshan Li, Yue Zhang and Yanhong Gu.

## Current manuscript record check - 2026-11-16

The current numerical comparison uses **24 paired runs per network** for Mandl, Mumford0 and Mumford1. Start with [`data/manuscript_24pairs/README.md`](data/manuscript_24pairs/README.md) and run `python code/verify_24pair_records.py`.

The record checks reproduce **3.42%** and **5.23%** mean cost reductions for Mandl and Mumford0, and **13.11% at 120 s / 1.81% after eight rounds** for Mumford1. They verify 1,968 Mumford1 main checkpoints and all 96 main endpoint records; the five-configuration ablation contains 240 endpoint records. Raw logs establishing the author's newly stated independent rerun provenance remain pending; the linked data README explains the scope.

The release page provides an additional `PORA_24pair_record_checks_20261116_v35.zip` asset with the same records and checker. Original release assets and numerical records are retained. The sections below document the original v1.0.0 snapshot; their ten-seed and six-seed summaries are historical, not the current manuscript's numerical tables.

## Original v1.0.0 contents and manuscript mapping

| Location | Contents and manuscript use |
|---|---|
| `data/manuscript/paired_runs.csv` | Frozen Mandl and Mumford0 N24 cohorts; main comparisons and Figure 4 |
| `data/manuscript/plot_data_long.csv` | Paired cost distributions used by Figure 4 |
| `data/manuscript/mumford1_*.json` | Frozen ten-seed main comparison, endpoint metrics and overhead |
| `data/manuscript/*training*.json` | Offline training tasks, preprocessing and frozen scorer parameters |
| `data/mumford1_main/` | All 20 raw main-comparison runs, including saved routes, decisions, feedback and checkpoints |
| `data/mumford1_ablation_all10/` | Complete 50-run, five-configuration, ten-seed ablation archive; both endpoints and original paired tests |
| `data/mumford1_ablation_six/` | Six-seed descriptive summary used in Section 4.4 and its supplied workbook |
| `data/mumford0_archive/` | Unified-online Mumford0 raw run archive and 40-pair records |
| `data/mandl_mumford0_original_archive/` | Original archived 40-pair records for each network |
| `code/figures/Figure3/` | Figure 3 source data and plotting script |
| `code/figures/Figure45/` | Figure 4 and Figure 5 plotting scripts, per-seed saved search paths and endpoint checks |
| `code/archive/work/` | Original project-specific feature, scorer, allocation, training, search-wrapper and analysis source snapshots |
| `provenance/` | Initial Mumford1 context, original-source hashes and required third-party file hashes |
| `figures/` | Final numerical figures for comparison with regenerated output |

## Verify the results without rerunning search

Use Python 3.9 or newer. Install `numpy`, `pandas` and `scipy`, then run from this directory:

```
python verify_release.py
python data/manuscript/reproduce_statistics.py
```

The first command checks every released file against `SHA256SUMS.json`, reconstructs objective components, verifies all 50 ablation runs and their saved endpoints, and checks the six-seed summary against the complete archive. It writes nothing. The second adapts only the output location of the original manuscript-table reproduction script; it writes `verification_output/recomputed_statistics.json` and reproduces the frozen N24 and ten-seed main comparison. Run the hash check before generating new files.

Figure 3 can be redrawn with `python code/figures/Figure3/draw_figure3.py`. Figure 4 and Figure 5 scripts are in `code/figures/Figure45/`; run them in that directory. These scripts write new figure files. Times New Roman (and Arial for Figures 4–5) should be installed to match the archived appearance. Numerical source data do not depend on those fonts.

## Cohorts, endpoints and provenance

Mandl and Mumford0 each report 24 parent-seed pairs (four parent networks and six seeds per parent), drawn from archived 40-pair collections. Their recorded cutoffs are 4 s and 7 s, respectively. The Mumford0 comparison combines matching keys from historical NEA/Count records and later Full/Ridge records, as described in the manuscript.

The Mumford1 main comparison is one fixed initial network with seeds 3011–3020 and two configurations. Its frozen mean reductions remain **13.73% at 120 s** and **2.46% after eight rounds**. The separate five-configuration ablation batch uses the same seed set. Its wall-clock checkpoints belong to that batch and must not replace the frozen main-comparison values.

The Section 4.4 six-seed summary uses seeds **3011, 3015, 3016, 3017, 3019 and 3020**. The supplied workbook records an extraction from the ten-seed ablation workbook; the corresponding numerical rows match the complete archive. This release preserves both the reported six-seed summary and all ten seeds. Separate independent six-seed rerun logs are not included. The six-seed table is descriptive; no ten-seed p-value is attached to it.

The original main comparison and complete ablation protocols used two-sided paired signed-rank tests. The complete ablation archive retains Holm correction over the three prespecified Full-versus-Count/Ridge/NoGain comparisons at each endpoint. All configurations and both endpoints are retained.

## Variables and units

`author` means the NEA baseline; `full` means PORA; `count` balances issued attempts; `ridge` uses a regression-trained frozen prior under the shared controller; `nogain` removes only the empirical mean-gain term while preserving the frozen SPO+ prior and attempt counts. `seed` is the search seed, not a city replicate. `parent` identifies the initial or current parent network as specified by each file.

`cost` is the dimensionless scalar objective, lower is better. `P_minutes` (or `*_P`) is mean passenger travel time including a 5-minute transfer penalty. `O_oneway_minutes` (or `*_O`) is the summed one-way route running time in minutes. The objective is P/66 + O/198 for Mandl, P/52 + O/312 for Mumford0, and P/88 + O/660 for Mumford1. `*_budget` is the saved cost at the network's reporting cutoff; `*_final` is the recorded final cost. `endpoint` is `120s` or `8round`; `last_completed_seconds` and other `*_seconds` fields are elapsed wall-clock seconds. `completed_passes`, counts and cache hits are integers. Positive `full_improvement_percent` means Full has lower cost than its comparator; ratio-of-means percentages and individual-pair percentages should be distinguished.

Raw JSONL contains one complete run per line. `fine_curve` stores completed-pass checkpoints, `fine_solutions` stores saved route lists (zero-based node indices), and `selection_log`/`feedback_log` preserve task-level auditing where enabled. The saved runtime environment was Python 3.9.13, PyTorch 1.12.1+cpu, NumPy 1.22.4 and SciPy 1.7.3, single-threaded. Offline training time is separate from the online run clock.

## Reusing code and rerunning optimization

The portable offline verifier runs directly from this release. Original search and training wrappers are archival source snapshots, retaining their original sibling-directory names and path conventions. They require reconstruction of the original workspace inputs and the original third-party runtime; they are not advertised as a one-command optimization installation. No search is launched by the verification commands above.

Obtain benchmark inputs from Christine Mumford's public UTRP collection: https://users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/Outline.html . Obtain the neural constructor, its environment specification and model download instructions from https://github.com/AHolliday/transit_learning (Holliday et al., 2025, DOI: 10.1080/21680566.2025.2561863). The frozen checkpoint used in this study is identified by its SHA-256 in `provenance/third_party_required_hashes.json`. Benchmark inputs and the pretrained neural checkpoint are not redistributed here. Third-party distribution and licence terms apply.

Source scripts are GPL-3.0 (see `LICENSE`); the authors' original numerical records and frozen scoring parameters are CC BY 4.0 (see `DATA_LICENSE.txt`). Publisher PDFs, third-party article text, credentials, manuscript author contact details and unrelated project files are excluded. This repository provides study outputs and software, not a deployment-ready operational bus plan.
