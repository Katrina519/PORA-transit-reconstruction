# PORA: prior-guided online reconstruction allocation

Code and experimental records for **Prior-guided online allocation of route-reconstruction tasks for transit network improvement** by Ziming Shu, Xiaoshan Li, Yue Zhang and Yanhong Gu.

## Current manuscript data

The current evaluation contains **24 paired experiments per network** for Mandl, Mumford0 and Mumford1. **Mumford1 uses four initial networks and six paired search seeds per initial network.** Methods within each comparison share the initial network and search seed. Both the eight-round endpoint and the completed incumbent at 120 s are retained.

Start with [the current data README](data/manuscript_24pairs/README.md). The [run-source index](data/manuscript_24pairs/run_sources.csv) connects every Mumford1 parent–seed–method record to its original run file and configuration. The [raw-record archive](data/manuscript_24pairs/run_records.zip) contains the indexed records and configurations; hashes and endpoint checks are provided alongside it.

| Comparison with NEA | Mean cost reduction |
|---|---:|
| Mandl, recorded 4-s cutoff | 3.42% |
| Mumford0, recorded 7-s cutoff | 5.23% |
| Mumford1, 120 s | 13.06% |
| Mumford1, eight rounds | 1.85% |

## Verify records and reproduce statistics

From the repository root, using Python 3.9 or newer:

```sh
python code/verify_run_sources.py
python code/verify_24pair_records.py
python code/recompute_current_statistics.py
```

These standard-library commands verify original run/configuration hashes, four initial-network identities, the 24 pair keys, all 96 main and 240 ablation endpoint records, objective components, and paired statistics. They do not launch optimization. The checks establish correspondence to the published run records. Original batch identifiers and source-workbook annotations are retained in [provenance24.json](data/manuscript_24pairs/provenance24.json).

This directional alternative addresses the research hypothesis that Full reduces cost relative to NEA and the ablated policies. The one-sided alternative was chosen after inspecting the results; the reported p-values do not adjust for this choice. Holm correction covers the three ablation contrasts within each endpoint. Both endpoints and all five configurations are available.

## Figures and rerunning search

Generate the current Figure 3 with `python code/figures/Figure3_current24/draw_figure3.py` after installing numpy, pandas, matplotlib and PyMuPDF. It uses the saved checkpoints and retains the 120–240 s inset. The current source and output are separate from earlier figure versions.

The original search wrappers are in `code/archive/work/`. Their environment, third-party model inputs and original path conventions are documented in the [original release instructions](ORIGINAL_V1_SNAPSHOT.md#reusing-code-and-rerunning-optimization). Statistical reproduction and full optimization reruns are separate workflows.

## Downloads and earlier versions

The [v1.0.0 release page](https://github.com/Katrina519/PORA-transit-reconstruction/releases/tag/v1.0.0) provides **PORA_24pair_sources_20261116_v41.zip**, containing the current data, source mapping and verifiers. Earlier assets remain available for traceability. Earlier ten-seed and six-seed summaries are documented only in [the original snapshot](ORIGINAL_V1_SNAPSHOT.md); they are not the current manuscript tables.

Code is GPL-3.0; original study data are CC BY 4.0. See LICENSE and DATA_LICENSE.txt. Third-party model and benchmark rights remain with their respective owners.
