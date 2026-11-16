# Current 24-pair experimental records

The manuscript evaluates 24 paired experiments on each of Mandl, Mumford0 and Mumford1. Mumford1 uses four initial networks, with six search seeds per initial network:

| Initial network | Search seeds |
|---|---|
| mumford1_reference_lineplan | 3011, 3013, 3015, 3016, 3019, 3020 |
| mumford1_new_01 | 3111, 3114, 3115, 3116, 3117, 3119 |
| mumford1_new_02 | 3121, 3123, 3125, 3127, 3129, 3130 |
| mumford1_new_03 | 3132, 3133, 3135, 3138, 3139, 3140 |

Main and ablation analyses use these same 24 parent–seed keys. Each record is identified by its original batch, parent, seed and method, so endpoint timings remain linked to the execution that produced them.

| File | Contents |
|---|---|
| mandl_mumford0_24pairs.csv | Unchanged Mandl and Mumford0 records, 24 pairs per network |
| main24.csv | 96 endpoint rows: 24 pairs × two methods × two endpoints |
| ablation24.csv | 240 endpoint rows: 24 pairs × five methods × two endpoints |
| pair_keys24.csv | Four initial networks, six seeds each |
| run_sources.csv | One row per physical run file, with method, roles, original source path and SHA-256 hashes |
| run_records.zip | Original run-record bytes and three original configuration records |
| run_source_checks.json | Verification of all 336 endpoint-to-run mappings |
| main24_checkpoints.csv | 1968 main-comparison checkpoints |
| statistics24.json / statistics24.csv | Means, sample SDs, paired outcomes and exact rank probabilities |
| components24.json / overhead24.json / mechanisms24.csv | Objective components, timing and task-history summaries |
| source_workbook_checks.json / provenance24.json | Workbook hash, source annotations and original batch mapping |

The source archive contains 132 distinct method-run files: 48 used in the main comparison and 120 used in ablations, with 36 files shared by the two analyses. These are method-run files, not 132 seed pairs. Records with a different original batch remain separate even when their costs agree. No new optimization was run while assembling this archive.

From the repository root run `python code/verify_run_sources.py`, `python code/verify_24pair_records.py`, and `python code/recompute_current_statistics.py`. All three use the Python standard library. The first checks original bytes, configurations, starting routes and saved endpoint values; the others check tabular consistency and reproduce the statistics. These checks do not repeat network search or independently re-evaluate every route cost.

Mean cost reductions relative to NEA are 3.42% for Mandl, 5.23% for Mumford0, and 13.06% at 120 s / 1.85% after eight rounds for Mumford1. The Mumford1 histories contain 9600 PORA reconstruction tasks, 9475 feasible proposals and 3430 positive-gain proposals. Allocation overhead is 2.02% of mean PORA online runtime.

This directional alternative addresses the research hypothesis that Full reduces cost relative to NEA and the ablated policies. The one-sided alternative was chosen after inspecting the results; the reported p-values do not adjust for this choice. Holm correction covers Count, Ridge and NoGain contrasts separately at each endpoint. Original source annotations are retained in provenance24.json, including the workbook's selection wording; a match to archived records establishes source identity, not the process used to choose the reporting pairs.
