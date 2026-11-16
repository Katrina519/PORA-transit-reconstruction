"""Read-only checks of the supplied 24-pair records; no search or provenance inference."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import math
import hashlib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "manuscript_24pairs"

def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def key(row):
    return row["parent"], row["seed"]

def verify():
    small = read("mandl_mumford0_24pairs.csv")
    main = read("main24.csv")
    ablation = read("ablation24.csv")
    checkpoints = read("main24_checkpoints.csv")
    mechanisms = read("mechanisms24.csv")
    means = {}
    for city in ("mandl", "mumford0"):
        rows = [r for r in small if r["city"] == city]
        assert len(rows) == len({key(r) for r in rows}) == 24
        assert len({r["parent"] for r in rows}) == 4
        assert set(Counter(r["parent"] for r in rows).values()) == {6}
        pscale, oscale = (66, 198) if city == "mandl" else (52, 312)
        for method in ("author", "full", "count", "ridge"):
            for r in rows:
                cost = float(r[method + "_budget"])
                reconstructed = float(r[method + "_P"]) / pscale + float(r[method + "_O"]) / oscale
                assert math.isfinite(cost) and abs(cost - reconstructed) < 1e-6
        nea = sum(float(r["author_budget"]) for r in rows) / 24
        full = sum(float(r["full_budget"]) for r in rows) / 24
        means[city] = {"nea": nea, "full": full, "mean_reduction_percent": 100 * (nea - full) / nea}
    pair_keys = {key(r) for r in main}
    assert len(pair_keys) == 24
    assert set(Counter(parent for parent, seed in pair_keys).values()) == {6}
    assert len({parent for parent, seed in pair_keys}) == 4
    assert len(main) == len({(*key(r), r["method"], r["endpoint"]) for r in main}) == 96
    assert len(ablation) == len({(*key(r), r["method"], r["endpoint"]) for r in ablation}) == 240
    for records, methods in [(main, ("author", "full")), (ablation, ("author", "full", "count", "ridge", "nogain"))]:
        for method in methods:
            for endpoint in ("8round", "120s"):
                rows = [r for r in records if r["method"] == method and r["endpoint"] == endpoint]
                assert len(rows) == 24 and {key(r) for r in rows} == pair_keys
                for r in rows:
                    assert abs(float(r["cost"]) - float(r["P_minutes"]) / 88 - float(r["O_oneway_minutes"]) / 660) < 1e-6
    groups = defaultdict(list)
    for r in checkpoints:
        groups[(*key(r), r["method"])].append(r)
    assert len(checkpoints) == 1968 and len(groups) == 48
    for rows in groups.values():
        assert len(rows) == 41
        assert sorted((int(r["iteration"]), int(r["pass_index"])) for r in rows) == [(0, 0)] + [(i, j) for i in range(1, 9) for j in range(1, 6)]
        ordered = sorted(rows, key=lambda r: (int(r["iteration"]), int(r["pass_index"])))
        assert all(float(b["seconds"]) >= float(a["seconds"]) for a, b in zip(ordered, ordered[1:]))
    discrepancies = []
    for r in main:
        rows = groups[(*key(r), r["method"])]
        eligible = rows if r["endpoint"] == "8round" else [q for q in rows if float(q["seconds"]) <= 120]
        saved = max(eligible, key=lambda q: float(q["seconds"]))
        discrepancies.append(abs(float(r["cost"]) - float(saved["cost"])))
    assert max(discrepancies) < 1e-12
    assert len(mechanisms) == 24 and {key(r) for r in mechanisms} == pair_keys
    feasible = int(sum(float(r["feasible_proposals"]) for r in mechanisms))
    positive = int(sum(float(r["positive_proposals"]) for r in mechanisms))
    overhead = json.loads((DATA / "overhead24.json").read_text(encoding="utf-8"))
    assert feasible == overhead["feasible_proposals"] == 9475
    assert positive == overhead["positive_proposals"] == 3430
    assert overhead["executed_neural_tasks"] == 24 * 400 == 9600
    means["mumford1"] = {}
    for endpoint in ("8round", "120s"):
        values = {m: sum(float(r["cost"]) for r in main if r["method"] == m and r["endpoint"] == endpoint) / 24 for m in ("author", "full")}
        means["mumford1"][endpoint] = {**values, "mean_reduction_percent": 100 * (values["author"] - values["full"]) / values["author"]}
    filenames = ["mandl_mumford0_24pairs.csv", "main24.csv", "ablation24.csv", "main24_checkpoints.csv", "mechanisms24.csv", "overhead24.json"]
    return {
        "status": "PASS_RECORD_CONSISTENCY", "pairs_per_network": {"mandl": 24, "mumford0": 24, "mumford1": 24},
        "mumford1_main_method_runs": 48, "mumford1_main_checkpoint_records": 1968,
        "checkpoints_per_main_run": 41, "main_endpoint_records_checked": 96,
        "maximum_endpoint_checkpoint_cost_difference": max(discrepancies),
        "ablation_endpoint_records": 240, "ablation_methods": 5, "endpoints": 2,
        "pora_reconstruction_tasks_recorded": 9600, "feasible_proposals": feasible, "positive_proposals": positive,
        "method_means": means,
        "scope": "Saved-record, key, component, count and checkpoint checks; no new optimization or independent route-cost replay.",
        "run_source_mapping": "See run_sources.csv and run the separate code/verify_run_sources.py check.",
        "input_sha256": {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in filenames}
    }

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
