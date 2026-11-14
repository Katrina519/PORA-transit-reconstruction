"""Read-only verification of archived PORA outputs; does not run search."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf-8'))
for name, expected in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name

cohort = pd.read_csv(ROOT/'data/manuscript/paired_runs.csv')
assert len(cohort) == 48
for city, group in cohort.groupby('city'):
    assert len(group) == 24 and group.parent.nunique() == 4
    assert group.groupby('parent').size().eq(6).all()
    p, o = (66,198) if city == 'mandl' else (52,312)
    for method in ['author','full','count','ridge']:
        assert np.max(abs(group[method+'_budget']-group[method+'_P']/p-group[method+'_O']/o)) < 1e-6

ablation = pd.read_csv(ROOT/'data/mumford1_ablation_all10/per_run_endpoints.csv')
assert len(ablation) == 100
assert not ablation[['seed','method','endpoint']].duplicated().any()
assert ablation.groupby(['method','endpoint']).size().eq(10).all()
assert np.max(abs(ablation.cost-ablation.P_minutes/88-ablation.O_oneway_minutes/660)) < 1e-6
seeds = [3011,3015,3016,3017,3019,3020]
six = ablation[ablation.seed.isin(seeds)]
summary = pd.read_csv(ROOT/'data/mumford1_ablation_six/summary.csv')
for row in summary.itertuples():
    values = six[(six.endpoint == row.endpoint)&(six.method == row.method)]
    full = six[(six.endpoint == row.endpoint)&(six.method == 'full')].cost.mean()
    assert len(values) == 6
    assert abs(values.cost.mean()-row.mean_cost) < 1e-12
    assert abs(100*(values.cost.mean()-full)/values.cost.mean()-row.full_reduction_percent) < 1e-9

for directory, count in [('mumford1_main',20),('mumford1_ablation_all10',50)]:
    runs = [json.loads(line) for line in (ROOT/'data'/directory/'raw_runs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(runs) == count
    for run in runs:
        assert run['iterations'] == 8 and len(run['fine_curve']) == 41
        assert run['generation_counts'] == [80,800]
        for point in run['fine_curve']:
            assert 0 <= point['solution_index'] < len(run['fine_solutions'])
        if directory == 'mumford1_ablation_all10':
            for endpoint in ['8round','120s']:
                row = ablation[(ablation.seed == run['seed'])&(ablation.method == run['variant'])&(ablation.endpoint == endpoint)].iloc[0]
                value = run['cost'] if endpoint == '8round' else [p for p in run['fine_curve'] if p['seconds'] <= 120][-1]['cost']
                assert abs(value-row.cost) < 1e-12
print('PASS: file hashes, 48 N24 pairs, 70 saved Mumford1 runs, all ten-seed ablation endpoints and six-seed descriptive means.')
