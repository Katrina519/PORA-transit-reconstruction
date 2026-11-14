"""Recompute draft tables from copied saved records; does not run transit search."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
TOL = 1e-6
frozen = pd.read_csv(ROOT / 'paired_runs.csv')
rows = pd.DataFrame(json.loads((ROOT / 'mumford1_rows.json').read_text()))
saved = json.loads((ROOT / 'mumford1_summary.json').read_text())
result = {'frozen_cohorts': [], 'mumford1': []}

def comparison(a, b):
    d = np.asarray(a) - np.asarray(b)
    return dict(gain_percent=100 * (np.mean(a) - np.mean(b)) / np.mean(a),
                wins=int((d > TOL).sum()), ties=int((abs(d) <= TOL).sum()),
                losses=int((d < -TOL).sum()))

for city, g in frozen.groupby('city', sort=False):
    assert len(g) == 24 and g['parent'].nunique() == 4
    assert g.groupby('parent').size().eq(6).all()
    assert not g[['parent', 'seed']].duplicated().any()
    denominator_P, denominator_O = (66, 198) if city == 'mandl' else (52, 312)
    for method in ['author', 'count', 'ridge', 'full']:
        cost = g[method + '_budget']
        reconstructed = g[method + '_P'] / denominator_P + g[method + '_O'] / denominator_O
        assert np.max(abs(cost - reconstructed)) < TOL
        result['frozen_cohorts'].append(dict(
            city=city, method=method, n=len(g), mean=float(cost.mean()),
            sd=float(cost.std(ddof=1)), passenger_minutes=float(g[method + '_P'].mean()),
            route_minutes=float(g[method + '_O'].mean()),
            **comparison(g['author_budget'], cost)))

for endpoint in ['8round', '120s']:
    g = rows[rows.endpoint == endpoint]
    a = g[g.method == 'author'].set_index('seed').sort_index()
    b = g[g.method == 'full'].set_index('seed').sort_index()
    assert len(a) == len(b) == 10 and a.index.equals(b.index)
    assert a.index.tolist() == list(range(3011, 3021))
    for x in [a, b]:
        assert np.max(abs(x.cost - x.P_minutes / 88 - x.O_oneway_minutes / 660)) < TOL
    test = wilcoxon(a.cost, b.cost, alternative='two-sided', method='exact')
    z = dict(endpoint=endpoint, n=10, author_mean=float(a.cost.mean()),
             full_mean=float(b.cost.mean()), author_sd=float(a.cost.std(ddof=1)),
             full_sd=float(b.cost.std(ddof=1)), p_two_sided=float(test.pvalue),
             author_P=float(a.P_minutes.mean()), full_P=float(b.P_minutes.mean()),
             author_O=float(a.O_oneway_minutes.mean()), full_O=float(b.O_oneway_minutes.mean()),
             **comparison(a.cost, b.cost))
    prior = next(x for x in saved if x['endpoint'] == endpoint)
    for key in ['author_mean', 'full_mean', 'author_sd', 'full_sd', 'p_two_sided']:
        assert abs(z[key] - prior[key]) < 1e-10, (key, z[key], prior[key])
    result['mumford1'].append(z)

full = rows[(rows.method == 'full') & (rows.endpoint == '8round')]
author = rows[(rows.method == 'author') & (rows.endpoint == '8round')]
result['overhead'] = dict(
    author_seconds=float(author.total_online_seconds.mean()),
    full_seconds=float(full.total_online_seconds.mean()),
    selection_seconds=float(full.selection_seconds.mean()),
    feedback_seconds=float(full.feedback_seconds.mean()),
    allocation_share_percent=float(100 * (full.selection_seconds.mean() + full.feedback_seconds.mean()) / full.total_online_seconds.mean()),
    cached_share_percent=float(100 * full.cache_hits.sum() / (full.score_calls.sum() + full.cache_hits.sum())))
output = ROOT.parent.parent / 'verification_output'
output.mkdir(exist_ok=True)
(output / 'recomputed_statistics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print('PASS: 48 frozen pairs, 10 Mumford1 pairs, objective components, paired tests and overhead reproduced.')
