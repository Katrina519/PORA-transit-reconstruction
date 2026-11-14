"""独立重算保存方案，配对比较冻结A/B及作者。"""
import json
import csv
import statistics
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    req = read(ROOT / '请求冻结.json'); done = read(ROOT / 'comparison/COMPLETE.json')
    runs = done['runs']
    assert all(sha(p) == s for p, s in done['input_hashes'].items())
    raw = [json.loads(s) for s in (ROOT / 'comparison/原始运行.jsonl').read_text(encoding='utf-8').splitlines()]
    assert raw == runs
    keys = [(r['parent'], r['seed'], r['method']) for r in runs]
    expected = {(c['parent'], s, m) for c in req['contexts'] for s in req['seeds'] for m in req['methods']}
    assert len(keys) == len(set(keys)) == req['planned_runs'] and set(keys) == expected
    spec = importlib.util.spec_from_file_location('independent_ab_audit', ROOT.parent / 'nea_spo_v2_joint_20261109_v1/audit.py')
    independent = importlib.util.module_from_spec(spec); spec.loader.exec_module(independent)
    rows, uncovered = [], []
    maximum = 0.; decisions = 0; references = 0
    for r in runs:
        assert r['target_samples'] == 811 and r['target_calls'] == 82 and len(r['fine_curve']) == 41
        assert r['fallbacks'] == 0
        decisions += independent.check_choices(r, 12)
        for solution in r['fine_solutions']:
            _, _, cost = independent.evaluate('mumford0', solution['routes'])
            maximum = max(maximum, abs(cost-solution['cost'])); references += 1
        previous_time = -1.
        for event in r['fine_curve']:
            solution = r['fine_solutions'][event['solution_index']]
            assert previous_time <= event['seconds'] and solution['first_seconds'] <= event['seconds']
            assert solution['cost'] == event['cost']; previous_time = event['seconds']
        if r['fine_curve'][-1]['seconds'] < req['primary_seconds']:
            uncovered.append((r['parent'], r['seed'], r['method']))
        for scope, seconds in [('primary', req['primary_seconds']), ('fixed_8', None)]:
            event = r['fine_curve'][-1] if seconds is None else next(
                (e for e in reversed(r['fine_curve']) if e['seconds'] <= seconds), None)
            assert event is not None
            p, o, cost = independent.evaluate('mumford0', r['fine_solutions'][event['solution_index']]['routes'])
            assert abs(cost-event['cost']) < 1e-6
            rows.append({'parent': r['parent'], 'seed': r['seed'], 'method': r['method'], 'scope': scope,
                'cost': event['cost'], 'P_minutes': p, 'O_single_minutes': o,
                'event_seconds': event['seconds'], 'online_8round_seconds': r['seconds'],
                'selection_8round_seconds': r['selection_seconds'], 'configuration_seconds': r['configuration_seconds']})
    assert maximum < 1e-6
    results = {}
    for scope in ['primary', 'fixed_8']:
        if scope == 'primary' and uncovered:
            results[scope] = {'status': 'insufficient_time_coverage', 'uncovered': uncovered}
            continue
        part = [r for r in rows if r['scope'] == scope]
        means = {m: {k: statistics.mean(r[k] for r in part if r['method'] == m) for k in
            ['cost', 'P_minutes', 'O_single_minutes', 'online_8round_seconds', 'selection_8round_seconds', 'configuration_seconds']}
            for m in req['methods']}
        for m in means:
            means[m]['cost_sd'] = statistics.stdev(r['cost'] for r in part if r['method'] == m)
        index = {(r['parent'], r['seed'], r['method']): r for r in part}
        comparisons = {}
        for base, target in [('author', 'A_spo'), ('author', 'B_spo'), ('A_spo', 'B_spo')]:
            pairs, parents = [], []
            for context in req['contexts']:
                parent = context['parent']
                for seed in req['seeds']:
                    a = index[parent, seed, base]; b = index[parent, seed, target]
                    pairs.append({'parent': parent, 'seed': seed, 'base_cost': a['cost'], 'target_cost': b['cost'],
                        'delta': a['cost']-b['cost'], 'improvement_percent': 100*(a['cost']-b['cost'])/a['cost'],
                        'P_seconds_delta': 60*(b['P_minutes']-a['P_minutes']),
                        'O_minutes_delta': b['O_single_minutes']-a['O_single_minutes']})
                a = statistics.mean(index[parent, s, base]['cost'] for s in req['seeds'])
                b = statistics.mean(index[parent, s, target]['cost'] for s in req['seeds'])
                parents.append({'parent': parent, 'base_mean': a, 'target_mean': b,
                    'improvement_percent': 100*(a-b)/a})
            a, b = means[base]['cost'], means[target]['cost']
            comparisons[target + '_vs_' + base] = {'improvement_percent': 100*(a-b)/a,
                'wins': sum(p['delta'] > 1e-6 for p in pairs), 'ties': sum(abs(p['delta']) <= 1e-6 for p in pairs),
                'losses': sum(p['delta'] < -1e-6 for p in pairs), 'pairs': pairs, 'parents': parents,
                'gross_loss_per_pair': statistics.mean(max(-p['delta'], 0.) for p in pairs),
                'gross_gain_per_pair': statistics.mean(max(p['delta'], 0.) for p in pairs)}
        results[scope] = {'means': means, 'comparisons': comparisons}
    out = {'runs': len(runs), 'paired_triples': len(runs)//3, 'parent_clusters': len(req['contexts']),
        'max_cost_error': maximum, 'saved_solution_references': references, 'unique_saved_solutions': len(independent.CACHE),
        'cost_samples': sum(r['target_samples'] for r in runs), 'selection_events': decisions,
        'input_hashes_checked': len(done['input_hashes']), 'all_inputs_unchanged': True, 'results': results,
        'wall_seconds': done['wall_seconds'], 'import_seconds': done['import_seconds'], 'raw_io_seconds': done['raw_io_seconds'],
        'configuration_seconds': sum(c['seconds'] for c in done['method_configuration']),
        'scope': 'four new parent clusters, three seeds each; A/B frozen whole-model comparison, no tuning or fusion'}
    with (ROOT / '对照统计.json').open('x', encoding='utf-8') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
    with (ROOT / '逐次结果.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'runs': len(runs), 'max_cost_error': maximum, 'input_hashes_checked': len(done['input_hashes']),
        'summary': {scope: {name: {k: d[k] for k in ['improvement_percent', 'wins', 'ties', 'losses']}
            for name, d in value.get('comparisons', {}).items()} for scope, value in results.items()}}, indent=2))


if __name__ == '__main__':
    main()
