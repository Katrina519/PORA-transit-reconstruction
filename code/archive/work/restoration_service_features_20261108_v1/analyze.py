"""固定开发数据，分别回放绕行余量与服务效率损失。"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    with (ROOT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    native = load('native_replay', WORK / 'restoration_native_transfer_20261105_v1/analyze.py')
    selector = native.b.a.s
    np, c = native.np, native.c
    source = WORK / 'restoration_native_transfer_20261105_v1'
    batches = [WORK / x for x in ('restoration_first_line_risk_20261106_v1',
                                 'restoration_rule_comparison_20261107_v1')]
    frozen_path = WORK / 'restoration_two_sided_validation_20261106_v1/部署冻结.json'
    paths = [Path(__file__), ROOT / '预先协议.md', source / '统一任务池.json',
             source / '父网与特征.json', source / 'COMPLETE.json', frozen_path,
             Path(selector.__file__), Path(native.__file__), Path(c.__file__),
             c.INPUT / 'MandlTravelTimes.txt', c.INPUT / 'MandlDemand.txt']
    paths += [p / name for p in batches for name in ('部署冻结.json', 'COMPLETE.json')]
    for folder in [source] + batches:
        for name, expected in read(folder / '交付文件哈希.json').items():
            assert sha(folder / name) == expected
    hashes = {str(p): sha(p) for p in paths}
    dump('INPUT_RECEIPT.json', {'hashes': hashes, 'new_generator_calls': 0,
                               'planned_fits': 36, 'scope': 'development_and_posthoc_replay'})
    tasks, parent_data, prior = (read(source / n) for n in
                               ('统一任务池.json', '父网与特征.json', 'COMPLETE.json'))
    frozen = read(frozen_path)
    lookup = {t['key']: t for t in tasks}
    assert len(tasks) == 79 and sum(len(t['rows']) for t in tasks) == 460
    roads = np.loadtxt(c.INPUT / 'MandlTravelTimes.txt')
    q = np.loadtxt(c.INPUT / 'MandlDemand.txt')
    Q = float(q.sum())
    mask = (q > 0) & ~np.eye(len(q), dtype=bool)
    assert np.all(np.diag(q) == 0)
    road = roads.copy()
    for k in range(len(roads)):
        road = np.minimum(road, road[:, k, None] + road[None, k, :])
    contexts = []
    for folder in batches:
        f, results = read(folder / '部署冻结.json'), read(folder / 'COMPLETE.json')
        for ctx, summary in zip(f['contexts'], results['summaries']):
            assert ctx['parent'] == summary['parent']
            contexts.append((ctx, summary, folder.name))
    parents = dict(parent_data['routes'])
    parents.update({ctx['parent']: ctx['routes'] for ctx, _, _ in contexts})
    feature_map = {}
    max_distance_error = max_sum_error = 0.
    tick = time.perf_counter()
    for state, routes in parents.items():
        D = c.unlimited(c.direct(roads, routes))
        max_distance_error = max(max_distance_error, c.compare(D, native.identity_distances(roads, routes)))
        assert np.all(D[mask] >= road[mask] - 1e-10)
        for j in range(6):
            remaining = routes[:j] + routes[j + 1:]
            residual = c.unlimited(c.direct(roads, remaining))
            max_distance_error = max(max_distance_error, c.compare(residual, native.identity_distances(roads, remaining)))
            affected = mask & ~np.isfinite(residual)
            H = float(np.sum(q[affected] * (D[affected] - road[affected])) / Q)
            K = float(np.sum(q[mask] * (1. / D[mask] - 1. / residual[mask])) / Q)
            hsum = sum(float(q[u, v] * (D[u, v] - road[u, v])) for u, v in zip(*np.where(affected))) / Q
            ksum = sum(float(q[u, v] * (1. / D[u, v] - 1. / residual[u, v])) for u, v in zip(*np.where(mask))) / Q
            max_sum_error = max(max_sum_error, abs(H - hsum), abs(K - ksum))
            assert H >= -1e-12 and K >= -1e-12
            feature_map[state + '|' + str(j)] = {'H_minutes': H, 'K_inverse_minutes': K,
                'H': H / 33., 'K': K * 33., 'affected_demand_fraction': float(q[affected].sum() / Q)}
    feature_seconds = time.perf_counter() - tick
    max_fit_error = 0.

    def fit(train_keys, targets, variant):
        nonlocal max_fit_error
        X = [lookup[k]['x'] + ([feature_map[k][variant]] if variant != 'original' else []) for k in train_keys]
        model = selector.fit(X, targets, list(range(len(X[0]))))
        Z = np.column_stack([np.ones(len(X)), (np.asarray(X) - model['mean']) / model['scale']])
        regularizer = np.eye(Z.shape[1]); regularizer[0, 0] = 0.
        independent = np.linalg.lstsq(np.vstack([Z, regularizer]),
                                    np.concatenate([targets, np.zeros(Z.shape[1])]), rcond=None)[0]
        max_fit_error = max(max_fit_error, float(np.max(np.abs(independent - model['coef']))))
        return model

    def score(model, state, X, variant):
        data = [x + ([feature_map[state + '|' + str(j)][variant]] if variant != 'original' else [])
                for j, x in enumerate(X)]
        return selector.scores(model, data)

    variants = ['original', 'H', 'K']
    development = []
    for fold, old_summary in zip(prior['folds'], prior['summary']):
        state = fold['held_ancestor']; keys = fold['original']['train_keys']
        assert all(state not in lookup[k]['ancestors'] for k in keys)
        group = [lookup[state + '|' + str(j)] for j in range(6)]
        record = {'parent': state, 'methods': {}, 'models': {}}
        for variant in variants:
            model = fit(keys, fold['original']['targets'], variant)
            scores = score(model, state, [t['x'] for t in group], variant)
            chosen = selector.choose(scores)
            record['models'][variant] = model
            record['methods'][variant] = {'choice': chosen + 1, 'gain': group[chosen]['value']}
            if variant == 'original':
                assert chosen == old_summary['choices']['original']
                assert abs(group[chosen]['value'] - old_summary['methods']['original']['gain']) < 1e-12
        development.append(record)
    models = {v: fit(frozen['train_keys'], frozen['targets'], v) for v in variants}
    assert np.max(np.abs(np.asarray(models['original']['coef']) - frozen['original_model']['coef'])) < 1e-12
    diagnostic = []
    for ctx, summary, batch in contexts:
        state = ctx['parent']; allowed = [j for j, x in enumerate(ctx['structural_details']) if x['count'] > 0]
        assert allowed
        gains = [t['gain'] for t in summary['tasks']]
        record = {'parent': state, 'batch': batch, 'allowed': [j + 1 for j in allowed],
                  'gains': gains, 'features': [feature_map[state + '|' + str(j)] for j in range(6)],
                  'best_allowed_gain': max(gains[j] for j in allowed), 'methods': {}}
        for variant in variants:
            values = score(models[variant], state, ctx['features'], variant)
            rank = min(allowed, key=lambda j: (-values[j], j))
            guard = 0 if 0 in allowed else rank
            all_pick = selector.choose(values)
            record['methods'][variant] = {'scores': values, 'rank_choice': rank + 1,
                'guard_choice': guard + 1, 'all_choice': all_pick + 1,
                'rank_gain': gains[rank], 'guard_gain': gains[guard], 'all_gain': gains[all_pick]}
            if variant == 'original':
                assert np.max(np.abs(np.asarray(values) - ctx['original_scores'])) < 1e-12
                assert guard == ctx['choices']['guarded']
                assert abs(gains[guard] - summary['methods']['guarded']['gain']) < 1e-12
        diagnostic.append(record)

    means = {v: {mode: float(np.mean([r['methods'][v][mode + '_gain'] for r in diagnostic]))
                 for mode in ('rank', 'guard', 'all')} for v in variants}
    comparisons = {}
    for v in ('H', 'K'):
        comparisons[v] = {}
        for mode in ('rank', 'guard', 'all'):
            differences = [r['methods'][v][mode + '_gain'] - r['methods']['original'][mode + '_gain'] for r in diagnostic]
            comparisons[v][mode] = {'better': sum(d > 1e-12 for d in differences),
                'worse': sum(d < -1e-12 for d in differences), 'same': sum(abs(d) <= 1e-12 for d in differences)}
        correct = lambda r, v: abs(r['methods'][v]['rank_gain'] - r['best_allowed_gain']) < 1e-12
        comparisons[v]['sample_best'] = {'original_correct': sum(correct(r, 'original') for r in diagnostic),
            'retained': sum(correct(r, 'original') and correct(r, v) for r in diagnostic),
            'new_correct': sum(correct(r, v) for r in diagnostic)}
    assert max_fit_error < 1e-9 and max_distance_error < 1e-8 and max_sum_error < 1e-10
    assert all(sha(Path(p)) == expected for p, expected in hashes.items())
    dump('部署冻结.json', {'train_keys': frozen['train_keys'], 'targets': frozen['targets'], 'models': models})
    dump('特征.json', feature_map)
    result = {'development': development, 'diagnostic': diagnostic, 'means': means,
        'comparisons': comparisons, 'development_means': {v: float(np.mean([r['methods'][v]['gain'] for r in development])) for v in variants},
        'checks': {'max_distance_error': max_distance_error, 'max_sum_error': max_sum_error,
                   'max_fit_error': max_fit_error, 'inputs_unchanged': True},
        'feature_with_independent_verification_seconds': feature_seconds,
        'new_generator_calls': 0, 'new_small_model_fits': 36,
        'scope': '11_ancestor_development_and_9_known_parent_posthoc_replay'}
    dump('COMPLETE.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('development', 'diagnostic')}, ensure_ascii=True))
    print(json.dumps([{'parent': r['parent'], 'methods': {v: {k: x for k, x in r['methods'][v].items() if k != 'scores'} for v in variants}} for r in diagnostic]))


if __name__ == '__main__':
    main()
