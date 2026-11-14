"""用独立的线路分层Dijkstra核验保存解与全部配对结果。"""
import json
import csv
import math
import heapq
import statistics
import random
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROADS = [[float(v) for v in row.split()] for row in
         (ROOT / 'inputs/Mumford0TravelTimes.txt').read_text().splitlines() if row.strip()]
Q = [[float(v) for v in row.split()] for row in
     (ROOT / 'inputs/Mumford0Demand.txt').read_text().splitlines() if row.strip()]
CACHE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def evaluate(routes):
    key = tuple(tuple(rr) for rr in routes)
    if key in CACHE:
        return CACHE[key]
    assert len(routes) == 12 and set().union(*map(set, routes)) == set(range(30))
    assert all(2 <= len(rr) <= 15 and len(set(rr)) == len(rr) for rr in routes)
    adj, at = {}, [[] for _ in range(30)]
    for ri, rr in enumerate(routes):
        for u in rr:
            at[u].append(ri); adj[ri, u] = []
        for u, v in zip(rr, rr[1:]):
            assert 0 < ROADS[u][v] < math.inf
            adj[ri, u].append((ROADS[u][v], ri, v))
            adj[ri, v].append((ROADS[v][u], ri, u))
    total = 0.
    for s in range(30):
        queue = [(0., ri, s) for ri in at[s]]; heapq.heapify(queue)
        dist = {(ri, s): 0. for ri in at[s]}; best = [math.inf] * 30; best[s] = 0.
        while queue:
            value, ri, u = heapq.heappop(queue)
            if value != dist[ri, u]:
                continue
            best[u] = min(best[u], value)
            for w, rj, v in adj[ri, u] + [(5., rj, u) for rj in at[u] if rj != ri]:
                trial = value + w
                if trial < dist.get((rj, v), math.inf):
                    dist[rj, v] = trial; heapq.heappush(queue, (trial, rj, v))
        for t in range(30):
            if Q[s][t] > 0:
                assert math.isfinite(best[t])
                total += Q[s][t] * best[t]
    p = total / sum(map(sum, Q))
    o = sum(ROADS[u][v] for rr in routes for u, v in zip(rr, rr[1:]))
    CACHE[key] = (p, o)
    return p, o


def check_selection(item, n):
    details = [v for batch in item['selection_log'] for v in batch]
    if item['method'] != 'v2_spo':
        assert not details and item['score_calls'] == item['cache_hits'] == 0
        return 0
    assert details
    for row in details:
        if 'fallback' in row:
            continue
        assert row['allowed'] == list(range(n)) and len(row['scores']) == n
        assert all(math.isfinite(v) for v in row['scores'])
        assert row['new'] == min(range(n), key=lambda j: (-row['scores'][j], j))
    scored = sum('fallback' not in row for row in details)
    assert item['score_calls'] + item['cache_hits'] == scored
    return scored


def at(item, seconds):
    return next((e for e in item['fine_curve'][::-1] if e['seconds'] <= seconds), None)


def main():
    request = read(ROOT / '请求冻结.json'); budget = read(ROOT / '预算冻结.json')
    complete = read(ROOT / 'comparison/COMPLETE.json'); runs = complete['runs']
    model = read(ROOT / '模型冻结.json')
    assert sha(model['source']) == model['source_sha256']
    assert model['model'] == read(model['source'])['models']['v2_spo']
    assert all(sha(p) == h for p, h in complete['input_hashes'].items())
    expected = {(c['parent'], s, m) for c in request['contexts'] for s in request['seeds'] for m in request['methods']}
    idx = {(r['parent'], r['seed'], r['method']): r for r in runs}
    assert len(runs) == len(idx) == 18 and set(idx) == expected
    max_error, references, decisions = 0., 0, 0
    rows = []
    t = budget['primary_seconds']; h = budget['secondary_seconds']
    scopes = {'half_time': h, 'primary': t, 'fixed_8': None}
    for item in runs:
        assert item['target_samples'] == 811 and item['target_calls'] == 82
        assert len(item['fine_curve']) == 41
        assert all(p['samples'] == 10+20*i for i, p in enumerate(item['fine_curve']))
        assert all(a['seconds'] <= b['seconds'] and a['cost'] >= b['cost'] for a, b in zip(item['fine_curve'], item['fine_curve'][1:]))
        decisions += check_selection(item, 12)
        for solution in item['fine_solutions']:
            p, o = evaluate(solution['routes']); references += 1
            max_error = max(max_error, abs(p/52+o/312-solution['cost']))
        for point in item['fine_curve']:
            solution = item['fine_solutions'][point['solution_index']]
            assert solution['first_seconds'] <= point['seconds'] and solution['cost'] == point['cost']
        for label, seconds in scopes.items():
            point = item['fine_curve'][-1] if seconds is None else at(item, seconds)
            covered = seconds is None or item['fine_curve'][-1]['seconds'] >= seconds
            if point is None or not covered:
                rows.append({'parent': item['parent'], 'seed': item['seed'], 'method': item['method'],
                             'scope': label, 'comparable': False, 'cost': None, 'P_minutes': None, 'O_single_minutes': None})
                continue
            solution = item['fine_solutions'][point['solution_index']]
            p, o = evaluate(solution['routes'])
            rows.append({'parent': item['parent'], 'seed': item['seed'], 'method': item['method'],
                         'scope': label, 'comparable': True, 'cost': point['cost'], 'P_minutes': p, 'O_single_minutes': o})
        assert max_error < 1e-6
    summary = {}
    rng = random.Random(20261109)
    for label in scopes:
        subset = [v for v in rows if v['scope'] == label]
        if not all(v['comparable'] for v in subset):
            summary[label] = {'comparable': False, 'reason': '未覆盖预算或截止前无完成评价，不用未来结果补齐'}
            continue
        means = {m: {k: statistics.mean(v[k] for v in subset if v['method'] == m)
                     for k in ['cost', 'P_minutes', 'O_single_minutes']} for m in request['methods']}
        keyed = {(v['parent'], v['seed'], v['method']): v for v in subset}
        differences, parents = [], []
        for context in request['contexts']:
            parent = context['parent']; ds = []
            for seed in request['seeds']:
                delta = keyed[parent, seed, 'author']['cost']-keyed[parent, seed, 'v2_spo']['cost']
                differences.append(delta); ds.append(delta)
            aa = statistics.mean(keyed[parent, s, 'author']['cost'] for s in request['seeds'])
            bb = statistics.mean(keyed[parent, s, 'v2_spo']['cost'] for s in request['seeds'])
            parents.append({'parent': parent, 'author_mean': aa, 'v2_mean': bb,
                            'improvement_percent': 100*(aa-bb)/aa, 'cost_difference': statistics.mean(ds)})
        cluster = [v['cost_difference'] for v in parents]
        boot = sorted(statistics.mean(rng.choice(cluster) for _ in cluster) for _ in range(10000))
        summary[label] = {'comparable': True, 'means': means,
            'improvement_percent': 100*(means['author']['cost']-means['v2_spo']['cost'])/means['author']['cost'],
            'wins': sum(v > 1e-6 for v in differences), 'ties': sum(abs(v) <= 1e-6 for v in differences),
            'losses': sum(v < -1e-6 for v in differences), 'parent_details': parents,
            'parent_cluster_bootstrap95_cost_difference': [boot[249], boot[9749]]}
    timing = {m: {k: statistics.mean(v[k] for v in runs if v['method'] == m)
                  for k in ['seconds', 'selection_seconds', 'timing_log_seconds', 'wrapper_preparation_seconds', 'score_calls', 'cache_hits']}
              for m in request['methods']}
    audit = {'runs': len(runs), 'pairs': 9, 'parent_clusters': 3, 'primary_seconds': t,
        'max_independent_cost_error': max_error, 'saved_solution_references': references, 'unique_solutions': len(CACHE),
        'scored_type1_choices': decisions, 'cost_samples': sum(v['target_samples'] for v in runs),
        'all_model_and_input_hashes_unchanged': True, 'timing': timing, 'results': summary,
        'phase_import_seconds': complete['import_seconds'], 'phase_wall_seconds_excluding_import': complete['wall_seconds'],
        'raw_io_seconds': complete['raw_io_seconds'], 'new_training': False,
        'scope': '冻结Mandl训练评分在一个Mumford0实例三个新父网上的小规模迁移；全部18次保留'}
    dump(ROOT / '结果与核验.json', audit)
    with (ROOT / '逐次结果.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
