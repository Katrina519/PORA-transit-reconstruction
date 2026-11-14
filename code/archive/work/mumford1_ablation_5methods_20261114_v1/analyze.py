"""五配置结果统计、选择重放和独立线路层评价；不运行搜索。"""
import sys
import csv
import json
import math
import statistics
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import experiment as e


def write_csv(name, rows):
    with (ROOT / name).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    output = [0.] * len(values)
    previous = 0.
    for position, index in enumerate(order):
        previous = max(previous, min(1., (len(values) - position) * values[index]))
        output[index] = previous
    return output


def outcome(delta, epsilon=1e-6):
    return dict(wins=int(sum(v > epsilon for v in delta)),
                ties=int(sum(abs(v) <= epsilon for v in delta)),
                losses=int(sum(v < -epsilon for v in delta)))


def analyze():
    req = e.read(ROOT / '请求冻结.json')
    e.check(req['hashes'])
    analysis_plan = e.read(ROOT / '分析方案冻结.json')
    e.check(analysis_plan['hashes'])
    done = e.read(ROOT / 'COMPLETE.json')
    assert done['passed'] and done['completed_runs'] == 50
    assert done['request_sha256'] == e.sha(ROOT / '请求冻结.json')
    raw = [json.loads(line) for line in (ROOT / '原始运行.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(raw) == 50
    assert [(r['seed'], r['variant']) for r in raw] == [(t['seed'], t['variant']) for t in req['tasks']]
    audit = e.module('frozen_independent_route_evaluator', e.PREVIOUS / 'analyze.py')
    audit.e = SimpleNamespace(PILOT=e.WORK / 'mumford1_frozen_B_pilot_20261112_v1')
    maximum = 0.
    replayed = 0
    rows, checkpoints, mechanisms = [], [], []
    score_records = {}
    for run_index, run in enumerate(raw, 1):
        method, seed = run['variant'], run['seed']
        assert run['generation_counts'] == [80, 800]
        assert run['target_samples'] == 811 and run['target_calls'] == 82
        assert run['iterations'] == 8 and not run['repair'] and not run['fallbacks']
        assert len(run['fine_curve']) == 41
        start = run['fine_solutions'][run['fine_curve'][0]['solution_index']]
        assert start['routes'] == req['context']['routes']
        for solution in run['fine_solutions']:
            _, _, cost = audit.evaluate(solution['routes'])
            maximum = max(maximum, abs(cost - solution['cost']))
        for i, point in enumerate(run['fine_curve']):
            solution = run['fine_solutions'][point['solution_index']]
            assert point['cost'] == solution['cost']
            assert solution['first_seconds'] <= point['seconds']
            assert point['samples'] == 10 + 20 * i
            if i:
                assert run['fine_curve'][i-1]['seconds'] <= point['seconds']
                assert point['cost'] <= run['fine_curve'][i-1]['cost'] + 1e-7
            checkpoints.append(dict(seed=seed, method=method, iteration=point['iteration'],
                pass_index=point['pass'], seconds=point['seconds'], cost=point['cost'], samples=point['samples']))
        if method == 'author':
            assert not run['selection_log'] and not run['feedback_log']
            changed_choices = 0
        else:
            assert len(run['selection_log']) == 80 and len(run['feedback_log']) == 400
            assert run['score_calls'] + run['cache_hits'] == 400
            allocator = e.Allocator(.05, use_feedback=method != 'nogain')
            counts = {}
            changed_choices = 0
            for batch in range(80):
                events = [event for event in run['feedback_log'] if event['batch_id'] == batch]
                details = run['selection_log'][batch]
                assert len(events) == len(details) == 5
                costs, valid = [0.] * 10, [False] * 10
                for event, detail in zip(events, details):
                    key = tuple(map(tuple, event['parent_routes']))
                    assert event['j'] == detail['new'] and event['bee'] == detail['bee']
                    assert event['eligible'] == detail['allowed'] == list(range(15))
                    scores = detail['scores']
                    assert len(scores) == 15 and all(math.isfinite(x) for x in scores)
                    if method == 'count':
                        assert scores == [0.] * 15 and event.get('reward') is None
                        visits = counts.setdefault(key, [0] * 15)
                        chosen = min(range(15), key=lambda j: (visits[j], j))
                        visits[chosen] += 1
                    else:
                        model_key = ('ridge' if method == 'ridge' else 'full', key)
                        if model_key in score_records:
                            assert score_records[model_key] == scores
                        else:
                            score_records[model_key] = scores
                        chosen = allocator.choose(key, scores, event['bee'], event['parent_cost'])
                        state = allocator.states[key]
                        visits_before = allocator.log[-1]['visits_before']
                        no_gain_choice = min(range(15), key=lambda j: (
                            -.05 * state['prior'][j] * math.sqrt(max(1, sum(visits_before))) /
                            (1 + visits_before[j]), j))
                        changed_choices += int(chosen != no_gain_choice)
                    assert chosen == event['j']
                    _, _, cost = audit.evaluate(event['parent_routes'])
                    maximum = max(maximum, abs(cost - event['parent_cost']))
                    if event['valid']:
                        _, _, cost = audit.evaluate(event['candidate_routes'])
                        maximum = max(maximum, abs(cost - event['candidate_cost']))
                    costs[event['bee']] = event['candidate_cost']
                    valid[event['bee']] = event['valid']
                    replayed += 1
                if method != 'count':
                    allocator.observe(costs, valid)
                    for event, log in zip(events, allocator.log[-5:]):
                        assert abs(event['reward'] - log['reward']) < 1e-12
            assert not allocator.pending
            visits = Counter(tuple(map(tuple, event['parent_routes'])) for event in run['feedback_log'])
            targets = {(tuple(map(tuple, event['parent_routes'])), event['j']) for event in run['feedback_log']}
            mechanisms.append(dict(seed=seed, method=method, distinct_parents=len(visits),
                mean_selections_per_parent=400 / len(visits), distinct_parent_route_tasks=len(targets),
                task_coverage=len(targets) / (15 * len(visits)),
                feasible_proposals=sum(event['valid'] for event in run['feedback_log']),
                positive_proposals=sum(event['valid'] and event['parent_cost'] - event['candidate_cost'] > 1e-6
                                       for event in run['feedback_log']),
                gain_changed_choice_count=changed_choices,
                gain_changed_choice_fraction=changed_choices / 400,
                model_evaluations=0 if method == 'count' else run['score_calls'],
                cache_hits=run['cache_hits']))
        for endpoint in ['8round', '120s']:
            point = run['fine_curve'][-1] if endpoint == '8round' else [
                p for p in run['fine_curve'] if p['seconds'] <= 120.][-1]
            p, o, cost = audit.evaluate(run['fine_solutions'][point['solution_index']]['routes'])
            maximum = max(maximum, abs(cost - point['cost']))
            rows.append(dict(city='mumford1', parent=run['parent'], seed=seed, method=method,
                endpoint=endpoint, cost=point['cost'], P_minutes=p, O_oneway_minutes=o,
                last_completed_seconds=point['seconds'], completed_passes=(point['samples']-10)//20,
                total_online_seconds=run['seconds'], selection_seconds=run['selection_seconds'],
                feedback_seconds=run['feedback_seconds'], preparation_seconds=run['wrapper_preparation_seconds'],
                model_evaluations=0 if method in ['author', 'count'] else run['score_calls'],
                cache_hits=run['cache_hits'], gain_changed_choice_count=changed_choices))
        print('AUDIT', run_index, '/50', seed, method, 'networks', len(audit.CACHE),
              'max_error', maximum, flush=True)
    assert replayed == 16000 and maximum < 2e-6
    summary, pairs, comparisons = [], [], []
    import numpy as np
    bootstrap = np.random.RandomState(20261114)
    for endpoint in ['8round', '120s']:
        index = {(r['seed'], r['method']): r for r in rows if r['endpoint'] == endpoint}
        author = [index[s, 'author']['cost'] for s in req['seeds']]
        for method in req['methods']:
            values = [index[s, method]['cost'] for s in req['seeds']]
            mean = statistics.mean(values)
            summary.append(dict(endpoint=endpoint, method=method, n=10, mean_cost=mean,
                sd_cost=statistics.stdev(values), improvement_vs_NEA_percent=100*(statistics.mean(author)-mean)/statistics.mean(author),
                mean_P_minutes=statistics.mean(index[s, method]['P_minutes'] for s in req['seeds']),
                mean_O_oneway_minutes=statistics.mean(index[s, method]['O_oneway_minutes'] for s in req['seeds']),
                mean_online_seconds=statistics.mean(index[s, method]['total_online_seconds'] for s in req['seeds']),
                mean_selection_seconds=statistics.mean(index[s, method]['selection_seconds'] for s in req['seeds']),
                mean_feedback_seconds=statistics.mean(index[s, method]['feedback_seconds'] for s in req['seeds']),
                **outcome([a-b for a,b in zip(author, values)])))
        for comparator in ['author', 'count', 'ridge', 'nogain']:
            control = [index[s, comparator]['cost'] for s in req['seeds']]
            full = [index[s, 'full']['cost'] for s in req['seeds']]
            delta = np.array(control) - np.array(full)
            ci = np.percentile(delta[bootstrap.randint(0, 10, size=(20000, 10))].mean(axis=1), [2.5, 97.5])
            test = audit.signed_rank_exact(delta)
            comparisons.append(dict(endpoint=endpoint, comparator=comparator, n=10,
                comparator_mean=statistics.mean(control), full_mean=statistics.mean(full),
                full_improvement_percent=100*(statistics.mean(control)-statistics.mean(full))/statistics.mean(control),
                mean_absolute_gain=float(delta.mean()), gain_ci95_low=float(ci[0]), gain_ci95_high=float(ci[1]),
                p_holm=None, prespecified_mechanism_family=comparator != 'author',
                **outcome(delta), **test))
            for seed in req['seeds']:
                a, b = index[seed, comparator], index[seed, 'full']
                pairs.append(dict(seed=seed, endpoint=endpoint, comparator=comparator,
                    comparator_cost=a['cost'], full_cost=b['cost'], absolute_gain=a['cost']-b['cost'],
                    full_improvement_percent=100*(a['cost']-b['cost'])/a['cost'],
                    comparator_P=a['P_minutes'], full_P=b['P_minutes'],
                    comparator_O=a['O_oneway_minutes'], full_O=b['O_oneway_minutes']))
        family = [r for r in comparisons if r['endpoint'] == endpoint and r['prespecified_mechanism_family']]
        for row, adjusted in zip(family, holm([r['p_two_sided'] for r in family])):
            row['p_holm'] = adjusted
    write_csv('主汇总.csv', summary)
    write_csv('配对检验.csv', comparisons)
    write_csv('逐次方法结果.csv', rows)
    write_csv('逐种子配对.csv', pairs)
    write_csv('全部检查点.csv', checkpoints)
    write_csv('机制概览.csv', mechanisms)
    result = dict(passed=True, summary=summary, comparisons=comparisons, rows=rows,
        pairs=pairs, mechanisms=mechanisms, checkpoints=checkpoints,
        method_runs=50, paired_seeds=10, replayed_selections=replayed,
        independently_evaluated_networks=len(audit.CACHE), max_cost_error=maximum,
        protected_sources=len(req['hashes']), primary_endpoint='8round', secondary_endpoint='120s',
        one_fixed_initial_network=True, existing_seed_extension=True,
        request_sha256=e.sha(ROOT/'请求冻结.json'), analysis_sha256=e.sha(__file__),
        common_import_seconds=done['import_seconds'], raw_io_seconds=done['raw_io_seconds_this_invocation'])
    e.dump(ROOT / '统计与核验.json', result)
    e.check(req['hashes'])
    print(json.dumps(dict(summary=summary, comparisons=comparisons,
                         replays=replayed, networks=len(audit.CACHE), maximum=maximum), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    analyze()
