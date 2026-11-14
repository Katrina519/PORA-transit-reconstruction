"""Mumford1冻结B迁移小测；只适配数据、约束和实例接口。"""
import os
import sys
import json
import time
import copy
import ast
import inspect
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
DATA = WORK / 'paper_code_learning_20261030_v1/dln/author_source/Codes and Instances/Networks and Line plans'
PRIOR = WORK / 'nea_frozen_multiseed_20261111_v1/请求冻结.json'
SPEC = {'city': 'Mumford1', 'n': 70, 'routes': 15, 'min': 10, 'max': 30, 'diameter': 44.}
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_runtime():
    spec = importlib.util.spec_from_file_location('mumford1_frozen_driver', WORK / 'nea_paper_final_confirm_20261110_v1/driver.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def configure(runtime, document):
    d = runtime.d
    base, q, np = d.b.base, d.q, d.q.np
    assert document == read(WORK / 'nea_paper_final_confirm_20261110_v1/模型冻结.json')['full']['mumford0']
    assert document['encoding'] == 'raw'
    base.SPEC = dict(SPEC)
    q.ROADS = np.loadtxt(ROOT / 'inputs/Mumford1TravelTimes.txt')
    q.Q = np.loadtxt(ROOT / 'inputs/Mumford1Demand.txt')
    q.MODEL = copy.deepcopy(base.MODEL)
    q.a.build = base.compile_function('build', [
        ("'city':'Mandl'", "'city':SPEC['city']", 1),
        ('RouteGenBatchState(data,cost,6,2,8)',
         "RouteGenBatchState(data,cost,SPEC['routes'],SPEC['min'],SPEC['max'])", 1)],
        {**q.a.__dict__, 'INPUTS': ROOT / 'inputs'})
    q.assess = base.compile_function('assess', [
        ('len(routes)==6', "len(routes)==SPEC['routes']", 1),
        ('set(range(15))', "set(range(SPEC['n']))", 1),
        ('2<=len(rr)<=8', "SPEC['min']<=len(rr)<=SPEC['max']", 1)], q.__dict__)

    def scoring(x):
        # 保留B的原始十维编码及冻结标准化，不能按新城市重估。
        return d.b.learner.scores(document, x, 'mumford0').tolist()

    base.BASE_SELECTION = base.compile_function('selection', [
        ('range(15)', "range(SPEC['n'])", 1),
        ('range(6)', "range(SPEC['routes'])", 3),
        ('j+1:],15,pieces)', "j+1:],SPEC['n'],pieces)", 1),
        ('ref.selector.scores(MODEL,[x+[H[j]] for j,x in enumerate(X)])',
         'joint_score([x+[H[j]] for j,x in enumerate(X)])', 1)],
        {**q.__dict__, 'joint_score': scoring})
    q.selection = base.BASE_SELECTION


def adapted_one(runtime):
    source = inspect.getsource(runtime.d.one)
    changes = [
        ("nr, limit, diameter = (6, 8, 33.) if city == 'mandl' else (12, 15, 26.)",
         'nr, limit, diameter = 15, 30, 44.'),
        ("b.configure(city, document if city == 'mumford0' else None)",
         'configure_mumford1(document)')]
    for old, new in changes:
        assert source.count(old) == 1
        source = source.replace(old, new)
    anchor = "                     'cost': value, 'samples': cost.samples, 'solution_index': len(solutions)-1})"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor +
        "\n        progress(method, int(iteration), int(pass_index), stamp, value)")
    scope = dict(runtime.d.__dict__)
    scope.update(configure_mumford1=lambda doc: configure(runtime, doc), progress=progress)
    exec(compile(source, str(ROOT / 'pilot.py') + '::instance_adapter', 'exec'), scope)
    return scope['one']


def progress(method, iteration, pass_index, seconds, value):
    event = dict(method=method, iteration=iteration, pass_index=pass_index,
                 seconds=seconds, cost=value)
    with (ROOT / '进度.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
    print('PASS', method, iteration, pass_index, round(seconds, 3), round(value, 8), flush=True)


def check_hashes(hashes):
    mismatches = [p for p, expected in hashes.items() if sha(p) != expected]
    assert not mismatches, mismatches


def prepare():
    import numpy as np
    assert not (ROOT / '请求冻结.json').exists()
    previous = read(PRIOR)
    check_hashes(previous['hashes'])
    input_dir = ROOT / 'inputs'
    input_dir.mkdir(exist_ok=False)
    for original, target in [('Mumford1Network.txt', 'Mumford1TravelTimes.txt'),
                             ('Mumford1Demand.txt', 'Mumford1Demand.txt'),
                             ('Mumford1Coords.txt', 'Mumford1Coords.txt')]:
        (input_dir / target).write_bytes((DATA / original).read_bytes())
    roads = np.loadtxt(DATA / 'Mumford1Network.txt')
    demand = np.loadtxt(DATA / 'Mumford1Demand.txt')
    assert roads.shape == demand.shape == (70, 70)
    assert np.array_equal(roads, roads.T) and np.array_equal(demand, demand.T)
    distance = roads.copy()
    for k in range(70):
        distance = np.minimum(distance, distance[:, k, None] + distance[None, k, :])
    assert np.isfinite(distance).all() and float(distance.max()) == 44.
    routes = [[int(v)-1 for v in line.split()] for line in
              (DATA / 'Mumford1Lineplan.txt').read_text().splitlines() if line.strip()]
    assert len(routes) == 15 and set().union(*map(set, routes)) == set(range(70))
    assert all(10 <= len(r) <= 30 and len(r) == len(set(r)) and
               all(0 < roads[u,v] < np.inf for u,v in zip(r,r[1:])) for r in routes)
    hashes = dict(previous['hashes'])
    for p in list(input_dir.iterdir()) + [DATA / 'Mumford1Lineplan.txt', Path(__file__), ROOT / 'test_adapter.py', ROOT / '协议.md']:
        hashes[str(p)] = sha(p)
    request = dict(date='2026-11-12', spec=SPEC,
                   context=dict(city='mumford1', parent='mumford1_reference_lineplan', routes=routes),
                   seed=3001, methods=['author', 'full'], iterations=8, checkpoints_seconds=[30., 60., 120.],
                   primary_checkpoint_seconds=120., hashes=hashes,
                   frozen_B_source=str(WORK / 'nea_spo_v2_joint_20261109_v1/模型冻结.json'),
                   source_request_sha256=sha(PRIOR), no_training=True, no_author_or_generator_changes=True)
    dump(ROOT / '请求冻结.json', request)
    print('FROZEN', len(hashes), 'source hashes; one reference parent; seed 3001', flush=True)


def snapshot(request, result, point, runtime):
    index = point['solution_index']
    routes = result['fine_solutions'][index]['routes']
    metrics = runtime.d.q.assess(routes)
    assert metrics['feasible']
    expected = metrics['P']/88. + metrics['O']/660.
    assert abs(expected-point['cost']) < 2e-6
    return dict(cost=point['cost'], passenger_minutes=metrics['P'],
                route_minutes_oneway=metrics['O'], finished_seconds=point['seconds'])


def run():
    request = read(ROOT / '请求冻结.json')
    check_hashes(request['hashes'])
    assert read(ROOT / '门禁.json')['passed']
    assert not (ROOT / '原始运行.jsonl').exists()
    tick = time.perf_counter()
    runtime = load_runtime()
    import_seconds = time.perf_counter()-tick
    fn = adapted_one(runtime)
    runs = []
    for method in request['methods']:
        value = fn(request['context'], request['seed'], 'author' if method == 'author' else 'ours',
                   repair=False, iterations=request['iterations'])
        value['variant'] = method
        assert value['generation_counts'] == [80, 800] and value['target_samples'] == 811
        assert len(value['fine_curve']) == 41
        assert all(b['cost'] <= a['cost']+1e-7 for a,b in zip(value['fine_curve'], value['fine_curve'][1:]))
        rows = {}
        for budget in request['checkpoints_seconds']:
            eligible = [p for p in value['fine_curve'] if p['seconds'] <= budget]
            rows[str(budget)] = snapshot(request, value, eligible[-1], runtime) if eligible else None
        final = runtime.d.q.assess(value['routes'])
        assert final['feasible'] and abs(final['P']/88.+final['O']/660.-value['cost']) < 2e-6
        value['checked_endpoints'] = rows
        value['independent_final'] = final
        with (ROOT / '原始运行.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        runs.append(value)
        print('DONE', method, value['cost'], round(value['seconds'], 3), flush=True)
        check_hashes(request['hashes'])
    comparison = {}
    for budget in request['checkpoints_seconds']:
        a, b = [v['checked_endpoints'][str(budget)] for v in runs]
        comparison[str(budget)] = dict(author=a, full=b,
            improvement_percent=100*(a['cost']-b['cost'])/a['cost'] if a and b else None,
            completed_all_8_iterations={v['variant']:v['seconds']<=budget for v in runs})
    a, b = runs
    comparison['fixed_8_iterations'] = dict(author_cost=a['cost'], full_cost=b['cost'],
        improvement_percent=100*(a['cost']-b['cost'])/a['cost'])
    dump(ROOT / '结果与核验.json', dict(passed=True, comparison=comparison,
         import_seconds=import_seconds, search_seconds={v['variant']:v['seconds'] for v in runs},
         selection_seconds=b['selection_seconds'], score_calls=b['score_calls'], cache_hits=b['cache_hits'],
         target_samples_each=811, generation_counts_each=[80,800], all_sources_unchanged=True,
         scope='one paired pilot; not merged with prior paper statistics'))
    print(json.dumps(comparison, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        assert sys.argv[1] == 'run'
        run()
