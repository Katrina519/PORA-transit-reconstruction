"""固定模型的Mumford0接口门禁和预登记直接对照。"""
import sys
import time
import json
import math
import csv
import shutil
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
IMPORT_START = time.perf_counter()
sys.path.insert(0, str(ROOT))
import interface as app
IMPORT_SECONDS = time.perf_counter() - IMPORT_START


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def canonical(routes):
    return tuple(sorted(min(tuple(rr), tuple(rr[::-1])) for rr in routes))


def verifier():
    return app.q.load('mumford0_independent_verifier', ROOT / 'verify.py')


def source_hashes():
    old = read(WORK / 'nea_spo_v2_matched_ablation_20261108_v1/comparison/运行冻结.json')['input_hashes']
    assert all(sha(p) == h for p, h in old.items())
    paths = list(ROOT.glob('*.py')) + [ROOT / '协议与实施计划.md', ROOT / '模型冻结.json',
            ROOT / '请求冻结.json', ROOT / '数据与准备核验.json'] + list((ROOT / 'inputs').glob('*.txt'))
    old.update({str(p): sha(p) for p in paths})
    return old


def prepare():
    started = time.perf_counter()
    inputs = ROOT / 'inputs'
    inputs.mkdir(exist_ok=False)
    for original, target in [('Mumford0Network.txt', 'Mumford0TravelTimes.txt'),
                             ('Mumford0Demand.txt', 'Mumford0Demand.txt'),
                             ('Mumford0Coords.txt', 'Mumford0Coords.txt')]:
        shutil.copyfile(app.DLN / original, inputs / target)
    app.configure()
    manifest = read(WORK / '开源公交复现/idea1_exchange_final_factorial_v1/INPUTS.json')['config']
    spec = manifest['instances']['mumford0']
    cache = Path('C:/Users/m/.codex/worktrees/public-od-benchmark/bus-network-agent/data/raw/transit_benchmark_cache')
    rows = {}
    hashes = {}
    for kind, record in spec['files'].items():
        path = cache / record['path']
        assert sha(path) == record['sha256']
        rows[kind] = list(csv.DictReader(path.read_text(encoding='utf-8-sig').splitlines()))
        hashes[str(path)] = sha(path)
    np, q = app.np, app.q
    roads = np.full((30, 30), np.inf); np.fill_diagonal(roads, 0.)
    demand = np.zeros((30, 30))
    for row in rows['links']:
        roads[int(row['from'])-1, int(row['to'])-1] = float(row['travel_time'])
    for row in rows['demand']:
        demand[int(row['from'])-1, int(row['to'])-1] = float(row['demand'])
    assert np.array_equal(roads, q.ROADS) and np.array_equal(demand, q.Q)
    coords = np.loadtxt(inputs / 'Mumford0Coords.txt', skiprows=1)
    assert np.array_equal(coords, [[float(v['lon']), float(v['lat'])] for v in rows['nodes']])
    d = roads.copy()
    for k in range(30):
        d = np.minimum(d, d[:, k, None] + d[None, k, :])
    assert float(d.max()) == 26. and np.array_equal(roads, roads.T) and np.array_equal(demand, demand.T)
    finite = np.isfinite(roads)
    assert np.array_equal(roads[finite], d[finite])
    instance = q.c.BenchmarkInstance(tuple(range(1, 31)),
        {(int(v['from']), int(v['to'])): float(v['travel_time']) for v in rows['links']},
        {(int(v['from']), int(v['to'])): float(v['demand']) for v in rows['demand']},
        frozenset(int(v['id']) for v in rows['nodes'] if v['terminal'] == '1'), {})
    contexts, seen = [], set()
    init_started = time.perf_counter()
    for ident in range(62600, 62604):
        seed = q.ref.stream(ident, 'initialization').getrandbits(64)
        routes = [[u-1 for u in rr] for rr in q.ref.make_initial_routes(instance, 12, (2, 15), seed)]
        assert q.assess(routes)['feasible'] and canonical(routes) not in seen
        seen.add(canonical(routes))
        contexts.append({'parent': 'mumford0-12-' + str(ident), 'routes': routes,
                         'initialization_seed': seed, 'base': q.assess(routes)})
    init_seconds = time.perf_counter() - init_started
    dump(ROOT / '模型冻结.json', {'model': app.MODEL, 'source': str(app.MODEL_PATH),
        'source_sha256': sha(app.MODEL_PATH), 'new_training': False,
        'feature_divisors_unchanged': [420, 8, 15, 33], 'risk_H_divisor_unchanged': 33})
    dump(ROOT / '请求冻结.json', {'calibration': contexts[0], 'contexts': contexts[1:],
        'calibration_seed': 509, 'seeds': [510, 511, 512], 'methods': ['author', 'v2_spo'],
        'max_iterations': 8, 'time_rule': 'max(4,ceil(2*max(calibration_author_seconds,calibration_v2_seconds)))',
        'instance': app.SPEC, 'all_results_retained': True})
    source_files = [app.DLN / n for n in ['Mumford0Network.txt', 'Mumford0Demand.txt', 'Mumford0Coords.txt']]
    hashes.update({str(p): sha(p) for p in source_files})
    dump(ROOT / '数据与准备核验.json', {'input_hashes': hashes, 'nodes': 30, 'directed_links': 180,
        'positive_od': 870, 'road_diameter_minutes': 26., 'source_matrices_equal': True,
        'coords_equal': True, 'all_road_edges_are_shortest': True,
        'initialization_seconds': init_seconds, 'total_prepare_seconds': time.perf_counter()-started,
        'import_seconds': IMPORT_SECONDS, 'unique_initial_networks_including_calibration': len(seen),
        'spec_input_only_not_old_transfer_limit': True})
    print('PREPARED 3 formal parents + 1 calibration', flush=True)


def run_phase(phase):
    started = time.perf_counter()
    folder = ROOT / phase
    folder.mkdir(exist_ok=False)
    request = read(ROOT / '请求冻结.json')
    hashes = source_hashes()
    hashes.update(read(ROOT / '数据与准备核验.json')['input_hashes'])
    if phase == 'comparison':
        gate_record = read(ROOT / '通过门禁.json')
        assert sha(gate_record['path']) == gate_record['sha256']
        gate = read(gate_record['path'])
        assert gate['passed']
        assert all(sha(p) == h for p, h in gate['input_hashes'].items())
        budget = read(ROOT / '预算冻结.json')
        hashes[str(ROOT / '预算冻结.json')] = sha(ROOT / '预算冻结.json')
    dump(folder / '运行冻结.json', {'request': request, 'input_hashes': hashes, 'phase': phase,
        'python': sys.version, 'torch': app.torch.__version__, 'numpy': app.np.__version__,
        'threads': app.torch.get_num_threads(), 'cuda_available': app.torch.cuda.is_available(),
        'new_training': False, 'environment_rebuilt': False})
    runs, io_seconds = [], 0.
    with (folder / '原始运行.jsonl').open('x', encoding='utf-8') as stream:
        def collect(context, seed, method, iterations, one, instance):
            nonlocal io_seconds
            item = app.run_one(one, context, seed, method, iterations)
            item['instance'] = instance
            assert item['target_samples'] == 11 + 100 * iterations
            assert len(item['fine_curve']) == 1 + 5 * iterations
            runs.append(item)
            tick = time.perf_counter()
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush(); io_seconds += time.perf_counter()-tick
            print(phase, context['parent'], seed, method, round(item['cost'], 8), round(item['seconds'], 3), flush=True)
            return item
        if phase.startswith('gate'):
            app.configure('mandl'); one = app.make_one()
            old_req = read(WORK / 'nea_spo_v2_matched_ablation_20261108_v1/请求冻结.json')
            old_runs = read(WORK / 'nea_spo_v2_matched_ablation_20261108_v1/comparison/COMPLETE.json')['runs']
            for method in ['author', 'v2_spo']:
                x = collect(old_req['contexts'][0], 330, method, 8, one, 'Mandl')
                old = next(v for v in old_runs if v['parent'] == x['parent'] and v['seed'] == 330 and v['method'] == method)
                for key in ['routes', 'history', 'rng', 'metrics', 'cost', 'selection_log',
                            'target_samples', 'target_calls', 'score_calls', 'cache_hits']:
                    assert x[key] == old[key], (method, key)
            app.configure(); one = app.make_one(); calibration = []
            for method in ['author', 'identity', 'v2_spo']:
                calibration.append(collect(request['calibration'], 509, method, 1, one, 'Mumford0'))
            for key in ['routes', 'history', 'rng', 'metrics', 'cost', 'selection_log', 'target_samples', 'target_calls']:
                assert calibration[0][key] == calibration[1][key], key
            module = verifier()
            evaluate, check_selection = module.evaluate, module.check_selection
            for item in calibration:
                check_selection(item, 12)
                for solution in item['fine_solutions']:
                    p, o = evaluate(solution['routes'])
                    assert abs(p/52+o/312-solution['cost']) < 1e-6
            assert calibration[-1]['score_calls'] > 0 and calibration[-1]['fallbacks'] == 0
            total = max(4, math.ceil(2*max(calibration[0]['seconds'], calibration[2]['seconds'])))
            dump(ROOT / '预算冻结.json', {'primary_seconds': total, 'secondary_seconds': total/2,
                'rule': request['time_rule'], 'calibration_author_seconds': calibration[0]['seconds'],
                'calibration_v2_seconds': calibration[2]['seconds'], 'formal_outcomes_seen': False})
        else:
            app.configure(); one = app.make_one()
            for pi, context in enumerate(request['contexts']):
                for si, seed in enumerate(request['seeds']):
                    methods = request['methods'] if (pi+si)%2 == 0 else request['methods'][::-1]
                    for method in methods:
                        collect(context, seed, method, 8, one, 'Mumford0')
    assert app.q.MODEL == app.MODEL == read(app.MODEL_PATH)['models']['v2_spo']
    assert all(sha(p) == h for p, h in hashes.items())
    dump(folder / 'COMPLETE.json', {'runs': runs, 'passed': True, 'input_hashes': hashes,
        'source_replacements': app.REUSE, 'import_seconds': IMPORT_SECONDS,
        'raw_io_seconds': io_seconds, 'wall_seconds': time.perf_counter()-started,
        'scope': 'gate includes historical identity checks; comparison is the full predeclared 18 runs'})
    if phase.startswith('gate'):
        dump(ROOT / '通过门禁.json', {'path': str(folder / 'COMPLETE.json'),
                                    'sha256': sha(folder / 'COMPLETE.json')})
    print(phase, 'COMPLETE', len(runs), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['prepare', 'gate', 'gate_retry', 'comparison', 'check_verifier'], required=True)
    args = ap.parse_args()
    if args.phase == 'check_verifier':
        module = verifier()
        assert Path(module.__file__).resolve() == (ROOT / 'verify.py').resolve()
        assert callable(module.evaluate)
        print('LOCAL_VERIFIER_OK', module.__file__)
    elif args.phase == 'prepare':
        prepare()
    else:
        run_phase(args.phase)
