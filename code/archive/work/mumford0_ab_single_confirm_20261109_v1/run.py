"""仅替换冻结评分，执行身份门禁及36次预定对照。"""
import time
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_io import ROOT, WORK, JOINT, MUM, b, IMPORT_SECONDS, read, sha, dump, models


def main(phase):
    started = time.perf_counter()
    request = read(ROOT / '请求冻结.json'); source = read(ROOT / '输入冻结_接口修正.json')
    frozen = read(ROOT / '模型冻结.json')['documents']
    assert models() == frozen
    hashes = dict(source['input_hashes']); hashes[str(ROOT / '输入冻结_接口修正.json')] = sha(ROOT / '输入冻结_接口修正.json')
    assert all(sha(p) == s for p, s in hashes.items())
    if phase == 'comparison':
        gate = read(ROOT / 'gate_retry/COMPLETE.json'); assert gate['passed']
        assert all(sha(p) == s for p, s in gate['input_hashes'].items())
    folder = ROOT / phase; folder.mkdir(exist_ok=False)
    dump(folder / '运行冻结.json', {'phase': phase, 'request': request, 'input_hashes': hashes,
                                  'threads': b.torch.get_num_threads(), 'device': 'cpu'})
    audit = b.q.load('ab_single_cost_audit', JOINT / 'audit.py')
    runs, configuration = [], []
    io_seconds = 0.
    with (folder / '原始运行.jsonl').open('x', encoding='utf-8') as stream:
        def collect(context, seed, method):
            nonlocal io_seconds
            tick = time.perf_counter()
            b.configure('mumford0', frozen[method] if method == 'B_spo' else None)
            assert b.base.MODEL == frozen['A_spo']['model']
            one = b.base.make_one()
            prep = time.perf_counter()-tick
            configuration.append({'parent': context['parent'], 'seed': seed, 'method': method, 'seconds': prep})
            v = b.base.run_one(one, context, seed, 'author' if method == 'author' else 'v2_spo', 8)
            v['search_seconds_before_configuration'] = v['seconds']
            for event in v['fine_curve'] + v['curve']:
                event['seconds'] += prep
            for solution in v['fine_solutions']:
                solution['first_seconds'] += prep
            v['seconds'] += prep
            v['fine_checkpoints'] = {str(t): b.base.stage.at_budget(v['fine_curve'], t) for t in [2., 4., 7.]}
            v['checkpoints'] = {str(t): next((e['cost'] for e in reversed(v['curve']) if e['seconds'] <= t), None)
                                for t in [1., 2., 4., 7.]}
            v['initialized_at_checkpoint'] = {str(t): v['curve'][0]['seconds'] <= t for t in [1., 2., 4., 7.]}
            v.update(method=method, city='mumford0', configuration_seconds=prep)
            assert v['target_samples'] == 811 and v['target_calls'] == 82 and len(v['fine_curve']) == 41
            audit.check_choices(v, 12); runs.append(v)
            tick = time.perf_counter()
            stream.write(__import__('json').dumps(v, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
            io_seconds += time.perf_counter()-tick
            print(phase, context['parent'], seed, method, round(v['cost'], 8), round(v['seconds'], 3), flush=True)
            return v
        if phase in ['gate', 'gate_retry']:
            fields = ['routes', 'history', 'rng', 'metrics', 'cost', 'selection_log',
                      'target_samples', 'target_calls', 'score_calls', 'cache_hits']
            for method, oldfolder, oldphase, oldmethod in [
                    ('author', MUM, 'comparison', 'author'), ('A_spo', MUM, 'comparison', 'v2_spo'),
                    ('B_spo', JOINT, 'confirmation', 'joint_spo')]:
                req = read(oldfolder / '请求冻结.json')
                context = next(c for c in req['contexts'] if c['parent'].startswith('mumford0'))
                seed = req['seeds'][0]
                v = collect(context, seed, method)
                old = next(r for r in read(oldfolder / oldphase / 'COMPLETE.json')['runs']
                           if r['parent'] == context['parent'] and r['seed'] == seed and r['method'] == oldmethod)
                for key in fields:
                    assert v[key] == old[key], (method, key)
                for solution in v['fine_solutions']:
                    _, _, cost = audit.evaluate('mumford0', solution['routes'])
                    assert abs(cost-solution['cost']) < 1e-6
        else:
            for pi, context in enumerate(request['contexts']):
                for si, seed in enumerate(request['seeds']):
                    offset = (pi+si) % 3
                    methods = request['methods'][offset:] + request['methods'][:offset]
                    for method in methods:
                        collect(context, seed, method)
            assert len(runs) == request['planned_runs']
    assert all(sha(p) == s for p, s in hashes.items())
    dump(folder / 'COMPLETE.json', {'runs': runs, 'passed': True, 'phase': phase, 'input_hashes': hashes,
        'wall_seconds': time.perf_counter()-started, 'import_seconds': IMPORT_SECONDS,
        'method_configuration': configuration, 'raw_io_seconds': io_seconds,
        'author_and_generator_unchanged': True, 'models_unchanged': True})
    print(phase, 'COMPLETE', len(runs), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['gate', 'gate_retry', 'comparison'], required=True)
    main(parser.parse_args().phase)
