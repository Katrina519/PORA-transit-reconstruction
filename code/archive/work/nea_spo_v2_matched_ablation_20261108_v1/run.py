"""第2版与两个匹配消融：复用冻结模型及原运行内核。"""
import sys
import time
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
IMPORT_START = time.perf_counter()
sys.path.insert(0, str(WORK / 'nea_spo_cost_scale_20261108_v1'))
import run_stage as stage
r, q, torch = stage.r, stage.q, stage.torch
read, dump = stage.read, stage.dump
IMPORT_SECONDS = time.perf_counter() - IMPORT_START


def prepare():
    source = WORK / 'nea_spo_balanced_tasks_20261108_v1'
    original = read(source / '候选冻结.json')
    old = read(source / '请求冻结.json')
    mapping = {'v2_spo': 'balanced_spo', 'no_spo_ridge': 'balanced_ridge', 'old_feedback_spo': 'previous_spo'}
    models = {name: original['models'][key] for name, key in mapping.items()}
    for key in ['indices', 'mean', 'scale', 'lambda']:
        assert models['v2_spo'][key] == models['no_spo_ridge'][key]
    for key in ['indices', 'mean', 'scale']:
        assert models['v2_spo'][key] == models['old_feedback_spo'][key]
    assert abs(models['v2_spo']['lambda'] / original['training_tasks'] - models['old_feedback_spo']['lambda'] / original['old_tasks']) < 1e-12
    dump(ROOT / '模型冻结.json', {'models': models, 'mapping': mapping,
        'source': str(source / '候选冻结.json'), 'source_sha256': q.sha(source / '候选冻结.json'),
        'training_tasks': {'v2_spo': 115, 'no_spo_ridge': 115, 'old_feedback_spo': 79},
        'new_training': False})
    dump(ROOT / '请求冻结.json', {'contexts': old['contexts'], 'seeds': old['seeds'],
        'methods': ['author', 'v2_spo', 'no_spo_ridge', 'old_feedback_spo'],
        'max_iterations': 8, 'primary_seconds': 4., 'role': '既有起点匹配消融，非新增独立确认',
        'source': str(source / '请求冻结.json'), 'source_sha256': q.sha(source / '请求冻结.json')})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--phase', choices=['gate', 'comparison'])
    args = ap.parse_args()
    if args.prepare:
        prepare()
        return
    started = time.perf_counter()
    models = read(ROOT / '模型冻结.json')['models']
    request = read(ROOT / '请求冻结.json')
    one = stage.timed_one()
    if args.phase == 'gate':
        contexts, seeds, methods = request['contexts'][:1], request['seeds'][:1], ['author', 'v2_spo']
        refs = read(WORK / 'nea_spo_balanced_tasks_20261108_v1/direct/COMPLETE.json')['runs'] + read(WORK / 'nea_spo_balanced_tasks_20261108_v1/ablation/COMPLETE.json')['runs']
    else:
        contexts, seeds, methods = request['contexts'], request['seeds'], request['methods']
    folder = ROOT / args.phase
    folder.mkdir(exist_ok=False)
    paths = [ROOT / n for n in ['run.py', '协议.md', '模型冻结.json', '请求冻结.json']]
    hashes = {str(p): q.sha(p) for p in paths}
    upstream = read(WORK / 'nea_spo_balanced_tasks_20261108_v1/ablation/运行冻结.json')
    for p, h in upstream['input_hashes'].items():
        assert q.sha(Path(p)) == h, p
        hashes[p] = h
    dump(folder / '运行冻结.json', {'request': {'contexts': contexts, 'seeds': seeds, 'methods': methods},
        'input_hashes': hashes, 'threads': torch.get_num_threads(), 'device': 'cpu', 'phase': args.phase})
    preparation = time.perf_counter() - started
    runs, io_seconds = [], 0.
    with (folder / '原始运行.jsonl').open('x', encoding='utf-8') as stream:
        for pi, ctx in enumerate(contexts):
            for si, seed in enumerate(seeds):
                shift = (pi + si) % len(methods)
                for method in methods[shift:] + methods[:shift]:
                    if method != 'author':
                        q.MODEL = models[method]
                    cache, counts = {}, [0, 0]

                    def select(R, model, state, base, probe):
                        key = tuple(tuple(rr) for rr in R)
                        if key not in cache:
                            result = dict(stage.BASE_SELECTION(R, model, state, base, probe))
                            result['allowed'] = list(range(6))
                            result['ranking'] = sorted(range(6), key=lambda j: (-result['scores'][j], j))
                            result['chosen'] = result['ranking'][0]
                            cache[key] = result
                            counts[0] += 1
                        else:
                            counts[1] += 1
                        return cache[key]

                    q.selection = select
                    x = one(ctx, seed, 'author' if method == 'author' else 'selector', 8)
                    x.update(method=method, score_calls=counts[0], cache_hits=counts[1])
                    assert x['target_samples'] == 811 and len(x['fine_curve']) == 41
                    assert all(p['samples'] == 10 + 20 * n for n, p in enumerate(x['fine_curve']))
                    assert all(x['fine_checkpoints'][str(b)] is not None for b in [2., 4.])
                    if args.phase == 'gate':
                        old_name = 'author' if method == 'author' else 'balanced_spo_no_risk'
                        ref = next(v for v in refs if v['parent'] == ctx['parent'] and v['seed'] == seed and v['method'] == old_name)
                        for key in ['routes', 'history', 'rng', 'metrics', 'cost', 'selection_log', 'target_samples', 'target_calls', 'score_calls', 'cache_hits']:
                            assert x[key] == ref[key], key
                    runs.append(x)
                    t = time.perf_counter()
                    stream.write(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    io_seconds += time.perf_counter() - t
                    print(args.phase, ctx['parent'], seed, method, round(x['fine_checkpoints']['4.0']['cost'], 8), round(x['seconds'], 3), flush=True)
    assert all(q.sha(Path(p)) == h for p, h in hashes.items())
    dump(folder / 'COMPLETE.json', {'runs': runs, 'request': {'contexts': contexts, 'seeds': seeds, 'methods': methods},
        'inputs_unchanged': True, 'import_seconds': IMPORT_SECONDS, 'preparation_seconds': preparation,
        'raw_io_seconds': io_seconds, 'wall_seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
