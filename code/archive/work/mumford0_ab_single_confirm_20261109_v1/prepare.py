"""冻结四个未见父网、两版参数与来源，不训练。"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_io import ROOT, WORK, JOINT, MUM, b, IMPORT_SECONDS, read, sha, dump, canonical, models


def main():
    started = time.perf_counter()
    assert not (ROOT / '请求冻结.json').exists()
    hashes = {}
    for folder, phase in [(MUM, 'comparison'), (JOINT, 'confirmation')]:
        done = read(folder / phase / 'COMPLETE.json')
        assert all(sha(p) == s for p, s in done['input_hashes'].items())
        for path, value in done['input_hashes'].items():
            assert path not in hashes or hashes[path] == value
            hashes[path] = value
    ledger = WORK / 'nea_feasible_quota_20261108_v1/环境说明.json'
    env = read(ledger)
    assert sha(env['search_command_interpreter']) == env['python_executable_sha256_now']
    assert b.torch.get_num_threads() == 1 and not b.torch.cuda.is_available()
    hashes[str(ledger)] = sha(ledger)
    hashes[env['search_command_interpreter']] = sha(env['search_command_interpreter'])
    known = set()

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            if len(value) == 12 and all(isinstance(r, list) and r and
                    all(type(v) is int and 0 <= v < 30 for v in r) for r in value):
                if set().union(*map(set, value)) == set(range(30)):
                    known.add(canonical(value))
                return
            for child in value:
                if isinstance(child, (dict, list)):
                    visit(child)

    scanned = []
    for folder in [MUM, JOINT, WORK / 'mandl_multistep_spo_retrain_20261109_v1']:
        files = list(folder.glob('*.json')) + list(folder.glob('*/COMPLETE.json'))
        for path in sorted(set(files)):
            visit(read(path)); hashes[str(path)] = sha(path); scanned.append(str(path))
        path = folder / '反馈原始.jsonl'
        if path.exists():
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    visit(__import__('json').loads(line))
            hashes[str(path)] = sha(path); scanned.append(str(path))
    b.configure('mumford0')
    known.add(canonical(b.base.reference_routes()))
    q, np, spec = b.q, b.np, b.base.SPEC
    instance = q.c.BenchmarkInstance(tuple(range(1, 31)),
        {(u+1, v+1): float(q.ROADS[u, v]) for u in range(30) for v in range(30)
         if u != v and np.isfinite(q.ROADS[u, v])},
        {(u+1, v+1): float(q.Q[u, v]) for u in range(30) for v in range(30)},
        frozenset(range(1, 31)), {})
    contexts = []
    known_count = len(known)
    for ident in [62631, 62632, 62633, 62634]:
        seed = q.ref.stream(ident, 'initialization').getrandbits(64)
        routes = [[v-1 for v in rr] for rr in q.ref.make_initial_routes(instance, 12, (2, 15), seed)]
        key = canonical(routes)
        assert key not in known and q.assess(routes)['feasible'], ident
        known.add(key)
        contexts.append({'city': 'mumford0', 'parent': 'mumford0-12-' + str(ident),
                         'routes': routes, 'initialization_seed': seed, 'base': q.assess(routes)})
    docs = models()
    dump(ROOT / '模型冻结.json', {'documents': docs, 'new_training': False,
        'A_source': str(WORK / 'nea_spo_v2_matched_ablation_20261108_v1/模型冻结.json'),
        'B_source': str(JOINT / '模型冻结.json')})
    dump(ROOT / '请求冻结.json', {'contexts': contexts, 'seeds': [810, 811, 812],
        'methods': ['author', 'A_spo', 'B_spo'], 'primary_seconds': 7., 'max_iterations': 8,
        'planned_runs': 36, 'all_results_retained': True, 'new_training': False})
    for path in list(ROOT.glob('*.py')) + [ROOT / '协议.md', ROOT / '执行协议.md', ROOT / '模型冻结.json', ROOT / '请求冻结.json']:
        hashes[str(path)] = sha(path)
    dump(ROOT / '输入冻结_接口修正.json', {'input_hashes': hashes, 'canonical_history_count': known_count,
        'history_files': scanned, 'new_contexts': len(contexts), 'spec': spec,
        'prepare_seconds': time.perf_counter()-started, 'import_seconds': IMPORT_SECONDS,
        'runtime_binary_unchanged': True, 'search_threads': b.torch.get_num_threads()})
    print('PREPARED', len(contexts), 'new contexts;', known_count, 'known canonical plans', flush=True)


if __name__ == '__main__':
    main()
