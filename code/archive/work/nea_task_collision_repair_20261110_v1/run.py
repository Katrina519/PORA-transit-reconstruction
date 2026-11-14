"""两实例固定模型；唯一候选变化为同批任务预约。"""
import sys
import json
import time
import inspect
import hashlib
import argparse
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
IMPORT_START = time.perf_counter()
sys.path.insert(0, str(WORK/'nea_spo_feedback_allocation_20261108_v1'))
import run_feedback as old
sys.path.insert(0, str(WORK/'nea_spo_v2_joint_20261109_v1'))
import bridge as b
sys.path.insert(0, str(ROOT))
from batch_policy import BatchAllocator, available
r, q, torch = old.r, old.q, old.torch
IMPORT_SECONDS = time.perf_counter()-IMPORT_START


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(city):
    if city == 'mandl':
        return read(WORK/'nea_spo_feedback_allocation_20261108_v1/确认冻结.json')['model']
    return read(WORK/'nea_spo_v2_joint_20261109_v1/模型冻结.json')['document']


def one(ctx, seed, method, repair=True, iterations=8):
    started = time.perf_counter()
    city = ctx['city']; nr, limit, diameter = (6, 8, 33.) if city == 'mandl' else (12, 15, 26.)
    document = frozen(city)
    b.configure(city, document if city == 'mumford0' else None)
    b.base.make_one()  # 恢复冻结colony并安装双方共同的pass-best观察器。
    if city == 'mandl':
        q.MODEL = document
        score_function = old.BASE
    else:
        score_function = b.base.BASE_SELECTION
    torch.manual_seed(0)
    model, state = q.a.build('neural', ctx['routes'])
    selection_state = state.clone(); state.clear_routes()
    init = q.a.tu.get_batch_tensor_from_routes([ctx['routes']], max_route_len=limit)
    allocator = BatchAllocator(.05, repair=repair)
    cache, counts, logs, feedback, pending, fine, solutions, curve = {}, [0, 0], [], [], [], [], [], []
    selection_seconds = 0.; feedback_seconds = 0.; fallbacks = 0

    class Observer(r.Observer):
        def __call__(self, current):
            nonlocal feedback_seconds
            result = super().__call__(current)
            if pending:
                tick = time.perf_counter()
                metrics = result.get_metrics()
                valid = ((metrics['# disconnected node pairs'] == 0) & (metrics['# stops out of bounds'] == 0)).tolist()
                for event in pending:
                    bee = event['bee']
                    event.update(candidate_routes=r.routes(current.routes[bee]), candidate_cost=float(result.cost[bee]), valid=bool(valid[bee]))
                if city == 'mandl':
                    allocator.observe(result.cost.tolist(), valid)
                    for event, entry in zip(pending, allocator.log[-len(pending):]):
                        event['reward'] = entry['reward']
                pending.clear()
                feedback_seconds += time.perf_counter()-tick
            return result
    cost = Observer(r.cost_module())

    def hook(networks, chosen, type1):
        nonlocal selection_seconds, fallbacks
        tick = time.perf_counter(); out = chosen.clone(); details = []; reservations = {}
        assert not pending and not allocator.pending
        batch_id = len(logs)
        for bee in type1.tolist():
            routes = r.routes(networks[0, bee]); base = q.assess(routes)
            if not base['feasible']:
                fallbacks += 1; details.append({'bee': bee, 'fallback': 'infeasible_parent'}); continue
            key = tuple(tuple(rr) for rr in routes)
            if key not in cache:
                cache[key] = score_function(routes, model, selection_state, base, False); counts[0] += 1
            else:
                counts[1] += 1
            scores = cache[key]['scores']
            used = reservations.setdefault(key, set())
            eligible = available(nr, used) if repair else list(range(nr))
            if city == 'mandl':
                j = allocator.choose(key, scores, bee, base['P']/(2*diameter)+base['O']/(nr*diameter))
                assert j in eligible
            else:
                j = min(eligible, key=lambda k: (-scores[k], k))
            used.add(j); out[0, bee] = j
            event = {'batch_id': batch_id, 'bee': bee, 'j': j, 'parent_routes': routes,
                     'parent_cost': base['P']/(2*diameter)+base['O']/(nr*diameter), 'eligible': eligible}
            pending.append(event); feedback.append(event)
            details.append({'bee': bee, 'old': int(chosen[0, bee]), 'new': j, 'scores': scores, 'allowed': list(range(nr))})
        mask = torch.ones_like(chosen, dtype=torch.bool); mask[:, type1] = False
        assert torch.equal(out[mask], chosen[mask])
        selection_seconds += time.perf_counter()-tick; logs.append(details)
        return out

    def record_best(networks, values, iteration, pass_index):
        stamp = time.perf_counter()-started; value = float(values[0])
        if not fine or value < fine[-1]['cost']:
            solutions.append({'cost': value, 'routes': r.routes(networks[0]), 'first_seconds': stamp})
        fine.append({'iteration': int(iteration), 'pass': int(pass_index), 'seconds': stamp,
                     'cost': value, 'samples': cost.samples, 'solution_index': len(solutions)-1})

    class Writer:
        def add_scalar(self, name, value, step):
            if curve and curve[-1]['step'] == step:
                return
            frame = inspect.currentframe().f_back
            curve.append({'step': int(step), 'seconds': time.perf_counter()-started,
                          'cost': float(frame.f_locals['best_costs'][0]), 'samples': cost.samples})

    fn = r.ORIGINAL if method == 'author' else r.modified(hook)
    generation = [0, 0]; native_generate = r.entry.get_neural_variants
    def counted(*args, **kwargs):
        generation[0] += 1; generation[1] += int(args[2].shape[0]*args[2].shape[1])
        return native_generate(*args, **kwargs)
    if method != 'author':
        fn.__globals__['get_neural_variants'] = counted
    torch.manual_seed(seed)
    with torch.no_grad(), patch.object(r.entry, 'get_mutants', fn), \
         patch.object(r.entry, 'best_observer', record_best, create=True), \
         patch.object(r.entry, 'get_neural_variants', counted):
        out, hist = r.entry.bee_colony(state, cost, init, n_bees=10, n_type1_bees=5, n_type2_bees=5,
            n_iterations=iterations, passes_per_it=5, mod_steps_per_pass=2, shorten_prob=.2,
            silent=True, bee_model=model, sum_writer=Writer())
    rng = torch.get_rng_state().tolist(); result = cost(out)
    assert not pending and not allocator.pending
    assert cost.samples == 11+100*iterations and len(fine) == 1+5*iterations
    assert generation == [10*iterations, 100*iterations], generation
    return {'parent': ctx['parent'], 'city': city, 'seed': seed, 'method': method, 'repair': repair,
        'cost': float(result.cost[0]), 'metrics': {k: v.tolist() for k, v in result.get_metrics().items()},
        'routes': r.routes(out.routes[0]), 'history': hist.tolist(), 'rng': rng,
        'seconds': time.perf_counter()-started, 'selection_seconds': selection_seconds,
        'feedback_seconds': feedback_seconds, 'selection_log': logs, 'feedback_log': feedback,
        'fine_curve': fine, 'fine_solutions': solutions, 'curve': curve,
        'target_samples': cost.samples, 'target_calls': cost.calls, 'generation_counts': generation,
        'score_calls': counts[0], 'cache_hits': counts[1], 'fallbacks': fallbacks}


def contexts(city):
    folder = 'nea_spo_feedback_allocation_20261108_v1/确认冻结.json' if city == 'mandl' else 'mumford0_ab_single_confirm_20261109_v1/请求冻结.json'
    return [dict(c, city=city) for c in read(WORK/folder)['contexts']]


def main(phase):
    folder = ROOT/phase; folder.mkdir(exist_ok=False)
    sources = read(WORK/'nea_spo_feedback_allocation_20261108_v1/direct/运行冻结.json')['input_hashes']
    sources.update(read(WORK/'mumford0_ab_single_confirm_20261109_v1/输入冻结_接口修正.json')['input_hashes'])
    sources.update({str(p): sha(p) for p in ROOT.glob('*.py')})
    assert all(sha(p) == h for p, h in sources.items())
    if phase.startswith('gate'):
        tasks = [(city, c, seed, method) for city, ident, seed in [('mandl', 62523, 351), ('mumford0', 62634, 810)]
                 for c in contexts(city) if c['parent'].endswith(str(ident)) for method in ['author', 'ours']]
    else:
        assert read(ROOT/'gate_retry/COMPLETE.json')['passed']
        tasks = [(city, c, seed, method) for city, ids, seeds in [('mandl', [62521,62523], [350,351,352,353]), ('mumford0',[62633,62634],[810,812])]
                 for c in contexts(city) if int(c['parent'].split('-')[-1]) in ids for seed in seeds
                 for method in (['author','ours'] if seed%2 == 0 else ['ours','author'])]
    dump(folder/'运行冻结.json', {'tasks': [{'city': city, 'context': ctx, 'seed': seed, 'method': method} for city,ctx,seed,method in tasks],
        'input_hashes': sources, 'import_seconds': IMPORT_SECONDS, 'threads': torch.get_num_threads()})
    runs = []
    with (folder/'原始运行.jsonl').open('x', encoding='utf-8') as stream:
        for city, ctx, seed, method in tasks:
            value = one(ctx, seed, method, repair=not phase.startswith('gate'))
            if phase.startswith('gate'):
                oldfolder = 'nea_spo_feedback_allocation_20261108_v1/direct' if city == 'mandl' else 'mumford0_ab_single_confirm_20261109_v1/comparison'
                oldmethod = 'author' if method == 'author' else ('feedback_spo' if city == 'mandl' else 'B_spo')
                ref = next(x for x in read(WORK/oldfolder/'COMPLETE.json')['runs'] if x['parent'] == ctx['parent'] and x['seed'] == seed and x['method'] == oldmethod)
                for key in ['routes','history','rng','metrics','cost','selection_log','target_samples','target_calls','score_calls','cache_hits']:
                    assert value[key] == ref[key], (city, method, key)
            runs.append(value); stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
            t = 4. if city == 'mandl' else 7.
            point = next(p for p in value['fine_curve'][::-1] if p['seconds'] <= t)
            print(phase, city, ctx['parent'], seed, method, round(point['cost'],8), round(value['cost'],8), round(value['seconds'],3), flush=True)
    assert all(sha(p) == h for p, h in sources.items())
    dump(folder/'COMPLETE.json', {'passed': True, 'input_hashes': sources, 'runs': runs, 'import_seconds': IMPORT_SECONDS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['gate','gate_retry','development'], required=True)
    main(parser.parse_args().phase)
