"""Mumford1五配置扩展消融；只在新目录保存，不修改冻结来源。"""
import os
import sys
import json
import time
import inspect
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PREVIOUS = WORK / 'mumford1_unified_online_10pairs_20261112_v1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled')
sys.path.insert(0, str(WORK / 'nea_spo_feedback_allocation_20261108_v1'))
from allocator import Allocator

METHODS = ['author', 'count', 'ridge', 'nogain', 'full']
ACTIVE = {}


class NoGainAllocator(Allocator):
    def __init__(self, c=.05, repair=False):
        assert not repair
        super().__init__(c=c, use_feedback=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def check(hashes):
    changed = [p for p, expected in hashes.items() if sha(p) != expected]
    assert not changed, changed


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def load_runtime():
    previous = module('previous_mumford1_interface', PREVIOUS / 'experiment.py')
    pilot, runtime = previous.load_runtime()
    runtime.d.torch.set_num_threads(1)
    return pilot, runtime


def documents():
    return read(WORK / 'nea_paper_final_confirm_20261110_v1' / '模型冻结.json')


def configure(pilot, runtime, document):
    models = documents()
    assert document in [models[m]['mumford0'] for m in ['full', 'ridge']]
    # 共用既有Mumford1实例适配及神经构造器；仅替换冻结评分回调。
    pilot.configure(runtime, models['full']['mumford0'])
    runtime.d.b.base.BASE_SELECTION.__globals__['joint_score'] = (
        lambda x: runtime.d.b.learner.scores(document, x, 'mumford0').tolist())


def progress(method, iteration, pass_index, seconds, cost):
    event = dict(ACTIVE, iteration=int(iteration), pass_index=int(pass_index),
                 seconds=float(seconds + ACTIVE.get('preparation', 0.)), cost=float(cost))
    with (ROOT / '进度.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
    print('PASS', event['phase'], event['seed'], event['variant'],
          event['iteration'], event['pass_index'], round(event['seconds'], 3),
          round(event['cost'], 8), flush=True)


def make(pilot, runtime, variant):
    assert variant in METHODS
    source = inspect.getsource(runtime.d.one)
    changes = [
        ("nr, limit, diameter = (6, 8, 33.) if city == 'mandl' else (12, 15, 26.)",
         'nr, limit, diameter = 15, 30, 44.'),
        ("b.configure(city, document if city == 'mumford0' else None)",
         'configure_mumford1(document)'),
        ("                if city == 'mandl':\n                    allocator.observe",
         '                if online:\n                    allocator.observe'),
        ("            if city == 'mandl':\n                j = allocator.choose",
         '            if online:\n                j = allocator.choose')]
    if variant == 'count':
        changes.extend([
            ('allocator = BatchAllocator(.05, repair=repair)',
             'allocator = BatchAllocator(.05, repair=repair)\n    counter = CountPolicy()'),
            ('cache[key] = score_function(routes, model, selection_state, base, False); counts[0] += 1',
             "cache[key] = {'scores': [0.] * nr}; counts[0] += 1"),
            ('j = min(eligible, key=lambda k: (-scores[k], k))', 'j = counter.choose(key, nr)')])
    for old, new in changes:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    anchor = "                     'cost': value, 'samples': cost.samples, 'solution_index': len(solutions)-1})"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor +
        '\n        progress(method, int(iteration), int(pass_index), stamp, value)' +
        "\n        fine[-1]['seconds'] = time.perf_counter() - started" +
        "\n        if solutions[-1]['first_seconds'] == stamp:" +
        "\n            solutions[-1]['first_seconds'] = fine[-1]['seconds']")
    model = documents()['ridge' if variant == 'ridge' else 'full']['mumford0']
    scope = dict(runtime.d.__dict__, online=variant in ['full', 'ridge', 'nogain'],
                 frozen=lambda city: model,
                 configure_mumford1=lambda document: configure(pilot, runtime, document),
                 progress=progress, CountPolicy=runtime.CountPolicy)
    if variant == 'nogain':
        scope['BatchAllocator'] = NoGainAllocator
    exec(compile(source, str(ROOT / 'experiment.py') + '::' + variant, 'exec'), scope)
    return scope['one']


def one(pilot, runtime, context, seed, variant, iterations=8, phase='formal'):
    global ACTIVE
    started = time.perf_counter()
    ACTIVE = dict(phase=phase, seed=seed, variant=variant)
    fn = make(pilot, runtime, variant)
    preparation = time.perf_counter() - started
    ACTIVE['preparation'] = preparation
    value = fn(context, seed, 'author' if variant == 'author' else 'ours',
               repair=False, iterations=iterations)
    value['kernel_seconds'] = value['seconds']
    for field in ['fine_curve', 'curve']:
        for point in value[field]:
            point['seconds'] += preparation
    for solution in value['fine_solutions']:
        solution['first_seconds'] += preparation
    value['wrapper_preparation_seconds'] = preparation
    value['seconds'] = time.perf_counter() - started
    value.update(variant=variant, iterations=iterations,
                 use_gain=variant in ['full', 'ridge'],
                 model_source='ridge_mumford0' if variant == 'ridge' else 'full_mumford0')
    return value


def prepare():
    old = read(PREVIOUS / '请求冻结.json')
    check(old['hashes'])
    context = old['context']
    assert old['seeds'] == list(range(3011, 3021))
    tasks = []
    for index, seed in enumerate(old['seeds']):
        order = [(j + index % 5) % 5 for j in [0, 1, 4, 2, 3]]
        if index >= 5:
            order.reverse()
        tasks.extend(dict(seed=seed, variant=METHODS[j], position=position + 1)
                     for position, j in enumerate(order))
    hashes = dict(old['hashes'])
    paths = list(ROOT.glob('*.py')) + [ROOT / '协议.md', PREVIOUS / '请求冻结.json',
             WORK / 'nea_task_collision_repair_20261110_v1' / 'batch_policy.py',
             PREVIOUS / '原始运行.jsonl', PREVIOUS / '逐次方法结果.csv',
             WORK / 'mumford0_unified_online_excel_20261112_v1' / '全部逐次_统一在线方法.csv',
             WORK / 'PORA_manuscript_figure_revision_20261114_v7' / 'main.pdf']
    manuscript = WORK / 'PORA_manuscript_figure_revision_20261114_v7'
    paths += list(manuscript.glob('*.tex')) + list(manuscript.glob('*.bib'))
    paths += list((manuscript / 'sections').glob('*.tex'))
    paths += list((manuscript / 'texpage').glob('*.tex'))
    for path in paths:
        hashes[str(path)] = sha(path)
    dump(ROOT / '请求冻结.json', dict(date='2026-11-14', context=context,
         seeds=old['seeds'], methods=METHODS, tasks=tasks, iterations=8,
         primary_endpoint='8round', secondary_seconds=120., c=.05,
         contrasts=['full_vs_count', 'full_vs_ridge', 'full_vs_nogain'],
         correction='Holm within the three prespecified contrasts at each endpoint',
         epsilon=1e-6, no_training=True, no_tuning=True, repair=False,
         source_request=str(PREVIOUS / '请求冻结.json'), hashes=hashes,
         scope='one fixed initial network; existing seed set; expanded ablation batch'))
    print('FROZEN', len(tasks), 'runs;', len(hashes), 'protected sources', flush=True)


def gate():
    req = read(ROOT / '请求冻结.json')
    check(req['hashes'])
    pilot, runtime = load_runtime()
    old_runs = [json.loads(line) for line in
                (PREVIOUS / '原始运行.jsonl').read_text(encoding='utf-8').splitlines()]
    checks = []
    for variant in METHODS:
        result = one(pilot, runtime, req['context'], 3011, variant,
                     iterations=1, phase='gate')
        assert result['generation_counts'] == [10, 100]
        assert result['target_samples'] == 111 and len(result['fine_curve']) == 6
        if variant in ['author', 'full']:
            reference = next(r for r in old_runs if r['seed'] == 3011 and r['variant'] == variant)
            for a, b in zip(result['fine_curve'], reference['fine_curve'][:6]):
                assert a['cost'] == b['cost'] and a['samples'] == b['samples']
                assert result['fine_solutions'][a['solution_index']]['routes'] == reference['fine_solutions'][b['solution_index']]['routes']
        assert len(result['feedback_log']) == (0 if variant == 'author' else 50)
        dump(ROOT / ('门禁_' + variant + '.json'), result)
        checks.append(dict(variant=variant, passed=True, cost=result['cost'], seconds=result['seconds']))
        check(req['hashes'])
    dump(ROOT / '门禁.json', dict(passed=True, checks=checks,
         environment=dict(python=sys.version, torch=runtime.d.torch.__version__,
                          threads=runtime.d.torch.get_num_threads(), cuda=False)))
    print('GATE PASSED all five methods; author/full prefixes preserved', flush=True)


def run():
    req = read(ROOT / '请求冻结.json')
    check(req['hashes'])
    assert read(ROOT / '门禁.json')['passed']
    assert read(ROOT / '独立接口审查.json')['launch_allowed']
    assert not (ROOT / 'COMPLETE.json').exists()
    tick = time.perf_counter()
    pilot, runtime = load_runtime()
    imported = time.perf_counter() - tick
    path = ROOT / '原始运行.jsonl'
    previous = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    assert [(r['seed'], r['variant']) for r in previous] == [(t['seed'], t['variant']) for t in req['tasks'][:len(previous)]]
    started = time.perf_counter()
    io = 0.
    with path.open('a' if path.exists() else 'x', encoding='utf-8') as stream:
        for index, task in enumerate(req['tasks'][len(previous):], start=len(previous) + 1):
            result = one(pilot, runtime, req['context'], task['seed'], task['variant'])
            assert result['generation_counts'] == [80, 800]
            assert result['target_samples'] == 811 and result['target_calls'] == 82
            assert len(result['fine_curve']) == 41 and not result['fallbacks']
            assert len(result['feedback_log']) == (0 if task['variant'] == 'author' else 400)
            tick = time.perf_counter()
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            io += time.perf_counter() - tick
            point = [p for p in result['fine_curve'] if p['seconds'] <= 120.][-1]
            print('DONE', index, '/50', task['seed'], task['variant'],
                  '120s', round(point['cost'], 8), '8round', round(result['cost'], 8),
                  'seconds', round(result['seconds'], 3), flush=True)
            check(req['hashes'])
    dump(ROOT / 'COMPLETE.json', dict(passed=True, completed_runs=50,
         request_sha256=sha(ROOT / '请求冻结.json'), import_seconds=imported,
         batch_seconds_this_invocation=time.perf_counter() - started,
         raw_io_seconds_this_invocation=io, all_protected_sources_unchanged=True))
    print('COMPLETE 50', flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'gate': gate, 'run': run}[sys.argv[1]]()
