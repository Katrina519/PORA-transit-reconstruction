"""只补用户最终Excel中的Mumford0在线Full及匹配Ridge。"""
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
SOURCE = WORK / 'nea_frozen_multiseed_20261111_v1'
REQUEST = ROOT / '请求冻结_40组.json'
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(hashes):
    bad = [p for p, h in hashes.items() if sha(p) != h]
    assert not bad, bad


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def load_runtime():
    pilot = module('prior_online_pilot', WORK / 'mumford0_B_allocation_check_20261112_v1/experiment.py')
    return pilot, pilot.load_runtime()


def make(pilot, runtime, variant):
    assert variant in ['full', 'ridge']
    fn = pilot.make(runtime, True)
    if variant == 'ridge':
        models = read(WORK / 'nea_paper_final_confirm_20261110_v1/模型冻结.json')['ridge']
        fn.__globals__['frozen'] = lambda city: models[city]
    return fn


def one(pilot, runtime, ctx, seed, variant):
    assert ctx['city'] == 'mumford0'
    started = time.perf_counter()
    function = make(pilot, runtime, variant)
    preparation = time.perf_counter() - started
    result = function(ctx, seed, 'ours', repair=False)
    result['kernel_seconds'] = result['seconds']
    for field in ['fine_curve', 'curve']:
        for point in result[field]:
            point['seconds'] += preparation
    for solution in result['fine_solutions']:
        solution['first_seconds'] += preparation
    result['wrapper_preparation_seconds'] = preparation
    result['seconds'] = time.perf_counter() - started
    result['variant'] = variant
    result['allocation'] = 'prior_plus_online_gain_c005'
    return result


def prepare():
    source = read(SOURCE / '请求冻结.json')
    check(source['hashes'])
    table = read(ROOT / '输入CSV.json')
    rows = table['rows']
    assert len(rows) == 80 and sum(r['city']=='mandl' for r in rows)==40
    selected = [r for r in rows if r['city']=='mumford0']
    assert len(selected)==40
    contexts = {c['parent']: c for c in source['contexts'] if c['city']=='mumford0'}
    assert sorted({r['parent'] for r in selected}) == sorted(contexts)
    tasks = []
    for pi, parent in enumerate(sorted(contexts)):
        seeds = [r['seed'] for r in selected if r['parent']==parent]
        assert seeds == list(range(2000,2010))
        for si, seed in enumerate(seeds):
            variants = ['full','ridge'] if (pi+si)%2==0 else ['ridge','full']
            tasks.extend([contexts[parent],seed,v] for v in variants)
    hashes = dict(source['hashes'])
    paths = [Path(table['source']), SOURCE/'comparison/原始运行.jsonl', ROOT/'输入CSV.json',
             WORK/'mumford0_B_allocation_check_20261112_v1/experiment.py',
             WORK/'mumford0_B_allocation_check_20261112_v1/COMPLETE.json']
    paths += list(ROOT.glob('*.py')) + list(ROOT.glob('*.md'))
    for p in paths:
        hashes[str(p)] = sha(p)
    assert sha(table['source']) == table['source_sha256']
    dump(REQUEST, dict(date='2026-11-12', source_table=table['source'],
         source_table_sha256=table['source_sha256'], rows=rows, tasks=tasks, hashes=hashes,
         primary_endpoint='fixed_8_rounds', historical_budget={'mandl':4.,'mumford0':7.},
         no_author_runs=True, no_training=True, no_tuning=True, methods=['author','full','count','ridge']))
    print('FROZEN',len(tasks),'new runs;',len(hashes),'source hashes',flush=True)


def gate():
    req = read(REQUEST); check(req['hashes'])
    pilot, runtime = load_runtime()
    reference = next(r for r in read(WORK/'mumford0_B_allocation_check_20261112_v1/COMPLETE.json')['runs']
                     if r['parent']=='mumford0-12-62631' and r['seed']==63001 and r['variant']=='online_B')
    ctx = next(c for c in runtime.d.contexts('mumford0') if c['parent']==reference['parent'])
    result = one(pilot,runtime,ctx,63001,'full')
    fields = ['routes','history','rng','metrics','cost','selection_log','feedback_log',
              'target_samples','target_calls','generation_counts','score_calls','cache_hits','fallbacks']
    for field in fields:
        assert result[field]==reference[field], field
    dump(ROOT/'门禁原始运行.json',result)
    dump(ROOT/'门禁.json',dict(passed=True,fields=fields,seconds=result['seconds']))
    print('GATE PASSED',fields,flush=True)


def run():
    req = read(REQUEST); check(req['hashes'])
    assert read(ROOT/'门禁.json')['passed'] and (ROOT/'独立接口审查.md').exists()
    tick = time.perf_counter(); pilot,runtime=load_runtime(); imported=time.perf_counter()-tick
    raw = ROOT/'原始运行.jsonl'
    runs = [json.loads(line) for line in raw.read_text(encoding='utf-8').splitlines()] if raw.exists() else []
    assert not (ROOT/'COMPLETE.json').exists()
    for result,(ctx,seed,variant) in zip(runs,req['tasks']):
        assert (result['parent'],result['seed'],result['variant'])==(ctx['parent'],seed,variant)
    started=time.perf_counter();io=0.
    with raw.open('a' if raw.exists() else 'x',encoding='utf-8') as stream:
        for i,(ctx,seed,variant) in enumerate(req['tasks'][len(runs):],start=len(runs)+1):
            result=one(pilot,runtime,ctx,seed,variant)
            assert result['generation_counts']==[80,800] and result['target_samples']==811 and result['target_calls']==82
            assert len(result['fine_curve'])==41 and len(result['feedback_log'])==400
            runs.append(result);tick=time.perf_counter()
            stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();io+=time.perf_counter()-tick
            point=[p for p in result['fine_curve'] if p['seconds']<=7.][-1]
            print('RUN',i,'/80',ctx['parent'],seed,variant,'7s',round(point['cost'],8),
                  '8round',round(result['cost'],8),'seconds',round(result['seconds'],3),flush=True)
    assert len(runs)==80
    check(req['hashes'])
    dump(ROOT/'COMPLETE.json',dict(passed=True,runs=runs,hashes=req['hashes'],
        request_sha256=sha(REQUEST),import_seconds=imported,
        batch_seconds=time.perf_counter()-started,raw_io_seconds=io))
    print('COMPLETE',len(runs),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'gate':gate,'run':run}[sys.argv[1]]()
