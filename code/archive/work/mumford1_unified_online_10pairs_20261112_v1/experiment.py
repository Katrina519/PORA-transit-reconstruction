"""Mumford1十组配对；作者内核不改，仅Full启用冻结在线分配。"""
import os
import sys
import time
import json
import inspect
import hashlib
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
PILOT=WORK/'mumford1_frozen_B_pilot_20261112_v1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',WANDB_MODE='disabled')
ACTIVE={}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def check(hashes):
    bad=[p for p,h in hashes.items() if sha(p)!=h]
    assert not bad,bad


def load_runtime():
    pilot=module('validated_mumford1_interface',PILOT/'pilot.py')
    runtime=pilot.load_runtime();runtime.d.torch.set_num_threads(1)
    return pilot,runtime


def progress(method,iteration,pass_index,seconds,value):
    event=dict(phase=ACTIVE['phase'],seed=ACTIVE['seed'],variant=ACTIVE['variant'],
               iteration=int(iteration),pass_index=int(pass_index),
               seconds=seconds+ACTIVE.get('preparation',0.),cost=value)
    with (ROOT/'进度.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,allow_nan=False)+'\n')
    print('PASS',event['phase'],event['seed'],event['variant'],event['iteration'],event['pass_index'],
          round(event['seconds'],3),round(value,8),flush=True)


def make(pilot,runtime,variant):
    assert variant in ['author','full']
    source=inspect.getsource(runtime.d.one)
    changes=[
        ("nr, limit, diameter = (6, 8, 33.) if city == 'mandl' else (12, 15, 26.)",'nr, limit, diameter = 15, 30, 44.'),
        ("b.configure(city, document if city == 'mumford0' else None)",'configure_mumford1(document)'),
        ("                if city == 'mandl':\n                    allocator.observe",'                if online:\n                    allocator.observe'),
        ("            if city == 'mandl':\n                j = allocator.choose",'            if online:\n                j = allocator.choose')]
    for old,new in changes:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    anchor="                     'cost': value, 'samples': cost.samples, 'solution_index': len(solutions)-1})"
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'\n        progress(method, int(iteration), int(pass_index), stamp, value)')
    namespace=dict(runtime.d.__dict__,online=variant=='full',
                   configure_mumford1=lambda doc:pilot.configure(runtime,doc),progress=progress)
    exec(compile(source,str(ROOT/'experiment.py')+'::frozen_mumford1_online', 'exec'),namespace)
    return namespace['one']


def one(pilot,runtime,ctx,seed,variant,iterations=8,phase='formal'):
    global ACTIVE
    started=time.perf_counter()
    ACTIVE=dict(phase=phase,seed=seed,variant=variant)
    fn=make(pilot,runtime,variant)
    preparation=time.perf_counter()-started
    ACTIVE['preparation']=preparation
    result=fn(ctx,seed,'author' if variant=='author' else 'ours',repair=False,iterations=iterations)
    result['kernel_seconds']=result['seconds']
    for field in ['fine_curve','curve']:
        for point in result[field]:point['seconds']+=preparation
    for solution in result['fine_solutions']:solution['first_seconds']+=preparation
    result['wrapper_preparation_seconds']=preparation
    result['seconds']=time.perf_counter()-started
    result['variant']=variant
    result['iterations']=iterations
    result['allocation']='author_original' if variant=='author' else 'SPO_prior_online_gain_c005'
    return result


def prepare():
    import numpy as np
    previous=read(PILOT/'请求冻结.json');check(previous['hashes'])
    ctx=previous['context']
    roads=np.loadtxt(PILOT/'inputs/Mumford1TravelTimes.txt')
    demand=np.loadtxt(PILOT/'inputs/Mumford1Demand.txt')
    assert roads.shape==demand.shape==(70,70) and len(ctx['routes'])==15
    assert set().union(*map(set,ctx['routes']))==set(range(70))
    assert all(10<=len(r)<=30 and len(r)==len(set(r)) and all(0<roads[u,v]<np.inf for u,v in zip(r,r[1:])) for r in ctx['routes'])
    seeds=list(range(3011,3021));tasks=[]
    for index,seed in enumerate(seeds):
        variants=['author','full'] if index%2==0 else ['full','author']
        tasks.extend([ctx,seed,v] for v in variants)
    frozen=ROOT/'frozen_N24';frozen.mkdir(exist_ok=False)
    n24=WORK/'nea_N24_two_sided_20261112_v1'
    result=read(n24/'双侧检验结果.json')
    assert sha(result['source'])==result['source_sha256'] and result['mandl_unchanged']
    copies=[(Path(result['source']),'指定N24数据.xlsx'),(n24/'双侧检验结果.json','N24检验记录.json'),
            (n24/'统计说明.md','N24来源说明.md'),
            (WORK/'mumford0_unified_online_excel_20261112_v1/全部逐次_统一在线方法.csv','完整40组每网络记录.csv')]
    for original,name in copies:(frozen/name).write_bytes(original.read_bytes())
    dump(frozen/'版本封存.json',dict(date='2026-11-12',scope=result['scope'],timing_scope=result['timing_scope'],
         main=result['main'],sources={str(p):sha(p) for p,_ in copies},
         copies={p.name:sha(p) for p in frozen.iterdir() if p.is_file()}))
    hashes=dict(previous['hashes'])
    for p in [PILOT/'请求冻结.json',PILOT/'原始运行.jsonl',PILOT/'逐方案复核.json']+list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(frozen.iterdir()):
        hashes[str(p)]=sha(p)
    dump(ROOT/'请求冻结.json',dict(date='2026-11-12',context=ctx,seeds=seeds,methods=['author','full'],tasks=tasks,
         primary_endpoint='fixed_8_rounds',secondary_seconds=120.,iterations=8,c=.05,repair=False,
         frozen_model_source=str(WORK/'nea_spo_v2_joint_20261109_v1/模型冻结.json'),hashes=hashes,
         no_training=True,no_tuning=True,no_author_or_generator_changes=True,one_fixed_parent=True))
    print('FROZEN',len(tasks),'runs;',len(hashes),'source hashes',flush=True)


def gate():
    req=read(ROOT/'请求冻结.json');check(req['hashes']);pilot,runtime=load_runtime()
    old=next(r for r in [json.loads(line) for line in (PILOT/'原始运行.jsonl').read_text(encoding='utf-8').splitlines()] if r['variant']=='author')
    result=one(pilot,runtime,req['context'],3001,'author',iterations=1,phase='gate')
    assert len(result['fine_curve'])==6 and result['target_samples']==111 and result['generation_counts']==[10,100]
    for current,reference in zip(result['fine_curve'],old['fine_curve'][:6]):
        assert current['cost']==reference['cost'] and current['samples']==reference['samples']
        assert result['fine_solutions'][current['solution_index']]['routes']==old['fine_solutions'][reference['solution_index']]['routes']
    assert not result['selection_log'] and not result['feedback_log']
    dump(ROOT/'门禁原始运行.json',result)
    dump(ROOT/'门禁.json',dict(passed=True,author_prefix_passes=5,source_seed=3001,
         independent_formal_seeds=req['seeds'],seconds=result['seconds'],
         environment=dict(python=sys.version,torch=runtime.d.torch.__version__,threads=runtime.d.torch.get_num_threads(),cuda=runtime.d.torch.cuda.is_available())))
    print('GATE PASSED unchanged author first iteration',flush=True)


def run():
    req=read(ROOT/'请求冻结.json');check(req['hashes'])
    assert read(ROOT/'门禁.json')['passed'] and (ROOT/'独立接口审查.md').exists()
    assert not (ROOT/'COMPLETE.json').exists()
    tick=time.perf_counter();pilot,runtime=load_runtime();imported=time.perf_counter()-tick
    path=ROOT/'原始运行.jsonl'
    runs=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    assert len(runs)<=20
    for r,(ctx,seed,v) in zip(runs,req['tasks']):assert (r['parent'],r['seed'],r['variant'])==(ctx['parent'],seed,v)
    started=time.perf_counter();io=0.
    with path.open('a' if path.exists() else 'x',encoding='utf-8') as stream:
        for index,(ctx,seed,variant) in enumerate(req['tasks'][len(runs):],start=len(runs)+1):
            result=one(pilot,runtime,ctx,seed,variant)
            assert result['generation_counts']==[80,800] and result['target_samples']==811 and result['target_calls']==82
            assert len(result['fine_curve'])==41
            assert len(result['feedback_log'])==(0 if variant=='author' else 400)
            runs.append(result);tick=time.perf_counter()
            stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();io+=time.perf_counter()-tick
            point=[p for p in result['fine_curve'] if p['seconds']<=120.][-1]
            print('DONE',index,'/20',seed,variant,'120s',round(point['cost'],8),'8round',round(result['cost'],8),
                  'seconds',round(result['seconds'],3),flush=True)
            check(req['hashes'])
    assert len(runs)==20
    dump(ROOT/'COMPLETE.json',dict(passed=True,runs=runs,hashes=req['hashes'],request_sha256=sha(ROOT/'请求冻结.json'),
         import_seconds=imported,batch_seconds_this_invocation=time.perf_counter()-started,raw_io_seconds=io))
    print('COMPLETE 20',flush=True)


if __name__=='__main__':{'prepare':prepare,'gate':gate,'run':run}[sys.argv[1]]()
