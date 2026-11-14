"""仅扩充固定四组的随机重复；复用旧driver，不改搜索逻辑。"""
import sys,json,time,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'nea_paper_final_confirm_20261110_v1'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def load_runtime():
    spec=importlib.util.spec_from_file_location('multiseed_unchanged_paper_driver',OLD/'driver.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.d.torch.set_num_threads(1)
    return module


def check(req):
    assert all(sha(p)==h for p,h in req['hashes'].items())


def schedule(contexts,seeds,methods):
    tasks=[]
    for pi,ctx in enumerate(contexts):
        for si,seed in enumerate(seeds):
            shift=(pi+si)%len(methods)
            tasks.extend([ctx,seed,m] for m in methods[shift:]+methods[:shift])
    return tasks


def prepare():
    previous=read(OLD/'请求冻结.json');sources=read(OLD/'comparison_timed/COMPLETE.json')['hashes']
    assert all(sha(p)==h for p,h in sources.items())
    sources.update({str(OLD/n):sha(OLD/n) for n in ['driver.py','请求冻结.json','模型冻结.json','gate_timed/COMPLETE.json','交付封存.json']})
    sources.update({str(p):sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md']})
    contexts=previous['contexts'];seeds=list(range(2000,2010));methods=previous['methods']
    assert len(contexts)==8 and methods==['author','full','count','ridge'] and not(set(seeds)&set(previous['seeds']))
    tasks=schedule(contexts,seeds,methods);assert len(tasks)==320
    req={'contexts':contexts,'seeds':seeds,'methods':methods,'budgets':previous['budgets'],'iterations':8,
         'tasks':tasks,'hashes':sources,'no_training_or_parameter_changes':True,'new_initial_parents':False,
         'scope':'320 fixed runs; ten new search seeds per existing parent; negative results retained',
         'source_request_sha256':sha(OLD/'请求冻结.json'),'date':'2026-11-11'}
    check(req);dump(ROOT/'请求冻结.json',req)
    print('FROZEN',len(contexts),'parents',len(seeds),'seeds',len(tasks),'formal runs',flush=True)


def main(phase):
    assert phase in ['gate','comparison']
    req=read(ROOT/'请求冻结.json');check(req)
    if phase=='comparison':assert read(ROOT/'gate/COMPLETE.json')['passed']
    started_import=time.perf_counter();runtime=load_runtime();import_seconds=time.perf_counter()-started_import
    previous_gate=read(OLD/'gate_timed/COMPLETE.json')['runs'] if phase=='gate' else None
    if phase=='gate':
        contexts=runtime.d.contexts('mandl')[:1]+runtime.d.contexts('mumford0')[:1]
        tasks=[[ctx,350 if ctx['city']=='mandl' else 810,m] for ctx in contexts for m in req['methods']]
    else:tasks=req['tasks']
    folder=ROOT/phase
    if not folder.exists():
        folder.mkdir();dump(folder/'运行冻结.json',{'tasks':tasks,'hashes':req['hashes'],'threads':runtime.d.torch.get_num_threads(),'request_sha256':sha(ROOT/'请求冻结.json')})
    frozen=read(folder/'运行冻结.json');assert frozen['tasks']==tasks and frozen['request_sha256']==sha(ROOT/'请求冻结.json')
    assert not (folder/'COMPLETE.json').exists()
    raw=folder/'原始运行.jsonl';runs=[json.loads(s) for s in raw.read_text('utf-8').splitlines()] if raw.exists() else []
    assert len(runs)<=len(tasks)
    for r,(ctx,seed,method) in zip(runs,tasks):assert (r['parent'],r['seed'],r['variant'])==(ctx['parent'],seed,method)
    started=time.perf_counter();io=0.
    with raw.open('a' if raw.exists() else 'x',encoding='utf-8') as stream:
        for ctx,seed,method in tasks[len(runs):]:
            r=runtime.one(ctx,seed,method)
            assert r['generation_counts']==[80,800] and r['target_samples']==811 and r['target_calls']==82
            assert len(r['fine_curve'])==41
            if phase=='gate':
                ref=next(x for x in previous_gate if (x['parent'],x['seed'],x['variant'])==(ctx['parent'],seed,method))
                for key in ['routes','cost','history','rng','metrics','selection_log','feedback_log','target_samples','target_calls','generation_counts','score_calls','cache_hits','fallbacks']:
                    assert r[key]==ref[key],(ctx['city'],method,key)
            runs.append(r);tick=time.perf_counter();stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();io+=time.perf_counter()-tick
            point=[p for p in r['fine_curve'] if p['seconds']<=req['budgets'][ctx['city']]][-1]
            print(phase,len(runs),'/',len(tasks),ctx['parent'],seed,method,'budget',point['cost'],'final',r['cost'],'seconds',round(r['seconds'],3),flush=True)
    check(req)
    dump(folder/'COMPLETE.json',{'passed':True,'runs':runs,'hashes':req['hashes'],'phase_seconds':time.perf_counter()-started,
         'raw_io_seconds_this_invocation':io,'import_seconds_this_invocation':import_seconds,'request_sha256':sha(ROOT/'请求冻结.json')})


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:main(sys.argv[1])
