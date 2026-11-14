"""一次冻结的四组最终确认；支持按原清单安全续跑。"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from driver import d,WORK,one,IMPORT_SECONDS

def hashes():
    values=d.read(WORK/'nea_selection_final_repair_20261110_v1/gate/COMPLETE.json')['hashes']
    values.update({str(p):d.sha(p) for p in ROOT.iterdir() if p.suffix in ['.py','.md','.json'] and p.is_file()})
    return values

def main(phase):
    assert phase in ['gate','comparison']
    folder=ROOT/(phase+'_timed')
    if phase=='comparison':assert d.read(ROOT/'gate_timed/COMPLETE.json')['passed']
    req=d.read(ROOT/'请求冻结.json');tasks=[]
    if phase=='gate':
        for city in ['mandl','mumford0']:
            context=d.contexts(city)[0];seed=350 if city=='mandl' else 810
            tasks.extend((context,seed,m) for m in ['author','full','count','ridge'])
    else:
        for pi,context in enumerate(req['contexts']):
            for si,seed in enumerate(req['seeds']):
                methods=req['methods'];shift=(pi+si)%4
                tasks.extend((context,seed,m) for m in methods[shift:]+methods[:shift])
    if not folder.exists():
        folder.mkdir();sources=hashes()
        if phase=='comparison':assert all(sources.get(p)==h for p,h in d.read(ROOT/'gate_timed/COMPLETE.json')['hashes'].items())
        d.dump(folder/'运行冻结.json',{'tasks':tasks,'hashes':sources,'threads':d.torch.get_num_threads(),'import_seconds':IMPORT_SECONDS})
    frozen=d.read(folder/'运行冻结.json');sources=frozen['hashes']
    assert json.loads(json.dumps(tasks))==frozen['tasks'] and all(d.sha(p)==h for p,h in sources.items())
    assert not (folder/'COMPLETE.json').exists()
    raw=folder/'原始运行.jsonl';runs=[json.loads(s) for s in raw.read_text(encoding='utf-8').splitlines()] if raw.exists() else []
    assert len(runs)<=len(tasks)
    for r,(c,s,m) in zip(runs,tasks):assert (r['parent'],r['seed'],r['variant'])==(c['parent'],s,m)
    started=time.perf_counter();io_seconds=0.
    with raw.open('a' if raw.exists() else 'x',encoding='utf-8') as f:
        for ctx,seed,method in tasks[len(runs):]:
            result=one(ctx,seed,method)
            if phase=='gate' and method in ['author','full']:
                path='nea_spo_feedback_allocation_20261108_v1/direct' if ctx['city']=='mandl' else 'mumford0_ab_single_confirm_20261109_v1/comparison'
                label='author' if method=='author' else 'feedback_spo' if ctx['city']=='mandl' else 'B_spo'
                ref=next(v for v in d.read(WORK/path/'COMPLETE.json')['runs'] if v['parent']==ctx['parent'] and v['seed']==seed and v['method']==label)
                for key in ['routes','cost','history','rng','metrics','selection_log','target_samples','target_calls']:assert result[key]==ref[key],(ctx['city'],method,key)
            if phase=='gate' and method=='count':
                assert [x['new'] for x in result['selection_log'][0]]==list(range(5))
            if phase=='gate' and method=='ridge':
                document=d.read(ROOT/'模型冻结.json')['ridge'][ctx['city']]
                X=d.b.features(ctx['routes'])
                predicted=d.q.ref.selector.scores(document,X) if ctx['city']=='mandl' else d.b.learner.scores(document,X,ctx['city']).tolist()
                assert max(abs(a-b) for a,b in zip(predicted,result['selection_log'][0][0]['scores']))<1e-9
            runs.append(result);tick=time.perf_counter();f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();io_seconds+=time.perf_counter()-tick
            points=[p for p in result['fine_curve'] if p['seconds']<=req['budgets'][ctx['city']]]
            print(phase,len(runs),'/',len(tasks),ctx['parent'],seed,method,'budget',points[-1]['cost'] if points else None,'final',result['cost'],flush=True)
    assert all(d.sha(p)==h for p,h in sources.items())
    d.dump(folder/'COMPLETE.json',{'passed':True,'runs':runs,'hashes':sources,'phase_seconds':time.perf_counter()-started,'raw_io_seconds_this_invocation':io_seconds,'import_seconds_this_invocation':IMPORT_SECONDS})

if __name__=='__main__':main(sys.argv[1])
