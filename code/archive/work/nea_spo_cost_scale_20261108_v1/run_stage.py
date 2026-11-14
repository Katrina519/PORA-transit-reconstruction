"""固定请求与共同细计时的NEA短对照；不改作者搜索。"""
import argparse, inspect, json, sys, time, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(WORK/'nea_frozen_selector_direct_20261108_v1'))
import run as r
q=r.q;torch=r.torch
BASE_SELECTION=q.selection

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def at_budget(events,budget):
    return next((p for p in reversed(events) if p['seconds']<=budget),None)

def timed_one():
    colony=inspect.getsource(r.entry.bee_colony)
    for anchor,extra in [
        ('    best_metrics = bee_metrics[batch_idxs, best_idxs]',
         '\n    best_observer(best_networks,best_costs,0,0)'),
        ('            best_metrics[is_improvement] = new_best',
         '\n            best_observer(best_networks,best_costs,iteration+1,pi+1)')]:
        assert colony.count(anchor)==1
        colony=colony.replace(anchor,anchor+extra)
    exec(compile(colony,'shared_fine_best_observer','exec'),r.entry.__dict__)
    source=inspect.getsource(r.one)
    old='logs=[];selection_seconds=0.;fallbacks=0;curve=[]'
    assert source.count(old)==1
    source=source.replace(old,old+';fine=[];solutions=[];timing_log_seconds=0.')
    callback='''    def record_best(networks,values,iteration,pass_index):
        nonlocal timing_log_seconds
        tick=time.perf_counter();stamp=tick-started;value=float(values[0])
        if not fine or value<fine[-1]['cost']:
            solutions.append({'cost':value,'routes':routes(networks[0]),'first_seconds':stamp})
        fine.append({'iteration':int(iteration),'pass':int(pass_index),'seconds':stamp,'cost':value,'samples':cost.samples,'solution_index':len(solutions)-1})
        timing_log_seconds+=time.perf_counter()-tick
'''
    source=source.replace('    class Writer:',callback+'    class Writer:')
    source=source.replace("patch.object(entry,'get_mutants',fn):",
                          "patch.object(entry,'get_mutants',fn),patch.object(entry,'best_observer',record_best,create=True):")
    source=source.replace("'initialized_at_checkpoint':",
        "'fine_curve':fine,'fine_solutions':solutions,'timing_log_seconds':timing_log_seconds,'fine_checkpoints':{str(b):at_budget(fine,b) for b in [2.,4.]},'initialized_at_checkpoint':")
    ns=dict(r.__dict__);ns['at_budget']=at_budget
    exec(compile(source,'shared_fine_one','exec'),ns)
    return ns['one']

def prepare_requests():
    # Both development and independent input batches are frozen before testing models.
    old=read(WORK/'nea_spo_context_20261108_v1/运行冻结.json')['contexts']
    prior=read(WORK/'nea_spo_exact_reuse_confirm_20261108_v1/运行冻结.json')['contexts']
    dev={'contexts':old+prior,'seeds':[224,225,226],'max_iterations':8,
         'methods':['author','old_spo','weighted_spo','tuned_spo'],
         'role':'开发，含既有看过的父网；按全批4秒均值选择一个候选'}
    assert [int(c['parent'].split('-')[-1]) for c in dev['contexts']]==list(range(62265,62271))
    dump(ROOT/'请求开发.json',dev)
    instance=q.c.BenchmarkInstance(tuple(range(1,16)),
        {(u+1,v+1):float(q.ROADS[u,v]) for u in range(15) for v in range(15)
         if u!=v and q.np.isfinite(q.ROADS[u,v])},
        {(u+1,v+1):float(q.Q[u,v]) for u in range(15) for v in range(15)},
        frozenset(range(1,16)),{})
    known=set()
    history=[]
    canon=lambda R:tuple(sorted(min(tuple(rr),tuple(reversed(rr))) for rr in R))
    for path in list(WORK.glob('restoration*/部署冻结.json'))+list(WORK.glob('nea*/运行冻结.json')):
        data=read(path)
        for c in data.get('contexts',[]):
            if 'routes' in c:known.add(canon(c['routes']))
        history.append({'path':str(path),'sha256':sha(path)})
    train=read(WORK/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes']
    known.update(canon(R) for R in train.values())
    contexts=[]
    for ident in range(62501,62507):
        seed=q.ref.stream(ident,'initialization').getrandbits(64)
        R=[[u-1 for u in rr] for rr in q.ref.make_initial_routes(instance,6,(2,8),seed)]
        assert q.assess(R)['feasible'] and canon(R) not in known,ident
        known.add(canon(R))
        contexts.append({'parent':'mandl1-6-'+str(ident),'routes':R,'initialization_seed':seed})
    conf={'contexts':contexts,'seeds':[230,231,232],'max_iterations':8,
          'methods':['author','PENDING_SELECTED_DEVELOPMENT'],
          'role':'预先冻结的新父网，只在开发选定模型后运行','history_receipt':history}
    dump(ROOT/'请求确认输入.json',conf)
    print('REQUESTS_FROZEN',len(dev['contexts']),len(contexts),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--request',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.prepare:
        prepare_requests();return
    wall=time.perf_counter()
    f=read(ROOT/'候选冻结.json');models=f['models']
    one=timed_one()
    if args.gate:
        req=read(WORK/'nea_fixed_replay_timing_20261108_v1/请求冻结.json')
        legacy=read(WORK/'nea_fixed_replay_timing_20261108_v1/run_001/COMPLETE.json')['runs']
        request={'contexts':req['contexts'][:1],'seeds':[215],
                 'methods':['author','old_spo'],'max_iterations':8}
        args.output=ROOT/'gate'
    else:
        request=read(args.request)
        assert all(m=='author' or m in models or m.endswith('_no_risk') for m in request['methods'])
    args.output.mkdir(exist_ok=False)
    inputs={str(ROOT/n):sha(ROOT/n) for n in
            ['run_stage.py','候选冻结.json','weighted_spo.py','研究协议.md']}
    inputs[str(Path(r.entry.__file__))]=sha(Path(r.entry.__file__))
    for p,h in read(WORK/'nea_spo_exact_reuse_confirm_20261108_v1/运行冻结.json')['input_hashes'].items():
        assert sha(p)==h,p
        inputs[p]=h
    if args.request:inputs[str(args.request.resolve())]=sha(args.request)
    dump(args.output/'运行冻结.json',{'request':request,'input_hashes':inputs,
         'cpu_threads':torch.get_num_threads(),'torch':torch.__version__,'timing':'initial + every best pass update',
         'offline_fit_seconds':f['fit_seconds'],'risk_ablation':'只去筛查决策，保持其计算以隔离决策作用'})
    prep=time.perf_counter()-wall;runs=[];io=0.
    with (args.output/'原始运行.jsonl').open('x',encoding='utf-8') as stream:
        for pi,ctx in enumerate(request['contexts']):
            for si,seed in enumerate(request['seeds']):
                methods=request['methods'];shift=(pi+si)%len(methods)
                for method in methods[shift:]+methods[:shift]:
                    no_risk=method.endswith('_no_risk')
                    name=method[:-8] if no_risk else method
                    if name!='author':q.MODEL=models[name]
                    cache={};counts=[0,0]
                    def select(R,model,state,base,probe):
                        key=tuple(tuple(rr) for rr in R)
                        if key not in cache:
                            sel=BASE_SELECTION(R,model,state,base,probe)
                            if no_risk:
                                sel=dict(sel);sel['allowed']=list(range(6))
                                sel['ranking']=sorted(range(6),key=lambda j:(-sel['scores'][j],j))
                                sel['chosen']=sel['ranking'][0]
                            cache[key]=sel;counts[0]+=1
                        else:counts[1]+=1
                        return cache[key]
                    q.selection=select
                    x=one(ctx,seed,'author' if method=='author' else 'selector',request['max_iterations'])
                    x['method']=method;x['score_calls']=counts[0];x['cache_hits']=counts[1]
                    assert x['target_samples']==11+100*request['max_iterations']
                    assert len(x['fine_curve'])==1+5*request['max_iterations']
                    assert all(p['samples']==10+20*n for n,p in enumerate(x['fine_curve']))
                    assert all(x['fine_checkpoints'][str(b)] is not None for b in [2.,4.])
                    if args.gate:
                        old=next(v for v in legacy if v['parent']==ctx['parent'] and v['seed']==seed
                                 and v['method']==('author' if method=='author' else 'cached_spo'))
                        for k in ['routes','history','rng','metrics','cost','selection_log','target_samples','target_calls','score_calls','cache_hits']:
                            assert x[k]==old[k],k
                    runs.append(x);tick=time.perf_counter()
                    stream.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    io+=time.perf_counter()-tick
                    print(ctx['parent'],seed,method,round(x['cost'],8),
                          '4s',round(x['fine_checkpoints']['4.0']['cost'],8),round(x['seconds'],3),flush=True)
    assert all(sha(p)==h for p,h in inputs.items())
    dump(args.output/'COMPLETE.json',{'runs':runs,'request':request,'inputs_unchanged':True,
         'identity_gate':bool(args.gate),'preparation_seconds':prep,
         'raw_io_seconds':io,'wall_seconds':time.perf_counter()-wall})

if __name__=='__main__':main()

