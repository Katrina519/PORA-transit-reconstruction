"""固定两次重建；保留全开销、原始输出和预定截止成绩。"""
import sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
t_import=time.perf_counter()
sys.path.insert(0,str(WORK/'restoration_equal_time_20261108_v1'))
import common as q
IMPORT_SECONDS=time.perf_counter()-t_import
METHODS=['pair','H2','cycle']
def dump(n,x):
    with (ROOT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def orders(sel,R,method,offset):
    if method=='cycle':return [offset%6,(offset+1)%6]
    allowed=set(sel['allowed']);s=sel['scores']
    h=sorted(range(6),key=lambda j:(j not in allowed,-s[j],j))
    if method=='H2':return h[:2]
    candidates=[j for j in allowed if j!=h[0]]
    lengths=[sum(q.ROADS[u,v] for u,v in zip(r,r[1:])) for r in R]
    second=min(candidates,key=lambda j:(-lengths[j],j)) if candidates else h[1]
    return [h[0],second]
def one(ctx,pi,rep,method):
    R=ctx['routes'];start=time.perf_counter()
    q.a.torch.manual_seed(0);model,state=q.a.build('neural',R)
    init=time.perf_counter()-start;base=q.assess(R)
    ts=time.perf_counter()
    sel=q.selection(R,model,state,base,False) if method!='cycle' else None
    order=orders(sel,R,method,(pi+rep)%6)
    selection_seconds=time.perf_counter()-ts
    tensor=q.a.tu.get_batch_tensor_from_routes([R],max_route_len=8)
    ready=time.perf_counter()-start;rows=[]
    for call,j in enumerate(order):
        begin=time.perf_counter()-start;seed=1450000+pi*10000+rep*100
        q.a.torch.manual_seed(seed)
        with q.a.torch.no_grad():
            out=q.p.get_neural_variants(model,state.clone(),tensor.clone(),q.a.torch.tensor([j],dtype=q.a.torch.long))
        genend=time.perf_counter()-start
        rr=[[int(u) for u in line.tolist() if u>=0] for line in out[0]]
        value=q.assess(rr)
        gain=max(0.,(base['P']-value['P'])/66+(base['O']-value['O'])/840) if value['feasible'] else 0.
        row={'call':call,'j':j,'seed':seed,'routes':rr,**value,'gain':gain,'started_seconds':begin,'generation_done_seconds':genend}
        rows.append(row);row['finished_seconds']=time.perf_counter()-start
    elapsed=time.perf_counter()-start
    return {'parent':ctx['parent'],'pi':pi,'repeat':rep,'method':method,'base':base,'order':order,'selection':sel,
      'initialization_seconds':init,'selection_seconds':selection_seconds,'ready_seconds':ready,'elapsed_seconds':elapsed,'rows':rows,
      'two_calls':q.best_at(rows,elapsed,base),'checkpoints':{str(b):q.best_at(rows,b,base) for b in [.20,.35]}}
def main():
    tprep=time.perf_counter();historical=list(WORK.glob('restoration*/部署冻结.json'))
    known=[]
    for f in historical:known.extend(c['routes'] for c in q.read(f).get('contexts',[]) if 'routes' in c)
    seen={q.ref.canonical(r) for r in known}
    instance=q.c.BenchmarkInstance(tuple(range(1,16)),{(u+1,v+1):float(q.ROADS[u,v]) for u,v in q.np.ndindex(q.ROADS.shape) if u!=v and q.np.isfinite(q.ROADS[u,v])},
        {(u+1,v+1):float(q.Q[u,v]) for u,v in q.np.ndindex(q.Q.shape)},frozenset(range(1,16)),{})
    contexts=[];screen=[]
    for init in range(62249,62299):
        seed=q.ref.stream(init,'initialization').getrandbits(64)
        try:R=[[u-1 for u in r] for r in q.ref.make_initial_routes(instance,6,(2,8),seed)]
        except ValueError as e:
            if not str(e).startswith('initialization_failed:'):raise
            screen.append([init,'initialization_failed']);continue
        key=q.ref.canonical(R)
        if key in seen:screen.append([init,'duplicate']);continue
        seen.add(key);assert q.assess(R)['feasible']
        contexts.append({'parent':'mandl1-6-'+str(init),'routes':R,'seed':seed});screen.append([init,'selected'])
        if len(contexts)==4:break
    assert len(contexts)==4
    dump('部署冻结.json',{'contexts':contexts,'screen':screen,'model':q.MODEL,'methods':METHODS,'repeats':6,'budgets':[.2,.35]})
    paths={str(p):q.sha(p) for p in ROOT.iterdir() if p.is_file()}
    prior=q.read(WORK/'restoration_equal_time_20261108_v1/INPUT_RECEIPT.json')['hashes']
    assert all(q.sha(Path(p))==h for p,h in prior.items());paths.update(prior)
    paths[str(Path(q.__file__))]=q.sha(Path(q.__file__))
    dump('INPUT_RECEIPT.json',{'hashes':paths,'import_seconds':IMPORT_SECONDS,'parent_preparation_seconds':time.perf_counter()-tprep,
      'python':sys.version,'torch':q.a.torch.__version__,'threads':q.a.torch.get_num_threads(),'cuda_available':q.a.torch.cuda.is_available(),'device':'cpu'})
    old=q.read(WORK/'restoration_attachment_validation_20261108_v1/部署冻结.json')['contexts'][0]
    warm=[one(old,99,0,m) for m in METHODS];dump('WARMUP.json',warm)
    assert all(len(set(r['order']))==2 for r in warm)
    runs=[]
    with (ROOT/'原始运行.jsonl').open('x',encoding='utf8') as f:
        for pi,ctx in enumerate(contexts):
            for rep in range(6):
                shift=(pi+rep)%3
                for method in METHODS[shift:]+METHODS[:shift]:
                    r=one(ctx,pi,rep,method);runs.append(r);f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            print(ctx['parent'],'complete',flush=True)
    assert len(runs)==72 and all(q.sha(Path(p))==h for p,h in paths.items())
    dump('COMPLETE.json',{'runs':runs,'new_fits':0,'calls':144,'warmup_calls':6})
    print('DONE',flush=True)
if __name__=='__main__':main()
