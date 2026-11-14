"""固定NEA与计时口径，轮换运行直接对照和消融。"""
import sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(WORK/'nea_spo_cost_scale_20261108_v1'))
import run_stage as stage
r=stage.r;q=r.q;torch=r.torch
read=stage.read

def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def prepare():
    instance=q.c.BenchmarkInstance(tuple(range(1,16)),
        {(u+1,v+1):float(q.ROADS[u,v]) for u in range(15) for v in range(15) if u!=v and q.np.isfinite(q.ROADS[u,v])},
        {(u+1,v+1):float(q.Q[u,v]) for u in range(15) for v in range(15)},frozenset(range(1,16)),{})
    canon=lambda R:tuple(sorted(min(tuple(rr),tuple(reversed(rr))) for rr in R))
    known={canon(R) for R in read(WORK/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes'].values()}
    for path in list(WORK.glob('restoration*/部署冻结.json'))+list(WORK.glob('nea*/运行冻结.json')):
        data=read(path)
        known.update(canon(c['routes']) for c in data.get('contexts',[]) if 'routes' in c)
    known.update(canon(c['routes']) for c in read(ROOT/'诊断冻结.json')['contexts'])
    known.update(canon(c['routes']) for c in read(WORK/'nea_spo_cost_scale_20261108_v1/请求确认.json')['contexts'])
    contexts=[]
    for ident in range(62511,62515):
        seed=q.ref.stream(ident,'initialization').getrandbits(64)
        R=[[u-1 for u in rr] for rr in q.ref.make_initial_routes(instance,6,(2,8),seed)]
        assert q.assess(R)['feasible'] and canon(R) not in known,ident
        known.add(canon(R));contexts.append({'parent':'mandl1-6-'+str(ident),'routes':R,'initialization_seed':seed})
    req={'contexts':contexts,'seeds':list(range(330,335)),'max_iterations':8,'primary_seconds':4.,
         'direct_methods':['author','previous_spo','balanced_spo'],
         'ablation_methods':['initial_only_spo','balanced_spo_no_risk','balanced_ridge']}
    dump(ROOT/'请求冻结.json',req);print('REQUEST_FROZEN',len(contexts),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--phase',choices=['direct','ablation','gate']);args=ap.parse_args()
    if args.prepare:prepare();return
    started=time.perf_counter();models=read(ROOT/'候选冻结.json')['models'];req=read(ROOT/'请求冻结.json')
    if args.phase=='gate':
        old=read(WORK/'nea_spo_cost_scale_20261108_v1/confirmation/COMPLETE.json')['runs']
        ctx=read(WORK/'nea_spo_cost_scale_20261108_v1/请求确认.json')['contexts'][-1]
        contexts=[ctx];seeds=[230];methods=['author','previous_spo']
    else:contexts=req['contexts'];seeds=req['seeds'];methods=req[args.phase+'_methods']
    out=ROOT/args.phase;out.mkdir(exist_ok=False)
    paths=[ROOT/n for n in ['run.py','train.py','协议.md','候选冻结.json','请求冻结.json','诊断冻结.json','诊断完成.json']]
    paths += [Path(stage.__file__),Path(r.__file__),Path(q.__file__),Path(r.entry.__file__)]
    hashes={str(p):q.sha(p) for p in paths}
    for p,h in read(WORK/'nea_spo_exact_reuse_confirm_20261108_v1/运行冻结.json')['input_hashes'].items():
        assert q.sha(Path(p))==h,p
        hashes[p]=h
    for p,h in read(ROOT/'候选冻结.json')['source_hashes'].items():assert q.sha(Path(p))==h,p
    dump(out/'运行冻结.json',{'request':req,'phase':args.phase,'input_hashes':hashes,'threads':torch.get_num_threads(),'device':'cpu','selection_interface_only':True})
    one=stage.timed_one();runs=[];io_seconds=0.
    with (out/'原始运行.jsonl').open('x',encoding='utf-8') as stream:
        for pi,ctx in enumerate(contexts):
            for si,seed in enumerate(seeds):
                shift=(pi+si)%len(methods)
                for method in methods[shift:]+methods[:shift]:
                    no_risk=method.endswith('_no_risk');name=method[:-8] if no_risk else method
                    if name!='author':q.MODEL=models[name]
                    cache={};counts=[0,0]
                    def select(R,model,state,base,probe):
                        key=tuple(tuple(rr) for rr in R)
                        if key not in cache:
                            sel=stage.BASE_SELECTION(R,model,state,base,probe)
                            if no_risk:
                                sel=dict(sel);sel['allowed']=list(range(6));sel['ranking']=sorted(range(6),key=lambda j:(-sel['scores'][j],j));sel['chosen']=sel['ranking'][0]
                            cache[key]=sel;counts[0]+=1
                        else:counts[1]+=1
                        return cache[key]
                    q.selection=select
                    x=one(ctx,seed,'author' if method=='author' else 'selector',8)
                    x.update(method=method,score_calls=counts[0],cache_hits=counts[1])
                    assert x['target_samples']==811 and len(x['fine_curve'])==41
                    assert all(p['samples']==10+20*n for n,p in enumerate(x['fine_curve']))
                    assert all(x['fine_checkpoints'][str(b)] is not None for b in [2.,4.])
                    if args.phase=='gate':
                        reference=next(v for v in old if v['parent']==ctx['parent'] and v['seed']==seed and v['method']==('author' if method=='author' else 'weighted_spo'))
                        for k in ['routes','history','rng','metrics','cost','selection_log','target_samples','target_calls','score_calls','cache_hits']:assert x[k]==reference[k],k
                    runs.append(x);tick=time.perf_counter();stream.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();io_seconds+=time.perf_counter()-tick
                    print(args.phase,ctx['parent'],seed,method,round(x['fine_checkpoints']['4.0']['cost'],8),round(x['seconds'],3),flush=True)
    assert all(q.sha(Path(p))==h for p,h in hashes.items())
    dump(out/'COMPLETE.json',{'runs':runs,'request':req,'phase':args.phase,'inputs_unchanged':True,'raw_io_seconds':io_seconds,'wall_seconds':time.perf_counter()-started})

if __name__=='__main__':main()
