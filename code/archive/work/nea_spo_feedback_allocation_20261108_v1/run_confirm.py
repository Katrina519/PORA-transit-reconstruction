"""新父网确认与消融；额外日志计入在线时间。"""
import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(ROOT));import run_feedback as base
from audited import AuditedAllocator
r=base.r;q=base.q;torch=base.torch;read=base.read;dump=base.dump
base.Allocator=AuditedAllocator
class AuditedObserver(base.FeedbackObserver):
    def __call__(self,state):
        for key,j,bee,old,index in self.allocator.pending:
            self.allocator.log[index]['candidate_routes']=r.routes(state.routes[bee])
        return super().__call__(state)
base.FeedbackObserver=AuditedObserver

def prepare():
    dev=read(ROOT/'开发统计.json');assert dev['proceed_confirmation']
    old=read(ROOT/'冻结.json');selected=dev['selected'];coef=old['coefficients'][selected]
    instance=q.c.BenchmarkInstance(tuple(range(1,16)),
        {(u+1,v+1):float(q.ROADS[u,v]) for u in range(15) for v in range(15) if u!=v and q.np.isfinite(q.ROADS[u,v])},
        {(u+1,v+1):float(q.Q[u,v]) for u in range(15) for v in range(15)},frozenset(range(1,16)),{})
    canon=lambda R:tuple(sorted(min(tuple(rr),tuple(reversed(rr))) for rr in R))
    known={canon(R) for R in read(WORK/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes'].values()}
    for path in list(WORK.glob('restoration*/部署冻结.json'))+list(WORK.glob('nea*/运行冻结.json')):
        known.update(canon(c['routes']) for c in read(path).get('contexts',[]) if 'routes' in c)
    for path in [ROOT/'冻结.json',WORK/'nea_spo_balanced_tasks_20261108_v1/请求冻结.json',WORK/'nea_spo_cost_scale_20261108_v1/请求确认.json']:
        known.update(canon(c['routes']) for c in read(path)['contexts'])
    contexts=[]
    for ident in range(62521,62525):
        seed=q.ref.stream(ident,'initialization').getrandbits(64)
        R=[[u-1 for u in rr] for rr in q.ref.make_initial_routes(instance,6,(2,8),seed)]
        assert q.assess(R)['feasible'] and canon(R) not in known,ident
        known.add(canon(R));contexts.append({'parent':'mandl1-6-'+str(ident),'routes':R,'initialization_seed':seed})
    dump(ROOT/'确认冻结.json',{'contexts':contexts,'seeds':list(range(350,355)),'selected':selected,'coefficient':coef,'model':old['model'],
         'direct_methods':['author','static_spo','feedback_spo'],'ablation_methods':['uniform_prior','no_feedback'],
         'logging_change':'新增父网/候选/批次字段和独立回放，未改变选择决定','primary_seconds':4.})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--phase',choices=['gate_audit','direct','ablation']);args=ap.parse_args()
    if args.prepare:prepare();return
    started=time.perf_counter();f=read(ROOT/'确认冻结.json');q.MODEL=f['model'];base.instrument()
    if args.phase=='gate_audit':
        contexts=read(ROOT/'冻结.json')['contexts'][-1:];seeds=[330];methods=['static_spo',f['selected']]
        old=read(ROOT/'development/COMPLETE.json')['runs']
    else:contexts=f['contexts'];seeds=f['seeds'];methods=f[args.phase+'_methods']
    folder=ROOT/args.phase;folder.mkdir(exist_ok=False)
    paths=[ROOT/n for n in ['allocator.py','audited.py','run_feedback.py','run_confirm.py','协议.md','数学依据.md','冻结.json','确认冻结.json','开发统计.json']]
    hashes={str(p):q.sha(p) for p in paths}
    for p,h in read(ROOT/'development/运行冻结.json')['input_hashes'].items():assert q.sha(Path(p))==h;hashes[p]=h
    dump(folder/'运行冻结.json',{'request':{'contexts':contexts,'seeds':seeds,'methods':methods},'input_hashes':hashes,'threads':torch.get_num_threads(),'device':'cpu'})
    runs=[]
    with (folder/'原始运行.jsonl').open('x',encoding='utf-8') as stream:
        for pi,ctx in enumerate(contexts):
            for si,seed in enumerate(seeds):
                shift=(pi+si)%len(methods)
                for mode in methods[shift:]+methods[:shift]:
                    x=base.one(ctx,seed,mode,f['coefficient'])
                    assert all(x['fine_checkpoints'][str(b)] is not None for b in [2.,4.])
                    if args.phase=='gate_audit':
                        reference=next(v for v in old if v['parent']==ctx['parent'] and v['seed']==seed and v['method']==mode)
                        for k in ['routes','history','rng','metrics','cost','selection_log','target_samples','target_calls','score_calls','cache_hits']:assert x[k]==reference[k],k
                        for e,a in zip(reference['feedback_log'],x['feedback_log']):assert all(e[k]==a[k] for k in e)
                    runs.append(x);stream.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    print(args.phase,ctx['parent'],seed,mode,round(x['fine_checkpoints']['4.0']['cost'],8),round(x['seconds'],3),flush=True)
    assert all(q.sha(Path(p))==h for p,h in hashes.items())
    dump(folder/'COMPLETE.json',{'runs':runs,'request':{'contexts':contexts,'seeds':seeds,'methods':methods},'inputs_unchanged':True,'wall_seconds':time.perf_counter()-started})

if __name__=='__main__':main()
