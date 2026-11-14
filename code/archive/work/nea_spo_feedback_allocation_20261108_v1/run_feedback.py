"""作者NEA观察包装与SPO先验任务分配；不改变生成和接受。"""
import sys,json,time,inspect,argparse
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(ROOT));from allocator import Allocator
sys.path.insert(0,str(WORK/'nea_spo_cost_scale_20261108_v1'));import run_stage as stage
r=stage.r;q=r.q;torch=r.torch
BASE=stage.BASE_SELECTION
read=stage.read

def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

class FeedbackObserver(r.Observer):
    def __init__(self,cost,allocator):super().__init__(cost);self.allocator=allocator;self.feedback_seconds=0.
    def __call__(self,state):
        result=super().__call__(state)
        if self.allocator.pending:
            tick=time.perf_counter();m=result.get_metrics()
            valid=((m['# disconnected node pairs']==0)&(m['# stops out of bounds']==0)).tolist()
            self.allocator.observe(result.cost.tolist(),valid)
            self.feedback_seconds+=time.perf_counter()-tick
        return result

def one(ctx,seed,mode,coefficient=.05):
    started=time.perf_counter();torch.manual_seed(0)
    model,state=q.a.build('neural',ctx['routes'])
    allocator=Allocator(coefficient,enabled=mode not in ['author','static_spo'],use_prior=mode!='uniform_prior',use_feedback=mode!='no_feedback')
    cost=FeedbackObserver(r.cost_module(),allocator)
    selection_state=state.clone();state.clear_routes()
    init=q.a.tu.get_batch_tensor_from_routes([ctx['routes']],max_route_len=8)
    logs=[];selection_seconds=0.;fallbacks=0;curve=[];fine=[];solutions=[];cache={};counts=[0,0]
    def hook(networks,chosen,type1):
        nonlocal selection_seconds,fallbacks
        assert not allocator.pending
        t=time.perf_counter();out=chosen.clone();details=[]
        for bee in type1.tolist():
            R=r.routes(networks[0,bee]);base=q.assess(R)
            if not base['feasible']:
                fallbacks+=1;details.append({'bee':bee,'fallback':'infeasible_parent'});continue
            key=tuple(tuple(rr) for rr in R)
            if key not in cache:cache[key]=BASE(R,model,selection_state,base,False);counts[0]+=1
            else:counts[1]+=1
            sel=cache[key];j=allocator.choose(key,sel['scores'],bee,base['P']/66+base['O']/198)
            out[0,bee]=j;details.append({'bee':bee,'old':int(chosen[0,bee]),'new':j,'scores':sel['scores'],'allowed':list(range(6))})
        mask=torch.ones_like(chosen,dtype=torch.bool);mask[:,type1]=False
        assert torch.equal(out[mask],chosen[mask])
        selection_seconds+=time.perf_counter()-t;logs.append(details);return out
    def record_best(networks,values,iteration,pass_index):
        stamp=time.perf_counter()-started;val=float(values[0])
        if not fine or val<fine[-1]['cost']:solutions.append({'cost':val,'routes':r.routes(networks[0]),'first_seconds':stamp})
        fine.append({'iteration':int(iteration),'pass':int(pass_index),'seconds':stamp,'cost':val,'samples':cost.samples,'solution_index':len(solutions)-1})
    class Writer:
        def add_scalar(self,name,val,step):
            if curve and curve[-1]['step']==step:return
            frame=inspect.currentframe().f_back
            curve.append({'step':int(step),'seconds':time.perf_counter()-started,'cost':float(frame.f_locals['best_costs'][0]),'samples':cost.samples})
    fn=r.ORIGINAL if mode=='author' else r.modified(hook)
    torch.manual_seed(seed)
    with torch.no_grad(),patch.object(r.entry,'get_mutants',fn),patch.object(r.entry,'best_observer',record_best,create=True):
        out,hist=r.entry.bee_colony(state,cost,init,n_bees=10,n_type1_bees=5,n_type2_bees=5,n_iterations=8,
            passes_per_it=5,mod_steps_per_pass=2,shorten_prob=.2,silent=True,bee_model=model,sum_writer=Writer())
    rng=torch.get_rng_state().tolist();res=cost(out);metrics={k:v.tolist() for k,v in res.get_metrics().items()};elapsed=time.perf_counter()-started
    assert not allocator.pending
    assert mode in ['author','static_spo'] or len(allocator.log)==400-fallbacks
    assert len(fine)==41 and cost.samples==811
    final=float(res.cost[0]);assert abs(final-float(hist[0,-1]))<1e-5
    return {'parent':ctx['parent'],'seed':seed,'method':mode,'cost':final,'metrics':metrics,'routes':r.routes(out.routes[0]),'history':hist.tolist(),
        'seconds':elapsed,'selection_seconds':selection_seconds,'selection_calls':len(logs),'selection_log':logs,'fallbacks':fallbacks,
        'target_samples':cost.samples,'target_calls':cost.calls,'curve':curve,'fine_curve':fine,'fine_solutions':solutions,
        'fine_checkpoints':{str(b):stage.at_budget(fine,b) for b in [2.,4.]},'rng':rng,'score_calls':counts[0],'cache_hits':counts[1],
        'feedback_seconds':cost.feedback_seconds,'feedback_log':allocator.log,'feedback_states':len(allocator.states)}

def instrument():
    # Exact same two observation callbacks already used by all prior paired methods.
    stage.timed_one()

def prepare():
    old=read(WORK/'nea_spo_balanced_tasks_20261108_v1/请求冻结.json')
    model=read(WORK/'nea_spo_balanced_tasks_20261108_v1/候选冻结.json')['models']['balanced_spo']
    dump(ROOT/'冻结.json',{'contexts':old['contexts'],'seeds':[330,331,332],'methods':['author','static_spo','feedback_005','feedback_020'],
         'model':model,'coefficients':{'feedback_005':.05,'feedback_020':.20},'primary_seconds':4.})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--gate',action='store_true');args=ap.parse_args()
    if args.prepare:prepare();return
    f=read(ROOT/'冻结.json');q.MODEL=f['model'];instrument();started=time.perf_counter()
    hashes={str(p):q.sha(p) for p in [ROOT/'allocator.py',Path(__file__),ROOT/'协议.md',ROOT/'冻结.json',Path(r.entry.__file__),Path(r.__file__),Path(stage.__file__)]}
    for p,h in read(WORK/'nea_spo_exact_reuse_confirm_20261108_v1/运行冻结.json')['input_hashes'].items():assert q.sha(Path(p))==h;hashes[p]=h
    folder=ROOT/('gate' if args.gate else 'development');folder.mkdir(exist_ok=False)
    contexts=f['contexts'][:1] if args.gate else f['contexts'];seeds=[330] if args.gate else f['seeds'];methods=['author','static_spo'] if args.gate else f['methods']
    dump(folder/'运行冻结.json',{'request':{'contexts':contexts,'seeds':seeds,'methods':methods},'input_hashes':hashes,'threads':torch.get_num_threads(),'device':'cpu'})
    old=read(WORK/'nea_spo_balanced_tasks_20261108_v1/direct/COMPLETE.json')['runs']+read(WORK/'nea_spo_balanced_tasks_20261108_v1/ablation/COMPLETE.json')['runs'] if args.gate else []
    runs=[]
    with (folder/'原始运行.jsonl').open('x',encoding='utf-8') as stream:
        for pi,ctx in enumerate(contexts):
            for si,seed in enumerate(seeds):
                shift=(pi+si)%len(methods)
                for mode in methods[shift:]+methods[:shift]:
                    x=one(ctx,seed,mode,f['coefficients'].get(mode,.05))
                    assert all(x['fine_checkpoints'][str(b)] is not None for b in [2.,4.])
                    if args.gate:
                        reference=next(v for v in old if v['parent']==ctx['parent'] and v['seed']==seed and v['method']==('author' if mode=='author' else 'balanced_spo_no_risk'))
                        for k in ['routes','history','rng','metrics','cost','selection_log','target_samples','target_calls','score_calls','cache_hits']:assert x[k]==reference[k],k
                    runs.append(x);stream.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    print(ctx['parent'],seed,mode,round(x['fine_checkpoints']['4.0']['cost'],8),round(x['seconds'],3),flush=True)
    assert all(q.sha(Path(p))==h for p,h in hashes.items())
    dump(folder/'COMPLETE.json',{'runs':runs,'request':{'contexts':contexts,'seeds':seeds,'methods':methods},'inputs_unchanged':True,'wall_seconds':time.perf_counter()-started})

if __name__=='__main__':main()
