"""作者NEA与冻结选线模块直接配对；只替换type1选择。"""
import sys,time,json,inspect,hashlib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
timport=time.perf_counter()
sys.path.insert(0,str(WORK/'restoration_equal_time_20261108_v1'))
import common as q
entry=q.load('nea_index_fixed_entry',WORK/'开源公交复现/ea_nea_index_alignment_smoke_v1/source_fixed/learning/bee_colony.py')
torch=q.a.torch
IMPORT_SECONDS=time.perf_counter()-timport
def dump(n,v):
    with (ROOT/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def routes(t):return [[int(u) for u in r.tolist() if u>=0] for r in t]
ORIGINAL=entry.get_mutants
src=inspect.getsource(ORIGINAL)
anchor='    type3_idxs = scen_idxs[n_type1 + n_type2:]\n'
assert src.count(anchor)==1 and 'new_routes[:, torch.argsort(scen_idxs)]' in src
insertion='''
    chosen_route_idxs = selection_hook(bee_networks, chosen_route_idxs, type1_idxs)
    gather_idx = chosen_route_idxs[..., None, None].expand(-1, -1, -1, max_n_nodes)
    modified_routes = bee_networks.gather(2, gather_idx).squeeze(2)
'''
def modified(hook):
    scope=dict(ORIGINAL.__globals__);scope['selection_hook']=hook
    exec(compile(src.replace(anchor,anchor+insertion),str(ROOT/'run.py')+'::selection_adapter','exec'),scope)
    return scope['get_mutants']
def cost_module():
    cfg=q.a.OmegaConf.load(q.a.SRC/'cfg/experiment/cost_function/mine.yaml')
    cfg.kwargs.variable_weights=False
    return q.a.get_cost_module_from_cfg(cfg,False,True)
class Observer:
    def __init__(self,cost):self.inner=cost;self.samples=0;self.calls=0
    def __getattr__(self,k):return getattr(self.inner,k)
    def __call__(self,state):
        self.samples+=int(state.batch_size);self.calls+=1
        return self.inner(state)
def one(ctx,seed,method,niter=5):
    started=time.perf_counter();torch.manual_seed(0)
    model,state=q.a.build('neural',ctx['routes']);cost=Observer(cost_module())
    # Keep an independent one-network state for the frozen static risk interface.
    selection_state=state.clone();state.clear_routes()
    init=q.a.tu.get_batch_tensor_from_routes([ctx['routes']],max_route_len=8)
    logs=[];selection_seconds=0.;fallbacks=0;curve=[]
    def hook(networks,chosen,type1):
        nonlocal selection_seconds,fallbacks
        t=time.perf_counter();out=chosen.clone();details=[]
        for bee in type1.tolist():
            R=routes(networks[0,bee]);base=q.assess(R)
            if not base['feasible']:
                fallbacks+=1;details.append({'bee':bee,'fallback':'infeasible_parent'});continue
            sel=q.selection(R,model,selection_state,base,False)
            out[0,bee]=sel['chosen'];details.append({'bee':bee,'old':int(chosen[0,bee]),'new':int(out[0,bee]),'scores':sel['scores'],'allowed':sel['allowed']})
        mask=torch.ones_like(chosen,dtype=torch.bool);mask[:,type1]=False
        assert torch.equal(out[mask],chosen[mask])
        selection_seconds+=time.perf_counter()-t;logs.append(details)
        return out
    class Writer:
        def add_scalar(self,name,val,step):
            if curve and curve[-1]['step']==step:return
            frame=inspect.currentframe().f_back
            curve.append({'step':int(step),'seconds':time.perf_counter()-started,'cost':float(frame.f_locals['best_costs'][0]),'samples':cost.samples})
    fn=ORIGINAL if method=='author' else modified((lambda n,c,t:c) if method=='identity' else hook)
    torch.manual_seed(seed)
    with torch.no_grad(),patch.object(entry,'get_mutants',fn):
        out,hist=entry.bee_colony(state,cost,init,n_bees=10,n_type1_bees=5,n_type2_bees=5,n_iterations=niter,
            passes_per_it=5,mod_steps_per_pass=2,shorten_prob=.2,silent=True,bee_model=model,sum_writer=Writer())
    rng=torch.get_rng_state().tolist();res=cost(out);metric={k:v.tolist() for k,v in res.get_metrics().items()}
    final=float(res.cost[0]);assert abs(final-float(hist[0,-1]))<1e-5
    rr=routes(out.routes[0]);elapsed=time.perf_counter()-started
    return {'parent':ctx['parent'],'seed':seed,'method':method,'cost':final,'metrics':metric,'routes':rr,'history':hist.tolist(),
        'seconds':elapsed,'selection_seconds':selection_seconds,'selection_calls':len(logs),'fallbacks':fallbacks,'selection_log':logs,
        'target_samples':cost.samples,'target_calls':cost.calls,'curve':curve,'rng':rng,
        'checkpoints':{str(b):next((x['cost'] for x in reversed(curve) if x['seconds']<=b),curve[0]['cost']) for b in [1.,2.,4.]},
        'initialized_at_checkpoint':{str(b):curve[0]['seconds']<=b for b in [1.,2.,4.]}}
def main():
    contexts=q.read(WORK/'restoration_complementary_pair_20261108_v1/部署冻结.json')['contexts'][:3]
    dump('部署冻结.json',{'contexts':contexts,'seeds':[200,201,202],'methods':['author','selector'],'iterations':5,'model':q.MODEL,
      'source':str(Path(entry.__file__)),'source_sha':q.sha(Path(entry.__file__)),'import_seconds':IMPORT_SECONDS})
    hashes=dict(q.read(WORK/'restoration_complementary_pair_20261108_v1/INPUT_RECEIPT.json')['hashes'])
    hashes.update({str(p):q.sha(p) for p in ROOT.iterdir() if p.is_file()})
    hashes[str(Path(entry.__file__))]=q.sha(Path(entry.__file__))
    assert all(q.sha(Path(p))==h for p,h in hashes.items())
    dump('INPUT_RECEIPT.json',{'hashes':hashes,'python':sys.version,'torch':torch.__version__,'threads':torch.get_num_threads(),'new_training':0})
    old=q.read(WORK/'restoration_attachment_validation_20261108_v1/部署冻结.json')['contexts'][0]
    a=one(old,99,'author',1);b=one(old,99,'identity',1)
    for k in ['routes','history','rng','metrics','cost','target_samples']:assert a[k]==b[k],k
    dump('PREFLIGHT.json',{'identity_match':True,'author':a,'identity':b})
    print('identity gate PASS',flush=True)
    runs=[]
    with (ROOT/'原始运行.jsonl').open('x',encoding='utf8') as f:
        for pi,ctx in enumerate(contexts):
            for si,seed in enumerate([200,201,202]):
                methods=['author','selector'] if (pi+si)%2==0 else ['selector','author']
                for method in methods:
                    r=one(ctx,seed,method);assert r['target_samples']==511
                    runs.append(r);f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                    print(ctx['parent'],seed,method,r['cost'],round(r['seconds'],3),flush=True)
    assert len(runs)==18 and all(q.sha(Path(p))==h for p,h in hashes.items())
    dump('COMPLETE.json',{'runs':runs,'new_training':0,'target_samples':sum(r['target_samples'] for r in runs)})
if __name__=='__main__':main()
