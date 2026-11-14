"""顺序选取新父网，在完整生成之前冻结首片段评价和全部对照。"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x):
    with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    ref=load('risk_reference',WORK/'restoration_first_line_risk_20261106_v1/screen.py')
    a,np,c=ref.a,ref.np,ref.c
    old=WORK/'restoration_service_features_20261108_v1';prior=WORK/'restoration_attachment_probe_20261108_v1'
    frozen=read(old/'部署冻结.json');models={v:frozen['models'][v] for v in ['original','H']}
    known=list(read(WORK/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes'].values())
    historical=list(WORK.glob('restoration*/部署冻结.json'))
    for p in historical:
        f=read(p);known.extend(x['routes'] for x in f.get('contexts',[]) if 'routes' in x)
        if isinstance(f.get('routes'),list):known.append(f['routes'])
    seen={ref.canonical(r) for r in known}
    paths=[ROOT/n for n in ['预先协议.md','prepare.py','sample.py','verify.py']]+historical
    paths+=[Path(ref.__file__),Path(ref.selector.__file__),Path(a.__file__),Path(c.__file__),a.WEIGHT,
            c.INPUT/'MandlTravelTimes.txt',c.INPUT/'MandlDemand.txt',WORK/'restoration_native_transfer_20261105_v1/父网与特征.json',
            WORK/'开源公交复现/ea_nea_mandl_smoke_v1/source/learning/models.py',
            WORK/'开源公交复现/ea_nea_mandl_smoke_v1/source/learning/bee_colony.py']
    paths+=list(a.SRC.rglob('*.py'))+list(a.SRC.rglob('*.yaml'))
    for folder in [old,prior]:
        for name,h in read(folder/'交付文件哈希.json').items():assert sha(folder/name)==h;paths.append(folder/name)
    hashes={str(p):sha(p) for p in paths}
    dump('INPUT_RECEIPT.json',{'hashes':hashes,'window':[62241,62260],'generator_calls':96,'distribution_queries':24,'new_fits':0})
    roads=np.loadtxt(c.INPUT/'MandlTravelTimes.txt');q=np.loadtxt(c.INPUT/'MandlDemand.txt')
    instance=c.BenchmarkInstance(tuple(range(1,16)),
        {(u+1,v+1):float(roads[u,v]) for u,v in np.ndindex(roads.shape) if u!=v and np.isfinite(roads[u,v])},
        {(u+1,v+1):float(q[u,v]) for u,v in np.ndindex(q.shape)},frozenset(range(1,16)),{})
    evaluator=SimpleNamespace(direct=c.direct,limited=c.unlimited,objective=c.objective)
    road=roads.copy()
    for k in range(15):road=np.minimum(road,road[:,k,None]+road[None,k,:])
    contexts=[];entries=[]
    for init in range(62241,62261):
        init_seed=ref.stream(init,'initialization').getrandbits(64)
        try:R=[[u-1 for u in rr] for rr in ref.make_initial_routes(instance,6,(2,8),init_seed)]
        except ValueError as error:
            if not str(error).startswith('initialization_failed:'):raise
            entries.append({'id':init,'status':'initialization_failed'});continue
        key=ref.canonical(R)
        if key in seen:entries.append({'id':init,'status':'duplicate'});continue
        seen.add(key);start=time.perf_counter()
        X=[r['x'] for r in ref.selector.features(R,roads,q,evaluator)];D0=c.unlimited(c.direct(roads,R))
        H=[]
        for j in range(6):
            Dr=c.unlimited(c.direct(roads,R[:j]+R[j+1:]));mask=(q>0)&~np.isfinite(Dr)
            H.append(float(np.sum(q[mask]*(D0[mask]-road[mask]))/q.sum()/33.))
        scores={v:ref.selector.scores(models[v],[x+([H[j]] if v=='H' else []) for j,x in enumerate(X)]) for v in models}
        feature_seconds=time.perf_counter()-start
        tick=time.perf_counter();a.torch.manual_seed(0);model,state=a.build('neural',R);load_seconds=time.perf_counter()-tick
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        pieces=ref.first_pieces(model,state);details=[ref.candidate_count(R[:j]+R[j+1:],15,pieces) for j in range(6)]
        allowed=[j for j,d in enumerate(details) if d['count']>0];base={'P':c.objective(D0,q),'O':X[0][8]*420.}
        tasks=[]
        for j in range(6):
            tick=time.perf_counter();rem=R[:j]+R[j+1:];st=state.clone()
            st.replace_routes(a.tu.get_batch_tensor_from_routes([rem],max_route_len=8));st=model.setup_planning(st)
            assert a.route(st)==();setup=time.perf_counter()-tick;rng=a.torch.random.get_rng_state()
            tick=time.perf_counter();halt,options=a.distribution(model,st);query=time.perf_counter()-tick
            assert halt==0. and a.torch.equal(rng,a.torch.random.get_rng_state())
            _,route,prob=max(options,key=lambda x:(x[2],tuple(-u for u in x[1])))
            tick=time.perf_counter();new=rem+[list(route)];D=c.unlimited(c.direct(roads,new));P=c.objective(D,q)
            O=float(sum(roads[u,v] for rr in new for u,v in zip(rr,rr[1:])))
            legal=(P is not None and set().union(*map(set,new))==set(range(15)) and
                   all(2<=len(rr)<=8 and len(set(rr))==len(rr) and all(0<roads[u,v]<np.inf for u,v in zip(rr,rr[1:])) for rr in new))
            raw=(base['P']-P)/66+(base['O']-O)/840 if legal else None
            tasks.append({'j':j,'route':list(route),'prob':prob,'P':P,'O':O,'feasible':bool(legal),'raw_gain':raw,
                          'times':{'setup':setup,'query':query,'evaluation':time.perf_counter()-tick}})
        assert all(a.torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        pick=lambda values,ix:min(ix,key=lambda j:(-values[j],j))
        choices={'first':0}
        for v in models:
            rank=pick(scores[v],allowed) if allowed else 0
            choices[v+'_rank']=rank;choices[v+'_guard']=0 if 0 in allowed else rank
        valid=[j for j in allowed if tasks[j]['feasible']]
        choices['H_probe']=pick(scores['H'],valid) if valid else choices['H_rank']
        choices['C_rank']=pick([t['raw_gain'] if t['raw_gain'] is not None else -1e100 for t in tasks],valid) if valid else choices['H_rank']
        choices['C_guard']=0 if 0 in allowed else choices['C_rank']
        seconds=time.perf_counter()-start
        contexts.append({'parent':'mandl1-6-'+str(init),'initialization_seed':init_seed,'routes':R,'base':base,
            'features':X,'H':H,'scores':scores,'allowed':allowed,'probe_valid':valid,'structural_details':details,
            'probes':tasks,'choices':choices,'seeds':[1323100+len(contexts)*100+i for i in range(4)],
            'times':{'selection_total':seconds,'feature_and_scores':feature_seconds,'model_state_init':load_seconds,
                     'probe_tasks':sum(sum(t['times'].values()) for t in tasks)}})
        entries.append({'id':init,'status':'selected'})
        if len(contexts)==4:break
    assert len(contexts)==4
    assert all(sha(Path(p))==h for p,h in hashes.items())
    dump('SCREEN.json',{'entries':entries,'selection_uses_scores_or_outcomes':False})
    dump('部署冻结.json',{'models':models,'train_keys':frozen['train_keys'],'contexts':contexts})
    print(json.dumps([(x['parent'],x['choices']) for x in contexts]),flush=True)


if __name__=='__main__':main()
