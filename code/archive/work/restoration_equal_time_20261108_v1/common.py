"""固定方法与截止成绩；只用于本轮选线小测。"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(name,value):
    with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

ref=load('risk_reference_time',WORK/'restoration_first_line_risk_20261106_v1/screen.py')
p=ref.reference.sample.p
a,np,c=ref.a,ref.np,ref.c
ROADS=np.loadtxt(c.INPUT/'MandlTravelTimes.txt')
Q=np.loadtxt(c.INPUT/'MandlDemand.txt')
MODEL=read(WORK/'restoration_service_features_20261108_v1/部署冻结.json')['models']['H']
METHODS=['C_single','H_single','direct_multi','H_multi']
BUDGETS=[.20,.35]

def assess(routes):
    P=c.objective(c.unlimited(c.direct(ROADS,routes)),Q)
    O=float(sum(ROADS[u,v] for rr in routes for u,v in zip(rr,rr[1:])))
    legal=(len(routes)==6 and P is not None and set().union(*map(set,routes))==set(range(15)) and
           all(2<=len(rr)<=8 and len(set(rr))==len(rr) and all(0<ROADS[u,v]<np.inf for u,v in zip(rr,rr[1:])) for rr in routes))
    return {'P':P,'O':O,'feasible':bool(legal)}

def selection(routes,model,state,base,probe):
    start=time.perf_counter()
    evaluator=SimpleNamespace(direct=c.direct,limited=c.unlimited,objective=c.objective)
    X=[x['x'] for x in ref.selector.features(routes,ROADS,Q,evaluator)]
    D0=c.unlimited(c.direct(ROADS,routes));road=ROADS.copy()
    for k in range(15):road=np.minimum(road,road[:,k,None]+road[None,k,:])
    H=[]
    for j in range(6):
        Dr=c.unlimited(c.direct(ROADS,routes[:j]+routes[j+1:]));mask=(Q>0)&~np.isfinite(Dr)
        H.append(float(np.sum(Q[mask]*(D0[mask]-road[mask]))/Q.sum()/33))
    scores=ref.selector.scores(MODEL,[x+[H[j]] for j,x in enumerate(X)])
    pieces=ref.first_pieces(model,state)
    allowed=[j for j in range(6) if ref.candidate_count(routes[:j]+routes[j+1:],15,pieces)['count']>0]
    ranking=sorted(allowed,key=lambda j:(-scores[j],j)) if allowed else [0]
    h_seconds=time.perf_counter()-start
    probes=[];chosen=ranking[0]
    if probe:
        for j in range(6):
            remaining=routes[:j]+routes[j+1:];st=state.clone()
            st.replace_routes(a.tu.get_batch_tensor_from_routes([remaining],max_route_len=8))
            st=model.setup_planning(st);assert a.route(st)==()
            halt,options=a.distribution(model,st);assert halt==0.
            _,rr,prob=max(options,key=lambda x:(x[2],tuple(-u for u in x[1])))
            metrics=assess(remaining+[list(rr)])
            raw=((base['P']-metrics['P'])/66+(base['O']-metrics['O'])/840) if metrics['feasible'] else None
            probes.append({'j':j,'route':list(rr),'prob':prob,**metrics,'raw_gain':raw})
        valid=[j for j in allowed if probes[j]['feasible']]
        if valid:chosen=min(valid,key=lambda j:(-probes[j]['raw_gain'],j))
    return {'chosen':chosen,'ranking':ranking,'allowed':allowed,'scores':scores,'probes':probes,
            'H_seconds':h_seconds,'selection_seconds':time.perf_counter()-start}

def best_at(rows,budget,base):
    eligible=[x for x in rows if x['finished_seconds']<=budget]
    best=max(eligible,key=lambda x:x['gain']) if eligible else None
    positive=best is not None and best['gain']>0
    return {'gain':best['gain'] if positive else 0.,'P':best['P'] if positive else base['P'],
            'O':best['O'] if positive else base['O'],'completed':len(eligible),
            'feasible_completed':sum(x['feasible'] for x in eligible),
            'distinct_tasks':len({x['j'] for x in eligible}),
            'best_call':best['call'] if positive else None}
