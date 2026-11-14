"""独立服务距离核验；preflight不调用生成器。"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
spec=importlib.util.spec_from_file_location('independent_distance',WORK/'restoration_attachment_probe_20261108_v1/verify.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
distances=module.distances
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(name,x):
    with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    frozen=read(ROOT/'部署冻结.json');receipt=read(ROOT/'INPUT_RECEIPT.json');old=read(WORK/'restoration_service_features_20261108_v1/部署冻结.json')
    assert frozen['models']=={v:old['models'][v] for v in ['original','H']} and frozen['train_keys']==old['train_keys']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in receipt['hashes'].items())
    roads=np.loadtxt(WORK/'paper_code_learning_20261030_v1/holliday/inputs/MandlTravelTimes.txt')
    q=np.loadtxt(WORK/'paper_code_learning_20261030_v1/holliday/inputs/MandlDemand.txt');mask=q>0;Q=float(q.sum());errors=[]
    road=roads.copy()
    for k in range(15):road=np.minimum(road,road[:,k,None]+road[None,k,:])
    def assess(routes):
        D=distances(roads,routes);P=float(np.sum(D[mask]*q[mask])/Q) if np.isfinite(D[mask]).all() else None
        O=float(sum(roads[u,v] for rr in routes for u,v in zip(rr,rr[1:])))
        legal=(len(routes)==6 and P is not None and set().union(*map(set,routes))==set(range(15)) and
               all(2<=len(rr)<=8 and len(rr)==len(set(rr)) and all(0<roads[u,v]<np.inf for u,v in zip(rr,rr[1:])) for rr in routes))
        return D,P,O,bool(legal)
    for i,ctx in enumerate(frozen['contexts']):
        R=ctx['routes'];D0,P0,O0,legal=assess(R);assert legal
        errors.extend([abs(P0-ctx['base']['P']),abs(O0-ctx['base']['O'])])
        assert ctx['seeds']==[1323100+i*100+j for j in range(4)]
        for t in ctx['probes']:
            j=t['j'];rem=R[:j]+R[j+1:];Dr=distances(roads,rem);D,P,O,legal=assess(rem+[t['route']])
            assert legal==t['feasible'] and (P is None)==(t['P'] is None)
            errors.append(abs(O-t['O']))
            if P is not None:errors.append(abs(P-t['P']))
            if legal:errors.append(abs((P0-P)/66+(O0-O)/840-t['raw_gain']))
            affected=mask&~np.isfinite(Dr);h=float(np.sum(q[affected]*(D0[affected]-road[affected]))/Q/33)
            errors.append(abs(h-ctx['H'][j]))
        for v,m in frozen['models'].items():
            X=np.asarray([x+([ctx['H'][j]] if v=='H' else []) for j,x in enumerate(ctx['features'])])
            ss=m['coef'][0]+((X-m['mean'])/m['scale'])@np.asarray(m['coef'][1:]);errors.append(float(np.max(np.abs(ss-ctx['scores'][v]))))
            j=min(ctx['allowed'],key=lambda j:(-ss[j],j)) if ctx['allowed'] else 0
            assert ctx['choices'][v+'_rank']==j and ctx['choices'][v+'_guard']==(0 if 0 in ctx['allowed'] else j)
        valid=[j for j in ctx['allowed'] if ctx['probes'][j]['feasible']];assert valid==ctx['probe_valid']
        expected=min(valid,key=lambda j:(-ctx['probes'][j]['raw_gain'],j)) if valid else ctx['choices']['H_rank']
        assert expected==ctx['choices']['C_rank'] and ctx['choices']['C_guard']==(0 if 0 in ctx['allowed'] else expected)
        hp=min(valid,key=lambda j:(-ctx['scores']['H'][j],j)) if valid else ctx['choices']['H_rank']
        assert hp==ctx['choices']['H_probe']
    assert max(errors)<1e-10
    if '--preflight' in sys.argv:
        dump('PREFLIGHT.json',{'hashes':len(receipt['hashes']),'probes':24,'max_error':max(errors),'full_generation_calls':0})
        print('preflight passed',max(errors));return
    r=read(ROOT/'COMPLETE.json');sr=read(ROOT/'采样INPUT_RECEIPT.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sr['hashes'].items())
    assert r['rows']==[json.loads(line) for line in (ROOT/'原始记录.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(r['rows'])==96
    direct=[];retention={ref:{'old_correct':0,'retained':0,'C_correct':0} for ref in ['original_rank','H_rank','H_guard','C_guard']}
    per_parent=[]
    for ctx,s in zip(frozen['contexts'],r['summaries']):
        assert ctx['parent']==s['parent'];pool=[x for x in r['rows'] if x['parent']==ctx['parent']]
        assert {(x['j'],x['repeat'],x['seed']) for x in pool}=={(j,i,seed) for j in range(6) for i,seed in enumerate(ctx['seeds'])}
        for x in pool:
            assert x['routes'][:-1]==ctx['routes'][:x['j']]+ctx['routes'][x['j']+1:]
            D,P,O,legal=assess(x['routes']);assert legal==x['feasible'] and (P is None)==(x['P'] is None)
            gain=max(0.,(ctx['base']['P']-P)/66+(ctx['base']['O']-O)/840) if legal else 0.
            expected=[O,gain,P if gain>0 else ctx['base']['P'],O if gain>0 else ctx['base']['O']]
            actual=[x[k] for k in ['O','gain','accepted_P','accepted_O']]
            if P is not None:expected.append(P);actual.append(x['P'])
            errors.append(float(np.max(np.abs(np.asarray(expected)-actual))))
        gs=[float(np.mean([x['gain'] for x in pool if x['j']==j])) for j in range(6)]
        for method,j in ctx['choices'].items():
            for field in ['gain','accepted_P','accepted_O']:
                errors.append(abs(np.mean([x[field] for x in pool if x['j']==j])-s['methods'][method][field]))
        best=max(gs[j] for j in (ctx['allowed'] or range(6)))
        cg=s['methods']['C_rank']['gain']
        for ref,counts in retention.items():
            oc=abs(s['methods'][ref]['gain']-best)<1e-12;cc=abs(cg-best)<1e-12
            counts['old_correct']+=oc;counts['retained']+=oc and cc;counts['C_correct']+=cc
        ccost=ctx['times']['selection_total']+float(np.mean([x['generation_seconds']+x['evaluation_seconds'] for x in pool if x['j']==ctx['choices']['C_rank']]))
        dcosts=[];dg=[]
        for repeat in range(4):
            xs=[x for x in pool if x['repeat']==repeat];choice=min(xs,key=lambda x:(-x['gain'],x['j']))
            dg.append(choice['gain']);dcosts.append(sum(x['generation_seconds']+x['evaluation_seconds'] for x in xs))
        per_parent.append({'parent':ctx['parent'],'best_gain':best,'C_gain':cg,'C_regret':best-cg,
            'selection_seconds':ctx['times'],'C_selection_plus_one_seconds':ccost,
            'direct_six_gain':float(np.mean(dg)),'direct_six_generation_eval_seconds':float(np.mean(dcosts)),
            'direct_six_plus_model_state_init_seconds':float(np.mean(dcosts))+ctx['times']['model_state_init']})
    assert max(errors)<1e-10
    out={'max_error':max(errors),'hashes':len(sr['hashes']),'probes_checked':24,'records_checked':96,
         'retention':retention,'parents':per_parent,
         'cost_note':'组件计时，直接六次与选线一次预算不同；不作严格等墙钟结论'}
    dump('VERIFICATION.json',out);print(json.dumps(out))


if __name__=='__main__':main()
