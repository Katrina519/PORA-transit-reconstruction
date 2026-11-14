"""独立服务图及截止记录复核，不重跑计时或生成器。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
spec=importlib.util.spec_from_file_location('service_graph_check',WORK/'restoration_attachment_probe_20261108_v1/verify.py')
graph=importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)
roads=np.loadtxt(WORK/'paper_code_learning_20261030_v1/holliday/inputs/MandlTravelTimes.txt')
q=np.loadtxt(WORK/'paper_code_learning_20261030_v1/holliday/inputs/MandlDemand.txt');mask=q>0

def assess(routes):
    D=graph.distances(roads,routes)
    P=float(np.sum(D[mask]*q[mask])/q.sum()) if np.isfinite(D[mask]).all() else None
    O=float(sum(roads[u,v] for rr in routes for u,v in zip(rr,rr[1:])))
    ok=(len(routes)==6 and P is not None and set().union(*map(set,routes))==set(range(15)) and
        all(2<=len(rr)<=8 and len(rr)==len(set(rr)) and all(0<roads[u,v]<np.inf for u,v in zip(rr,rr[1:])) for rr in routes))
    return P,O,bool(ok)

def main():
    frozen=read(ROOT/'部署冻结.json');data=read(ROOT/'COMPLETE.json');runs=data['runs'];receipt=read(ROOT/'RUN_RECEIPT.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in receipt['hashes'].items())
    assert runs==[json.loads(x) for x in (ROOT/'原始运行.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(runs)==64
    expected=[]
    for pi,ctx in enumerate(frozen['contexts']):
        for rep in range(4):
            shift=(pi+rep)%4
            expected.extend((ctx['parent'],rep,m) for m in frozen['methods'][shift:]+frozen['methods'][:shift])
    assert [(r['parent'],r['repeat'],r['method']) for r in runs]==expected
    errors=[];shared={};probe_count=0;gen_count=0;late=0;decisions={}
    for run in runs:
        ctx=frozen['contexts'][run['pi']];R=ctx['routes'];assert run['parent']==ctx['parent']
        P0,O0,ok=assess(R);assert ok
        errors.extend([abs(P0-run['base']['P']),abs(O0-run['base']['O'])])
        sel=run['selection']
        if sel is not None:
            ranking=sorted(sel['allowed'],key=lambda j:(-sel['scores'][j],j)) if sel['allowed'] else [0]
            assert sel['ranking']==ranking
            if run['method']=='C_single':
                assert len(sel['probes'])==6
                for t in sel['probes']:
                    P,O,ok=assess(R[:t['j']]+R[t['j']+1:]+[t['route']]);probe_count+=1
                    assert ok==t['feasible'] and (P is None)==(t['P'] is None)
                    errors.append(abs(O-t['O']))
                    if P is not None:errors.append(abs(P-t['P']))
                    if ok:errors.append(abs((P0-P)/66+(O0-O)/840-t['raw_gain']))
                valid=[j for j in sel['allowed'] if sel['probes'][j]['feasible']]
                chosen=min(valid,key=lambda j:(-sel['probes'][j]['raw_gain'],j)) if valid else ranking[0]
                assert sel['chosen']==chosen
            else:assert not sel['probes'] and sel['chosen']==ranking[0]
            key=(run['parent'],run['method'])
            decision=(sel['chosen'],sel['ranking'],sel['scores'])
            if key in decisions:assert decision==decisions[key]
            else:decisions[key]=decision
        if run['method']=='direct_multi':
            offset=(run['pi']+run['repeat'])%6;order=list(range(6))[offset:]+list(range(6))[:offset]
        elif run['method']=='H_multi':order=sel['ranking']
        else:order=[sel['chosen']]
        assert order==run['order']
        assert 0<run['initialization_seconds']<=run['ready_seconds']<=run['elapsed_seconds']
        visits=[0]*6;last=run['ready_seconds']
        for i,x in enumerate(run['rows']):
            j=order[i%len(order)];assert x['call']==i and x['j']==j and x['visit']==visits[j]
            assert x['seed']==1340000+run['pi']*10000+run['repeat']*1000+visits[j];visits[j]+=1
            assert last<=x['started_seconds']<=x['generation_done_seconds']<=x['finished_seconds']<=run['elapsed_seconds']
            assert x['started_seconds']<.35
            last=x['finished_seconds'];late+=last>.35
            assert x['routes'][:-1]==R[:j]+R[j+1:]
            P,O,ok=assess(x['routes']);assert ok==x['feasible'] and (P is None)==(x['P'] is None)
            g=max(0.,(P0-P)/66+(O0-O)/840) if ok else 0.
            errors.extend([abs(O-x['O']),abs(g-x['gain'])])
            if P is not None:errors.append(abs(P-x['P']))
            key=(run['parent'],run['repeat'],j,x['visit'])
            if key in shared:assert shared[key]==x['routes']
            else:shared[key]=x['routes']
            gen_count+=1
        for b in frozen['budgets']:
            eligible=[x for x in run['rows'] if x['finished_seconds']<=b]
            best=max(eligible,key=lambda x:x['gain']) if eligible else None
            positive=best is not None and best['gain']>0
            expected={'gain':best['gain'] if positive else 0.,'P':best['P'] if positive else P0,
                'O':best['O'] if positive else O0,'completed':len(eligible),'feasible_completed':sum(x['feasible'] for x in eligible),
                'distinct_tasks':len({x['j'] for x in eligible}),'best_call':best['call'] if positive else None}
            actual=run['checkpoints'][str(b)]
            for k in ['gain','P','O']:errors.append(abs(expected[k]-actual[k]))
            for k in ['completed','feasible_completed','distinct_tasks','best_call']:assert expected[k]==actual[k]
    assert gen_count==data['calls'] and max(errors)<1e-10
    summary={}
    for b in frozen['budgets']:
        key=str(b);means={};parents=[]
        for m in frozen['methods']:
            subset=[r for r in runs if r['method']==m]
            means[m]={k:float(np.mean([r['checkpoints'][key][k] for r in subset])) for k in ['gain','P','O','completed','feasible_completed','distinct_tasks']}
            means[m].update({k:float(np.mean([r[k] for r in subset])) for k in ['initialization_seconds','ready_seconds','elapsed_seconds']})
        for ctx in frozen['contexts']:
            ps={m:float(np.mean([r['checkpoints'][key]['gain'] for r in runs if r['method']==m and r['parent']==ctx['parent']])) for m in frozen['methods']}
            parents.append({'parent':ctx['parent'],'gains':ps})
        comparisons={}
        for m in frozen['methods'][1:]:
            ds=[x['gains']['C_single']-x['gains'][m] for x in parents]
            paired=[]
            for ctx in frozen['contexts']:
                for rep in range(4):
                    pair={r['method']:r['checkpoints'][key]['gain'] for r in runs if r['parent']==ctx['parent'] and r['repeat']==rep}
                    paired.append(pair['C_single']-pair[m])
            comparisons[m]={'parent_better':sum(d>1e-12 for d in ds),'parent_worse':sum(d< -1e-12 for d in ds),
                'parent_same':sum(abs(d)<=1e-12 for d in ds),'parent_differences':ds,
                'paired_better':sum(d>1e-12 for d in paired),'paired_worse':sum(d< -1e-12 for d in paired),
                'paired_same':sum(abs(d)<=1e-12 for d in paired)}
        summary[key]={'means':means,'parents':parents,'comparisons':comparisons}
    result={'max_error':max(errors),'hashes':len(receipt['hashes']),'probes':probe_count,'generations':gen_count,
        'late_results_excluded':late,'same_task_seed_identical':True,'summary':summary}
    with (ROOT/'VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result))

if __name__=='__main__':main()
