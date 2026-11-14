"""用独立线路身份图核验首片段和保存结果，不调用神经模型。"""
import collections
import hashlib
import heapq
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def distances(roads,routes):
    at=collections.defaultdict(list);adj=collections.defaultdict(list);n=len(roads)
    for j,rr in enumerate(routes):
        for u in rr:at[u].append((j,u))
        for u,v in zip(rr,rr[1:]):
            adj[j,u].append(((j,v),float(roads[u,v])));adj[j,v].append(((j,u),float(roads[v,u])))
    for u,states in at.items():
        for a in states:adj[a].extend((b,5.) for b in states if b!=a)
    D=np.full((n,n),np.inf);np.fill_diagonal(D,0.)
    for src in range(n):
        dd={a:0. for a in at[src]};todo=[(0.,a) for a in at[src]];heapq.heapify(todo)
        while todo:
            value,a=heapq.heappop(todo)
            if value!=dd[a]:continue
            D[src,a[1]]=min(D[src,a[1]],value)
            for b,t in adj[a]:
                nv=value+t
                if nv<dd.get(b,np.inf):dd[b]=nv;heapq.heappush(todo,(nv,b))
    return D


def main():
    receipt=read(ROOT/'INPUT_RECEIPT.json');assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in receipt['hashes'].items())
    frozen=read(ROOT/'探测冻结.json');result=read(ROOT/'COMPLETE.json')
    roads=np.loadtxt(WORK/'paper_code_learning_20261030_v1/holliday/inputs/MandlTravelTimes.txt')
    q=np.loadtxt(WORK/'paper_code_learning_20261030_v1/holliday/inputs/MandlDemand.txt');Q=float(q.sum());mask=q>0
    errs=[];raw_count=0;choices=0;batch_cache={}
    def stats(routes):
        D=distances(roads,routes);P=float(np.sum(q[mask]*D[mask])/Q) if np.isfinite(D[mask]).all() else None
        O=float(sum(roads[u,v] for rr in routes for u,v in zip(rr,rr[1:])))
        legal=(len(routes)==6 and set().union(*map(set,routes))==set(range(15)) and P is not None and
               all(2<=len(rr)<=8 and len(set(rr))==len(rr) and all(0<roads[u,v]<np.inf for u,v in zip(rr,rr[1:])) for rr in routes))
        return D,P,O,bool(legal)
    for e,rec in zip(frozen['entries'],result['records']):
        assert e['parent']==rec['parent'];R=e['routes'];D0,P0,O0,_=stats(R)
        errs.extend([abs(P0-e['base']['P']),abs(O0-e['base']['O'])])
        for t in e['tasks']:
            j=t['j'];rem=R[:j]+R[j+1:];Dr=distances(roads,rem);D,P,O,legal=stats(rem+[t['route']]);S=mask&np.isfinite(Dr)
            G=sum(float(q[u,v]*(Dr[u,v]-D[u,v])) for u,v in zip(*np.where(S)))/Q
            V=sum(float(q[u,v]*(D0[u,v]-D[u,v])) for u,v in zip(*np.where(S)))/Q
            L=sum(float(q[u,v]*(Dr[u,v]-D0[u,v])) for u,v in zip(*np.where(S)))/Q
            errs.extend([abs(G-t['G']),abs(V-t['V']),abs(L-t['L']),abs(O-t['O'])])
            assert legal==t['feasible'] and (P is None)==(t['P'] is None)
            if P is not None:errs.append(abs(P-t['P']))
            if legal:errs.append(abs((P0-P)/66+(O0-O)/840-t['raw_gain']))
        allowed=e['allowed'];valid=[j for j in allowed if e['tasks'][j]['feasible']];assert valid==e['probe_valid']
        pick=lambda s,ix:min(ix,key=lambda j:(-s[j],j))
        expected={v:pick(e['scores'][v],allowed) for v in ['original','H']}
        for method,field in [('H_probe',None),('G_probe','G'),('C_probe','raw_gain')]:
            values=e['scores']['H'] if field is None else [t[field] if t[field] is not None else -1e100 for t in e['tasks']]
            expected[method]=pick(values,valid) if valid else expected['H']
        for method,j in expected.items():
            assert e['choices'][method]=={'rank':j,'guard':0 if 0 in allowed else j};choices+=2
        if e['batch'] not in batch_cache:batch_cache[e['batch']]=read(WORK/e['batch']/'COMPLETE.json')
        rows=[x for x in batch_cache[e['batch']]['rows'] if x['parent']==e['parent']]
        for x in rows:
            D,P,O,legal=stats(x['routes']);assert legal==x['feasible'] and (P is None)==(x['P'] is None)
            g=max(0.,(P0-P)/66+(O0-O)/840) if legal else 0.
            errs.extend([abs(O-x['O']),abs(g-x['gain'])]);raw_count+=1
            if P is not None:errs.append(abs(P-x['P']))
        for method,modes in e['choices'].items():
            for mode,j in modes.items():
                got=rec['methods'][method][mode]
                assert got['j']==j
                for k in ['gain','accepted_P','accepted_O']:errs.append(abs(got[k]-np.mean([x[k] for x in rows if x['j']==j])))
    assert max(errs)<1e-10 and raw_count==312
    out={'hashes':len(receipt['hashes']),'probes':78,'saved_records':raw_count,'decisions':choices,'max_error':max(errs)}
    with (ROOT/'VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out))


if __name__=='__main__':main()
