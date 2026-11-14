"""独立层图最短路，核验双网络冻结配对结果。"""
import json
import math
import heapq
import hashlib
import csv
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
DATA={}
CACHE={}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(city,routes):
    key=(city,tuple(tuple(rr) for rr in routes))
    if key in CACHE:return CACHE[key]
    if city not in DATA:
        folder=WORK/('paper_code_learning_20261030_v1/holliday/inputs' if city=='mandl' else 'nea_spo_v2_mumford0_20261109_v1/inputs')
        prefix='Mandl' if city=='mandl' else 'Mumford0'
        DATA[city]=tuple([[float(v) for v in row.split()] for row in (folder/(prefix+suffix+'.txt')).read_text().splitlines() if row.strip()]
                         for suffix in ['TravelTimes','Demand'])
    road,Q=DATA[city];n=len(road);nr=6 if city=='mandl' else 12;limit=8 if city=='mandl' else 15
    d=33. if city=='mandl' else 26.
    assert len(routes)==nr and set().union(*map(set,routes))==set(range(n))
    assert all(2<=len(rr)<=limit and len(rr)==len(set(rr)) for rr in routes)
    at=[[] for _ in range(n)];adj={}
    for j,rr in enumerate(routes):
        for u in rr:at[u].append(j);adj[j,u]=[]
        for u,v in zip(rr,rr[1:]):
            assert 0<road[u][v]<math.inf
            adj[j,u].append((road[u][v],j,v));adj[j,v].append((road[v][u],j,u))
    total=0.
    for s in range(n):
        queue=[(0.,j,s) for j in at[s]];heapq.heapify(queue);dist={(j,s):0. for j in at[s]};best=[math.inf]*n;best[s]=0.
        while queue:
            val,j,u=heapq.heappop(queue)
            if val!=dist[j,u]:continue
            best[u]=min(best[u],val)
            for w,k,v in adj[j,u]+[(5.,k,u) for k in at[u] if k!=j]:
                trial=val+w
                if trial<dist.get((k,v),math.inf):dist[k,v]=trial;heapq.heappush(queue,(trial,k,v))
        for t in range(n):
            if Q[s][t]>0:assert math.isfinite(best[t]);total+=Q[s][t]*best[t]
    p=total/sum(map(sum,Q));o=sum(road[u][v] for rr in routes for u,v in zip(rr,rr[1:]))
    CACHE[key]=(p,o,p/(2*d)+o/(nr*d));return CACHE[key]


def check_choices(item,n):
    details=[v for batch in item['selection_log'] for v in batch]
    if item['method']=='author':assert not details;return 0
    for v in details:
        if 'fallback' in v:continue
        assert v['allowed']==list(range(n)) and len(v['scores'])==n
        assert all(math.isfinite(s) for s in v['scores'])
        assert v['new']==min(range(n),key=lambda j:(-v['scores'][j],j))
    counted=sum('fallback' not in v for v in details)
    assert item['score_calls']+item['cache_hits']==counted
    return counted


def main():
    request=read(ROOT/'请求冻结.json');complete=read(ROOT/'confirmation/COMPLETE.json');runs=complete['runs']
    assert all(sha(p)==h for p,h in complete['input_hashes'].items())
    keys={(c['city'],c['parent'],s,m) for c in request['contexts'] for s in request['seeds'] for m in request['methods']}
    idx={(v['city'],v['parent'],v['seed'],v['method']):v for v in runs}
    assert len(idx)==len(runs)==24 and set(idx)==keys
    rows=[];maximum=0.;decisions=0
    for v in runs:
        city=v['city'];n=6 if city=='mandl' else 12
        assert v['target_samples']==811 and v['target_calls']==82 and len(v['fine_curve'])==41
        decisions+=check_choices(v,n)
        for solution in v['fine_solutions']:
            _,_,cost=evaluate(city,solution['routes']);maximum=max(maximum,abs(cost-solution['cost']))
        for event in v['fine_curve']:
            solution=v['fine_solutions'][event['solution_index']]
            assert solution['first_seconds']<=event['seconds'] and solution['cost']==event['cost']
        for label,seconds in [('primary',request['primary_seconds'][city]),('fixed_8',None)]:
            covered=seconds is None or v['fine_curve'][-1]['seconds']>=seconds
            event=v['fine_curve'][-1] if seconds is None else next((p for p in v['fine_curve'][::-1] if p['seconds']<=seconds),None)
            assert covered and event is not None, ('insufficient budget',city,v['parent'],v['seed'])
            p,o,_=evaluate(city,v['fine_solutions'][event['solution_index']]['routes'])
            rows.append({'city':city,'parent':v['parent'],'seed':v['seed'],'method':v['method'],'scope':label,
                         'cost':event['cost'],'P_minutes':p,'O_single_minutes':o,'online_seconds':v['seconds'],
                         'selection_seconds':v['selection_seconds']})
    assert maximum<1e-6
    results={}
    for city in ['mandl','mumford0']:
        results[city]={}
        for label in ['primary','fixed_8']:
            part=[r for r in rows if r['city']==city and r['scope']==label]
            means={m:{k:statistics.mean(r[k] for r in part if r['method']==m) for k in
                     ['cost','P_minutes','O_single_minutes','online_seconds','selection_seconds']} for m in request['methods']}
            for m in means:means[m]['cost_sd']=statistics.stdev(r['cost'] for r in part if r['method']==m)
            index={(r['parent'],r['seed'],r['method']):r for r in part};pairs=[];parents=[]
            for parent in sorted({r['parent'] for r in part}):
                for seed in request['seeds']:
                    a=index[parent,seed,'author']['cost'];b=index[parent,seed,'joint_spo']['cost']
                    pairs.append({'parent':parent,'seed':seed,'author':a,'joint_spo':b,'delta':a-b,'improvement_percent':100*(a-b)/a})
                a=statistics.mean(index[parent,s,'author']['cost'] for s in request['seeds'])
                b=statistics.mean(index[parent,s,'joint_spo']['cost'] for s in request['seeds'])
                parents.append({'parent':parent,'author':a,'joint_spo':b,'improvement_percent':100*(a-b)/a})
            a,b=means['author']['cost'],means['joint_spo']['cost']
            results[city][label]={'means':means,'improvement_percent':100*(a-b)/a,
                'wins':sum(v['delta']>1e-6 for v in pairs),'ties':sum(abs(v['delta'])<=1e-6 for v in pairs),
                'losses':sum(v['delta']< -1e-6 for v in pairs),'pairs':pairs,'parents':parents}
    out={'runs':len(runs),'max_cost_error':maximum,'unique_solutions':len(CACHE),'choices':decisions,
         'cost_samples':sum(v['target_samples'] for v in runs),'results':results,
         'import_seconds':complete['import_seconds'],'wall_seconds':complete['wall_seconds'],
         'all_inputs_unchanged':True,'scope':'两实例、各两新父网各三种子的小规模确认，不混算原始cost'}
    with (ROOT/'确认统计.json').open('x',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2)
    with (ROOT/'逐次结果.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
