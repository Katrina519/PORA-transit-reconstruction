"""固定四组最终确认：完整统计、选择重放及独立目标复核。"""
import sys,json,csv,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from driver import d,CountPolicy
from allocator import Allocator
import numpy as np

def point(r,budget):
    points=[p for p in r['fine_curve'] if p['seconds']<=budget]
    assert points,(r['parent'],r['seed'],r['variant'],'no completed initial plan')
    return points[-1]

def summarize(group,method,metric):
    a=np.array([r['author_'+metric] for r in group]);b=np.array([r[method+'_'+metric] for r in group])
    am=float(a.mean());bm=float(b.mean());roots=sorted({r['parent'] for r in group})
    rng=np.random.default_rng(20261110);bootstrap=[]
    perroot={root:[r for r in group if r['parent']==root] for root in roots}
    for _ in range(5000):
        selected=rng.integers(0,len(roots),len(roots));sample=[row for i in selected for row in perroot[roots[i]]]
        av=sum(r['author_'+metric] for r in sample)/len(sample);bv=sum(r[method+'_'+metric] for r in sample)/len(sample)
        bootstrap.append(100*(av-bv)/av)
    return {'method':method,'metric':metric,'n':len(group),'parents':len(roots),'author_mean':am,'method_mean':bm,
        'author_std':float(a.std(ddof=1)),'method_std':float(b.std(ddof=1)), 'improvement_percent':100*(am-bm)/am,
        'wins':int(np.sum(b<a-1e-6)),'ties':int(np.sum(np.abs(b-a)<=1e-6)),'losses':int(np.sum(b>a+1e-6)),
        'worst_percent':float(np.min(100*(a-b)/a)), 'parent_cluster_descriptive_95_interval':np.quantile(bootstrap,[.025,.975]).tolist()}

def main():
    folder=ROOT/'comparison_timed';doc=d.read(folder/'COMPLETE.json');gate=d.read(ROOT/'gate_timed/COMPLETE.json')
    assert doc['passed'] and gate['passed']
    assert all(doc['hashes'].get(p)==h and d.sha(p)==h for p,h in gate['hashes'].items())
    assert all(d.sha(p)==h for p,h in doc['hashes'].items())
    runs=doc['runs'];raw=[json.loads(s) for s in (folder/'原始运行.jsonl').read_text(encoding='utf-8').splitlines()]
    assert runs==raw and len(runs)==96
    tasks=d.read(folder/'运行冻结.json')['tasks']
    assert len(tasks)==len(runs)
    for r,(c,s,m) in zip(runs,tasks):assert (r['parent'],r['seed'],r['variant'])==(c['parent'],s,m)
    req=d.read(ROOT/'请求冻结.json');groups={};count_choices=0;feedback_choices=0;static_choices=0
    for r in runs:
        assert r['generation_counts']==[80,800] and r['target_samples']==811 and r['target_calls']==82
        assert len(r['fine_curve'])==41 and all(r['fine_curve'][i]['seconds']<=r['fine_curve'][i+1]['seconds'] for i in range(40))
        assert all(r['fine_curve'][i]['cost']>=r['fine_curve'][i+1]['cost'] for i in range(40))
        assert r['seconds']>=r['kernel_seconds']+r['wrapper_preparation_seconds']-1e-6
        assert all(r['metrics'][k]==[0.] for k in ['# disconnected node pairs','# stops out of bounds'])
        key=(r['city'],r['parent'],r['seed']);assert r['variant'] not in groups.setdefault(key,{})
        groups[key][r['variant']]=r
        if r['variant']=='author':continue
        alloc=Allocator(.05);counter=CountPolicy()
        for batch,logs in enumerate(r['selection_log']):
            events=[e for e in r['feedback_log'] if e['batch_id']==batch];bybee={e['bee']:e for e in events}
            costs=[0.]*10;valid=[False]*10
            for log in logs:
                if 'fallback' in log:continue
                e=bybee[log['bee']];key=tuple(tuple(rr) for rr in e['parent_routes'])
                if r['variant']=='count':
                    expected=counter.choose(key,len(log['scores']));assert all(v==0. for v in log['scores']);count_choices+=1
                elif r['city']=='mandl':expected=alloc.choose(key,log['scores'],e['bee'],e['parent_cost']);feedback_choices+=1
                else:expected=min(range(len(log['scores'])),key=lambda j:(-log['scores'][j],j));static_choices+=1
                assert expected==e['j']==log['new'];assert log['allowed']==list(range(len(log['scores'])))
                costs[e['bee']]=e['candidate_cost'];valid[e['bee']]=e['valid']
            if r['city']=='mandl' and r['variant']!='count':
                alloc.observe(costs,valid)
                for e,entry in zip(events,alloc.log[-len(events):]):assert abs(e['reward']-entry['reward'])<1e-12
    assert len(groups)==24 and all(set(v)==set(req['methods']) for v in groups.values())
    rows=[];target_error=0.;targets=0
    for city in ['mandl','mumford0']:
        d.b.configure(city);nr,diam=(6,33.) if city=='mandl' else (12,26.)
        for (c,parent,seed),pair in groups.items():
            if c!=city:continue
            row={'city':city,'parent':parent,'seed':seed}
            for method in req['methods']:
                r=pair[method];p=point(r,req['budgets'][city]);solution=r['fine_solutions'][p['solution_index']]
                for routes,expected in [(solution['routes'],p['cost']),(r['routes'],r['cost'])]:
                    metrics=d.q.assess(routes);assert metrics['feasible'];actual=metrics['P']/(2*diam)+metrics['O']/(nr*diam)
                    target_error=max(target_error,abs(actual-expected));assert abs(actual-expected)<1e-6;targets+=1
                budgetmetrics=d.q.assess(solution['routes'])
                row.update({method+'_budget':p['cost'],method+'_final':r['cost'],method+'_P':budgetmetrics['P'],method+'_O':budgetmetrics['O'],
                    method+'_online_seconds':r['seconds'],method+'_wrapper_seconds':r['wrapper_preparation_seconds'],
                    method+'_selection_seconds':r['selection_seconds'],method+'_feedback_seconds':r['feedback_seconds']})
            rows.append(row)
    with (ROOT/'全部逐次.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[];parent_rows=[];comparisons=[]
    for city in ['mandl','mumford0']:
        group=[r for r in rows if r['city']==city]
        for method in ['full','count','ridge']:
            for metric in ['budget','final']:summary.append(dict(city=city,**summarize(group,method,metric)))
        for root in sorted({r['parent'] for r in group}):
            subset=[r for r in group if r['parent']==root]
            parent_rows.append({'city':city,'parent':root,**{m:statistics.mean(x[m+'_budget'] for x in subset) for m in req['methods']}})
        for comparator in ['count','ridge']:
            a=statistics.mean(r[comparator+'_budget'] for r in group);b=statistics.mean(r['full_budget'] for r in group)
            comparisons.append({'city':city,'comparator':comparator,'full_improvement_percent':100*(a-b)/a,
                 'wins':sum(r['full_budget']<r[comparator+'_budget']-1e-6 for r in group),
                 'ties':sum(abs(r['full_budget']-r[comparator+'_budget'])<=1e-6 for r in group),
                 'losses':sum(r['full_budget']>r[comparator+'_budget']+1e-6 for r in group)})
    out={'summary':summary,'parent_means':parent_rows,'full_vs_controls':comparisons,'choices_replayed':count_choices+feedback_choices+static_choices,
      'count_choices':count_choices,'feedback_choices':feedback_choices,'static_choices':static_choices,'target_records':targets,'target_max_error':target_error,
      'hashes_verified':True,'scope':'仅同两图的4新父网簇、每簇3种子；区间为描述性父网簇重采样，不据运行种子虚增独立网络样本。'}
    d.dump(ROOT/'统计与核验.json',out)
    print(json.dumps({'summary':summary,'full_vs_controls':comparisons,'choices_replayed':out['choices_replayed'],'target_max_error':target_error},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
