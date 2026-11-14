"""按原80行CSV逐项替换新Mumford0 Full/Ridge，保留来源与两种终点。"""
import sys
import csv
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e


def write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def point(run,seconds):
    assert run['fine_curve'][-1]['seconds']>=seconds
    return [p for p in run['fine_curve'] if p['seconds']<=seconds][-1]


def metrics(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    return dict(control_mean=ma,method_mean=mb,improvement_percent=100*(ma-mb)/ma,
                wins=sum(x-y>1e-6 for x,y in zip(a,b)),ties=sum(abs(x-y)<=1e-6 for x,y in zip(a,b)),
                losses=sum(x-y< -1e-6 for x,y in zip(a,b)),n=len(a))


def analyze():
    from scipy.stats import wilcoxon,rankdata
    req=e.read(e.REQUEST);e.check(req['hashes'])
    completed=e.read(ROOT/'COMPLETE.json')
    raw=[json.loads(line) for line in (ROOT/'原始运行.jsonl').read_text(encoding='utf-8').splitlines()]
    assert completed['runs']==raw and len(raw)==80 and completed['request_sha256']==e.sha(e.REQUEST)
    assert [(r['parent'],r['seed'],r['variant']) for r in raw]==[(c['parent'],s,v) for c,s,v in req['tasks']]
    oldraw=[json.loads(line) for line in (e.SOURCE/'comparison/原始运行.jsonl').read_text(encoding='utf-8').splitlines()]
    old={(r['city'],r['parent'],r['seed'],r['variant']):r for r in oldraw}
    new={(r['city'],r['parent'],r['seed'],r['variant']):r for r in raw}
    assert len(old)==320 and len(new)==80
    audit=e.module('unified_independent_service_graph',e.WORK/'nea_spo_v2_joint_20261109_v1/audit.py')
    sys.path.insert(0,str(e.WORK/'nea_spo_feedback_allocation_20261108_v1'))
    from allocator import Allocator
    max_error=0.;replays=0;score_records={}
    for r in raw:
        assert r['generation_counts']==[80,800] and r['target_samples']==811 and r['target_calls']==82
        assert not r['repair'] and not r['fallbacks'] and len(r['feedback_log'])==400 and len(r['fine_curve'])==41
        assert r['score_calls']+r['cache_hits']==400
        for i,p in enumerate(r['fine_curve']):
            s=r['fine_solutions'][p['solution_index']]
            assert p['cost']==s['cost'] and s['first_seconds']<=p['seconds'] and p['samples']==10+20*i
            assert i==0 or r['fine_curve'][i-1]['seconds']<=p['seconds']
        for s in r['fine_solutions']:
            _,_,c=audit.evaluate('mumford0',s['routes']);max_error=max(max_error,abs(c-s['cost']))
        allocator=Allocator(.05)
        for batch in range(80):
            events=[v for v in r['feedback_log'] if v['batch_id']==batch]
            details=r['selection_log'][batch];assert len(events)==len(details)==5
            costs,valid=[0.]*10,[False]*10
            for event,detail in zip(events,details):
                key=tuple(map(tuple,event['parent_routes']))
                assert event['bee']==detail['bee'] and event['j']==detail['new']
                assert event['eligible']==detail['allowed']==list(range(12))
                scorekey=(r['variant'],key)
                if scorekey in score_records:assert score_records[scorekey]==detail['scores']
                else:score_records[scorekey]=detail['scores']
                assert all(math.isfinite(v) for v in detail['scores'])
                assert allocator.choose(key,detail['scores'],event['bee'],event['parent_cost'])==event['j']
                _,_,c=audit.evaluate('mumford0',event['parent_routes']);max_error=max(max_error,abs(c-event['parent_cost']))
                if event['valid']:
                    _,_,c=audit.evaluate('mumford0',event['candidate_routes']);max_error=max(max_error,abs(c-event['candidate_cost']))
                costs[event['bee']],valid[event['bee']]=event['candidate_cost'],event['valid'];replays+=1
            allocator.observe(costs,valid)
            for event,log in zip(events,allocator.log[-5:]):assert abs(event['reward']-log['reward'])<1e-12
    assert replays==32000 and max_error<1e-6
    rows=[];records=[];legacy=[]
    for original in req['rows']:
        row=dict(original);city,parent,seed=row['city'],row['parent'],row['seed']
        budget=req['historical_budget'][city]
        for method in req['methods']:
            oldrun=old[city,parent,seed,method]
            oldpoint=point(oldrun,budget)
            assert abs(original[method+'_budget']-oldpoint['cost'])<1e-12
            assert abs(original[method+'_final']-oldrun['cost'])<1e-12
            updated=city=='mumford0' and method in ['full','ridge']
            r=new[city,parent,seed,method] if updated else oldrun
            p=point(r,budget)
            pb,ob,cb=audit.evaluate(city,r['fine_solutions'][p['solution_index']]['routes'])
            pf,of,cf=audit.evaluate(city,r['routes'])
            max_error=max(max_error,abs(cb-p['cost']),abs(cf-r['cost']))
            if updated:
                row.update({method+'_budget':p['cost'],method+'_final':r['cost'],method+'_P':pb,method+'_O':ob,
                    method+'_online_seconds':r['seconds'],method+'_selection_seconds':r['selection_seconds'],
                    method+'_feedback_seconds':r['feedback_seconds'],
                    method+'_completed_passes':sum(x['seconds']<=budget for x in r['fine_curve'][1:])})
                legacy.append(dict(city=city,parent=parent,seed=seed,method=method,
                    old_budget=oldpoint['cost'],new_budget=p['cost'],old_final=oldrun['cost'],new_final=r['cost']))
            records.append(dict(city=city,parent=parent,seed=seed,method=method,
                budget=budget,cost_budget=p['cost'],cost_final=r['cost'],P_budget=pb,O_budget=ob,P_final=pf,O_final=of,
                online_seconds=r['seconds'],selection_seconds=r['selection_seconds'],feedback_seconds=r['feedback_seconds'],
                source='本轮新增在线版' if updated else '原CSV冻结记录',
                allocation='作者原选线' if method=='author' else '简单计数' if method=='count' else '先验与在线收益分配'))
        if city=='mumford0':
            for method in ['full','count','ridge']:
                for suffix in ['budget','final']:
                    key=method+('_improvement_percent' if suffix=='budget' else '_final_improvement_percent')
                    row[key]=100*(row['author_'+suffix]-row[method+'_'+suffix])/row['author_'+suffix]
        if city=='mandl':
            assert all(row[k]==original[k] for k in original)
        else:
            assert all(row[k]==original[k] for k in original if k.startswith(('author_','count_')) or k in ['city','parent','seed'])
        rows.append(row)
    assert len(rows)==80 and len(records)==320 and len(legacy)==80 and max_error<1e-6
    summary=[];ablations=[];parents=[]
    for endpoint in ['final','budget']:
        for city in ['mandl','mumford0']:
            subset=[v for v in records if v['city']==city]
            pairs=defaultdict(dict)
            for r in subset:pairs[r['parent'],r['seed']][r['method']]=r
            index=[pairs[k] for k in sorted(pairs)]
            assert len(index)==40
            authors=[v['author']['cost_'+endpoint] for v in index]
            for method in req['methods']:
                values=[v[method]['cost_'+endpoint] for v in index]
                stat=metrics(authors,values)
                stat.update(endpoint=endpoint,city=city,method=method,cost_sd=statistics.stdev(values),
                    P_mean=statistics.mean(v[method]['P_'+endpoint] for v in index),
                    O_mean=statistics.mean(v[method]['O_'+endpoint] for v in index))
                summary.append(stat)
            full=[v['full']['cost_'+endpoint] for v in index]
            for control in ['author','count','ridge']:
                values=[v[control]['cost_'+endpoint] for v in index]
                stat=metrics(values,full)
                delta=[0. if abs(a-b)<=1e-6 else a-b for a,b in zip(values,full)]
                if any(delta):
                    stat['one_sided_p_full_lower']=float(wilcoxon(delta,alternative='greater',zero_method='wilcox').pvalue)
                    nz=[v for v in delta if v];ranks=rankdata([abs(v) for v in nz])
                    stat['rank_biserial_positive_full_better']=float(sum(rank*(1 if d>0 else -1) for rank,d in zip(ranks,nz))/sum(ranks))
                else:
                    stat['one_sided_p_full_lower']=1.;stat['rank_biserial_positive_full_better']=0.
                stat.update(endpoint=endpoint,city=city,control=control);ablations.append(stat)
            for parent in sorted({r['parent'] for r in subset}):
                group=[p for key,p in sorted(pairs.items()) if key[0]==parent]
                for method in req['methods']:
                    stat=metrics([v['author']['cost_'+endpoint] for v in group],[v[method]['cost_'+endpoint] for v in group])
                    stat.update(endpoint=endpoint,city=city,parent=parent,method=method);parents.append(stat)
    write_csv(ROOT/'全部逐次_统一在线方法.csv',rows)
    write_csv(ROOT/'方法逐次与来源.csv',records)
    write_csv(ROOT/'旧新Mumford0对照.csv',legacy)
    write_csv(ROOT/'汇总.csv',summary)
    write_csv(ROOT/'消融配对.csv',ablations)
    write_csv(ROOT/'按父网汇总.csv',parents)
    output=dict(passed=True,rows=rows,records=records,summary=summary,ablations=ablations,parents=parents,legacy=legacy,
                new_runs=80,reused_runs=240,new_feedback_replays=replays,independent_legal_networks=len(audit.CACHE),
                max_cost_error=max_error,hashes_checked=len(req['hashes']),source_csv=req['source_table'],
                source_csv_sha256=req['source_table_sha256'],primary_endpoint='final',analysis_sha256=e.sha(__file__))
    e.dump(ROOT/'统计与核验.json',output)
    e.check(req['hashes'])
    print(json.dumps(dict(new_runs=80,replays=replays,max_error=max_error,summary=summary,ablations=ablations),ensure_ascii=False,indent=2))


if __name__=='__main__':analyze()
