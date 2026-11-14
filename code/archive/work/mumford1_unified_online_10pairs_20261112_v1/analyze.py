"""仅汇总十组冻结运行；独立线路状态图核验和精确双侧秩检验。"""
import sys
import json
import csv
import math
import heapq
import statistics
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e
CACHE={}
DATA=None


def evaluate(routes):
    global DATA
    key=tuple(map(tuple,routes))
    if key in CACHE:return CACHE[key]
    if DATA is None:
        DATA=tuple([[float(v) for v in row.split()] for row in (e.PILOT/'inputs'/name).read_text().splitlines() if row.strip()]
                   for name in ['Mumford1TravelTimes.txt','Mumford1Demand.txt'])
    roads,demand=DATA;n=70
    assert len(routes)==15 and set().union(*map(set,routes))==set(range(n))
    assert all(10<=len(r)<=30 and len(r)==len(set(r)) for r in routes)
    at=[[] for _ in range(n)];adj={}
    for j,line in enumerate(routes):
        for u in line:at[u].append(j);adj[j,u]=[]
        for u,v in zip(line,line[1:]):
            assert 0<roads[u][v]<math.inf
            adj[j,u].append((roads[u][v],j,v));adj[j,v].append((roads[v][u],j,u))
    total=0.
    for s in range(n):
        queue=[(0.,j,s) for j in at[s]];heapq.heapify(queue)
        dist={(j,s):0. for j in at[s]};best=[math.inf]*n;best[s]=0.
        while queue:
            value,j,u=heapq.heappop(queue)
            if value!=dist[j,u]:continue
            best[u]=min(best[u],value)
            for cost,k,v in adj[j,u]+[(5.,k,u) for k in at[u] if k!=j]:
                trial=value+cost
                if trial<dist.get((k,v),math.inf):dist[k,v]=trial;heapq.heappush(queue,(trial,k,v))
        for t in range(n):
            if demand[s][t]>0:assert math.isfinite(best[t]);total+=demand[s][t]*best[t]
    p=total/sum(map(sum,demand));o=sum(roads[u][v] for line in routes for u,v in zip(line,line[1:]))
    CACHE[key]=(p,o,p/88.+o/660.)
    return CACHE[key]


def signed_rank_exact(delta):
    from scipy.stats import rankdata
    nonzero=[float(v) for v in delta if v!=0.]
    if not nonzero:return dict(W=0.,p_two_sided=1.,nonzero=0,method='exact conditional signed-rank')
    ranks=rankdata([abs(v) for v in nonzero])
    scaled=[int(round(2*v)) for v in ranks]
    observed=sum(r for r,v in zip(scaled,nonzero) if v>0)
    total=sum(scaled);ways=[0]*(total+1);ways[0]=1
    for r in scaled:
        for value in range(total,r-1,-1):ways[value]+=ways[value-r]
    denominator=2**len(nonzero)
    p=min(1.,2*min(sum(ways[:observed+1]),sum(ways[observed:]))/denominator)
    return dict(W=min(observed,total-observed)/2.,p_two_sided=p,nonzero=len(nonzero),
                method='exact conditional signed-rank; zero differences removed; average ranks for ties')


def write_csv(name,rows):
    with (ROOT/name).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze():
    req=e.read(ROOT/'请求冻结.json');e.check(req['hashes'])
    complete=e.read(ROOT/'COMPLETE.json')
    raw=[json.loads(line) for line in (ROOT/'原始运行.jsonl').read_text(encoding='utf-8').splitlines()]
    assert complete['runs']==raw and len(raw)==20 and complete['request_sha256']==e.sha(ROOT/'请求冻结.json')
    assert [(r['seed'],r['variant']) for r in raw]==[(s,v) for _,s,v in req['tasks']]
    assert len({(r['seed'],r['variant']) for r in raw})==20
    sys.path.insert(0,str(e.WORK/'nea_spo_feedback_allocation_20261108_v1'))
    from allocator import Allocator
    maximum=0.;replays=0;rows=[];traces=[];mechanisms=[];score_records={}
    for run in raw:
        assert run['generation_counts']==[80,800] and run['target_samples']==811 and run['target_calls']==82
        assert run['iterations']==8 and len(run['fine_curve'])==41 and not run['repair']
        initial=run['fine_solutions'][run['fine_curve'][0]['solution_index']]
        assert initial['routes']==req['context']['routes'],'each run must start from the frozen same network'
        assert abs(initial['cost']-3.36954665)<2e-7
        for i,point in enumerate(run['fine_curve']):
            solution=run['fine_solutions'][point['solution_index']]
            assert point['cost']==solution['cost'] and solution['first_seconds']<=point['seconds']
            assert point['samples']==10+20*i
            assert i==0 or run['fine_curve'][i-1]['seconds']<=point['seconds']
            traces.append(dict(seed=run['seed'],variant=run['variant'],iteration=point['iteration'],
                pass_index=point['pass'],seconds=point['seconds'],cost=point['cost'],samples=point['samples']))
        for solution in run['fine_solutions']:
            _,_,cost=evaluate(solution['routes']);maximum=max(maximum,abs(cost-solution['cost']))
        if run['variant']=='author':assert not run['selection_log'] and not run['feedback_log']
        else:
            assert not run['fallbacks'] and len(run['feedback_log'])==400 and run['score_calls']+run['cache_hits']==400
            alloc=Allocator(.05)
            for batch in range(80):
                events=[v for v in run['feedback_log'] if v['batch_id']==batch]
                details=run['selection_log'][batch];assert len(events)==len(details)==5
                costs,valid=[0.]*10,[False]*10
                for event,detail in zip(events,details):
                    key=tuple(map(tuple,event['parent_routes']))
                    assert event['j']==detail['new'] and event['bee']==detail['bee']
                    assert event['eligible']==detail['allowed']==list(range(15))
                    assert len(detail['scores'])==15 and all(math.isfinite(v) for v in detail['scores'])
                    if key in score_records:assert detail['scores']==score_records[key]
                    else:score_records[key]=detail['scores']
                    assert alloc.choose(key,detail['scores'],event['bee'],event['parent_cost'])==event['j']
                    # 父网与合法候选均以独立线路状态图核验。
                    _,_,cost=evaluate(event['parent_routes']);maximum=max(maximum,abs(cost-event['parent_cost']))
                    if event['valid']:
                        _,_,cost=evaluate(event['candidate_routes']);maximum=max(maximum,abs(cost-event['candidate_cost']))
                    costs[event['bee']],valid[event['bee']]=event['candidate_cost'],event['valid'];replays+=1
                alloc.observe(costs,valid)
                for event,log in zip(events,alloc.log[-5:]):assert abs(event['reward']-log['reward'])<1e-12
            assert not alloc.pending
        for endpoint in ['8round','120s']:
            point=run['fine_curve'][-1] if endpoint=='8round' else [p for p in run['fine_curve'] if p['seconds']<=120.][-1]
            p,o,cost=evaluate(run['fine_solutions'][point['solution_index']]['routes'])
            maximum=max(maximum,abs(cost-point['cost']))
            rows.append(dict(city='mumford1',parent=run['parent'],seed=run['seed'],method=run['variant'],endpoint=endpoint,
                cost=point['cost'],P_minutes=p,O_oneway_minutes=o,last_completed_seconds=point['seconds'],
                completed_passes=(point['samples']-10)//20,completed_all_before_120=run['seconds']<=120.,
                total_online_seconds=run['seconds'],selection_seconds=run['selection_seconds'],feedback_seconds=run['feedback_seconds'],
                preparation_seconds=run['wrapper_preparation_seconds'],score_calls=run['score_calls'],cache_hits=run['cache_hits']))
        if run['variant']=='full':
            visits=Counter(tuple(map(tuple,v['parent_routes'])) for v in run['feedback_log'])
            tasks={(tuple(map(tuple,v['parent_routes'])),v['j']) for v in run['feedback_log']}
            mechanisms.append(dict(seed=run['seed'],distinct_parents=len(visits),selections_per_parent=400/len(visits),
                distinct_parent_route_tasks=len(tasks),task_coverage=len(tasks)/(15*len(visits)),
                feasible_proposals=sum(v['valid'] for v in run['feedback_log']),
                positive_proposals=sum(v['valid'] and v['parent_cost']-v['candidate_cost']>1e-6 for v in run['feedback_log'])))
    assert replays==4000 and maximum<2e-6
    summary=[];pairs=[]
    for endpoint in ['8round','120s']:
        index={(r['seed'],r['method']):r for r in rows if r['endpoint']==endpoint}
        a=[index[s,'author']['cost'] for s in req['seeds']]
        b=[index[s,'full']['cost'] for s in req['seeds']]
        test=signed_rank_exact([y-x for x,y in zip(a,b)])
        mean_a,mean_b=statistics.mean(a),statistics.mean(b)
        summary.append(dict(endpoint=endpoint,n=10,author_mean=mean_a,full_mean=mean_b,
            author_sd=statistics.stdev(a),full_sd=statistics.stdev(b),
            improvement_percent=100*(mean_a-mean_b)/mean_a,wins=sum(x-y>1e-6 for x,y in zip(a,b)),
            ties=sum(abs(x-y)<=1e-6 for x,y in zip(a,b)),losses=sum(x-y< -1e-6 for x,y in zip(a,b)),**test))
        for seed in req['seeds']:
            x,y=index[seed,'author'],index[seed,'full']
            pairs.append(dict(seed=seed,endpoint=endpoint,author_cost=x['cost'],full_cost=y['cost'],
                improvement_percent=100*(x['cost']-y['cost'])/x['cost'],author_P=x['P_minutes'],full_P=y['P_minutes'],
                author_O=x['O_oneway_minutes'],full_O=y['O_oneway_minutes'],author_seconds=x['total_online_seconds'],
                full_seconds=y['total_online_seconds'],full_selection_seconds=y['selection_seconds'],full_feedback_seconds=y['feedback_seconds']))
    write_csv('逐次方法结果.csv',rows);write_csv('十组配对结果.csv',pairs);write_csv('全部检查点.csv',traces);write_csv('任务反馈概览.csv',mechanisms)
    result=dict(passed=True,paired_runs=10,method_runs=20,summary=summary,rows=rows,pairs=pairs,traces=traces,
                mechanisms=mechanisms,new_feedback_replays=replays,independent_legal_networks=len(CACHE),
                max_independent_cost_error=maximum,source_hashes_checked=len(req['hashes']),
                primary_endpoint='8round',secondary_endpoint='120s',one_fixed_initial_network=True,
                frozen_N24=read_snapshot(),analysis_sha256=e.sha(__file__))
    e.dump(ROOT/'统计与核验.json',result);e.check(req['hashes'])
    print(json.dumps(dict(summary=summary,replays=replays,networks=len(CACHE),maximum=maximum),ensure_ascii=False,indent=2))


def read_snapshot():return e.read(ROOT/'frozen_N24/版本封存.json')


if __name__=='__main__':analyze()
