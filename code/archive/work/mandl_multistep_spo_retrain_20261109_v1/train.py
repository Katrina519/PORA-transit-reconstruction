"""先冻结探测训练集与新起点，再进行一次SPO+微调。"""
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;JOINT=WORK/'nea_spo_v2_joint_20261109_v1'
IMPORT_START=time.perf_counter();sys.path.insert(0,str(ROOT));sys.path.insert(0,str(JOINT))
import bridge as b
from learner import fit_multi
IMPORT_SECONDS=time.perf_counter()-IMPORT_START
FEEDBACK=WORK/'mandl_multistep_feedback_20261109_v1'
TRACE=WORK/'mandl_multistep_credit_trace_20261109_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def canonical(R):return tuple(sorted(min(tuple(r),tuple(reversed(r))) for r in R))


def prepare():
    started=time.perf_counter();prior=read(JOINT/'模型冻结.json')['document'];b.configure('mandl',prior)
    source=read(FEEDBACK/'请求冻结.json');seed_set=set(source['development'])
    raw=[json.loads(s) for s in (FEEDBACK/'development.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(raw)==108 and {r['seed'] for r in raw}==seed_set and not seed_set&set(source['confirmation'])
    tasks=[];training=[];feature_seconds=0.
    for case in source['cases']:
        R=case['snapshot']['routes'];tick=time.perf_counter();X=b.features(R);feature_seconds+=time.perf_counter()-tick
        scores=b.learner.scores(prior,X,'mandl').tolist()
        assert max(abs(x-y) for x,y in zip(scores,case['snapshot']['scores']))<1e-10
        Q=[sum(r['cost'] for r in raw if r['id']==case['id'] and r['j']==j)/6 for j in range(6)]
        span=max(Q)-min(Q);assert span>0
        training.append({'id':case['id'],'routes':R,'Q':Q,'span':span,'source_parent':case['context']['parent'],
                         'source_search_seed':case['context']['search_seed']})
        for j in range(6):
            tasks.append({'state':case['id'],'j':j,'x':X[j],'gain':max(Q)-Q[j],
                'terminal_cost':Q[j],'n':6,'city':'mandl','ancestors':[case['context']['parent']]})
    assert len(tasks)==18
    known={canonical(c['snapshot']['routes']) for c in source['cases']}
    oldroutes=read(WORK/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes']
    known.update(canonical(R) for R in oldroutes.values())
    olddiag=read(WORK/'nea_spo_balanced_tasks_20261108_v1/诊断完成.json')['records']
    known.update(canonical(v['routes']) for v in olddiag)
    known.update(canonical(c['routes']) for c in read(JOINT/'输入冻结.json')['contexts'] if c['city']=='mandl')
    history=[]
    for folder in WORK.iterdir():
        if not folder.is_dir() or folder==ROOT or not folder.name.startswith(('nea_','mandl_')):continue
        for name in ['请求冻结.json','部署冻结.json','运行冻结.json','请求确认输入.json']:
            path=folder/name
            if not path.is_file():continue
            data=read(path);history.append({'path':str(path),'sha256':sha(path)})
            for c in data.get('contexts',[]):
                if c.get('routes') and len(c['routes'])==6:known.add(canonical(c['routes']))
    for run in read(JOINT/'confirmation/COMPLETE.json')['runs']:
        if run['city']=='mandl':known.update(canonical(s['routes']) for s in run['fine_solutions'])
    q=b.q
    instance=q.c.BenchmarkInstance(tuple(range(1,16)),
        {(u+1,v+1):float(q.ROADS[u,v]) for u in range(15) for v in range(15) if u!=v and q.np.isfinite(q.ROADS[u,v])},
        {(u+1,v+1):float(q.Q[u,v]) for u in range(15) for v in range(15)},frozenset(range(1,16)),{})
    contexts=[]
    for ident in [62621,62622]:
        name='mandl1-6-'+str(ident)
        assert all(name not in Path(v['path']).read_text(encoding='utf-8-sig') for v in history),'previous parent ID'
        seed=q.ref.stream(ident,'initialization').getrandbits(64)
        R=[[u-1 for u in r] for r in q.ref.make_initial_routes(instance,6,(2,8),seed)]
        assert canonical(R) not in known and q.assess(R)['feasible']
        known.add(canonical(R));contexts.append({'city':'mandl','parent':name,'routes':R,'initialization_seed':seed})
    hashes=dict(source['input_hashes'])
    for folder in [FEEDBACK,TRACE,JOINT,WORK/'nea_spo_v2_mumford0_20261109_v1']:
        for name,h in read(folder/'封存清单.json')['files'].items():
            assert sha(folder/name)==h;hashes[str(folder/name)]=h
    hashes.update({str(p):sha(p) for p in ROOT.iterdir() if p.is_file()})
    for v in history:hashes[v['path']]=v['sha256']
    assert all(sha(p)==h for p,h in hashes.items())
    dump(ROOT/'训练任务.json',tasks)
    dump(ROOT/'请求冻结.json',{'contexts':contexts,'seeds':[790,791,792],
        'methods':['author','single_spo','multi_spo'],'primary_seconds':4.,'max_iterations':8,
        'role':'two new parents, three paired seeds; no hyperparameter changes after outcomes'})
    dump(ROOT/'输入冻结.json',{'input_hashes':hashes,'prior':prior,'training_states':training,
        'training_seeds':sorted(seed_set),'training_feedback_rows':108,'training_tasks':18,
        'confirmation_feedback_used':False,'training_file_sha256':sha(ROOT/'训练任务.json'),
        'test_request_sha256':sha(ROOT/'请求冻结.json'),'history_receipts':history,
        'feature_seconds':feature_seconds,'prepare_seconds':time.perf_counter()-started,'import_seconds':IMPORT_SECONDS,
        'reused_feedback_branch_seconds':sum(r['seconds'] for r in raw),
        'old_full_feedback_wall_seconds':read(FEEDBACK/'COMPLETE.json')['wall_seconds']})
    print('PREPARED 18 tasks / 108 development branches / 2 new parents / 0 confirmation labels',flush=True)


def train():
    started=time.perf_counter();frozen=read(ROOT/'输入冻结.json');tasks=read(ROOT/'训练任务.json')
    assert all(sha(p)==h for p,h in frozen['input_hashes'].items())
    assert sha(ROOT/'训练任务.json')==frozen['training_file_sha256']
    assert sha(ROOT/'请求冻结.json')==frozen['test_request_sha256']
    fit_started=time.perf_counter();document,checks=fit_multi(tasks,frozen['prior']);fit_seconds=time.perf_counter()-fit_started
    def regret(doc):
        out=[]
        for state in dict.fromkeys(t['state'] for t in tasks):
            group=[t for t in tasks if t['state']==state];scores=b.learner.scores(doc,[t['x'] for t in group],'mandl')
            j=min(range(6),key=lambda k:(-scores[k],k))
            out.append({'state':state,'choice':j,'regret':max(t['gain'] for t in group)-group[j]['gain']})
        return out
    dump(ROOT/'模型冻结.json',{'document':document,'checks':checks,'fit_seconds':fit_seconds,
        'training_phase_seconds':time.perf_counter()-started,'import_seconds':IMPORT_SECONDS,
        'source_sha256':sha(ROOT/'训练任务.json'),'old_training_regret':regret(frozen['prior']),
        'new_training_regret':regret(document),'training_states':3,'training_tasks':18,
        'confirmation_labels_used':False,'new_test_outcomes_seen':False,'Mandl_only':True})
    print('MODEL FROZEN','fit',round(fit_seconds,5),'KKT',checks['KKT_stationarity'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','train'],required=True)
    args=parser.parse_args();prepare() if args.phase=='prepare' else train()
