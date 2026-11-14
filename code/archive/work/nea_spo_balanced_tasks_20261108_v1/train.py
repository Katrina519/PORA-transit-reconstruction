"""保持模型表示与损失不变，仅补完整任务反馈。"""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent

def balanced_labels(rows,n_actions,seeds):
    expected={(j,s) for j in range(n_actions) for s in seeds}
    actual=[(x['j'],x['seed']) for x in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('missing or duplicate state-action-seed observations')
    return [sum(x['gain'] for x in rows if x['j']==j)/len(seeds) for j in range(n_actions)]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(n,v):
    with (ROOT/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    start=time.perf_counter()
    sys.path.insert(0,str(WORK/'nea_spo_cost_scale_20261108_v1'));import weighted_spo as fitref
    import run_stage as stage
    prep=stage.q.load('balanced_context_features',WORK/'nea_spo_context_20261108_v1/prepare.py')
    oldtasks,oldfreeze=fitref.load_data();oldmodel=read(WORK/'nea_spo_cost_scale_20261108_v1/候选冻结.json')['models']['weighted_spo']
    d=read(ROOT/'诊断完成.json');added=[]
    for context in d['records']:
        selected=[t for t in d['labels'] if t['state']==context['parent']]
        rows=[r for t in selected for r in t['rows']]
        gains=balanced_labels(rows,6,list(range(310,315)))
        X=prep.features(context['routes'])
        for j in range(6):
            assert abs(gains[j]-context['gains'][j])<1e-12
            added.append({'key':context['parent']+'::'+str(j),'state':context['parent'],'j':j,
                'ancestors':[context['parent'].split('-round2')[0]],'gain':gains[j],'n':5,'x':X[j]})
    full=oldtasks+added;initial=oldtasks+[t for t in added if '-round2-' not in t['state']]
    models={'previous_spo':oldmodel};checks={}
    for name,tasks in [('balanced_spo',full),('initial_only_spo',initial)]:
        # Keep regularization per mean task loss constant when the task count changes.
        lam=.1*len(tasks)/len(oldtasks)
        models[name],checks[name]=fitref.fit(tasks,lam,True,oldmodel['mean'],oldmodel['scale'])
    # Same complete feedback and features, replace only the decision loss with square loss.
    X=np.asarray([t['x'] for t in full]);Z=(X-oldmodel['mean'])/oldmodel['scale']
    y=np.asarray([t['gain'] for t in full]);groups=[np.asarray([i for i,t in enumerate(full) if t['state']==s]) for s in dict.fromkeys(t['state'] for t in full)]
    maxima=[float(y[ids].max()) for ids in groups];counts=[len(ids) for ids in groups]
    ws=fitref.normalized_weights(maxima,counts);weights=np.zeros(len(full))
    for ids,m,w in zip(groups,maxima,ws):y[ids]/=max(m,1e-12);weights[ids]=w
    Z1=np.column_stack([np.ones(len(Z)),Z]);pen=np.eye(Z1.shape[1]);pen[0,0]=0
    lam=.1*len(full)/len(oldtasks)
    coef=np.linalg.solve(Z1.T@(weights[:,None]*Z1)+lam*pen,Z1.T@(weights*y))
    models['balanced_ridge']=dict(models['balanced_spo'],coef=coef.tolist())
    paths=[Path(__file__),ROOT/'协议.md',ROOT/'test_training.py',ROOT/'诊断冻结.json',ROOT/'诊断完成.json',Path(fitref.__file__),Path(prep.__file__),WORK/'nea_spo_cost_scale_20261108_v1/候选冻结.json']
    result={'models':models,'fit_seconds':time.perf_counter()-start,'checks':checks,
            'training_tasks':len(full),'training_states':len(groups),'training_samples':sum(t['n'] for t in full),
            'old_tasks':len(oldtasks),'added_tasks':len(added),'added_samples':180,
            'lambda_per_mean_loss':.1/len(oldtasks),'lambda_rule':'.1 * N_tasks / 79',
            'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'diagnostic_replay':{name:fitref.regret(model,added) for name,model in models.items()}}
    dump('训练任务.json',full);dump('候选冻结.json',result)
    print('TRAIN_FROZEN',len(full),len(groups),result['fit_seconds'],flush=True)
    for name,rows in result['diagnostic_replay'].items():print(name,[(r['state'],r['choice'],round(r['regret'],6)) for r in rows],flush=True)

if __name__=='__main__':main()
