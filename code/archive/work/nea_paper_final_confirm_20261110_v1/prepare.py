"""仅已有训练数据内拟合回归；新父网仅初始化，不运行搜索。"""
import sys,json,time,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from driver import d,WORK
import numpy as np

def fit_ridge(tasks,encoding,lam):
    learner=d.b.learner
    X=np.array([learner.encode(t['x'],learner.SPECS[t['city']],encoding) for t in tasks])
    mean=X.mean(0);scale=X.std(0);scale[scale<1e-12]=1.
    states=list(dict.fromkeys(t['state'] for t in tasks));ids=[np.array([i for i,t in enumerate(tasks) if t['state']==s]) for s in states]
    cities=[tasks[ix[0]]['city'] for ix in ids];p=learner.probabilities(cities)
    y=np.array([t['gain'] for t in tasks]);maxima=np.array([y[ix].max() for ix in ids]);reference=float(p@maxima)
    assert reference>0
    group_weights=p*maxima/reference;weights=np.zeros(len(tasks))
    for ix,m,w in zip(ids,maxima,group_weights):y[ix]/=max(m,1e-12);weights[ix]=w/len(ix)
    Z=np.column_stack([np.ones(len(X)),(X-mean)/scale]);pen=np.eye(Z.shape[1]);pen[0,0]=0
    A=Z.T@(weights[:,None]*Z)+lam*pen;rhs=Z.T@(weights*y)
    coef=np.linalg.solve(A,rhs);residual=float(np.max(np.abs(A@coef-rhs)))
    assert residual<1e-8
    return {'encoding':encoding,'model':{'indices':list(range(X.shape[1])),'mean':mean.tolist(),'scale':scale.tolist(),'coef':coef.tolist(),'lambda':lam}}, {'normal_equation_residual':residual,'weights_sum':float(weights.sum())}

def canonical(routes):return tuple(sorted(min(tuple(r),tuple(r[::-1])) for r in routes))
def discover(value,known):
    if isinstance(value,dict):
        for v in value.values():discover(v,known)
    elif isinstance(value,list):
        if len(value) in [6,12] and all(isinstance(rr,list) and rr and all(type(n)==int and n>=0 for n in rr) for rr in value):
            known.add(('mandl' if len(value)==6 else 'mumford0',canonical(value)))
        else:
            for v in value:discover(v,known)

def main():
    started=time.perf_counter();models={'full':{city:d.frozen(city) for city in ['mandl','mumford0']},'ridge':{}}
    old=d.read(WORK/'nea_spo_v2_matched_ablation_20261108_v1/模型冻结.json')
    models['ridge']['mandl']=old['models']['no_spo_ridge']
    assert models['full']['mandl']==old['models']['v2_spo']
    trained=d.read(WORK/'nea_spo_v2_joint_20261109_v1/模型冻结.json')
    tasks=d.read(WORK/'nea_spo_v2_joint_20261109_v1/训练任务.json')
    assert len(tasks)==211 and trained['training_source_sha256']==d.sha(WORK/'nea_spo_v2_joint_20261109_v1/训练任务.json')
    cv=[]
    for numerator in [.1,1.,10.]:
        lam=numerator/79.;rows=[]
        for fold in trained['folds']:
            fit,check=fit_ridge([tasks[i] for i in fold['train_ids']],trained['selected_encoding'],lam)
            rows.extend(d.b.learner.regret(fit,[tasks[i] for i in fold['test_ids']])['rows'])
        city_means={city:float(np.mean([r['regret'] for r in rows if r['city']==city])) for city in ['mandl','mumford0']}
        cv.append({'lambda':lam,'balanced_regret':sum(city_means.values())/2,'city_means':city_means,'rows':rows})
    selected=min(cv,key=lambda r:(r['balanced_regret'],r['lambda']))
    document,check=fit_ridge(tasks,trained['selected_encoding'],selected['lambda'])
    for field in ['mean','scale','indices']:assert document['model'][field]==trained['document']['model'][field]
    models['ridge']['mumford0']=document
    d.dump(ROOT/'模型冻结.json',{'full':models['full'],'ridge':models['ridge'],'mumford_ridge_CV':cv,'selected':selected,'checks':check,
       'training_tasks':{'mandl':115,'mumford0':211},'test_outcomes_seen':False,'fit_seconds':time.perf_counter()-started,
       'scope':'Mumford0按同祖先闭包分组CV、同特征编码、同状态收益归一化和状态权重；组内平方损失按任务平均。回归和SPO+损失数值尺度不同，不声称正则效果相同。Mandl复用原匹配回归。'})
    known=set();paths=set()
    for pattern in ['*/请求冻结.json','*/验证冻结.json','*/冻结.json','*/输入冻结.json','*/父网与特征.json']:
        paths.update(WORK.glob(pattern))
    for p in sorted(paths):
        if p.parent==ROOT:continue
        discover(d.read(p),known)
    d.dump(ROOT/'新起点查重.json',{'sources':{str(p):d.sha(p) for p in sorted(paths) if p.parent!=ROOT},'known_services':len(known)})
    contexts=[]
    for city in ['mandl','mumford0']:
        d.b.configure(city);q=d.q;n,nr,limit=(15,6,8) if city=='mandl' else (30,12,15)
        instance=q.c.BenchmarkInstance(tuple(range(1,n+1)),{(u+1,v+1):float(q.ROADS[u,v]) for u in range(n) for v in range(n) if u!=v and np.isfinite(q.ROADS[u,v])},{(u+1,v+1):float(q.Q[u,v]) for u in range(n) for v in range(n)},frozenset(range(1,n+1)),{})
        for ident in range(62811,62815):
            seed=q.ref.stream(ident,'initialization').getrandbits(64)
            routes=[[u-1 for u in rr] for rr in q.ref.make_initial_routes(instance,nr,(2,limit),seed)]
            assert q.assess(routes)['feasible'] and (city,canonical(routes)) not in known, (city,ident,'duplicate or infeasible')
            known.add((city,canonical(routes)))
            contexts.append({'city':city,'parent':('mandl1-6-' if city=='mandl' else 'mumford0-12-')+str(ident),'routes':routes,'initialization_seed':seed})
    d.dump(ROOT/'请求冻结.json',{'contexts':contexts,'seeds':[1500,1501,1502],'methods':['author','full','count','ridge'],'iterations':8,
       'budgets':{'mandl':4.,'mumford0':7.},'new_training_feedback':False,'scope':'已知道路/OD图上未参与打磨的新父网；不宣称跨城市或完全未见状态泛化'})
    print('FROZEN',len(contexts),'parents; 96 runs; ridge lambda',selected['lambda'],flush=True)

if __name__=='__main__':main()
