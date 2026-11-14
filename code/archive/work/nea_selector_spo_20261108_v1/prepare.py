import sys,time,json,copy,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(WORK/'nea_frozen_selector_direct_20261108_v1'))
import run as r
q=r.q
def dump(name,obj):
 with (ROOT/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 start=time.perf_counter()
 frozen=q.read(WORK/'restoration_service_features_20261108_v1/部署冻结.json')
 tasks=q.read(WORK/'restoration_native_transfer_20261105_v1/统一任务池.json')
 parents=q.read(WORK/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes']
 fm=q.read(WORK/'restoration_service_features_20261108_v1/特征.json')
 lookup={t['key']:t for t in tasks};tasks=[lookup[k] for k in frozen['train_keys']]
 old=frozen['models']['H'];X=np.array([t['x']+[fm[t['key']]['H']] for t in tasks]);Z=(X-old['mean'])/old['scale'];Z1=np.column_stack([np.ones(len(X)),Z])
 penalty=np.eye(11);penalty[0,0]=0
 def ridge(y):return np.linalg.solve(Z1.T@Z1+penalty,Z1.T@y)
 assert np.max(abs(ridge(frozen['targets'])-old['coef']))<1e-10
 groups={s:np.array([i for i,t in enumerate(tasks) if t['state']==s]) for s in parents}
 model,state=q.a.build('neural',next(iter(parents.values())));cost=r.cost_module()
 diameter=float(state.drive_times.max());nr=int(state.n_routes_to_plan[0]);assert abs(diameter-1980)<1e-5 and nr==6
 def C(v):return .5*v['P']*60/diameter+.5*2*v['O']*60/(diameter*nr+1e-6)
 gains=[];oldg=[];checks=[];records=[]
 for t in tasks:
  base=q.assess(parents[t['state']]);assert base['feasible'];vals=[];ov=[]
  for row in t['rows']:
   v=q.assess(row['routes']);assert v['feasible']==row['feasible']
   if v['feasible']:
    assert abs(v['P']-row['P'])<1e-8 and abs(v['O']-row['O'])<1e-8
    st=state.clone();st.replace_routes(q.a.tu.get_batch_tensor_from_routes([row['routes']],max_route_len=8))
    with r.torch.no_grad():actual=float(cost(st).cost[0])
    checks.append(abs(actual-C(v)));gain=max(0.,C(base)-C(v));og=max(0.,(base['P']-v['P'])/66+(base['O']-v['O'])/840)
   else:gain=og=0.
   vals.append(gain);ov.append(og)
  gains.append(float(np.mean(vals)));oldg.append(float(np.mean(ov)))
  assert abs(oldg[-1]-t['value'])<1e-10
  records.append({'key':t['key'],'n':len(vals),'old_gain':oldg[-1],'author_gain':gains[-1]})
 assert max(checks)<1e-6
 gains=np.array(gains);oldg=np.array(oldg);y=gains.copy();yo=oldg.copy();changes=[]
 for s,ids in groups.items():
  y[ids]/=max(gains[ids].max(),1e-12);yo[ids]/=max(oldg[ids].max(),1e-12)
  before=sorted(ids,key=lambda i:(-oldg[i],tasks[i]['j']));after=sorted(ids,key=lambda i:(-gains[i],tasks[i]['j']))
  changes.append({'parent':s,'observed_tasks':len(ids),'old_rank':[tasks[i]['j']+1 for i in before],'author_rank':[tasks[i]['j']+1 for i in after],'best_changed':tasks[before[0]]['j']!=tasks[after[0]]['j'],'values':[records[i] for i in ids]})
 assert np.max(abs(yo-frozen['targets']))<1e-10
 prep_seconds=time.perf_counter()-start
 tick=time.perf_counter();coef=ridge(y);ridge_seconds=time.perf_counter()-tick
 aligned=copy.deepcopy(old);aligned['coef']=coef.tolist()
 # Epigraph QP: t_g >= 2*(z_j-z_best)w + y_best-y_j.
 gs=list(groups.values());G=len(gs);A=[];b=[];weights=np.array([len(ids) for ids in gs],float)
 for g,ids in enumerate(gs):
  best=ids[np.argmax(y[ids])]
  for j in ids:
   row=np.zeros(10+G);row[:10]=-2*(Z[j]-Z[best]);row[10+g]=1;A.append(row);b.append(y[best]-y[j])
 A=np.array(A);b=np.array(b)
 def fun(v):return weights@v[10:]+v[:10]@v[:10]
 def jac(v):return np.r_[2*v[:10],weights]
 initial=np.r_[np.zeros(10),[max(y[ids])-min(y[ids]) for ids in gs]]
 tick=time.perf_counter();res=minimize(fun,initial,jac=jac,constraints={'type':'ineq','fun':lambda v:A@v-b,'jac':lambda v:A},method='SLSQP',options={'ftol':1e-10,'maxiter':2000});spo_seconds=time.perf_counter()-tick
 assert res.success,res.message
 assert min(A@res.x-b)>-1e-7
 spo=copy.deepcopy(old);spo['coef']=[0.]+res.x[:10].tolist()
 # Algebraic upper bound witness, independent finite-action formula.
 rng=np.random.default_rng(721);bound_slack=[]
 for _ in range(1000):
  yy=rng.random(6);ss=rng.normal(size=6);best=int(np.argmax(yy));pick=int(np.argmax(ss))
  loss=max(2*ss-yy)-2*ss[best]+yy[best];regret=max(yy)-yy[pick];assert loss>=regret-1e-12;bound_slack.append(loss-regret)
 train={}
 for name,m in [('old',old),('aligned',aligned),('spo',spo)]:
  ss=Z1@np.array(m['coef']);rows=[]
  for s,ids in groups.items():
   pick=min(ids,key=lambda i:(-ss[i],tasks[i]['j']));rows.append({'parent':s,'choice':tasks[pick]['j']+1,'gain':float(gains[pick]),'regret':float(max(gains[ids])-gains[pick])})
  train[name]=rows
 models={'old':old,'aligned':aligned,'spo':spo}
 dump('排序变化.json',{'changes':changes,'rank_changed':sum(x['old_rank']!=x['author_rank'] for x in changes),'best_changed':sum(x['best_changed'] for x in changes)})
 dump('模型冻结.json',{'models':models,'targets':y.tolist(),'keys':[t['key'] for t in tasks],'train_replay':train,'prep_seconds':prep_seconds,'fit_seconds':{'old':0.,'aligned':ridge_seconds,'spo':spo_seconds},'checks':{'old_reproduced':True,'max_author_cost_error':max(checks),'upper_bound_witnesses':1000,'solver_success':bool(res.success),'solver_iterations':int(res.nit),'constraint_min':float(min(A@res.x-b)),'objective':float(res.fun)},'partial_groups':{s:len(ids) for s,ids in groups.items() if len(ids)<6}})
 paths=[Path(__file__),ROOT/'预先协议.md',WORK/'restoration_service_features_20261108_v1/部署冻结.json',WORK/'restoration_service_features_20261108_v1/特征.json',WORK/'restoration_native_transfer_20261105_v1/统一任务池.json',WORK/'restoration_native_transfer_20261105_v1/父网与特征.json',Path(r.__file__),Path(q.__file__)]
 hashes=dict(q.read(WORK/'nea_frozen_selector_direct_20261108_v1/INPUT_RECEIPT.json')['hashes']);assert all(q.sha(Path(p))==h for p,h in hashes.items());hashes.update({str(p):q.sha(p) for p in paths})
 dump('INPUT_RECEIPT.json',{'hashes':hashes,'python':sys.version,'cpu_threads':r.torch.get_num_threads()})
 print('PREP PASS',sum(x['best_changed'] for x in changes),sum(x['old_rank']!=x['author_rank'] for x in changes),ridge_seconds,spo_seconds,flush=True)
if __name__=='__main__':main()
