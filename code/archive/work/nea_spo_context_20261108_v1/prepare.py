import sys,time,json,copy,collections
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize,nnls
P=Path(__file__).resolve().parent;W=P.parent
sys.path.insert(0,str(W/'nea_frozen_selector_direct_20261108_v1'));import run as r
q=r.q

def dump(n,v):
 with (P/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def features(R):
 ev=SimpleNamespace(direct=q.c.direct,limited=q.c.unlimited,objective=q.c.objective)
 X=[v['x'] for v in q.ref.selector.features(R,q.ROADS,q.Q,ev)]
 D0=q.c.unlimited(q.c.direct(q.ROADS,R));road=q.ROADS.copy()
 for k in range(15):road=np.minimum(road,road[:,k,None]+road[None,k,:])
 for j in range(6):
  dr=q.c.unlimited(q.c.direct(q.ROADS,R[:j]+R[j+1:]));mask=(q.Q>0)&~np.isfinite(dr);h=float(np.sum(q.Q[mask]*(D0[mask]-road[mask]))/q.Q.sum()/33)
  X[j]=X[j]+[h]
 return X

def expand(x):return x+[x[j]*x[k] for j in [0,1,2,3,4,5,6,9] for k in [7,8]]

def main():
 start=time.perf_counter();old=q.read(W/'nea_selector_spo_20261108_v1/模型冻结.json');tasks=q.read(W/'restoration_native_transfer_20261105_v1/统一任务池.json');parents=q.read(W/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes'];lookup={t['key']:t for t in tasks}
 pool=collections.defaultdict(lambda:collections.defaultdict(list));Rs={};sources=collections.Counter()
 def add(R,j,g,source):
  key=tuple(tuple(v) for v in R);Rs[key]=R;pool[key][j].append(float(g));sources[source]+=1
 for key in old['keys']:
  t=lookup[key];R=parents[t['state']];base=q.assess(R);C0=base['P']/66+base['O']/198
  for row in t['rows']:
   m=q.assess(row['routes']);g=max(0,C0-m['P']/66-m['O']/198) if m['feasible'] else 0;add(R,t['j'],g,'original')
 replay=q.read(W/'nea_selector_replay_20261108_v1/COMPLETE.json')['runs']
 for x in replay:
  for t in x['trace']:
   for bee in t['type1']:
    R=t['before'][bee]
    if not q.assess(R)['feasible']:continue
    add(R,t['chosen'][bee],max(0,t['before_cost'][bee]-t['after_cost'][bee]),'replay')
 X=[];y=[];groups=[];records=[]
 for key,items in pool.items():
  if len(items)<2:continue
  fs=features(Rs[key]);ids=[];vals={j:float(np.mean(v)) for j,v in items.items()};scale=max(max(vals.values()),1e-12)
  for j in sorted(items):
   ids.append(len(X));X.append(expand(fs[j]));y.append(vals[j]/scale)
  groups.append(np.array(ids));records.append({'routes':Rs[key],'actions':[{'j':j,'n':len(items[j]),'gain':vals[j]} for j in sorted(items)]})
 X=np.array(X);y=np.array(y);mean=X.mean(0);scale=X.std(0);scale[scale<1e-12]=1;Z=(X-mean)/scale;D=Z.shape[1];G=len(groups);A=[];b=[];weights=np.array([len(ids) for ids in groups],float)
 for g,ids in enumerate(groups):
  best=ids[np.argmax(y[ids])]
  for j in ids:
   row=np.zeros(D+G);row[:D]=-2*(Z[j]-Z[best]);row[D+g]=1;A.append(row);b.append(y[best]-y[j])
 A=np.array(A);b=np.array(b);fun=lambda v:float(weights@v[D:]+v[:D]@v[:D]);jac=lambda v:np.r_[2*v[:D],weights]
 init=np.r_[np.zeros(D),[max(y[ids])-min(y[ids]) for ids in groups]]
 tick=time.perf_counter();res=minimize(fun,init,jac=jac,constraints={'type':'ineq','fun':lambda v:A@v-b,'jac':lambda v:A},method='SLSQP',options={'ftol':1e-9,'maxiter':1000});fit=time.perf_counter()-tick;assert res.success,res.message
 slack=A@res.x-b;active=slack<1e-5;dual,_=nnls(A[active].T,jac(res.x),maxiter=100000);stat=float(np.max(abs(A[active].T@dual-jac(res.x))));assert slack.min()>-1e-6 and stat<1e-3
 model={'indices':list(range(D)),'mean':mean.tolist(),'scale':scale.tolist(),'coef':[0.]+res.x[:D].tolist(),'lambda':1.}
 paths=[W/'nea_selector_replay_20261108_v1/COMPLETE.json',W/'nea_selector_spo_20261108_v1/模型冻结.json',W/'restoration_native_transfer_20261105_v1/统一任务池.json',Path(__file__),P/'准备与协议.md']
 hashes=dict(q.read(W/'nea_selector_spo_20261108_v1/INPUT_RECEIPT.json')['hashes']);assert all(q.sha(Path(p))==h for p,h in hashes.items());hashes.update({str(p):q.sha(p) for p in paths})
 dump('模型冻结.json',{'model':model,'old_spo':old['models']['spo'],'sources':dict(sources),'training_groups':G,'training_tasks':len(X),'features':D,'fit_seconds':fit,'preparation_seconds':time.perf_counter()-start,'KKT_stationarity':stat,'constraint_min':float(slack.min()),'iterations':int(res.nit)})
 dump('训练任务.json',records);dump('INPUT_RECEIPT.json',{'hashes':hashes,'python':sys.version,'torch':r.torch.__version__,'device':'cpu','threads':r.torch.get_num_threads()})
 print('PREP PASS',D,G,len(X),'KKT',stat,'fit',fit)
if __name__=='__main__':main()
