"""原稿与近期小测的输入和换乘口径对齐，只重评旧站序。"""
import os,sys,json,math,csv,heapq,hashlib,time
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
OLDREPO=Path('C:/Users/m/.codex/worktrees/public-od-selector-v2/bus-network-agent');sys.path.insert(0,str(OLDREPO))
from src.transit_benchmark_data import BenchmarkInstance
from src.transit_evaluator import shortest_service_times,evaluate_routes
CACHE=Path('C:/Users/m/.codex/worktrees/public-od-benchmark/bus-network-agent/data/raw/transit_benchmark_cache/Mandl/Mandl1')
INPUT=WORK/'paper_code_learning_20261030_v1/holliday/inputs'
EARLY=WORK/'开源公交复现/ea_nea_mandl_smoke_v1/inputs/cache_adapter'
def dump(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def readcsv(p):return list(csv.DictReader(p.read_text(encoding='utf-8-sig').splitlines()))
def can(r):return min(tuple(r),tuple(r)[::-1])
def direct(roads,routes):
 A=np.full_like(roads,np.inf);np.fill_diagonal(A,0.)
 for rr in routes:
  cs=[0.]
  for u,v in zip(rr,rr[1:]):cs.append(cs[-1]+roads[u,v])
  for i,u in enumerate(rr):
   for j,v in enumerate(rr):A[u,v]=min(A[u,v],abs(cs[i]-cs[j]))
 return A
def limited(A):
 best=A.copy();layer=A.copy()
 for k in range(2):
  layer=np.min(layer[:,:,None]+A[None,:,:],axis=1)+5.;best=np.minimum(best,layer)
 np.fill_diagonal(best,0.);return best
def unlimited(A):
 D=A+5.;np.fill_diagonal(D,0.)
 for k in range(len(A)):D=np.minimum(D,D[:,k,None]+D[None,k,:])
 off=~np.eye(len(A),dtype=bool);D[off]-=5.;return D
def layer_dijkstra(roads,routes):
 n=len(roads);at=[[] for _ in range(n)];adj={}
 for ri,rr in enumerate(routes):
  for u in rr:at[u].append(ri);adj[ri,u]=[]
  for u,v in zip(rr,rr[1:]):adj[ri,u].append((v,roads[u,v]));adj[ri,v].append((u,roads[v,u]))
 D=np.full_like(roads,np.inf);np.fill_diagonal(D,0.)
 for origin in range(n):
  best={};queue=[]
  for ri in at[origin]:best[origin,ri,0]=0.;heapq.heappush(queue,(0.,origin,ri,0))
  while queue:
   t,u,ri,tr=heapq.heappop(queue)
   if t!=best[u,ri,tr]:continue
   D[origin,u]=min(D[origin,u],t)
   nexts=[(t+w,v,ri,tr) for v,w in adj[ri,u]]
   if tr<2:nexts += [(t+5.,u,rj,tr+1) for rj in at[u] if rj!=ri]
   for nt,v,rj,ntr in nexts:
    if nt<best.get((v,rj,ntr),math.inf):best[v,rj,ntr]=nt;heapq.heappush(queue,(nt,v,rj,ntr))
 return D
def compare(A,B):
 assert np.array_equal(np.isfinite(A),np.isfinite(B))
 return float(np.max(np.abs(A[np.isfinite(A)]-B[np.isfinite(B)])))
def objective(D,q):
 mask=q>0
 return float((D[mask]*q[mask]).sum()/q[mask].sum()) if np.isfinite(D[mask]).all() else None
def dom(a,b):return a['P2']<=b['P2']+1e-8 and a['O']<=b['O']+1e-8 and (a['P2']<b['P2']-1e-8 or a['O']<b['O']-1e-8)
def main():
 assert not (ROOT/'COMPLETE.json').exists();tick=time.perf_counter()
 sources=[WORK/'service_logit_correction_20261103_v1/COMPLETE.json',WORK/'service_budget_mix_20261103_v1/COMPLETE.json',WORK/'service_recovery_control_20261103_v1/COMPLETE.json',WORK/'service_stop_diagnosis_20261103_v1/COMPLETE.json']
 ctxp=WORK/'idea4_applicability_20261101_v1/contexts.json';contexts=json.loads(ctxp.read_text(encoding='utf-8'))
 papersrc=WORK/'开源公交复现/idea1_exchange_final_factorial_v1/INPUTS.json'
 files=[Path(__file__),ROOT/'核对范围.md',ctxp,papersrc,OLDREPO/'src/transit_evaluator.py',OLDREPO/'src/transit_benchmark_data.py']+sources+list(CACHE.glob('*.txt'))+[INPUT/n for n in ('MandlTravelTimes.txt','MandlDemand.txt','MandlCoords.txt')]+[EARLY/n for n in ('MandlTravelTimes.txt','MandlDemand.txt','MandlCoords.txt')]
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};dump('INPUT_RECEIPT.json',{'hashes':hashes,'no_search':True,'no_neural_model':True})
 links=readcsv(CACHE/'mandl1_links.txt');ods=readcsv(CACHE/'mandl1_demand.txt');nodes=readcsv(CACHE/'mandl1_nodes.txt');n=len(nodes)
 roads=np.full((n,n),np.inf);np.fill_diagonal(roads,0.);q=np.zeros((n,n))
 for r in links:roads[int(r['from'])-1,int(r['to'])-1]=float(r['travel_time'])
 for r in ods:q[int(r['from'])-1,int(r['to'])-1]=float(r['demand'])
 recent_roads=np.loadtxt(INPUT/'MandlTravelTimes.txt');recent_q=np.loadtxt(INPUT/'MandlDemand.txt')
 data={'nodes':n,'directed_links':len(links),'undirected_links':len(links)//2,'positive_od':int((q>0).sum()),'demand_sum':float(q.sum()),'recent_road_equal':np.array_equal(roads,recent_roads),'recent_OD_equal':np.array_equal(q,recent_q),'early_road_equal':np.array_equal(roads,np.loadtxt(EARLY/'MandlTravelTimes.txt')),'early_OD_equal':np.array_equal(q,np.loadtxt(EARLY/'MandlDemand.txt'))}
 assert data['recent_road_equal'] and data['recent_OD_equal']
 paper=json.loads(papersrc.read_text(encoding='utf-8'));spec=paper['config']['instances']['mandl1']['files'];data['paper_source_hashes_match']={k:hashes[str(CACHE/Path(v['path']).name)]==v['sha256'] for k,v in spec.items()};assert all(data['paper_source_hashes_match'].values())
 data['coordinate_files_byte_equal']=(INPUT/'MandlCoords.txt').read_bytes()==(EARLY/'MandlCoords.txt').read_bytes()
 data['coordinate_note']='recent author-format coordinates differ from early CSV-lon/lat adapter; no verified geometry equivalence'
 dump('DATA_ALIGNMENT.json',data)
 instance=BenchmarkInstance(tuple(range(1,n+1)),{(int(r['from']),int(r['to'])):float(r['travel_time']) for r in links},{(int(r['from']),int(r['to'])):float(r['demand']) for r in ods},frozenset(range(1,n+1)),{})
 # 边界见证，不是算法效果实验。
 tr=np.full((5,5),np.inf);np.fill_diagonal(tr,0.)
 for i in range(4):tr[i,i+1]=tr[i+1,i]=1.
 toy_routes=[[0,1],[1,2],[2,3],[3,4]];toy2=limited(direct(tr,toy_routes));toyinf=unlimited(direct(tr,toy_routes));assert toy2[0,3]==13. and not np.isfinite(toy2[0,4]) and toyinf[0,4]==19.;assert compare(toy2,layer_dijkstra(tr,toy_routes))==0
 candidates={name:{} for name in ('structural_1','structural_2')}
 def add(name,route,origin):
  rr=can(route)
  if len(rr)>=2:candidates[name].setdefault(rr,set()).add(origin)
 for path in sources:
  x=json.loads(path.read_text(encoding='utf-8'))
  if 'runs' in x:
   for run in x['runs']:
    name=run['parent']
    if name not in candidates:continue
    for rr in run.get('common',[])+run.get('records',[])+run.get('recovered',[]):add(name,rr['route'],path.parent.name)
    for d in run.get('details',[]):
     for rr in d.get('next_routes',[]):add(name,rr,path.parent.name+'/scored')
  else:
   for rr in x.get('records',[]):add(rr['parent'],rr['route'],path.parent.name)
 rows=[];maxerr=0.;oldmaxerr=0.;evalPmaxerr=0.
 for name,collection in candidates.items():
  remaining=contexts[name]['parent'][1:]
  for rr,origins in sorted(collection.items()):
   routes=remaining+[list(rr)];assert all(2<=len(t)<=8 and len(t)==len(set(t)) and all(np.isfinite(roads[u,v]) and roads[u,v]>0 for u,v in zip(t,t[1:])) for t in routes)
   A=direct(roads,routes);D2=limited(A);Du=unlimited(A);ind=layer_dijkstra(roads,routes);maxerr=max(maxerr,compare(D2,ind));assert maxerr<1e-8
   onebased=tuple(tuple(u+1 for u in t) for t in routes);old=shortest_service_times(instance,onebased,5.,2)
   for (u,v),demand in instance.demand.items():
    if demand>0:
     expected=D2[u-1,v-1];actual=old.get((u,v),math.inf);assert np.isfinite(expected)==math.isfinite(actual)
     if math.isfinite(actual):oldmaxerr=max(oldmaxerr,abs(expected-actual))
   p2=objective(D2,q);pu=objective(Du,q);O=float(sum(roads[u,v] for t in routes for u,v in zip(t,t[1:])))
   original_eval=evaluate_routes(instance,onebased,6,2,8,5.,2);assert original_eval.feasible==(p2 is not None)
   if p2 is not None:evalPmaxerr=max(evalPmaxerr,abs(p2-original_eval.passenger_minutes));assert abs(O-original_eval.operator_minutes)<1e-8
   finite=np.isfinite(D2)&np.isfinite(Du);diff=D2[finite]-Du[finite];assert (diff>=-1e-8).all()
   positive=q>0;became_unreachable=int((positive & np.isfinite(Du) & ~np.isfinite(D2)).sum());increased=int((positive & finite & (np.where(finite,D2,0)-np.where(finite,Du,0)>1e-8)).sum())
   routeO=float(sum(roads[u,v] for u,v in zip(rr,rr[1:])));rows.append({'parent':name,'route':list(rr),'sources':sorted(origins),'P2':p2,'P_unlimited':pu,'P_delta_seconds':60*(p2-pu) if p2 is not None and pu is not None else None,'O':O,'replacement_O':routeO,'finite_unlimited':pu is not None,'finite_M2':p2 is not None,'positive_OD_newly_unreachable':became_unreachable,'positive_OD_increased':increased,'within_development_limits':O<=70+1e-8 and routeO<=35+1e-8})
 summary=[]
 for name in candidates:
  rr=[r for r in rows if r['parent']==name];good=[r for r in rr if r['P2'] is not None];front=[r for r in good if not any(dom(t,r) for t in good)]
  summary.append({'parent':name,'saved_unique_routes':len(rr),'unlimited_feasible':sum(r['finite_unlimited'] for r in rr),'M2_feasible':len(good),'became_infeasible':sum(r['finite_unlimited'] and not r['finite_M2'] for r in rr),'finite_P_changed':sum(r['P_delta_seconds'] is not None and abs(r['P_delta_seconds'])>1e-7 for r in rr),'max_finite_P_increase_seconds':max((r['P_delta_seconds'] for r in rr if r['P_delta_seconds'] is not None),default=0.),'finite_front_M2':front})
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 out={'data':data,'rows':rows,'summary':summary,'checks':{'layer_graph_max_distance_error':maxerr,'original_paper_OD_max_error':oldmaxerr,'original_paper_P_max_error':evalPmaxerr,'toy_three_rides_13_minutes':True,'toy_four_rides_disallowed_M2':True},'inputs_unchanged':True,'elapsed_seconds':time.perf_counter()-tick,'scope':'saved_route_reevaluation_no_search_no_model_not_new_algorithm_effect'};dump('COMPLETE.json',out)
 print(json.dumps({'data':data,'summary':[{k:v for k,v in r.items() if k!='finite_front_M2'} for r in summary],'checks':out['checks'],'elapsed':out['elapsed_seconds']},ensure_ascii=False))
if __name__=='__main__':main()
