import sys,time,json
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parent;wall=time.perf_counter()
sys.path.insert(0,str(W/'nea_frozen_selector_direct_20261108_v1'));import run as r
q=r.q;original_selection=q.selection;cache={};score_calls=0;cache_hits=0

def reuse(R,model,state,base,probe):
 global score_calls,cache_hits
 assert not probe
 key=tuple(tuple(line) for line in R)
 if key not in cache:cache[key]=original_selection(R,model,state,base,probe);score_calls+=1
 else:cache_hits+=1
 return cache[key]

def dump(n,v):
 with (P/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 global cache,score_calls,cache_hits
 tick=time.perf_counter();prior=W/'nea_spo_exact_reuse_20261108_v1'
 for n,h in q.read(prior/'交付文件哈希.json').items():assert q.sha(prior/n)==h,n
 f=q.read(prior/'运行冻结.json');seen=set();history=list(W.glob('restoration*/部署冻结.json'))+list(W.glob('nea*/运行冻结.json'))+list(W.glob('nea*/部署冻结.json'))
 for path in history:
  for ctx in q.read(path).get('contexts',[]):
   if 'routes' in ctx:seen.add(q.ref.canonical(ctx['routes']))
 for R in q.read(W/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes'].values():seen.add(q.ref.canonical(R))
 for x in q.read(W/'nea_selector_replay_20261108_v1/COMPLETE.json')['runs']:
  for frame in x['trace']:
   for R in frame['before']+frame['after']:seen.add(q.ref.canonical(R))
 for path in W.glob('nea*/COMPLETE.json'):
  for x in q.read(path).get('runs',[]):
   if 'routes' in x:seen.add(q.ref.canonical(x['routes']))
 known=len(seen)
 instance=q.c.BenchmarkInstance(tuple(range(1,16)),{(u+1,v+1):float(q.ROADS[u,v]) for u,v in q.np.ndindex(q.ROADS.shape) if u!=v and q.np.isfinite(q.ROADS[u,v])},{(u+1,v+1):float(q.Q[u,v]) for u,v in q.np.ndindex(q.Q.shape)},frozenset(range(1,16)),{})
 contexts=[];screen=[]
 for init in range(62268,62318):
  seed=q.ref.stream(init,'initialization').getrandbits(64)
  try:R=[[u-1 for u in line] for line in q.ref.make_initial_routes(instance,6,(2,8),seed)]
  except ValueError as e:
   if not str(e).startswith('initialization_failed:'):raise
   screen.append([init,'initialization_failed']);continue
  k=q.ref.canonical(R)
  if k in seen:screen.append([init,'duplicate']);continue
  if not q.assess(R)['feasible']:screen.append([init,'infeasible']);continue
  seen.add(k);contexts.append({'parent':'mandl1-6-'+str(init),'routes':R,'seed':seed});screen.append([init,'selected'])
  if len(contexts)==3:break
 assert len(contexts)==3;preparation=time.perf_counter()-tick
 model=q.read(W/'nea_selector_spo_20261108_v1/模型冻结.json')['models']['spo'];q.MODEL=model
 hashes=dict(f['input_hashes']);hashes.update({str(p):q.sha(p) for p in [Path(__file__),P/'运行协议.md',prior/'运行冻结.json']});assert all(q.sha(Path(p))==h for p,h in hashes.items())
 dump('运行冻结.json',{'contexts':contexts,'seeds':[215,216,217],'methods':['author','cached_spo'],'max_iterations':8,'screen':screen,'known_networks':known,'input_hashes':hashes,'script_sha':q.sha(Path(__file__))})
 runs=[]
 with (P/'原始运行.jsonl').open('x',encoding='utf8') as out:
  for pi,ctx in enumerate(contexts):
   for si,seed in enumerate([215,216,217]):
    methods=['author','cached_spo'] if (pi+si)%2==0 else ['cached_spo','author']
    for method in methods:
     cache={};score_calls=cache_hits=0;q.selection=reuse if method=='cached_spo' else original_selection
     x=r.one(ctx,seed,'author' if method=='author' else 'selector',8);x['method']=method;x['score_calls']=score_calls;x['cache_hits']=cache_hits;x['cache_entries']=len(cache)
     assert x['target_samples']==811
     runs.append(x);out.write(json.dumps(x,ensure_ascii=False)+'\n');out.flush();print(ctx['parent'],seed,method,x['cost'],round(x['seconds'],3),flush=True)
 assert all(q.sha(Path(p))==h for p,h in hashes.items())
 dump('COMPLETE.json',{'runs':runs,'model_unchanged':True,'preparation_seconds':preparation,'wall_seconds':time.perf_counter()-wall,'inputs_unchanged':True})
if __name__=='__main__':main()
