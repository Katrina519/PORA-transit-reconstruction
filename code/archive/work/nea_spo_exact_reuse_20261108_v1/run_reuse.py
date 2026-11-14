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
 f=q.read(W/'nea_spo_context_20261108_v1/运行冻结.json');contexts=f['contexts'];previous=q.read(W/'nea_spo_context_20261108_v1/COMPLETE.json')['runs'];idx={(x['parent'],x['seed'],x['method']):x for x in previous}
 model=q.read(W/'nea_selector_spo_20261108_v1/模型冻结.json')['models']['spo'];q.MODEL=model
 hashes=dict(f['input_hashes']);hashes.update({str(p):q.sha(p) for p in [Path(__file__),P/'运行协议.md',W/'nea_spo_context_20261108_v1/COMPLETE.json']});assert all(q.sha(Path(p))==h for p,h in hashes.items())
 dump('运行冻结.json',{'contexts':contexts,'seeds':[212,213,214],'methods':['author','cached_spo'],'max_iterations':8,'input_hashes':hashes,'script_sha':q.sha(Path(__file__))})
 runs=[]
 with (P/'原始运行.jsonl').open('x',encoding='utf8') as out:
  for pi,ctx in enumerate(contexts):
   for si,seed in enumerate([212,213,214]):
    methods=['author','cached_spo'] if (pi+si)%2==0 else ['cached_spo','author']
    for method in methods:
     cache={};score_calls=cache_hits=0;q.selection=reuse if method=='cached_spo' else original_selection
     x=r.one(ctx,seed,'author' if method=='author' else 'selector',8);x['method']=method;x['score_calls']=score_calls;x['cache_hits']=cache_hits;x['cache_entries']=len(cache)
     y=idx[ctx['parent'],seed,'author' if method=='author' else 'spo']
     for k in ['routes','history','rng','metrics','cost','selection_log']:assert x[k]==y[k],k
     assert x['target_samples']==811
     runs.append(x);out.write(json.dumps(x,ensure_ascii=False)+'\n');out.flush();print(ctx['parent'],seed,method,x['cost'],round(x['seconds'],3),flush=True)
 assert all(q.sha(Path(p))==h for p,h in hashes.items())
 dump('COMPLETE.json',{'runs':runs,'exact_trajectory_match':True,'wall_seconds':time.perf_counter()-wall,'inputs_unchanged':True})
if __name__=='__main__':main()
