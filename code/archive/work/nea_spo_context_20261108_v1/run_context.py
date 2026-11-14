import sys,time,json,inspect
from pathlib import Path
P=Path(__file__).resolve().parent;W=P.parent;wall=time.perf_counter()
sys.path.insert(0,str(W/'nea_frozen_selector_direct_20261108_v1'));import run as r
q=r.q
old_scores=q.ref.selector.scores
def context_scores(model,X):
 if len(model['indices'])==26:
  X=[list(x)+[x[j]*x[k] for j in [0,1,2,3,4,5,6,9] for k in [7,8]] for x in X]
 return old_scores(model,X)
q.ref.selector.scores=context_scores

def dump(n,v):
 with (P/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

original_colony=r.entry.bee_colony;original_mutant=r.ORIGINAL;original_src=r.src
fail={};cache={};pending=[];feedback_seconds=0.;triggers=0;score_computations=0

def capture(networks,chosen,type1):
 global pending
 pending=[(b,(tuple(tuple(line) for line in r.routes(networks[0,b])),int(chosen[0,b]))) for b in type1.tolist()]

def feedback(bc,ac):
 global feedback_seconds
 tick=time.perf_counter();group={}
 for b,key in pending:group[key]=group.get(key,False) or bool(ac[0,b]<bc[0,b])
 for key,success in group.items():fail[key]=0 if success else fail.get(key,0)+1
 feedback_seconds+=time.perf_counter()-tick

s=inspect.getsource(r.one);start=s.index('        for bee in type1.tolist():');end=s.index('        mask=torch.ones_like',start)
s=s[:start]+'''        global triggers,score_computations
        counts={}
        for bee in type1.tolist():
            R=routes(networks[0,bee]);key=tuple(tuple(line) for line in R)
            if key not in cache:
                base=q.assess(R)
                cache[key]=q.selection(R,model,selection_state,base,False) if base['feasible'] else None
                score_computations+=1
            sel=cache[key]
            if sel is None:
                fallbacks+=1;details.append({'bee':bee,'fallback':'infeasible_parent'});continue
            count=counts.get(key,0);counts[key]=count+1;ranking=sel['ranking'];top=ranking[0]
            triggered=fail.get((key,top),0)>=2
            j=ranking[count%len(ranking)] if triggered else top
            triggers+=int(j!=top);out[0,bee]=j
            details.append({'bee':bee,'old':int(chosen[0,bee]),'new':j,'scores':sel['scores'],'allowed':sel['allowed'],'group_slot':count,'failure_batches':fail.get((key,top),0),'triggered':triggered})
'''+s[end:]
scope=dict(r.__dict__);scope.update(fail=fail,cache=cache,triggers=0,score_computations=0);exec(compile(s,'feedback_selector','exec'),scope);one_feedback=scope['one']

def main():
 global fail,cache,pending,feedback_seconds,triggers,score_computations
 tick=time.perf_counter();receipt=q.read(P/'INPUT_RECEIPT.json');assert all(q.sha(Path(p))==h for p,h in receipt['hashes'].items())
 f=q.read(P/'模型冻结.json');models={'spo':f['old_spo'],'context_spo':f['model']};loading=time.perf_counter()-tick
 tick=time.perf_counter();seen=set();history=list(W.glob('restoration*/部署冻结.json'))+list(W.glob('nea*/运行冻结.json'))+list(W.glob('nea*/部署冻结.json'))
 for path in history:
  for ctx in q.read(path).get('contexts',[]):
   if 'routes' in ctx:seen.add(q.ref.canonical(ctx['routes']))
 for R in q.read(W/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes'].values():seen.add(q.ref.canonical(R))
 instance=q.c.BenchmarkInstance(tuple(range(1,16)),{(u+1,v+1):float(q.ROADS[u,v]) for u,v in q.np.ndindex(q.ROADS.shape) if u!=v and q.np.isfinite(q.ROADS[u,v])},{(u+1,v+1):float(q.Q[u,v]) for u,v in q.np.ndindex(q.Q.shape)},frozenset(range(1,16)),{})
 contexts=[];screen=[]
 for init in range(62265,62315):
  seed=q.ref.stream(init,'initialization').getrandbits(64)
  try:R=[[u-1 for u in line] for line in q.ref.make_initial_routes(instance,6,(2,8),seed)]
  except ValueError as e:
   if not str(e).startswith('initialization_failed:'):raise
   screen.append([init,'initialization_failed']);continue
  key=q.ref.canonical(R)
  if key in seen:screen.append([init,'duplicate']);continue
  if not q.assess(R)['feasible']:screen.append([init,'infeasible']);continue
  seen.add(key);contexts.append({'parent':'mandl1-6-'+str(init),'routes':R,'seed':seed});screen.append([init,'selected'])
  if len(contexts)==3:break
 assert len(contexts)==3;prep=time.perf_counter()-tick
 hashes=dict(receipt['hashes']);hashes.update({str(p):q.sha(p) for p in [Path(__file__),P/'准备与协议.md',P/'模型冻结.json']+history})
 dump('运行冻结.json',{'contexts':contexts,'screen':screen,'methods':['author','spo','context_spo'],'seeds':[212,213,214],'model_sha':q.sha(P/'模型冻结.json'),'script_sha':q.sha(Path(__file__)),'input_hashes':hashes})
 # Frozen previous SPO trajectory witness before running new evaluation parents.
 old=q.read(W/'nea_author_aligned_spo_20261108_v1/运行冻结.json')['contexts'][0];q.MODEL=models['spo'];check=r.one(old,200,'selector',2)
 ref=next(x for x in q.read(W/'nea_author_aligned_spo_20261108_v1/COMPLETE.json')['runs'] if x['parent']==old['parent'] and x['seed']==200 and x['method']=='spo')
 assert check['history'][0]==ref['history'][0][:3] and check['selection_log']==ref['selection_log'][:20]
 dump('PREFLIGHT.json',{'old_spo_prefix_match':True,'diagnostic_seconds':check['seconds']})
 # Add observational feedback only to our method.
 ms=original_src.replace('    # modify type 1 routes','    capture(bee_networks,chosen_route_idxs,type1_idxs)\n    # modify type 1 routes',1)
 gs=dict(original_mutant.__globals__);gs['capture']=capture;exec(compile(ms,'feedback_capture','exec'),gs);captured=gs['get_mutants']
 cs=inspect.getsource(original_colony);anchor='                better_idxs = new_bee_costs < bee_costs';assert cs.count(anchor)==1
 cs=cs.replace(anchor,'                feedback(bee_costs,new_bee_costs)\n'+anchor)
 r.entry.__dict__['feedback']=feedback;exec(compile(cs,'feedback_colony','exec'),r.entry.__dict__);feedback_colony=r.entry.bee_colony
 runs=[];io=0.
 with (P/'原始运行.jsonl').open('x',encoding='utf8') as out:
  for pi,ctx in enumerate(contexts):
   for si,seed in enumerate([212,213,214]):
    methods=['author','spo','context_spo'];shift=(pi+si)%3;methods=methods[shift:]+methods[:shift]
    for method in methods:
     fail={};cache={};pending=[];feedback_seconds=0.;triggers=0;score_computations=0
     scope.update(fail=fail,cache=cache,triggers=0,score_computations=0)
     active=method=='context_spo'
     r.entry.bee_colony=feedback_colony if active else original_colony
     r.ORIGINAL=captured if active else original_mutant;r.src=ms if active else original_src
     q.MODEL=models.get(method,models['spo']);fn=one_feedback if active else r.one
     x=fn(ctx,seed,'author' if method=='author' else 'selector',8);x['method']=method;x['feedback_seconds']=feedback_seconds;x['switches']=scope['triggers'];x['score_computations']=scope['score_computations'];assert x['target_samples']==811
     runs.append(x);tick=time.perf_counter();out.write(json.dumps(x,ensure_ascii=False)+'\n');out.flush();io+=time.perf_counter()-tick
     print(ctx['parent'],seed,method,x['cost'],round(x['seconds'],3),flush=True)
 assert len(runs)==27 and all(q.sha(Path(p))==h for p,h in hashes.items())
 dump('COMPLETE.json',{'runs':runs,'inputs_unchanged':True,'wall_seconds':time.perf_counter()-wall,'loading_seconds':loading,'parent_preparation_seconds':prep,'raw_io_seconds':io,'preflight_seconds':check['seconds'],'new_fit_seconds':f['fit_seconds'],'historical_prepare_seconds':f['preparation_seconds']})
if __name__=='__main__':main()
