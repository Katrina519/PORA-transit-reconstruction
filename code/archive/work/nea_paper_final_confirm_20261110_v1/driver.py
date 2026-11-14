"""论文收尾四组接口，复用既有冻结搜索内核。"""
import sys,inspect,importlib.util,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
IMPORT_STARTED=time.perf_counter()
spec=importlib.util.spec_from_file_location('paper_frozen_search',WORK/'nea_task_collision_repair_20261110_v1/run.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
IMPORT_SECONDS=time.perf_counter()-IMPORT_STARTED

class CountPolicy:
    def __init__(self):self.counts={}
    def choose(self,key,n):
        counts=self.counts.setdefault(key,[0]*n)
        j=min(range(n),key=lambda k:(counts[k],k));counts[j]+=1
        return j

def make(method):
    assert method in ['author','full','count','ridge']
    if method in ['author','full']:return d.one
    source=inspect.getsource(d.one);namespace=dict(d.__dict__)
    if method=='ridge':
        models=d.read(ROOT/'模型冻结.json')['ridge']
        namespace['frozen']=lambda city:models[city]
    else:
        changes={
          'allocator = BatchAllocator(.05, repair=repair)':'allocator = BatchAllocator(.05, repair=False)\n    counter = CountPolicy()',
          "if city == 'mandl':\n                    allocator.observe":'if False:\n                    allocator.observe',
          "scores = cache[key]['scores']":"scores = cache[key]['scores']",
          'cache[key] = score_function(routes, model, selection_state, base, False); counts[0] += 1':"cache[key] = {'scores':[0.]*nr}; counts[0] += 1",
          "j = allocator.choose(key, scores, bee, base['P']/(2*diameter)+base['O']/(nr*diameter))":'j = counter.choose(key,nr)',
          'j = min(eligible, key=lambda k: (-scores[k], k))':'j = counter.choose(key,nr)',
        }
        for a,b in changes.items():assert source.count(a)==1,a;source=source.replace(a,b)
        namespace['CountPolicy']=CountPolicy
    exec(compile(source,str(ROOT/'driver.py')+'::'+method,'exec'),namespace)
    return namespace['one']

def one(ctx,seed,method):
    started=time.perf_counter()
    fn=make(method)
    preparation=time.perf_counter()-started
    result=fn(ctx,seed,'author' if method=='author' else 'ours',repair=False)
    result['kernel_seconds']=result['seconds']
    for field in ['fine_curve','curve']:
        for point in result[field]:point['seconds']+=preparation
    for solution in result['fine_solutions']:solution['first_seconds']+=preparation
    result['wrapper_preparation_seconds']=preparation
    result['seconds']=time.perf_counter()-started
    result['variant']=method
    return result
