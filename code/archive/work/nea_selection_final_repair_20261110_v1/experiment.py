"""一次冻结修复轮：身份核验、旧案例回归、新种子确认。"""
import sys
import json
import inspect
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
spec = importlib.util.spec_from_file_location('frozen_driver', WORK/'nea_task_collision_repair_20261110_v1/run.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
sys.path.insert(0, str(ROOT))
from policy import MarginalAllocator, RetryOrder


def modified():
    source = inspect.getsource(d.one)
    changes = {
        'allocator = BatchAllocator(.05, repair=repair)': 'allocator = BatchAllocator(.05, repair=False)\n    retry_order = RetryOrder()',
        'pending.clear()': "if city == 'mumford0':\n                    retry_order.observe(pending)\n                pending.clear()",
        'j = min(eligible, key=lambda k: (-scores[k], k))': 'j = retry_order.choose(key, scores)',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = dict(d.__dict__, BatchAllocator=MarginalAllocator, RetryOrder=RetryOrder)
    exec(compile(source, str(ROOT/'experiment.py')+'::modified', 'exec'), namespace)
    return namespace['one']


def main(phase):
    folder = ROOT/phase
    folder.mkdir(exist_ok=False)
    contexts = [c for city in ['mandl','mumford0'] for c in d.contexts(city)]
    seeds = {'mandl': [350,351,352], 'mumford0': [810,811,812]}
    if phase == 'fresh':
        seeds = {'mandl': [1400,1401,1402], 'mumford0': [1400,1401,1402]}
    tasks = []
    for i, ctx in enumerate(contexts):
        for k, seed in enumerate(seeds[ctx['city']]):
            methods = ['author','repair'] if (i+k)%2 == 0 else ['repair','author']
            if phase == 'gate':
                if i not in [0,4] or k != 0:
                    continue
                methods = ['author','original']
            for method in methods:
                tasks.append((ctx,seed,method))
    sources = d.read(WORK/'nea_spo_feedback_allocation_20261108_v1/direct/运行冻结.json')['input_hashes']
    sources.update(d.read(WORK/'mumford0_ab_single_confirm_20261109_v1/输入冻结_接口修正.json')['input_hashes'])
    sources.update({str(p): d.sha(p) for p in ROOT.glob('*.py')})
    sources[str(Path(d.__file__))] = d.sha(d.__file__)
    assert all(d.sha(p) == h for p,h in sources.items())
    d.dump(folder/'运行冻结.json', {'tasks':tasks,'hashes':sources,'budgets':{'mandl':4,'mumford0':7},'iterations':8,'threads':d.torch.get_num_threads()})
    fn = modified()
    runs = []
    with (folder/'原始运行.jsonl').open('x',encoding='utf-8') as stream:
        for ctx,seed,method in tasks:
            run = (fn if method == 'repair' else d.one)(ctx,seed,'author' if method == 'author' else 'ours',repair=False)
            run['variant'] = method
            if phase == 'gate':
                path = 'nea_spo_feedback_allocation_20261108_v1/direct' if ctx['city']=='mandl' else 'mumford0_ab_single_confirm_20261109_v1/comparison'
                label = 'author' if method=='author' else ('feedback_spo' if ctx['city']=='mandl' else 'B_spo')
                ref = next(x for x in d.read(WORK/path/'COMPLETE.json')['runs'] if x['parent']==ctx['parent'] and x['seed']==seed and x['method']==label)
                for field in ['routes','cost','history','rng','selection_log','target_samples','target_calls']:
                    assert run[field] == ref[field], (ctx['city'],method,field)
            runs.append(run)
            stream.write(json.dumps(run,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
            budget = 4 if ctx['city']=='mandl' else 7
            eligible = [x for x in run['fine_curve'] if x['seconds'] <= budget]
            point = eligible[-1]['cost'] if eligible else None
            print(phase,ctx['city'],ctx['parent'],seed,method,point,run['cost'],round(run['seconds'],3),flush=True)
    assert all(d.sha(p)==h for p,h in sources.items())
    d.dump(folder/'COMPLETE.json',{'passed':True,'runs':runs,'hashes':sources})


if __name__=='__main__':
    assert sys.argv[1] in ['gate','regression','fresh']
    if sys.argv[1]!='gate':
        assert d.read(ROOT/'gate/COMPLETE.json')['passed']
    main(sys.argv[1])
