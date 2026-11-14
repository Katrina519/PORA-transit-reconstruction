"""同父网同种子，运行作者完整生成，不训练。"""
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
spec = importlib.util.spec_from_file_location('probe',WORK/'reconstruction_task_probe_20261104_v1/run_probe.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
np, c = p.np, p.c
from src.transit_evaluator import validate_route_set


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(name,value):
    with (ROOT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def metrics(rows):
    return {'count':len(rows),'feasible_count':sum(r['feasible'] for r in rows),
        'positive_count':sum(r['gain']>0 for r in rows),
        **{k:float(np.mean([r[k] for r in rows])) for k in ('gain','accepted_P','accepted_O')}}


def main():
    frozen = read(ROOT/'部署冻结.json')
    hashes = read(ROOT/'INPUT_RECEIPT.json')['hashes']
    hashes.update({str(ROOT/n):p.sha(ROOT/n) for n in ('部署冻结.json','SCREEN.json')})
    assert p.sha(p.a.SRC/'learning/models.py')==p.sha(p.AUTHOR/'learning/models.py')
    assert p.source_function(p.a.SRC/'learning/bee_colony.py')==p.source_function(p.AUTHOR/'learning/bee_colony.py')
    assert all(p.sha(Path(x))==h for x,h in hashes.items())
    dump('采样INPUT_RECEIPT.json',{'hashes':hashes,'generation_calls':96,'new_model_fits':0,
        'all_choices_frozen_before_generation':True,'models_byte_identical':True,'generation_function_AST_equal':True})
    roads,q = np.loadtxt(c.INPUT/'MandlTravelTimes.txt'),np.loadtxt(c.INPUT/'MandlDemand.txt')
    instance = c.BenchmarkInstance(tuple(range(1,16)),
        {(u+1,v+1):float(roads[u,v]) for u,v in np.ndindex(roads.shape) if u!=v and np.isfinite(roads[u,v])},
        {(u+1,v+1):float(q[u,v]) for u,v in np.ndindex(q.shape)},frozenset(range(1,16)),{})
    rows,summaries = [],[]
    with (ROOT/'原始记录.jsonl').open('x',encoding='utf-8') as log:
        for ctx in frozen['contexts']:
            parent,base = ctx['routes'],ctx['base']
            p.a.torch.manual_seed(0)
            model,state = p.a.build('neural',parent)
            tensor = p.a.tu.get_batch_tensor_from_routes([parent],max_route_len=8)
            before = [[r.tolist() for r in batch] for batch in state.routes]
            records=[]
            for repeat,seed in enumerate(ctx['seeds']):
                order=list(range(6));order=order[repeat:]+order[:repeat]
                for j in order:
                    p.a.torch.manual_seed(seed)
                    tick=time.perf_counter()
                    with p.a.torch.no_grad():
                        out=p.get_neural_variants(model,state.clone(),tensor.clone(),p.a.torch.tensor([j],dtype=p.a.torch.long))
                    seconds=time.perf_counter()-tick
                    eval_tick=time.perf_counter()
                    routes=[[int(u) for u in r.tolist() if u>=0] for r in out[0]]
                    assert routes[:-1]==parent[:j]+parent[j+1:]
                    assert [[r.tolist() for r in batch] for batch in state.routes]==before
                    legal=validate_route_set(instance,tuple(tuple(u+1 for u in r) for r in routes),6,2,8) is None
                    P=c.objective(c.unlimited(c.direct(roads,routes)),q)
                    O=float(sum(roads[u,v] for r in routes for u,v in zip(r,r[1:])))
                    feasible=bool(legal and P is not None)
                    gain=max(0.,(base['P']-P)/66.+(base['O']-O)/840.) if feasible else 0.
                    row={'parent':ctx['parent'],'j':j,'seed':seed,'repeat':repeat,'routes':routes,
                        'P':P,'O':O,'feasible':feasible,'gain':gain,'accepted_P':P if gain>0 else base['P'],
                        'accepted_O':O if gain>0 else base['O'],'generation_seconds':seconds,'evaluation_seconds':time.perf_counter()-eval_tick}
                    rows.append(row);records.append(row)
                    log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            tasks=[{'j':j,**metrics([r for r in records if r['j']==j])} for j in range(6)]
            methods={name:metrics([r for r in records if r['j']==j]) for name,j in ctx['choices'].items()}
            methods['uniform']=metrics(records)
            summaries.append({'parent':ctx['parent'],'base':base,'choices':ctx['choices'],'tasks':tasks,'methods':methods})
            print(json.dumps({'parent':ctx['parent'],'methods':methods}),flush=True)
    assert len(rows)==96 and all(p.sha(Path(x))==h for x,h in hashes.items())
    means={name:{k:float(np.mean([s['methods'][name][k] for s in summaries])) for k in ('gain','accepted_P','accepted_O')}
           for name in summaries[0]['methods']}
    comparisons={}
    for reference in ('H_guard','H_rank','C_guard','H_probe','original_guard'):
        differences=[s['methods']['C_rank']['gain']-s['methods'][reference]['gain'] for s in summaries]
        comparisons[reference]={'better':sum(d>1e-12 for d in differences),'worse':sum(d < -1e-12 for d in differences),
            'same':sum(abs(d)<=1e-12 for d in differences),'differences':differences}
    dump('COMPLETE.json',{'summaries':summaries,'means':means,'comparisons':comparisons,'rows':rows,
                          'new_generator_calls':96,'new_model_fits':0,'scope':'four_contiguous_new_parents_fixed_models'})
    print(json.dumps({'means':means,'comparisons':comparisons}),flush=True)


if __name__=='__main__':
    main()
