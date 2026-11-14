"""冻结重训评分，只与作者NEA直接成对比较。"""
import sys
import time
import argparse
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e


def collect(ctx,seed,method,iterations=8,old_model=False):
    if method=='author':
        value=e.d.one(ctx,seed,'author',repair=False,iterations=iterations)
    else:
        path=e.ROOT/('原模型冻结.json' if old_model else '归一化模型冻结.json')
        def load(city):
            assert city=='mumford0'
            doc=e.read(path)
            return doc if old_model else doc['document']
        with patch.object(e.d,'frozen',load):
            value=e.d.one(ctx,seed,'ours',repair=False,iterations=iterations)
    value['method_description']='author_NEA' if method=='author' else 'city_normalized_retrained_B'
    value['variant']='retrained_plain_B'
    assert not value['repair']
    return value


def main(phase):
    request=e.read(e.ROOT/'请求冻结.json');folder=e.ROOT/phase;folder.mkdir(exist_ok=False)
    hashes=e.read(e.ROOT/'归一化拟合冻结.json')['input_hashes']
    assert all(e.sha(p)==h for p,h in hashes.items())
    hashes.update({str(p):e.sha(p) for p in e.ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json']})
    if phase=='gate':
        ctx=request['old_contexts'][-1]
        tasks=[{'city':'mumford0','context':ctx,'seed':810,'method':m,'iterations':it,'old_model':old}
            for m,it,old in [('author',2,False),('ours',2,True),('ours',1,False)]]
    else:
        assert e.read(e.ROOT/'gate/COMPLETE.json')['passed']
        contexts=request['old_contexts'] if phase=='regression' else request['fresh_contexts']
        seeds=request['development_seeds'] if phase=='regression' else request['independent_seeds']
        tasks=[]
        for pi,ctx in enumerate(contexts):
            for si,seed in enumerate(seeds):
                methods=['author','ours'] if (pi+si)%2==0 else ['ours','author']
                tasks.extend({'city':'mumford0','context':ctx,'seed':seed,'method':m,'iterations':8,'old_model':False} for m in methods)
    e.dump(folder/'运行冻结.json',{'tasks':tasks,'input_hashes':hashes,'request':request,
        'model_sha256':e.sha(e.ROOT/'归一化模型冻结.json'),'import_seconds':e.IMPORT_SECONDS,
        'threads':e.torch.get_num_threads(),'device':'cpu'})
    runs=[];started=time.perf_counter();io_seconds=0.
    with (folder/'原始运行.jsonl').open('x',encoding='utf-8') as stream:
        for i,t in enumerate(tasks):
            value=collect(t['context'],t['seed'],t['method'],t['iterations'],t['old_model'])
            if phase=='gate':
                if t['method']=='author' or t['old_model']:
                    oldmethod='author' if t['method']=='author' else 'B_spo'
                    old=next(v for v in e.read(e.WORK/'mumford0_ab_single_confirm_20261109_v1/comparison/COMPLETE.json')['runs']
                        if v['parent']==value['parent'] and v['seed']==810 and v['method']==oldmethod)
                    assert value['selection_log']==old['selection_log'][:20]
                    assert value['history'][0]==old['history'][0][:3]
                    assert [p['cost'] for p in value['fine_curve']]==[p['cost'] for p in old['fine_curve'][:11]]
                    event=old['fine_curve'][10];solution=old['fine_solutions'][event['solution_index']]
                    assert value['cost']==event['cost'] and value['routes']==solution['routes']
                else:
                    document=e.read(e.ROOT/'归一化模型冻结.json')['document']
                    e.b.configure('mumford0',document)
                    scores=e.b.learner.scores(document,e.b.features(t['context']['routes']),'mumford0')
                    actual=value['selection_log'][0]
                    assert all(max(abs(a-b) for a,b in zip(v['scores'],scores))<1e-12 for v in actual)
            assert value['target_samples']==11+100*t['iterations']
            runs.append(value);tick=time.perf_counter()
            stream.write(__import__('json').dumps(value,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            io_seconds+=time.perf_counter()-tick
            point=next(p for p in value['fine_curve'][::-1] if p['seconds']<=7.)
            print(phase,i+1,len(tasks),value['parent'],value['seed'],value['method'],
                round(point['cost'],8),round(value['cost'],8),round(value['seconds'],3),flush=True)
    assert all(e.sha(p)==h for p,h in hashes.items())
    e.dump(folder/'COMPLETE.json',{'runs':runs,'passed':True,'input_hashes':hashes,'phase':phase,
        'wall_seconds':time.perf_counter()-started,'raw_io_seconds':io_seconds,'source_models_unchanged':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['gate','regression','independent'])
    main(parser.parse_args().phase)
