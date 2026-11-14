"""冻结联合SPO+之后的身份门禁与双图新起点直接对照。"""
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(ROOT));IMPORT_START=time.perf_counter()
import bridge as b
IMPORT_SECONDS=time.perf_counter()-IMPORT_START


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main(phase):
    started=time.perf_counter();folder=ROOT/phase;folder.mkdir(exist_ok=False)
    source=read(ROOT/'输入冻结.json');request=read(ROOT/'请求冻结.json');model=read(ROOT/'模型冻结.json')
    assert all(sha(p)==h for p,h in source['input_hashes'].items())
    assert sha(ROOT/'训练任务.json')==model['training_source_sha256']
    hashes=dict(source['input_hashes'])
    paths=list(ROOT.glob('*.py'))+[ROOT/n for n in ['协议.md','模型冻结.json','输入冻结.json','请求冻结.json','训练任务.json','反馈完成.json']]
    hashes.update({str(p):sha(p) for p in paths})
    if phase=='confirmation':
        gate=read(ROOT/'gate/COMPLETE.json');assert gate['passed']
        assert all(sha(p)==h for p,h in gate['input_hashes'].items())
    dump(folder/'运行冻结.json',{'input_hashes':hashes,'request':request,'phase':phase,'threads':b.torch.get_num_threads()})
    audit=b.q.load('joint_independent_audit',ROOT/'audit.py')
    runs=[];io_seconds=0.
    with (folder/'原始运行.jsonl').open('x',encoding='utf-8') as stream:
        def collect(ctx,seed,method,one,it):
            nonlocal io_seconds
            call='v2_spo' if method in ['joint_spo','v2_spo'] else method
            v=b.base.run_one(one,ctx,seed,call,it);v.update(method=method,city=ctx['city'])
            assert v['target_samples']==11+100*it and len(v['fine_curve'])==1+5*it
            runs.append(v);tick=time.perf_counter();stream.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            io_seconds+=time.perf_counter()-tick
            print(phase,ctx['parent'],seed,method,round(v['cost'],8),round(v['seconds'],3),flush=True)
            return v
        if phase=='gate':
            sources={'mandl':WORK/'nea_spo_v2_independent_confirm_20261108_v1',
                     'mumford0':WORK/'nea_spo_v2_mumford0_20261109_v1'}
            for city,oldfolder in sources.items():
                oldreq=read(oldfolder/'请求冻结.json');oldruns=read(oldfolder/'comparison/COMPLETE.json')['runs']
                ctx=dict(oldreq['contexts'][0],city=city);seed=oldreq['seeds'][0]
                b.configure(city);one=b.base.make_one()
                for method in ['author','v2_spo']:
                    v=collect(ctx,seed,method,one,8)
                    old=next(x for x in oldruns if x['parent']==ctx['parent'] and x['seed']==seed and x['method']==method)
                    for key in ['routes','history','rng','metrics','cost','selection_log','target_samples','target_calls','score_calls','cache_hits']:
                        assert v[key]==old[key],(city,method,key)
                b.configure(city,model['document']);one=b.base.make_one()
                v=collect(ctx,529,'joint_spo',one,1)
                assert v['score_calls']>0 and v['fallbacks']==0
                predicted=b.learner.scores(model['document'],b.features(ctx['routes']),city)
                actual=v['selection_log'][0][0]['scores']
                assert max(abs(a-c) for a,c in zip(predicted,actual))<1e-12
                audit.check_choices(v,6 if city=='mandl' else 12)
                for solution in v['fine_solutions']:
                    _,_,cost=audit.evaluate(city,solution['routes']);assert abs(cost-solution['cost'])<1e-6
        else:
            for city in ['mandl','mumford0']:
                b.configure(city,model['document']);one=b.base.make_one()
                for pi,ctx in enumerate(c for c in request['contexts'] if c['city']==city):
                    for si,seed in enumerate(request['seeds']):
                        methods=request['methods'] if (pi+si)%2==0 else request['methods'][::-1]
                        for method in methods:collect(ctx,seed,method,one,8)
    assert all(sha(p)==h for p,h in hashes.items())
    dump(folder/'COMPLETE.json',{'runs':runs,'passed':True,'input_hashes':hashes,'phase':phase,
         'import_seconds':IMPORT_SECONDS,'wall_seconds':time.perf_counter()-started,'raw_io_seconds':io_seconds,
         'source_replacements':b.base.REUSE,'model_unchanged':True,'generator_and_author_unchanged':True})
    print(phase,'COMPLETE',len(runs),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['gate','confirmation'],required=True)
    main(parser.parse_args().phase)
