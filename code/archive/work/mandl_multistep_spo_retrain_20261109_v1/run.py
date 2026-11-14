"""原NEA、冻结单步和一次多步微调的同起点直接对照。"""
import sys
import time
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));IMPORT_START=time.perf_counter()
from train import read,sha,dump,b,JOINT
IMPORT_SECONDS=time.perf_counter()-IMPORT_START


def main(phase):
    started=time.perf_counter();folder=ROOT/phase;folder.mkdir(exist_ok=False)
    frozen=read(ROOT/'输入冻结.json');request=read(ROOT/'请求冻结.json');model=read(ROOT/'模型冻结.json')
    assert all(sha(p)==h for p,h in frozen['input_hashes'].items())
    assert sha(ROOT/'训练任务.json')==model['source_sha256']
    hashes=dict(frozen['input_hashes'])
    hashes.update({str(p):sha(p) for p in ROOT.glob('*.py')})
    for name in ['协议.md','输入冻结.json','请求冻结.json','模型冻结.json','训练任务.json']:
        hashes[str(ROOT/name)]=sha(ROOT/name)
    if phase=='comparison':
        gate=read(ROOT/'gate/COMPLETE.json');assert gate['passed']
        assert all(sha(p)==h for p,h in gate['input_hashes'].items())
    dump(folder/'运行冻结.json',{'request':request,'input_hashes':hashes,'phase':phase,'threads':b.torch.get_num_threads()})
    audit=b.q.load('frozen_independent_check',JOINT/'audit.py');runs=[];configuration=[];io_seconds=0.
    with (folder/'原始运行.jsonl').open('x',encoding='utf-8') as stream:
        def collect(ctx,seed,method,iterations):
            nonlocal io_seconds
            document=model['document'] if method=='multi_spo' else frozen['prior']
            tick=time.perf_counter();b.configure('mandl',document);one=b.base.make_one()
            prep=time.perf_counter()-tick;configuration.append({'parent':ctx['parent'],'seed':seed,'method':method,'seconds':prep})
            v=b.base.run_one(one,ctx,seed,'author' if method=='author' else 'v2_spo',iterations)
            v.update(method=method,city='mandl',configuration_seconds=prep)
            assert v['target_samples']==11+100*iterations and v['target_calls']==2+10*iterations
            assert len(v['fine_curve'])==1+5*iterations
            audit.check_choices(v,6);runs.append(v);tick=time.perf_counter()
            stream.write(__import__('json').dumps(v,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            io_seconds+=time.perf_counter()-tick
            print(phase,ctx['parent'],seed,method,round(v['cost'],8),round(v['seconds'],3),flush=True)
            return v
        if phase=='gate':
            oldreq=read(JOINT/'请求冻结.json');ctx=next(v for v in oldreq['contexts'] if v['city']=='mandl');seed=532
            oldruns=read(JOINT/'confirmation/COMPLETE.json')['runs']
            fields=['routes','history','rng','metrics','cost','selection_log','target_samples','target_calls','score_calls','cache_hits']
            for method in ['author','single_spo']:
                v=collect(ctx,seed,method,8)
                ref=next(r for r in oldruns if r['parent']==ctx['parent'] and r['seed']==seed and r['method']==('author' if method=='author' else 'joint_spo'))
                for key in fields:assert v[key]==ref[key],(method,key)
            v=collect(ctx,789,'multi_spo',1)
            predicted=b.learner.scores(model['document'],b.features(ctx['routes']),'mandl').tolist()
            actual=v['selection_log'][0][0]['scores'];assert max(abs(x-y) for x,y in zip(predicted,actual))<1e-12
            for s in v['fine_solutions']:
                _,_,c=audit.evaluate('mandl',s['routes']);assert abs(c-s['cost'])<1e-6
        else:
            for pi,ctx in enumerate(request['contexts']):
                for si,seed in enumerate(request['seeds']):
                    offset=(pi+si)%3;methods=request['methods'][offset:]+request['methods'][:offset]
                    for method in methods:collect(ctx,seed,method,request['max_iterations'])
    assert all(sha(p)==h for p,h in hashes.items())
    dump(folder/'COMPLETE.json',{'runs':runs,'input_hashes':hashes,'passed':True,'phase':phase,
        'wall_seconds':time.perf_counter()-started,'import_seconds':IMPORT_SECONDS,
        'method_configuration':configuration,'raw_io_seconds':io_seconds,'no_parameter_tuning':True,
        'author_and_generator_unchanged':True,'Mumford0_unchanged':True})
    print(phase,'COMPLETE',len(runs),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['gate','comparison'],required=True)
    main(parser.parse_args().phase)
