"""单线程实际墙钟截止，固定父网和方法，不从历史计时拼接。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def one(ctx,pi,repeat,method):
    R=ctx['routes'];start=time.perf_counter()
    a.torch.manual_seed(0);model,state=a.build('neural',R)
    init_seconds=time.perf_counter()-start
    base=assess(R)
    sel=selection(R,model,state,base,method=='C_single') if method!='direct_multi' else None
    if method=='direct_multi':
        offset=(pi+repeat)%6;order=list(range(6))[offset:]+list(range(6))[:offset]
    elif method=='H_multi':order=sel['ranking']
    else:order=[sel['chosen']]
    tensor=a.tu.get_batch_tensor_from_routes([R],max_route_len=8)
    ready=time.perf_counter()-start;rows=[];visits=[0]*6
    while time.perf_counter()-start<BUDGETS[-1]:
        call=len(rows);j=order[call%len(order)];visit=visits[j];visits[j]+=1
        seed=1340000+pi*10000+repeat*1000+visit
        begin=time.perf_counter()-start;a.torch.manual_seed(seed)
        with a.torch.no_grad():
            out=p.get_neural_variants(model,state.clone(),tensor.clone(),a.torch.tensor([j],dtype=a.torch.long))
        gen_end=time.perf_counter()-start
        rr=[[int(u) for u in line.tolist() if u>=0] for line in out[0]]
        value=assess(rr)
        gain=max(0.,(base['P']-value['P'])/66+(base['O']-value['O'])/840) if value['feasible'] else 0.
        rows.append({'call':call,'j':j,'visit':visit,'seed':seed,'routes':rr,**value,'gain':gain,
                     'started_seconds':begin,'generation_done_seconds':gen_end})
        rows[-1]['finished_seconds']=time.perf_counter()-start
    elapsed=time.perf_counter()-start
    return {'parent':ctx['parent'],'pi':pi,'repeat':repeat,'method':method,'base':base,
            'initialization_seconds':init_seconds,'ready_seconds':ready,'elapsed_seconds':elapsed,
            'selection':sel,'order':order,'rows':rows,
            'checkpoints':{str(b):best_at(rows,b,base) for b in BUDGETS}}

def main():
    assert read(ROOT/'PREFLIGHT.json')['old_parents']==4
    hashes=read(ROOT/'INPUT_RECEIPT.json')['hashes'];assert all(sha(Path(x))==h for x,h in hashes.items())
    frozen=read(ROOT/'部署冻结.json')
    # 统一预热在历史父网执行，不使用本批结果。
    old=read(WORK/'restoration_attachment_validation_20261108_v1/部署冻结.json')['contexts'][0]
    for method in METHODS:one(old,99,0,method)
    dump('RUN_RECEIPT.json',{'hashes':{**hashes,str(ROOT/'PREFLIGHT.json'):sha(ROOT/'PREFLIGHT.json')},
        'warmup_parent':old['parent'],'warmup_methods':METHODS,'budget':BUDGETS[-1]})
    runs=[]
    with (ROOT/'原始运行.jsonl').open('x',encoding='utf-8') as f:
        for pi,ctx in enumerate(frozen['contexts']):
            for repeat in range(4):
                shift=(pi+repeat)%4;order=METHODS[shift:]+METHODS[:shift]
                for method in order:
                    result=one(ctx,pi,repeat,method);runs.append(result)
                    f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            print(ctx['parent'],'completed',flush=True)
    assert len(runs)==64 and all(sha(Path(x))==h for x,h in hashes.items())
    dump('COMPLETE.json',{'runs':runs,'new_fits':0,'new_parents':4,'calls':sum(len(x['rows']) for x in runs)})
    print('done',sum(len(x['rows']) for x in runs),'calls',flush=True)

if __name__=='__main__':main()
