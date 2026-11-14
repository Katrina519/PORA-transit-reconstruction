"""先测试时限和上一批选择一致性，不产生本批训练标签。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def main():
    base={'P':10.,'O':20.}
    def row(t,g,i):return {'finished_seconds':t,'gain':g,'P':9.,'O':20.,'call':i,'j':i,'feasible':True}
    rows=[row(.1,.01,0),row(.35,.02,1),row(.350001,.99,2)]
    assert best_at(rows,.35,base)['best_call']==1
    assert best_at(rows,.09,base)['gain']==0 and best_at(rows,.09,base)['P']==10
    assert best_at([row(.1,0.,0)],.35,base)['best_call'] is None
    hashes=read(ROOT/'INPUT_RECEIPT.json')['hashes'];assert all(sha(Path(x))==h for x,h in hashes.items())
    assert p.sha(p.a.SRC/'learning/models.py')==p.sha(p.AUTHOR/'learning/models.py')
    assert p.source_function(p.a.SRC/'learning/bee_colony.py')==p.source_function(p.AUTHOR/'learning/bee_colony.py')
    old=read(WORK/'restoration_attachment_validation_20261108_v1/部署冻结.json');errors=[]
    for ctx in old['contexts']:
        a.torch.manual_seed(0);model,state=a.build('neural',ctx['routes'])
        weights={k:v.clone() for k,v in model.state_dict().items()};rng=a.torch.random.get_rng_state()
        routes=[[rr.tolist() for rr in batch] for batch in state.routes]
        sel=selection(ctx['routes'],model,state,ctx['base'],True)
        assert a.torch.equal(rng,a.torch.random.get_rng_state())
        assert all(a.torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
        assert routes==[[rr.tolist() for rr in batch] for batch in state.routes]
        assert sel['chosen']==ctx['choices']['C_rank'] and sel['ranking'][0]==ctx['choices']['H_rank']
        assert sel['allowed']==ctx['allowed']
        errors.extend(abs(x-y) for x,y in zip(sel['scores'],ctx['scores']['H']))
        for x,y in zip(sel['probes'],ctx['probes']):
            assert x['route']==y['route'] and x['feasible']==y['feasible']
            for key in ['prob','P','O','raw_gain']:
                assert (x[key] is None)==(y[key] is None)
                if x[key] is not None:errors.append(abs(x[key]-y[key]))
    assert max(errors)<1e-10
    dump('PREFLIGHT.json',{'deadline_tests':3,'old_parents':4,'old_probes':24,'max_error':max(errors),
        'hashes':len(hashes),'models_byte_identical':True,'generation_AST_equal':True,'random_state_and_model_unchanged':True})
    print('preflight passed',max(errors))

if __name__=='__main__':main()
