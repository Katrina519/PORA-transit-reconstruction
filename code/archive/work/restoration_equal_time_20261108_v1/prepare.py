"""先冻结父网、方法与文件身份；不读取新生成结果。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def main():
    old=WORK/'restoration_attachment_validation_20261108_v1'
    for name,h in read(old/'交付文件哈希.json').items():assert sha(old/name)==h
    paths=[x for x in ROOT.iterdir() if x.is_file()]
    prior=read(old/'采样INPUT_RECEIPT.json')['hashes']
    assert all(sha(Path(path))==h for path,h in prior.items())
    historical=list(WORK.glob('restoration*/部署冻结.json'))
    known=list(read(WORK/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes'].values())
    for path in historical:
        f=read(path);known.extend(x['routes'] for x in f.get('contexts',[]) if 'routes' in x)
        if isinstance(f.get('routes'),list):known.append(f['routes'])
    seen={ref.canonical(r) for r in known}
    instance=c.BenchmarkInstance(tuple(range(1,16)),
        {(u+1,v+1):float(ROADS[u,v]) for u,v in np.ndindex(ROADS.shape) if u!=v and np.isfinite(ROADS[u,v])},
        {(u+1,v+1):float(Q[u,v]) for u,v in np.ndindex(Q.shape)},frozenset(range(1,16)),{})
    contexts=[];screen=[]
    for init in range(62245,62265):
        seed=ref.stream(init,'initialization').getrandbits(64)
        try:R=[[u-1 for u in rr] for rr in ref.make_initial_routes(instance,6,(2,8),seed)]
        except ValueError as e:
            if not str(e).startswith('initialization_failed:'):raise
            screen.append({'id':init,'status':'initialization_failed'});continue
        key=ref.canonical(R)
        if key in seen:screen.append({'id':init,'status':'duplicate'});continue
        seen.add(key);base=assess(R);assert base['feasible']
        contexts.append({'parent':'mandl1-6-'+str(init),'initialization_seed':seed,'routes':R,'base':base})
        screen.append({'id':init,'status':'selected'})
        if len(contexts)==4:break
    assert len(contexts)==4
    dump('部署冻结.json',{'models':{'H':MODEL},'contexts':contexts,'methods':METHODS,'budgets':BUDGETS,'repeats':4,'screen':screen})
    paths+=historical+[old/x for x in read(old/'交付文件哈希.json')]+[ROOT/'部署冻结.json']
    prior.update({str(path):sha(path) for path in paths})
    dump('INPUT_RECEIPT.json',{'hashes':prior,'runtime':{'python':__import__('sys').version,'torch':a.torch.__version__,
        'numpy':np.__version__,'device':'cpu','threads':a.torch.get_num_threads()},'new_fits':0})
    print([x['parent'] for x in contexts])

if __name__=='__main__':main()
