"""Freeze an initial-route risk policy and screen fresh parents before sampling."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PREVIOUS = WORK/'restoration_first_step_disagreement_20261106_v1'
sys.path.insert(0,str(WORK/'restoration_first_step_validation_20261106_v1'))
import prepare as reference
from src.transit_actions import make_initial_routes
from src.transit_operator_policies_v2 import stream

a, np, c = reference.adapter, reference.np, reference.c
read, sha, selector = reference.read, reference.sha, reference.selector


def dump(name,value):
    with (ROOT/name).open('x',encoding='utf-8') as file:
        json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)


def canonical(routes):
    return tuple(sorted(min(tuple(r),tuple(r[::-1])) for r in routes))


def residual_structure(routes,node_count):
    components = []
    for line in map(set,routes):
        touching = [s for s in components if s & line]
        components = [s for s in components if not s & line]
        components.append(line.union(*touching))
    missing = set(range(node_count)) - set().union(*components)
    return missing,components


def first_pieces(model,state):
    assert not model.force_linking_unlinked and model.serial_halting
    limit = int(state.max_route_len[0])
    if model.max_act_len is not None:
        limit = min(limit,int(model.max_act_len))
    table = state.get_shortest_path_sequences()[0].tolist()
    valid = state.valid_terms_mat[0].tolist()
    pieces = []
    for i,j in np.ndindex((len(table),len(table))):
        line = frozenset(v for v in table[i][j] if v>=0)
        if valid[i][j] and 2<=len(line)<=limit:
            pieces.append(line)
    return pieces


def candidate_count(remaining,node_count,pieces):
    missing,components = residual_structure(remaining,node_count)
    count = sum(missing.issubset(piece) and all(piece & component for component in components)
                for piece in pieces)
    return {'missing':[u+1 for u in sorted(missing)],'components':len(components),'count':count}


def main():
    frozen = read(PREVIOUS/'部署冻结.json')
    hashes = read(PREVIOUS/'采样INPUT_RECEIPT.json')['hashes']
    hashes.update({str(PREVIOUS/path):digest for path,digest in read(PREVIOUS/'交付文件哈希.json').items()})
    hashes.update({str(path):sha(path) for path in [Path(__file__),ROOT/'sample.py',ROOT/'预先协议.md']})
    assert all(sha(Path(path))==digest for path,digest in hashes.items())
    dump('INPUT_RECEIPT.json',{'hashes':hashes,'window':[62216,62415],'new_fits':0})

    roads = np.loadtxt(c.INPUT/'MandlTravelTimes.txt')
    demand = np.loadtxt(c.INPUT/'MandlDemand.txt')
    n = len(roads)
    instance = c.BenchmarkInstance(tuple(range(1,n+1)),
        {(u+1,v+1):float(roads[u,v]) for u,v in np.ndindex(roads.shape)
         if u!=v and np.isfinite(roads[u,v])},
        {(u+1,v+1):float(demand[u,v]) for u,v in np.ndindex(demand.shape)},
        frozenset(range(1,n+1)),{})
    evaluator = SimpleNamespace(direct=c.direct,limited=c.unlimited,objective=c.objective)
    source = reference.ref.p.SOURCE
    old = read(source/'父网与特征.json')['routes']
    seen = {canonical(parent) for parent in old.values()}
    for directory in [reference.OLD,reference.ROOT,PREVIOUS]:
        seen.update(canonical(ctx['routes']) for ctx in read(directory/'部署冻结.json')['contexts'])

    a.torch.manual_seed(0)
    model,state = a.build('neural',frozen['contexts'][0]['routes'])
    pieces = first_pieces(model,state)
    assert len(pieces)>0
    quotas = {'risk':3,'normal':2}
    entries,selected = [],[]
    for init in range(62216,62416):
        seed = stream(init,'initialization').getrandbits(64)
        try:
            parent = [[u-1 for u in r] for r in make_initial_routes(instance,6,(2,8),seed)]
        except ValueError as error:
            if not str(error).startswith('initialization_failed:'):
                raise
            entries.append({'id':init,'status':'initialization_failed'})
            continue
        key = canonical(parent)
        if key in seen:
            entries.append({'id':init,'status':'duplicate'})
            continue
        seen.add(key)
        details = [candidate_count(parent[:j]+parent[j+1:],n,pieces) for j in range(6)]
        eligible = [j for j,d in enumerate(details) if d['count']>0]
        group = 'normal' if 0 in eligible else ('risk' if eligible else 'no_eligible')
        take = group in quotas and quotas[group]>0
        entries.append({'id':init,'status':group,'counts':[d['count'] for d in details],'selected':take})
        if not take:
            continue
        rng = a.torch.random.get_rng_state()
        for j,expected in enumerate(details):
            remaining = parent[:j]+parent[j+1:]
            st = state.clone()
            st.replace_routes(a.tu.get_batch_tensor_from_routes([remaining],max_route_len=8))
            st = model.setup_planning(st)
            assert a.route(st)==()
            assert first_pieces(model,st)==pieces
            halt,initial = a.distribution(model,st)
            assert halt==0.
            realized = candidate_count(remaining,n,[frozenset(line) for _,line,p in initial if p>0.])
            assert realized['count']==expected['count']
        assert a.torch.equal(a.torch.random.get_rng_state(),rng)
        features = [r['x'] for r in selector.features(parent,roads,demand,evaluator)]
        scores = selector.scores(frozen['original_model'],features)
        replacement = 0 if group=='normal' else min(eligible,key=lambda j:(-scores[j],j))
        offset = len(selected)*100
        selected.append({'parent':'mandl1-6-'+str(init),'group':group,'initialization_seed':seed,
            'routes':parent,'base':{'P':features[0][7]*33.,'O':features[0][8]*420.},
            'features':features,'original_scores':scores,'structural_details':details,
            'choices':{'original':selector.choose(scores),'guarded':replacement,'first':0},
            'seeds':[1320600+offset+k for k in range(4)],
            'probe_candidates':len(pieces),'probe_median_seconds':None})
        quotas[group]-=1
        if not any(quotas.values()):
            break

    assert all(sha(Path(path))==digest for path,digest in hashes.items())
    dump('SCREEN.json',{'entries':entries,'remaining_quotas':quotas,
                        'selected':[ctx['parent'] for ctx in selected]})
    dump('部署冻结.json',{'original_model':frozen['original_model'],'train_keys':frozen['train_keys'],
                        'contexts':selected})
    print(len(entries),[(s['parent'],s['group'],s['choices'],[d['count'] for d in s['structural_details']])
                         for s in selected])


if __name__=='__main__':
    main()

