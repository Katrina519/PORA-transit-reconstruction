"""复用已验证双实例接口，只替换我们的线性评分计算。"""
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
import model as learner
spec = importlib.util.spec_from_file_location('frozen_mumford_interface_joint', WORK/'nea_spo_v2_mumford0_20261109_v1/interface.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
q, r, np, torch = base.q, base.r, base.np, base.torch
CITY = 'mandl'


def configure(city, document=None):
    global CITY
    CITY = city; base.configure(city)
    if document is not None:
        def scoring(x):
            return learner.scores(document, x, city).tolist()
        changes = [
            ('range(15)', "range(SPEC['n'])", 1),
            ('range(6)', "range(SPEC['routes'])", 3),
            ('j+1:],15,pieces)', "j+1:],SPEC['n'],pieces)", 1),
            ('ref.selector.scores(MODEL,[x+[H[j]] for j,x in enumerate(X)])',
             'joint_score([x+[H[j]] for j,x in enumerate(X)])', 1)]
        base.BASE_SELECTION = base.compile_function('selection', changes, {**q.__dict__, 'joint_score':scoring})


def features(routes):
    ev = SimpleNamespace(direct=q.c.direct, limited=q.c.unlimited, objective=q.c.objective)
    x = [v['x'] for v in q.ref.selector.features(routes, q.ROADS, q.Q, ev)]
    d0 = q.c.unlimited(q.c.direct(q.ROADS, routes)); road = q.ROADS.copy()
    for k in range(len(road)):
        road = np.minimum(road, road[:,k,None]+road[None,k,:])
    for j in range(len(routes)):
        dr = q.c.unlimited(q.c.direct(q.ROADS, routes[:j]+routes[j+1:]))
        mask = (q.Q>0) & ~np.isfinite(dr)
        x[j].append(float(np.sum(q.Q[mask]*(d0[mask]-road[mask]))/q.Q.sum()/33.))
    return x
