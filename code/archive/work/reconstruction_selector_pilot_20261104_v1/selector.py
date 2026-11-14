"""生成前服务描述与极小回归器；不调用生成器，不读取测试收益。"""
import numpy as np

NAMES = ['old_O', 'old_nodes', 'exclusive_nodes', 'unreachable_demand',
         'finite_delay', 'lost_direct', 'lost_one_transfer', 'parent_P', 'parent_O']
STATIC = [0, 1, 7, 8]


def features(parent, roads, demand, c):
    total = float(demand.sum())
    parent_direct = c.direct(roads, parent)
    parent_D = c.limited(parent_direct)
    parent_P = c.objective(parent_D, demand)
    assert parent_P is not None
    parent_O = sum(roads[u, v] for rr in parent for u, v in zip(rr, rr[1:]))
    def one_transfer(A):
        return np.minimum(A, np.min(A[:, :, None] + A[None, :, :], axis=1) + 5.)
    parent_one = one_transfer(parent_direct)
    records = []
    for j, rr in enumerate(parent):
        remaining = parent[:j] + parent[j + 1:]
        A = c.direct(roads, remaining)
        D = c.limited(A)
        finite = (demand > 0) & np.isfinite(D)
        unreachable = float(demand[(demand > 0) & ~np.isfinite(D)].sum())
        delay = float((demand[finite] * (D[finite] - parent_D[finite])).sum() / total)
        uncovered = set(rr) - set().union(*map(set, remaining))
        old_O = float(sum(roads[u, v] for u, v in zip(rr, rr[1:])))
        lost_direct = float(demand[np.isfinite(parent_direct) & ~np.isfinite(A)].sum()) / total
        lost_one = float(demand[np.isfinite(parent_one) & ~np.isfinite(one_transfer(A))].sum()) / total
        vector = [old_O / 420., len(rr) / 8., len(uncovered) / 15.,
                  unreachable / total, delay / 33., lost_direct, lost_one,
                  parent_P / 33., float(parent_O) / 420.]
        records.append({'j': j, 'x': vector, 'unreachable_demand': unreachable, 'finite_delay': delay})
    assert np.isfinite([r['x'] for r in records]).all()
    return records


def fit(X, y, indices):
    X, y = np.asarray(X, dtype=float)[:, indices], np.asarray(y, dtype=float)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-12] = 1.
    Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    penalty = np.eye(Z.shape[1])
    penalty[0, 0] = 0.
    coef = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
    return {'indices': indices, 'mean': mean.tolist(), 'scale': scale.tolist(),
            'coef': coef.tolist(), 'lambda': 1.}


def scores(model, X):
    X = np.asarray(X, dtype=float)[:, model['indices']]
    Z = np.column_stack([np.ones(len(X)), (X - model['mean']) / model['scale']])
    return (Z @ np.asarray(model['coef'])).tolist()


def choose(values):
    return min(range(len(values)), key=lambda j: (-values[j], j))
