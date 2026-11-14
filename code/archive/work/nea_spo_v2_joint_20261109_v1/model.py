"""双网络选线的训练数据尺度与SPO+目标。"""
import numpy as np
from scipy.optimize import minimize, nnls

SPECS = {'mandl': {'routes': 6, 'diameter': 33., 'max': 8, 'n': 15},
         'mumford0': {'routes': 12, 'diameter': 26., 'max': 15, 'n': 30}}


def encode(x, spec, kind):
    out = np.asarray(x, dtype=float).copy()
    assert kind in ['raw', 'scaled']
    if kind == 'scaled':
        factor = np.ones(10)
        factor[[0, 8]] = 420./(spec['routes']*spec['diameter'])
        factor[1] = 8./spec['max']; factor[2] = 15./spec['n']
        factor[[4, 7, 9]] = 33./spec['diameter']
        out *= factor
    return out


def probabilities(cities):
    unique = set(cities)
    return np.array([1./(len(unique)*cities.count(c)) for c in cities])


def loss(scores, gains):
    scores, gains = np.asarray(scores), np.asarray(gains)
    best = int(np.argmax(gains))
    return float(np.max(2*scores-gains)-2*scores[best]+gains[best])


def fit(tasks, encoding, lam):
    X = np.array([encode(t['x'], SPECS[t['city']], encoding) for t in tasks])
    mean, scale = X.mean(0), X.std(0); scale[scale < 1e-12] = 1.
    Z = (X-mean)/scale
    states = list(dict.fromkeys(t['state'] for t in tasks))
    ids = [np.array([i for i,t in enumerate(tasks) if t['state'] == s]) for s in states]
    cities = [tasks[index[0]]['city'] for index in ids]
    p = probabilities(cities)
    y = np.array([t['gain'] for t in tasks], dtype=float)
    maxima = np.array([y[index].max() for index in ids])
    reference = float(p @ maxima)
    if reference <= 0:
        raise ValueError('all task gains are zero')
    weights = p*maxima/reference
    for index, maximum in zip(ids, maxima):
        y[index] /= max(maximum, 1e-12)
    d, g = Z.shape[1], len(ids)
    A, b = [], []
    for k, index in enumerate(ids):
        best = index[int(np.argmax(y[index]))]
        for j in index:
            row = np.zeros(d+g); row[:d] = -2*(Z[j]-Z[best]); row[d+k] = 1.
            A.append(row); b.append(y[best]-y[j])
    A, b = np.array(A), np.array(b)
    fun = lambda v: float(weights @ v[d:] + lam*(v[:d] @ v[:d]))
    jac = lambda v: np.r_[2*lam*v[:d], weights]
    initial = np.r_[np.zeros(d), [float(y[index].max()-y[index].min()) for index in ids]]
    result = minimize(fun, initial, jac=jac, method='SLSQP',
                      constraints={'type': 'ineq', 'fun': lambda v:A@v-b, 'jac': lambda v:A},
                      options={'ftol':1e-10, 'maxiter':2000})
    slack = A @ result.x-b; active = slack < 1e-5
    dual, _ = nnls(A[active].T, jac(result.x), maxiter=100000)
    stationarity = float(np.max(np.abs(A[active].T @ dual-jac(result.x))))
    complementarity = float(np.max(np.abs(dual*slack[active])))
    assert result.success and slack.min() > -1e-7 and stationarity < 1e-4 and complementarity < 1e-5
    model = {'indices': list(range(d)), 'mean':mean.tolist(), 'scale':scale.tolist(),
             'coef':[0.] + result.x[:d].tolist(), 'lambda':lam}
    checks = {'KKT_stationarity':stationarity, 'KKT_complementarity':complementarity,
              'constraint_min':float(slack.min()), 'iterations':int(result.nit), 'success':bool(result.success),
              'group_probabilities':p.tolist(), 'group_cities':cities, 'group_maxima':maxima.tolist(),
              'reference_gain':reference, 'objective':float(result.fun)}
    return {'model':model, 'encoding':encoding}, checks


def scores(document, X, city):
    fit = document['model']
    x = encode(X, SPECS[city], document['encoding'])
    return ((x-np.asarray(fit['mean']))/np.asarray(fit['scale'])) @ np.asarray(fit['coef'][1:]) + fit['coef'][0]


def regret(document, tasks):
    values = [float(scores(document, t['x'], t['city'])) for t in tasks]
    rows = []
    for state in dict.fromkeys(t['state'] for t in tasks):
        ids = [i for i,t in enumerate(tasks) if t['state']==state]
        chosen = min(ids, key=lambda i:(-values[i], tasks[i]['j']))
        rows.append({'state':state, 'city':tasks[chosen]['city'], 'choice':tasks[chosen]['j'],
                     'regret':max(tasks[i]['gain'] for i in ids)-tasks[chosen]['gain']})
    city_means = {c:float(np.mean([v['regret'] for v in rows if v['city']==c])) for c in {v['city'] for v in rows}}
    return {'rows':rows, 'city_means':city_means, 'balanced_mean':float(np.mean(list(city_means.values())))}
