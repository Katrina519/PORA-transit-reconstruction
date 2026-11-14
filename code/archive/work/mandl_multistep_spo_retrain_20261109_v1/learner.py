"""固定原尺度，以原系数为参照的多步SPO+正则微调。"""
import copy
import numpy as np
from scipy.optimize import minimize,nnls


def fit_multi(tasks,prior):
    assert prior['encoding']=='raw'
    previous=prior['model'];mean=np.asarray(previous['mean']);scale=np.asarray(previous['scale'])
    w0=np.asarray(previous['coef'][1:]);lam=previous['lambda']
    Z=(np.asarray([t['x'] for t in tasks])-mean)/scale
    groups=list(dict.fromkeys(t['state'] for t in tasks))
    ids=[np.array([i for i,t in enumerate(tasks) if t['state']==s]) for s in groups]
    y=np.array([t['gain'] for t in tasks]);maxima=np.array([y[i].max() for i in ids]);p=np.ones(len(ids))/len(ids)
    reference=float(p@maxima)
    if reference<=0:raise ValueError('no informative multi-step feedback')
    weights=p*maxima/reference
    for i,m in zip(ids,maxima):y[i]/=max(m,1e-12)
    d,g=Z.shape[1],len(ids);A=[];rhs=[];initial_loss=[]
    for k,ix in enumerate(ids):
        best=ix[int(np.argmax(y[ix]))];scores=Z[ix]@w0
        initial_loss.append(float(np.max(2*scores-y[ix])-2*(Z[best]@w0)+y[best]))
        for j in ix:
            row=np.zeros(d+g);row[:d]=-2*(Z[j]-Z[best]);row[d+k]=1.
            A.append(row);rhs.append(y[best]-y[j])
    A,rhs=np.array(A),np.array(rhs)
    fun=lambda v:float(weights@v[d:]+lam*((v[:d]-w0)@(v[:d]-w0)))
    jac=lambda v:np.r_[2*lam*(v[:d]-w0),weights]
    result=minimize(fun,np.r_[w0,initial_loss],jac=jac,method='SLSQP',
        constraints={'type':'ineq','fun':lambda v:A@v-rhs,'jac':lambda v:A},options={'ftol':1e-10,'maxiter':2000})
    slack=A@result.x-rhs;active=slack<1e-5
    dual,_=nnls(A[active].T,jac(result.x),maxiter=100000)
    stationarity=float(np.max(np.abs(A[active].T@dual-jac(result.x))))
    complementarity=float(np.max(np.abs(dual*slack[active])))
    assert result.success and slack.min()>-1e-7 and stationarity<1e-4 and complementarity<1e-5
    document=copy.deepcopy(prior);document['model']['coef']=[0.]+result.x[:d].tolist()
    checks={'success':bool(result.success),'iterations':int(result.nit),'objective':float(result.fun),
        'initial_objective':float(fun(np.r_[w0,initial_loss])),'KKT_stationarity':stationarity,
        'KKT_complementarity':complementarity,'constraint_min':float(slack.min()),
        'group_weights':weights.tolist(),'group_spans':maxima.tolist(),'reference_span':reference,
        'states':groups,'coefficient_change_l2':float(np.linalg.norm(result.x[:d]-w0)),
        'lambda':lam,'regularizer':'lambda * squared distance to frozen prior coefficients'}
    return document,checks
