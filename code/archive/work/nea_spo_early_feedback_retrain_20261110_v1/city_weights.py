"""等城市权重；城内仍按任务最大可见收益分配。"""


def weights(maxima,cities):
    unique=set(cities)
    totals={c:sum(v for v,k in zip(maxima,cities) if k==c) for c in unique}
    if any(t<=0 for t in totals.values()):
        raise ValueError('each included city needs positive gain evidence')
    return [v/(len(unique)*totals[c]) for v,c in zip(maxima,cities)]
