"""实际完整重建反馈，不挑样本。"""


def aggregate(rows,n_actions,seeds):
    actual=[(r['j'],r['seed']) for r in rows]
    expected={(j,s) for j in range(n_actions) for s in seeds}
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('missing or duplicate task-seed feedback')
    return [sum(max(0.,r['parent_cost']-r['cost']) if r['feasible'] else 0.
                for r in rows if r['j']==j)/len(seeds) for j in range(n_actions)]
