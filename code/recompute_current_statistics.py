"""Reproduce current paired statistics without running optimization."""
from pathlib import Path
from statistics import mean,stdev
import csv,json,math
from statistics_core import signed_rank,holm

DATA=Path(__file__).resolve().parents[1]/'data/manuscript_24pairs'
expected=json.loads((DATA/'statistics24.json').read_text(encoding='utf-8'))
for role in ['main','ablation']:
    with (DATA/(role+'24.csv')).open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    actual=[]
    for ep in ['8round','120s']:
        paired={}
        for r in records:
            if r['endpoint']==ep:paired.setdefault((r['parent'],r['seed']),{})[r['method']]=float(r['cost'])
        assert len(paired)==24
        for method in (['author'] if role=='main' else ['author','count','ridge','nogain']):
            a=[r[method] for _,r in sorted(paired.items())];b=[r['full'] for _,r in sorted(paired.items())]
            d=[x-y for x,y in zip(a,b)]
            actual.append(dict(endpoint=ep,comparator=method,n=24,comparator_mean=mean(a),comparator_sd=stdev(a),
                full_mean=mean(b),full_sd=stdev(b),improvement_percent=100*mean(d)/mean(a),mean_gain=mean(d),
                wins=sum(x>1e-6 for x in d),ties=sum(abs(x)<=1e-6 for x in d),losses=sum(x< -1e-6 for x in d),**signed_rank(d)))
        family=[r for r in actual if r['endpoint']==ep and r['comparator']!='author']
        for r,p in zip(family,holm([r['p_one_sided'] for r in family])):r['p_holm']=p
    for a,b in zip(actual,expected[role]):
        for k,v in a.items():
            assert (v==b[k] if isinstance(v,str) else math.isclose(v,b[k],rel_tol=1e-11,abs_tol=1e-13)),(role,k,v,b[k])
print('PASS: main and all ablation means, SDs, gains, outcomes, exact one/two-sided probabilities and Holm corrections reproduced.')
