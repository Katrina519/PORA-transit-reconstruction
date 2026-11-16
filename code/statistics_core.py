"""精确条件单侧配对秩检验；零差移除，并列使用平均秩。"""
from fractions import Fraction


def signed_rank(delta):
    values=[float(x) for x in delta if float(x)!=0.]
    if not values:return dict(W_plus=0.,n_nonzero=0,p_one_sided=1.,p_two_sided=1.)
    ordered=sorted(range(len(values)),key=lambda i:abs(values[i]))
    ranks=[0]*len(values);i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and abs(values[ordered[j]])==abs(values[ordered[i]]):j+=1
        for k in ordered[i:j]:ranks[k]=i+j+1
        i=j
    observed=sum(r for r,v in zip(ranks,values) if v>0);total=sum(ranks)
    ways=[0]*(total+1);ways[0]=1
    for rank in ranks:
        for k in range(total,rank-1,-1):ways[k]+=ways[k-rank]
    upper=Fraction(sum(ways[observed:]),2**len(values))
    lower=Fraction(sum(ways[:observed+1]),2**len(values))
    return dict(W_plus=observed/2.,n_nonzero=len(values),p_one_sided=float(upper),
                p_two_sided=float(min(1,2*min(lower,upper))))


def holm(values):
    out=[0.]*len(values);previous=0.
    for position,i in enumerate(sorted(range(len(values)),key=lambda k:values[k])):
        previous=max(previous,min(1.,(len(values)-position)*values[i]));out[i]=previous
    return out
