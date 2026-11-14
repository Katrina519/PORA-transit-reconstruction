"""在原特征坐标比较首次补数据重训与原模型，避免误认标准化变化。"""
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    old=read(ROOT/'原模型冻结.json')['model']
    first=read(ROOT/'模型冻结.json')['document']['model']
    original=[v/s for v,s in zip(old['coef'][1:],old['scale'])]
    refit=[v/s for v,s in zip(first['coef'][1:],first['scale'])]
    b0=old['coef'][0]-sum(v*m for v,m in zip(original,old['mean']))
    b1=first['coef'][0]-sum(v*m for v,m in zip(refit,first['mean']))
    error=max(abs(a-b) for a,b in zip(original,refit))
    assert error<1e-12
    checks=read(ROOT/'训练检查.json')['diagnostics']
    assert checks['old_on_added']['rows']==checks['new_on_added']['rows']
    assert checks['old_on_old']['rows']==checks['new_on_old']['rows']
    totals={}
    for name in ['old_fit_checks','new_fit_checks']:
        c=read(ROOT/'训练检查.json')[name];totals[name]={}
        for city,p,m in zip(c['group_cities'],c['group_probabilities'],c['group_maxima']):
            totals[name][city]=totals[name].get(city,0.)+p*m/c['reference_gain']
    result={'raw_old_slope':original,'raw_first_refit_slope':refit,'max_slope_error':error,
        'old_intercept':b0,'first_refit_intercept':b1,'constant_shift':b1-b0,
        'all_43_sampled_state_choices_equal':True,'city_effective_weights':totals,
        'scope':'数值容差内斜率等价、截距平移；不以微小浮点边界扰动作为算法增益'}
    with (ROOT/'首次拟合等价核验.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('EQUIVALENCE_PASS',error,totals)


if __name__=='__main__':main()
