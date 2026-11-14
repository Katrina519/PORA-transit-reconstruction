"""仅替换训练状态权重的归一化；沿用原优化与SPO+。"""
import sys
import inspect
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e
from city_weights import weights


def main():
    started=time.perf_counter();request=e.read(e.ROOT/'请求冻结.json');tasks=e.read(e.ROOT/'训练任务.json')
    hashes=e.read(e.ROOT/'拟合输入冻结.json')['input_hashes']
    assert all(e.sha(p)==h for p,h in hashes.items())
    hashes.update({str(p):e.sha(p) for p in e.ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.json']})
    e.dump(e.ROOT/'归一化拟合冻结.json',{'input_hashes':hashes,'request_sha256':e.sha(e.ROOT/'请求冻结.json'),
        'change':'仅w_s按城内最大收益和归一，城市总损失各半；lambda/features/data/loss形式不变'})
    source=inspect.getsource(e.b.learner.fit)
    anchor='weights = p*maxima/reference';assert source.count(anchor)==1
    scope=dict(e.b.learner.__dict__,city_weights=weights)
    exec(compile(source.replace(anchor,'weights = city_weights(maxima,cities)'),__file__+'::same_SPO_city_weights','exec'),scope)
    document,checks=scope['fit'](tasks,request['encoding'],request['lambda'])
    totals={c:sum(w for w,k in zip(weights(checks['group_maxima'],checks['group_cities']),checks['group_cities']) if k==c)
        for c in set(checks['group_cities'])}
    assert max(abs(v-.5) for v in totals.values())<1e-12
    old=e.read(e.ROOT/'原模型冻结.json');newstates={c['state'] for c in request['training_contexts']}
    oldtasks=e.read(request['old_tasks_source']);added=[t for t in tasks if t['state'] in newstates]
    diagnostics={'old_on_added':e.b.learner.regret(old,added),'normalized_on_added':e.b.learner.regret(document,added),
        'old_on_old':e.b.learner.regret(old,oldtasks),'normalized_on_old':e.b.learner.regret(document,oldtasks)}
    e.dump(e.ROOT/'归一化模型冻结.json',{'document':document,'encoding':request['encoding'],'lambda':request['lambda'],
        'training_data_sha256':e.sha(e.ROOT/'训练任务.json'),'training_tasks':len(tasks),
        'city_loss_weight_totals':totals,'change':'equal_city_weight_normalization','confirmation_outcomes_seen':False})
    e.dump(e.ROOT/'归一化训练检查.json',{'KKT':checks,'diagnostics':diagnostics,'fit_seconds':time.perf_counter()-started})
    assert all(e.sha(p)==h for p,h in hashes.items())
    print('CITY_NORMALIZED',totals,'added old/new regret',diagnostics['old_on_added']['balanced_mean'],
        diagnostics['normalized_on_added']['balanced_mean'],flush=True)


if __name__=='__main__':main()
