"""确认完成后诊断62733；不把新探测用于本轮重训或重算确认成绩。"""
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e
from labels import aggregate


def main():
    started=time.perf_counter();request=e.read(e.ROOT/'请求冻结.json')
    ctx=next(c for c in request['fresh_contexts'] if c['parent'].endswith('62733'))
    document=e.read(e.ROOT/'归一化模型冻结.json')['document']
    e.b.configure('mumford0',document);X=e.b.features(ctx['routes'])
    scores=e.b.learner.scores(document,X,'mumford0').tolist()
    saved=next(v for v in e.read(e.ROOT/'independent/COMPLETE.json')['runs'] if v['parent']==ctx['parent'] and v['method']=='ours')
    assert max(abs(a-b) for a,b in zip(scores,saved['selection_log'][0][0]['scores']))<1e-12
    e.torch.manual_seed(0);net,state=e.q.a.build('neural',ctx['routes']);cost=e.r.cost_module()
    original=state.clone();original.replace_routes(e.q.a.tu.get_batch_tensor_from_routes([ctx['routes']],max_route_len=15))
    c0=float(cost(original).cost[0]);rows=[];seeds=[1310,1311,1312]
    for j in range(12):
        for seed in seeds:
            e.torch.manual_seed(seed)
            tensor=e.q.a.tu.get_batch_tensor_from_routes([ctx['routes']],max_route_len=15)
            with e.torch.no_grad():draft=e.r.entry.get_neural_variants(net,state.clone(),tensor,e.torch.tensor([j]))
            generated=e.r.routes(draft[0]);assert generated[:-1]==ctx['routes'][:j]+ctx['routes'][j+1:]
            after=[rr[:] for rr in ctx['routes']];after[j]=generated[-1];metrics=e.q.assess(after)
            afterstate=state.clone();afterstate.replace_routes(e.q.a.tu.get_batch_tensor_from_routes([after],max_route_len=15))
            ca=float(cost(afterstate).cost[0])
            if metrics['feasible']:
                _,_,verify=e.audit.evaluate('mumford0',after);assert abs(verify-ca)<1e-6
            rows.append({'j':j,'seed':seed,'parent_cost':c0,'cost':ca,'routes':after,**metrics})
    gains=aggregate(rows,12,seeds)
    tasks=[{'j':j,'score':scores[j],'mean_positive_gain':gains[j],
        'legal':sum(r['feasible'] for r in rows if r['j']==j),'features':X[j]} for j in range(12)]
    tasks.sort(key=lambda t:(-t['mean_positive_gain'],t['j']))
    result={'parent':ctx,'parent_cost':c0,'seeds':seeds,'drafts':36,'rows':rows,'ranked_tasks':tasks,
        'model_selected':min(range(12),key=lambda j:(-scores[j],j)),
        'confirmation_complete_before_probe':True,'not_used_to_fit_current_model':True,
        'scope':'三个新探测种子的诊断均值，不声称期望最优或完整终态因果','seconds':time.perf_counter()-started}
    e.dump(e.ROOT/'62733初始任务探测.json',result)
    print('CURRENT_CHOICE',result['model_selected'],'gain',gains[result['model_selected']],flush=True)
    print('TOP_PROBED',[(t['j'],round(t['mean_positive_gain'],7),t['legal'],round(t['score'],5)) for t in tasks[:5]],flush=True)


if __name__=='__main__':main()
