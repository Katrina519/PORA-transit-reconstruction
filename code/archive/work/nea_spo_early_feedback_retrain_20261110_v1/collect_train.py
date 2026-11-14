"""沿用原B反馈定义与拟合器，只增加早期全线路实际结果。"""
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e
from labels import aggregate


def collect():
    start=time.perf_counter();request=e.read(e.ROOT/'请求冻结.json');e.check_sources()
    old=e.read(request['old_tasks_source']);tasks=[dict(t) for t in old];rows=[]
    assert len(tasks)==211
    for context in request['training_contexts']:
        e.b.configure('mumford0');X=e.b.features(context['routes'])
        e.torch.manual_seed(0);model,state=e.q.a.build('neural',context['routes']);cost=e.r.cost_module()
        parent=state.clone();parent.replace_routes(e.q.a.tu.get_batch_tensor_from_routes([context['routes']],max_route_len=15))
        base=float(cost(parent).cost[0]);local=[]
        for j in range(12):
            for seed in request['label_seeds']:
                e.torch.manual_seed(seed)
                tensor=e.q.a.tu.get_batch_tensor_from_routes([context['routes']],max_route_len=15)
                with e.torch.no_grad():
                    draft=e.r.entry.get_neural_variants(model,state.clone(),tensor,e.torch.tensor([j]))
                generated=e.r.routes(draft[0]);assert generated[:-1]==context['routes'][:j]+context['routes'][j+1:]
                after=[rr[:] for rr in context['routes']];after[j]=generated[-1]
                metrics=e.q.assess(after)
                current=state.clone();current.replace_routes(e.q.a.tu.get_batch_tensor_from_routes([after],max_route_len=15))
                value=float(cost(current).cost[0])
                if metrics['feasible']:
                    _,_,check=e.audit.evaluate('mumford0',after);assert abs(check-value)<1e-6
                local.append({'state':context['state'],'city':'mumford0','j':j,'seed':seed,
                    'parent_cost':base,'cost':value,'routes':after,**metrics})
        means=aggregate(local,12,request['label_seeds']);rows.extend(local)
        for j in range(12):
            tasks.append({'key':context['state']+'|'+str(j),'state':context['state'],'city':'mumford0',
                'ancestors':context['ancestors'],'j':j,'x':X[j],'gain':means[j],'n':3})
        print('FEEDBACK',context['state'],'best',max(range(12),key=lambda j:means[j]),'gain',round(max(means),7),flush=True)
    e.dump(e.ROOT/'补充反馈.json',{'rows':rows,'generator_calls':len(rows),'seconds':time.perf_counter()-start,
        'all_trials_retained':True,'generator_unchanged':True})
    e.dump(e.ROOT/'训练任务.json',tasks)
    print('COLLECTED',len(rows),'drafts',len(tasks),'tasks',flush=True)


def train():
    start=time.perf_counter();request=e.read(e.ROOT/'请求冻结.json');e.check_sources()
    frozen=e.read(e.ROOT/'拟合输入冻结.json')['input_hashes']
    assert all(e.sha(p)==h for p,h in frozen.items())
    old=e.read(request['old_tasks_source']);tasks=e.read(e.ROOT/'训练任务.json')
    original=e.read(e.ROOT/'原模型冻结.json');learner=e.b.learner
    reproduced,check=learner.fit(old,request['encoding'],request['lambda'])
    assert reproduced['encoding']==original['encoding']
    errors={key:float(e.b.np.max(e.b.np.abs(e.b.np.asarray(reproduced['model'][key])-original['model'][key])))
        for key in ['mean','scale','coef']}
    assert max(errors.values())<1e-7,errors
    document,checks=learner.fit(tasks,request['encoding'],request['lambda'])
    oldstates={t['state'] for t in old};new=[t for t in tasks if t['state'] not in oldstates]
    diagnostics={'old_on_old':learner.regret(original,old),'new_on_old':learner.regret(document,old),
        'old_on_added':learner.regret(original,new),'new_on_added':learner.regret(document,new)}
    e.dump(e.ROOT/'模型冻结.json',{'document':document,'encoding':request['encoding'],'lambda':request['lambda'],
        'training_tasks':len(tasks),'training_states':len({t['state'] for t in tasks}),
        'training_data_sha256':e.sha(e.ROOT/'训练任务.json'),'old_tasks_preserved':tasks[:len(old)]==old,
        'hyperparameters_unchanged':True,'confirmation_outcomes_seen':False})
    e.dump(e.ROOT/'训练检查.json',{'old_model_reproduction_errors':errors,'old_fit_checks':check,'new_fit_checks':checks,
        'fit_seconds':time.perf_counter()-start,'diagnostics':diagnostics,
        'interpretation':'旧样本/新增样本上的离线排序核验，不是作者性能对照，不作为独立测试成绩'})
    assert all(e.sha(p)==h for p,h in frozen.items())
    print('TRAINED',len(tasks),'tasks',len({t['state'] for t in tasks}),'states; old/new added regret',
        diagnostics['old_on_added']['balanced_mean'],diagnostics['new_on_added']['balanced_mean'],flush=True)


if __name__=='__main__':{'collect':collect,'train':train}[sys.argv[1]]()
