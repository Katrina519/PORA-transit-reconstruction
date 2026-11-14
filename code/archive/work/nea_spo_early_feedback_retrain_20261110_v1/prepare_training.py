"""在补反馈前登记早期状态、保留原模型，并冻结新确认输入。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e


def main():
    hashes=e.check_sources();folder=e.ROOT/'prefix';folder.mkdir(exist_ok=False)
    contexts=e.d.contexts('mumford0')
    old=e.read(e.WORK/'mumford0_ab_single_confirm_20261109_v1/comparison/COMPLETE.json')['runs']
    training=[];seen={};prefix=[]
    def add(root,tag,routes):
        key=e.canonical(routes)
        if key in seen:
            seen[key]['sources'].append(tag);return
        c={'state':root+'-early-'+tag,'city':'mumford0','routes':routes,'ancestors':[root],'sources':[tag]}
        training.append(c);seen[key]=c
    for ctx in contexts:
        add(ctx['parent'],'initial',ctx['routes'])
        run=e.d.one(ctx,810,'ours',repair=False,iterations=2)
        ref=next(v for v in old if v['parent']==ctx['parent'] and v['seed']==810 and v['method']=='B_spo')
        assert run['selection_log']==ref['selection_log'][:20]
        assert [v['cost'] for v in run['fine_curve']]==[v['cost'] for v in ref['fine_curve'][:11]]
        assert run['history'][0]==ref['history'][0][:3]
        for sol in run['fine_solutions']:
            _,_,cost=e.audit.evaluate('mumford0',sol['routes']);assert abs(cost-sol['cost'])<1e-6
        batches=[10,16] if ctx['parent'].endswith('62633') else [10]
        for batch in batches:
            events=[x for x in run['feedback_log'] if x['batch_id']==batch]
            selected=min(events,key=lambda x:x['parent_cost'])
            add(ctx['parent'],'batch'+str(batch),selected['parent_routes'])
        prefix.append(run)
        print('PREFIX_IDENTITY',ctx['parent'],len(training),flush=True)
    e.dump(folder/'COMPLETE.json',{'runs':prefix,'passed':True,'original_B_prefix_unchanged':True})
    e.b.configure('mumford0');q=e.q
    instance=q.c.BenchmarkInstance(tuple(range(1,31)),
        {(u+1,v+1):float(q.ROADS[u,v]) for u in range(30) for v in range(30) if u!=v and q.np.isfinite(q.ROADS[u,v])},
        {(u+1,v+1):float(q.Q[u,v]) for u in range(30) for v in range(30)},frozenset(range(1,31)),{})
    history=e.q.load('early_history_helper',e.PREVIOUS/'history_check.py');known=set();receipts=[]
    for pattern in ['nea_*','mandl_*','mumford0_*','restoration_*']:
        for directory in e.WORK.glob(pattern):
            if not directory.is_dir() or directory==e.ROOT:continue
            for p in directory.rglob('*.json'):
                history.walk(e.read(p),known);receipts.append({'path':str(p),'sha256':e.sha(p)})
    fresh=[]
    for ident in range(62731,62735):
        seed=q.ref.stream(ident,'initialization').getrandbits(64)
        routes=[[v-1 for v in rr] for rr in q.ref.make_initial_routes(instance,12,(2,15),seed)]
        assert q.assess(routes)['feasible'] and ('mumford0',e.canonical(routes)) not in known
        known.add(('mumford0',e.canonical(routes)))
        fresh.append({'parent':'mumford0-12-'+str(ident),'city':'mumford0','routes':routes,'initialization_seed':seed})
    original=e.read(e.WORK/'nea_spo_v2_joint_20261109_v1/模型冻结.json')
    e.dump(e.ROOT/'原模型冻结.json',original['document'])
    e.dump(e.ROOT/'请求冻结.json',{'training_contexts':training,'old_contexts':contexts,'fresh_contexts':fresh,
        'label_seeds':[1300,1301,1302],'development_seeds':[810,812],'independent_seeds':[960,961,962],
        'encoding':original['selected_encoding'],'lambda':original['selected_lambda'],
        'methods':['author','retrained_spo'],'primary_seconds':7.,'iterations':8,
        'old_tasks_source':str(e.WORK/'nea_spo_v2_joint_20261109_v1/训练任务.json'),
        'scope':'旧62631—34转为训练开发；62731—34只作一次独立确认，原作者/生成器不变，无容量规则'})
    e.dump(e.ROOT/'历史与来源.json',{'history_files':receipts,'historical_service_plans':len(known)-4,'new_initial_duplicates':[],
        'input_hashes':hashes,'previous_manifest_sha256':e.sha(e.PREVIOUS/'交付封存.json'),
        'model_sha256':e.sha(e.WORK/'nea_spo_v2_joint_20261109_v1/模型冻结.json'),
        'training_contexts':len(training),'prefix_runs':len(prefix),'new_training_not_started':True})
    print('PREPARED',len(training),'training contexts; 4 independent parents',flush=True)


if __name__=='__main__':main()
