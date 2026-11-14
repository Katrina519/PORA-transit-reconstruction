"""预登记两图反馈采集、分组CV与一次冻结联合模型。"""
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
import model as learner
IMPORT_START = time.perf_counter()
import bridge as b
IMPORT_SECONDS = time.perf_counter()-IMPORT_START


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, obj):
    with (ROOT/name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)


def canonical(routes):
    return tuple(sorted(min(tuple(rr), tuple(rr[::-1])) for rr in routes))


def prepare():
    started = time.perf_counter()
    sources = {'mandl': WORK/'nea_spo_v2_independent_confirm_20261108_v1',
               'mumford0': WORK/'nea_spo_v2_mumford0_20261109_v1'}
    oldtasks = read(WORK/'nea_spo_balanced_tasks_20261108_v1/训练任务.json')
    for task in oldtasks:
        task['city'] = 'mandl'
    oldroutes = read(WORK/'restoration_native_transfer_20261105_v1/父网与特征.json')['routes']
    oldroutes.update({v['parent']:v['routes'] for v in read(WORK/'nea_spo_balanced_tasks_20261108_v1/诊断完成.json')['records']})
    assert all(t['state'] in oldroutes for t in oldtasks)
    known = {('mandl', canonical(oldroutes[t['state']])) for t in oldtasks}
    contexts, held, receipts, diagnostic = [], [], {}, {}
    for city, source in sources.items():
        b.configure(city); q, np = b.q, b.np
        req = read(source/'请求冻结.json'); runs = read(source/'comparison/COMPLETE.json')['runs']
        for path in [source/'请求冻结.json', source/'comparison/COMPLETE.json', source/'封存清单.json']:
            receipts[str(path)] = sha(path)
        candidates = [dict(c, state=c['parent'], city=city, ancestors=[c['parent']], source='initial') for c in req['contexts']]
        for context in req['contexts'][:2]:
            run = next(v for v in runs if v['parent']==context['parent'] and v['seed']==req['seeds'][0] and v['method']=='author')
            event = next(v for v in run['fine_curve'] if v['iteration']==2 and v['pass']==5)
            candidates.append({'state': context['parent']+'-author-round2', 'city':city,
                'routes':run['fine_solutions'][event['solution_index']]['routes'],
                'ancestors':[context['parent']], 'source':'saved_author_round2'})
        initial_x = []
        for context in candidates:
            identity = (city, canonical(context['routes']))
            assert identity not in known, ('new training state duplicates old training', context['state'])
            assert q.assess(context['routes'])['feasible']
            same = next((c for c in contexts if (c['city'],canonical(c['routes']))==identity), None)
            if same:
                same['ancestors'] = sorted(set(same['ancestors']+context['ancestors']))
                continue
            contexts.append(context)
            if context['source']=='initial':
                x = b.features(context['routes']); initial_x.extend(x)
                predicted = q.ref.selector.scores(b.base.MODEL, x)
                saved = next(v for v in runs if v['parent']==context['state'] and v['method']=='v2_spo')['selection_log'][0][0]['scores']
                assert max(abs(a-c) for a,c in zip(predicted,saved)) < 1e-12
        z = (np.array(initial_x)-b.base.MODEL['mean'])/b.base.MODEL['scale']
        diagnostic[city] = {'rows':len(initial_x), 'max_abs_Z_by_feature':np.max(np.abs(z),axis=0).tolist(),
                            'golden_score_error':0., 'old_nodes_max':float(np.max(np.array(initial_x)[:,1]))}
        known.update((city, canonical(c['routes'])) for c in candidates)
        for run in runs:
            known.update((city, canonical(v['routes'])) for v in run['fine_solutions'])
        spec = b.base.SPEC
        instance = q.c.BenchmarkInstance(tuple(range(1,spec['n']+1)),
            {(u+1,v+1):float(q.ROADS[u,v]) for u in range(spec['n']) for v in range(spec['n']) if u!=v and np.isfinite(q.ROADS[u,v])},
            {(u+1,v+1):float(q.Q[u,v]) for u in range(spec['n']) for v in range(spec['n'])}, frozenset(range(1,spec['n']+1)), {})
        for ident in [62611,62612]:
            seed = q.ref.stream(ident,'initialization').getrandbits(64)
            routes = [[v-1 for v in rr] for rr in q.ref.make_initial_routes(instance,spec['routes'],(spec['min'],spec['max']),seed)]
            identity = (city,canonical(routes))
            assert identity not in known and q.assess(routes)['feasible']
            known.add(identity)
            held.append({'city':city, 'parent': ('mandl1-6-' if city=='mandl' else 'mumford0-12-')+str(ident),
                         'routes':routes, 'initialization_seed':seed})
    oldpath = WORK/'nea_spo_balanced_tasks_20261108_v1/训练任务.json'
    receipts[str(oldpath)] = sha(oldpath)
    previous = read(WORK/'nea_spo_v2_mumford0_20261109_v1/comparison/COMPLETE.json')['input_hashes']
    assert all(sha(p)==h for p,h in previous.items())
    receipts.update(previous)
    receipts.update({str(p):sha(p) for p in ROOT.glob('*.py')})
    receipts[str(ROOT/'协议.md')] = sha(ROOT/'协议.md')
    dump('输入冻结.json', {'old_tasks':oldtasks, 'contexts':contexts, 'label_seeds':[720,721,722],
        'new_roots':sorted({a for c in contexts for a in c['ancestors']}),
        'input_hashes':receipts, 'feature_diagnostic':diagnostic,
        'prepare_seconds':time.perf_counter()-started, 'import_seconds':IMPORT_SECONDS})
    dump('请求冻结.json', {'contexts':held, 'seeds':[530,531,532], 'methods':['author','joint_spo'],
        'primary_seconds':{'mandl':4.,'mumford0':7.}, 'max_iterations':8,
        'scope':'new confirmation, no feedback used for model selection', 'new_training_feedback':False})
    print('PREPARED',len(contexts),'training states;',len(held),'confirmation parents',flush=True)


def sample():
    started = time.perf_counter(); frozen = read(ROOT/'输入冻结.json')
    assert all(sha(p)==h for p,h in frozen['input_hashes'].items())
    rows, tasks = [], list(frozen['old_tasks'])
    with (ROOT/'反馈原始.jsonl').open('x',encoding='utf-8') as stream:
        for context in frozen['contexts']:
            b.configure(context['city']); q,r,torch = b.q,b.r,b.torch
            routes = context['routes']; base = q.assess(routes); assert base['feasible']
            X = b.features(routes)
            torch.manual_seed(0); net,state = q.a.build('neural',routes); cost = r.cost_module()
            tensor = q.a.tu.get_batch_tensor_from_routes([routes],max_route_len=b.base.SPEC['max'])
            original = state.clone(); original.replace_routes(tensor)
            c0 = float(cost(original).cost[0])
            d,nr = b.base.SPEC['diameter'],b.base.SPEC['routes']
            assert abs(c0-base['P']/(2*d)-base['O']/(nr*d)) < 1e-6
            for j in range(len(routes)):
                trial_rows = []
                for seed in frozen['label_seeds']:
                    tick=time.perf_counter(); torch.manual_seed(seed)
                    with torch.no_grad():
                        draft = r.entry.get_neural_variants(net,state.clone(),tensor.clone(),torch.tensor([j]))
                    generated = r.routes(draft[0]); assert generated[:-1]==routes[:j]+routes[j+1:]
                    after = [rr[:] for rr in routes]; after[j]=generated[-1]
                    metrics = q.assess(after)
                    st = state.clone(); st.replace_routes(q.a.tu.get_batch_tensor_from_routes([after],max_route_len=b.base.SPEC['max']))
                    ca = float(cost(st).cost[0])
                    if metrics['feasible']:
                        assert abs(ca-metrics['P']/(2*d)-metrics['O']/(nr*d)) < 1e-6
                    row = {'city':context['city'],'state':context['state'],'j':j,'seed':seed,'routes':after,
                           'cost':ca,'parent_cost':c0,'gain':max(0.,c0-ca) if metrics['feasible'] else 0.,
                           **metrics,'seconds':time.perf_counter()-tick}
                    rows.append(row);trial_rows.append(row)
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                tasks.append({'key':context['state']+'|'+str(j), 'state':context['state'], 'city':context['city'],
                    'ancestors':context['ancestors'],'j':j,'x':X[j],
                    'gain':sum(v['gain'] for v in trial_rows)/len(trial_rows),'n':len(trial_rows)})
            print('LABELLED',context['state'],len(routes),'tasks',flush=True)
    assert all(sha(p)==h for p,h in frozen['input_hashes'].items())
    for context in frozen['contexts']:
        expected = {(j,s) for j in range(len(context['routes'])) for s in frozen['label_seeds']}
        actual = [(v['j'],v['seed']) for v in rows if v['state']==context['state']]
        assert len(actual)==len(set(actual)) and set(actual)==expected
    dump('训练任务.json',tasks)
    dump('反馈完成.json',{'rows':rows,'new_tasks':len(tasks)-len(frozen['old_tasks']),
        'generation_calls':len(rows),'seconds':time.perf_counter()-started,'import_seconds':IMPORT_SECONDS,
        'inputs_unchanged':True,'generator_unchanged':True})
    print('LABEL_COMPLETE',len(tasks),len(rows),flush=True)


def train():
    started=time.perf_counter(); frozen=read(ROOT/'输入冻结.json');tasks=read(ROOT/'训练任务.json')
    assert all(sha(p)==h for p,h in frozen['input_hashes'].items())
    newstates={c['state'] for c in frozen['contexts']};folds=[];seen=set()
    for root in frozen['new_roots']:
        related={root}
        while True:
            enlarged=related|{a for t in tasks if set(t['ancestors'])&related for a in t['ancestors']}
            if enlarged==related:break
            related=enlarged
        identity=tuple(sorted(related))
        if identity in seen:continue
        seen.add(identity)
        train_ids=[i for i,t in enumerate(tasks) if not set(t['ancestors'])&related]
        test_ids=[i for i,t in enumerate(tasks) if t['state'] in newstates and set(t['ancestors'])&related]
        assert train_ids and test_ids
        folds.append({'roots':list(identity),'train_ids':train_ids,'test_ids':test_ids})
    cv=[]
    for encoding in ['raw','scaled']:
        for numerator in [.1,1.,10.]:
            lam=numerator/79.;validation=[];fold_checks=[]
            for fold in folds:
                fit,checks=learner.fit([tasks[i] for i in fold['train_ids']],encoding,lam)
                result=learner.regret(fit,[tasks[i] for i in fold['test_ids']])
                validation+=result['rows'];fold_checks.append({'roots':fold['roots'],**checks})
            assert len(validation)==len(newstates) and len({v['state'] for v in validation})==len(newstates)
            means={c:sum(v['regret'] for v in validation if v['city']==c)/sum(v['city']==c for v in validation)
                   for c in ['mandl','mumford0']}
            value=sum(means.values())/2
            cv.append({'encoding':encoding,'lambda':lam,'balanced_regret':value,'city_means':means,
                       'validation':validation,'fold_checks':fold_checks})
            print('CV',encoding,numerator,round(value,8),flush=True)
    selected=min(cv,key=lambda v:(v['balanced_regret'],v['lambda'],v['encoding']!='raw'))
    document,checks=learner.fit(tasks,selected['encoding'],selected['lambda'])
    old={'encoding':'raw','model':b.base.MODEL}
    olddev=learner.regret(old,[t for t in tasks if t['state'] in newstates])
    dump('模型冻结.json', {'document':document,'checks':checks,'CV':cv,'folds':folds,
        'selected_encoding':selected['encoding'],'selected_lambda':selected['lambda'],
        'CV_balanced_regret':selected['balanced_regret'],'old_model_development_regret':olddev,
        'training_tasks':len(tasks),'new_states':len(newstates),'total_states':len({t['state'] for t in tasks}),
        'fit_seconds':time.perf_counter()-started,'import_seconds':IMPORT_SECONDS,
        'training_source_sha256':sha(ROOT/'训练任务.json'),'confirmation_outcomes_seen':False})
    print('MODEL_FROZEN',selected['encoding'],selected['lambda'],'olddev',olddev['balanced_mean'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','sample','train'],required=True)
    args=parser.parse_args();{'prepare':prepare,'sample':sample,'train':train}[args.phase]()
