"""确定性首片段探测：不完整生成、不训练、不改既有评分。"""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
BATCHES=['restoration_first_line_risk_20261106_v1','restoration_rule_comparison_20261107_v1','restoration_slack_validation_20261108_v1']


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x):
    with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=load('fixed_probe',WORK/'reconstruction_task_probe_20261104_v1/run_probe.py')
    a,np,c=p.a,p.np,p.c
    roads=np.loadtxt(a.INPUTS/'MandlTravelTimes.txt');q=np.loadtxt(a.INPUTS/'MandlDemand.txt');Q=float(q.sum())
    feature_dir=WORK/'restoration_service_features_20261108_v1'
    fresh=WORK/BATCHES[-1]
    f=read(feature_dir/'部署冻结.json');models={v:f['models'][v] for v in ['original','H']}
    hf=read(feature_dir/'特征.json')
    paths=[Path(__file__),ROOT/'预先协议.md',ROOT/'verify.py',Path(a.__file__),Path(c.__file__),Path(p.__file__),a.WEIGHT,
           a.INPUTS/'MandlTravelTimes.txt',a.INPUTS/'MandlDemand.txt']
    paths+=list(a.SRC.rglob('*.py'))+list(a.SRC.rglob('*.yaml'))
    for folder in [WORK/b for b in BATCHES]+[feature_dir,WORK/'restoration_service_pairs_20261107_v1']:
        for name,h in read(folder/'交付文件哈希.json').items():
            assert sha(folder/name)==h,(folder,name)
            paths.append(folder/name)
    assert sha(a.SRC/'learning/models.py')==sha(p.AUTHOR/'learning/models.py')
    receipt={str(x):sha(x) for x in paths}
    dump('INPUT_RECEIPT.json',{'hashes':receipt,'planned_distribution_queries':78,'full_generation_calls':0,'fits':0})
    entries=[];timings=[];six=read(WORK/'restoration_service_pairs_20261107_v1/FIRST_ACTION_RESULTS.json');oldfirst={(str(x['parent']),x['line']):x for x in six}
    first_matches=0
    for batch in BATCHES:
        contexts=read(WORK/batch/'部署冻结.json')['contexts']
        for ctx in contexts:
            R=ctx['routes'];statekey=ctx['parent'];base=ctx['base'];D0=c.unlimited(c.direct(roads,R))
            assert abs(c.objective(D0,q)-base['P'])<1e-10
            a.torch.manual_seed(0);tick=time.perf_counter();model,state=a.build('neural',R)
            loading=time.perf_counter()-tick
            weights={k:t.detach().clone() for k,t in model.state_dict().items()}
            H=ctx.get('H',[hf.get(statekey+'|'+str(j),{}).get('H') for j in range(6)])
            scores={}
            for v,m in models.items():
                X=np.asarray([x+([H[j]] if v=='H' else []) for j,x in enumerate(ctx['features'])])
                scores[v]=(m['coef'][0]+((X-m['mean'])/m['scale'])@np.asarray(m['coef'][1:])).tolist()
            expected=ctx.get('original_scores',ctx.get('scores',{}).get('original'))
            np.testing.assert_allclose(scores['original'],expected,atol=1e-12,rtol=0)
            tasks=[]
            for j in range(6):
                tick=time.perf_counter();remaining=R[:j]+R[j+1:];st=state.clone()
                st.replace_routes(a.tu.get_batch_tensor_from_routes([remaining],max_route_len=8));st=model.setup_planning(st)
                assert a.route(st)==();setup=time.perf_counter()-tick
                rng=a.torch.random.get_rng_state();tick=time.perf_counter();halt,options=a.distribution(model,st)
                query=time.perf_counter()-tick;assert halt==0. and a.torch.equal(rng,a.torch.random.get_rng_state())
                _,route,prob=max(options,key=lambda x:(x[2],tuple(-u for u in x[1])))
                tick=time.perf_counter();new=remaining+[list(route)];Dr=c.unlimited(c.direct(roads,remaining));D=c.unlimited(c.direct(roads,new))
                mask=(q>0)&np.isfinite(Dr);G=float(np.sum(q[mask]*(Dr[mask]-D[mask]))/Q)
                V=float(np.sum(q[mask]*(D0[mask]-D[mask]))/Q);L=float(np.sum(q[mask]*(Dr[mask]-D0[mask]))/Q)
                assert G>=-1e-10 and abs(V-(G-L))<1e-10 and abs(L-ctx['features'][j][4]*33)<1e-10
                P=c.objective(D,q);O=float(sum(roads[u,v] for rr in new for u,v in zip(rr,rr[1:])))
                covered=set().union(*map(set,new));missing=sorted(set(range(15))-covered)
                legal=(not missing and P is not None and all(2<=len(rr)<=8 and len(set(rr))==len(rr) and all(0<roads[u,v]<np.inf for u,v in zip(rr,rr[1:])) for rr in new))
                raw=(base['P']-P)/66+(base['O']-O)/840 if legal else None
                evaluation=time.perf_counter()-tick
                row={'j':j,'route':list(route),'prob':prob,'option_count':len(options),'G':G,'V':V,'L':L,'P':P,'O':O,
                     'feasible':bool(legal),'missing':[u+1 for u in missing],'raw_gain':raw,
                     'times':{'setup':setup,'query':query,'evaluation':evaluation}}
                old=oldfirst.get((statekey[-5:],j+1))
                if old is not None:
                    assert [u+1 for u in route]==old['top_first_route']
                    for k,got in [('prob',prob),('P',P),('O',O),('raw_gain',raw)]:assert abs(got-old[k])<1e-10
                    first_matches+=1
                tasks.append(row)
            assert all(a.torch.equal(t,model.state_dict()[k]) for k,t in weights.items())
            allowed=[j for j,d in enumerate(ctx['structural_details']) if d['count']>0];assert allowed
            pick=lambda values,ix:min(ix,key=lambda j:(-values[j],j))
            picks={v:pick(scores[v],allowed) for v in scores}
            valid=[j for j in allowed if tasks[j]['feasible']]
            picks['H_probe']=pick(scores['H'],valid) if valid else picks['H']
            picks['G_probe']=pick([x['G'] for x in tasks],valid) if valid else picks['H']
            picks['C_probe']=pick([x['raw_gain'] if x['raw_gain'] is not None else -1e100 for x in tasks],valid) if valid else picks['H']
            choices={v:{'rank':j,'guard':0 if 0 in allowed else j} for v,j in picks.items()}
            entries.append({'parent':statekey,'batch':batch,'routes':R,'base':base,'scores':scores,'tasks':tasks,
                            'allowed':allowed,'probe_valid':valid,'fallback':not bool(valid),'choices':choices,'load_seconds':loading})
    assert len(entries)==13 and sum(len(x['tasks']) for x in entries)==78 and first_matches==6
    assert all(sha(Path(k))==v for k,v in receipt.items())
    dump('探测冻结.json',{'entries':entries,'original_six_matched':first_matches,'queries':78,'full_generation_calls':0,'fits':0})
    # 全部决策冻结后，才加载完整生成结果进行回放。
    outcomes={b:read(WORK/b/'COMPLETE.json') for b in BATCHES};records=[]
    for e in entries:
        pool=[r for r in outcomes[e['batch']]['rows'] if r['parent']==e['parent']]
        tasks=[]
        for j in range(6):
            xs=[x for x in pool if x['j']==j];assert len(xs)==4
            tasks.append({'j':j,'gain':float(np.mean([x['gain'] for x in xs])),
                          'feasible':sum(x['feasible'] for x in xs),
                          'accepted_P':float(np.mean([x['accepted_P'] for x in xs])),
                          'accepted_O':float(np.mean([x['accepted_O'] for x in xs]))})
        best=max(tasks[j]['gain'] for j in e['allowed']);methods={}
        for name,modes in e['choices'].items():
            methods[name]={mode:{**tasks[j],'regret':best-tasks[j]['gain']} for mode,j in modes.items()}
        records.append({'parent':e['parent'],'batch':e['batch'],'tasks':tasks,'best':best,'methods':methods})
    stats={}
    for label,group in [('known9',records[:9]),('recent4',records[9:])]:
        stats[label]={}
        for name in records[0]['methods']:
            stats[label][name]={}
            for mode in ['rank','guard']:
                vals=[x['methods'][name][mode] for x in group]
                summary={'gain':float(np.mean([x['gain'] for x in vals])),
                         'regret':float(np.mean([x['regret'] for x in vals])),
                         'correct':sum(x['regret']<1e-12 for x in vals),'vs':{}}
                for reference in ['original','H','H_probe','C_probe']:
                    base=[x['methods'][reference][mode] for x in group];diff=[a['gain']-b['gain'] for a,b in zip(vals,base)]
                    bm=float(np.mean([x['gain'] for x in base]));summary['vs'][reference]={
                        'better':sum(x>1e-12 for x in diff),'worse':sum(x< -1e-12 for x in diff),'same':sum(abs(x)<=1e-12 for x in diff),
                        'relative_change':(summary['gain']-bm)/bm if bm else None,
                        'old_correct':sum(x['regret']<1e-12 for x in base),
                        'retained':sum(a['regret']<1e-12 and b['regret']<1e-12 for a,b in zip(vals,base))}
                stats[label][name][mode]=summary
    costs={'load_sum':sum(e['load_seconds'] for e in entries),
           **{k+'_median_task':float(np.median([t['times'][k] for e in entries for t in e['tasks']])) for k in ['setup','query','evaluation']},
           'probe_median_parent':float(np.median([sum(sum(t['times'].values()) for t in e['tasks']) for e in entries])),
           'historical_full_generation_median':float(np.median([x['generation_seconds'] for r in outcomes.values() for x in r['rows']]))}
    assert all(sha(Path(k))==v for k,v in receipt.items())
    dump('COMPLETE.json',{'records':records,'summary':stats,'costs':costs,'full_generation_calls':0,'fits':0,'queries':78,'inputs_unchanged':True})
    print(json.dumps({'summary':{g:{m:{mode:{k:v for k,v in d.items() if k!='vs'} for mode,d in a.items()} for m,a in b.items()} for g,b in stats.items()},'costs':costs}))


if __name__=='__main__':main()
