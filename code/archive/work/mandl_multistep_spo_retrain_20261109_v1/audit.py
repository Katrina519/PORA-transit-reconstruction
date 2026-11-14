"""三方法完整配对、时间截止与P/O的独立重算。"""
import json
import csv
import statistics
import hashlib
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    req=read(ROOT/'请求冻结.json');done=read(ROOT/'comparison/COMPLETE.json');runs=done['runs']
    assert all(sha(p)==h for p,h in done['input_hashes'].items())
    raw=[json.loads(s) for s in (ROOT/'comparison/原始运行.jsonl').read_text(encoding='utf-8').splitlines()];assert raw==runs
    expected={(c['parent'],s,m) for c in req['contexts'] for s in req['seeds'] for m in req['methods']}
    keys=[(r['parent'],r['seed'],r['method']) for r in runs];assert len(keys)==len(set(keys))==18 and set(keys)==expected
    spec=importlib.util.spec_from_file_location('independent_mandl_audit',ROOT.parent/'nea_spo_v2_joint_20261109_v1/audit.py')
    independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
    rows=[];maximum=0.;decisions=0;uncovered=[]
    for r in runs:
        assert r['target_samples']==811 and r['target_calls']==82 and len(r['fine_curve'])==41
        decisions+=independent.check_choices(r,6)
        for s in r['fine_solutions']:
            _,_,c=independent.evaluate('mandl',s['routes']);maximum=max(maximum,abs(c-s['cost']))
        for e in r['fine_curve']:
            s=r['fine_solutions'][e['solution_index']]
            assert s['first_seconds']<=e['seconds'] and s['cost']==e['cost']
        if r['fine_curve'][-1]['seconds']<req['primary_seconds']:uncovered.append((r['parent'],r['seed'],r['method']))
        for scope,seconds in [('primary',req['primary_seconds']),('fixed_8',None)]:
            event=r['fine_curve'][-1] if seconds is None else next((e for e in reversed(r['fine_curve']) if e['seconds']<=seconds),None)
            assert event is not None
            p,o,c=independent.evaluate('mandl',r['fine_solutions'][event['solution_index']]['routes'])
            rows.append({'parent':r['parent'],'seed':r['seed'],'method':r['method'],'scope':scope,
                'cost':event['cost'],'P_minutes':p,'O_minutes':o,'online_seconds':r['seconds'],
                'selection_seconds':r['selection_seconds'],'configuration_seconds':r['configuration_seconds'],
                'event_seconds':event['seconds']})
    assert maximum<1e-6
    results={}
    for scope in ['primary','fixed_8']:
        if scope=='primary' and uncovered:
            results[scope]={'status':'insufficient_time_coverage','uncovered':uncovered};continue
        part=[v for v in rows if v['scope']==scope]
        means={m:{k:statistics.mean(v[k] for v in part if v['method']==m) for k in
               ['cost','P_minutes','O_minutes','online_seconds','selection_seconds','configuration_seconds']} for m in req['methods']}
        for m in means:means[m]['cost_sd']=statistics.stdev(v['cost'] for v in part if v['method']==m)
        index={(v['parent'],v['seed'],v['method']):v for v in part};pairs=[];parents=[]
        for parent in sorted({v['parent'] for v in part}):
            for seed in req['seeds']:
                v={m:index[parent,seed,m]['cost'] for m in req['methods']};a=v['author']
                pairs.append({'parent':parent,'seed':seed,**v,
                    'single_vs_author_percent':100*(a-v['single_spo'])/a,
                    'multi_vs_author_percent':100*(a-v['multi_spo'])/a,
                    'multi_vs_single_percent':100*(v['single_spo']-v['multi_spo'])/v['single_spo']})
            v={m:statistics.mean(index[parent,s,m]['cost'] for s in req['seeds']) for m in req['methods']}
            parents.append({'parent':parent,**v,'multi_vs_author_percent':100*(v['author']-v['multi_spo'])/v['author']})
        a=means['author']['cost'];s=means['single_spo']['cost'];m=means['multi_spo']['cost']
        def counts(reference):
            values=[p[reference]-p['multi_spo'] for p in pairs]
            return {'wins':sum(x>1e-6 for x in values),'ties':sum(abs(x)<=1e-6 for x in values),'losses':sum(x<-1e-6 for x in values)}
        results[scope]={'means':means,'pairs':pairs,'parents':parents,
            'single_vs_author_percent':100*(a-s)/a,'multi_vs_author_percent':100*(a-m)/a,
            'multi_vs_single_percent':100*(s-m)/s,'multi_vs_author_counts':counts('author'),
            'multi_vs_single_counts':counts('single_spo')}
    out={'runs':18,'max_cost_error':maximum,'unique_saved_solutions':len(independent.CACHE),
        'cost_samples':sum(r['target_samples'] for r in runs),'selection_events':decisions,
        'all_input_hashes_unchanged':True,'input_hashes_checked':len(done['input_hashes']),
        'results':results,'wall_seconds':done['wall_seconds'],'import_seconds':done['import_seconds'],
        'raw_io_seconds':done['raw_io_seconds'],'configuration_seconds':sum(c['seconds'] for c in done['method_configuration']),
        'scope':'2 new parent clusters; small independent confirmation, no significance claim'}
    with (ROOT/'对照统计.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    with (ROOT/'逐次结果.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
