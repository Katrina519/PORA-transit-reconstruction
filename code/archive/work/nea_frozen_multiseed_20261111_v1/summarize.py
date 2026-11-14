"""完整逐次交付；不修改原记录，不省略负结果。"""
import sys,json,csv,statistics,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e


def point(run,budget):
    return [p for p in run['fine_curve'] if p['seconds']<=budget][-1]


def gain(author,method):return 100*(author-method)/author


def write_csv(name,rows):
    with (ROOT/name).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stats(rows,method,metric='budget'):
    a=[r['author_'+metric] for r in rows];b=[r[method+'_'+metric] for r in rows]
    return {'author_mean':statistics.mean(a),'method_mean':statistics.mean(b),
        'author_std':statistics.stdev(a),'method_std':statistics.stdev(b),
        'improvement_percent':gain(statistics.mean(a),statistics.mean(b)),
        'wins':sum(y<x-1e-6 for x,y in zip(a,b)),'ties':sum(abs(y-x)<=1e-6 for x,y in zip(a,b)),
        'losses':sum(y>x+1e-6 for x,y in zip(a,b)),'n':len(rows),
        'mean_paired_percent':statistics.mean(gain(x,y) for x,y in zip(a,b))}


def cluster_interval(rows,method):
    parents=sorted({r['parent'] for r in rows});roots={p:[r for r in rows if r['parent']==p] for p in parents};values=[]
    for sample in itertools.product(parents,repeat=len(parents)):
        rs=[r for p in sample for r in roots[p]]
        values.append(stats(rs,method)['improvement_percent'])
    values.sort()
    def quantile(q):
        at=(len(values)-1)*q;lo=int(at);hi=min(lo+1,len(values)-1)
        return values[lo]+(at-lo)*(values[hi]-values[lo])
    return [quantile(.025),quantile(.975)]


def main():
    req=e.read(ROOT/'请求冻结.json');e.check(req)
    doc=e.read(ROOT/'comparison/COMPLETE.json');raw=[json.loads(s) for s in (ROOT/'comparison/原始运行.jsonl').read_text('utf-8').splitlines()]
    assert doc['passed'] and doc['runs']==raw and len(raw)==320 and doc['request_sha256']==e.sha(ROOT/'请求冻结.json')
    assert all((r['parent'],r['seed'],r['variant'])==(c['parent'],s,m) for r,(c,s,m) in zip(raw,req['tasks']))
    assert e.read(ROOT/'gate/COMPLETE.json')['passed']
    groups={};count_replayed=0
    for r in raw:
        key=(r['city'],r['parent'],r['seed']);group=groups.setdefault(key,{})
        assert r['variant'] not in group;group[r['variant']]=r
        assert r['generation_counts']==[80,800] and r['target_samples']==811 and r['target_calls']==82
        assert len(r['fine_curve'])==41
        assert all(r['fine_curve'][i]['seconds']<=r['fine_curve'][i+1]['seconds'] for i in range(40))
        assert all(r['fine_curve'][i]['cost']>=r['fine_curve'][i+1]['cost'] for i in range(40))
        assert all(r['metrics'][k]==[0.] for k in ['# disconnected node pairs','# stops out of bounds'])
        if r['variant']=='count':
            counts={};events={(v['batch_id'],v['bee']):v for v in r['feedback_log']}
            for batch,logs in enumerate(r['selection_log']):
                for log in logs:
                    if 'fallback' in log:continue
                    event=events[batch,log['bee']];rr=tuple(tuple(line) for line in event['parent_routes'])
                    cc=counts.setdefault(rr,[0]*len(log['scores']));j=min(range(len(cc)),key=lambda k:(cc[k],k));cc[j]+=1
                    assert j==log['new']==event['j'] and all(s==0 for s in log['scores'])
                    count_replayed+=1
    assert len(groups)==80 and all(set(g)==set(req['methods']) for g in groups.values())
    runtime=e.load_runtime();rows=[];target_error=0.;target_records=0
    for city in ['mandl','mumford0']:
        runtime.d.b.configure(city);denP,denO=(66.,198.) if city=='mandl' else (52.,312.)
        for (c,parent,seed),group in sorted(groups.items()):
            if c!=city:continue
            row={'city':city,'parent':parent,'seed':seed}
            for method in req['methods']:
                run=group[method];p=point(run,req['budgets'][city]);solution=run['fine_solutions'][p['solution_index']]
                metrics=runtime.d.q.assess(solution['routes'])
                for routes,expected in [(solution['routes'],p['cost']),(run['routes'],run['cost'])]:
                    m=runtime.d.q.assess(routes);assert m['feasible']
                    actual=m['P']/denP+m['O']/denO;target_error=max(target_error,abs(actual-expected));target_records+=1
                    assert abs(actual-expected)<1e-6
                row.update({method+'_budget':p['cost'],method+'_final':run['cost'],method+'_P':metrics['P'],method+'_O':metrics['O'],
                    method+'_online_seconds':run['seconds'],method+'_selection_seconds':run['selection_seconds'],
                    method+'_feedback_seconds':run['feedback_seconds'],'%s_completed_passes'%method:len([x for x in run['fine_curve'][1:] if x['seconds']<=req['budgets'][city]])})
            for method in ['full','count','ridge']:
                row[method+'_improvement_percent']=gain(row['author_budget'],row[method+'_budget'])
                row[method+'_final_improvement_percent']=gain(row['author_final'],row[method+'_final'])
            rows.append(row)
    assert len(rows)==80
    write_csv('全部逐次.csv',rows)
    seedrows=[];parentrows=[];summary=[];controls=[]
    for city in ['mandl','mumford0']:
        rs=[r for r in rows if r['city']==city]
        for seed in req['seeds']:
            ss=[r for r in rs if r['seed']==seed];assert len(ss)==4
            sr={'city':city,'seed':seed,'parent_count':4}
            sr.update({m+'_mean_cost':statistics.mean(r[m+'_budget'] for r in ss) for m in req['methods']})
            sr.update({m+'_improvement_percent':gain(sr['author_mean_cost'],sr[m+'_mean_cost']) for m in ['full','count','ridge']});seedrows.append(sr)
        for parent in sorted({r['parent'] for r in rs}):
            ss=[r for r in rs if r['parent']==parent];assert len(ss)==10
            pr={'city':city,'parent':parent,'n':len(ss)}
            pr.update({m+'_mean_cost':statistics.mean(r[m+'_budget'] for r in ss) for m in req['methods']})
            pr.update({m+'_improvement_percent':gain(pr['author_mean_cost'],pr[m+'_mean_cost']) for m in ['full','count','ridge']});parentrows.append(pr)
        for method in ['full','count','ridge']:
            summary.append({'city':city,'method':method,'main':stats(rs,method),'final8':stats(rs,method,'final'),
                'parent_cluster_descriptive_95_interval':cluster_interval(rs,method)})
        for method in ['count','ridge']:
            full=statistics.mean(r['full_budget'] for r in rs);cc=statistics.mean(r[method+'_budget'] for r in rs)
            controls.append({'city':city,'comparator':method,'full_improvement_percent':gain(cc,full),
                'wins':sum(r['full_budget']<r[method+'_budget']-1e-6 for r in rs),
                'ties':sum(abs(r['full_budget']-r[method+'_budget'])<=1e-6 for r in rs),
                'losses':sum(r['full_budget']>r[method+'_budget']+1e-6 for r in rs)})
    write_csv('按种子汇总.csv',seedrows);write_csv('按起点汇总.csv',parentrows)
    lines=['# 全部逐次结果','', '2026-11-11。主预算Mandl4秒、Mumford07秒。cost越低越好；收益为100×(同组作者−当前方法)/同组作者。负值全部保留。每张图40行，四个旧起点×十个新种子。','']
    for city in ['mandl','mumford0']:
        lines+=['## '+city,'','|起点|种子|作者cost|原主方法cost|主方法收益%|计数cost|计数收益%|回归cost|回归收益%|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['city']==city:
                lines.append('|'+str(r['parent'].split('-')[-1])+'|'+str(r['seed'])+'|'+f"{r['author_budget']:.9f}|{r['full_budget']:.9f}|{r['full_improvement_percent']:+.4f}|{r['count_budget']:.9f}|{r['count_improvement_percent']:+.4f}|{r['ridge_budget']:.9f}|{r['ridge_improvement_percent']:+.4f}|" )
        lines+=['']
    with (ROOT/'全部逐次结果.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    e.check(req)
    e.dump(ROOT/'统计与核验.json',{'summary':summary,'full_vs_controls':controls,'parent_means':parentrows,'seed_means':seedrows,
        'formal_runs':len(raw),'paired_groups':len(groups),'count_choices_replayed':count_replayed,'target_records':target_records,
        'target_max_error':target_error,'source_hashes_unchanged':True,'request_sha256':e.sha(ROOT/'请求冻结.json'),
        'formal_phase_seconds':doc['phase_seconds'],'import_seconds':doc['import_seconds_this_invocation'],
        'scope':'新增十随机种子，原四初始线网；不把40运行当40独立网络；不合并旧批，不据输赢加测'})
    report=['# 冻结主方法十种子扩充结果','','完成320次正式运行和8次旧入口身份门禁，未训练、调参或更改作者NEA/生成器。沿用每图原四起点、十个新种子2000—2009；主口径为同时间，辅助8轮。完整80组成对明细见全部逐次结果.md、全部逐次.csv。','','|网络|方法|作者均值|方法均值|相对作者改善%|胜/平/负|父网簇描述性95%区间|','|---|---|---:|---:|---:|---|---|']
    for s in summary:
        v=s['main'];ci=s['parent_cluster_descriptive_95_interval']
        report.append(f"|{s['city']}|{s['method']}|{v['author_mean']:.10f}|{v['method_mean']:.10f}|{v['improvement_percent']:+.4f}|{v['wins']}/{v['ties']}/{v['losses']}|[{ci[0]:+.4f},{ci[1]:+.4f}]|")
    report+=['','## 必要解释','','总体改善是均值之比，逐次收益均值另在JSON保存。新旧批分别报告，不能按结果择取其中一批。四个独立父网簇的区间仅作描述；十种子增加随机重复，未新增独立道路/OD图或父网。总体正均值不自动表示显著/稳定或每个起点都胜。','', 'full对count/ridge是控制实验；count同时移除学习分数和在线收益作用，不能单独归因于某个组件。回归沿用冻结训练配方，不能推广为SPO+胜过一切调优回归。','', '## 固定8轮辅助结果','']
    for s in summary:report.append(f"- {s['city']} {s['method']}：相对作者{s['final8']['improvement_percent']:+.4f}%。")
    report+=['','## 主方法相对两个控制组','']
    for c in controls:report.append(f"- {c['city']} full对{c['comparator']}：均值改善{c['full_improvement_percent']:+.4f}%，胜/平/负{c['wins']}/{c['ties']}/{c['losses']}。")
    report+=['',f'来源SHA全通过；原始JSONL与COMPLETE及320项冻结顺序逐条一致；80组没有缺失/重复。count独立重放{count_replayed}次；{target_records}个预算/终态方案由既有独立评价接口核验全部合法，最大cost误差{target_error:.10g}。每次811个成本样本、82次调用、80批/800草稿，神经实际采用400。', '', f"正式批耗时{doc['phase_seconds']:.3f}秒（含本批外层落盘）；共享导入{doc['import_seconds_this_invocation']:.3f}秒另记。单次在线时钟按旧driver包含全部内部开销。",'', '全部结果保留；不因正负结果继续改方法或加测。参数及原主方法均未改变，后续只据本批完整结果写作。仓库原有未提交修改保留，无Git提交、合并或推送。']
    with (ROOT/'结果说明.md').open('x',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
    print(json.dumps({'summary':summary,'controls':controls,'target_max_error':target_error},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
