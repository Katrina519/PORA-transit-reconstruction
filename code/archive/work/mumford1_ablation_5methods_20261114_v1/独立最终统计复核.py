"""仅从冻结原始记录复核配对统计；不导入实验或主分析模块。"""
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'mumford1_unified_online_10pairs_20261112_v1' / '原始运行.jsonl'
METHODS = ['author', 'count', 'ridge', 'nogain', 'full']
SEEDS = list(range(3011, 3021))
EPS = 1e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def signed_rank_enumeration(differences):
    # 对实际非零差值的条件符号分布枚举；平均秩乘二后为整数。
    nonzero = [d for d in differences if d != 0]
    ranks2 = [sum(abs(e) < abs(d) for e in nonzero) * 2
              + sum(abs(e) == abs(d) for e in nonzero) + 1
              for d in nonzero]
    observed_plus2 = sum(r for r, d in zip(ranks2, nonzero) if d > 0)
    rank_total2 = sum(ranks2)
    observed_min2 = min(observed_plus2, rank_total2 - observed_plus2)
    sums2 = [sum(r for r, sign in zip(ranks2, signs) if sign)
             for signs in itertools.product([0, 1], repeat=len(nonzero))]
    extreme = sum(min(w, rank_total2-w) <= observed_min2 for w in sums2)
    return {'n_nonzero': len(nonzero), 'zeros_removed': len(differences)-len(nonzero),
            'average_ranks': [r/2 for r in ranks2],
            'positive_rank_sum': observed_plus2/2,
            'negative_rank_sum': (rank_total2-observed_plus2)/2,
            'statistic': observed_min2/2,
            'extreme_assignments': extreme, 'total_assignments': len(sums2),
            'p_raw': extreme/len(sums2)}


def holm(ps):
    ordered = sorted(ps, key=ps.get)
    previous = 0
    adjusted = {}
    for i, method in enumerate(ordered):
        previous = max(previous, min(1, (len(ordered)-i)*ps[method]))
        adjusted[method] = previous
    return adjusted


def summary(values):
    assert all(math.isfinite(x) for x in values)
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values), 'min': min(values), 'max': max(values)}


def endpoint(row, name):
    fine = row['fine_curve']
    assert len(fine) == 41
    assert [(c['iteration'], c['pass']) for c in fine] == [(0, 0)] + [
        (i, p) for i in range(1, 9) for p in range(1, 6)]
    assert all(math.isfinite(c['seconds']) and math.isfinite(c['cost']) for c in fine)
    assert all(a['seconds'] < b['seconds'] for a, b in zip(fine, fine[1:]))
    assert fine[-1]['cost'] == row['cost'] == row['metrics']['cost'][0]
    assert all(a['cost'] >= b['cost'] for a, b in zip(fine, fine[1:]))
    cp = fine[-1] if name == '8round' else [c for c in fine if c['seconds'] <= 120][-1]
    solution = row['fine_solutions'][cp['solution_index']]
    assert solution['cost'] == cp['cost']
    assert solution['first_seconds'] <= cp['seconds']
    return dict(cp)


def main():
    assert signed_rank_enumeration([1, 2, 3])['p_raw'] == .25
    assert signed_rank_enumeration([1, -1])['p_raw'] == 1
    assert signed_rank_enumeration([0, 0])['p_raw'] == 1
    assert signed_rank_enumeration([0, 1e-8])['n_nonzero'] == 1
    assert holm({'a': .01, 'b': .03, 'c': .04}) == {'a': .03, 'b': .06, 'c': .06}
    raw_path = ROOT / '原始运行.jsonl'
    rows = [json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines()]
    req = read_json(ROOT / '请求冻结.json')
    plan = read_json(ROOT / '分析方案冻结.json')
    complete = read_json(ROOT / 'COMPLETE.json')
    keyed = {(r['seed'], r['variant']): r for r in rows}
    assert len(rows) == len(keyed) == 50
    assert set(keyed) == set(itertools.product(SEEDS, METHODS))
    assert req['seeds'] == SEEDS and req['methods'] == METHODS
    assert [(r['seed'], r['variant']) for r in rows] == [(t['seed'], t['variant']) for t in req['tasks']]
    assert complete['passed'] and complete['completed_runs'] == 50
    assert sha(ROOT / '请求冻结.json') == complete['request_sha256']
    assert plan['win_tie_loss_tolerance'] == EPS
    assert plan['mechanism_comparators'] == ['count', 'ridge', 'nogain']
    for r in rows:
        assert r['generation_counts'] == [80,800]
        assert r['target_samples'] == 811 and r['target_calls'] == 82
        assert r['iterations'] == 8 and not r['repair'] and not r['fallbacks']
        assert r['fine_solutions'][r['fine_curve'][0]['solution_index']]['routes'] == req['context']['routes']
    position_counts = {m: {str(p): sum(t['variant']==m and t['position']==p
                                     for t in req['tasks']) for p in range(1, 6)} for m in METHODS}
    assert all(n == 2 for counts in position_counts.values() for n in counts.values())
    endpoints = {}
    bootstrap = np.random.RandomState(20261114)
    for name in ['8round', '120s']:
        cps = {m: [endpoint(keyed[s,m], name) for s in SEEDS] for m in METHODS}
        costs = {m: [cp['cost'] for cp in cps[m]] for m in METHODS}
        summaries = {m: summary(costs[m]) for m in METHODS}
        author_mean = summaries['author']['mean']
        for m in METHODS:
            summaries[m]['relative_to_author_ratio_of_means_percent'] = 100*(author_mean-summaries[m]['mean'])/author_mean
            summaries[m]['relative_to_author_mean_of_paired_percent'] = statistics.mean(
                100*(a-b)/a for a,b in zip(costs['author'], costs[m]))
            author_gains = [a-b for a,b in zip(costs['author'],costs[m])]
            summaries[m]['vs_author_wins'] = sum(x > EPS for x in author_gains)
            summaries[m]['vs_author_ties'] = sum(abs(x) <= EPS for x in author_gains)
            summaries[m]['vs_author_losses'] = sum(x < -EPS for x in author_gains)
        comparisons = {}
        for m in ['author', 'count', 'ridge', 'nogain']:
            gains = [a-b for a,b in zip(costs[m], costs['full'])]
            resample_indices = bootstrap.randint(0,10,(20000,10))
            bootstrap_means = np.asarray(gains)[resample_indices].mean(axis=1)
            ci = np.percentile(bootstrap_means,[2.5,97.5])
            comparisons[m] = {'contrast': 'full_vs_'+m, 'paired_n': 10,
                'gain_definition': 'comparator_cost - full_cost; positive favors Full',
                'absolute_gain': summary(gains),
                'relative_gain_ratio_of_means_percent': 100*statistics.mean(gains)/summaries[m]['mean'],
                'relative_gain_mean_of_paired_percent': statistics.mean(100*d/a for d,a in zip(gains,costs[m])),
                'full_win': sum(d > EPS for d in gains),
                'tie': sum(abs(d) <= EPS for d in gains),
                'full_loss': sum(d < -EPS for d in gains),
                'differences': gains,
                'unadjusted_bootstrap_ci95_mean_absolute_gain': [float(ci[0]),float(ci[1])],
                'wilcoxon': signed_rank_enumeration(gains)}
        adjusted = holm({m: comparisons[m]['wilcoxon']['p_raw'] for m in ['count','ridge','nogain']})
        for m in adjusted:
            comparisons[m]['p_holm'] = adjusted[m]
        comparisons['author']['p_holm'] = None
        endpoints[name] = {'summaries': summaries, 'comparisons': comparisons,
                           'checkpoint_by_method': cps, 'costs_by_method': costs}
    old_rows = [json.loads(line) for line in OLD.read_text(encoding='utf-8').splitlines()]
    old_keyed = {(r['seed'],r['variant']):r for r in old_rows}
    reproduction = []
    for s,m in itertools.product(SEEDS, ['author','full']):
        a,b=keyed[s,m],old_keyed[s,m]
        reproduction.append({'seed':s,'method':m,'final_cost_exact_equal':a['cost']==b['cost'],
            'all_41_checkpoint_costs_exact_equal':[c['cost'] for c in a['fine_curve']]==[c['cost'] for c in b['fine_curve']],
            'all_8round_history_costs_exact_equal':a['history']==b['history'],
            'final_routes_exact_equal':a['routes']==b['routes'],
            'new_120s_cost': endpoint(a,'120s')['cost'], 'old_120s_cost':endpoint(b,'120s')['cost'],
            'new_total_seconds':a['seconds'], 'old_total_seconds':b['seconds']})
    assert all(all(row[k] for k in ['final_cost_exact_equal','all_41_checkpoint_costs_exact_equal',
                                  'all_8round_history_costs_exact_equal','final_routes_exact_equal']) for row in reproduction)
    result = {'status':'统计通过；等待主分析结果文件逐项比对',
        'scope':'从raw提取终点和独立条件符号枚举；未做图算法重算和选择器重放',
        'inputs_sha256': {str(p):sha(p) for p in [raw_path,OLD,ROOT/'协议.md', ROOT/'请求冻结.json',ROOT/'分析方案冻结.json',ROOT/'COMPLETE.json']},
        'run_count':50,'unique_seed_method_pairs':50,'paired_seeds_per_method':10,
        'position_counts':position_counts,'endpoints':endpoints,'old_batch_reproduction':reproduction,
        'timing_scope':'120s只取本次新raw的fine_curve；旧计时仅作来源核对，不合并或覆盖',
        'conclusions':{
            'full_vs_count':'支持：两终点Full均10胜0平0负，Holm p=0.005859375',
            'full_vs_ridge':'不支持宣称优势：8轮Holm p=0.431640625；120s Full平均代价略高且Holm p=1',
            'full_vs_nogain':'尚未证实收益项额外优势：8轮Holm p=0.26171875；120s Holm p=0.125',
            'full_vs_author_reference':'8轮改善2.4599106661%但原始p=0.083984375；120s改善13.7398326414%、原始p=0.001953125',
            'causal_scope':'单一起点、既有十种子的Mumford1扩展消融；不支持单独归因网络规模'},
        'limitations':['一个固定起点和既有种子集合上的扩展消融，不是独立新种子确认',
                       'Wilcoxon条件符号检验依赖配对差值符号可交换/对称的工作假设，10对未能验证总体分布假设',
                       '未拒绝差异不证明方法等效；跨实例规模解释不构成因果证明']}
    main_stats = ROOT/'统计与核验.json'
    if main_stats.exists():
        official = read_json(main_stats)
        checks = []
        def close(label,a,b):
            assert abs(a-b) <= 1e-12, (label,a,b)
            checks.append(label)
        assert official['passed'] and official['method_runs']==50 and official['paired_seeds']==10
        assert len(official['summary'])==10 and len(official['comparisons'])==8
        assert len(official['rows'])==100 and len(official['pairs'])==80
        for row in official['summary']:
            v = endpoints[row['endpoint']]['summaries'][row['method']]
            for external,internal in [('mean_cost','mean'),('sd_cost','sample_sd'),
                ('improvement_vs_NEA_percent','relative_to_author_ratio_of_means_percent'),
                ('wins','vs_author_wins'),('ties','vs_author_ties'),('losses','vs_author_losses')]:
                close(f"summary.{row['endpoint']}.{row['method']}.{external}",row[external],v[internal])
        for row in official['comparisons']:
            v = endpoints[row['endpoint']]['comparisons'][row['comparator']]
            for external,value in [('mean_absolute_gain',v['absolute_gain']['mean']),
                ('full_improvement_percent',v['relative_gain_ratio_of_means_percent']),
                ('wins',v['full_win']),('ties',v['tie']),('losses',v['full_loss']),
                ('W',v['wilcoxon']['statistic']),('p_two_sided',v['wilcoxon']['p_raw']),
                ('nonzero',v['wilcoxon']['n_nonzero']),
                ('gain_ci95_low',v['unadjusted_bootstrap_ci95_mean_absolute_gain'][0]),
                ('gain_ci95_high',v['unadjusted_bootstrap_ci95_mean_absolute_gain'][1])]:
                close(f"comparison.{row['endpoint']}.{row['comparator']}.{external}",row[external],value)
            assert row['p_holm']==v['p_holm']
            assert row['prespecified_mechanism_family']==(row['comparator']!='author')
        for row in official['rows']:
            run=keyed[row['seed'],row['method']]
            cp=endpoint(run,row['endpoint'])
            for external,value in [('cost',cp['cost']),('last_completed_seconds',cp['seconds']),
                ('completed_passes',(cp['samples']-10)//20),('total_online_seconds',run['seconds']),
                ('selection_seconds',run['selection_seconds']),('feedback_seconds',run['feedback_seconds']),
                ('preparation_seconds',run['wrapper_preparation_seconds'])]:
                close(f"row.{row['endpoint']}.{row['seed']}.{row['method']}.{external}",row[external],value)
        for row in official['pairs']:
            ep=endpoints[row['endpoint']]['costs_by_method']; i=SEEDS.index(row['seed'])
            a,b=ep[row['comparator']][i],ep['full'][i]
            for external,value in [('comparator_cost',a),('full_cost',b),('absolute_gain',a-b),
                ('full_improvement_percent',100*(a-b)/a)]:
                close(f"pair.{row['endpoint']}.{row['seed']}.{row['comparator']}.{external}",row[external],value)
        result['main_analysis_comparison'] = {'passed':True,'numeric_checks':len(checks),
            'absolute_tolerance':1e-12,'sha256':sha(main_stats),
            'scope':'均值SD、改善分母、WTL、精确检验、Holm、bootstrap CI、100终点提取、80逐种子配对；不重复图评价与选择重放'}
        result['status']='统计通过；主分析结果逐项比对通过'
    else:
        result['main_analysis_comparison']={'passed':None,'reason':'统计与核验.json尚未出现'}
    (ROOT/'独立最终统计复核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for e in endpoints:
        print(e)
        for m,v in endpoints[e]['summaries'].items(): print(m,v)
        for m,v in endpoints[e]['comparisons'].items():
            print(m,'gain',v['absolute_gain']['mean'],'WTL',v['full_win'],v['tie'],v['full_loss'],
                  'W',v['wilcoxon']['statistic'],'p',v['wilcoxon']['p_raw'],'Holm',v['p_holm'])
    print('OLD_REPRODUCTION',len(reproduction),'exact costs/history/routes checks passed')


if __name__ == '__main__':
    main()
