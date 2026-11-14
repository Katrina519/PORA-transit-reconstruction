"""从核验统计生成本轮交付说明，不改旧表。"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e


def main():
    d=e.read(ROOT/'统计与核验.json');req=e.read(e.REQUEST);e.check(req['hashes'])
    assert d['passed'] and d['new_runs']==80 and d['analysis_sha256']==e.sha(ROOT/'analyze.py')
    name={'author':'作者NEA','full':'Full：SPO+先验＋在线分配','count':'简单计数','ridge':'冻结回归评分器＋在线分配'}
    city={'mandl':'Mandl','mumford0':'Mumford0'}
    lines=['# 统一在线方法：40组Mumford0更新与主实验、消融', '',
       '2026-11-12。按用户最新指定的全部逐次.csv更新。每网络40个起点—种子组合；新增Mumford0 Full和匹配Ridge各40次，共80次正式运行。Mandl160个方法结果、Mumford0作者/Count共80个方法结果全部复用，合计新表320个方法结果。作者NEA、生成器、目标和权重不改，所有种子保留。', '',
       '## 唯一主方法', '',
       '两网络统一采用SPO+评分先验＋原c=.05在线收益/尝试次数分配，沿用各自既有冻结评分权重。Mumford0不再以静态B作为主方法。Ridge对照保留同一在线分配，替换为原冻结平方损失评分器及其已冻结CV正则设置。Count替代整个学习评分/实际收益分配，而不是只删SPO+一个损失项。', '',
       '## 相同8轮主比较', '',
       '遵照用户要求，不重跑作者和Count。正式可比口径为相同8轮、相同80批生成与811评价样本，避免将不同运行批次的墙钟变化写成算法效率优势。', '',
       '| 网络 | 方法 | 平均cost | 标准差 | 相对作者改善 | 胜/平/负 |',
       '|---|---|---:|---:|---:|---:|']
    for r in d['summary']:
        if r['endpoint']!='final':continue
        improvement='—' if r['method']=='author' else '{:+.4f}%'.format(r['improvement_percent'])
        w='—' if r['method']=='author' else '{}/{}/{}'.format(r['wins'],r['ties'],r['losses'])
        lines.append('| {} | {} | {:.8f} | {:.8f} | {} | {} |'.format(city[r['city']],name[r['method']],r['method_mean'],r['cost_sd'],improvement,w))
    lines.extend(['','## 原表4/7秒预算记录','',
       '以下使用各自真实记录的墙钟：Mandl4秒、Mumford0 7秒。NEA、Count和Mandl为旧批次，新Mumford0 Full/Ridge为本轮；不是同批墙钟效率对照。不能拿其表面改善幅度替代上面的等生成次数结论，也不修改时钟来凑一致。','',
       '| 网络 | 方法 | 平均cost | 相对作者改善 | 胜/平/负 |','|---|---|---:|---:|---:|'])
    for r in d['summary']:
        if r['endpoint']!='budget':continue
        improvement='—' if r['method']=='author' else '{:+.4f}%'.format(r['improvement_percent'])
        w='—' if r['method']=='author' else '{}/{}/{}'.format(r['wins'],r['ties'],r['losses'])
        lines.append('| {} | {} | {:.8f} | {} | {} |'.format(city[r['city']],name[r['method']],r['method_mean'],improvement,w))
    lines.extend(['','## Full与消融的直接配对（8轮）','',
       '| 网络 | 对照 | Full相对对照改善 | Full胜/平/负 | 单侧p值 | 秩二列效应量 |','|---|---|---:|---:|---:|---:|'])
    for r in d['ablations']:
        if r['endpoint']!='final':continue
        lines.append('| {} | {} | {:+.4f}% | {}/{}/{} | {:.6g} | {:.4f} |'.format(city[r['city']],name[r['control']],
            r['improvement_percent'],r['wins'],r['ties'],r['losses'],r['one_sided_p_full_lower'],r['rank_biserial_positive_full_better']))
    lines.extend(['','单侧方向为Full cost更低，p值是配对运行层面的探索性检验。四个起点来自同一道路/OD网络，不等于40个独立网络。消融结果按实际方向解释，不因Full必须最好而更换反馈、参数或种子。','',
       '## 两项目标与开销','',
       '| 网络/方法 | 8轮乘客时间(分) | 8轮单向线路时间(分) | 完整在线耗时(秒) |','|---|---:|---:|---:|'])
    for r in d['summary']:
        if r['endpoint']!='final':continue
        part=[v for v in d['records'] if v['city']==r['city'] and v['method']==r['method']]
        seconds=sum(v['online_seconds'] for v in part)/len(part)
        lines.append('| {} / {} | {:.6f} | {:.6f} | {:.4f} |'.format(city[r['city']],name[r['method']],r['P_mean'],r['O_mean'],seconds))
    lines.extend(['','运行时间列保留各自批次，不直接解释为加速比。O为单向线路道路运行时间总和，不是费用、车辆或班次。若P/O方向不同，应按目标权衡解释。','',
       '## 文件与核验','',
       '- 原全部逐次.csv保持原样，输入SHA='+req['source_table_sha256']+'。',
       '- 全部逐次_统一在线方法.csv：保留原41列、原80行顺序；Mandl40行逐字段不变；Mumford0作者和Count字段不变。',
       '- Excel包含主实验汇总、消融、更新后的原格式逐次、方法逐次、Mumford0旧新对照、方法与口径、原始CSV。',
       '- 新增{}次正式运行，重放{}次新选线与反馈；{}个不同合法线网独立线路层图重算，最大cost误差{:.3g}。'.format(d['new_runs'],d['new_feedback_replays'],d['independent_legal_networks'],d['max_cost_error']),
       '- {}项冻结来源SHA保持不变；旧Full/Ridge保留在原始输入和旧新对照中，不覆盖。'.format(d['hashes_checked']),
       '- 两项专项测试、13字段在线历史轨迹门禁和运行前独立接口审查通过。',
       '- 原32任务请求在正式运行前已因用户改选CSV被废止，未执行；实际正式任务以请求冻结_40组.json为准。',
       '- 没有新训练、调参、作者/Count重跑、生成器修改或付费算力。仓库未提交、合并或推送。'])
    with (ROOT/'结果说明.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print('REPORT WRITTEN')


if __name__=='__main__':main()
