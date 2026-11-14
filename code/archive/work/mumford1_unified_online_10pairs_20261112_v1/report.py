"""从冻结20次运行的核验统计生成中文说明，不筛选种子。"""
import sys
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import experiment as e


def main():
    d=e.read(ROOT/'统计与核验.json');req=e.read(ROOT/'请求冻结.json');e.check(req['hashes'])
    assert d['passed'] and d['method_runs']==20 and d['analysis_sha256']==e.sha(ROOT/'analyze.py')
    lines=['# Mumford1统一在线Full：十组成对确认','',
       '日期：2026-11-12。一个固定合法Mumford1参考线网，种子3011—3020，每种子分别运行作者NEA和Full，共20次，全部完整8轮。所有正式任务均在开始前冻结，没有根据结果更换样本、参数、起点或主指标。','',
       'Full统一为SPO+评分先验＋c=.05已完成收益/尝试次数分配，沿用冻结Mumford0联合B模型的raw十维编码、尺度、标准化和系数；没有Mumford1训练或调参。作者源文件、生成器、接受与招募未修改，沿用已核验的共同索引修正版。','',
       '## 两个预先固定的终点','',
       '| 终点 | 作者cost均值±标准差 | Full cost均值±标准差 | Full相对作者改善 | 胜/平/负 | 双侧p |',
       '|---|---:|---:|---:|---:|---:|']
    for r in d['summary']:
        lines.append('| {} | {:.8f} ± {:.8f} | {:.8f} ± {:.8f} | {:+.4f}% | {}/{}/{} | {:.9g} |'.format(
            '完整8轮（主）' if r['endpoint']=='8round' else '120秒（辅助）',r['author_mean'],r['author_sd'],r['full_mean'],r['full_sd'],
            r['improvement_percent'],r['wins'],r['ties'],r['losses'],r['p_two_sided']))
    lines.extend(['','改善率为100×(作者均值−Full均值)/作者均值，正数表示Full更低。胜平负容差1e-6；双侧精确条件符号秩检验采用配对差、零差移除、并列平均秩。8轮主检验与120秒辅助检验不当作两批独立样本。','',
       '## 全部十组','',
       '| 种子 | 8轮作者 | 8轮Full | 8轮改善 | 120秒作者 | 120秒Full | 120秒改善 |','|---|---:|---:|---:|---:|---:|---:|'])
    index={(r['seed'],r['endpoint']):r for r in d['pairs']}
    for seed in req['seeds']:
        a,b=index[seed,'8round'],index[seed,'120s']
        lines.append('| {} | {:.8f} | {:.8f} | {:+.4f}% | {:.8f} | {:.8f} | {:+.4f}% |'.format(seed,
            a['author_cost'],a['full_cost'],a['improvement_percent'],b['author_cost'],b['full_cost'],b['improvement_percent']))
    lines.extend(['','## 两项目标和计算开销','',
       '| 终点/方法 | 乘客时间(分) | 单向线路时间(分) | 完整在线秒 | 选线秒 | 反馈秒 |','|---|---:|---:|---:|---:|---:|'])
    for endpoint in ['8round','120s']:
        for method in ['author','full']:
            part=[r for r in d['rows'] if r['endpoint']==endpoint and r['method']==method]
            values=[statistics.mean(r[k] for r in part) for k in ['P_minutes','O_oneway_minutes','total_online_seconds','selection_seconds','feedback_seconds']]
            lines.append('| {} / {} | {:.6f} | {:.6f} | {:.3f} | {:.3f} | {:.3f} |'.format(endpoint,method,*values))
    lines.extend(['','cost=P/88+O/660。P包含每次5分钟换乘惩罚、不限换乘；O只为单向线路道路运行时间总和，不是真实费用、车辆或班次。总在线秒覆盖完整8轮，不是120秒截点的耗时。两方计时均包含包装、模型、特征、缓存、选线、反馈、生成、评价及进度记录；共享导入与外层JSON写盘另列。','',
       '## 验证范围与来源','',
       '- 同一70站/15线参考起点的十个搜索种子，检验当前起点下的随机变化及冻结模型迁移，不是十个城市或十个独立父网。',
       '- 20次均为80批/800场景草稿、811目标样本、82次目标调用；Full实际type1选线400次/run。',
       '- 作者3001的首轮六个检查点与旧pilot的cost/线路/样本数严格一致；该门禁不计入正式样本。',
       '- 三项专项测试和运行前独立接口审查通过。',
       '- {}次Full选线与反馈重放通过；独立层图核验{}个合法线网，最大cost误差{:.3g}。'.format(d['new_feedback_replays'],d['independent_legal_networks'],d['max_independent_cost_error']),
       '- {}项来源SHA保持不变；旧Mandl和Mumford0未修改。'.format(d['source_hashes_checked']),
       '- frozen_N24保存指定48行表及原检验结果、来源说明和完整40组记录。它是既有版本快照，不与本轮检验混合。',
       '- 本轮没有新训练、参数修改、生成器修改、付费算力、Git提交/合并/推送。','',
       '主要输出：十组配对结果.csv、逐次方法结果.csv、全部检查点.csv、统计与核验.json与Excel。所有正式原始记录保存于原始运行.jsonl。'])
    with (ROOT/'结果说明.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print('REPORT WRITTEN')


if __name__=='__main__':main()
