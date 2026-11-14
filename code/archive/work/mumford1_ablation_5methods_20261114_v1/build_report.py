"""由已核验结果生成中文分析说明，不修改论文或实验记录。"""
import sys
import json
import statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import experiment as e


def main():
    data = e.read(ROOT/'统计与核验.json')
    independent = e.read(ROOT/'独立最终统计复核.json')
    excel = e.read(ROOT/'Excel独立核验.json')
    done = e.read(ROOT/'COMPLETE.json')
    assert data['passed'] and excel['passed']
    labels = dict(author='作者NEA',count='Count',ridge='Ridge',nogain='NoGain',full='Full（PORA）')
    text = ['# Mumford1五配置消融：结果与分析', '',
        '2026-11-14。本批50次正式运行全部完成，五配置各10个种子（3011—3020）；完整8轮为主终点，120秒为次终点。原Mandl、Mumford0、旧Mumford1冻结记录及论文正文保持不变。', '',
        '本批沿用同一70站、15条线路的公开参考初始线网和既有十个搜索种子，是扩展消融新批次。所有五配置同期串行运行，采用预定Williams平衡顺序；统计单位为十个配对种子。', '',
        '## 核心结论', '',
        'Full在Mumford1上明显优于简单Count：8轮平均cost降低9.92%，120秒降低38.40%，两终点均10胜0平0负；三个预设机制比较内Holm校正后p均为0.005859375。', '',
        'Full在8轮的平均cost最低。Full对Ridge、NoGain的差异均未达到校正后的0.05阈值。120秒时Ridge的平均cost略低于Full。因此，本轮支持任务优先级分配在Mumford1上的价值，尚未建立SPO+训练配方或在线收益项分别优于相应替代方案的显著证据。', '',
        '## 五配置汇总', '',
        '改善率为100×（同期NEA均值−配置均值）/同期NEA均值；标准差为十次运行的样本标准差。两个终点来自同一批完整运行。', '']
    for endpoint, title in [('8round','完整8轮：主终点'),('120s','120秒：次终点')]:
        text += ['### '+title, '', '| 配置 | 平均cost | 样本标准差 | 相对同期NEA改善 | 胜/平/负（相对NEA） |',
                 '|---|---:|---:|---:|---:|']
        for row in (r for r in data['summary'] if r['endpoint']==endpoint):
            wtl='—' if row['method']=='author' else f"{row['wins']}/{row['ties']}/{row['losses']}"
            improvement='—' if row['method']=='author' else f"{row['improvement_vs_NEA_percent']:+.4f}%"
            text.append(f"| {labels[row['method']]} | {row['mean_cost']:.8f} | {row['sd_cost']:.8f} | {improvement} | {wtl} |")
        text.append('')
    text += ['## Full的直接配对比较', '',
        '以下改善率以对应对照均值为分母，与上表相对NEA的百分比不同。p为双侧精确条件Wilcoxon检验；每个终点分别对Full–Count、Full–Ridge、Full–NoGain三个预设比较作Holm校正。', '',
        '| 终点 | 对照 | Full相对改善 | Full胜/平/负 | 原始双侧p | Holm p |',
        '|---|---|---:|---:|---:|---:|']
    for row in data['comparisons']:
        if row['comparator']=='author':continue
        endpoint='8轮' if row['endpoint']=='8round' else '120秒'
        text.append(f"| {endpoint} | {labels[row['comparator']]} | {row['full_improvement_percent']:+.4f}% | {row['wins']}/{row['ties']}/{row['losses']} | {row['p_two_sided']:.9f} | {row['p_holm']:.9f} |")
    text += ['', '相对NEA的参考比较另列：8轮平均改善2.4599%，7胜3负，原始双侧p=0.083984375；120秒平均改善13.7398%，10胜，原始p=0.001953125。NEA比较不混入上述三个机制对照的校正族。', '',
        '胜平负使用绝对cost容差1e-6；秩检验仅移除精确零差。120秒Full–NoGain有4个精确持平，检验非零组数为6。Excel中的95%区间是配对均值差的描述性bootstrap区间（20000次、统计种子20261114，未作多重校正）。该区间与Wilcoxon检验的统计目标和假设不同；显著性判断使用预设检验及其Holm结果。', '',
        '## 对研究问题的解释', '',
        '### 简单Count在Mumford1上是否足够', '',
        '本次Full–Count比较给出明确否定：两终点10个种子全部由Full取得更低cost，且多重校正后仍显著。Count的完整8轮平均cost为1.25452150，甚至高于NEA的1.15860033；其120秒cost为2.10815561，表明均衡尝试不能充分替代该实例上的任务优先级。结论针对本次Count定义、共同初始线网和冻结预算。', '',
        '### 是否支持“小网络计数足够，大网络学习选线更有价值”', '',
        'Mandl冻结报告队列中Full与Count接近；本次Mumford1中二者差距明显，结果与这一情境解释相符。建议论文使用“小规模线网/较少候选重建目标”，避免“小样本”与训练样本或种子数混淆。可写为：在所测试的网络和预算下，简单尝试均衡在Mandl上具有竞争力，而Mumford1上的学习优先级分配获得更明显的额外收益。跨实例时拓扑、需求、路线约束和预算同时变化，因此本轮没有单独识别网络规模的因果作用，也没有直接测量“试错更便宜”这一原因。', '',
        '### 收益反馈是否提供了独立增益', '',
        'NoGain保留相同SPO+先验、次数项和实际收益记录，仅在选择指数中关闭平均收益项。8轮Full相对NoGain均值改善2.6006%，7胜3负，但Holm p=0.26171875；120秒改善0.8304%，5胜4平1负，Holm p=0.125。本轮观察到平均收益方向为正，尚不足以将在线反馈写成已验证的必要组件。', '',
        '### SPO+是否优于回归先验', '',
        '8轮Full相对Ridge均值改善1.1384%，Holm p=0.431640625；120秒Ridge平均cost比Full更低，Full相对改善为−0.6097%，Holm p=1。两者在本轮未检出显著差异。Ridge沿用冻结Mumford0回归先验，均没有在Mumford1上训练或调参。这比较的是两个冻结训练配方，包含其评分尺度的影响。', '',
        '### 早期搜索收益主要说明什么', '',
        '120秒时Ridge、NoGain和Full相对同期NEA均有约13%—14%的均值改善，三者均明显优于Count。Full与NoGain有4个种子的截点cost完全相同。这与早期收益主要由任务先验和优先级驱动的解释相符；收益反馈实际参与决策，但其额外性能收益仍表现为趋势。', '',
        '## 机制、P/O和开销', '']
    full=next(r for r in data['summary'] if r['endpoint']=='8round' and r['method']=='full')
    allocation=full['mean_selection_seconds']+full['mean_feedback_seconds']
    changed=sum(r['gain_changed_choice_count'] for r in data['mechanisms'] if r['method']=='full')
    text += [f"Full完整8轮平均在线时间为{full['mean_online_seconds']:.4f}秒，选线和反馈合计{allocation:.4f}秒，占{100*allocation/full['mean_online_seconds']:.4f}%。在Full的4000次选择中，收益项相对于相同历史下仅用先验/计数的局部反事实选择改变了{changed}次（{changed/40:.2f}%）。该计数证明收益项实际影响选择；性能增量仍由Full–NoGain受控比较判断。", '',
        'P/O均已列入Excel。Full在Mumford1上的标量cost改善伴随乘客时间增加和线路运行时间减少，不能直接写成乘客体验或实际车辆费用改善。O是单向线路道路运行时间总和，未包含频率、车数、容量或货币成本。', '',
        f"共同模块导入为{done['import_seconds']:.6f}秒，原始JSON归档落盘合计{done['raw_io_seconds_this_invocation']:.6f}秒，均另列；完整批次用时{done['batch_seconds_this_invocation']:.3f}秒。每run在线计时包含包装准备、模型构建/加载、特征、缓存、选择、反馈、构造、评价及pass日志。Excel的选线/反馈/准备时间、模型调用、缓存和机制计数均为完整8轮全程值，120秒行也采用这一全程口径，未将其冒充120秒前开销。", '',
        '## 与原冻结结果的关系', '',
        '旧批次NEA和Full的20次八轮最终cost、全部41个检查点cost、最终线路和搜索history在本批均完全复现。新批120秒有1处取点变化：种子3014的NEA本批取到1.5634216070，旧批为1.5617163181，原因是实际墙钟下120秒落在不同检查点。由此本批参考改善为13.7398%，旧冻结主实验13.7301%继续保留。八轮改善2.4599%完全一致。本批没有覆盖任何旧记录，也没有把同一种子的重跑当作额外独立样本。', '',
        '## 验证与交付', '',
        f"50次正式运行、2050检查点、16000次选择重放全部通过；独立线路层图算法复算{data['independently_evaluated_networks']}个合法线网，最大cost误差{data['max_cost_error']:.10g}。371项保护来源均保持一致，覆盖既有模型、作者依赖、输入、冻结记录及论文文件。独立统计审查使用单独的符号枚举和Holm实现，并核对均值、SD、分母、胜平负、bootstrap与端点。", '',
        f"工作簿已核对{excel['numeric_checks']}个导出数值单元、{excel['saved_formulas']}个保存公式及7个工作表，公式错误扫描为零，七表均完成视觉检查。Excel中的均值、差值和改善率可沿公式追溯；p值、区间与胜平负为统计脚本生成的固定快照。", '',
        '- Excel：outputs/mumford1_ablation_20261114/Mumford1_五配置消融_10种子_8轮及120秒.xlsx',
        '- 原始运行：原始运行.jsonl；完整统计：统计与核验.json。',
        '- CSV：主汇总.csv、配对检验.csv、逐次方法结果.csv、逐种子配对.csv、全部检查点.csv、机制概览.csv。',
        '- 复核：独立最终统计复核.md、Excel独立核验.json。',
        '- 实际运行入口：experiment.py prepare、test_ablation.py -v、experiment.py gate、experiment.py run、analyze.py、build_workbook.mjs、verify_workbook.py。Python搜索环境与Node制表运行时路径见协议及启动记录。',
        '- 研究范围和统计口径已冻结，本轮未新增种子、训练或调参；正文和图表包没有修改。',
        '- Git没有提交、合并或推送；仓库原有未提交文件保留。', '']
    with (ROOT/'结果分析.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(text))
    print(ROOT/'结果分析.md')


if __name__=='__main__':
    main()
